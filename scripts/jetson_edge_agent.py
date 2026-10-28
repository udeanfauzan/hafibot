#!/usr/bin/env python3
'''
jetson_edge_agent.py - Ultra-Low-Latency Edge Streamer & Voice Gateway for Jetson Nano.

Architecture:
  - Thread 1: Camera Grabber (Decoupled, continuous camera draining, 0 frame backlog).
  - Thread 2: Network Streamer (Persistent HTTP Session with Keep-Alive, 15 FPS pacing).
  - Thread 3: Arduino Serial Gateway (Bidirectional USB UART communications).
  - Thread 4: USB Microphone Voice Grabber & VAD (Energy-based speech segmenter with Auto-Device Discovery).
  - Thread 5: I2S Speaker Audio Playback Queue (MAX98357A 4W amplifier player with ALSA aplay).
  - Thread 6: Proactive Audio Poller (Receives fall alert prompts & scheduled reminders).

Usage:
  python3 scripts/jetson_edge_agent.py --server http://100.86.76.95:5000 --enable-voice
'''

import argparse
import base64
import io
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

try:
    import serial
except ImportError:
    serial = None

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    num_orig = len(audio)
    num_target = int(round(num_orig * target_sr / orig_sr))
    x_orig = np.linspace(0, 1, num_orig, endpoint=False)
    x_target = np.linspace(0, 1, num_target, endpoint=False)
    return np.interp(x_target, x_orig, audio).astype(audio.dtype)



class JetsonEdgeAgent:
    def __init__(
        self,
        server_url: str = 'http://127.0.0.1:5000',
        serial_port: str = '/dev/ttyUSB0',
        baudrate: int = 115200,
        camera_sensor_id: int = 0,
        camera_sensor_mode: int = -1,
        capture_width: int = 1280,
        capture_height: int = 720,
        capture_fps: int = 30,
        output_width: int = 640,
        output_height: int = 480,
        target_fps: float = 15.0,
        jpeg_quality: int = 60,
        enable_voice: bool = True,
        mic_device: Optional[str] = None,
        speaker_device: str = 'default',
        vad_threshold: float = 0.030,
        silence_timeout: float = 0.8,
    ):
        self.server_url = server_url.rstrip('/')
        self.frame_endpoint = f'{self.server_url}/api/robot/frame'
        self.voice_endpoint = f'{self.server_url}/api/voice/interact'
        self.pending_audio_endpoint = f'{self.server_url}/api/voice/pending_audio'

        self.serial_port = serial_port
        self.baudrate = baudrate

        self.camera_sensor_id = camera_sensor_id
        self.camera_sensor_mode = camera_sensor_mode
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.capture_fps = capture_fps
        self.output_width = output_width
        self.output_height = output_height
        self.target_fps = max(1.0, float(target_fps))
        self.jpeg_quality = min(95, max(30, int(jpeg_quality)))

        self.enable_voice = enable_voice
        self.mic_device = mic_device
        self.speaker_device = speaker_device
        self.vad_threshold = float(vad_threshold)
        self.silence_timeout = float(silence_timeout)

        self.running = True

        # Decoupled Camera Frame Buffer (Thread-safe, 0 latency)
        self.latest_frame = None
        self.latest_frame_stamp = 0.0
        self.frame_lock = threading.Lock()
        self.camera_ready = threading.Event()

        # Serial state
        self.serial_conn = None
        self.serial_lock = threading.Lock()
        self.latest_telemetry = {'state': 'IDLE', 'us_left': 0.0, 'us_center': 0.0, 'us_right': 0.0}
        self.telemetry_lock = threading.Lock()

        # Audio Queues & State
        self.outgoing_frame_queue = queue.Queue(maxsize=2)
        self.audio_playback_queue = queue.Queue(maxsize=30)
        self.is_speaking = threading.Event()
        self.remote_voice_enabled = bool(enable_voice)
        self.playback_lock = threading.Lock()
        self.playback_process = None
        self.is_uploading = False
        self.is_fetching_audio = False
        self.upload_lock = threading.Lock()  # Echo suppression gate
        self.network_stats_lock = threading.Lock()
        self.delivered_frame_times = []
        self.last_video_error = ''

        # Dedicated Persistent HTTP Sessions (Zero Lock Contention)
        if HAS_REQUESTS:
            # Dedicated Session 1: High-Speed Video Streamer (15 FPS)
            self.video_session = requests.Session()
            v_adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=4, max_retries=0)
            self.video_session.mount('http://', v_adapter)
            self.video_session.mount('https://', v_adapter)

            # Dedicated Session 2: Asynchronous Voice I/O
            self.voice_session = requests.Session()
            a_adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=1)
            self.voice_session.mount('http://', a_adapter)
            self.voice_session.mount('https://', a_adapter)
        else:
            self.video_session = None
            self.voice_session = None

        print('=' * 65)
        print('  HAFIBOT - Low-Latency Edge Streamer & Voice Gateway (Jetson Nano)')
        print('=' * 65)
        print(f'  Target Server : {self.server_url}')
        print(f'  Arduino Serial: {self.serial_port} @ {self.baudrate} baud')
        print(f'  Stream Config : {self.output_width}x{self.output_height} @ {self.target_fps:.1f} FPS (JPEG Q={self.jpeg_quality})')
        print(f'  Voice Support : {"ENABLED (USB Mic + I2S Speaker)" if self.enable_voice else "DISABLED"}')
        if self.enable_voice:
            print(f'    Speaker ALSA: {self.speaker_device}')
            print(f'    VAD Sens.   : Threshold={self.vad_threshold:.4f}, Silence={self.silence_timeout:.1f}s')
        print(f'  HTTP Engine   : {"Requests Session (HTTP Keep-Alive)" if HAS_REQUESTS else "Urllib (Legacy)"}')
        print('=' * 65)

    # ==========================================================================
    # 1. ARDUINO SERIAL BRIDGE
    # ==========================================================================
    def _connect_serial(self) -> bool:
        if serial is None:
            print('  [SERIAL WARNING] pyserial belum ter-install. Jalankan: pip3 install pyserial')
            return False
        with self.serial_lock:
            if self.serial_conn is not None and self.serial_conn.is_open:
                return True
            candidate_ports = [
                self.serial_port,
                '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2',
                '/dev/ttyACM0', '/dev/ttyACM1',
            ]
            candidate_ports = list(dict.fromkeys(candidate_ports))

            for port in candidate_ports:
                if not os.path.exists(port):
                    continue
                try:
                    conn = serial.Serial(port, self.baudrate, timeout=0.05, write_timeout=0.2)
                    time.sleep(1.5)
                    conn.reset_input_buffer()
                    self.serial_conn = conn
                    self.serial_port = port
                    print(f'  [SERIAL SUCCESS] Connected to Arduino on {port}')
                    return True
                except Exception as e:
                    print(f'  [SERIAL ERROR] Gagal membuka {port}: {e}')
            return False

    def _write_serial(self, command_str: str):
        if not command_str:
            return
        with self.serial_lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    if not command_str.endswith('\n'):
                        command_str += '\n'
                    self.serial_conn.write(command_str.encode('ascii'))
                except Exception as e:
                    print(f'  [SERIAL ERROR] Write failed: {e}')
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
                    self.serial_conn = None

    def _serial_reader_loop(self):
        while self.running:
            if not self._connect_serial():
                time.sleep(1.0)
                continue

            try:
                line = self.serial_conn.readline()
                if not line:
                    continue
                text = line.decode('ascii', errors='ignore').strip()
                if text.startswith('S,'):
                    parts = text.split(',')
                    if len(parts) >= 5:
                        state = parts[1][:24]
                        us_l = float(parts[2])
                        us_c = float(parts[3])
                        us_r = float(parts[4])
                        with self.telemetry_lock:
                            self.latest_telemetry = {
                                'state': state,
                                'us_left': us_l,
                                'us_center': us_c,
                                'us_right': us_r,
                            }
                elif text.startswith('#'):
                    print(f'  [ARDUINO] {text}')
            except Exception as e:
                with self.serial_lock:
                    if self.serial_conn:
                        try:
                            self.serial_conn.close()
                        except Exception:
                            pass
                        self.serial_conn = None
                time.sleep(0.5)

    # ==========================================================================
    # 2. CAMERA CAPTURE PIPELINE (DECOUPLED, ZERO-LATENCY BUFFER)
    # ==========================================================================
    def _build_gstreamer_pipeline(self) -> str:
        sensor_mode_clause = f'sensor-mode={self.camera_sensor_mode} ' if self.camera_sensor_mode >= 0 else ''
        return (
            f'nvarguscamerasrc sensor-id={self.camera_sensor_id} {sensor_mode_clause}! '
            f'video/x-raw(memory:NVMM), width=(int){self.capture_width}, '
            f'height=(int){self.capture_height}, format=(string)NV12, '
            f'framerate=(fraction){self.capture_fps}/1 ! '
            f'queue max-size-buffers=1 leaky=downstream ! '
            f'nvvidconv flip-method=0 ! '
            f'video/x-raw, width=(int){self.output_width}, '
            f'height=(int){self.output_height}, format=(string)BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=(string)BGR ! '
            f'appsink max-buffers=1 drop=true sync=false'
        )

    def _open_camera(self):
        pipeline = self._build_gstreamer_pipeline()
        print('  [CAMERA] Opening GStreamer CSI pipeline...')
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print('  [CAMERA SUCCESS] CSI camera streaming via GStreamer!')
                    return cap
                time.sleep(0.1)
            cap.release()

        print('  [CAMERA FALLBACK] Trying V4L2 /dev/video0...')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.output_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.output_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print('  [CAMERA SUCCESS] Camera streaming via V4L2 (/dev/video0)!')
                return cap
            cap.release()

        return None

    def _restart_nvargus_daemon(self):
        print('  [RECOVERY] Restarting nvargus-daemon...')
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', 'nvargus-daemon'], timeout=5)
            time.sleep(2.0)
            print('  [RECOVERY] nvargus-daemon restarted.')
        except Exception as e:
            print(f'  [RECOVERY] Could not auto-restart nvargus-daemon: {e}')

    def _camera_worker_loop(self):
        cap = None
        for attempt in range(3):
            cap = self._open_camera()
            if cap is not None:
                break
            print(f'  [CAMERA RETRY] Attempt {attempt+1} failed. Resetting nvargus-daemon...')
            self._restart_nvargus_daemon()
            time.sleep(1.5)

        if cap is None:
            print('  [ERROR] Cannot open camera after 3 attempts. Exiting camera thread.')
            return

        consecutive_cam_errors = 0
        self.camera_ready.set()

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_cam_errors += 1
                if consecutive_cam_errors > 20:
                    print('  [CAMERA ERROR] Stream stalled. Reconnecting...')
                    cap.release()
                    self._restart_nvargus_daemon()
                    cap = self._open_camera()
                    consecutive_cam_errors = 0
                    if cap is None:
                        time.sleep(2.0)
                        continue
                time.sleep(0.01)
                continue

            consecutive_cam_errors = 0
            now_stamp = time.time()
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_frame_stamp = now_stamp

        cap.release()
        print('  [CAMERA] Camera worker stopped.')

    # ==========================================================================
    # 3. I2S SPEAKER PLAYBACK WORKER (MAX98357A AMPLIFIER)
    # ==========================================================================
    def _set_remote_voice_enabled(self, enabled: bool):
        enabled = bool(enabled) and self.enable_voice
        if enabled == self.remote_voice_enabled:
            return
        self.remote_voice_enabled = enabled
        state = 'ENABLED' if enabled else 'DISABLED'
        print(f'  [VOICE CONTROL] Laptop dashboard set voice to {state}')

        if not enabled:
            while True:
                try:
                    self.audio_playback_queue.get_nowait()
                    self.audio_playback_queue.task_done()
                except queue.Empty:
                    break
            with self.playback_lock:
                process = self.playback_process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            if HAS_SOUNDDEVICE:
                try:
                    sd.stop()
                except Exception:
                    pass

    def _play_wav_bytes(self, wav_bytes: bytes):
        """Plays WAV audio bytes on I2S ALSA hardware or sounddevice."""
        if not self.remote_voice_enabled or not wav_bytes or len(wav_bytes) < 100:
            return

        self.is_speaking.set()  # Enable echo suppression
        try:
            # 1. Try playing with ALSA aplay command directly (standard on Jetson)
            if sys.platform.startswith('linux'):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
                    tf.write(wav_bytes)
                    temp_path = tf.name
                try:
                    cmd = ['aplay', '-q']
                    if self.speaker_device and self.speaker_device != 'default':
                        cmd.extend(['-D', self.speaker_device])
                    cmd.append(temp_path)
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    with self.playback_lock:
                        self.playback_process = process
                    if not self.remote_voice_enabled:
                        process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                finally:
                    with self.playback_lock:
                        self.playback_process = None
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                return

            # 2. Cross-platform fallback: sounddevice
            if HAS_SOUNDDEVICE:
                bio = io.BytesIO(wav_bytes)
                with wave.open(bio, 'rb') as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    data = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(data, dtype=np.int16)
                    if ch > 1:
                        samples = samples.reshape((-1, ch))
                    sd.play(samples, samplerate=sr)
                    sd.wait()
        except Exception as e:
            print(f'  [SPEAKER ERROR] Playback failed: {e}')
        finally:
            time.sleep(0.3)
            self.is_speaking.clear()

    def _speaker_worker_loop(self):
        """Background thread playing audio from playback queue."""
        while self.running:
            try:
                item = self.audio_playback_queue.get(timeout=0.5)
                if item is None:
                    break
                wav_bytes = item if isinstance(item, bytes) else item.get('wav_bytes', b'')
                if self.remote_voice_enabled:
                    self._play_wav_bytes(wav_bytes)
                self.audio_playback_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f'  [SPEAKER ERROR] {e}')

    # ==========================================================================
    # 4. USB MICROPHONE DISCOVERY & VOICE ACTIVITY DETECTION (VAD)
    # ==========================================================================
    def _find_input_device_and_samplerate(self):
        if not HAS_SOUNDDEVICE:
            return None, 16000

        chosen_device = None
        if self.mic_device is not None:
            try:
                chosen_device = int(self.mic_device)
            except ValueError:
                chosen_device = self.mic_device
        else:
            try:
                devices = sd.query_devices()
                for idx, dev in enumerate(devices):
                    if dev.get('max_input_channels', 0) > 0:
                        name = dev.get('name', '').lower()
                        if any(k in name for k in ['usb', 'mic', 'audio', 'uac', 'c-media', 'headset', 'realsense', 'camera']):
                            print(f"  [VOICE AUTO-SELECT] Detected USB Microphone [{idx}]: {dev['name']}")
                            chosen_device = idx
                            break

                if chosen_device is None:
                    default_in = sd.default.device[0]
                    if default_in is not None and default_in >= 0:
                        print(f"  [VOICE AUTO-SELECT] Using Default Input [{default_in}]: {devices[default_in]['name']}")
                        chosen_device = default_in
            except Exception as e:
                print(f"  [VOICE WARNING] Error discovering microphone: {e}")

        # Candidate sample rates to test on hardware
        candidate_rates = [16000, 44100, 48000, 32000, 22050, 8000]
        try:
            if chosen_device is not None:
                dev_info = sd.query_devices(chosen_device)
                def_rate = int(dev_info.get('default_samplerate', 44100))
                if def_rate not in candidate_rates:
                    candidate_rates.insert(1, def_rate)
        except Exception:
            pass

        for rate in candidate_rates:
            try:
                sd.check_input_settings(device=chosen_device, samplerate=rate, channels=1, dtype='float32')
                print(f"  [VOICE CONFIG] Hardware sample rate negotiated: {rate} Hz (Device={chosen_device})")
                return chosen_device, rate
            except Exception:
                continue

        print("  [VOICE WARNING] Falling back to 16000 Hz")
        return chosen_device, 16000

    def _send_audio_to_server(self, wav_bytes: bytes):
        if not self.remote_voice_enabled or not HAS_REQUESTS or self.voice_session is None:
            return
        with self.upload_lock:
            if self.is_uploading:
                return
            self.is_uploading = True

        def _worker():
            try:
                if not self.remote_voice_enabled:
                    return
                t0 = time.monotonic()
                files = {'audio': ('mic.wav', wav_bytes, 'audio/wav')}
                # Generic Qwen replies run on CPU to protect vision CUDA and may
                # take longer on their first lazy model load. Voice has its own
                # session/thread, so this does not block camera delivery.
                resp = self.voice_session.post(self.voice_endpoint, files=files, timeout=(1.0, 45.0))
                if resp.status_code == 200:
                    data = resp.json()
                    transcript = data.get('transcript', '')
                    resp_text = data.get('response_text', '')
                    action = data.get('action', 'none')
                    audio_b64 = data.get('audio_base64', '')
                    rtt_ms = (time.monotonic() - t0) * 1000.0

                    if transcript:
                        print(f'  [VOICE SUCCESS] Heard: "{transcript}" -> Robot: "{resp_text}" (Action: {action}, {rtt_ms:.0f}ms)')
                    else:
                        print(f'  [VOICE UNRECOGNIZED] Robot: "{resp_text}" ({rtt_ms:.0f}ms)')

                    if audio_b64 and self.remote_voice_enabled:
                        audio_bytes = base64.b64decode(audio_b64)
                        self.audio_playback_queue.put(audio_bytes)
                else:
                    try:
                        error_data = resp.json()
                        if error_data.get('voice_enabled') is False:
                            self._set_remote_voice_enabled(False)
                    except Exception:
                        pass
                    print(f'  [VOICE HTTP ERROR] Server returned HTTP {resp.status_code}')
            except Exception as e:
                print(f'  [VOICE ERROR] Failed to send audio to server: {e}')
            finally:
                with self.upload_lock:
                    self.is_uploading = False
        self.is_fetching_audio = False

        threading.Thread(target=_worker, daemon=True).start()

    def _mic_vad_worker_loop(self):
        """C-callback non-blocking audio capture with zero ALSA xrun or GIL contention."""
        if not HAS_SOUNDDEVICE:
            print('  [VOICE WARNING] sounddevice is not available. Mic capture disabled.')
            return

        input_device, hw_samplerate = self._find_input_device_and_samplerate()
        chunk_size = int(hw_samplerate * 0.1)  # 100ms blocks
        audio_raw_queue = queue.Queue(maxsize=50)

        def _audio_callback(indata, frames, time_info, status):
            if self.remote_voice_enabled and indata is not None and not self.is_speaking.is_set():
                try:
                    audio_raw_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

        print(f'  [VOICE] Microphone listening at {hw_samplerate} Hz (Callback VAD, Threshold={self.vad_threshold:.4f})...')

        try:
            with sd.InputStream(
                device=input_device,
                samplerate=hw_samplerate,
                channels=1,
                dtype='float32',
                blocksize=chunk_size,
                callback=_audio_callback,
            ):
                recording = False
                recorded_frames = []
                silence_chunks = 0
                silence_chunks_limit = int(self.silence_timeout / 0.1)
                max_record_chunks = int(10.0 / 0.1)

                while self.running:
                    if not self.remote_voice_enabled:
                        recording = False
                        recorded_frames.clear()
                        silence_chunks = 0
                        while True:
                            try:
                                audio_raw_queue.get_nowait()
                            except queue.Empty:
                                break
                        time.sleep(0.1)
                        continue
                    try:
                        chunk = audio_raw_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if self.is_speaking.is_set():
                        recording = False
                        recorded_frames.clear()
                        silence_chunks = 0
                        continue

                    rms = math.sqrt(float(np.mean(chunk**2))) if chunk.size > 0 else 0.0

                    if not recording:
                        if rms >= self.vad_threshold:
                            recording = True
                            recorded_frames = [chunk.copy()]
                            silence_chunks = 0
                            print(f'  [MIC] ??? Speech detected (RMS: {rms:.4f} > {self.vad_threshold:.4f}), recording...')
                    else:
                        recorded_frames.append(chunk.copy())
                        if rms < self.vad_threshold:
                            silence_chunks += 1
                        else:
                            silence_chunks = 0

                        if silence_chunks >= silence_chunks_limit or len(recorded_frames) >= max_record_chunks:
                            recording = False
                            silence_chunks = 0
                            duration_sec = len(recorded_frames) * 0.1

                            if duration_sec >= 0.4:
                                print(f'  [MIC] ?? Speech captured ({duration_sec:.1f}s, {hw_samplerate}Hz). Resampling to 16kHz & uploading...')
                                full_audio = np.concatenate(recorded_frames, axis=0).flatten()
                                if hw_samplerate != 16000:
                                    full_audio = resample_audio(full_audio, hw_samplerate, 16000)
                                int_samples = (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                                bio = io.BytesIO()
                                with wave.open(bio, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(16000)
                                    wf.writeframes(int_samples.tobytes())
                                self._send_audio_to_server(bio.getvalue())
                            else:
                                print('  [MIC] Discarded noise click (< 0.4s)')
                            recorded_frames.clear()
        except Exception as e:
            print(f'  [VOICE ERROR] Microphone callback error: {e}')

    # ==========================================================================
    # PIPELINED NETWORK SENDER WORKER
    # ==========================================================================
    def _network_sender_loop(self):
        """Dedicated thread: continuously sends newest prepared frames with zero pacing block."""
        consecutive_net_errors = 0
        worker_session = None
        if HAS_REQUESTS:
            # Each worker owns its Session; requests.Session state is not shared
            # between concurrent threads, while HTTP keep-alive is preserved.
            worker_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
            worker_session.mount('http://', adapter)
            worker_session.mount('https://', adapter)
        try:
            while self.running:
                try:
                    item = self.outgoing_frame_queue.get(timeout=0.2)
                    jpeg_bytes, telem, capture_stamp = item
                except queue.Empty:
                    continue

                server_response = self._send_frame_http(jpeg_bytes, telem, capture_stamp, worker_session)

                if server_response and server_response.get('success'):
                    consecutive_net_errors = 0
                    if 'voice_enabled' in server_response:
                        self._set_remote_voice_enabled(server_response['voice_enabled'])
                    now_mono = time.monotonic()
                    with self.network_stats_lock:
                        self.delivered_frame_times.append(now_mono)
                        cutoff = now_mono - 5.0
                        self.delivered_frame_times = [t for t in self.delivered_frame_times if t >= cutoff]
                    motor_cmd = server_response.get('command')
                    if motor_cmd:
                        self._write_serial(motor_cmd)
                    if server_response.get('has_voice_pending') and not self.is_fetching_audio:
                        self._fetch_pending_audio_async()
                else:
                    consecutive_net_errors += 1
                    if consecutive_net_errors % 30 == 1:
                        detail = f' ({self.last_video_error})' if self.last_video_error else ''
                        print(f'  [NETWORK] Menunggu Laptop Server di {self.server_url}{detail}...')
                    if consecutive_net_errors >= 15:
                        self._write_serial('<S,0,0.500,0>\n')

                self.outgoing_frame_queue.task_done()
        finally:
            if worker_session is not None:
                worker_session.close()

    def _delivered_fps(self) -> float:
        with self.network_stats_lock:
            times = list(self.delivered_frame_times)
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return (len(times) - 1) / elapsed if elapsed > 0 else 0.0

    def _send_frame_http(self, jpeg_bytes: bytes, telemetry: dict, capture_stamp: float,
                         session=None) -> Optional[dict]:
        if HAS_REQUESTS and session is not None:
            try:
                files = {'frame': ('camera.jpg', jpeg_bytes, 'image/jpeg')}
                data = {
                    'telemetry': json.dumps(telemetry),
                    'capture_stamp': str(capture_stamp),
                }
                resp = session.post(self.frame_endpoint, files=files, data=data, timeout=(1.5, 2.0))
                if resp.status_code == 200:
                    self.last_video_error = ''
                    return resp.json()
                self.last_video_error = f'HTTP {resp.status_code}'
            except Exception as exc:
                self.last_video_error = str(exc)
                return None
        return None

    def _fetch_pending_audio_async(self):
        if (not self.remote_voice_enabled or not HAS_REQUESTS
                or self.voice_session is None or self.is_fetching_audio):
            return
        self.is_fetching_audio = True

        def _fetcher():
            try:
                resp = self.voice_session.get(self.pending_audio_endpoint, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('voice_enabled') is False:
                        self._set_remote_voice_enabled(False)
                    if (self.remote_voice_enabled and data.get('has_audio')
                            and data.get('audio_base64')):
                        audio_bytes = base64.b64decode(data['audio_base64'])
                        text = data.get('text', '')
                        reason = data.get('reason', '')
                        print(f'  [PROACTIVE VOICE] Playing {reason}: "{text}"')
                        self.audio_playback_queue.put(audio_bytes)
            except Exception:
                pass
            finally:
                self.is_fetching_audio = False

        threading.Thread(target=_fetcher, daemon=True).start()

    def run(self):
        serial_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        serial_thread.start()

        cam_thread = threading.Thread(target=self._camera_worker_loop, daemon=True)
        cam_thread.start()

        # Dual-Worker Pipelined Streamers (Delivers 15+ FPS over network RTT)
        threading.Thread(target=self._network_sender_loop, daemon=True).start()
        threading.Thread(target=self._network_sender_loop, daemon=True).start()

        if self.enable_voice:
            spk_thread = threading.Thread(target=self._speaker_worker_loop, daemon=True)
            spk_thread.start()

            mic_thread = threading.Thread(target=self._mic_vad_worker_loop, daemon=True)
            mic_thread.start()

        print('  [INIT] Waiting for camera initialization...')
        if not self.camera_ready.wait(timeout=25.0):
            print('  [ERROR] Camera initialization timeout.')
            self.running = False
            return

        print(f'  [STREAMING] Active at stable {self.target_fps:.1f} FPS. Sending to {self.frame_endpoint}...')
        frame_interval = 1.0 / self.target_fps
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        last_pacing_time = time.monotonic()
        fps_history = []
        last_fps_log_time = time.monotonic()

        try:
            while self.running:
                t_loop_start = time.monotonic()

                with self.frame_lock:
                    frame = self.latest_frame.copy() if self.latest_frame is not None else None
                    capture_stamp = self.latest_frame_stamp

                if frame is None:
                    time.sleep(0.005)
                    continue

                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                if not success:
                    continue
                jpeg_bytes = buffer.tobytes()

                with self.telemetry_lock:
                    telem = dict(self.latest_telemetry)

                # Pipelined enqueue: discard old unsent frame if queue is full (0 latency guarantee)
                try:
                    self.outgoing_frame_queue.put_nowait((jpeg_bytes, telem, capture_stamp))
                except queue.Full:
                    try:
                        self.outgoing_frame_queue.get_nowait()
                        self.outgoing_frame_queue.task_done()
                    except queue.Empty:
                        pass
                    try:
                        self.outgoing_frame_queue.put_nowait((jpeg_bytes, telem, capture_stamp))
                    except queue.Full:
                        pass

                # Track FPS
                now_mono = time.monotonic()
                dt = now_mono - last_pacing_time
                last_pacing_time = now_mono
                if dt > 0:
                    fps_history.append(1.0 / dt)
                    if len(fps_history) > 30:
                        fps_history.pop(0)

                if now_mono - last_fps_log_time >= 2.0:
                    current_fps = sum(fps_history) / max(1, len(fps_history))
                    delivered_fps = self._delivered_fps()
                    print(f'  [STREAM] Capture: {current_fps:4.1f} FPS | Delivered: {delivered_fps:4.1f} FPS | Arduino: {telem.get("state")} | Sonar: [{telem.get("us_left",0):.0f}, {telem.get("us_center",0):.0f}, {telem.get("us_right",0):.0f}]cm')
                    last_fps_log_time = now_mono

                elapsed = time.monotonic() - t_loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print('\n  [SHUTDOWN] Exiting gracefully...')
        finally:
            self.running = False
            self._write_serial('<S,0,0.500,0>\n')
            with self.serial_lock:
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
            print('  [CLEANUP] Serial closed. Program terminated.')


def main():
    parser = argparse.ArgumentParser(description='Hafibot Low-Latency Edge Streamer & Voice Gateway')
    parser.add_argument('--server', type=str, default='http://127.0.0.1:5000', help='Laptop Server URL (e.g. http://100.86.76.95:5000)')
    parser.add_argument('--serial', type=str, default='/dev/ttyUSB0', help='Arduino serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baudrate (default: 115200)')
    parser.add_argument('--fps', type=float, default=15.0, help='Target streaming FPS (default: 15.0)')
    parser.add_argument('--quality', type=int, default=60, help='JPEG compression quality 30-95 (default: 60)')
    parser.add_argument('--enable-voice', action='store_true', default=True, help='Enable USB mic capture & I2S speaker playback')
    parser.add_argument('--no-voice', dest='enable_voice', action='store_false', help='Disable voice subsystem')
    parser.add_argument('--mic-device', type=str, default=None, help='Microphone device name or index in sounddevice (default: auto-detect USB mic)')
    parser.add_argument('--speaker-device', type=str, default='default', help='ALSA speaker device (e.g. plughw:tegrasndt210ref,0 or default)')
    parser.add_argument('--vad-threshold', type=float, default=0.030, help='Mic VAD RMS energy threshold (default: 0.030)')
    parser.add_argument('--silence-timeout', type=float, default=0.8, help='VAD silence timeout in seconds (default: 0.8)')
    args = parser.parse_args()

    agent = JetsonEdgeAgent(
        server_url=args.server,
        serial_port=args.serial,
        baudrate=args.baud,
        target_fps=args.fps,
        jpeg_quality=args.quality,
        enable_voice=args.enable_voice,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        vad_threshold=args.vad_threshold,
        silence_timeout=args.silence_timeout,
    )
    agent.run()


if __name__ == '__main__':
    main()
