#!/usr/bin/env python3
"""
ROS Node untuk Fall Detection menggunakan 1D CNN-LSTM.
Dioptimasi untuk dijalankan pada NVIDIA Jetson Nano.

Arsitektur Composed Node (Zero-copy Python):
- Pembacaan kamera (GStreamer) langsung tergabung di node ini (tanpa topic /usb_cam/image_raw)
- Pose estimation (MediaPipe CPU) berjalan di background
- Menggunakan sensor_msgs/CompressedImage (20KB/frame) ke Flask (Bukan 900KB)
"""

import rospy
import cv2
import sys
import time
import json
import threading
import numpy as np
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point

# -- Mengatasi import dari root folder project asal --
import os
# GPU rendering via HDMI dummy plug â€” DISPLAY=:0 diset oleh docker-compose
# Hapus flag software rendering jika ada (bawaan dari konfigurasi lama)
if 'LIBGL_ALWAYS_SOFTWARE' in os.environ:
    del os.environ['LIBGL_ALWAYS_SOFTWARE']
import mediapipe as mp

class FallDetectionNode:
    def __init__(self):
        rospy.init_node('fall_detection_node', anonymous=True)

        # -- Parameter ROS --
        self.threshold = rospy.get_param('~threshold', 0.90)
        self.src_path = rospy.get_param('~src_path', '/workspace/FallDetectionCNN')
        self.pose_loss_grace_sec = rospy.get_param('~pose_loss_grace_sec', 2.0)
        self.fall_hold_sec = rospy.get_param('~fall_hold_sec', 5.0)
        self.pose_reacquire_confirmation_frames = int(
            rospy.get_param('~pose_reacquire_confirmation_frames', 3)
        )
        self.enable_posture_filter = bool(
            rospy.get_param('~enable_posture_filter', True)
        )
        self.min_upright_torso_angle = float(
            rospy.get_param('~min_upright_torso_angle', 50.0)
        )
        self.min_upright_aspect_ratio = float(
            rospy.get_param('~min_upright_aspect_ratio', 1.25)
        )
        self.min_dy_torso = float(
            rospy.get_param('~min_dy_torso', 0.04)
        )
        self.recording_dir = rospy.get_param(
            '~recording_dir',
            '/workspace/FallDetectionCNN/data/jetson_recordings'
        )
        self.recording_enabled_on_start = rospy.get_param(
            '~recording_enabled_on_start', False
        )
        self.stats_publish_hz = max(
            0.0, float(rospy.get_param('~stats_publish_hz', 1.0))
        )

        # -- Parameter kalibrasi shoulder-to-distance (polyfit 2 orde) --
        # Nilai default dari referensi ProgramNael (kamera 640x480, bahu dewasa).
        # x = raw shoulder-distance dalam pixel, y = jarak sesungguhnya dalam cm.
        # y = A*x^2 + B*x + C -> coefficients di-load dari rosparam list [A, B, C]
        # Kamu bisa re-kalibrasi via `scripts/calibrate_shoulder_distance.py`.
        default_coeff = [0.0244, -6.5757, 495.7]  # hasil polyfit ProgramNael
        self.shoulder_coeff = rospy.get_param('~shoulder_polyfit', default_coeff)
        # Lebar frame kamera (untuk konversi koordinat pixel -> normalized)
        # MediaPipe sudah kasih normalized 0..1 jadi ini info telemetry saja
        self.frame_width  = rospy.get_param('~frame_width', 640)
        self.frame_height = rospy.get_param('~frame_height', 480)
        self.target_min_shoulder_visibility = float(
            rospy.get_param('~target_min_shoulder_visibility', 0.6)
        )
        self.target_confirmation_frames = int(
            rospy.get_param('~target_confirmation_frames', 3)
        )
        self.target_distance_ema_alpha = float(
            rospy.get_param('~target_distance_ema_alpha', 0.35)
        )
        self.target_center_ema_alpha = float(
            rospy.get_param('~target_center_ema_alpha', 0.4)
        )
        self.target_max_distance_jump_cm = float(
            rospy.get_param('~target_max_distance_jump_cm', 90.0)
        )
        self.target_max_center_jump = float(
            rospy.get_param('~target_max_center_jump', 0.22)
        )
        self.enable_torso_fusion = bool(
            rospy.get_param('~enable_torso_fusion', True)
        )
        self.torso_shoulder_ratio = float(
            rospy.get_param('~torso_shoulder_ratio', 0.80)
        )
        self.target_min_hip_visibility = float(
            rospy.get_param('~target_min_hip_visibility', 0.5)
        )
        # IMX219 mode 2 is a native 1920x1080@30 mode. Pinning the sensor mode
        # prevents Argus from falling back to mode 5 (1280x720@120), which can
        # time out on Jetson Nano when the requested caps are unsupported.
        self.camera_sensor_id = int(rospy.get_param('~camera_sensor_id', 0))
        self.camera_sensor_mode = int(rospy.get_param('~camera_sensor_mode', 2))
        self.camera_capture_width = int(
            rospy.get_param('~camera_capture_width', 1920)
        )
        self.camera_capture_height = int(
            rospy.get_param('~camera_capture_height', 1080)
        )
        self.camera_capture_fps = int(rospy.get_param('~camera_capture_fps', 30))
        self.camera_output_fps = float(rospy.get_param('~camera_output_fps', 15.0))
        self.stream_jpeg_quality = min(
            95, max(30, int(rospy.get_param('~stream_jpeg_quality', 60)))
        )
        
        if self.src_path not in sys.path:
            sys.path.insert(0, self.src_path)
            
        try:
            from src.inference import FallDetector
            from src.human_following import ShoulderTargetFilter
            from src.jetson_recording import LandmarkSessionRecorder
        except ImportError as e:
            rospy.logerr(f"Gagal import FallDetector. Pastikan src_path benar: {self.src_path}")
            rospy.logerr(str(e))
            sys.exit(1)

        # -- Inisialisasi Detektor --
        rospy.loginfo("Inisialisasi Fall Detector Model...")
        self.detector = FallDetector(
            confidence_threshold=self.threshold,
            pose_loss_grace_sec=self.pose_loss_grace_sec,
            fall_hold_sec=self.fall_hold_sec,
            pose_reacquire_confirmation_frames=(
                self.pose_reacquire_confirmation_frames
            ),
            enable_posture_filter=self.enable_posture_filter,
            min_upright_torso_angle=self.min_upright_torso_angle,
            min_upright_aspect_ratio=self.min_upright_aspect_ratio,
            min_dy_torso=self.min_dy_torso,
        )
        self.landmarker = self.detector._create_landmarker()
        self.target_filter = ShoulderTargetFilter(
            coefficients=self.shoulder_coeff,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            min_visibility=self.target_min_shoulder_visibility,
            confirmation_frames=self.target_confirmation_frames,
            distance_alpha=self.target_distance_ema_alpha,
            center_alpha=self.target_center_ema_alpha,
            max_distance_jump_cm=self.target_max_distance_jump_cm,
            max_center_jump=self.target_max_center_jump,
            enable_torso_fusion=self.enable_torso_fusion,
            torso_shoulder_ratio=self.torso_shoulder_ratio,
            min_hip_visibility=self.target_min_hip_visibility,
        )
        self.recorder = LandmarkSessionRecorder(self.recording_dir)

        # -- Setup ROS Publisher (Menggunakan CompressedImage!) --
        self.image_pub = rospy.Publisher('/fall_detection/image_overlay', CompressedImage, queue_size=1)
        self.alert_pub = rospy.Publisher('/fall_detection/alert', Bool, queue_size=1)
        # /person_centroid : centroid tubuh (untuk UI/debug)
        #   x, y = normalized 0..1 (kiri/atas), z = 1.0 jika terdeteksi
        self.centroid_pub = rospy.Publisher('/person_centroid', Point, queue_size=1)
        # /person_target : payload untuk human following (robot)
        #   x = center_x_norm (0..1), y = distance_cm (0 = lost), z = detected (0/1)
        self.target_pub = rospy.Publisher('/person_target', Point, queue_size=1)
        # /fall_detection/landmarks : landmark pose sebagai JSON (untuk privacy notification)
        self.landmarks_pub = rospy.Publisher('/fall_detection/landmarks', String, queue_size=1)
        # Statistik berbasis timestamp pemrosesan, bukan sekadar 1 / latency model.
        self.processing_stats_pub = rospy.Publisher(
            '/fall_detection/processing_stats', String, queue_size=1, latch=True
        )
        self.set_recording_service = rospy.Service(
            '~set_recording', SetBool, self._set_recording_callback
        )
        self.recording_status_service = rospy.Service(
            '~recording_status', Trigger, self._recording_status_callback
        )

        # -- Decoupled Processing State --
        self._latest_frame = None           # Frame mentah terbaru dari kamera
        self._latest_frame_id = 0
        self._latest_frame_capture_unix = 0.0
        self._latest_frame_capture_monotonic = 0.0
        self._latest_frame_lock = threading.Lock()
        self._latest_result = None          # Hasil pose+fall detection terakhir
        self._latest_landmarks = None       # Landmarks terakhir
        self._result_lock = threading.Lock()

        # -- Variabel Perhitungan FPS --
        self._stream_fps_history = []       # FPS streaming (cepat)
        self._process_interval_history = [] # Jarak waktu antar pose sample
        self._processing_latency_history = []
        self._pose_detected_history = []
        self._processed_frame_count = 0
        self._first_process_start = None
        self._last_process_start = None
        self._last_stats_publish_monotonic = 0.0
        self._last_stream_time = time.time()
        self._latest_stream_jpeg_bytes = 0

        if self.recording_enabled_on_start:
            self._start_recording()

        rospy.on_shutdown(self._shutdown)

        # -- Mulai thread pose processing --
        self._pose_thread = threading.Thread(target=self._pose_processing_loop, daemon=True)
        self._pose_thread.start()

        # -- Mulai thread camera internal --
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()

        rospy.loginfo("Arsitektur Composed Node aktif! Zero-copy memory antara kamera dan MediaPipe.")

    def _camera_loop(self):
        """Thread terpisah: baca dari kamera GStreamer langsung, tanpa topic ROS."""
        sensor_mode_str = f"sensor-mode={self.camera_sensor_mode} " if self.camera_sensor_mode >= 0 else ""
        gstreamer_pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_sensor_id} {sensor_mode_str}! "
            f"video/x-raw(memory:NVMM), width=(int){self.camera_capture_width}, "
            f"height=(int){self.camera_capture_height}, format=(string)NV12, "
            f"framerate=(fraction){self.camera_capture_fps}/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.frame_width}, "
            f"height=(int){self.frame_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink max-buffers=1 drop=true sync=false"
        )
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            rospy.logerr("Gagal membuka kamera CSI di dalam fall_detection_node")
            return
            
        rospy.loginfo(
            "CSI camera mode %d: %dx%d@%d -> %dx%d@%.1f FPS",
            self.camera_sensor_mode,
            self.camera_capture_width,
            self.camera_capture_height,
            self.camera_capture_fps,
            self.frame_width,
            self.frame_height,
            self.camera_output_fps,
        )
        rate = rospy.Rate(self.camera_output_fps)
        
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn_throttle(5, "Gagal membaca frame kamera.")
                rate.sleep()
                continue
                
            # Simpan frame terbaru untuk pose processing thread
            capture_unix = time.time()
            capture_monotonic = time.monotonic()
            with self._latest_frame_lock:
                self._latest_frame = frame.copy()
                self._latest_frame_id += 1
                self._latest_frame_capture_unix = capture_unix
                self._latest_frame_capture_monotonic = capture_monotonic
                
            # Hitung FPS streaming
            now = time.time()
            dt = now - self._last_stream_time
            self._last_stream_time = now
            if dt > 0:
                self._stream_fps_history.append(1.0 / dt)
                if len(self._stream_fps_history) > 30:
                    self._stream_fps_history.pop(0)
            stream_fps = np.mean(self._stream_fps_history) if self._stream_fps_history else 0

            # Ambil hasil terakhir dari pose processing (non-blocking)
            with self._result_lock:
                result = self._latest_result
                landmarks = self._latest_landmarks

            # Gambar overlay menggunakan hasil terakhir yang tersedia
            if result is not None:
                annotated_frame = self.detector.draw_overlay(frame, result, landmarks, stream_fps)
            else:
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"FPS: {stream_fps:.1f} (menunggu pose...)", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Compress gambar menjadi JPEG untuk dikirim via ROS Topic (20KB vs 900KB)
            try:
                msg = CompressedImage()
                # Preserve capture time so downstream stream latency can be
                # measured instead of stamping an already-processed JPEG.
                msg.header.seq = self._latest_frame_id
                msg.header.stamp = rospy.Time.from_sec(capture_unix)
                msg.format = "jpeg"
                # Quality 60 materially reduces pose-overlay bandwidth while
                # leaving the raw frame used by MediaPipe untouched.
                encoded, buffer = cv2.imencode(
                    '.jpg', annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.stream_jpeg_quality],
                )
                if not encoded:
                    raise RuntimeError("OpenCV gagal encode JPEG")
                msg.data = buffer.tobytes()
                self._latest_stream_jpeg_bytes = len(msg.data)
                self.image_pub.publish(msg)
            except Exception as e:
                rospy.logerr_throttle(2, f"Gagal kompresi gambar: {e}")

            # Publikasi status jatuh
            is_fall = result.get("is_fall", False) if result else False
            alert_msg = Bool()
            alert_msg.data = is_fall
            self.alert_pub.publish(alert_msg)

            if is_fall:
                rospy.logwarn_throttle(1.0, "!!! FALL DETECTED !!!")

            # Publish landmarks JSON untuk privacy-preserving notifications
            if landmarks is not None:
                try:
                    lm_data = [{'x': float(lm.x), 'y': float(lm.y),
                                'z': float(lm.z), 'v': float(lm.visibility)}
                               for lm in landmarks]
                    self.landmarks_pub.publish(json.dumps(lm_data))
                except Exception:
                    pass  # Jangan crash jika serialisasi gagal

            self._publish_processing_stats(stream_fps, capture_monotonic)
                
            rate.sleep()
            
        cap.release()

    def _pose_processing_loop(self):
        """Thread terpisah: jalankan MediaPipe pose estimation secara kontinu."""
        rospy.loginfo("Pose processing thread dimulai...")
        
        # Throttling: saat tidak ada orang, MediaPipe menjalankan model deteksi penuh
        # yang sangat berat. Kita batasi agar tidak membombardir GPU.
        self._no_detection_count = 0
        DETECTION_SKIP_INTERVAL = 3  # Saat tidak ada orang, proses 1 dari setiap 3 frame
        last_consumed_frame_id = -1
        
        while not rospy.is_shutdown():
            with self._latest_frame_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
                capture_unix = self._latest_frame_capture_unix
                capture_monotonic = self._latest_frame_capture_monotonic
            
            if frame is None or frame_id == last_consumed_frame_id:
                time.sleep(0.01)
                continue

            # Throttle: saat tidak ada orang terdeteksi, skip beberapa frame
            # agar GPU tidak dipaksa menjalankan model deteksi penuh terus-menerus.
            if self._no_detection_count > 0:
                self._no_detection_count -= 1
                last_consumed_frame_id = frame_id
                time.sleep(0.005)
                continue

            last_consumed_frame_id = frame_id
            process_start = time.monotonic()

            try:
                # Konversi untuk MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Pose estimation (YOLOv8 TensorRT / ONNX / PyTorch atau MediaPipe)
                if hasattr(self.landmarker, 'detect'):
                    landmarks = self.landmarker.detect(frame)
                elif hasattr(self.landmarker, 'detect_for_video'):
                    timestamp_ms = int(time.time() * 1000)
                    if not hasattr(self, '_last_ts'):
                        self._last_ts = timestamp_ms - 1
                    if timestamp_ms <= self._last_ts:
                        timestamp_ms = self._last_ts + 1
                    self._last_ts = timestamp_ms
                    
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    result_mp = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    landmarks = None
                    if result_mp.pose_landmarks and len(result_mp.pose_landmarks) > 0:
                        landmarks = result_mp.pose_landmarks[0]
                else:
                    # Legacy Solutions API (MediaPipe 0.8.x di Jetson Nano)
                    result_mp = self.landmarker.process(frame_rgb)
                    
                    landmarks = None
                    if result_mp.pose_landmarks:
                        landmarks = result_mp.pose_landmarks.landmark

                # Throttling: jika tidak terdeteksi, skip beberapa frame berikutnya
                if landmarks is None:
                    self._no_detection_count = DETECTION_SKIP_INTERVAL

                # Hitung Centroid untuk Human Following
                centroid_msg = Point()
                target_msg = Point()
                if landmarks:
                    if len(landmarks) == 13:
                        # YOLO 13-joint: 1=L-Shoulder, 2=R-Shoulder, 7=L-Hip, 8=R-Hip
                        cx = (landmarks[1].x + landmarks[2].x + landmarks[7].x + landmarks[8].x) / 4.0
                        cy = (landmarks[1].y + landmarks[2].y + landmarks[7].y + landmarks[8].y) / 4.0
                    else:
                        # MediaPipe 33-joint: 11=L-Shoulder, 12=R-Shoulder, 23=L-Hip, 24=R-Hip
                        cx = (landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) / 4.0
                        cy = (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) / 4.0
                    centroid_msg.x = cx
                    centroid_msg.y = cy
                    centroid_msg.z = 1.0

                    target = self.target_filter.update(landmarks)
                    target_msg.x = target['center_x']
                    target_msg.y = float(target['distance_cm'])
                    target_msg.z = 1.0 if target['detected'] else 0.0
                    if not target['detected'] and target['reason'] != 'confirming':
                        rospy.logwarn_throttle(
                            2.0,
                            "Following target ditolak: %s",
                            target['reason'],
                        )
                else:
                    self.target_filter.reset()
                    centroid_msg.x = 0.5
                    centroid_msg.y = 0.5
                    centroid_msg.z = 0.0
                    target_msg.x = 0.5
                    target_msg.y = 0.0
                    target_msg.z = 0.0

                self.centroid_pub.publish(centroid_msg)
                self.target_pub.publish(target_msg)

                # Proses Landmark untuk deteksi jatuh (model CNN-LSTM di CUDA)
                result = self.detector.process_frame(landmarks)

                # Latency menunjukkan biaya satu inferensi. Throughput aktual
                # dihitung dari interval antar timestamp mulai pemrosesan.
                process_end = time.monotonic()
                processing_latency = max(0.0, process_end - process_start)
                sample_interval = None
                if self._last_process_start is not None:
                    sample_interval = max(
                        0.0, process_start - self._last_process_start
                    )
                if self._first_process_start is None:
                    self._first_process_start = process_start
                self._last_process_start = process_start
                self._processed_frame_count += 1

                with self._result_lock:
                    if sample_interval and sample_interval > 0.0:
                        self._process_interval_history.append(sample_interval)
                        if len(self._process_interval_history) > 60:
                            self._process_interval_history.pop(0)
                    self._processing_latency_history.append(processing_latency)
                    if len(self._processing_latency_history) > 60:
                        self._processing_latency_history.pop(0)
                    self._pose_detected_history.append(landmarks is not None)
                    if len(self._pose_detected_history) > 60:
                        self._pose_detected_history.pop(0)

                try:
                    self.recorder.record_frame(
                        landmarks=landmarks,
                        capture_unix_sec=capture_unix,
                        capture_monotonic_sec=capture_monotonic,
                        processing_start_monotonic_sec=process_start,
                        processing_end_monotonic_sec=process_end,
                        result=result,
                        camera_frame_id=frame_id,
                    )
                except Exception as e:
                    rospy.logerr_throttle(2, "Gagal merekam landmark: " + str(e))

                # Simpan hasil untuk dipakai oleh loop kamera utama
                with self._result_lock:
                    self._latest_result = result
                    self._latest_landmarks = landmarks

            except Exception as e:
                rospy.logerr("Error pose processing: " + str(e))
                time.sleep(0.1)

    def _runtime_recording_metadata(self):
        checkpoint = getattr(self.detector, 'checkpoint', {})
        return {
            'mediapipe_version': getattr(mp, '__version__', 'unknown'),
            'mediapipe_api': (
                'tasks_pose_landmarker_video'
                if hasattr(self.landmarker, 'detect_for_video')
                else 'legacy_solutions_pose'
            ),
            'pose_model_variant': 'lite',
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'checkpoint_version': checkpoint.get('checkpoint_version'),
            'checkpoint_epoch': checkpoint.get('epoch'),
            'training_config': checkpoint.get('training_config', {}),
        }

    def _start_recording(self):
        label = rospy.get_param('~recording_label', 'UNLABELED')
        subject_id = rospy.get_param('~recording_subject', 'unknown_subject')
        activity = rospy.get_param('~recording_activity', 'unspecified')
        path = self.recorder.start(
            label=label,
            subject_id=subject_id,
            activity=activity,
            metadata=self._runtime_recording_metadata(),
        )
        rospy.loginfo("Landmark recording dimulai: " + str(path))
        return path

    def _set_recording_callback(self, request):
        try:
            if request.data:
                if self.recorder.is_recording:
                    return SetBoolResponse(
                        success=True,
                        message="Recording sudah aktif: " + str(self.recorder.path),
                    )
                path = self._start_recording()
                return SetBoolResponse(
                    success=True,
                    message="Recording dimulai: " + str(path),
                )

            summary = self.recorder.stop()
            return SetBoolResponse(
                success=True,
                message="Recording dihentikan: " + json.dumps(summary),
            )
        except Exception as error:
            rospy.logerr("Gagal mengubah status recording: " + str(error))
            return SetBoolResponse(success=False, message=str(error))

    def _recording_status_callback(self, _request):
        try:
            return TriggerResponse(
                success=True,
                message=json.dumps(self.recorder.snapshot()),
            )
        except Exception as error:
            return TriggerResponse(success=False, message=str(error))

    def _publish_processing_stats(self, stream_fps, capture_monotonic=None):
        if self.stats_publish_hz <= 0.0:
            return

        now = time.monotonic()
        if now - self._last_stats_publish_monotonic < 1.0 / self.stats_publish_hz:
            return
        self._last_stats_publish_monotonic = now

        with self._result_lock:
            intervals = list(self._process_interval_history)
            latencies = list(self._processing_latency_history)
            detected = list(self._pose_detected_history)

        rolling_fps = 0.0
        if intervals:
            mean_interval = sum(intervals) / len(intervals)
            if mean_interval > 0.0:
                rolling_fps = 1.0 / mean_interval

        effective_fps = 0.0
        if (
            self._processed_frame_count > 1
            and self._first_process_start is not None
            and self._last_process_start is not None
            and self._last_process_start > self._first_process_start
        ):
            effective_fps = (
                (self._processed_frame_count - 1)
                / (self._last_process_start - self._first_process_start)
            )

        payload = {
            'timestamp_unix_sec': time.time(),
            'camera_stream_fps': float(stream_fps),
            'stream_frame_age_ms': (
                max(0.0, (time.monotonic() - capture_monotonic) * 1000.0)
                if capture_monotonic else 0.0
            ),
            'stream_jpeg_size_kb': self._latest_stream_jpeg_bytes / 1024.0,
            'processing_fps_rolling': rolling_fps,
            'processing_fps_effective': effective_fps,
            'processing_latency_ms_mean': (
                1000.0 * sum(latencies) / len(latencies) if latencies else 0.0
            ),
            'pose_detection_rate_rolling': (
                sum(1 for value in detected if value) / float(len(detected))
                if detected else 0.0
            ),
            'processed_frames': self._processed_frame_count,
            'recording': self.recorder.snapshot(),
        }
        self.processing_stats_pub.publish(json.dumps(payload))

    def _shutdown(self):
        try:
            if self.recorder.is_recording:
                summary = self.recorder.stop()
                rospy.loginfo("Landmark recording disimpan: " + json.dumps(summary))
        except Exception as error:
            rospy.logerr("Gagal menutup landmark recording: " + str(error))

        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except Exception:
                pass

    def run(self):
        rospy.spin()
        self._shutdown()

if __name__ == '__main__':
    try:
        node = FallDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

