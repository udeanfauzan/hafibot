"""
chatbot_service.py - Unified Voice Assistant & NLP Service for Hafibot.

Integrates:
  1. DistilBERT Multi-Head Classifier (Intent, Sentiment, Function)
  2. Contextual Middlelayer (Time, BMKG Weather, News)
  3. Your Friend's NLP Reminder Module (c_dateparser + ReminderAdd + ReminderCheck + ReminderCall)
  4. Qwen LoRA Response Generator with Contextual Dialogue Fallback
  5. Google Indonesian Speech Recognition (id-ID)
  6. Text-to-Speech Synthesis (gTTS + 16kHz Mono WAV Conversion with Tone Fallback)
  7. Cross-System Robot Action Bridge (Voice Commands + Fall Confirmation Dialogs)
  8. Proactive Background Reminder & Alarm Dispatcher
"""

import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from dateutil.relativedelta import relativedelta

# Ensure robotchatbot and middlelayer are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / 'robotchatbot' / 'm_middlelayer') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'robotchatbot' / 'm_middlelayer'))
if str(ROOT_DIR / 'robotchatbot') not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / 'robotchatbot'))

try:
    from c_dateparser import reminder_time_normalize, reminder_followup
    from reminder import ReminderAdd, ReminderCheck, ReminderCall
except Exception as e:
    ReminderAdd = None
    ReminderCheck = None
    ReminderCall = None
    reminder_time_normalize = None

logger = logging.getLogger('hafibot_chatbot')


# ==============================================================================
# 1. DISTILBERT MULTI-HEAD CLASSIFIER (OFFLINE LOADABLE)
# ==============================================================================
class DistilBertClassifier(nn.Module):
    def __init__(self, root_dir: Path, device: Optional[torch.device] = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        requested_device = os.getenv('HAFIBOT_DBERT_DEVICE', 'cpu').strip().lower()
        use_cuda = requested_device == 'cuda' and torch.cuda.is_available()
        self.device = device or torch.device('cuda' if use_cuda else 'cpu')

        vocab_dir = self.root_dir / 'models' / 'distilbert-base-uncased'
        if not vocab_dir.exists():
            vocab_dir = self.root_dir / 'robotchatbot' / 'models' / 'distilbert-base-uncased'

        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

        if vocab_dir.exists():
            self.config = DistilBertConfig.from_pretrained(str(vocab_dir))
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(vocab_dir))
        else:
            self.config = DistilBertConfig()
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.encoder = DistilBertModel(self.config)
        hidden = self.config.hidden_size
        sequence = 384
        dropout = 0.1
        intent_head = 15
        sentiment_head = 4
        function_head = 8

        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, intent_head),
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, sentiment_head),
        )
        self.function_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, function_head),
        )

        model_path_2 = self.root_dir / 'robotchatbot' / 'models' / 'DBERTModel2' / 'Model Result.pth'
        model_path_1 = self.root_dir / 'robotchatbot' / 'models' / 'DBERTModel' / 'Model Result.pth'
        model_path = model_path_2 if model_path_2.exists() else model_path_1
        if model_path.exists():
            logger.info('Loading DistilBERT weights from %s', model_path)
            ckpt = torch.load(model_path, map_location='cpu')
            state_dict = ckpt[0]['best_model'] if isinstance(ckpt, list) else ckpt
            self.load_state_dict(state_dict)
        else:
            logger.warning('DistilBERT checkpoint not found at %s', model_path)

        self.to(self.device)
        self.eval()

        labels_dir = self.root_dir / 'robotchatbot' / 'm_distilbert'
        self.intent_labels = self._load_label_map(labels_dir / 'intent_labels.csv', 15)
        self.sentiment_labels = self._load_label_map(labels_dir / 'sentiment_labels.csv', 4)
        self.function_labels = self._load_label_map(labels_dir / 'function_labels.csv', 8)

    def _load_label_map(self, csv_path: Path, default_count: int) -> Dict[int, str]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return dict(zip(df['id'], df['label']))
        return {i: f'label_{i}' for i in range(default_count)}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        cls_token = self.dropout(hidden_state[:, 0])
        return {
            'intent_logits': self.intent_head(cls_token),
            'sentiment_logits': self.sentiment_head(cls_token),
            'function_logits': self.function_head(cls_token),
        }

    def predict(self, text: str) -> Dict[str, str]:
        if not text or not text.strip():
            return {'intent': 'none', 'sentiment': 'neutral', 'function': 'chat'}

        inputs = self.tokenizer(text, truncation=True, max_length=128, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        intent_idx = torch.argmax(outputs['intent_logits'], dim=1).item()
        sentiment_idx = torch.argmax(outputs['sentiment_logits'], dim=1).item()
        func_idx = torch.argmax(outputs['function_logits'], dim=1).item()

        return {
            'intent': self.intent_labels.get(intent_idx, f'intent_{intent_idx}'),
            'sentiment': self.sentiment_labels.get(sentiment_idx, 'neutral'),
            'function': self.function_labels.get(func_idx, 'chat'),
        }


# ==============================================================================
# 2. MIDDLELAYER: EXTERNAL APIS
# ==============================================================================
def get_jakarta_time() -> Dict[str, Any]:
    try:
        from m_middlelayer.middlelayer import real_time
        rt = real_time()
        rt['display'] = f"{rt['day']}, {rt['date']} jam {rt['time'][:5]}"
        return rt
    except Exception:
        now = datetime.now()
        days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
        return {
            'date': now.strftime('%Y-%m-%d'),
            'day': days[now.weekday()],
            'month': months[now.month - 1],
            'year': now.strftime('%Y'),
            'time': now.strftime('%H:%M:%S'),
            'time_12h': now.strftime('%I:%M:%S %p'),
            'time_24h': now.strftime('%H:%M:%S'),
            'datetime': now.isoformat(timespec='seconds'),
            'timezone': 'UTC+07:00',
            'display': f"{days[now.weekday()]}, {now.strftime('%d-%m-%Y')} jam {now.strftime('%H:%M')}",
        }

def fetch_weather_bmkg(adm_code: str = '32.73.02.1005') -> Dict[str, Any]:
    url = f'https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm_code}'
    try:
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            cuaca = data['data'][0]['cuaca'][0][0]['weather_desc']
            suhu = data['data'][0]['cuaca'][0][0]['t']
            kelembapan = data['data'][0]['cuaca'][0][0]['hu']
            return {'cuaca': cuaca, 'suhu': suhu, 'kelembapan': kelembapan, 'success': True}
    except Exception as e:
        logger.debug('BMKG Weather fetch failed: %s', e)
    return {'cuaca': 'Cerah Berawan', 'suhu': '26', 'kelembapan': '75', 'success': False}

def fetch_news_tempo() -> Dict[str, Any]:
    url = 'https://berita-indo-api-next.vercel.app/api/tempo-news/tekno'
    try:
        resp = requests.get(url, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            news = data['data'][0]
            return {'title': news.get('title', ''), 'date': news.get('isoDate', ''), 'success': True}
    except Exception as e:
        logger.debug('News fetch failed: %s', e)
    return {
        'title': 'Perkembangan teknologi AI dan robotika di Indonesia terus berkembang pesat.',
        'date': datetime.now().isoformat(),
        'success': False,
    }


# ==============================================================================
# 3. REMINDER ENGINE (UTILIZING YOUR FRIEND'S REMINDER MODULE)
# ==============================================================================
class ReminderEngine:
    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self.lock = threading.RLock()
        if ReminderAdd is not None:
            self.r_add = ReminderAdd(json_path=str(self.json_path))
            self.r_check = ReminderCheck(json_path=str(self.json_path))
            self.r_call = ReminderCall(json_path=str(self.json_path))
        else:
            self.r_add = None
            self.r_check = None
            self.r_call = None

    @property
    def reminders(self) -> List[Dict[str, Any]]:
        if self.r_add:
            return self.r_add.reminders
        return []

    def add_reminder(self, task: str, target_dt: Any, repeat_days: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if isinstance(target_dt, datetime):
                startdate = target_dt.isoformat()
            else:
                startdate = str(target_dt)
            item = {
                'uuid': str(uuid.uuid4()),
                'task': task.strip(),
                'startdate': startdate,
                'repeat_type': repeat_days,
                'called_upon': False,
                'is_prayer': False,
            }
            if self.r_add:
                self.r_add.reminders.append(item)
                self.r_add._save()
            return item

    def add_from_text(self, text_ori: str) -> Tuple[Optional[Dict[str, Any]], str]:
        with self.lock:
            if not reminder_time_normalize or not self.r_add:
                return None, 'Sistem pengingat belum siap.'
            task = self.r_add.strip_text(text_ori)
            parsed = reminder_time_normalize(text_ori)
            startdate = parsed.get('result')
            repeat_type = parsed.get('repeat_type')
            if not startdate:
                return None, 'Maaf, saya tidak dapat memahami waktu pengingat yang dimaksud.'
            item = {
                'uuid': str(uuid.uuid4()),
                'task': task if task else text_ori,
                'startdate': startdate,
                'repeat_type': repeat_type,
                'called_upon': False,
                'is_prayer': False,
            }
            self.r_add.reminders.append(item)
            self.r_add._save()
            try:
                dt = datetime.fromisoformat(startdate)
                months = {1:'januari',2:'februari',3:'maret',4:'april',5:'mei',6:'juni',7:'juli',8:'agustus',9:'september',10:'oktober',11:'november',12:'desember'}
                month_name = months.get(dt.month, str(dt.month))
                hour = dt.hour
                period = 'pagi' if 1 <= hour < 12 else ('siang' if 12 <= hour < 15 else ('sore' if 15 <= hour < 18 else ('maghrib' if 18 <= hour < 19 else 'malam')))
                display_hour = f'{period} jam {hour if hour <= 12 else hour-12}' + (f' lewat {dt.minute} menit' if dt.minute else '')
                if repeat_type:
                    msg = f'Baik, pengingat untuk {item["task"]} setiap {repeat_type} hari mulai {dt.day} {month_name}, {display_hour} telah disimpan.'
                else:
                    msg = f'Baik, pengingat untuk {item["task"]} pada {dt.day} {month_name}, {display_hour} telah disimpan.'
            except Exception:
                msg = f'Baik, pengingat untuk {item["task"]} telah disimpan.'
            return item, msg

    def query_reminders(self, search_text: str = '') -> Tuple[str, List[Dict[str, Any]]]:
        with self.lock:
            if self.r_check:
                return self.r_check.query(search_text)
            return 'Tidak ada pengingat.', []

    def check_due_reminders(self) -> List[Dict[str, Any]]:
        with self.lock:
            if not self.r_check:
                return []
            is_due, reminder_id = self.r_check.check()
            if is_due and reminder_id:
                for r in self.r_check.reminders:
                    if r.get('uuid') == reminder_id:
                        return [dict(r)]
            return []


# ==============================================================================
# 4. CONTEXTUAL DIALOGUE & LLM GENERATOR
# ==============================================================================
class DialogueGenerator:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.qwen = None
        self._qwen_attempted = False
        self._qwen_init_lock = threading.Lock()
        self.qwen_enabled = os.getenv('HAFIBOT_QWEN_ENABLED', 'true').strip().lower() not in {
            '0', 'false', 'no', 'off'
        }
        if self.qwen_enabled:
            threading.Thread(target=self._init_qwen, daemon=True, name="qwen_preload").start()

    def _init_qwen(self):
        if not self.qwen_enabled:
            return
        with self._qwen_init_lock:
            if self._qwen_attempted:
                return
            self._qwen_attempted = True
            try:
                from m_qwen.response import QwenResponse
                self.qwen = QwenResponse()
                logger.info("QwenResponse initialized successfully with QwenModel2.")
            except Exception as e:
                logger.warning("QwenResponse not loaded (using deterministic companion dialogue): %s", e)

    def _generate_qwen(self, qwentext: Optional[dict]) -> str:
        if not qwentext or not self.qwen_enabled:
            return ''
        self._init_qwen()
        if self.qwen is None:
            return ''
        try:
            return (self.qwen.generate_response(qwentext) or '').strip()
        except Exception as e:
            logger.debug("Qwen generation failed (%s), using fallback", e)
            return ''

    def generate(self, user_text: str, intent: str, sentiment: str, function_label: str, qwentext: Optional[dict] = None) -> str:
        t = user_text.lower().strip()

        # 1. Direct motion commands (prioritize stop/berhenti over start/mulai)
        is_stop = intent == 'berhenti' or any(w in t for w in ['berhenti', 'stop', 'diam', 'jangan gerak', 'jangan ikuti', 'jangan jalan', 'jangan maju', 'tunggu'])
        is_start = not is_stop and (intent == 'mulai' or any(w in t for w in ['mulai', 'ikuti saya', 'ayo ikuti', 'jalan', 'ayo gerak', 'maju']))

        if is_stop:
            return 'Baik, saya berhenti bergerak sekarang.'
        if is_start:
            return 'Baik, saya akan mulai bergerak dan mengikuti Anda.'

        # 2. Emergency & Fall Checks
        if any(w in t for w in ['saya tidak apa-apa', 'tidak apa apa', 'aman', 'baik baik saja', 'hanya duduk']):
            return 'Syukurlah jika Anda baik-baik saja. Alarm telah saya matikan.'
        if any(w in t for w in ['tolong', 'sakit', 'jatuh', 'darurat', 'panggil bantuan']):
            return 'Peringatan darurat diaktifkan! Saya segera menghubungi pengasuh Anda.'

        # 3. Primary: Generate natural response via fine-tuned Qwen model
        qwen_resp = self._generate_qwen(qwentext)
        if qwen_resp and qwen_resp.strip():
            return qwen_resp.strip()

        # 4. Fallback: Contextual deterministic responses if Qwen is disabled or unavailable
        if function_label == 'time' or 'jam berapa' in t or 'waktu' in t:
            jt = get_jakarta_time()
            return f'Sekarang adalah hari {jt.get("day", "")}, tanggal {jt.get("date", "")} pukul {jt.get("time", "")[:5]} WIB.'

        if function_label == 'weather' or 'cuaca' in t or 'hujan' in t:
            w = fetch_weather_bmkg()
            return f'Prakiraan cuaca saat ini {w["cuaca"]} dengan suhu sekitar {w["suhu"]} derajat Celcius.'

        if function_label == 'news' or 'berita' in t:
            n = fetch_news_tempo()
            return f'Kabar terbaru hari ini: {n["title"]}'

        if intent in ['menyapa', 'sapaan'] or any(w in t for w in ['halo', 'hai', 'selamat pagi', 'selamat siang', 'selamat malam']):
            return 'Halo! Senang bertemu dengan Anda. Ada yang bisa saya bantu hari ini?'

        if intent in ['tanya_kabar'] or 'apa kabar' in t:
            return 'Kabar saya sangat baik dan siap menemani kegiatan Anda hari ini! Bagaimana perasaan Anda?'

        if intent in ['terimakasih'] or 'terima kasih' in t or 'makasih' in t:
            return 'Sama-sama! Saya selalu senang bisa membantu dan menemani Anda.'

        if 'kamu siapa' in t or 'nama kamu' in t:
            return 'Saya Hafibot, robot asisten pendamping lansia dengan sistem pemantau keselamatan dan pengikut manusia.'

        if sentiment == 'positive':
            return 'Senang sekali mendengarnya! Tetap semangat ya.'
        elif sentiment == 'negative':
            return 'Saya mengerti perasaan Anda. Istirahatlah sejenak, saya akan selalu menjaga Anda di sini.'

        return 'Baik, saya mengerti. Ada hal lain yang ingin Anda sampaikan atau tanyakan?'

class AudioEngine:
    def __init__(self, root_dir=None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent
        import speech_recognition as sr
        self.sr_recognizer = sr.Recognizer()

    @staticmethod
    def parse_number(words):
        ones = {
            "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
            "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9
        }
        total = 0
        current = 0
        i = 0
        while i < len(words):
            word = words[i]
            if word in ones:
                current += ones[word]
            elif word == "sepuluh":
                current += 10
            elif word == "sebelas":
                current += 11
            elif word == "seratus":
                total += 100
                current = 0
            elif word == "seribu":
                total += 1000
                current = 0
            elif word == "belas":
                if current == 0:
                    current = 10
                else:
                    current += 10
            elif word == "puluh":
                if current == 0:
                    current = 1
                current *= 10
            elif word == "ratus":
                if current == 0:
                    current = 1
                current *= 100
            elif word == "ribu":
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
            elif word == "juta":
                if current == 0:
                    current = 1
                total += current * 1_000_000
                current = 0
            i += 1
        return total + current

    def normalize_numbers(self, text):
        number_words = {
            "nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
            "sepuluh", "sebelas", "seratus", "seribu", "belas", "puluh", "ratus", "ribu", "juta"
        }
        words = text.lower().split()
        result = []
        number_buffer = []

        def flush_number():
            if number_buffer:
                number = self.parse_number(number_buffer)
                result.append(str(number))
                number_buffer.clear()

        for word in words:
            clean_word = word.strip(".,!?;:")
            if clean_word in number_words:
                number_buffer.append(clean_word)
            else:
                flush_number()
                result.append(word)

        flush_number()
        return " ".join(result)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        if not wav_bytes or len(wav_bytes) < 200:
            return ''
        import speech_recognition as sr

        # Google Indonesian Speech Recognition (id-ID) is the only ASR engine.
        try:
            bio = io.BytesIO(wav_bytes)
            with sr.AudioFile(bio) as source:
                audio_data = self.sr_recognizer.record(source)
            text = self.sr_recognizer.recognize_google(audio_data, language="id-ID").strip()
            if text:
                text = self.normalize_numbers(text)
                logger.info("Google Speech Recognition (id-ID) recognized: '%s'", text)
                return text
        except sr.UnknownValueError:
            logger.debug("Google ASR: UnknownValueError (silence/unintelligible)")
            return ''
        except Exception as e:
            logger.warning("Google ASR unavailable: %s", e)
            return ''

    @staticmethod
    def _clean_text_for_speech(text: str) -> str:
        # Remove markdown formatting (*, _, #, ~, bullet points)
        cleaned = re.sub(r'[*_#~]', '', text)
        cleaned = re.sub(r'^\s*[-+*]\s+', '', cleaned, flags=re.MULTILINE)
        # Remove emoji characters that confuse TTS pronunciation
        cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def synthesize_wav(self, text: str) -> bytes:
        if not text or not text.strip():
            return b''
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            speech_text = self._clean_text_for_speech(text) or text
            tts = gTTS(text=speech_text, lang='id')
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            audio = AudioSegment.from_file(mp3_buf, format='mp3')
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            wav_buf = io.BytesIO()
            audio.export(wav_buf, format='wav')
            return wav_buf.getvalue()
        except Exception as e:
            return self._synthesize_offline_tone_wav(text)

    def _synthesize_offline_tone_wav(self, text: str) -> bytes:
        sample_rate = 16000
        words = len(text.split())
        num_beeps = max(2, min(6, words // 2 + 1))
        freqs = [659.25, 783.99, 1046.50, 1318.51]  # High pleasant chime chords for companion assistant
        frames = bytearray()
        tone_samples = int(sample_rate * 0.10)
        gap_samples = int(sample_rate * 0.04)
        for i in range(num_beeps):
            freq = freqs[i % len(freqs)]
            for s in range(tone_samples):
                env = 1.0 - (s / tone_samples)
                val = int(14000 * env * math.sin(2.0 * math.pi * freq * s / sample_rate))
                frames.extend(struct.pack('<h', val))
            frames.extend(b'\x00\x00' * gap_samples)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(frames))
        return wav_buf.getvalue()

class ChatbotService:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, root_dir: Optional[Path] = None):
        with cls._lock:
            if cls._instance is None:
                root = root_dir or Path(__file__).resolve().parent.parent
                cls._instance = cls(root)
            return cls._instance

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        logger.info('Initializing Hafibot ChatbotService at %s', self.root_dir)

        self.classifier = DistilBertClassifier(self.root_dir)
        rem_path = self.root_dir / 'data' / 'monitoring' / 'reminders.json'
        self.reminder_engine = ReminderEngine(rem_path)
        self.dialogue_gen = DialogueGenerator(self.root_dir)
        self.audio_engine = AudioEngine(self.root_dir)

        self.proactive_audio_queue: List[Dict[str, Any]] = []
        self.queue_lock = threading.Lock()
        self.history: List[Dict[str, Any]] = []
        self.history_lock = threading.Lock()
        self.prev_resp = None
        self.enabled = True
        self.enabled_lock = threading.RLock()

        self.on_start_following = None
        self.on_stop_following = None
        self.on_reset_estop = None
        self.on_trigger_alarm = None

        self.running = True
        self.bg_thread = threading.Thread(target=self._reminder_worker_loop, daemon=True)
        self.bg_thread.start()
        logger.info('ChatbotService initialized successfully.')

    def toggle(self, enable: Optional[bool] = None) -> bool:
        with self.enabled_lock:
            if enable is not None:
                self.enabled = bool(enable)
            else:
                self.enabled = not self.enabled
            is_enabled = self.enabled

        if not is_enabled:
            # Do not play reminders/fall prompts queued before the user turned
            # the voice assistant off.
            with self.queue_lock:
                self.proactive_audio_queue.clear()
        logger.info('ChatbotService enabled set to: %s', is_enabled)
        return is_enabled

    def is_enabled(self) -> bool:
        with self.enabled_lock:
            return bool(self.enabled)

    @staticmethod
    def _disabled_result() -> Dict[str, Any]:
        return {
            'success': False,
            'voice_enabled': False,
            'error': 'voice_disabled',
            'transcript': '',
            'response_text': 'Asisten suara sedang dinonaktifkan.',
            'wav_bytes': b'',
            'action': 'none',
            'latency': {},
        }

    def _reminder_worker_loop(self):
        while self.running:
            try:
                if not self.is_enabled():
                    time.sleep(3.0)
                    continue
                if self.reminder_engine.r_check and self.reminder_engine.r_call:
                    is_due, reminder_id = self.reminder_engine.r_check.check()
                    if is_due and reminder_id:
                        self.reminder_engine.r_call.call(
                            reminder_id=reminder_id,
                            output_fn=lambda msg: self.queue_proactive_speech(msg, priority='high', reason='reminder')
                        )
            except Exception as e:
                logger.error('Error in reminder worker: %s', e)
            time.sleep(5.0)

    def queue_proactive_speech(self, text: str, priority: str = 'normal', reason: str = 'general') -> str:
        if not self.is_enabled():
            return ''
        wav_bytes = self.audio_engine.synthesize_wav(text)
        if not self.is_enabled():
            return ''
        item_id = str(uuid.uuid4())
        item = {
            'id': item_id,
            'text': text,
            'priority': priority,
            'reason': reason,
            'wav_bytes': wav_bytes,
            'created_at': time.time(),
        }
        with self.queue_lock:
            if priority == 'high':
                self.proactive_audio_queue.insert(0, item)
            else:
                self.proactive_audio_queue.append(item)
            if len(self.proactive_audio_queue) > 20:
                self.proactive_audio_queue.pop(-1)

        self._record_history(speaker='robot', text=text, intent='proactive', sentiment='neutral', action=reason)
        return item_id

    def pop_pending_audio(self) -> Optional[Dict[str, Any]]:
        if not self.is_enabled():
            return None
        with self.queue_lock:
            if self.proactive_audio_queue:
                return self.proactive_audio_queue.pop(0)
        return None

    def trigger_fall_alert_voice(self):
        prompt = "Perhatian! Terdeteksi Anda terjatuh. Apakah Anda baik-baik saja? Katakan 'Saya tidak apa-apa' untuk mematikan alarm, atau 'Tolong' untuk bantuan."
        self.queue_proactive_speech(prompt, priority='high', reason='fall_confirmation')

    def trigger_fall_inquiry_voice(self):
        prompt = "Apakah Anda baik-baik saja? Tolong bersuara jika Anda membutuhkan bantuan."
        self.queue_proactive_speech(prompt, priority='high', reason='fall_inquiry')

    def process_voice_interaction(self, wav_bytes: bytes) -> Dict[str, Any]:
        if not self.is_enabled():
            return self._disabled_result()
        t_start = time.perf_counter()
        t_asr_start = time.perf_counter()
        transcript = self.audio_engine.transcribe_wav_bytes(wav_bytes)
        asr_latency_ms = (time.perf_counter() - t_asr_start) * 1000.0

        if not transcript or not transcript.strip():
            logger.info('ASR finished in %.1f ms: No speech recognized.', asr_latency_ms)
            return {
                'success': False,
                'transcript': '',
                'response_text': 'Maaf, suara tidak terdengar jelas.',
                'wav_bytes': b'',
                'action': 'none',
                'latency_ms': (time.perf_counter() - t_start) * 1000.0,
            }

        logger.info('ASR finished in %.1f ms: Transcribed -> %s', asr_latency_ms, transcript)
        return self.process_text_interaction(transcript, asr_latency_ms=asr_latency_ms)

    def process_text_interaction(self, user_text: str, asr_latency_ms: float = 0.0) -> Dict[str, Any]:
        if not self.is_enabled():
            return self._disabled_result()
        t_start = time.perf_counter()
        t_nlu_start = time.perf_counter()
        nlu_res = self.classifier.predict(user_text)
        intent = nlu_res['intent']
        sentiment = nlu_res['sentiment']
        function_label = nlu_res['function']
        nlu_latency_ms = (time.perf_counter() - t_nlu_start) * 1000.0

        action_triggered = 'none'
        t_norm = user_text.lower().strip()

        is_stop = intent == 'berhenti' or any(w in t_norm for w in ['berhenti', 'stop', 'diam', 'jangan gerak', 'jangan ikuti', 'jangan jalan', 'jangan maju', 'tunggu'])
        is_start = not is_stop and (intent == 'mulai' or any(w in t_norm for w in ['mulai', 'ikuti saya', 'ayo ikuti', 'jalan', 'ayo gerak', 'maju']))

        if is_stop:
            action_triggered = 'STOP_FOLLOWING'
            if callable(self.on_stop_following):
                self.on_stop_following()
        elif is_start:
            action_triggered = 'START_FOLLOWING'
            if callable(self.on_start_following):
                self.on_start_following()
        elif any(w in t_norm for w in ['saya tidak apa-apa', 'tidak apa apa', 'aman', 'baik baik saja']):
            action_triggered = 'RESET_ESTOP'
            if callable(self.on_reset_estop):
                self.on_reset_estop()
        elif any(w in t_norm for w in ['tolong', 'sakit', 'darurat', 'jatuh', 'panggil bantuan']):
            action_triggered = 'EMERGENCY_ALARM'
            if callable(self.on_trigger_alarm):
                self.on_trigger_alarm()

        # 4. Context Enriching via middlelayer & Decision Making
        labels = {'intent': intent, 'sentiment': sentiment, 'function': function_label}
        try:
            from m_middlelayer import middlelayer as midlayer
            fulltext, qwentext = midlayer.context_builder(text=user_text, bert_inference=labels, prev_resp=self.prev_resp)
        except Exception:
            fulltext, qwentext = {'command_override': False, 'TBA_override': False}, {'input': user_text, 'context': {}}

        if action_triggered == 'STOP_FOLLOWING':
            response_text = 'Baik, saya akan berhenti bergerak'
        elif action_triggered == 'START_FOLLOWING':
            response_text = 'Baik, saya akan kembali bergerak'
        elif action_triggered == 'RESET_ESTOP':
            response_text = 'Syukurlah jika Anda baik-baik saja. Alarm telah saya matikan.'
        elif action_triggered == 'EMERGENCY_ALARM':
            response_text = 'Peringatan darurat diaktifkan! Saya segera menghubungi pengasuh Anda.'
        elif fulltext.get('command_override') and fulltext.get('command_text'):
            response_text = fulltext['command_text']
        elif fulltext.get('TBA_override') and fulltext.get('TBA_text'):
            response_text = fulltext['TBA_text']
        elif function_label == 'reminder' or intent in ['buatagenda', 'reminder']:
            item, msg = self.reminder_engine.add_from_text(user_text)
            response_text = msg
        elif function_label == 'agenda_query' or intent in ['jadwal', 'cek_jadwal']:
            response_text, _ = self.reminder_engine.query_reminders(user_text)
        elif function_label == 'repeat':
            response_text = self.prev_resp if self.prev_resp else 'Belum ada percakapan sebelumnya.'
        else:
            response_text = self.dialogue_gen.generate(user_text, intent, sentiment, function_label, qwentext=qwentext)

        if not fulltext.get('command_override') and not fulltext.get('TBA_override'):
            self.prev_resp = response_text

        # 5. Synthesize Audio
        t_tts_start = time.perf_counter()
        resp_wav_bytes = self.audio_engine.synthesize_wav(response_text)
        tts_latency_ms = (time.perf_counter() - t_tts_start) * 1000.0

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0 + asr_latency_ms

        # 6. Record history
        self._record_history(
            speaker='user',
            text=user_text,
            intent=intent,
            sentiment=sentiment,
            action=action_triggered,
            response=response_text,
        )

        return {
            'success': True,
            'transcript': user_text,
            'intent': intent,
            'sentiment': sentiment,
            'function': function_label,
            'action': action_triggered,
            'response_text': response_text,
            'wav_bytes': resp_wav_bytes,
            'latency': {
                'asr_ms': round(asr_latency_ms, 1),
                'nlu_ms': round(nlu_latency_ms, 1),
                'tts_ms': round(tts_latency_ms, 1),
                'total_ms': round(total_latency_ms, 1),
            },
        }

    def _record_history(self, speaker: str, text: str, intent: str, sentiment: str, action: str, response: str = ''):
        with self.history_lock:
            entry = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'speaker': speaker,
                'text': text,
                'intent': intent,
                'sentiment': sentiment,
                'action': action,
                'response': response,
            }
            self.history.append(entry)
            if len(self.history) > 50:
                self.history.pop(0)

    def get_history(self) -> List[Dict[str, Any]]:
        with self.history_lock:
            return list(self.history)
