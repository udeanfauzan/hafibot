#!/usr/bin/env python3
"""
ROS Node untuk Telegram Push Notification - Hafibot Fall Alert.
Privacy-Preserving Edition.

Fitur Keamanan & Privasi:
  1. Authorized Family IDs — hanya chat_id terdaftar yang bisa interaksi
  2. Privacy Frame — frame diblur + skeleton overlay (wajah tidak terlihat)
  3. Auto-Delete Messages — pesan otomatis dihapus setiap 2 jam dari server Telegram
  4. Command Handler — /status, /help (hanya untuk authorized users)

Setup Telegram Bot:
  1. Buka Telegram, cari @BotFather
  2. /newbot → beri nama → simpan TOKEN
  3. Tambahkan bot ke group chat keluarga
  4. Dapatkan chat_id dari https://api.telegram.org/bot<TOKEN>/getUpdates

Konfigurasi via ROS params atau environment variables:
  - bot_token        : Token dari BotFather
  - chat_id          : Chat ID (group/individual, koma untuk multiple)
  - authorized_ids   : ID user/group yang diizinkan interaksi (koma-separated)
  - cooldown_sec     : Cooldown antar notifikasi (default: 30)
  - auto_delete_sec  : Interval auto-delete pesan (default: 7200 = 2 jam)
"""

import rospy
import cv2
import os
import json
import time
import threading
import numpy as np
from io import BytesIO
from datetime import datetime
from collections import namedtuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String


# ============================================================
# POSE SKELETON CONSTANTS (sama dengan inference.py)
# ============================================================
POSE_CONNECTIONS = [
    # Wajah (tidak digambar untuk privasi)
    # Badan atas
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    # Tangan
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # Badan bawah
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Warna skeleton untuk privacy frame (BGR)
SKELETON_COLOR = (0, 255, 200)      # Cyan untuk garis
JOINT_COLOR = (0, 200, 255)         # Kuning untuk sendi
FALL_SKELETON_COLOR = (0, 0, 255)   # Merah saat jatuh
FALL_JOINT_COLOR = (0, 100, 255)    # Orange saat jatuh

# Landmark untuk dummy namedtuple
LandmarkPoint = namedtuple('LandmarkPoint', ['x', 'y', 'z', 'visibility'])


class TelegramNotifierNode:
    """ROS Node yang mengirim notifikasi fall detection ke Telegram.
    Privacy-preserving: frame diblur, skeleton-only overlay."""

    TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self):
        rospy.init_node('telegram_notifier_node', anonymous=True)

        # -- Konfigurasi --
        self.bot_token = rospy.get_param(
            '~bot_token',
            os.environ.get('TELEGRAM_BOT_TOKEN', '')
        )
        chat_id_raw = rospy.get_param(
            '~chat_id',
            os.environ.get('TELEGRAM_CHAT_ID', '')
        )
        authorized_raw = rospy.get_param(
            '~authorized_ids',
            os.environ.get('TELEGRAM_AUTHORIZED_IDS', '')
        )
        self.cooldown_sec = rospy.get_param('~cooldown_sec', 30)
        self.auto_delete_sec = rospy.get_param('~auto_delete_sec', 7200)  # 2 jam
        self.monitor_url = rospy.get_param(
            '~monitor_url',
            os.environ.get('HAFIBOT_MONITOR_URL', '')
        )
        self.blur_strength = rospy.get_param('~blur_strength', 45)  # Kernel size GaussianBlur

        # Parse chat_ids (target notifikasi)
        self.chat_ids = self._parse_ids(chat_id_raw)

        # Parse authorized_family_ids (whitelist interaksi)
        # Jika tidak diset, gunakan chat_ids sebagai default
        self.authorized_ids = self._parse_ids(authorized_raw)
        if not self.authorized_ids:
            self.authorized_ids = self.chat_ids.copy()

        # -- Validasi --
        if not HAS_REQUESTS:
            rospy.logerr("Library 'requests' tidak ditemukan! "
                         "Install dengan: pip3 install requests")
            rospy.signal_shutdown("Missing dependency: requests")
            return

        if not self.bot_token:
            rospy.logwarn("⚠️  TELEGRAM_BOT_TOKEN belum diset!")

        # -- State --
        self._current_frame = None        # Frame mentah (CompressedImage → decoded)
        self._current_landmarks = None    # Landmarks JSON terbaru
        self._frame_lock = threading.Lock()
        self._landmarks_lock = threading.Lock()
        self._last_notify_time = 0.0
        self._is_sending = False
        self._total_sent = 0
        self._total_failed = 0

        # -- Auto-Delete: simpan message_id yang dikirim --
        self._sent_messages = []  # List of (chat_id, message_id, timestamp)
        self._sent_messages_lock = threading.Lock()

        # -- Telegram update polling state --
        self._last_update_id = 0

        # -- ROS Subscribers --
        rospy.Subscriber(
            '/fall_detection/image_overlay', CompressedImage,
            self._image_callback, queue_size=1, buff_size=2**24
        )
        rospy.Subscriber(
            '/fall_detection/alert', Bool,
            self._alert_callback, queue_size=1
        )
        rospy.Subscriber(
            '/fall_detection/landmarks', String,
            self._landmarks_callback, queue_size=1
        )

        # -- Startup --
        self._log_startup_info()

        if self.bot_token and self.chat_ids:
            self._test_connection()

        # -- Start background threads --
        # Auto-delete thread
        if self.auto_delete_sec > 0:
            self._auto_delete_thread = threading.Thread(
                target=self._auto_delete_loop, daemon=True
            )
            self._auto_delete_thread.start()

        # Telegram command polling thread
        if self.bot_token and self.authorized_ids:
            self._polling_thread = threading.Thread(
                target=self._polling_loop, daemon=True
            )
            self._polling_thread.start()

    # ============================================================
    # HELPERS
    # ============================================================
    @staticmethod
    def _parse_ids(raw_str):
        """Parse koma-separated string menjadi list of int IDs."""
        ids = []
        if raw_str:
            for cid in str(raw_str).split(','):
                cid = cid.strip()
                if cid:
                    try:
                        ids.append(int(cid))
                    except ValueError:
                        pass
        return ids

    def _log_startup_info(self):
        """Log informasi konfigurasi saat startup."""
        rospy.loginfo("=" * 55)
        rospy.loginfo("🤖 Telegram Notifier - Hafibot (Privacy Edition)")
        rospy.loginfo("=" * 55)

        if self.bot_token:
            masked = self.bot_token[:8] + "..." + self.bot_token[-4:]
            rospy.loginfo(f"  Bot Token      : {masked}")
        else:
            rospy.loginfo("  Bot Token      : ❌ TIDAK DISET")

        rospy.loginfo(f"  Chat IDs       : {self.chat_ids or '❌ TIDAK DISET'}")
        rospy.loginfo(f"  Authorized IDs : {self.authorized_ids}")
        rospy.loginfo(f"  Cooldown       : {self.cooldown_sec} detik")
        rospy.loginfo(f"  Auto-Delete    : setiap {self.auto_delete_sec}s ({self.auto_delete_sec/3600:.1f} jam)")
        rospy.loginfo(f"  Blur Strength  : {self.blur_strength}")
        rospy.loginfo(f"  Privacy Frame  : ✅ Blur + Skeleton Only")
        rospy.loginfo("=" * 55)

    def _test_connection(self):
        """Test koneksi ke Telegram Bot API saat startup."""
        try:
            url = self.TELEGRAM_API_URL.format(
                token=self.bot_token, method='getMe'
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and resp.json().get('ok'):
                bot_info = resp.json()['result']
                rospy.loginfo(
                    f"  ✅ Bot: {bot_info.get('first_name')} "
                    f"(@{bot_info.get('username')})"
                )
            else:
                rospy.logwarn(f"  ⚠️ Bot API error: {resp.text[:100]}")
        except Exception as e:
            rospy.logwarn(f"  ⚠️ Koneksi Telegram gagal: {e}")

    def _is_authorized(self, chat_id):
        """Cek apakah chat_id termasuk dalam authorized family IDs."""
        return int(chat_id) in self.authorized_ids

    # ============================================================
    # ROS CALLBACKS
    # ============================================================
    def _image_callback(self, msg):
        """Callback: simpan frame terbaru (CompressedImage → decoded)."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with self._frame_lock:
                    self._current_frame = frame
        except Exception as e:
            rospy.logwarn_throttle(5, f"Gagal decode frame: {e}")

    def _landmarks_callback(self, msg):
        """Callback: simpan landmarks JSON terbaru."""
        try:
            landmarks = json.loads(msg.data)
            with self._landmarks_lock:
                self._current_landmarks = landmarks
        except Exception as e:
            rospy.logwarn_throttle(5, f"Gagal parse landmarks: {e}")

    def _alert_callback(self, msg):
        """Callback: trigger notifikasi saat fall terdeteksi."""
        if not msg.data:
            return

        if not self.bot_token or not self.chat_ids:
            return

        # Cooldown check
        now = time.time()
        if now - self._last_notify_time < self.cooldown_sec:
            return

        if self._is_sending:
            return

        self._last_notify_time = now

        send_thread = threading.Thread(
            target=self._send_notification, daemon=True
        )
        send_thread.start()

    # ============================================================
    # PRIVACY-PRESERVING FRAME
    # ============================================================
    def _create_privacy_frame(self, frame, landmarks_data):
        """
        Buat frame privacy-preserving:
        1. Blur frame asli (wajah & detail tidak terlihat)
        2. Gambar skeleton MediaPipe di atas frame yang sudah diblur
        
        Args:
            frame: Frame BGR asli
            landmarks_data: List of dicts [{'x','y','z','v'}, ...]
            
        Returns:
            Privacy-preserved frame (BGR)
        """
        h, w = frame.shape[:2]

        # Step 1: Blur seluruh frame
        k = self.blur_strength
        if k % 2 == 0:
            k += 1  # Kernel harus ganjil
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # Step 2: Gambar skeleton di atas blurred frame
        if landmarks_data:
            landmarks = [
                LandmarkPoint(
                    x=lm['x'], y=lm['y'],
                    z=lm['z'], visibility=lm.get('v', 0)
                )
                for lm in landmarks_data
            ]
            self._draw_skeleton(blurred, landmarks, w, h, is_fall=True)

        # Step 3: Tambahkan label "FALL DETECTED" di atas
        self._draw_fall_label(blurred, w, h)

        # Step 4: Tambahkan watermark privasi
        cv2.putText(
            blurred, "PRIVACY PROTECTED",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (200, 200, 200), 1, cv2.LINE_AA
        )

        return blurred

    def _draw_skeleton(self, frame, landmarks, w, h, is_fall=False):
        """Gambar skeleton pose pada frame (tanpa wajah untuk privasi)."""
        line_color = FALL_SKELETON_COLOR if is_fall else SKELETON_COLOR
        joint_color = FALL_JOINT_COLOR if is_fall else JOINT_COLOR

        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py, lm.visibility))

        # Gambar connections (skip wajah: index 0-10)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                s_vis = points[start_idx][2]
                e_vis = points[end_idx][2]
                if s_vis > 0.3 and e_vis > 0.3:
                    pt1 = (points[start_idx][0], points[start_idx][1])
                    pt2 = (points[end_idx][0], points[end_idx][1])
                    cv2.line(frame, pt1, pt2, line_color, 3, cv2.LINE_AA)

        # Gambar joints (skip wajah: hanya gambar index >= 11)
        for idx in range(11, len(points)):
            px, py, vis = points[idx]
            if vis > 0.3:
                cv2.circle(frame, (px, py), 5, joint_color, -1)
                cv2.circle(frame, (px, py), 6, (0, 0, 0), 1)

    def _draw_fall_label(self, frame, w, h):
        """Gambar label FALL DETECTED di tengah atas."""
        text = "!! FALL DETECTED !!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        x = (w - tw) // 2
        y = 45
        pad = 10

        # Background merah
        cv2.rectangle(frame, (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      (0, 0, 200), -1)
        cv2.rectangle(frame, (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      (255, 255, 255), 2)
        cv2.putText(frame, text, (x, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # ============================================================
    # SEND NOTIFICATION
    # ============================================================
    def _send_notification(self):
        """Kirim notifikasi privacy-preserving ke semua authorized chat_ids."""
        self._is_sending = True
        try:
            # Ambil frame dan landmarks terbaru
            with self._frame_lock:
                frame = self._current_frame.copy() if self._current_frame is not None else None
            with self._landmarks_lock:
                landmarks = self._current_landmarks

            if frame is None:
                rospy.logwarn("Tidak ada frame untuk notifikasi")
                return

            # Buat privacy frame (blur + skeleton)
            privacy_frame = self._create_privacy_frame(frame, landmarks)

            # Encode ke JPEG
            ret, jpeg_buffer = cv2.imencode(
                '.jpg', privacy_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                rospy.logerr("Gagal encode privacy frame ke JPEG")
                return

            jpeg_bytes = jpeg_buffer.tobytes()

            # Compose caption
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S WIB")
            caption = (
                "🚨 PERINGATAN JATUH TERDETEKSI!\n"
                "\n"
                f"📅 Waktu: {now_str}\n"
                "🤖 Sistem: Hafibot Fall Detection\n"
                "🔒 Gambar: Privacy-Protected (Blurred + Skeleton)\n"
                "\n"
                "⚠️ Segera periksa kondisi lansia."
            )

            if self.monitor_url:
                caption += f"\n\n🔗 Live Monitor: {self.monitor_url}"

            # Kirim ke semua authorized chat IDs
            for chat_id in self.chat_ids:
                if self._is_authorized(chat_id):
                    self._send_photo(chat_id, jpeg_bytes, caption)

        except Exception as e:
            rospy.logerr(f"Error mengirim notifikasi: {e}")
            self._total_failed += 1
        finally:
            self._is_sending = False

    def _send_photo(self, chat_id, jpeg_bytes, caption):
        """Kirim foto ke satu chat_id. Simpan message_id untuk auto-delete."""
        url = self.TELEGRAM_API_URL.format(
            token=self.bot_token, method='sendPhoto'
        )
        files = {
            'photo': ('fall_alert.jpg', BytesIO(jpeg_bytes), 'image/jpeg')
        }
        data = {
            'chat_id': chat_id,
            'caption': caption,
        }

        try:
            resp = requests.post(url, files=files, data=data, timeout=15)
            result = resp.json()

            if resp.status_code == 200 and result.get('ok'):
                self._total_sent += 1
                msg_id = result['result']['message_id']

                # Simpan untuk auto-delete nanti
                with self._sent_messages_lock:
                    self._sent_messages.append(
                        (chat_id, msg_id, time.time())
                    )

                rospy.loginfo(
                    f"✅ Notifikasi terkirim ke {chat_id} "
                    f"(msg_id={msg_id}, total={self._total_sent})"
                )
            else:
                self._total_failed += 1
                rospy.logerr(
                    f"❌ Gagal kirim ke {chat_id}: "
                    f"{result.get('description', 'Unknown')}"
                )

        except requests.exceptions.Timeout:
            self._total_failed += 1
            rospy.logerr(f"⏱️ Timeout mengirim ke {chat_id}")
        except requests.exceptions.ConnectionError:
            self._total_failed += 1
            rospy.logerr("🔌 Tidak ada koneksi internet")
        except Exception as e:
            self._total_failed += 1
            rospy.logerr(f"❌ Error: {e}")

    # ============================================================
    # AUTO-DELETE MESSAGES (Ephemeral Data)
    # ============================================================
    def _auto_delete_loop(self):
        """Background thread: hapus pesan yang sudah lebih dari auto_delete_sec."""
        rospy.loginfo(
            f"🗑️ Auto-delete thread aktif: "
            f"hapus pesan setiap {self.auto_delete_sec}s"
        )

        while not rospy.is_shutdown():
            time.sleep(60)  # Cek setiap 1 menit

            now = time.time()
            to_delete = []

            with self._sent_messages_lock:
                # Kumpulkan pesan yang sudah expired
                remaining = []
                for chat_id, msg_id, sent_time in self._sent_messages:
                    if now - sent_time >= self.auto_delete_sec:
                        to_delete.append((chat_id, msg_id))
                    else:
                        remaining.append((chat_id, msg_id, sent_time))
                self._sent_messages = remaining

            # Hapus pesan expired dari Telegram
            for chat_id, msg_id in to_delete:
                self._delete_message(chat_id, msg_id)

    def _delete_message(self, chat_id, message_id):
        """Hapus satu pesan dari Telegram via Bot API."""
        url = self.TELEGRAM_API_URL.format(
            token=self.bot_token, method='deleteMessage'
        )
        data = {
            'chat_id': chat_id,
            'message_id': message_id,
        }

        try:
            resp = requests.post(url, data=data, timeout=10)
            result = resp.json()

            if result.get('ok'):
                rospy.loginfo(
                    f"🗑️ Pesan {message_id} dihapus dari chat {chat_id}"
                )
            else:
                desc = result.get('description', '')
                # Pesan mungkin sudah dihapus manual oleh user
                if 'not found' not in desc.lower() and 'deleted' not in desc.lower():
                    rospy.logwarn(
                        f"⚠️ Gagal hapus pesan {message_id}: {desc}"
                    )
        except Exception as e:
            rospy.logwarn(f"⚠️ Error hapus pesan {message_id}: {e}")

    # ============================================================
    # TELEGRAM COMMAND POLLING (Authorized Users Only)
    # ============================================================
    def _polling_loop(self):
        """Background thread: poll Telegram updates untuk handle commands."""
        rospy.loginfo("📨 Command polling thread aktif (authorized users only)")

        while not rospy.is_shutdown():
            try:
                self._process_updates()
            except Exception as e:
                rospy.logwarn_throttle(30, f"Polling error: {e}")

            time.sleep(3)  # Poll setiap 3 detik

    def _process_updates(self):
        """Ambil dan proses Telegram updates (pesan masuk)."""
        url = self.TELEGRAM_API_URL.format(
            token=self.bot_token, method='getUpdates'
        )
        params = {
            'offset': self._last_update_id + 1,
            'timeout': 5,
            'allowed_updates': '["message"]',
        }

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return

        data = resp.json()
        if not data.get('ok'):
            return

        for update in data.get('result', []):
            update_id = update.get('update_id', 0)
            if update_id > self._last_update_id:
                self._last_update_id = update_id

            message = update.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            user_id = message.get('from', {}).get('id')
            text = message.get('text', '').strip()

            if not chat_id or not text:
                continue

            # === AUTHORIZATION CHECK ===
            # Cek apakah user_id ATAU chat_id termasuk authorized
            if not (self._is_authorized(user_id) or self._is_authorized(chat_id)):
                rospy.logwarn(
                    f"🚫 Unauthorized access dari user={user_id} chat={chat_id}: '{text}'"
                )
                # Abaikan — tidak merespon sama sekali
                continue

            # Handle command
            self._handle_command(chat_id, text)

    def _handle_command(self, chat_id, text):
        """Handle command dari authorized user."""
        cmd = text.lower().split()[0] if text else ''

        if cmd == '/start' or cmd == '/help':
            reply = (
                "🤖 *Hafibot Fall Detection Bot*\n"
                "\n"
                "Perintah tersedia:\n"
                "/status — Cek status sistem\n"
                "/help — Tampilkan bantuan\n"
                "\n"
                "Bot ini akan mengirim notifikasi otomatis "
                "saat jatuh terdeteksi.\n"
                "🔒 Gambar dikirim dalam mode privacy "
                "(blur + skeleton only).\n"
                f"🗑️ Pesan otomatis dihapus setelah "
                f"{self.auto_delete_sec // 3600} jam."
            )
            self._send_text(chat_id, reply)

        elif cmd == '/status':
            # Cek apakah ada frame terbaru (sistem aktif)
            with self._frame_lock:
                has_frame = self._current_frame is not None

            status = "🟢 Aktif" if has_frame else "🔴 Tidak ada feed kamera"
            reply = (
                f"📊 *Status Hafibot*\n"
                f"\n"
                f"Sistem: {status}\n"
                f"Notifikasi terkirim: {self._total_sent}\n"
                f"Gagal: {self._total_failed}\n"
                f"Cooldown: {self.cooldown_sec}s\n"
                f"Auto-delete: {self.auto_delete_sec // 3600} jam"
            )
            self._send_text(chat_id, reply)

        else:
            # Command tidak dikenal — abaikan (jangan beri petunjuk ke attacker)
            pass

    def _send_text(self, chat_id, text):
        """Kirim pesan teks dan simpan msg_id untuk auto-delete."""
        url = self.TELEGRAM_API_URL.format(
            token=self.bot_token, method='sendMessage'
        )
        data = {
            'chat_id': chat_id,
            'text': text,
        }

        try:
            resp = requests.post(url, data=data, timeout=10)
            result = resp.json()

            if result.get('ok'):
                msg_id = result['result']['message_id']
                with self._sent_messages_lock:
                    self._sent_messages.append(
                        (chat_id, msg_id, time.time())
                    )
        except Exception as e:
            rospy.logwarn(f"Gagal kirim teks ke {chat_id}: {e}")

    # ============================================================
    # MAIN
    # ============================================================
    def run(self):
        """Main loop ROS."""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = TelegramNotifierNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
