import logging
import os
import json
import time
import hashlib
import hmac
import re
import secrets
import uuid
import numpy as np
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from flask import Flask, Response, jsonify, send_from_directory, session, redirect, url_for, request
from functools import wraps
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from waitress import serve as waitress_serve
import cv2
import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('hafibot_server')

class _RospyShim:
    def loginfo(self, msg): logger.info(msg)
    def logwarn(self, msg): logger.warning(msg)
    def logerr(self, msg): logger.error(msg)
    def logwarn_throttle(self, period, msg): logger.warning(msg)
    def is_shutdown(self): return False

rospy = _RospyShim()
try:
    from pywebpush import webpush, WebPushException
    HAS_WEBPUSH = True
except ImportError:
    HAS_WEBPUSH = False

app = Flask(__name__)

def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# Deployment credentials are intentionally required. Insecure source-code
# defaults would allow a forged Flask session even after changing the password.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'hafibot-development-secret-key-32-chars-long')
AUTH_USERNAME = os.environ.get('HAFIBOT_USERNAME', 'caregiver')
AUTH_PASSWORD = os.environ.get('HAFIBOT_PASSWORD', 'hafibot123')

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=_env_bool('HAFIBOT_SECURE_COOKIES', False),
    SESSION_COOKIE_SAMESITE='Strict',
    PERMANENT_SESSION_LIFETIME=timedelta(
        hours=_env_float('HAFIBOT_SESSION_HOURS', 12.0)
    ),
    MAX_CONTENT_LENGTH=128 * 1024,
)

FRAME_STALE_SEC = max(0.5, _env_float('HAFIBOT_FRAME_STALE_SEC', 3.0))
DETECTOR_STALE_SEC = max(0.5, _env_float('HAFIBOT_DETECTOR_STALE_SEC', 3.0))
STATS_STALE_SEC = max(1.0, _env_float('HAFIBOT_STATS_STALE_SEC', 5.0))
SNAPSHOT_TOKEN_MAX_AGE_SEC = max(60, int(
    _env_float('HAFIBOT_SNAPSHOT_TOKEN_MAX_AGE_SEC', 300.0)
))
MAX_INCIDENTS = max(1, int(_env_float('HAFIBOT_MAX_INCIDENTS', 100.0)))
LOGIN_WINDOW_SEC = max(30, int(_env_float('HAFIBOT_LOGIN_WINDOW_SEC', 300.0)))
LOGIN_MAX_FAILURES = max(1, int(_env_float('HAFIBOT_LOGIN_MAX_FAILURES', 5.0)))

# Path ke folder img diset nanti saat node diinisialisasi
IMG_DIR = ""
INCIDENTS_FILE = ""
SNAPSHOTS_DIR = ""
CALIBRATION_FILE = str(Path(__file__).resolve().parent.parent / 'data' / 'monitoring' / 'shoulder_calibration.json')

DEFAULT_FOCAL_CONSTANT = 22000.0
current_focal_constant = DEFAULT_FOCAL_CONSTANT
latest_shoulder_px = 0.0
latest_effective_px = 0.0
last_valid_shoulder_px = 0.0
last_calibration_info = {
    'focal_constant': DEFAULT_FOCAL_CONSTANT,
    'calibrated_distance_cm': 200,
    'measured_shoulder_px': 110.0,
    'updated_at': None,
}

# Global variables untuk web server
current_jpeg_bytes = None
latest_status = None
_current_frame_id = 0
_current_frame_source_stamp = 0.0
_current_frame_received_monotonic = 0.0
_last_alert_received_monotonic = 0.0
_latest_processing_stats = {}
_last_processing_stats_monotonic = 0.0
_server_started_monotonic = time.monotonic()
_frame_condition = threading.Condition()

# Durable incident state is intentionally separate from the momentary detector
# signal. A fall remains visible until a caregiver acknowledges it.
_incidents = []
_incidents_lock = threading.RLock()
_active_signal_incident_id = None
_previous_alert_status = False
_fall_clear_started_monotonic = None
_alert_state_lock = threading.Lock()
FALL_CLEAR_DEBOUNCE_SEC = max(1.0, _env_float('HAFIBOT_FALL_CLEAR_DEBOUNCE_SEC', 10.0))

# Lightweight in-memory login throttling. Tailscale ACLs remain the outer
# network boundary, but the web login should still resist password guessing.
_login_failures = {}
_login_failures_lock = threading.Lock()

# Fall snapshot untuk push notification
_fall_snapshot_bytes = None      # JPEG bytes dari frame terakhir saat fall
_fall_snapshot_time = 0.0        # Timestamp snapshot

# Frame hitam kosong untuk ditampilkan saat belum ada gambar
_blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(_blank_frame, "Menunggu kamera...", (160, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
_, _blank_jpg = cv2.imencode('.jpg', _blank_frame)
_blank_bytes = _blank_jpg.tobytes()


def _utc_now_string():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _validate_security_config():
    """Fail closed when deployment credentials are missing or trivial."""
    problems = []
    if len(app.secret_key or '') < 32:
        problems.append('FLASK_SECRET_KEY must contain at least 32 characters')
    if not AUTH_USERNAME:
        problems.append('HAFIBOT_USERNAME is required')
    if len(AUTH_PASSWORD) < 12:
        problems.append('HAFIBOT_PASSWORD must contain at least 12 characters')
    if problems:
        raise RuntimeError('Insecure monitoring configuration: ' + '; '.join(problems))


def _set_private_permissions(path):
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _ensure_private_directory(path):
    if not path:
        return
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _write_json_atomic(path, value):
    directory = os.path.dirname(path)
    if directory:
        _ensure_private_directory(directory)
    temporary_path = '{}.{}.{}.{}.tmp'.format(
        path, os.getpid(), threading.get_ident(), uuid.uuid4().hex
    )
    try:
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        _set_private_permissions(temporary_path)

        # Windows can briefly deny os.replace while Defender, an indexer, or
        # another process has the destination open. Retry the atomic swap while
        # retaining this writer's unique temporary file.
        for attempt in range(6):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.025 * (2 ** attempt))
        _set_private_permissions(path)
    finally:
        try:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        except OSError:
            pass


def _public_incident(incident):
    if not incident:
        return None
    return {
        'id': incident.get('id'),
        'detected_at': incident.get('detected_at'),
        'detector_cleared_at': incident.get('detector_cleared_at'),
        'acknowledged_at': incident.get('acknowledged_at'),
        'acknowledged_by': incident.get('acknowledged_by'),
        'has_snapshot': bool(incident.get('snapshot_filename')),
    }


def _load_incidents():
    global _incidents
    if not INCIDENTS_FILE:
        return
    try:
        if os.path.exists(INCIDENTS_FILE):
            with open(INCIDENTS_FILE, 'r') as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list):
                raise ValueError('incident storage must contain a JSON list')
            _incidents = [item for item in loaded if isinstance(item, dict)][-MAX_INCIDENTS:]
            rospy.loginfo('Loaded {} monitoring incidents'.format(len(_incidents)))
    except Exception as error:
        rospy.logerr('Gagal load monitoring incidents: {}'.format(error))
        _incidents = []


def _load_calibration():
    global current_focal_constant, last_calibration_info, target_filter
    if not CALIBRATION_FILE or not os.path.exists(CALIBRATION_FILE):
        return
    try:
        with open(CALIBRATION_FILE, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and 'focal_constant' in loaded:
            focal_val = float(loaded['focal_constant'])
            if 1000.0 <= focal_val <= 200000.0:
                current_focal_constant = focal_val
                last_calibration_info = loaded
                if 'target_filter' in globals() and target_filter is not None:
                    target_filter.coefficients = (current_focal_constant,)
                rospy.loginfo(f'Loaded shoulder calibration: {current_focal_constant}')
    except Exception as error:
        rospy.logerr(f'Gagal load shoulder calibration: {error}')


def _save_calibration():
    if not CALIBRATION_FILE:
        return
    try:
        _write_json_atomic(CALIBRATION_FILE, last_calibration_info)
    except Exception as error:
        rospy.logerr(f'Gagal save shoulder calibration: {error}')


def _save_incidents_locked():
    if not INCIDENTS_FILE:
        return
    try:
        _write_json_atomic(INCIDENTS_FILE, _incidents[-MAX_INCIDENTS:])
    except Exception as error:
        rospy.logerr('Gagal save monitoring incidents: {}'.format(error))


def _find_incident_locked(incident_id):
    for incident in _incidents:
        if incident.get('id') == incident_id:
            return incident
    return None


def _attention_incident_locked():
    if _active_signal_incident_id:
        active = _find_incident_locked(_active_signal_incident_id)
        if active:
            return active
    # Only the newest episode controls the dashboard attention state. Older
    # unacknowledged records remain in incident history, but acknowledging the
    # newest event must not reveal a long backlog of detector-jitter duplicates.
    if _incidents:
        newest = _incidents[-1]
        if not newest.get('acknowledged_at'):
            return newest
    return None


def _create_incident():
    """Persist a new rising-edge fall incident and its protected snapshot."""
    global _fall_snapshot_bytes, _fall_snapshot_time
    incident_id = '{}-{}'.format(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        uuid.uuid4().hex[:8],
    )
    snapshot_filename = None
    snapshot = current_jpeg_bytes
    if snapshot is not None:
        _fall_snapshot_bytes = snapshot
        _fall_snapshot_time = time.time()
        try:
            _ensure_private_directory(SNAPSHOTS_DIR)
            snapshot_filename = incident_id + '.jpg'
            snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
            with open(snapshot_path, 'wb') as handle:
                handle.write(snapshot)
                handle.flush()
                os.fsync(handle.fileno())
            _set_private_permissions(snapshot_path)
        except Exception as error:
            snapshot_filename = None
            rospy.logwarn('Gagal menyimpan fall snapshot: {}'.format(error))

    incident = {
        'id': incident_id,
        'detected_at': _utc_now_string(),
        'detector_cleared_at': None,
        'acknowledged_at': None,
        'acknowledged_by': None,
        'snapshot_filename': snapshot_filename,
    }
    with _incidents_lock:
        _incidents.append(incident)
        if len(_incidents) > MAX_INCIDENTS:
            removed = _incidents[:-MAX_INCIDENTS]
            del _incidents[:-MAX_INCIDENTS]
            for old_incident in removed:
                old_filename = old_incident.get('snapshot_filename')
                if not old_filename:
                    continue
                old_path = os.path.abspath(os.path.join(SNAPSHOTS_DIR, old_filename))
                if old_path.startswith(os.path.abspath(SNAPSHOTS_DIR) + os.sep):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
        _save_incidents_locked()
    return dict(incident)


def _mark_incident_detector_cleared(incident_id):
    if not incident_id:
        return
    with _incidents_lock:
        incident = _find_incident_locked(incident_id)
        if incident and not incident.get('detector_cleared_at'):
            incident['detector_cleared_at'] = _utc_now_string()
            _save_incidents_locked()


def _acknowledge_incident(incident_id, username):
    with _incidents_lock:
        incident = _find_incident_locked(incident_id)
        if not incident:
            return None
        if not incident.get('acknowledged_at'):
            incident['acknowledged_at'] = _utc_now_string()
            incident['acknowledged_by'] = username
            _save_incidents_locked()
        return _public_incident(incident)


def _snapshot_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt='hafibot-fall-snapshot')


def _snapshot_token(incident_id):
    return _snapshot_serializer().dumps({'incident_id': incident_id})


def _recent_incidents(limit=10):
    with _incidents_lock:
        return [_public_incident(item) for item in reversed(_incidents[-limit:])]


def _status_payload():
    now = time.monotonic()
    frame_age = (
        now - _current_frame_received_monotonic
        if _current_frame_received_monotonic > 0.0 else None
    )
    detector_age = (
        now - _last_alert_received_monotonic
        if _last_alert_received_monotonic > 0.0 else None
    )
    stats_age = (
        now - _last_processing_stats_monotonic
        if _last_processing_stats_monotonic > 0.0 else None
    )
    camera_connected = frame_age is not None and frame_age <= FRAME_STALE_SEC
    detector_connected = detector_age is not None and detector_age <= DETECTOR_STALE_SEC
    stats_connected = stats_age is not None and stats_age <= STATS_STALE_SEC

    with _incidents_lock:
        attention_incident = _public_incident(_attention_incident_locked())

    if latest_status is True or (
        attention_incident and not attention_incident.get('acknowledged_at')
    ):
        state = 'fall'
    elif not camera_connected or not detector_connected:
        if (
            frame_age is None and detector_age is None
            and now - _server_started_monotonic <= max(FRAME_STALE_SEC, DETECTOR_STALE_SEC)
        ):
            state = 'initializing'
        else:
            state = 'offline'
    else:
        state = 'normal'

    stats = dict(_latest_processing_stats) if stats_connected else {}
    return {
        'state': state,
        'is_fall': latest_status is True,
        'active_incident': attention_incident,
        'camera': {
            'connected': camera_connected,
            'frame_age_ms': round(frame_age * 1000.0, 1) if frame_age is not None else None,
            'stream_fps': stats.get('camera_stream_fps'),
        },
        'detector': {
            'connected': detector_connected,
            'status_age_ms': round(detector_age * 1000.0, 1) if detector_age is not None else None,
            'processing_fps': stats.get('processing_fps_rolling'),
            'processing_latency_ms': stats.get('processing_latency_ms_mean'),
            'pose_detection_rate': stats.get('pose_detection_rate_rolling'),
        },
        'robot': {
            'control_mode': globals().get('robot_control_mode', 'auto'),
            'following_enabled': globals().get('following_enabled', True),
            'estop_latched': globals().get('fall_safety_latch', FallEStopLatch()).latched,
            'state': globals().get('robot_latest_telemetry', {}).get('state', 'IDLE') if not globals().get('fall_safety_latch', FallEStopLatch()).latched else 'FALL_ESTOP_LATCHED',
            'target_distance_cm': globals().get('latest_target_distance_cm', 0),
            'target_detected': globals().get('latest_target_detected', False),
            'sonar_left_cm': globals().get('robot_latest_telemetry', {}).get('us_left', 0.0),
            'sonar_center_cm': globals().get('robot_latest_telemetry', {}).get('us_center', 0.0),
            'sonar_right_cm': globals().get('robot_latest_telemetry', {}).get('us_right', 0.0),
        },
        'notifications': {
            'configured': bool(HAS_WEBPUSH and VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY),
        },
        'voice': {
            'status': 'ACTIVE' if chatbot_service.is_enabled() else 'DISABLED',
            'enabled': chatbot_service.is_enabled(),
            'pending_audio_count': len(chatbot_service.proactive_audio_queue) if 'chatbot_service' in globals() else 0,
            'recent_history': (
                chatbot_service.get_history()[-6:]
                if 'chatbot_service' in globals()
                else []
            ),
            'reminders_count': len(chatbot_service.reminder_engine.reminders) if 'chatbot_service' in globals() else 0,
        },
        'recent_incidents': _recent_incidents(10),
        'server_time_unix_sec': time.time(),
        'calibration': {
            'focal_constant': round(float(globals().get('current_focal_constant', DEFAULT_FOCAL_CONSTANT)), 1),
            'default_focal_constant': DEFAULT_FOCAL_CONSTANT,
            'shoulder_px': round(float(globals().get('latest_shoulder_px', 0.0)), 1),
            'effective_px': round(float(globals().get('latest_effective_px', 0.0)), 1),
            'last_valid_shoulder_px': round(float(globals().get('last_valid_shoulder_px', 0.0)), 1),
            'target_distance_cm': int(globals().get('latest_target_distance_cm', 0)),
            'target_detected': bool(globals().get('latest_target_detected', False)),
            'calibrated_at': globals().get('last_calibration_info', {}).get('updated_at'),
            'calibrated_distance_cm': globals().get('last_calibration_info', {}).get('calibrated_distance_cm', 200),
        },
    }


def _csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token


def _login_retry_after(client_key):
    now = time.monotonic()
    with _login_failures_lock:
        failures = [
            value for value in _login_failures.get(client_key, [])
            if now - value < LOGIN_WINDOW_SEC
        ]
        _login_failures[client_key] = failures
        if len(failures) < LOGIN_MAX_FAILURES:
            return 0
        return max(1, int(LOGIN_WINDOW_SEC - (now - failures[0])))


def _record_login_failure(client_key):
    with _login_failures_lock:
        _login_failures.setdefault(client_key, []).append(time.monotonic())


def _clear_login_failures(client_key):
    with _login_failures_lock:
        _login_failures.pop(client_key, None)


# ============================================================
# WEB PUSH NOTIFICATION SYSTEM
# ============================================================
VAPID_PRIVATE_KEY = ""
VAPID_PUBLIC_KEY = ""
VAPID_EMAIL = ""
SUBSCRIPTIONS_FILE = ""
_push_subscriptions = []  # List of push subscription dicts
_push_lock = threading.Lock()
_push_send_lock = threading.Lock()
_last_push_time = 0.0
PUSH_COOLDOWN_SEC = 30


def _init_vapid():
    """Inisialisasi VAPID keys dari environment variables."""
    global VAPID_PRIVATE_KEY, VAPID_PUBLIC_KEY, VAPID_EMAIL
    VAPID_PRIVATE_KEY = os.environ.get('VAPID_PRIVATE_KEY', '')
    VAPID_PUBLIC_KEY = os.environ.get('VAPID_PUBLIC_KEY', '')
    VAPID_EMAIL = os.environ.get('VAPID_EMAIL', 'mailto:hafibot@example.com')

    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        vapid_txt = PROJECT_ROOT / 'Vapid Keys.txt'
        if vapid_txt.exists():
            try:
                with vapid_txt.open('r', encoding='utf-8') as f:
                    content = f.read()
                m = re.search(r'\{[\s\S]*\}', content)
                if m:
                    data = json.loads(m.group(0))
                    VAPID_PUBLIC_KEY = data.get('publicKey', '').strip()
                    VAPID_PRIVATE_KEY = data.get('privateKey', '').strip()
                    subj = data.get('subject', 'mailto:muhammaddaffa09fauzan@gmail.com')
                    VAPID_EMAIL = subj.replace('<', '').replace('>', '').replace(' ', '').strip()
                    rospy.loginfo('  [VAPID] Berhasil memuat VAPID keys dari Vapid Keys.txt')
            except Exception as e:
                rospy.logwarn(f'Gagal membaca Vapid Keys.txt: {e}')

    if not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        rospy.logwarn("=" * 55)
        rospy.logwarn("⚠️  VAPID keys belum diset!")
        rospy.logwarn("   Web Push Notification TIDAK AKTIF.")
        rospy.logwarn("")
        rospy.logwarn("   Generate keys dengan menjalankan di PC/Jetson:")
        rospy.logwarn("     python3 -c \"from pywebpush import webpush; from py_vapid import Vapid; v=Vapid(); v.generate_keys(); print('PRIVATE:', v.private_pem()); print('PUBLIC:', v.public_key)\"")
        rospy.logwarn("")
        rospy.logwarn("   Atau gunakan tool online:")
        rospy.logwarn("     https://vapidkeys.com")
        rospy.logwarn("")
        rospy.logwarn("   Lalu set di .env docker:")
        rospy.logwarn("     VAPID_PRIVATE_KEY=<private_key>")
        rospy.logwarn("     VAPID_PUBLIC_KEY=<public_key>")
        rospy.logwarn("=" * 55)
    else:
        rospy.loginfo(f"  VAPID Public Key: {VAPID_PUBLIC_KEY[:20]}...")
        rospy.loginfo(f"  VAPID Email     : {VAPID_EMAIL}")


def _load_subscriptions():
    """Load push subscriptions dari JSON file."""
    global _push_subscriptions
    if not SUBSCRIPTIONS_FILE:
        return
    try:
        if os.path.exists(SUBSCRIPTIONS_FILE):
            with open(SUBSCRIPTIONS_FILE, 'r') as f:
                _push_subscriptions = json.load(f)
            rospy.loginfo(f"  📋 Loaded {len(_push_subscriptions)} push subscriptions")
    except Exception as e:
        rospy.logwarn(f"Gagal load subscriptions: {e}")
        _push_subscriptions = []


def _save_subscriptions():
    """Simpan push subscriptions ke JSON file."""
    if not SUBSCRIPTIONS_FILE:
        return
    try:
        subscription_dir = os.path.dirname(SUBSCRIPTIONS_FILE)
        if subscription_dir:
            _ensure_private_directory(subscription_dir)
        with open(SUBSCRIPTIONS_FILE, 'w') as f:
            json.dump(_push_subscriptions, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        _set_private_permissions(SUBSCRIPTIONS_FILE)
    except Exception as e:
        rospy.logwarn(f"Gagal save subscriptions: {e}")


def _sub_id(sub):
    """Generate unique ID dari subscription endpoint."""
    return hashlib.sha256(sub.get('endpoint', '').encode()).hexdigest()[:16]


def _send_push_notifications(incident):
    """Kirim push notification ke semua subscriber (background thread)."""
    global _last_push_time

    if not HAS_WEBPUSH or not VAPID_PRIVATE_KEY:
        return

    with _push_send_lock:
        now = time.time()
        if now - _last_push_time < PUSH_COOLDOWN_SEC:
            rospy.logwarn('Push fall event dilewati karena cooldown anti-duplikat')
            return
        _last_push_time = now

    with _push_lock:
        subs = list(_push_subscriptions)

    if not subs:
        rospy.loginfo("&#128276; Fall detected tapi belum ada subscriber push notification")
        return

    now_str = datetime.now().astimezone().strftime('%H:%M:%S %Z')
    image_url = ''
    if incident and incident.get('snapshot_filename'):
        image_url = '/api/incidents/{}/snapshot?token={}'.format(
            incident['id'], _snapshot_token(incident['id'])
        )

    payload = json.dumps({
        "title": "🚨 Jatuh Terdeteksi!",
        "body": f"Hafibot mendeteksi kejadian jatuh pada {now_str}. Segera periksa kondisi lansia.",
        "icon": "/img/hafibot_logo.png",
        "badge": "/img/hafibot_logo.png",
        "image": image_url,
        "url": "/?incident={}".format(incident['id']) if incident else "/",
        "timestamp": now_str,
    })

    failed_subs = []
    sent_count = 0

    for sub in subs:
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims={"sub": VAPID_EMAIL},
                timeout=10,
            )
            sent_count += 1
        except WebPushException as e:
            resp = e.response
            # 410 Gone atau 404 = subscription expired/invalid → hapus
            if resp is not None and resp.status_code in (404, 410):
                failed_subs.append(_sub_id(sub))
                rospy.loginfo(f"🗑️ Subscription expired, dihapus: {_sub_id(sub)}")
            else:
                rospy.logwarn(f"⚠️ Push gagal: {e}")
        except Exception as e:
            rospy.logwarn(f"⚠️ Push error: {e}")

    # Hapus subscription yang expired
    if failed_subs:
        with _push_lock:
            _push_subscriptions[:] = [
                s for s in _push_subscriptions
                if _sub_id(s) not in failed_subs
            ]
            _save_subscriptions()

    if sent_count > 0:
        rospy.loginfo(f"&#128276; Push notification + snapshot terkirim ke {sent_count}/{len(subs)} subscriber")


# ============================================================
# AUTH HELPERS
# ============================================================
def login_required(f):
    """Decorator: redirect ke login page jika belum login."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            if request.path == '/status' or request.path.startswith('/api/'):
                return jsonify({'error': 'authentication_required'}), 401
            if request.path == '/video_feed':
                return Response(status=401)
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.before_request
def protect_state_changing_requests():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return None
    if request.path.startswith('/api/robot/') or request.path.startswith('/api/voice/'):
        return None
    expected = session.get('csrf_token', '')
    provided = request.headers.get('X-CSRF-Token', '') or request.form.get('csrf_token', '')
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'invalid_csrf_token'}), 403
        return Response('Invalid CSRF token', status=403, mimetype='text/plain')
    return None


@app.after_request
def add_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault(
        'Permissions-Policy',
        'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
    )
    response.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; base-uri 'self'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; worker-src 'self'; manifest-src 'self'"
    )
    response.headers.setdefault(
        'Strict-Transport-Security', 'max-age=31536000; includeSubDomains'
    )
    if request.endpoint in ('index', 'login', 'get_status', 'list_incidents'):
        response.headers.setdefault('Cache-Control', 'no-store')
    return response


# ============================================================
# STREAM GENERATOR
# ============================================================
def encode_frame():
    """Yield each new ROS frame once and skip frames for slow clients.

    The previous 30 FPS polling loop repeatedly queued the same 15 FPS source
    frame. Waitress and the browser then played those queued JPEGs in order,
    making a numerically smooth stream appear seconds behind real time.
    """
    last_frame_id = -1

    while True:
        with _frame_condition:
            _frame_condition.wait_for(
                lambda: (
                    rospy.is_shutdown()
                    or (
                        current_jpeg_bytes is not None
                        and _current_frame_id != last_frame_id
                    )
                ),
                timeout=1.0,
            )

            if rospy.is_shutdown():
                return

            if current_jpeg_bytes is None:
                # Send the placeholder once, then wait instead of queueing it.
                if last_frame_id == 0:
                    continue
                frame_bytes = _blank_bytes
                frame_id = 0
                source_stamp = 0.0
            else:
                # Copy references while holding the lock. If this client was
                # blocked by the network, this is the newest frame, not the
                # next old frame in a server-side queue.
                frame_bytes = current_jpeg_bytes
                frame_id = _current_frame_id
                source_stamp = _current_frame_source_stamp

        last_frame_id = frame_id
        source_age_ms = (
            max(0.0, (time.time() - source_stamp) * 1000.0)
            if source_stamp > 0.0 else -1.0
        )
        part_header = (
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: {}\r\n"
            "X-Frame-Id: {}\r\n"
            "X-Source-Age-Ms: {:.1f}\r\n\r\n"
        ).format(len(frame_bytes), frame_id, source_age_ms).encode('ascii')
        yield part_header + frame_bytes + b'\r\n'


# ============================================================
# ROUTES
# ============================================================
@app.route('/img/<path:filename>')
def serve_img(filename):
    """Serve gambar dari folder img (publik, untuk login page background)."""
    return send_from_directory(IMG_DIR, filename)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Halaman login Hafibot."""
    error_msg = ''
    response_status = 200
    retry_after = 0
    client_key = request.remote_addr or 'unknown'
    csrf_token = _csrf_token()
    if request.method == 'POST':
        retry_after = _login_retry_after(client_key)
        if retry_after:
            error_msg = 'Terlalu banyak percobaan. Coba lagi dalam {} detik.'.format(retry_after)
            response_status = 429
        else:
            username = request.form.get('username', '')
            password = request.form.get('password', '')
            username_ok = hmac.compare_digest(username, AUTH_USERNAME)
            password_ok = hmac.compare_digest(password, AUTH_PASSWORD)
            if username_ok and password_ok:
                _clear_login_failures(client_key)
                session.clear()
                session.permanent = True
                session['logged_in'] = True
                session['username'] = AUTH_USERNAME
                _csrf_token()
                return redirect(url_for('index'))
            _record_login_failure(client_key)
            error_msg = 'Username atau password salah!'
            response_status = 401

    html = f"""
    <!DOCTYPE html>
    <html lang="id">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hafibot - Login</title>
        <meta name="description" content="Login ke Hafibot Fall Detection Live Monitoring System">
        <link rel="manifest" href="/manifest.json">
        <meta name="theme-color" content="#E87E24">
        <link rel="icon" href="/img/hafibot_app_icon.svg" type="image/svg+xml">
        <link rel="apple-touch-icon" href="/img/hafibot_logo.png">
        <style>
          *, *::before, *::after {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }}

          html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
          }}

          body {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
              radial-gradient(circle at 12% 12%, rgba(20, 138, 103, 0.12), transparent 34%),
              radial-gradient(circle at 90% 88%, rgba(232, 126, 36, 0.12), transparent 34%),
              #f4f7f6;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
          }}

          .login-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            border: 3px solid #2d2d2d;
            padding: 40px 36px 36px;
            width: 380px;
            max-width: 90vw;
            box-shadow:
              0 8px 32px rgba(0,0,0,0.18),
              0 2px 8px rgba(0,0,0,0.10);
            text-align: center;
            animation: fadeInUp 0.5s ease-out;
          }}

          @keyframes fadeInUp {{
            from {{
              opacity: 0;
              transform: translateY(20px);
            }}
            to {{
              opacity: 1;
              transform: translateY(0);
            }}
          }}

          .login-card img.logo {{
            height: 54px;
            width: auto;
            margin-bottom: 4px;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
          }}

          .login-brand {{
            margin-bottom: 2px;
            color: #147a5d;
            font-size: 34px;
            font-weight: 900;
            letter-spacing: -1.5px;
          }}

          .login-brand span {{ color: #E87E24; }}

          .login-card .subtitle {{
            font-size: 16px;
            font-weight: 700;
            color: #E87E24;
            letter-spacing: 1px;
            margin-bottom: 28px;
          }}

          .form-group {{
            margin-bottom: 18px;
            text-align: left;
          }}

          .form-group label {{
            display: block;
            font-size: 14px;
            font-weight: 700;
            color: #333;
            margin-bottom: 6px;
          }}

          .form-group input {{
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #ddd;
            border-radius: 12px;
            font-family: 'Nunito', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #333;
            background: #fafafa;
            transition: all 0.25s ease;
            outline: none;
          }}

          .form-group input:focus {{
            border-color: #E87E24;
            background: #fff;
            box-shadow: 0 0 0 3px rgba(232, 126, 36, 0.15);
          }}

          .form-group input::placeholder {{
            color: #aaa;
            font-weight: 400;
          }}

          .btn-login {{
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #E87E24, #D46A10);
            color: #fff;
            border: none;
            border-radius: 12px;
            font-family: 'Nunito', sans-serif;
            font-size: 16px;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.25s ease;
            letter-spacing: 0.5px;
            margin-top: 8px;
          }}

          .btn-login:hover {{
            background: linear-gradient(135deg, #D46A10, #B85A08);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(232, 126, 36, 0.35);
          }}

          .btn-login:active {{
            transform: translateY(0);
          }}

          .error-msg {{
            background: #FEE2E2;
            color: #DC2626;
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 16px;
            display: {'block' if error_msg else 'none'};
            animation: shake 0.4s ease;
          }}

          @keyframes shake {{
            0%, 100% {{ transform: translateX(0); }}
            25% {{ transform: translateX(-6px); }}
            75% {{ transform: translateX(6px); }}
          }}

          .badge-container {{
            margin-top: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
          }}

          .badge-container img {{
            height: 50px;
            width: auto;
            filter: drop-shadow(0 2px 6px rgba(0,0,0,0.08));
          }}

          .badge-symbol {{
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border-radius: 11px;
            color: #fff;
            background: #147a5d;
            font-weight: 900;
          }}

          .badge-container span {{
            font-size: 11px;
            color: #888;
            font-weight: 600;
          }}

          @media (max-width: 480px) {{
            .login-card {{
              padding: 28px 24px 24px;
              border-radius: 18px;
            }}
            .login-card img.logo {{
              height: 42px;
            }}
            .login-card .subtitle {{
              font-size: 14px;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="login-card">
          <h1 class="login-brand" aria-label="Hafibot">hafi<span>bot</span></h1>
          <div class="subtitle">Live Monitoring</div>

          <div class="error-msg" id="errorMsg" role="alert" aria-live="assertive">{error_msg}</div>

          <form method="POST" action="/login">
            <input type="hidden" name="csrf_token" value="{csrf_token}">
            <div class="form-group">
              <label for="username">Username</label>
              <input type="text" id="username" name="username" placeholder="Masukkan username" autocomplete="username" required autofocus>
            </div>
            <div class="form-group">
              <label for="password">Password</label>
              <input type="password" id="password" name="password" placeholder="Masukkan password" autocomplete="current-password" required>
            </div>
            <button type="submit" class="btn-login">Masuk</button>
          </form>

          <div class="badge-container">
            <span class="badge-symbol" aria-hidden="true">H</span>
            <span>Fall Detection System</span>
          </div>
        </div>
      </body>
    </html>
    """
    headers = {'Retry-After': str(retry_after)} if retry_after else None
    return Response(html, status=response_status, headers=headers, mimetype='text/html')


@app.route('/logout', methods=['POST'])
def logout():
    """Logout dan redirect ke login page."""
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    csrf_token = _csrf_token()
    return """
    <!DOCTYPE html>
    <html lang="id">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hafibot - Live Monitoring</title>
        <meta name="description" content="Hafibot Fall Detection Live Monitoring System - Real-time elderly fall detection using AI">
        <meta name="csrf-token" content="__CSRF_TOKEN__">
        <link rel="manifest" href="/manifest.json">
        <meta name="theme-color" content="#E87E24">
        <link rel="icon" href="/img/hafibot_app_icon.svg" type="image/svg+xml">
        <link rel="apple-touch-icon" href="/img/hafibot_logo.png">
        <style>
          /* ===== RESET & BASE ===== */
          *, *::before, *::after {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }

          html, body {
            width: 100%;
            height: 100%;
            overflow-x: hidden;
          }

          body {
            font-family: 'Nunito', sans-serif;
            background: #f3f7f5;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            position: relative;
          }

          /* ===== TOP BAR (Logout + Notif) ===== */
          .top-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 12px 16px;
            gap: 10px;
            z-index: 10;
          }

          .top-btn {
            background: rgba(45, 45, 45, 0.85);
            color: #fff;
            border: 2px solid rgba(255,255,255,0.15);
            padding: 8px 16px;
            border-radius: 10px;
            font-family: 'Nunito', sans-serif;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.25s ease;
            backdrop-filter: blur(6px);
            display: inline-flex;
            align-items: center;
            gap: 6px;
          }

          .top-btn:hover {
            transform: translateY(-1px);
          }

          .top-btn.logout:hover {
            background: rgba(220, 38, 38, 0.9);
            border-color: rgba(220, 38, 38, 0.5);
          }

          .top-btn.notif {
            background: rgba(232, 126, 36, 0.9);
            border-color: rgba(232, 126, 36, 0.5);
          }

          .top-btn.notif:hover {
            background: rgba(212, 106, 16, 0.95);
          }

          .top-btn.notif.subscribed {
            background: rgba(43, 110, 79, 0.9);
            border-color: rgba(43, 110, 79, 0.5);
          }

          .top-btn.notif.unsupported {
            background: rgba(100, 100, 100, 0.7);
            cursor: not-allowed;
            opacity: 0.6;
          }

          /* ===== HEADER / LOGO ===== */
          .header {
            text-align: center;
            margin-bottom: 20px;
            z-index: 2;
          }

          .header img.typography-logo {
            height: 60px;
            width: auto;
            display: block;
            margin: 0 auto 4px auto;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.08));
          }

          .header .subtitle {
            font-size: 20px;
            font-weight: 700;
            color: #E87E24;
            letter-spacing: 1px;
            text-shadow: 0 1px 4px rgba(0,0,0,0.06);
          }

          /* ===== VIDEO CONTAINER ===== */
          .video-container {
            position: relative;
            width: 640px;
            max-width: 90vw;
            background: #ffffff;
            border-radius: 24px;
            border: 4px solid #2d2d2d;
            overflow: hidden;
            box-shadow:
              0 8px 32px rgba(0,0,0,0.18),
              0 2px 8px rgba(0,0,0,0.10),
              inset 0 1px 0 rgba(255,255,255,0.1);
            z-index: 2;
          }

          .video-container img.video-feed {
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 4/3;
            object-fit: contain;
            background: #000;
          }

          /* ===== FOOTER / STATUS BAR ===== */
          .footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 640px;
            max-width: 90vw;
            margin-top: 20px;
            z-index: 2;
          }

          /* Status Indicator */
          #status {
            font-size: 26px;
            font-weight: 800;
            color: #2B6E4F;
            transition: all 0.3s ease;
            text-shadow: 0 1px 4px rgba(0,0,0,0.06);
          }

          #status.fall {
            color: #D32F2F;
            animation: pulse-fall 1s ease-in-out infinite;
          }

          @keyframes pulse-fall {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.04); }
          }

          /* Badge Logo (bottom right) */
          .badge-logo {
            height: 80px;
            width: auto;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.10));
            transition: transform 0.3s ease;
          }

          .badge-logo:hover {
            transform: scale(1.06);
          }

          /* ===== TOAST NOTIFICATION ===== */
          .toast {
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%) translateY(80px);
            background: rgba(45, 45, 45, 0.95);
            color: #fff;
            padding: 12px 24px;
            border-radius: 12px;
            font-family: 'Nunito', sans-serif;
            font-size: 14px;
            font-weight: 700;
            z-index: 100;
            opacity: 0;
            transition: all 0.35s ease;
            backdrop-filter: blur(8px);
            pointer-events: none;
          }

          .toast.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
          }

          /* ===== RESPONSIVE ===== */
          @media (max-width: 700px) {
            .header img.typography-logo {
              height: 44px;
            }
            .header .subtitle {
              font-size: 16px;
            }
            .video-container {
              border-radius: 16px;
              border-width: 3px;
            }
            #status {
              font-size: 20px;
            }
            .badge-logo {
              height: 56px;
            }
            .footer {
              margin-top: 14px;
            }
          }

          @media (max-width: 480px) {
            .header img.typography-logo {
              height: 36px;
            }
            .header .subtitle {
              font-size: 14px;
            }
            #status {
              font-size: 16px;
            }
            .badge-logo {
              height: 44px;
            }
            .top-btn {
              font-size: 11px;
              padding: 6px 12px;
            }
          }

          /* ===== DEPLOYMENT DASHBOARD ===== */
          :root {
            --green: #147a5d;
            --green-dark: #0d5a44;
            --green-soft: #e7f5ef;
            --orange: #df741c;
            --red: #c82727;
            --red-dark: #991b1b;
            --red-soft: #fff0f0;
            --amber: #a85d00;
            --amber-soft: #fff7df;
            --ink: #17211e;
            --muted: #65726d;
            --line: #d9e2df;
            --surface: #ffffff;
            --page: #f3f7f5;
          }

          html, body { height: auto; min-height: 100%; }

          body {
            display: block;
            min-height: 100vh;
            padding: 24px;
            color: var(--ink);
            background:
              radial-gradient(circle at 5% 0%, rgba(20, 122, 93, 0.09), transparent 26rem),
              radial-gradient(circle at 100% 100%, rgba(223, 116, 28, 0.08), transparent 28rem),
              var(--page);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }

          button, input { font: inherit; }

          .skip-link {
            position: fixed;
            top: 8px;
            left: 8px;
            z-index: 999;
            padding: 10px 14px;
            border-radius: 8px;
            color: #fff;
            background: var(--ink);
            transform: translateY(-150%);
          }
          .skip-link:focus { transform: translateY(0); }

          .app-shell {
            width: min(1120px, 100%);
            margin: 0 auto;
          }

          .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            min-height: 60px;
            margin-bottom: 20px;
          }

          .brand { display: flex; align-items: center; gap: 12px; }
          .brand-mark {
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            border-radius: 13px;
            color: #fff;
            background: linear-gradient(145deg, var(--green), var(--green-dark));
            box-shadow: 0 7px 18px rgba(20, 122, 93, 0.22);
            font-weight: 900;
          }
          .brand-name { font-size: 22px; font-weight: 900; letter-spacing: -0.04em; }
          .brand-name .accent { color: var(--orange); }
          .brand-copy { color: var(--muted); font-size: 12px; font-weight: 700; }

          .header-actions { display: flex; align-items: center; gap: 10px; }
          .action-btn {
            min-height: 44px;
            padding: 10px 14px;
            border: 1px solid var(--line);
            border-radius: 11px;
            color: var(--ink);
            background: rgba(255,255,255,0.9);
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            text-decoration: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
          }
          .action-btn:hover { border-color: #9dafaa; box-shadow: 0 5px 14px rgba(23,33,30,0.08); }
          .action-btn:active { transform: translateY(1px); }
          .action-btn.primary { border-color: var(--green); color: #fff; background: var(--green); }
          .action-btn.subscribed { border-color: var(--green); color: var(--green-dark); background: var(--green-soft); }
          .action-btn.unsupported { cursor: not-allowed; opacity: 0.65; }
          .action-btn.danger:hover { border-color: #f2b8b8; color: var(--red-dark); background: var(--red-soft); }
          .logout-form { display: inline; }

          .status-banner {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 14px;
            min-height: 82px;
            margin-bottom: 18px;
            padding: 16px 18px;
            border: 1px solid #ecd59a;
            border-radius: 16px;
            background: var(--amber-soft);
            box-shadow: 0 7px 22px rgba(23,33,30,0.06);
          }
          .status-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            color: #fff;
            background: var(--amber);
            font-size: 20px;
            font-weight: 900;
          }
          .status-title { font-size: 18px; font-weight: 900; }
          .status-detail { margin-top: 2px; color: var(--muted); font-size: 13px; font-weight: 650; }
          .status-time { color: var(--muted); font-size: 12px; font-weight: 750; white-space: nowrap; }
          .status-banner.normal { border-color: #b8dfd0; background: var(--green-soft); }
          .status-banner.normal .status-icon { background: var(--green); }
          .status-banner.offline { border-color: #efc489; background: #fff4e5; }
          .status-banner.offline .status-icon { background: var(--orange); }
          .status-banner.fall { border-color: #ef9c9c; background: var(--red-soft); }
          .status-banner.fall .status-icon { background: var(--red); animation: pulse-fall 1.1s ease-in-out infinite; }

          .dashboard-grid {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(280px, 0.85fr);
            gap: 18px;
            align-items: start;
          }
          .card {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: 0 8px 26px rgba(23,33,30,0.07);
          }
          .video-card { padding: 12px; }
          .video-container {
            width: 100%;
            max-width: none;
            border: 0;
            border-radius: 13px;
            box-shadow: none;
            background: #111;
          }
          .video-container img.video-feed { object-fit: cover; }
          .stream-overlay {
            position: absolute;
            inset: 0;
            display: none;
            place-items: center;
            padding: 24px;
            color: #fff;
            background: rgba(18, 29, 26, 0.76);
            text-align: center;
            backdrop-filter: blur(3px);
          }
          .stream-overlay.show { display: grid; }
          .stream-overlay strong { display: block; font-size: 18px; }
          .stream-overlay span { display: block; margin-top: 5px; color: #dce7e3; font-size: 13px; }
          .fall-overlay { background: rgba(118, 12, 12, 0.86); }
          .fall-overlay .ack-btn { margin-top: 14px; border-color: #fff; color: var(--red-dark); background: #fff; }

          .video-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 8px 14px;
            padding: 12px 4px 2px;
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
          }
          .live-label { display: inline-flex; align-items: center; gap: 7px; color: var(--ink); }
          .live-dot { width: 9px; height: 9px; border-radius: 50%; background: #aab5b1; }
          .live-dot.online { background: #20a775; box-shadow: 0 0 0 4px rgba(32,167,117,0.13); }

          .side-stack { display: grid; gap: 18px; }
          .card-section { padding: 18px; }
          .card-title { margin-bottom: 14px; font-size: 14px; font-weight: 900; }
          .health-list { display: grid; gap: 12px; }
          .health-row { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 10px; }
          .health-dot { width: 10px; height: 10px; border-radius: 50%; background: #aab5b1; }
          .health-dot.online { background: #20a775; }
          .health-dot.offline { background: var(--red); }
          .health-label { font-size: 13px; font-weight: 800; }
          .health-value { color: var(--muted); font-size: 12px; font-weight: 700; text-align: right; }
          .notification-actions { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 14px; }
          .notification-actions .action-btn { width: 100%; }

          .events-card { margin-top: 18px; padding: 18px; }
          .calibration-card { margin-top: 18px; padding: 18px; }
          .events-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
          .events-header h2 { font-size: 15px; }
          .events-note { color: var(--muted); font-size: 12px; }
          .incident-list { display: grid; gap: 8px; }
          .incident-row {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 11px;
            min-height: 50px;
            padding: 9px 10px;
            border-radius: 11px;
            background: #f7f9f8;
          }
          .incident-row.pending { background: var(--red-soft); }
          .incident-symbol { width: 31px; height: 31px; display: grid; place-items: center; border-radius: 9px; color: var(--red-dark); background: #ffdcdc; font-weight: 900; }
          .incident-main { min-width: 0; }
          .incident-title { font-size: 12px; font-weight: 850; }
          .incident-time { margin-top: 2px; color: var(--muted); font-size: 11px; }
          .incident-state { color: var(--muted); font-size: 11px; font-weight: 800; white-space: nowrap; }
          .empty-state { padding: 18px 8px; color: var(--muted); font-size: 13px; text-align: center; }

          .toast { max-width: min(420px, calc(100vw - 32px)); text-align: center; }
          :focus-visible { outline: 3px solid rgba(223,116,28,0.42); outline-offset: 2px; }

          @media (max-width: 820px) {
            body { padding: 16px; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .side-stack { grid-template-columns: 1fr; }
          }
          @media (max-width: 560px) {
            body { padding: 12px; padding-bottom: calc(12px + env(safe-area-inset-bottom)); }
            .app-header { align-items: flex-start; }
            .brand-copy { display: none; }
            .header-actions { gap: 6px; }
            .header-actions .action-btn { min-width: 44px; padding: 10px; font-size: 0; }
            .header-actions .action-btn .button-icon { font-size: 16px; }
            .status-banner { grid-template-columns: auto 1fr; padding: 14px; }
            .status-time { grid-column: 2; }
            .status-title { font-size: 16px; }
            .card, .status-banner { border-radius: 14px; }
            .video-card { padding: 7px; }
            .card-section, .events-card, .calibration-card { padding: 15px; }
            .notification-actions { grid-template-columns: 1fr; }
          }
          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
          }
        </style>
      </head>
      <body>

        <a class="skip-link" href="#mainContent">Lewati ke pemantauan</a>
        <main class="app-shell" id="mainContent">
          <header class="app-header">
            <div class="brand" aria-label="Hafibot Live Monitoring">
              <span class="brand-mark" aria-hidden="true">H</span>
              <div>
                <div class="brand-name">hafi<span class="accent">bot</span></div>
                <div class="brand-copy">Pemantauan deteksi jatuh</div>
              </div>
            </div>
            <div class="header-actions">
              <button class="action-btn primary" id="notifBtn" type="button">
                <span class="button-icon" aria-hidden="true">&#128276;</span>
                <span id="notifBtnText">Aktifkan notifikasi</span>
              </button>
              <form class="logout-form" method="POST" action="/logout">
                <input type="hidden" name="csrf_token" value="__CSRF_TOKEN__">
                <button class="action-btn danger" id="logoutBtn" type="submit">
                  <span class="button-icon" aria-hidden="true">&#8618;</span>
                  <span>Keluar</span>
                </button>
              </form>
            </div>
          </header>

          <section class="status-banner" id="statusBanner" aria-live="polite" aria-atomic="true">
            <div class="status-icon" id="statusIcon" aria-hidden="true">&hellip;</div>
            <div>
              <div class="status-title" id="statusTitle">Menghubungkan ke Hafibot</div>
              <div class="status-detail" id="statusDetail">Menunggu kamera dan detektor memberikan data.</div>
            </div>
            <time class="status-time" id="statusTime">Belum ada data</time>
          </section>

          <div class="dashboard-grid">
            <section class="card video-card" aria-label="Kamera langsung">
              <div class="video-container">
                <img class="video-feed" id="videoFeed" src="/video_feed" alt="Siaran kamera Hafibot secara langsung">
                <div class="stream-overlay" id="streamOverlay">
                  <div>
                    <strong id="streamOverlayTitle">Kamera tidak tersedia</strong>
                    <span id="streamOverlayDetail">Mencoba menyambungkan kembali&hellip;</span>
                  </div>
                </div>
                <div class="stream-overlay fall-overlay" id="fallOverlay" role="alert">
                  <div>
                    <strong>Jatuh terdeteksi</strong>
                    <span id="fallOverlayTime">Segera periksa kondisi orang yang dipantau.</span>
                    <button class="action-btn ack-btn" id="ackBtn" type="button">Tandai sudah diperiksa</button>
                  </div>
                </div>
              </div>
              <div class="video-meta">
                <span class="live-label"><span class="live-dot" id="liveDot"></span><span id="cameraMeta">Menunggu kamera</span></span>
                <span id="frameAge">Frame belum tersedia</span>
              </div>
            </section>

            <aside class="side-stack" aria-label="Kesehatan sistem">
              <section class="card card-section">
                <h2 class="card-title">Kesehatan sistem</h2>
                <div class="health-list">
                  <div class="health-row">
                    <span class="health-dot" id="cameraHealthDot"></span>
                    <span class="health-label">Kamera</span>
                    <span class="health-value" id="cameraHealthValue">Memeriksa&hellip;</span>
                  </div>
                  <div class="health-row">
                    <span class="health-dot" id="detectorHealthDot"></span>
                    <span class="health-label">Detektor AI</span>
                    <span class="health-value" id="detectorHealthValue">Memeriksa&hellip;</span>
                  </div>
                  <div class="health-row">
                    <span class="health-dot" id="notificationHealthDot"></span>
                    <span class="health-label">Notifikasi perangkat</span>
                    <span class="health-value" id="notificationHealthValue">Memeriksa&hellip;</span>
                  </div>
                </div>
                <div class="notification-actions">
                  <button class="action-btn" id="notificationCardBtn" type="button">Atur notifikasi</button>
                  <button class="action-btn" id="testNotifBtn" type="button" disabled>Uji</button>
                </div>
              </section>
              <section class="card card-section" aria-label="Kontrol Robot Hafibot">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                  <h2 class="card-title" style="margin-bottom:0;">Kontrol Robot</h2>
                  <div style="display: flex; background: #eef3f1; border-radius: 8px; padding: 2px; gap: 2px;">
                    <button id="tabAutoMode" type="button" style="border: none; background: #fff; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; cursor: pointer; color: var(--green); box-shadow: 0 1px 3px rgba(0,0,0,0.08);">&#129302; Auto</button>
                    <button id="tabManualMode" type="button" style="border: none; background: transparent; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; cursor: pointer; color: var(--muted);">&#127918; Manual</button>
                  </div>
                </div>
                <div class="health-list">
                  <div class="health-row">
                    <span class="health-dot" id="robotStateDot"></span>
                    <span class="health-label">Status Robot</span>
                    <span class="health-value" id="robotStateValue">Memeriksa...</span>
                  </div>
                  <div class="health-row">
                    <span class="health-dot" id="targetDistDot"></span>
                    <span class="health-label">Jarak Target</span>
                    <span class="health-value" id="targetDistValue">-</span>
                  </div>
                  <div class="health-row">
                    <span class="health-dot" id="sonarDot"></span>
                    <span class="health-label">Sonar Ultrasonik</span>
                    <span class="health-value" id="sonarValue">[0, 0, 0] cm</span>
                  </div>
                </div>
                <!-- Auto Following Panel -->
                <div id="autoControlsPanel" style="display: grid; gap: 8px; margin-top: 14px;">
                  <button class="action-btn" id="toggleFollowingBtn" type="button" style="background: var(--green); color: white; border: none; font-weight: bold; width: 100%;">Ikuti Pengguna: AKTIF</button>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <button class="action-btn" id="resetEstopBtn" type="button" style="background: var(--amber-soft); color: var(--amber); border-color: var(--amber); font-weight: bold;">&#128275; Reset ESTOP</button>
                    <button class="action-btn" id="manualEstopBtn" type="button" style="background: var(--red-soft); color: var(--red-dark); border-color: var(--red); font-weight: bold;">&#9940; Rem E-STOP</button>
                  </div>
                </div>

                <!-- Manual Takeover D-Pad Panel -->
                <div id="manualControlsPanel" style="display: none; margin-top: 14px;">
                  <div style="text-align: center; margin-bottom: 8px; font-size: 11px; color: var(--muted); font-weight: 700;">
                    Gunakan tombol / keyboard: <b>W / A / S / D</b>
                  </div>
                  <div class="dpad-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; max-width: 220px; margin: 0 auto;">
                    <button class="dpad-btn" id="dpadPivotLeft" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: #f7f9f8; font-weight: bold; cursor: pointer;">&#8634; Q</button>
                    <button class="dpad-btn" id="dpadForward" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: var(--green-soft); color: var(--green); font-weight: bold; font-size: 16px; cursor: pointer;">&#9650; W</button>
                    <button class="dpad-btn" id="dpadPivotRight" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: #f7f9f8; font-weight: bold; cursor: pointer;">&#8635; E</button>
                    
                    <button class="dpad-btn" id="dpadLeft" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: #f7f9f8; font-weight: bold; font-size: 16px; cursor: pointer;">&#9668; A</button>
                    <button class="dpad-btn" id="dpadStop" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: var(--red-soft); color: var(--red-dark); font-weight: bold; font-size: 13px; cursor: pointer;">&#9632; S</button>
                    <button class="dpad-btn" id="dpadRight" type="button" style="padding: 10px 0; border: 1px solid var(--line); border-radius: 8px; background: #f7f9f8; font-weight: bold; font-size: 16px; cursor: pointer;">&#9658; D</button>
                  </div>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px;">
                    <button class="action-btn" id="resetEstopBtnManual" type="button" style="background: var(--amber-soft); color: var(--amber); border-color: var(--amber); font-weight: bold; font-size: 11px;">&#128275; Reset ESTOP</button>
                    <button class="action-btn" id="manualEstopBtnManual" type="button" style="background: var(--red-soft); color: var(--red-dark); border-color: var(--red); font-weight: bold; font-size: 11px;">&#9940; Rem E-STOP</button>
                  </div>
                </div>
              </section>
            </aside>
          </div>

          <!-- Voice Assistant & Chatbot Card -->
          <section class="card voice-card" aria-label="Asisten Suara & Chatbot Robot" style="margin-top: 14px;">
            <div class="events-header" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
              <h2 style="font-size: 18px; font-weight: 800; color: var(--navy); display: flex; align-items: center; gap: 8px;">
                <span>&#127897;&#65039;</span> Asisten Suara & Chatbot (NLU + Google ASR)
              </h2>
              <div style="display: flex; align-items: center; gap: 8px;">
                <button class="action-btn" id="toggleVoiceBtn" type="button" aria-pressed="true" style="background: var(--green); color: white; border: none; font-weight: bold; font-size: 12px; padding: 5px 12px; border-radius: 8px; cursor: pointer;">Asisten Suara: AKTIF</button>
                <span class="events-note" id="voiceStatusBadge" style="background: var(--green-soft); color: var(--green-dark); padding: 4px 10px; border-radius: 8px; font-weight: bold; font-size: 12px;">AKTIF (USB Mic + I2S Speaker)</span>
              </div>
            </div>

            <!-- Conversation Display Box -->
            <div id="voiceHistoryBox" style="background: #f7faf8; border: 1px solid var(--line); border-radius: 12px; padding: 14px; min-height: 90px; max-height: 200px; overflow-y: auto; margin-top: 12px; display: flex; flex-direction: column; gap: 8px;">
              <div style="font-size: 13px; color: var(--muted); text-align: center; margin: auto;">Menunggu ucapan suara dari mikrofon atau input teks...</div>
            </div>

            <!-- Quick Chips -->
            <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;">
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Robot mulai ikuti saya')" style="font-size: 12px; padding: 6px 12px; background: var(--green-soft); color: var(--green-dark); border: 1px solid var(--green); cursor: pointer;">&#9654; Mulai Ikuti</button>
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Robot berhenti')" style="font-size: 12px; padding: 6px 12px; background: var(--red-soft); color: var(--red-dark); border: 1px solid var(--red); cursor: pointer;">&#9209; Berhenti</button>
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Sekarang jam berapa?')" style="font-size: 12px; padding: 6px 12px; background: #eef2f6; border: 1px solid var(--line); cursor: pointer;">&#128340; Jam Berapa?</button>
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Bagaimana cuaca hari ini?')" style="font-size: 12px; padding: 6px 12px; background: #eef2f6; border: 1px solid var(--line); cursor: pointer;">&#9925; Cuaca BMKG</button>
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Cek jadwal pengingat')" style="font-size: 12px; padding: 6px 12px; background: #eef2f6; border: 1px solid var(--line); cursor: pointer;">&#128203; Jadwal Solat / Obat</button>
              <button class="action-btn" type="button" onclick="sendVoicePrompt('Saya tidak apa-apa')" style="font-size: 12px; padding: 6px 12px; background: var(--amber-soft); color: var(--amber); border: 1px solid var(--amber); cursor: pointer;">&#128275; Saya Tidak Apa-apa</button>
            </div>

            <!-- Text Prompt Input -->
            <form id="voicePromptForm" onsubmit="handleVoiceFormSubmit(event)" style="display: flex; gap: 8px; margin-top: 10px;">
              <input type="text" id="voicePromptInput" placeholder="Ketik perintah suara atau ajak bicara robot..." style="flex: 1; padding: 10px 14px; border: 1px solid var(--line); border-radius: 10px; font-family: inherit; font-size: 14px;">
              <button class="action-btn primary" type="submit" style="padding: 10px 18px; font-weight: bold; cursor: pointer;">Kirim</button>
            </form>
            <audio id="voiceAudioPlayer" style="display: none;"></audio>
          </section>

          <section class="card events-card" aria-labelledby="eventsTitle">
            <div class="events-header">
              <h2 id="eventsTitle">Kejadian terbaru</h2>
              <span class="events-note">Kejadian tersimpan sampai diperiksa</span>
            </div>
            <div class="incident-list" id="incidentList">
              <div class="empty-state">Memuat riwayat kejadian&hellip;</div>
            </div>
          </section>

          <!-- Rekalibrasi Bahu & Jarak Following -->
          <section class="card calibration-card" aria-labelledby="calibTitle">
            <div class="events-header" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
              <h2 id="calibTitle" style="font-size: 16px; font-weight: 800; color: var(--navy); display: flex; align-items: center; gap: 8px; margin: 0;">
                <span>&#128207;</span> Kalibrasi Lebar Bahu &amp; Jarak Following
              </h2>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span id="calibStatusBadge" style="background: var(--green-soft); color: var(--green-dark); padding: 4px 10px; border-radius: 8px; font-weight: bold; font-size: 12px;">Memuat...</span>
              </div>
            </div>

            <p style="margin: 10px 0 16px; font-size: 13px; color: var(--muted); line-height: 1.5;">
              Sesuaikan konstanta estimasi jarak kamera agar akurat untuk postur subjek pengguna (bahu sempit, sedang, atau lebar).
              Berdirilah menghadap kamera di jarak yang diukur dengan meteran lantai (misal 2.0 meter / 200 cm), lalu tekan <strong>Kalibrasi Sekarang</strong>.
            </p>

            <!-- Grid Nilai Live -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
              <div style="background: #f8faf9; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;">
                <div style="font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase;">Lebar Bahu Kamera (Live)</div>
                <div style="font-size: 20px; font-weight: 900; color: var(--ink); margin-top: 4px;" id="liveShoulderPx">0.0 px</div>
                <div style="font-size: 11px; margin-top: 2px;" id="liveShoulderStatus">Menunggu target...</div>
              </div>
              <div style="background: #f8faf9; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;">
                <div style="font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase;">Estimasi Jarak Saat Ini</div>
                <div style="font-size: 20px; font-weight: 900; color: var(--green); margin-top: 4px;" id="liveTargetDist">0 cm</div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 2px;">Dihitung dari model optik</div>
              </div>
              <div style="background: #f8faf9; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px;">
                <div style="font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase;">Konstanta Fokal Aktif</div>
                <div style="font-size: 20px; font-weight: 900; color: var(--orange); margin-top: 4px;" id="liveFocalConst">22000.0</div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 2px;" id="lastCalibTime">Default bawaan</div>
              </div>
            </div>

            <!-- Formulir & Tombol Aksi Kalibrasi -->
            <div style="background: #ffffff; border: 1px solid var(--line); border-radius: 14px; padding: 16px; display: flex; flex-wrap: wrap; align-items: flex-end; gap: 14px;">
              <div style="flex: 1; min-width: 220px;">
                <label for="calibDistanceInput" style="display: block; font-size: 12px; font-weight: 750; color: var(--ink); margin-bottom: 6px;">
                  Jarak Berdiri Subjek Saat Ini (cm):
                </label>
                <div style="display: flex; gap: 8px; align-items: center;">
                  <input type="number" id="calibDistanceInput" min="50" max="500" value="200" step="5" style="flex: 1; padding: 9px 12px; border: 1px solid var(--line); border-radius: 8px; font-size: 14px; font-weight: 700;">
                  <span style="font-size: 13px; font-weight: 700; color: var(--muted);">cm</span>
                </div>
              </div>

              <!-- Tombol Preset Jarak Cepat -->
              <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <button type="button" class="action-btn" onclick="setPresetDist(150)" style="font-size: 11px; padding: 8px 10px;">1.5 m (150cm)</button>
                <button type="button" class="action-btn" onclick="setPresetDist(200)" style="font-size: 11px; padding: 8px 10px;">2.0 m (200cm)</button>
                <button type="button" class="action-btn" onclick="setPresetDist(250)" style="font-size: 11px; padding: 8px 10px;">2.5 m (250cm)</button>
              </div>

              <!-- Tombol Kalibrasi & Reset -->
              <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <button type="button" class="action-btn primary" id="btnCalibrateNow" style="padding: 10px 18px; font-size: 13px; font-weight: 800;">
                  &#127919; Kalibrasi Sekarang
                </button>
                <button type="button" class="action-btn" id="btnResetCalib" style="padding: 10px 14px; font-size: 13px; font-weight: 700; color: var(--muted); border-color: var(--line);">
                  &#128260; Reset Default (22000)
                </button>
              </div>
            </div>
          </section>
        </main>

        <div class="toast" id="toast" role="status" aria-live="polite"></div>

        <script>
          // ===== TOAST HELPER =====
          function showToast(msg, duration = 3000) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), duration);
          }

          const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
          const videoFeed = document.getElementById('videoFeed');
          let currentIncidentId = null;
          let lastAnnouncedIncidentId = null;
          let videoErrored = false;
          let voiceEnabled = true;
          let voiceToggleBusy = false;

          function renderVoiceControls(isEnabled) {
            voiceEnabled = Boolean(isEnabled);
            const vBtn = document.getElementById('toggleVoiceBtn');
            const vBadge = document.getElementById('voiceStatusBadge');
            if (vBtn) {
              vBtn.innerText = voiceToggleBusy
                ? 'Memproses...'
                : (voiceEnabled ? 'Asisten Suara: AKTIF' : 'Asisten Suara: NONAKTIF');
              vBtn.style.background = voiceEnabled ? 'var(--green)' : 'var(--muted)';
              vBtn.disabled = voiceToggleBusy;
              vBtn.setAttribute('aria-pressed', voiceEnabled ? 'true' : 'false');
            }
            if (vBadge) {
              vBadge.innerText = voiceEnabled ? 'AKTIF (USB Mic + I2S Speaker)' : 'NONAKTIF';
              vBadge.style.background = voiceEnabled ? 'var(--green-soft)' : '#f0f2f5';
              vBadge.style.color = voiceEnabled ? 'var(--green-dark)' : 'var(--muted)';
            }
            document.querySelectorAll('.voice-card button:not(#toggleVoiceBtn), .voice-card input').forEach((control) => {
              control.disabled = !voiceEnabled;
              control.style.opacity = voiceEnabled ? '1' : '0.5';
            });
            if (!voiceEnabled) {
              const player = document.getElementById('voiceAudioPlayer');
              if (player) {
                player.pause();
                player.removeAttribute('src');
                player.load();
              }
            }
          }

          function formatEventTime(value) {
            if (!value) return 'Waktu tidak tersedia';
            const date = new Date(value);
            if (Number.isNaN(date.getTime())) return value;
            return new Intl.DateTimeFormat('id-ID', {
              dateStyle: 'medium', timeStyle: 'short'
            }).format(date);
          }

          function formatMetric(value, suffix, digits = 1) {
            const number = Number(value);
            return Number.isFinite(number) ? `${number.toFixed(digits)} ${suffix}` : 'Belum tersedia';
          }

          function setHealth(dotId, valueId, connected, value) {
            const dot = document.getElementById(dotId);
            dot.className = `health-dot ${connected ? 'online' : 'offline'}`;
            document.getElementById(valueId).textContent = value;
          }

          function renderIncidents(incidents) {
            const list = document.getElementById('incidentList');
            list.replaceChildren();
            if (!incidents || incidents.length === 0) {
              const empty = document.createElement('div');
              empty.className = 'empty-state';
              empty.textContent = 'Belum ada kejadian jatuh yang tercatat.';
              list.appendChild(empty);
              return;
            }
            incidents.forEach((incident) => {
              const row = document.createElement('div');
              row.className = `incident-row ${incident.acknowledged_at ? '' : 'pending'}`;

              const symbol = document.createElement('span');
              symbol.className = 'incident-symbol';
              symbol.textContent = '!';

              const main = document.createElement('div');
              main.className = 'incident-main';
              const title = document.createElement('div');
              title.className = 'incident-title';
              title.textContent = 'Jatuh terdeteksi';
              const eventTime = document.createElement('div');
              eventTime.className = 'incident-time';
              eventTime.textContent = formatEventTime(incident.detected_at);
              main.append(title, eventTime);

              const state = document.createElement(incident.has_snapshot ? 'a' : 'span');
              state.className = 'incident-state';
              state.textContent = incident.acknowledged_at ? 'Sudah diperiksa' : 'Perlu diperiksa';
              if (incident.has_snapshot) {
                state.href = `/api/incidents/${encodeURIComponent(incident.id)}/snapshot`;
                state.target = '_blank';
                state.rel = 'noopener';
                state.title = 'Buka snapshot kejadian';
              }
              row.append(symbol, main, state);
              list.appendChild(row);
            });
          }

          function renderDisconnected(detail) {
            const banner = document.getElementById('statusBanner');
            banner.className = 'status-banner offline';
            document.getElementById('statusIcon').textContent = '!';
            document.getElementById('statusTitle').textContent = 'Koneksi pemantauan terputus';
            document.getElementById('statusDetail').textContent = detail;
            document.getElementById('statusTime').textContent = 'Data tidak dapat diperbarui';
            document.getElementById('streamOverlay').classList.add('show');
            document.getElementById('streamOverlayTitle').textContent = 'Siaran tidak dapat diverifikasi';
            setHealth('cameraHealthDot', 'cameraHealthValue', false, 'Terputus');
            setHealth('detectorHealthDot', 'detectorHealthValue', false, 'Terputus');
          }

          function renderStatus(data) {
            const banner = document.getElementById('statusBanner');
            const icon = document.getElementById('statusIcon');
            const title = document.getElementById('statusTitle');
            const detail = document.getElementById('statusDetail');
            const overlay = document.getElementById('streamOverlay');
            const fallOverlay = document.getElementById('fallOverlay');
            const incident = data.active_incident;
            currentIncidentId = incident ? incident.id : null;
            banner.className = `status-banner ${data.state}`;

            if (data.state === 'fall') {
              icon.textContent = '!';
              title.textContent = incident && incident.acknowledged_at
                ? 'Jatuh masih terdeteksi — sudah diperiksa'
                : 'Jatuh terdeteksi — perlu diperiksa';
              detail.textContent = 'Segera pastikan kondisi orang yang dipantau.';
              banner.setAttribute('role', 'alert');
              fallOverlay.classList.add('show');
              document.getElementById('fallOverlayTime').textContent = incident
                ? formatEventTime(incident.detected_at)
                : 'Segera periksa kondisi orang yang dipantau.';
              document.getElementById('ackBtn').hidden = Boolean(incident && incident.acknowledged_at);
              if (incident && incident.id !== lastAnnouncedIncidentId) {
                lastAnnouncedIncidentId = incident.id;
                if ('vibrate' in navigator) navigator.vibrate([300, 120, 300]);
              }
            } else if (data.state === 'normal') {
              icon.textContent = '✓';
              title.textContent = 'Tidak ada jatuh terdeteksi';
              detail.textContent = 'Kamera dan detektor aktif. Status ini bukan jaminan keselamatan.';
              banner.removeAttribute('role');
              fallOverlay.classList.remove('show');
            } else if (data.state === 'offline') {
              icon.textContent = '!';
              title.textContent = 'Pemantauan tidak lengkap';
              const missing = [];
              if (!data.camera.connected) missing.push('kamera');
              if (!data.detector.connected) missing.push('detektor');
              detail.textContent = `${missing.join(' dan ')} tidak mengirim data terbaru.`;
              banner.removeAttribute('role');
              fallOverlay.classList.remove('show');
            } else {
              icon.textContent = '&hellip;';
              title.textContent = 'Menghubungkan ke Hafibot';
              detail.textContent = 'Menunggu kamera dan detektor memberikan data.';
              banner.removeAttribute('role');
              fallOverlay.classList.remove('show');
            }

            document.getElementById('statusTime').textContent = 'Diperbarui baru saja';
            const cameraOnline = Boolean(data.camera.connected) && !videoErrored;
            overlay.classList.toggle('show', !cameraOnline);
            document.getElementById('liveDot').className = `live-dot ${cameraOnline ? 'online' : ''}`;
            document.getElementById('cameraMeta').textContent = cameraOnline
              ? `Kamera live${data.camera.stream_fps ? ` &bull; ${Number(data.camera.stream_fps).toFixed(1)} FPS` : ''}`
              : 'Kamera tidak tersedia';
            document.getElementById('frameAge').textContent = data.camera.frame_age_ms == null
              ? 'Frame belum tersedia'
              : `Usia frame ${Math.round(data.camera.frame_age_ms)} ms`;

            setHealth(
              'cameraHealthDot', 'cameraHealthValue', cameraOnline,
              cameraOnline ? formatMetric(data.camera.stream_fps, 'FPS') : 'Terputus'
            );
            setHealth(
              'detectorHealthDot', 'detectorHealthValue', Boolean(data.detector.connected),
              data.detector.connected
                ? formatMetric(data.detector.processing_latency_ms, 'ms')
                : 'Terputus'
            );

            // Robot following & telemetry controls
            if (data.robot) {
              const r = data.robot;
              const isEstop = Boolean(r.estop_latched);
              const rState = isEstop ? 'ESTOP TERKUNCI' : (!r.following_enabled ? 'NONAKTIF' : (r.target_detected ? 'FOLLOWING' : 'MENCARI TARGET'));
              setHealth('robotStateDot', 'robotStateValue', !isEstop && (r.following_enabled || r.target_detected), rState);
              if (isEstop) {
                const dot = document.getElementById('robotStateDot');
                if (dot) dot.className = 'health-dot offline';
              }
              setHealth('targetDistDot', 'targetDistValue', Boolean(r.target_detected), r.target_detected ? `${r.target_distance_cm} cm` : 'Tidak Ada');
              const sL = Math.round(r.sonar_left_cm || 0);
              const sC = Math.round(r.sonar_center_cm || 0);
              const sR = Math.round(r.sonar_right_cm || 0);
              setHealth('sonarDot', 'sonarValue', true, `[${sL}, ${sC}, ${sR}] cm`);

              const toggleBtn = document.getElementById('toggleFollowingBtn');
              if (toggleBtn) {
                if (r.following_enabled) {
                  toggleBtn.innerText = 'Ikuti Pengguna: AKTIF';
                  toggleBtn.style.background = 'var(--green)';
                } else {
                  toggleBtn.innerText = 'Ikuti Pengguna: NONAKTIF';
                  toggleBtn.style.background = 'var(--muted)';
                }
              }
              const resetBtn = document.getElementById('resetEstopBtn');
              if (resetBtn) {
                resetBtn.disabled = !isEstop;
                resetBtn.style.opacity = isEstop ? '1.0' : '0.4';
              }
            }
            serverPushConfigured = Boolean(data.notifications.configured);

            // Update voice assistant status & toggle button
            if (data.voice) {
              renderVoiceControls(data.voice.enabled);
            }

            // Update voice assistant logs
            if (data.voice && Array.isArray(data.voice.recent_history)) {
              const box = document.getElementById('voiceHistoryBox');
              const history = data.voice.recent_history;
              if (box && history.length > 0) {
                const historyKey = JSON.stringify(history.map(m => (m.timestamp || '') + m.text + (m.response || '')));
                if (box.getAttribute('data-history-key') !== historyKey) {
                  box.setAttribute('data-history-key', historyKey);
                  box.replaceChildren();
                  history.forEach(h => {
                    if (h.speaker === 'robot' && !h.response) {
                      const rMsg = document.createElement('div');
                      rMsg.style.cssText = 'padding: 6px 10px; border-radius: 8px; background: #e7f5ef; color: var(--green-dark); font-size: 13px; align-self: flex-end; max-width: 85%;';
                      rMsg.innerHTML = '<span style="font-size:10px; opacity:0.6; display:block;">[' + (h.timestamp || '') + ']</span><strong>Hafibot:</strong> ' + h.text + (h.action && h.action !== 'none' ? ' <span style="font-size: 11px; background: var(--orange); color: white; padding: 1px 5px; border-radius: 4px;">' + h.action + '</span>' : '');
                      box.appendChild(rMsg);
                    } else {
                      const uMsg = document.createElement('div');
                      uMsg.style.cssText = 'padding: 6px 10px; border-radius: 8px; background: #e8f4fc; font-size: 13px; align-self: flex-start; max-width: 85%;';
                      uMsg.innerHTML = '<span style="font-size:10px; opacity:0.6; display:block;">[' + (h.timestamp || '') + ']</span><strong>Pengguna:</strong> ' + h.text + (h.intent ? ' <span style="font-size: 11px; opacity: 0.7;">(' + h.intent + ')</span>' : '');
                      box.appendChild(uMsg);

                      if (h.response) {
                        const rMsg = document.createElement('div');
                        rMsg.style.cssText = 'padding: 6px 10px; border-radius: 8px; background: #e7f5ef; color: var(--green-dark); font-size: 13px; align-self: flex-end; max-width: 85%; margin-top: 2px;';
                        rMsg.innerHTML = '<strong>Hafibot:</strong> ' + h.response + (h.action && h.action !== 'none' ? ' <span style="font-size: 11px; background: var(--green); color: white; padding: 1px 5px; border-radius: 4px;">' + h.action + '</span>' : '');
                        box.appendChild(rMsg);
                      }
                    }
                  });
                  box.scrollTop = box.scrollHeight;
                }
              }
            }

            // Update shoulder calibration live telemetry
            if (data.calibration) {
              const c = data.calibration;
              const badge = document.getElementById('calibStatusBadge');
              const shPxEl = document.getElementById('liveShoulderPx');
              const shStatusEl = document.getElementById('liveShoulderStatus');
              const distEl = document.getElementById('liveTargetDist');
              const focalEl = document.getElementById('liveFocalConst');
              const timeEl = document.getElementById('lastCalibTime');

              const livePx = c.shoulder_px > 0 ? c.shoulder_px : (c.last_valid_shoulder_px > 0 ? c.last_valid_shoulder_px : 0);
              if (shPxEl) shPxEl.textContent = livePx > 0 ? livePx.toFixed(1) + ' px' : '0.0 px';

              if (shStatusEl) {
                if (c.target_detected) {
                  shStatusEl.innerHTML = '<span style="color:var(--green); font-weight:700;">&#10003; Terdeteksi (Bahu Terlihat)</span>';
                } else if (livePx > 5) {
                  shStatusEl.innerHTML = '<span style="color:var(--orange); font-weight:700;">&#8987; Mengonfirmasi target...</span>';
                } else {
                  shStatusEl.innerHTML = '<span style="color:var(--muted);">&#10007; Target tidak terlihat</span>';
                }
              }

              if (distEl) distEl.textContent = c.target_distance_cm + ' cm';
              if (focalEl) focalEl.textContent = Number(c.focal_constant).toFixed(1);

              if (timeEl) {
                timeEl.textContent = c.calibrated_at ? 'Terkalibrasi: ' + formatEventTime(c.calibrated_at) : 'Default bawaan (22000)';
              }

              if (badge) {
                const isDefault = Math.abs(c.focal_constant - c.default_constant) < 1;
                badge.textContent = isDefault ? 'Default: 22000' : 'Terkalibrasi: ' + Math.round(c.focal_constant);
                badge.style.background = isDefault ? '#f0f2f5' : 'var(--green-soft)';
                badge.style.color = isDefault ? 'var(--muted)' : 'var(--green-dark)';
              }
            }

            updateNotifButton();
            renderIncidents(data.recent_incidents);
          }

          async function pollStatus() {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 3500);
            try {
              const response = await fetch('/status', {
                cache: 'no-store', signal: controller.signal,
                headers: { 'Accept': 'application/json' }
              });
              if (response.status === 401 || response.redirected) {
                window.location.assign('/login');
                return;
              }
              if (!response.ok) throw new Error(`Status HTTP ${response.status}`);
              renderStatus(await response.json());
            } catch (error) {
              renderDisconnected('Server tidak dapat dihubungi. Hafibot mencoba menyambungkan kembali.');
            } finally {
              clearTimeout(timeout);
              window.setTimeout(pollStatus, 1000);
            }
          }

          videoFeed.addEventListener('load', () => { videoErrored = false; });
          videoFeed.addEventListener('error', () => {
            videoErrored = true;
            renderDisconnected('Siaran kamera berhenti. Mencoba menyambungkan kembali.');
            window.setTimeout(() => {
              videoFeed.src = `/video_feed?reconnect=${Date.now()}`;
            }, 2500);
          });

          document.getElementById('ackBtn').addEventListener('click', async () => {
            if (!currentIncidentId) return;
            const button = document.getElementById('ackBtn');
            button.disabled = true;
            try {
              const response = await fetch(`/api/incidents/${encodeURIComponent(currentIncidentId)}/acknowledge`, {
                method: 'POST', headers: { 'X-CSRF-Token': csrfToken, 'Accept': 'application/json' }
              });
              if (!response.ok) throw new Error(`Acknowledge HTTP ${response.status}`);
              showToast('Kejadian ditandai sudah diperiksa.');
            } catch (error) {
              showToast('Gagal memperbarui kejadian. Coba lagi.');
            } finally {
              button.disabled = false;
            }
          });

          // Robot control event listeners
          const toggleBtn = document.getElementById('toggleFollowingBtn');
          if (toggleBtn) {
            toggleBtn.addEventListener('click', async () => {
              try {
                const res = await fetch('/api/robot/following/toggle', {
                  method: 'POST', headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' }
                });
                const d = await res.json();
                showToast(d.following_enabled ? 'Following diaktifkan' : 'Following dinonaktifkan');
              } catch(e) { showToast('Gagal mengubah mode following'); }
            });
          }
          const resetEstopBtn = document.getElementById('resetEstopBtn');
          if (resetEstopBtn) {
            resetEstopBtn.addEventListener('click', async () => {
              try {
                const res = await fetch('/api/robot/following/reset_estop', {
                  method: 'POST', headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' }
                });
                const d = await res.json();
                showToast(d.message || 'ESTOP di-reset');
              } catch(e) { showToast('Gagal reset ESTOP'); }
            });
          }
          const manualEstopBtn = document.getElementById('manualEstopBtn');
          if (manualEstopBtn) {
            manualEstopBtn.addEventListener('click', async () => {
              try {
                const res = await fetch('/api/robot/following/estop', {
                  method: 'POST', headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' }
                });
                showToast('&#9940; REM DARURAT (E-STOP) DIAKTIFKAN!');
              } catch(e) { showToast('Gagal mengaktifkan E-STOP'); }
            });
          }

          // Mode Switcher (Auto vs Manual)
          let currentControlMode = 'auto';
          const tabAuto = document.getElementById('tabAutoMode');
          const tabManual = document.getElementById('tabManualMode');
          const panelAuto = document.getElementById('autoControlsPanel');
          const panelManual = document.getElementById('manualControlsPanel');

          async function setMode(mode) {
            currentControlMode = mode;
            if (mode === 'auto') {
              tabAuto.style.background = '#fff';
              tabAuto.style.color = 'var(--green)';
              tabAuto.style.boxShadow = '0 1px 3px rgba(0,0,0,0.08)';
              tabManual.style.background = 'transparent';
              tabManual.style.color = 'var(--muted)';
              tabManual.style.boxShadow = 'none';
              panelAuto.style.display = 'grid';
              panelManual.style.display = 'none';
            } else {
              tabManual.style.background = '#fff';
              tabManual.style.color = 'var(--orange)';
              tabManual.style.boxShadow = '0 1px 3px rgba(0,0,0,0.08)';
              tabAuto.style.background = 'transparent';
              tabAuto.style.color = 'var(--muted)';
              tabAuto.style.boxShadow = 'none';
              panelAuto.style.display = 'none';
              panelManual.style.display = 'block';
            }
            try {
              await fetch('/api/robot/control_mode', {
                method: 'POST',
                headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
              });
              showToast(mode === 'auto' ? 'Mode Otomatis (Following) Aktif' : 'Mode Manual Takeover Aktif');
            } catch(e) {}
          }

          if (tabAuto) tabAuto.addEventListener('click', () => setMode('auto'));
          if (tabManual) tabManual.addEventListener('click', () => setMode('manual'));

          // Manual Driving Helper
          let driveInterval = null;
          async function sendDrive(direction) {
            try {
              await fetch('/api/robot/manual_drive', {
                method: 'POST',
                headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' },
                body: JSON.stringify({ dir: direction })
              });
            } catch(e) {}
          }

          function setupHoldButton(btnId, direction) {
            const btn = document.getElementById(btnId);
            if (!btn) return;
            const start = (e) => {
              e.preventDefault();
              sendDrive(direction);
              if (driveInterval) clearInterval(driveInterval);
              driveInterval = setInterval(() => sendDrive(direction), 250);
            };
            const stop = (e) => {
              if (driveInterval) clearInterval(driveInterval);
              driveInterval = null;
              sendDrive('stop');
            };
            btn.addEventListener('mousedown', start);
            btn.addEventListener('mouseup', stop);
            btn.addEventListener('mouseleave', stop);
            btn.addEventListener('touchstart', start);
            btn.addEventListener('touchend', stop);
          }

          setupHoldButton('dpadForward', 'forward');
          setupHoldButton('dpadLeft', 'left');
          setupHoldButton('dpadRight', 'right');
          setupHoldButton('dpadPivotLeft', 'pivot_left');
          setupHoldButton('dpadPivotRight', 'pivot_right');

          const dpadStop = document.getElementById('dpadStop');
          if (dpadStop) dpadStop.addEventListener('click', () => sendDrive('stop'));

          // Toggle Voice Assistant Event Listener
          const toggleVoiceBtn = document.getElementById('toggleVoiceBtn');
          if (toggleVoiceBtn) {
            toggleVoiceBtn.addEventListener('click', async () => {
              if (voiceToggleBusy) return;
              const requestedState = !voiceEnabled;
              voiceToggleBusy = true;
              renderVoiceControls(voiceEnabled);
              try {
                const res = await fetch('/api/voice/toggle', {
                  method: 'POST',
                  headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' },
                  body: JSON.stringify({ enable: requestedState })
                });
                const d = await res.json();
                if (!res.ok || !d.success) throw new Error(d.error || `Voice toggle HTTP ${res.status}`);
                voiceEnabled = Boolean(d.voice_enabled);
                showToast(voiceEnabled ? 'Asisten suara diaktifkan' : 'Asisten suara dinonaktifkan');
              } catch(e) {
                showToast('Gagal mengubah status asisten suara');
              } finally {
                voiceToggleBusy = false;
                renderVoiceControls(voiceEnabled);
              }
            });
          }

          // Voice Assistant Chat Helper Functions
          async function sendVoicePrompt(text) {
            if (!text || !text.trim()) return;
            if (!voiceEnabled) {
              showToast('Aktifkan asisten suara terlebih dahulu.');
              return;
            }
            const box = document.getElementById('voiceHistoryBox');
            if (box) {
              const uMsg = document.createElement('div');
              uMsg.style.cssText = 'padding: 8px 12px; border-radius: 8px; background: #e8f4fc; font-size: 13px; align-self: flex-start; max-width: 85%;';
              uMsg.innerHTML = '<strong>Pengguna:</strong> ' + text;
              box.appendChild(uMsg);
              box.scrollTop = box.scrollHeight;
            }
            try {
              const res = await fetch('/api/voice/text_query', {
                method: 'POST',
                headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
              });
              const d = await res.json();
              if (res.status === 409 && d.error === 'voice_disabled') {
                renderVoiceControls(false);
                showToast('Asisten suara sedang dinonaktifkan.');
                return;
              }
              if (!res.ok) throw new Error(d.error || `Voice query HTTP ${res.status}`);
              if (d.success && box) {
                const rMsg = document.createElement('div');
                rMsg.style.cssText = 'padding: 8px 12px; border-radius: 8px; background: #e7f5ef; color: var(--green-dark); font-size: 13px; align-self: flex-end; max-width: 85%; margin-top: 4px;';
                rMsg.innerHTML = '<strong>Hafibot:</strong> ' + d.response_text + (d.action && d.action !== 'none' ? ' <span style="font-size: 11px; background: var(--green); color: white; padding: 2px 6px; border-radius: 4px;">' + d.action + '</span>' : '');
                box.appendChild(rMsg);
                box.scrollTop = box.scrollHeight;

                if (d.audio_base64) {
                  const player = document.getElementById('voiceAudioPlayer');
                  if (player) {
                    player.src = 'data:audio/wav;base64,' + d.audio_base64;
                    player.play().catch(() => {});
                  }
                }
              }
            } catch(e) {
              showToast('Gagal memproses pesan suara');
            }
          }

          function handleVoiceFormSubmit(event) {
            event.preventDefault();
            const inp = document.getElementById('voicePromptInput');
            if (!inp || !inp.value.trim()) return;
            sendVoicePrompt(inp.value.trim());
            inp.value = '';
          }

          // Keyboard listeners for W/A/S/D/Q/E
          const keyMap = {
            'KeyW': 'forward', 'ArrowUp': 'forward',
            'KeyA': 'left', 'ArrowLeft': 'left',
            'KeyD': 'right', 'ArrowRight': 'right',
            'KeyQ': 'pivot_left',
            'KeyE': 'pivot_right',
            'KeyS': 'stop', 'Space': 'stop'
          };
          const pressedKeys = new Set();
          window.addEventListener('keydown', (e) => {
            if (currentControlMode !== 'manual') return;
            if (['input', 'textarea'].includes(document.activeElement.tagName.toLowerCase())) return;
            const dir = keyMap[e.code];
            if (dir && !pressedKeys.has(e.code)) {
              pressedKeys.add(e.code);
              sendDrive(dir);
              if (driveInterval) clearInterval(driveInterval);
              driveInterval = setInterval(() => sendDrive(dir), 250);
            }
          });
          window.addEventListener('keyup', (e) => {
            if (currentControlMode !== 'manual') return;
            if (pressedKeys.has(e.code)) {
              pressedKeys.delete(e.code);
              if (driveInterval) clearInterval(driveInterval);
              driveInterval = null;
              sendDrive('stop');
            }
          });

          // Reset & ESTOP handlers for manual panel
          const resetBtnM = document.getElementById('resetEstopBtnManual');
          if (resetBtnM) resetBtnM.addEventListener('click', () => document.getElementById('resetEstopBtn').click());
          const manualEstopM = document.getElementById('manualEstopBtnManual');
          if (manualEstopM) manualEstopM.addEventListener('click', () => document.getElementById('manualEstopBtn').click());

          // ===== WEB PUSH NOTIFICATION =====
          let swRegistration = null;
          let isSubscribed = false;
          let subscriptionSynced = false;
          let serverPushConfigured = null;

          function urlBase64ToUint8Array(base64String) {
            const padding = '='.repeat((4 - base64String.length % 4) % 4);
            const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
            const raw = atob(base64);
            const arr = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; ++i) arr[i] = raw.charCodeAt(i);
            return arr;
          }

          async function postJSON(url, value) {
            const response = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
              body: JSON.stringify(value),
            });
            if (response.status === 401) window.location.assign('/login');
            if (!response.ok) throw new Error(`${url} HTTP ${response.status}`);
            return response.json();
          }

          function updateNotifButton() {
            const btn = document.getElementById('notifBtn');
            const cardBtn = document.getElementById('notificationCardBtn');
            const testBtn = document.getElementById('testNotifBtn');
            const txt = document.getElementById('notifBtnText');
            const pushApisSupported = 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;
            const secureContext = window.isSecureContext;
            let healthText = 'Belum aktif';
            btn.classList.remove('unsupported', 'subscribed');
            btn.disabled = false;
            cardBtn.disabled = false;

            if (!secureContext) {
              txt.textContent = 'HTTPS diperlukan';
              cardBtn.textContent = 'Buka melalui HTTPS';
              healthText = 'HTTPS diperlukan';
              btn.classList.add('unsupported');
              btn.disabled = true;
              cardBtn.disabled = true;
            } else if (!pushApisSupported) {
              txt.textContent = 'Tidak didukung';
              cardBtn.textContent = 'Browser tidak mendukung';
              healthText = 'Tidak didukung';
              btn.classList.add('unsupported');
              btn.disabled = true;
              cardBtn.disabled = true;
            } else if (serverPushConfigured === false) {
              txt.textContent = 'Belum dikonfigurasi';
              cardBtn.textContent = 'Server belum siap';
              healthText = 'Server belum siap';
              btn.classList.add('unsupported');
              btn.disabled = true;
              cardBtn.disabled = true;
            } else if (Notification.permission === 'denied') {
              txt.textContent = 'Diblokir browser';
              cardBtn.textContent = 'Izin diblokir';
              healthText = 'Diblokir';
              btn.classList.add('unsupported');
              btn.disabled = true;
              cardBtn.disabled = true;
            } else if (isSubscribed && subscriptionSynced) {
              txt.textContent = 'Notifikasi aktif';
              cardBtn.textContent = 'Nonaktifkan';
              healthText = 'Aktif dan tersinkron';
              btn.classList.add('subscribed');
            } else if (isSubscribed) {
              txt.textContent = 'Perlu sinkronisasi';
              cardBtn.textContent = 'Sinkronkan ulang';
              healthText = 'Perlu sinkronisasi';
            } else {
              txt.textContent = 'Aktifkan notifikasi';
              cardBtn.textContent = 'Aktifkan notifikasi';
            }

            const healthy = isSubscribed && subscriptionSynced;
            setHealth('notificationHealthDot', 'notificationHealthValue', healthy, healthText);
            testBtn.disabled = !healthy;
          }

          async function registerSubscription(sub) {
            await postJSON('/api/push/subscribe', sub.toJSON());
            subscriptionSynced = true;
          }

          async function initPush() {
            updateNotifButton();
            if (!window.isSecureContext || !('serviceWorker' in navigator) || !('PushManager' in window)) return;
            try {
              swRegistration = await navigator.serviceWorker.register('/sw.js');
              const sub = await swRegistration.pushManager.getSubscription();
              isSubscribed = sub !== null;
              if (sub) await registerSubscription(sub);
            } catch (error) {
              subscriptionSynced = false;
              console.error('Inisialisasi push gagal:', error);
            }
            updateNotifButton();
          }

          async function togglePushNotification() {
            if (!swRegistration) {
              showToast('Layanan notifikasi belum siap. Muat ulang halaman.');
              return;
            }
            try {
              const existing = await swRegistration.pushManager.getSubscription();
              if (existing && isSubscribed && subscriptionSynced) {
                await postJSON('/api/push/unsubscribe', existing.toJSON());
                await existing.unsubscribe();
                isSubscribed = false;
                subscriptionSynced = false;
                showToast('Notifikasi dinonaktifkan.');
              } else {
                const permission = await Notification.requestPermission();
                if (permission !== 'granted') throw new Error('Izin notifikasi tidak diberikan');
                const response = await fetch('/api/push/vapid-key', { headers: { 'Accept': 'application/json' } });
                if (!response.ok) throw new Error(`VAPID HTTP ${response.status}`);
                const { vapid_public_key } = await response.json();
                if (!vapid_public_key) throw new Error('Server belum memiliki VAPID key');
                const sub = existing || await swRegistration.pushManager.subscribe({
                  userVisibleOnly: true,
                  applicationServerKey: urlBase64ToUint8Array(vapid_public_key),
                });
                isSubscribed = true;
                await registerSubscription(sub);
                showToast('Notifikasi berhasil diaktifkan. Gunakan tombol Uji untuk verifikasi.');
              }
            } catch (error) {
              subscriptionSynced = false;
              console.error('Perubahan notifikasi gagal:', error);
              showToast('Gagal mengubah notifikasi. Periksa HTTPS dan izin browser.');
            }
            updateNotifButton();
          }

          async function testPushNotification() {
            const sub = swRegistration && await swRegistration.pushManager.getSubscription();
            if (!sub) return;
            const button = document.getElementById('testNotifBtn');
            button.disabled = true;
            try {
              await postJSON('/api/push/test', sub.toJSON());
              showToast('Notifikasi uji berhasil dikirim.');
            } catch (error) {
              showToast('Notifikasi uji gagal dikirim.');
            } finally {
              button.disabled = false;
            }
          }

          document.getElementById('notifBtn').addEventListener('click', togglePushNotification);
          document.getElementById('notificationCardBtn').addEventListener('click', togglePushNotification);
          document.getElementById('testNotifBtn').addEventListener('click', testPushNotification);

          // Shoulder Calibration Helpers & Event Handlers
          function setPresetDist(cm) {
            const input = document.getElementById('calibDistanceInput');
            if (input) input.value = cm;
          }

          const btnCalib = document.getElementById('btnCalibrateNow');
          if (btnCalib) {
            btnCalib.addEventListener('click', async () => {
              const distInput = document.getElementById('calibDistanceInput');
              const distVal = parseFloat(distInput ? distInput.value : 200);
              if (!distVal || distVal < 50 || distVal > 500) {
                showToast('Masukkan jarak kalibrasi antara 50 - 500 cm');
                return;
              }
              btnCalib.disabled = true;
              btnCalib.innerHTML = '&#8987; Mengalibrasi...';
              try {
                const response = await fetch('/api/robot/calibration/shoulder', {
                  method: 'POST',
                  headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' },
                  body: JSON.stringify({ distance_cm: distVal })
                });
                const json = await response.json();
                if (json.success) {
                  showToast(json.message || 'Kalibrasi bahu berhasil disimpan!');
                } else {
                  showToast(json.message || 'Kalibrasi gagal.');
                }
              } catch (e) {
                showToast('Gagal menghubungi server.');
              } finally {
                btnCalib.disabled = false;
                btnCalib.innerHTML = '&#127919; Kalibrasi Sekarang';
              }
            });
          }

          const btnResetCalib = document.getElementById('btnResetCalib');
          if (btnResetCalib) {
            btnResetCalib.addEventListener('click', async () => {
              if (!confirm('Kembalikan konstanta fokal bahu ke default bawaan (22000.0)?')) return;
              btnResetCalib.disabled = true;
              try {
                const response = await fetch('/api/robot/calibration/shoulder/reset', {
                  method: 'POST',
                  headers: { 'X-CSRF-Token': csrfToken, 'Content-Type': 'application/json' }
                });
                const json = await response.json();
                showToast(json.message || 'Reset kalibrasi berhasil.');
              } catch (e) {
                showToast('Gagal reset kalibrasi.');
              } finally {
                btnResetCalib.disabled = false;
              }
            });
          }
          initPush();
          pollStatus();
        </script>
      </body>
    </html>
    """.replace('__CSRF_TOKEN__', csrf_token)

@app.route('/video_feed')
@login_required
def video_feed():
    response = Response(
        encode_frame(),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/status')
@login_required
def get_status():
    return jsonify(_status_payload())


@app.route('/healthz')
def healthz():
    """Unauthenticated process liveness endpoint for Docker."""
    return jsonify({'status': 'ok'})


@app.route('/readyz')
def readyz():
    """Minimal detector readiness endpoint without exposing private telemetry."""
    payload = _status_payload()
    ready = payload['camera']['connected'] and payload['detector']['connected']
    return jsonify({'status': 'ready' if ready else 'not_ready'}), 200 if ready else 503


@app.route('/api/incidents')
@login_required
def list_incidents():
    return jsonify({'incidents': _recent_incidents(20)})


@app.route('/api/incidents/<incident_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_incident(incident_id):
    incident = _acknowledge_incident(
        incident_id, session.get('username', AUTH_USERNAME)
    )
    if not incident:
        return jsonify({'error': 'incident_not_found'}), 404
    return jsonify({'status': 'ok', 'incident': incident})


@app.route('/api/incidents/<incident_id>/snapshot')
def incident_snapshot(incident_id):
    authorized = bool(session.get('logged_in'))
    if not authorized:
        token = request.args.get('token', '')
        try:
            token_data = _snapshot_serializer().loads(
                token, max_age=SNAPSHOT_TOKEN_MAX_AGE_SEC
            )
            authorized = hmac.compare_digest(
                str(token_data.get('incident_id', '')), incident_id
            )
        except (BadSignature, SignatureExpired, TypeError, ValueError):
            authorized = False
    if not authorized:
        return Response(status=401)

    with _incidents_lock:
        incident = _find_incident_locked(incident_id)
        filename = incident.get('snapshot_filename') if incident else None
    if not filename:
        return Response(status=404)
    snapshot_path = os.path.abspath(os.path.join(SNAPSHOTS_DIR, filename))
    snapshots_root = os.path.abspath(SNAPSHOTS_DIR) + os.sep
    if not snapshot_path.startswith(snapshots_root) or not os.path.isfile(snapshot_path):
        return Response(status=404)
    with open(snapshot_path, 'rb') as handle:
        snapshot_bytes = handle.read()
    return Response(
        snapshot_bytes,
        mimetype='image/jpeg',
        headers={
            'Cache-Control': 'private, no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
        },
    )


# ============================================================
# WEB PUSH API ROUTES
# ============================================================
@app.route('/api/push/vapid-key')
@login_required
def get_vapid_key():
    """Return VAPID public key untuk browser subscription."""
    return jsonify({"vapid_public_key": VAPID_PUBLIC_KEY})


@app.route('/api/push/subscribe', methods=['POST'])
@login_required
def push_subscribe():
    """Terima dan simpan push subscription dari browser."""
    sub = request.get_json(silent=True)
    keys = sub.get('keys', {}) if isinstance(sub, dict) else {}
    if (
        not isinstance(sub, dict)
        or not isinstance(sub.get('endpoint'), str)
        or not sub['endpoint'].startswith('https://')
        or len(sub['endpoint']) > 4096
        or not isinstance(keys.get('p256dh'), str)
        or not isinstance(keys.get('auth'), str)
    ):
        return jsonify({"error": "Invalid subscription"}), 400

    sub_id = _sub_id(sub)

    with _push_lock:
        # Cek duplikat
        existing_ids = [_sub_id(s) for s in _push_subscriptions]
        if sub_id not in existing_ids:
            _push_subscriptions.append(sub)
            _save_subscriptions()
            rospy.loginfo(f"&#128276; Push subscription baru: {sub_id} (total: {len(_push_subscriptions)})")
        else:
            rospy.loginfo(f"&#128276; Push subscription sudah ada: {sub_id}")

    return jsonify({"status": "ok", "id": sub_id})


@app.route('/api/push/unsubscribe', methods=['POST'])
@login_required
def push_unsubscribe():
    """Hapus push subscription."""
    sub = request.get_json(silent=True)
    if not isinstance(sub, dict) or not isinstance(sub.get('endpoint'), str):
        return jsonify({"error": "Invalid data"}), 400

    sub_id = _sub_id(sub)

    with _push_lock:
        before = len(_push_subscriptions)
        _push_subscriptions[:] = [
            s for s in _push_subscriptions
            if _sub_id(s) != sub_id
        ]
        if len(_push_subscriptions) < before:
            _save_subscriptions()
            rospy.loginfo(f"🔕 Push subscription dihapus: {sub_id} (sisa: {len(_push_subscriptions)})")

    return jsonify({"status": "ok"})


@app.route('/api/push/test', methods=['POST'])
@login_required
def push_test():
    """Send a delivery test only to the requesting browser subscription."""
    if not HAS_WEBPUSH or not VAPID_PRIVATE_KEY or not VAPID_PUBLIC_KEY:
        return jsonify({'error': 'push_not_configured'}), 503
    sub = request.get_json(silent=True)
    if not isinstance(sub, dict) or not isinstance(sub.get('endpoint'), str):
        return jsonify({'error': 'invalid_subscription'}), 400
    sub_id = _sub_id(sub)
    with _push_lock:
        registered = any(_sub_id(item) == sub_id for item in _push_subscriptions)
    if not registered:
        return jsonify({'error': 'subscription_not_registered'}), 409
    payload = json.dumps({
        'title': '✅ Notifikasi Hafibot aktif',
        'body': 'Perangkat ini siap menerima peringatan jatuh.',
        'icon': '/img/hafibot_logo.png',
        'badge': '/img/hafibot_logo.png',
        'url': '/',
    })
    try:
        webpush(
            subscription_info=sub,
            data=payload,
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={'sub': VAPID_EMAIL},
            timeout=10,
        )
    except Exception as error:
        rospy.logwarn('Push test gagal untuk {}: {}'.format(sub_id, error))
        return jsonify({'error': 'delivery_failed'}), 502
    return jsonify({'status': 'delivered'})


# ============================================================
# SERVICE WORKER & PWA MANIFEST
# ============================================================
@app.route('/sw.js')
def service_worker():
    """Serve Service Worker untuk push notifications."""
    sw_code = """
// Hafibot Service Worker - Push Notification Handler
const CACHE_NAME = 'hafibot-v2';

// Install: cache assets penting
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll([
                '/img/hafibot_app_icon.svg',
            ]).catch(() => {});
        })
    );
    self.skipWaiting();
});

// Activate: cleanup old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((keys) => Promise.all(
                keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
            ))
            .then(() => self.clients.claim())
    );
});

// Push: tampilkan notifikasi native dengan gambar privacy frame
self.addEventListener('push', (event) => {
    let data = {
        title: '🚨 Hafibot Alert',
        body: 'Kejadian jatuh terdeteksi! Segera periksa.',
        icon: '/img/hafibot_logo.png',
        badge: '/img/hafibot_logo.png',
        image: '',
        url: '/',
    };

    try {
        if (event.data) {
            data = Object.assign(data, event.data.json());
        }
    } catch (e) {}

    const options = {
        body: data.body,
        icon: data.icon,
        badge: data.badge,
        vibrate: [300, 100, 300, 100, 300],
        tag: 'hafibot-fall-alert',
        renotify: true,
        timestamp: Date.now(),
        data: { url: data.url },
    };

    if (data.image) {
        options.image = data.image;
    }

    event.waitUntil(
        self.registration.showNotification(data.title, options).catch((err) => {
            // Fallback tanpa gambar/actions jika browser mobile menolak
            delete options.image;
            delete options.actions;
            return self.registration.showNotification(data.title, options);
        })
    );
});

// Notification Click: buka halaman monitoring
self.addEventListener('notificationclick', (event) => {
    event.notification.close();

    if (event.action === 'dismiss') return;

    const url = event.notification.data?.url || '/';

    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
            // Jika sudah ada tab terbuka, fokuskan
            for (const client of clientList) {
                if (client.url.includes(self.registration.scope) && 'focus' in client) {
                    if ('navigate' in client) {
                        return client.navigate(url).then((navigated) => navigated.focus());
                    }
                    return client.focus();
                }
            }
            // Jika belum, buka tab baru
            return clients.openWindow(url);
        })
    );
});
"""
    return Response(sw_code, mimetype='application/javascript',
                    headers={'Service-Worker-Allowed': '/'})


@app.route('/manifest.json')
def pwa_manifest():
    """Serve PWA manifest agar bisa di-install sebagai app."""
    manifest = {
        "name": "Hafibot Fall Detection Monitor",
        "short_name": "Hafibot",
        "description": "Real-time elderly fall detection monitoring system",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a1a2e",
        "theme_color": "#E87E24",
        "orientation": "any",
        "icons": [
            {
                "src": "/img/hafibot_app_icon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            }
        ]
    }
    return jsonify(manifest)


# ============================================================
# FALL SNAPSHOT ROUTE
# ============================================================
@app.route('/api/fall-snapshot')
@login_required
def fall_snapshot():
    """Backward-compatible authenticated view of the latest fall snapshot."""
    if _fall_snapshot_bytes is None:
        return Response(b'', status=204)

    # Snapshot hanya valid selama 5 menit setelah kejadian
    if time.time() - _fall_snapshot_time > 300:
        return Response(b'', status=204)

    return Response(
        _fall_snapshot_bytes,
        mimetype='image/jpeg',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
        }
    )


# ============================================================
# ROS CALLBACKS
# ============================================================
def image_callback(msg):
    global current_jpeg_bytes, _current_frame_id, _current_frame_source_stamp
    global _current_frame_received_monotonic
    try:
        frame_bytes = bytes(msg.data)
        source_stamp = msg.header.stamp.to_sec()
        with _frame_condition:
            current_jpeg_bytes = frame_bytes
            _current_frame_id += 1
            _current_frame_source_stamp = source_stamp
            _current_frame_received_monotonic = time.monotonic()
            _frame_condition.notify_all()
    except Exception as e:
        rospy.logwarn_throttle(2, f"Gagal membaca compressed image: {e}")


def processing_stats_callback(msg):
    global _latest_processing_stats, _last_processing_stats_monotonic
    try:
        payload = json.loads(msg.data)
        if isinstance(payload, dict):
            _latest_processing_stats = payload
            _last_processing_stats_monotonic = time.monotonic()
    except (TypeError, ValueError) as error:
        rospy.logwarn_throttle(5, 'Processing stats JSON tidak valid: {}'.format(error))


def alert_callback(msg):
    global latest_status, _last_alert_received_monotonic
    global _previous_alert_status, _active_signal_incident_id
    global _fall_clear_started_monotonic
    current_status = bool(msg.data)
    now_mono = time.monotonic()
    _last_alert_received_monotonic = now_mono
    incident = None
    cleared_incident_id = None

    # Treat brief normal predictions as detector jitter, not a new fall episode.
    # The first True frame still alerts immediately; rearming requires a
    # continuous normal period so one physical fall creates only one incident.
    with _alert_state_lock:
        if current_status:
            _fall_clear_started_monotonic = None
            if not _previous_alert_status:
                incident = _create_incident()
                _active_signal_incident_id = incident['id']
                _previous_alert_status = True
        elif _previous_alert_status:
            if _fall_clear_started_monotonic is None:
                _fall_clear_started_monotonic = now_mono
            elif now_mono - _fall_clear_started_monotonic >= FALL_CLEAR_DEBOUNCE_SEC:
                cleared_incident_id = _active_signal_incident_id
                _active_signal_incident_id = None
                _previous_alert_status = False
                _fall_clear_started_monotonic = None

        latest_status = _previous_alert_status

    if cleared_incident_id:
        _mark_incident_detector_cleared(cleared_incident_id)

    if incident:
        rospy.logwarn('Monitoring incident created: {}'.format(incident['id']))
        if HAS_WEBPUSH and VAPID_PRIVATE_KEY:
            push_thread = threading.Thread(
                target=_send_push_notifications, args=(incident,), daemon=True
            )
            push_thread.start()



# ============================================================
# AI & ROBOT GATEWAY INTEGRATION
# ============================================================
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, FALL_CONFIDENCE_THRESHOLD, MODEL_DIR, YOLO_POSE_MODEL_PATH
from src.inference import FallDetector
from src.runtime_loader import create_server_fall_detector
from src.yolo_detector import YOLOPoseDetector
from src.human_following import ShoulderTargetFilter
from src.hafibot_protocol import encode_command, FOLLOW, STOP, ESTOP, RESET_ESTOP, FallEStopLatch
from src.chatbot_service import ChatbotService

print('=' * 60)
print('  INITIALIZING HAFIBOT GPU AI PIPELINE ON LAPTOP')
print('=' * 60)

pose_model_file = str(YOLO_POSE_MODEL_PATH)
pose_detector = YOLOPoseDetector(model_path=pose_model_file, device='cuda' if torch.cuda.is_available() else 'cpu')
fall_detector = create_server_fall_detector(os.environ.get('HAFIBOT_FALL_RUNTIME_CALIBRATION'))
target_filter = ShoulderTargetFilter(coefficients=[DEFAULT_FOCAL_CONSTANT], confirmation_frames=2, min_visibility=0.35)
_load_calibration()
fall_safety_latch = FallEStopLatch()
following_enabled = True
robot_control_mode = 'auto'  # 'auto' or 'manual'
manual_direction = 'stop'
manual_direction_expires = 0.0
reset_estop_pending = False
robot_latest_telemetry = {'state': 'IDLE', 'us_left': 0.0, 'us_center': 0.0, 'us_right': 0.0}

# Initialize Chatbot & Voice Assistant Service
chatbot_service = ChatbotService.get_instance(PROJECT_ROOT)

def _cb_start_following():
    global following_enabled
    following_enabled = True
    logger.info("[VOICE ACTION] Following Enabled")

def _cb_stop_following():
    global following_enabled
    following_enabled = False
    logger.info("[VOICE ACTION] Following Disabled")

def _cb_reset_estop():
    global reset_estop_pending
    if not fall_safety_latch.fall_active:
        fall_safety_latch.request_reset()
        reset_estop_pending = True
        logger.info("[VOICE ACTION] ESTOP Reset Triggered via Voice")

def _cb_trigger_alarm():
    logger.warning("[VOICE ACTION] Emergency Alarm Escalated via Voice")

chatbot_service.on_start_following = _cb_start_following
chatbot_service.on_stop_following = _cb_stop_following
chatbot_service.on_reset_estop = _cb_reset_estop
chatbot_service.on_trigger_alarm = _cb_trigger_alarm

# Dynamic FPS & Latency Trackers
_last_frame_recv_time = time.monotonic()
_stream_fps_history = []
_proc_latency_history = []
_frame_inference_lock = threading.Lock()

class FakeCompressedImageMsg:
    def __init__(self, data_bytes, stamp_sec, seq=1):
        self.data = data_bytes
        self.format = 'jpeg'
        self.header = type('Header', (), {'stamp': type('Stamp', (), {'to_sec': lambda *a, **k: stamp_sec})(), 'seq': seq})()

class FakeBoolMsg:
    def __init__(self, data): self.data = bool(data)

class FakeStringMsg:
    def __init__(self, data): self.data = str(data)

@app.route('/api/robot/frame', methods=['POST'])
def api_robot_frame():
    # The temporal fall detector and CUDA pose model must process one frame at
    # a time. The Jetson still pipelines two HTTP requests to hide Tailscale RTT.
    with _frame_inference_lock:
        return _process_robot_frame()

def _process_robot_frame():
    # Receives frame + telemetry from Jetson Nano Edge Agent and returns motor commands.
    global following_enabled, robot_latest_telemetry, latest_target_distance_cm, latest_target_detected
    global robot_control_mode, manual_direction, manual_direction_expires
    
    file = request.files.get('frame')
    if file:
        telem_raw = request.form.get('telemetry', '{}')
        file_bytes = file.read()
    else:
        telem_raw = request.headers.get('X-Telemetry', '{}')
        file_bytes = request.get_data()

    try:
        telem = json.loads(telem_raw)
        robot_latest_telemetry = telem
    except Exception:
        telem = robot_latest_telemetry
        
    if not file_bytes:
        return jsonify({'success': False, 'error': 'No frame provided'}), 400
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'success': False, 'error': 'Decode failed'}), 400
        
    global _last_frame_recv_time, _stream_fps_history, _proc_latency_history
    t_ai_start = time.perf_counter()
    t_now = time.monotonic()
    dt_frame = t_now - _last_frame_recv_time
    _last_frame_recv_time = t_now
    if dt_frame > 0:
        _stream_fps_history.append(1.0 / dt_frame)
        if len(_stream_fps_history) > 30:
            _stream_fps_history.pop(0)
    real_stream_fps = float(sum(_stream_fps_history) / max(1, len(_stream_fps_history))) if _stream_fps_history else 15.0
    
    # Sonar telemetry for cross-checking
    us_l = float(telem.get('us_left', 0.0))
    us_c = float(telem.get('us_center', 0.0))
    us_r = float(telem.get('us_right', 0.0))
    valid_sonars = [v for v in (us_l, us_c, us_r) if v > 0.0]
    min_sonar_cm = min(valid_sonars) if valid_sonars else 999.0

    # 1. Pose estimation on GPU
    landmarks = pose_detector.detect(frame)

    # 2. Fall detection (1D CNN-LSTM + Kinematic Gating)
    fall_res = fall_detector.process_frame(landmarks, frame_size=(frame.shape[1], frame.shape[0]))
    is_fall = bool(fall_res.get('is_fall', False))

    # 3. Following target calculation





    # probe
    if landmarks and len(landmarks) >= 13:
        target = target_filter.update(landmarks)
    else:
        target_filter.reset()
        target = {'center_x': 0.5, 'distance_cm': 0, 'detected': False, 'reason': 'no_pose'}
        
    target_dist = int(max(0, min(500, target['distance_cm'])))
    target_cx = float(max(0.0, min(1.0, target['center_x'])))
    target_det = bool(target['detected'])
    latest_target_distance_cm = target_dist
    latest_target_detected = target_det

    # Cross-check: Target / Sonar / Occlusion consistency
    # 1. Target not confirmed or distance 0 cm while following
    # 2. Sonar indicates very close object (< 25 cm) but target distance 0 cm or target inconsistent
    # 3. Target discontinuity resets fall detector buffer so static objects suddenly appearing don't latch fall
    target_reason = str(target.get('reason', ''))
    if target_reason == 'target_discontinuity':
        fall_detector.reset_buffer()

    posture_data = fall_res.get('posture') or {}
    # Torso-centric validity: as long as the torso is visible and reliable,
    # it is confirmed as a valid human pose. Do NOT require human-following target confirmation
    # because a fallen/lying person naturally loses standing target lock (distance=0 cm).
    # An alert is only downgraded to doubtful if torso anchors are missing or
    # an obstacle is right against the bumper (< 18 cm) with no valid torso detected.
    has_reliable_torso = bool(
        fall_res.get('pose_valid', True)
        and posture_data.get('is_torso_reliable', True)
    )
    extreme_sonar_blindspot = bool(min_sonar_cm < 18.0 and not has_reliable_torso)
    is_plausible_pose = bool(has_reliable_torso and not extreme_sonar_blindspot)

    # Action branching: Confirmed Plausible vs Doubtful (Uncertain)
    doubtful_fall = False
    if is_fall and not is_plausible_pose:
        # Downgrade to uncertain: log snapshot, ask via voice, do not latch ESTOP or push
        doubtful_fall = True
        is_fall = False

    global latest_shoulder_px, latest_effective_px, last_valid_shoulder_px
    spx = float(target.get('shoulder_px', 0.0))
    eff_px = float(target.get('effective_px', spx))
    latest_shoulder_px = round(spx, 1)
    latest_effective_px = round(eff_px, 1)
    if spx > 5.0:
        last_valid_shoulder_px = latest_shoulder_px
    
    # 4. Latched ESTOP & Motor Command
    global reset_estop_pending
    was_latched = fall_safety_latch.latched
    fall_safety_latch.observe(is_fall, pause_on_fall=True)
    if fall_safety_latch.latched:
        cmd_mode = ESTOP
        robot_state = 'FALL_ESTOP_LATCHED'
        if not was_latched:
            chatbot_service.trigger_fall_alert_voice()
    elif doubtful_fall:
        # Uncertain alert mode: ask user gently via voice without locking motors in ESTOP
        cmd_mode = STOP
        robot_state = 'UNCERTAIN_FALL_INQUIRY'
        chatbot_service.trigger_fall_inquiry_voice()
    elif reset_estop_pending:
        cmd_mode = RESET_ESTOP
        target_dist, target_cx, target_det = 0, 0.500, False
        robot_state = 'RESETTING_ESTOP'
        if not fall_safety_latch.fall_active:
            reset_estop_pending = False
    elif robot_control_mode == 'manual':
        # Manual takeover driving mode
        now_mono = time.monotonic()
        if now_mono > manual_direction_expires or manual_direction == 'stop':
            cmd_mode = STOP
            target_dist, target_cx, target_det = 0, 0.5, False
            robot_state = 'MANUAL_STOP'
        elif manual_direction == 'forward':
            cmd_mode = FOLLOW
            target_dist, target_cx, target_det = 380, 0.500, True
            robot_state = 'MANUAL_FORWARD'
        elif manual_direction == 'left':
            cmd_mode = FOLLOW
            target_dist, target_cx, target_det = 300, 0.200, True
            robot_state = 'MANUAL_LEFT'
        elif manual_direction == 'right':
            cmd_mode = FOLLOW
            target_dist, target_cx, target_det = 300, 0.800, True
            robot_state = 'MANUAL_RIGHT'
        elif manual_direction == 'pivot_left':
            cmd_mode = FOLLOW
            target_dist, target_cx, target_det = 250, 0.050, True
            robot_state = 'MANUAL_PIVOT_LEFT'
        elif manual_direction == 'pivot_right':
            cmd_mode = FOLLOW
            target_dist, target_cx, target_det = 250, 0.950, True
            robot_state = 'MANUAL_PIVOT_RIGHT'
        else:
            cmd_mode = STOP
            target_dist, target_cx, target_det = 0, 0.5, False
            robot_state = 'MANUAL_STOP'
    elif not following_enabled:
        cmd_mode = STOP
        robot_state = 'DISABLED'
    elif not target_det:
        cmd_mode = STOP
        robot_state = 'NO_TARGET'
    else:
        cmd_mode = FOLLOW
        robot_state = 'FOLLOWING'
        
    motor_cmd = encode_command(cmd_mode, target_dist, target_cx, target_det).decode('ascii')
    
    # Measure AI latency and throughput
    t_ai_end = time.perf_counter()
    ai_latency_ms = (t_ai_end - t_ai_start) * 1000.0
    _proc_latency_history.append(ai_latency_ms)
    if len(_proc_latency_history) > 30:
        _proc_latency_history.pop(0)
    mean_ai_latency_ms = float(sum(_proc_latency_history) / max(1, len(_proc_latency_history)))
    real_ai_fps = float(1000.0 / max(0.1, mean_ai_latency_ms))

    # 5. Draw HUD Overlay
    annotated = fall_detector.draw_overlay(frame, fall_res, landmarks, fps=real_stream_fps)
    
    # Draw sonar telemetry overlay at bottom
    h_img, w_img = annotated.shape[:2]
    status_tag = 'LOCKED' if target_det else ('NO POSE' if not landmarks else 'CONFIRMING...')
    telem_str = f'Target: {target_dist}cm ({status_tag}) | Sonar: [{us_l:.0f}, {us_c:.0f}, {us_r:.0f}]cm | State: {robot_state}'
    color_tag = (0, 255, 0) if target_det else ((0, 0, 255) if fall_safety_latch.latched else (0, 255, 255))
    cv2.putText(annotated, telem_str, (12, h_img - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_tag, 1, cv2.LINE_AA)
    
    # 6. Compress and notify Flask stream callbacks
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
    annotated_jpeg = buf.tobytes()
    
    image_callback(FakeCompressedImageMsg(annotated_jpeg, time.time()))
    alert_callback(FakeBoolMsg(is_fall))
    
    stats_payload = {
        'timestamp_unix_sec': time.time(),
        'camera_stream_fps': float(round(real_stream_fps, 1)),
        'processing_fps_rolling': float(round(real_ai_fps, 1)),
        'processing_latency_ms_mean': float(round(mean_ai_latency_ms, 1)),
        'pose_detection_rate_rolling': 1.0 if landmarks else 0.0,
    }
    processing_stats_callback(FakeStringMsg(json.dumps(stats_payload)))
    
    return jsonify({
        'success': True,
        'command': motor_cmd,
        'is_fall': is_fall,
        'fall_prob': fall_res.get('fall_prob', 0.0),
        'prediction': fall_res.get('prediction', 'NORMAL'),
        'confidence': fall_res.get('confidence', 0.0),
        'robot_state': robot_state,
        'target': {'distance': target_dist, 'center_x': target_cx, 'detected': target_det},
        'has_voice_pending': bool(chatbot_service.proactive_audio_queue),
        'voice_enabled': chatbot_service.is_enabled(),
    })

@app.route('/api/robot/following/toggle', methods=['POST'])
def api_toggle_following():
    global following_enabled
    data = request.get_json(silent=True) or {}
    if 'enable' in data:
        following_enabled = bool(data['enable'])
    else:
        following_enabled = not following_enabled
    return jsonify({'success': True, 'following_enabled': following_enabled})

@app.route('/api/robot/following/reset_estop', methods=['POST'])
def api_reset_estop():
    global reset_estop_pending
    if latest_status is True:
        return jsonify({'success': False, 'message': 'Fall signal masih aktif'}), 400
    fall_safety_latch.request_reset()
    reset_estop_pending = True
    return jsonify({'success': True, 'message': 'ESTOP berhasil di-reset', 'estop_latched': fall_safety_latch.latched})

@app.route('/api/robot/following/estop', methods=['POST'])
def api_manual_estop():
    fall_safety_latch.latched = True
    return jsonify({'success': True, 'message': 'E-STOP diaktifkan', 'estop_latched': True})

@app.route('/api/robot/control_mode', methods=['POST'])
def api_set_control_mode():
    global robot_control_mode, manual_direction
    data = request.get_json(silent=True) or {}
    mode = str(data.get('mode', 'auto')).lower()
    if mode in ('auto', 'manual'):
        robot_control_mode = mode
        manual_direction = 'stop'
        return jsonify({'success': True, 'control_mode': robot_control_mode})
    return jsonify({'success': False, 'error': 'Invalid mode'}), 400

@app.route('/api/robot/manual_drive', methods=['POST'])
def api_manual_drive():
    global manual_direction, manual_direction_expires
    data = request.get_json(silent=True) or {}
    direction = str(data.get('dir', 'stop')).lower()
    valid_dirs = ('forward', 'left', 'right', 'pivot_left', 'pivot_right', 'stop')
    if direction in valid_dirs:
        manual_direction = direction
        manual_direction_expires = time.monotonic() + 0.6
        return jsonify({'success': True, 'direction': manual_direction})
    return jsonify({'success': False, 'error': 'Invalid direction'}), 400

@app.route('/api/robot/calibration/shoulder', methods=['GET'])
@login_required
def api_get_calibration():
    return jsonify({
        'success': True,
        'focal_constant': round(float(current_focal_constant), 1),
        'default_focal_constant': DEFAULT_FOCAL_CONSTANT,
        'shoulder_px': round(float(globals().get('latest_shoulder_px', 0.0)), 1),
        'effective_px': round(float(globals().get('latest_effective_px', 0.0)), 1),
        'last_valid_shoulder_px': round(float(globals().get('last_valid_shoulder_px', 0.0)), 1),
        'target_distance_cm': int(globals().get('latest_target_distance_cm', 0)),
        'target_detected': bool(globals().get('latest_target_detected', False)),
        'calibrated_at': last_calibration_info.get('updated_at'),
        'calibrated_distance_cm': last_calibration_info.get('calibrated_distance_cm', 200),
    })

@app.route('/api/robot/calibration/shoulder', methods=['POST'])
@login_required
def api_set_calibration():
    global current_focal_constant, last_calibration_info, target_filter
    data = request.get_json(silent=True) or {}

    # Mode 1: Calibrate from current measured shoulder at known distance
    if 'distance_cm' in data:
        try:
            dist_cm = float(data['distance_cm'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Jarak tidak valid'}), 400
        if dist_cm < 50.0 or dist_cm > 500.0:
            return jsonify({'success': False, 'message': 'Jarak kalibrasi harus antara 50 - 500 cm'}), 400

        measured_px = float(globals().get('latest_effective_px', 0.0))
        if measured_px <= 5.0:
            measured_px = float(globals().get('latest_shoulder_px', 0.0))
        if measured_px <= 5.0:
            measured_px = float(globals().get('last_valid_shoulder_px', 0.0))

        if measured_px <= 5.0:
            return jsonify({
                'success': False,
                'message': 'Bahu/target tidak terdeteksi! Berdirilah terlihat di kamera robot sebelum kalibrasi.',
            }), 400

        new_focal = float(round(dist_cm * measured_px, 1))
        current_focal_constant = new_focal
        if 'target_filter' in globals() and target_filter is not None:
            target_filter.coefficients = (current_focal_constant,)
        last_calibration_info = {
            'focal_constant': current_focal_constant,
            'calibrated_distance_cm': int(round(dist_cm)),
            'measured_shoulder_px': round(measured_px, 1),
            'updated_at': _utc_now_string(),
        }
        _save_calibration()
        logger.info(f'[CALIBRATION] Recalibrated shoulder: dist={dist_cm}cm, px={measured_px}, focal={new_focal}')
        return jsonify({
            'success': True,
            'focal_constant': current_focal_constant,
            'calibrated_distance_cm': int(round(dist_cm)),
            'measured_shoulder_px': round(measured_px, 1),
            'message': f'Kalibrasi berhasil! Konstanta fokal baru: {new_focal:.1f} (di jarak {int(round(dist_cm))} cm, lebar bahu {measured_px:.1f} px)',
        })

    # Mode 2: Direct set focal_constant
    elif 'focal_constant' in data:
        try:
            focal_val = float(data['focal_constant'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'message': 'Nilai konstanta tidak valid'}), 400
        if focal_val < 5000.0 or focal_val > 100000.0:
            return jsonify({'success': False, 'message': 'Konstanta fokal harus antara 5000 - 100000'}), 400

        current_focal_constant = round(focal_val, 1)
        if 'target_filter' in globals() and target_filter is not None:
            target_filter.coefficients = (current_focal_constant,)
        last_calibration_info['focal_constant'] = current_focal_constant
        last_calibration_info['updated_at'] = _utc_now_string()
        _save_calibration()
        logger.info(f'[CALIBRATION] Manually set shoulder focal constant: {current_focal_constant}')
        return jsonify({
            'success': True,
            'focal_constant': current_focal_constant,
            'message': f'Konstanta fokal berhasil diatur ke {current_focal_constant:.1f}',
        })

    return jsonify({'success': False, 'message': 'Parameter tidak lengkap'}), 400

@app.route('/api/robot/calibration/shoulder/reset', methods=['POST'])
@login_required
def api_reset_calibration():
    global current_focal_constant, last_calibration_info, target_filter
    current_focal_constant = DEFAULT_FOCAL_CONSTANT
    if 'target_filter' in globals() and target_filter is not None:
        target_filter.coefficients = (current_focal_constant,)
    last_calibration_info = {
        'focal_constant': DEFAULT_FOCAL_CONSTANT,
        'calibrated_distance_cm': 200,
        'measured_shoulder_px': 110.0,
        'updated_at': _utc_now_string(),
    }
    _save_calibration()
    logger.info('[CALIBRATION] Reset shoulder calibration to default (22000.0)')
    return jsonify({
        'success': True,
        'focal_constant': current_focal_constant,
        'message': 'Konstanta fokal dikembalikan ke default bawaan (22000.0)',
    })

@app.route('/api/voice/interact', methods=['POST'])
def api_voice_interact():
    import base64
    if not chatbot_service.is_enabled():
        return jsonify({
            'success': False,
            'voice_enabled': False,
            'error': 'voice_disabled',
            'response_text': 'Asisten suara sedang dinonaktifkan.',
        }), 409
    file = request.files.get('audio')
    if not file:
        data = request.get_data()
        if data and len(data) > 100:
            wav_bytes = data
        else:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    else:
        wav_bytes = file.read()

    result = chatbot_service.process_voice_interaction(wav_bytes)
    audio_b64 = base64.b64encode(result.get('wav_bytes', b'')).decode('ascii') if result.get('wav_bytes') else ''

    return jsonify({
        'success': result['success'],
        'transcript': result.get('transcript', ''),
        'intent': result.get('intent', ''),
        'sentiment': result.get('sentiment', ''),
        'function': result.get('function', ''),
        'action': result.get('action', ''),
        'response_text': result.get('response_text', ''),
        'audio_base64': audio_b64,
        'latency': result.get('latency', {}),
    })

@app.route('/api/voice/text_query', methods=['POST'])
def api_voice_text_query():
    import base64
    if not chatbot_service.is_enabled():
        return jsonify({
            'success': False,
            'voice_enabled': False,
            'error': 'voice_disabled',
            'response_text': 'Asisten suara sedang dinonaktifkan.',
        }), 409
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'}), 400

    result = chatbot_service.process_text_interaction(text)
    audio_b64 = base64.b64encode(result.get('wav_bytes', b'')).decode('ascii') if result.get('wav_bytes') else ''

    return jsonify({
        'success': result['success'],
        'transcript': result.get('transcript', ''),
        'intent': result.get('intent', ''),
        'sentiment': result.get('sentiment', ''),
        'function': result.get('function', ''),
        'action': result.get('action', ''),
        'response_text': result.get('response_text', ''),
        'audio_base64': audio_b64,
        'latency': result.get('latency', {}),
    })

@app.route('/api/voice/pending_audio', methods=['GET'])
def api_voice_pending_audio():
    import base64
    if not chatbot_service.is_enabled():
        return jsonify({'has_audio': False, 'voice_enabled': False})
    item = chatbot_service.pop_pending_audio()
    if not item:
        return jsonify({'has_audio': False, 'voice_enabled': True})
    audio_b64 = base64.b64encode(item['wav_bytes']).decode('ascii')
    return jsonify({
        'has_audio': True,
        'id': item['id'],
        'text': item['text'],
        'reason': item['reason'],
        'priority': item['priority'],
        'audio_base64': audio_b64,
        'voice_enabled': True,
    })

@app.route('/api/voice/toggle', methods=['POST'])
def api_voice_toggle():
    data = request.get_json(silent=True) or {}
    enable = data['enable'] if 'enable' in data else None
    is_enabled = chatbot_service.toggle(enable)
    return jsonify({'success': True, 'voice_enabled': is_enabled})

@app.route('/api/voice/status', methods=['GET'])
def api_voice_status():
    is_enabled = chatbot_service.is_enabled()
    return jsonify({
        'status': 'ACTIVE' if is_enabled else 'DISABLED',
        'voice_enabled': is_enabled,
        'reminders': chatbot_service.reminder_engine.reminders,
        'history': chatbot_service.get_history(),
        'pending_audio_count': len(chatbot_service.proactive_audio_queue),
    })

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hafibot Laptop AI Server & Web Monitoring Dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    default_threads = max(4, int(_env_float('HAFIBOT_WEB_THREADS', 8.0)))
    parser.add_argument('--threads', type=int, default=default_threads, help='Waitress worker threads (default: HAFIBOT_WEB_THREADS or 8)')
    args = parser.parse_args()

    global IMG_DIR, INCIDENTS_FILE, SNAPSHOTS_DIR, SUBSCRIPTIONS_FILE
    global CALIBRATION_FILE
    
    src_path = str(PROJECT_ROOT)
    IMG_DIR = str(PROJECT_ROOT / 'img')
    monitoring_data_dir = str(PROJECT_ROOT / 'data' / 'monitoring')
    SUBSCRIPTIONS_FILE = str(PROJECT_ROOT / 'data' / 'monitoring' / 'push_subscriptions.json')
    INCIDENTS_FILE = str(PROJECT_ROOT / 'data' / 'monitoring' / 'incidents.json')
    SNAPSHOTS_DIR = str(PROJECT_ROOT / 'data' / 'monitoring' / 'snapshots')
    CALIBRATION_FILE = str(PROJECT_ROOT / 'data' / 'monitoring' / 'shoulder_calibration.json')
    
    _ensure_private_directory(SNAPSHOTS_DIR)
    _init_vapid()
    _load_subscriptions()
    _load_incidents()
    _load_calibration()
    
    print('=' * 60)
    print('  HAFIBOT LAPTOP AI SERVER ACTIVE (NVIDIA GPU ACCELERATED)')
    print(f'  URL         : http://{args.host}:{args.port}')
    print(f'  Local Login : http://localhost:{args.port}')
    print(f'  Username    : {AUTH_USERNAME} / {AUTH_PASSWORD}')
    print(f'  Web Push    : {"ACTIVE" if HAS_WEBPUSH and VAPID_PUBLIC_KEY else "DISABLED (Set VAPID in .env)"}')
    print('=' * 60)
    
    waitress_serve(
        app,
        host=args.host,
        port=args.port,
        threads=args.threads,
        channel_timeout=300,
        recv_bytes=65536,
        send_bytes=9000,
        outbuf_high_watermark=16384,
        connection_limit=50,
        cleanup_interval=30,
        log_untrusted_proxy_headers=False,
    )

if __name__ == '__main__':
    main()
