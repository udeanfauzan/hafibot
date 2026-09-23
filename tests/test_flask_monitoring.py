"""Focused safety and security tests for the monitoring web application."""

import importlib.util
import os
import sys
import tempfile
import shutil
import time
import types
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault('FLASK_SECRET_KEY', 'test-only-secret-key-that-is-longer-than-32-characters')
os.environ.setdefault('HAFIBOT_USERNAME', 'caregiver')
os.environ.setdefault('HAFIBOT_PASSWORD', 'test-password-12345')
os.environ.setdefault('HAFIBOT_SECURE_COOKIES', 'false')


def _install_ros_stubs():
    rospy = types.ModuleType('rospy')
    rospy.is_shutdown = lambda: False
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.logwarn = lambda *args, **kwargs: None
    rospy.logerr = lambda *args, **kwargs: None
    rospy.logwarn_throttle = lambda *args, **kwargs: None
    rospy.init_node = lambda *args, **kwargs: None
    rospy.get_param = lambda _name, default=None: default
    rospy.Subscriber = lambda *args, **kwargs: None
    rospy.spin = lambda: None
    sys.modules['rospy'] = rospy

    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.CompressedImage = type('CompressedImage', (), {})
    sensor_msgs.msg = sensor_msgs_msg
    sys.modules['sensor_msgs'] = sensor_msgs
    sys.modules['sensor_msgs.msg'] = sensor_msgs_msg

    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Bool = type('Bool', (), {})
    std_msgs_msg.String = type('String', (), {})
    std_msgs.msg = std_msgs_msg
    sys.modules['std_msgs'] = std_msgs
    sys.modules['std_msgs.msg'] = std_msgs_msg

    waitress = types.ModuleType('waitress')
    waitress.serve = lambda *args, **kwargs: None
    sys.modules['waitress'] = waitress


_install_ros_stubs()
try:
    import flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

MODULE_PATH = Path(__file__).parents[1] / 'src' / 'server_app.py'
if HAS_FLASK:
    SPEC = importlib.util.spec_from_file_location('hafibot_flask_stream', str(MODULE_PATH))
    monitor = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(monitor)
else:
    monitor = None

@unittest.skipUnless(HAS_FLASK, 'flask is required for monitoring tests')
class FlaskMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(self.tmp_dir)
        monitor.app.config.update(TESTING=True, SESSION_COOKIE_SECURE=False)
        monitor.current_jpeg_bytes = None
        monitor.latest_status = None
        monitor._current_frame_id = 0
        monitor._current_frame_source_stamp = 0.0
        monitor._current_frame_received_monotonic = 0.0
        monitor._last_alert_received_monotonic = 0.0
        monitor._latest_processing_stats = {}
        monitor._last_processing_stats_monotonic = 0.0
        monitor._server_started_monotonic = time.monotonic()
        monitor._incidents = []
        monitor._active_signal_incident_id = None
        monitor._previous_alert_status = False
        monitor._fall_clear_started_monotonic = None
        monitor._login_failures = {}
        monitor._push_subscriptions = []
        monitor.HAS_WEBPUSH = False
        monitor.VAPID_PRIVATE_KEY = ''
        monitor.VAPID_PUBLIC_KEY = ''
        monitor.INCIDENTS_FILE = str(tmp_path / 'incidents.json')
        monitor.SNAPSHOTS_DIR = str(tmp_path / 'snapshots')
        monitor.SUBSCRIPTIONS_FILE = str(tmp_path / 'subscriptions.json')
        monitor.IMG_DIR = str(Path(__file__).parents[1] / 'img')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _authenticate(self, client, csrf='test-csrf-token'):
        with client.session_transaction() as flask_session:
            flask_session['logged_in'] = True
            flask_session['username'] = monitor.AUTH_USERNAME
            flask_session['csrf_token'] = csrf
        return csrf

    def test_security_configuration_fails_closed(self):
        orig_key = monitor.app.secret_key
        try:
            monitor.app.secret_key = 'short'
            with self.assertRaises(RuntimeError):
                monitor._validate_security_config()
        finally:
            monitor.app.secret_key = orig_key

    def test_unknown_or_stale_detector_is_never_reported_normal(self):
        self.assertEqual(monitor._status_payload()['state'], 'initializing')
        monitor._server_started_monotonic = time.monotonic() - 30.0
        self.assertEqual(monitor._status_payload()['state'], 'offline')
        now = time.monotonic()
        monitor.latest_status = False
        monitor._current_frame_received_monotonic = now
        monitor._last_alert_received_monotonic = now
        self.assertEqual(monitor._status_payload()['state'], 'normal')

    def test_status_requires_authentication_and_exposes_security_headers(self):
        client = monitor.app.test_client()
        response = client.get('/status')
        self.assertEqual(response.status_code, 401)
        self._authenticate(client)
        response = client.get('/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])

    def test_dashboard_starts_in_connecting_state_not_adl(self):
        client = monitor.app.test_client()
        self._authenticate(client)
        response = client.get('/')
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Menghubungkan ke Hafibot', html)
        self.assertNotIn('Status: ADL', html)
        self.assertIn('Tandai sudah diperiksa', html)

    def test_login_and_logout_require_valid_csrf(self):
        client = monitor.app.test_client()
        self.assertEqual(client.post('/login', data={'username': monitor.AUTH_USERNAME, 'password': monitor.AUTH_PASSWORD}).status_code, 403)
        client.get('/login')
        with client.session_transaction() as flask_session:
            csrf = flask_session['csrf_token']
        response = client.post('/login', data={'csrf_token': csrf, 'username': monitor.AUTH_USERNAME, 'password': monitor.AUTH_PASSWORD})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(client.get('/logout').status_code, 405)
        with client.session_transaction() as flask_session:
            csrf = flask_session['csrf_token']
        self.assertEqual(client.post('/logout', data={'csrf_token': csrf}).status_code, 302)
        self.assertEqual(client.get('/status').status_code, 401)

    def test_fall_rising_edge_creates_one_durable_acknowledgeable_incident(self):
        monitor.current_jpeg_bytes = b'\xff\xd8test-jpeg\xff\xd9'
        true_message = types.SimpleNamespace(data=True)
        false_message = types.SimpleNamespace(data=False)
        monitor.alert_callback(true_message)
        monitor.alert_callback(true_message)
        self.assertEqual(len(monitor._incidents), 1)
        incident = monitor._incidents[0]
        self.assertTrue(Path(monitor.INCIDENTS_FILE).is_file())
        self.assertTrue(Path(monitor.SNAPSHOTS_DIR, incident['snapshot_filename']).is_file())
        self.assertEqual(monitor._status_payload()['state'], 'fall')
        monitor.alert_callback(false_message)
        self.assertIsNone(monitor._incidents[0]['detector_cleared_at'])
        monitor._fall_clear_started_monotonic = (
            time.monotonic() - monitor.FALL_CLEAR_DEBOUNCE_SEC - 0.1
        )
        monitor.alert_callback(false_message)
        self.assertIsNotNone(monitor._incidents[0]['detector_cleared_at'])
        self.assertEqual(monitor._status_payload()['state'], 'fall')
        client = monitor.app.test_client()
        csrf = self._authenticate(client)
        response = client.post('/api/incidents/{}/acknowledge'.format(incident['id']), headers={'X-CSRF-Token': csrf})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.get_json()['incident']['acknowledged_at'])

    def test_transient_normal_predictions_do_not_create_duplicate_incidents(self):
        monitor.current_jpeg_bytes = b'\xff\xd8test-jpeg\xff\xd9'
        true_message = types.SimpleNamespace(data=True)
        false_message = types.SimpleNamespace(data=False)

        monitor.alert_callback(true_message)
        for _ in range(5):
            monitor.alert_callback(false_message)
            monitor._fall_clear_started_monotonic = (
                time.monotonic() - monitor.FALL_CLEAR_DEBOUNCE_SEC + 0.1
            )
            monitor.alert_callback(true_message)

        self.assertEqual(len(monitor._incidents), 1)
        self.assertTrue(monitor._previous_alert_status)
        self.assertIsNone(monitor._incidents[0]['detector_cleared_at'])

    def test_atomic_json_write_retries_transient_windows_access_denied(self):
        target = str(Path(self.tmp_dir) / 'retry.json')
        original_replace = os.replace
        attempts = []

        def flaky_replace(source, destination):
            attempts.append((source, destination))
            if len(attempts) < 3:
                raise PermissionError(5, 'Access is denied', destination)
            return original_replace(source, destination)

        with mock.patch.object(monitor.os, 'replace', side_effect=flaky_replace):
            monitor._write_json_atomic(target, {'ok': True})

        self.assertEqual(len(attempts), 3)
        self.assertEqual(Path(target).read_text(encoding='utf-8').strip(), '{\n  "ok": true\n}')
        self.assertEqual(list(Path(self.tmp_dir).glob('retry.json.*.tmp')), [])

    def test_acknowledged_latest_incident_does_not_surface_old_duplicate_backlog(self):
        monitor._incidents = [
            {'id': 'old-duplicate', 'acknowledged_at': None},
            {'id': 'latest', 'acknowledged_at': '2026-09-11T08:00:00Z'},
        ]
        monitor._active_signal_incident_id = None
        self.assertIsNone(monitor._attention_incident_locked())

    def test_snapshot_requires_login_or_short_lived_signed_token(self):
        monitor.current_jpeg_bytes = b'\xff\xd8test-jpeg\xff\xd9'
        monitor.alert_callback(types.SimpleNamespace(data=True))
        incident_id = monitor._incidents[0]['id']
        client = monitor.app.test_client()
        self.assertEqual(client.get('/api/incidents/{}/snapshot'.format(incident_id)).status_code, 401)
        token = monitor._snapshot_token(incident_id)
        response = client.get('/api/incidents/{}/snapshot?token={}'.format(incident_id, token))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/jpeg')

    def test_incident_acknowledgement_rejects_missing_csrf(self):
        monitor.current_jpeg_bytes = b'\xff\xd8test-jpeg\xff\xd9'
        monitor.alert_callback(types.SimpleNamespace(data=True))
        incident_id = monitor._incidents[0]['id']
        client = monitor.app.test_client()
        self._authenticate(client)
        response = client.post('/api/incidents/{}/acknowledge'.format(incident_id))
        self.assertEqual(response.status_code, 403)

if __name__ == '__main__':
    unittest.main()
