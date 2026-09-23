#!/usr/bin/env python3
"""Safety bridge between ROS following targets and the Arduino controller.

Serial protocol v2 (Jetson -> Arduino):
    <mode,distance_cm,center_x,detected>\n

Modes are F=follow, S=soft stop, E=latched emergency stop, and R=explicit
emergency-stop reset.  A fall is latched here and in the Arduino; a later
``False`` alert never resumes motion by itself.
"""

import math
import sys
import threading
import time

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

try:
    import serial
except ImportError as error:
    serial = None
    _serial_import_error = error
else:
    _serial_import_error = None


class HumanFollowingNode:
    def __init__(self):
        rospy.init_node('human_following_node', anonymous=False)

        self.src_path = rospy.get_param('~src_path', '/workspace/FallDetectionCNN')
        if self.src_path not in sys.path:
            sys.path.insert(0, self.src_path)
        from src.hafibot_protocol import (  # pylint: disable=import-outside-toplevel
            ESTOP,
            FallEStopLatch,
            FOLLOW,
            RESET_ESTOP,
            STOP,
            encode_command,
        )

        self.MODE_FOLLOW = FOLLOW
        self.MODE_STOP = STOP
        self.MODE_ESTOP = ESTOP
        self.MODE_RESET = RESET_ESTOP
        self._encode_command = encode_command

        self.serial_port = rospy.get_param('~serial_port', '/dev/ttyUSB0')
        self.baudrate = int(rospy.get_param('~baudrate', 115200))
        self.serial_timeout = max(
            0.01, float(rospy.get_param('~serial_timeout', 0.05))
        )
        self.send_rate_hz = max(
            1.0, float(rospy.get_param('~send_rate_hz', 10.0))
        )
        self.target_timeout_sec = max(
            0.1, float(rospy.get_param('~target_timeout', 1.0))
        )
        self.pause_on_fall = bool(rospy.get_param('~pause_on_fall', True))
        # Safe default: an operator must explicitly enable following after boot.
        self.enabled = bool(rospy.get_param('~enabled_on_start', False))

        self._target = {'distance': 0, 'center_x': 0.5, 'detected': 0}
        self._target_lock = threading.Lock()
        self._last_target_monotonic = time.monotonic()

        self._state_lock = threading.RLock()
        self._fall_safety = FallEStopLatch()
        self._reset_pending = False
        self._last_safety_state = None

        self._serial = None
        self._serial_lock = threading.RLock()

        self.state_pub = rospy.Publisher('/hafibot/state', String, queue_size=5)
        self.ultra_pub = rospy.Publisher(
            '/hafibot/ultrasonic', Point, queue_size=5
        )
        self.safety_pub = rospy.Publisher(
            '/human_following/safety_state', String, queue_size=1, latch=True
        )

        rospy.Subscriber(
            '/person_target', Point, self._target_callback, queue_size=1
        )
        rospy.Subscriber(
            '/fall_detection/alert', Bool, self._alert_callback, queue_size=1
        )
        rospy.Subscriber(
            '/human_following/enable', Bool, self._enable_callback, queue_size=1
        )
        self.reset_estop_service = rospy.Service(
            '~reset_estop', Trigger, self._reset_estop_callback
        )

        self._connect_serial()

        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True
        )
        self._sender_thread.start()
        self._reader_thread.start()
        rospy.on_shutdown(self._stop_arduino)

        self._publish_safety_state('DISABLED' if not self.enabled else 'NO_TARGET')
        rospy.loginfo(
            "Human Following v2 siap. Serial=%s@%d, enabled=%s",
            self.serial_port,
            self.baudrate,
            self.enabled,
        )

    # ------------------------------------------------------------------
    # Serial lifecycle
    # ------------------------------------------------------------------
    def _connect_serial(self):
        if serial is None:
            rospy.logerr(
                "pyserial belum ter-install: %s. Motor commands unavailable.",
                _serial_import_error,
            )
            return False

        with self._serial_lock:
            if self._serial is not None and self._serial.is_open:
                return True
            try:
                port = serial.Serial(
                    self.serial_port,
                    self.baudrate,
                    timeout=self.serial_timeout,
                    write_timeout=0.2,
                )
                self._serial = port
            except Exception as error:
                rospy.logwarn_throttle(
                    5.0, "Serial reconnect gagal: %s", error
                )
                self._serial = None
                return False

        # Opening USB serial normally resets the Arduino.  Do not hold the
        # serial lock while waiting for its bootloader.
        time.sleep(2.0)
        with self._serial_lock:
            try:
                if self._serial is port and port.is_open:
                    port.reset_input_buffer()
                    rospy.loginfo("Serial ke Arduino terbuka: %s", self.serial_port)
                    return True
            except Exception as error:
                rospy.logwarn("Serial initialization gagal: %s", error)
                self._drop_serial_locked(port)
        return False

    def _drop_serial_locked(self, expected_port=None):
        if expected_port is not None and self._serial is not expected_port:
            return
        port = self._serial
        self._serial = None
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def _write_packet(self, packet):
        with self._serial_lock:
            if self._serial is None or not self._serial.is_open:
                return False
            try:
                self._serial.write(packet)
                return True
            except Exception as error:
                rospy.logwarn_throttle(5.0, "Serial write error: %s", error)
                self._drop_serial_locked(self._serial)
                return False

    def _write_mode(self, mode, distance=0, center_x=0.5, detected=False):
        return self._write_packet(
            self._encode_command(mode, distance, center_x, detected)
        )

    # ------------------------------------------------------------------
    # ROS callbacks and latched safety state
    # ------------------------------------------------------------------
    def _target_callback(self, msg):
        values = (float(msg.x), float(msg.y), float(msg.z))
        valid = all(math.isfinite(value) for value in values)
        with self._target_lock:
            if valid:
                center_x = max(0.0, min(1.0, values[0]))
                distance = int(max(0.0, min(500.0, values[1])))
                detected = 1 if values[2] >= 0.5 and distance > 0 else 0
                self._target = {
                    'center_x': center_x if detected else 0.5,
                    'distance': distance if detected else 0,
                    'detected': detected,
                }
            else:
                self._target = {
                    'center_x': 0.5,
                    'distance': 0,
                    'detected': 0,
                }
                rospy.logwarn_throttle(
                    2.0, "Target non-finite diterima; fail-safe STOP."
                )
            self._last_target_monotonic = time.monotonic()

    def _alert_callback(self, msg):
        should_send_estop = False
        with self._state_lock:
            self._fall_safety.observe(msg.data, self.pause_on_fall)
            if self.pause_on_fall and self._fall_safety.fall_active:
                self._reset_pending = False
                should_send_estop = True

        if should_send_estop:
            self._publish_safety_state('FALL_ESTOP_LATCHED')
            # Bypass the normal send period so a fall stop is transmitted as
            # soon as the ROS callback runs.
            self._write_mode(self.MODE_ESTOP)
            rospy.logerr_throttle(
                2.0,
                "FALL detected -> ESTOP latched; manual reset required.",
            )

    def _enable_callback(self, msg):
        with self._state_lock:
            self.enabled = bool(msg.data)
            enabled = self.enabled
            estop_latched = self._fall_safety.latched

        if not enabled:
            self._write_mode(self.MODE_STOP)
            self._publish_safety_state('DISABLED')
        elif estop_latched:
            self._publish_safety_state('FALL_ESTOP_LATCHED')
            rospy.logwarn(
                "Following enable diterima, tetapi ESTOP masih ter-latch."
            )
        else:
            self._publish_safety_state('NO_TARGET')
        rospy.loginfo("Human following enabled=%s", enabled)

    def _reset_estop_callback(self, _request):
        with self._state_lock:
            if self._fall_safety.fall_active:
                return TriggerResponse(
                    success=False,
                    message="Fall signal masih aktif; ESTOP tidak di-reset.",
                )
            if not self._fall_safety.latched:
                return TriggerResponse(
                    success=True, message="ESTOP sudah tidak aktif."
                )
            self._fall_safety.request_reset()
            self._reset_pending = True

        self._publish_safety_state('DISABLED' if not self.enabled else 'NO_TARGET')
        rospy.logwarn("ESTOP reset diminta operator; menunggu konfirmasi serial.")
        return TriggerResponse(
            success=True,
            message="Reset ESTOP diantrikan; robot tetap mengikuti status enable.",
        )

    def _publish_safety_state(self, state):
        if state == self._last_safety_state:
            return
        self._last_safety_state = state
        self.safety_pub.publish(String(data=state))

    # ------------------------------------------------------------------
    # Sender and telemetry loops
    # ------------------------------------------------------------------
    def _sender_loop(self):
        rate = rospy.Rate(self.send_rate_hz)
        while not rospy.is_shutdown():
            try:
                if not self._connect_serial():
                    self._publish_safety_state('SERIAL_OFFLINE')
                    rate.sleep()
                    continue

                with self._target_lock:
                    target = dict(self._target)
                    target_age = (
                        time.monotonic() - self._last_target_monotonic
                    )

                with self._state_lock:
                    if self._fall_safety.latched:
                        mode = self.MODE_ESTOP
                        safety_state = 'FALL_ESTOP_LATCHED'
                    elif self._reset_pending:
                        mode = self.MODE_RESET
                        safety_state = 'RESETTING_ESTOP'
                    elif not self.enabled:
                        mode = self.MODE_STOP
                        safety_state = 'DISABLED'
                    elif target_age > self.target_timeout_sec:
                        mode = self.MODE_STOP
                        safety_state = 'TARGET_STALE'
                    elif not target['detected']:
                        mode = self.MODE_STOP
                        safety_state = 'NO_TARGET'
                    else:
                        mode = self.MODE_FOLLOW
                        safety_state = 'FOLLOWING'

                    if mode == self.MODE_FOLLOW:
                        sent = self._write_mode(
                            mode,
                            target['distance'],
                            target['center_x'],
                            True,
                        )
                    else:
                        sent = self._write_mode(mode)

                    if (
                        sent
                        and mode == self.MODE_RESET
                        and not self._fall_safety.fall_active
                    ):
                        self._reset_pending = False

                self._publish_safety_state(safety_state)
            except Exception as error:
                rospy.logerr_throttle(5.0, "Sender loop error: %s", error)

            rate.sleep()

    def _reader_loop(self):
        while not rospy.is_shutdown():
            with self._serial_lock:
                port = self._serial
            if port is None or not port.is_open:
                time.sleep(0.2)
                continue

            try:
                line = port.readline()
                if not line:
                    continue
                text = line.decode('ascii', errors='ignore').strip()
                if not text:
                    continue

                if text.startswith('S,'):
                    parts = text.split(',')
                    if len(parts) >= 5:
                        state = parts[1][:24]
                        us_left = float(parts[2])
                        us_center = float(parts[3])
                        us_right = float(parts[4])
                        if not all(
                            math.isfinite(value)
                            for value in (us_left, us_center, us_right)
                        ):
                            continue
                        self.state_pub.publish(String(data=state))
                        self.ultra_pub.publish(
                            Point(x=us_left, y=us_center, z=us_right)
                        )
                elif text.startswith('#'):
                    rospy.loginfo("[Arduino] %s", text)
                else:
                    rospy.logdebug("[Arduino] %s", text)
            except Exception as error:
                rospy.logwarn_throttle(5.0, "Serial reader error: %s", error)
                with self._serial_lock:
                    self._drop_serial_locked(port)
                time.sleep(0.2)

    def _stop_arduino(self):
        """Send a soft stop and close serial during orderly ROS shutdown."""
        try:
            self._write_mode(self.MODE_STOP)
            with self._serial_lock:
                if self._serial is not None and self._serial.is_open:
                    self._serial.flush()
                    time.sleep(0.1)
                self._drop_serial_locked(self._serial)
            rospy.loginfo("Serial Arduino ditutup; robot STOP.")
        except Exception as error:
            rospy.logwarn("Gagal shutdown serial: %s", error)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        HumanFollowingNode().run()
    except rospy.ROSInterruptException:
        pass
