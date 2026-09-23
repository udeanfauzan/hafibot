#!/usr/bin/env python3
"""Play a non-blocking ALSA beep pattern on each new Fall alert."""

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading

import rospy
from std_msgs.msg import Bool


class FallAlarmNode:
    def __init__(self):
        rospy.init_node("fall_alarm_node", anonymous=False)
        src_path = rospy.get_param("~src_path", "/workspace/FallDetectionCNN")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from src.fall_alarm import (  # pylint: disable=import-outside-toplevel
            AlertRisingEdgeGate,
            write_beep_pattern,
        )

        self.enabled = bool(rospy.get_param("~enabled", True))
        self.alert_topic = rospy.get_param(
            "~alert_topic", "/fall_detection/alert"
        )
        self.alsa_device = str(rospy.get_param("~alsa_device", "default"))
        self._gate = AlertRisingEdgeGate(
            cooldown_sec=float(rospy.get_param("~cooldown_sec", 2.0))
        )
        self._pending = queue.Queue(maxsize=1)
        self._shutdown = threading.Event()
        self._wave_path = os.path.join(
            tempfile.gettempdir(), "hafibot_fall_alarm_{}.wav".format(os.getpid())
        )
        write_beep_pattern(
            self._wave_path,
            frequency_hz=float(rospy.get_param("~frequency_hz", 880.0)),
            beep_duration_sec=float(
                rospy.get_param("~beep_duration_sec", 0.25)
            ),
            beep_count=int(rospy.get_param("~beep_count", 3)),
            gap_sec=float(rospy.get_param("~gap_sec", 0.15)),
            volume=float(rospy.get_param("~volume", 0.8)),
        )

        self._aplay = shutil.which("aplay")
        if self.enabled and self._aplay is None:
            rospy.logerr(
                "Fall alarm enabled tetapi 'aplay' tidak ditemukan. "
                "Install package alsa-utils dan rebuild container."
            )

        self._worker = threading.Thread(target=self._playback_loop)
        self._worker.daemon = True
        self._worker.start()
        self._subscriber = rospy.Subscriber(
            self.alert_topic, Bool, self._alert_callback, queue_size=1
        )
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "Fall alarm siap: enabled=%s, topic=%s, ALSA=%s",
            self.enabled,
            self.alert_topic,
            self.alsa_device,
        )

    def _alert_callback(self, message):
        if not self._gate.observe(message.data) or not self.enabled:
            return
        if self._aplay is None:
            rospy.logerr_throttle(30.0, "Fall alarm cannot play: aplay missing")
            return
        try:
            self._pending.put_nowait(True)
        except queue.Full:
            rospy.logwarn("Fall alarm beep already queued; duplicate ignored")

    def _playback_loop(self):
        while not self._shutdown.is_set():
            try:
                self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                result = subprocess.call([
                    self._aplay,
                    "-q",
                    "-D",
                    self.alsa_device,
                    self._wave_path,
                ])
                if result != 0:
                    rospy.logerr(
                        "Fall alarm aplay gagal (exit=%d, ALSA device=%s)",
                        result,
                        self.alsa_device,
                    )
                else:
                    rospy.loginfo("Fall alarm beep played")
            except OSError as error:
                rospy.logerr("Fall alarm playback error: %s", error)
            finally:
                self._pending.task_done()

    def _on_shutdown(self):
        self._shutdown.set()
        try:
            os.remove(self._wave_path)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        FallAlarmNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
