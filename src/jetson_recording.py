"""Timestamped Jetson landmark recording for deployment-matched training data."""

from __future__ import print_function

import datetime
import json
import os
import threading
import uuid
from collections import deque
from pathlib import Path


SCHEMA_VERSION = 1
NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 4
FEATURES_PER_FRAME = NUM_LANDMARKS * FEATURES_PER_LANDMARK


def _utc_now():
    """Return a timezone-aware UTC datetime on Python 3.6 and newer."""
    return datetime.datetime.now(datetime.timezone.utc)


def _safe_component(value, fallback):
    """Return a filesystem-safe identifier without accepting path fragments."""
    value = str(value or "").strip()
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in value
    ).strip("_")
    return safe[:80] or fallback


def flatten_landmarks(landmarks):
    """Convert MediaPipe landmarks to the 132-value training representation."""
    if landmarks is None:
        return [0.0] * FEATURES_PER_FRAME

    flattened = []
    for index in range(NUM_LANDMARKS):
        if index >= len(landmarks):
            flattened.extend((0.0, 0.0, 0.0, 0.0))
            continue

        landmark = landmarks[index]
        flattened.extend((
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 0.0)),
        ))
    return flattened


class LandmarkSessionRecorder(object):
    """Thread-safe JSON Lines writer for one Jetson capture session at a time."""

    def __init__(self, output_dir, rolling_window=60):
        self.output_dir = Path(output_dir)
        self.rolling_window = max(2, int(rolling_window))
        self._lock = threading.RLock()
        self._handle = None
        self._path = None
        self._session_id = None
        self._metadata = None
        self._frame_count = 0
        self._detected_count = 0
        self._first_process_monotonic = None
        self._last_process_monotonic = None
        self._last_process_start = None
        self._sample_intervals = deque(maxlen=self.rolling_window)
        self._processing_latencies = deque(maxlen=self.rolling_window)

    @property
    def is_recording(self):
        with self._lock:
            return self._handle is not None

    @property
    def path(self):
        with self._lock:
            return self._path

    def start(self, label, subject_id, activity, metadata=None):
        """Start a uniquely named recording and write its metadata header."""
        with self._lock:
            if self._handle is not None:
                return self._path

            safe_label = _safe_component(label, "UNLABELED")
            safe_subject = _safe_component(subject_id, "unknown_subject")
            safe_activity = _safe_component(activity, "unspecified")
            timestamp = _utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
            self._session_id = "%s_%s" % (timestamp, uuid.uuid4().hex[:8])

            self.output_dir.mkdir(parents=True, exist_ok=True)
            filename = "%s_%s_%s_%s.jsonl" % (
                self._session_id,
                safe_label,
                safe_subject,
                safe_activity,
            )
            self._path = self.output_dir / filename
            self._handle = self._path.open("w", encoding="utf-8", buffering=1)
            self._frame_count = 0
            self._detected_count = 0
            self._first_process_monotonic = None
            self._last_process_monotonic = None
            self._last_process_start = None
            self._sample_intervals.clear()
            self._processing_latencies.clear()

            self._metadata = {
                "record_type": "metadata",
                "schema_version": SCHEMA_VERSION,
                "session_id": self._session_id,
                "created_utc": _utc_now().isoformat(),
                "label": safe_label,
                "subject_id": safe_subject,
                "activity": safe_activity,
                "features_per_frame": FEATURES_PER_FRAME,
                "landmark_order": "33_landmarks_x_y_z_visibility",
                "missing_pose_representation": "132_zeros",
                "runtime": metadata or {},
            }
            self._write_json(self._metadata)
            return self._path

    def record_frame(
        self,
        landmarks,
        capture_unix_sec,
        capture_monotonic_sec,
        processing_start_monotonic_sec,
        processing_end_monotonic_sec,
        result=None,
        camera_frame_id=None,
    ):
        """Append one uniquely captured and pose-processed camera frame."""
        with self._lock:
            if self._handle is None:
                return False

            process_start = float(processing_start_monotonic_sec)
            process_end = float(processing_end_monotonic_sec)
            latency_sec = max(0.0, process_end - process_start)
            sample_interval_sec = None
            if self._last_process_start is not None:
                sample_interval_sec = max(0.0, process_start - self._last_process_start)
                if sample_interval_sec > 0.0:
                    self._sample_intervals.append(sample_interval_sec)

            if self._first_process_monotonic is None:
                self._first_process_monotonic = process_start
            self._last_process_monotonic = process_start
            self._last_process_start = process_start
            self._processing_latencies.append(latency_sec)

            pose_detected = landmarks is not None and len(landmarks) >= NUM_LANDMARKS
            if pose_detected:
                self._detected_count += 1

            result = result or {}
            record = {
                "record_type": "frame",
                "frame_index": self._frame_count,
                "camera_frame_id": camera_frame_id,
                "capture_unix_sec": float(capture_unix_sec),
                "capture_monotonic_sec": float(capture_monotonic_sec),
                "processing_start_monotonic_sec": process_start,
                "processing_end_monotonic_sec": process_end,
                "processing_latency_sec": latency_sec,
                "sample_interval_sec": sample_interval_sec,
                "pose_detected": pose_detected,
                "landmarks": flatten_landmarks(landmarks),
                "prediction": result.get("prediction"),
                "fall_probability": float(result.get("fall_prob", 0.0)),
                "model_is_fall": bool(result.get("model_is_fall", False)),
                "alert_active": bool(result.get("is_fall", False)),
            }
            self._write_json(record)
            self._frame_count += 1
            return True

    def snapshot(self):
        """Return current session statistics without stopping the recording."""
        with self._lock:
            duration_sec = 0.0
            if (
                self._first_process_monotonic is not None
                and self._last_process_monotonic is not None
            ):
                duration_sec = max(
                    0.0,
                    self._last_process_monotonic - self._first_process_monotonic,
                )

            effective_fps = 0.0
            if self._frame_count > 1 and duration_sec > 0.0:
                effective_fps = (self._frame_count - 1) / duration_sec

            rolling_fps = 0.0
            if self._sample_intervals:
                mean_interval = sum(self._sample_intervals) / len(self._sample_intervals)
                if mean_interval > 0.0:
                    rolling_fps = 1.0 / mean_interval

            mean_latency_ms = 0.0
            if self._processing_latencies:
                mean_latency_ms = (
                    1000.0
                    * sum(self._processing_latencies)
                    / len(self._processing_latencies)
                )

            detection_rate = 0.0
            if self._frame_count:
                detection_rate = self._detected_count / float(self._frame_count)

            return {
                "recording": self._handle is not None,
                "session_id": self._session_id,
                "path": str(self._path) if self._path is not None else None,
                "frames": self._frame_count,
                "pose_detected_frames": self._detected_count,
                "pose_detection_rate": detection_rate,
                "duration_sec": duration_sec,
                "effective_processing_fps": effective_fps,
                "rolling_processing_fps": rolling_fps,
                "mean_processing_latency_ms": mean_latency_ms,
            }

    def stop(self):
        """Write a final summary, flush it to disk, and close the session."""
        with self._lock:
            if self._handle is None:
                return self.snapshot()

            summary = self.snapshot()
            summary["record_type"] = "summary"
            summary["stopped_utc"] = _utc_now().isoformat()
            summary["recording"] = False
            self._write_json(summary)
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
            return summary

    def _write_json(self, payload):
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
