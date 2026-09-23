import json
import tempfile
import unittest
from pathlib import Path

from src.jetson_recording import LandmarkSessionRecorder, flatten_landmarks


class FakeLandmark(object):
    def __init__(self, value):
        self.x = value
        self.y = value + 0.1
        self.z = value + 0.2
        self.visibility = value + 0.3


class LandmarkSessionRecorderTests(unittest.TestCase):
    def test_missing_pose_is_saved_as_132_zeros(self):
        self.assertEqual(flatten_landmarks(None), [0.0] * 132)

    def test_recording_contains_metadata_frames_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = LandmarkSessionRecorder(directory)
            path = recorder.start(
                label="ADL",
                subject_id="subject/01",
                activity="controlled lying",
                metadata={"mediapipe_api": "legacy_solutions"},
            )
            landmarks = [FakeLandmark(index / 100.0) for index in range(33)]

            recorder.record_frame(
                landmarks=landmarks,
                capture_unix_sec=1000.0,
                capture_monotonic_sec=10.0,
                processing_start_monotonic_sec=10.1,
                processing_end_monotonic_sec=10.2,
                result={"fall_prob": 0.2, "is_fall": False},
                camera_frame_id=7,
            )
            recorder.record_frame(
                landmarks=None,
                capture_unix_sec=1000.25,
                capture_monotonic_sec=10.25,
                processing_start_monotonic_sec=10.35,
                processing_end_monotonic_sec=10.45,
                result={"fall_prob": 0.7, "model_is_fall": True, "is_fall": True},
                camera_frame_id=8,
            )
            summary = recorder.stop()

            self.assertNotIn("/", path.name)
            self.assertIn("subject_01", path.name)
            self.assertIn("controlled_lying", path.name)
            self.assertEqual(summary["frames"], 2)
            self.assertAlmostEqual(summary["effective_processing_fps"], 4.0)
            self.assertAlmostEqual(summary["pose_detection_rate"], 0.5)

            records = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual([item["record_type"] for item in records], [
                "metadata", "frame", "frame", "summary"
            ])
            self.assertEqual(len(records[1]["landmarks"]), 132)
            self.assertEqual(records[2]["landmarks"], [0.0] * 132)
            self.assertEqual(records[2]["camera_frame_id"], 8)
            self.assertFalse(records[3]["recording"])


if __name__ == "__main__":
    unittest.main()
