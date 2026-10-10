"""Runtime replay must not credit an alert that only precedes a true Fall."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.evaluate_runtime_replay import replay_checkpoint, summarize_events


class FakeDetector:
    checkpoint = {}

    def __init__(self, **kwargs):
        self.reset_buffer()

    def reset_buffer(self):
        self.index = 0

    def process_frame(self, landmarks):
        index = self.index
        self.index += 1
        return {
            "is_fall": index == 20, "model_evaluated": True,
            "posture_gated": False, "pose_detected": True,
            "buffer_reprimed": False,
        }


class RuntimeReplayTests(unittest.TestCase):
    def test_pre_fall_false_alarm_is_not_credited_as_detected_fall(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for category in ("ADL", "Fall"):
                (root / category).mkdir()
                np.save(root / category / "clip.npy", np.ones((60, 52), dtype=np.float32))
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({
                "schema_version": 1,
                "videos": {"clip": {"fall_start_frame": 30, "fall_end_frame": 45}},
            }))
            source = {
                "name": "test", "processed_adl_dir": root / "ADL",
                "processed_fall_dir": root / "Fall", "fall_annotations_path": annotations,
            }
            with mock.patch("scripts.evaluate_runtime_replay.FallDetector", FakeDetector), \
                 mock.patch("scripts.evaluate_runtime_replay.training_dataset_sources", return_value=[source]), \
                 mock.patch("scripts.evaluate_runtime_replay.split_video_indices", return_value=([], [], [0, 1])):
                report = replay_checkpoint(root / "candidate.pt")
            self.assertEqual(report["summary"]["confusion_matrix"], [[0, 1], [1, 0]])
            self.assertEqual(report["summary"]["pre_fall_alert_videos"], 1)
            self.assertIsNone(report["videos"][1]["detection_delay_sec"])

    def test_summary_reports_fall_recall_and_alert_precision_separately(self):
        summary = summarize_events([
            {"label": 0, "detected": False},
            {"label": 0, "detected": True},
            {"label": 1, "detected": True, "detection_delay_sec": 0.4},
            {"label": 1, "detected": False},
        ])
        self.assertEqual(summary["fall_recall"], 0.5)
        self.assertEqual(summary["fall_precision"], 0.5)
        self.assertEqual(summary["median_detection_delay_sec"], 0.4)


if __name__ == "__main__":
    unittest.main()
