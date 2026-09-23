"""Tests for pseudonymous per-video subject metadata."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.manage_subject_metadata import (
    draw_subject_overlay,
    filtered_video_ids,
)
from src.dataset import split_video_indices
from src.event_calibration import (
    event_threshold_rows,
    select_safety_first_threshold,
)
from src.train_loso import choose_validation_subject
from src.subject_metadata import (
    discover_training_videos,
    load_subject_metadata,
    new_subject_payload,
    resolve_subject_ids,
    subject_summary,
    sync_subject_metadata,
    validate_subject_id,
)


class SubjectMetadataTests(unittest.TestCase):
    def test_loso_validation_uses_smallest_remaining_subject(self):
        counts = {"S01": 136, "S02": 60, "S03": 43, "S04": 33}
        self.assertEqual(
            choose_validation_subject(
                "S02", ["S01", "S02", "S03", "S04"], counts
            ),
            "S04",
        )
        self.assertEqual(
            choose_validation_subject(
                "S04", ["S01", "S02", "S03", "S04"], counts
            ),
            "S03",
        )

    def test_event_threshold_selection_preserves_recall_then_reduces_fp(self):
        rows = event_threshold_rows(
            [0, 0, 1, 1],
            [0.60, 0.96, 0.95, 0.99],
            thresholds=[0.50, 0.90, 0.95, 0.97],
        )
        selected = select_safety_first_threshold(rows)
        self.assertEqual(selected["threshold"], 0.95)
        self.assertEqual(selected["tp"], 2)
        self.assertEqual(selected["fp"], 1)

    def test_subject_split_has_no_participant_leakage(self):
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        subjects = ["S01", "S01", "S02", "S02", "S03", "S03", "S04", "S04"]
        train_idx, val_idx, test_idx = split_video_indices(
            labels,
            subjects,
            split_strategy="subject",
            train_subject_ids=("S01", "S03"),
            val_subject_ids=("S04",),
            test_subject_ids=("S02",),
        )
        self.assertEqual({subjects[i] for i in train_idx}, {"S01", "S03"})
        self.assertEqual({subjects[i] for i in val_idx}, {"S04"})
        self.assertEqual({subjects[i] for i in test_idx}, {"S02"})
        self.assertFalse(set(train_idx) & set(val_idx))
        self.assertFalse(set(train_idx) & set(test_idx))

    def test_subject_split_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "lebih dari satu split"):
            split_video_indices(
                [0, 1, 0, 1],
                ["S01", "S01", "S02", "S02"],
                split_strategy="subject",
                train_subject_ids=("S01",),
                val_subject_ids=("S02",),
                test_subject_ids=("S01",),
            )

    def test_interactive_filter_resumes_only_unassigned_videos(self):
        payload = {"videos": {
            "old:assigned": {
                "source": "old", "video_stem": "assigned",
                "source_relative_path": "ADL/assigned.mp4",
                "class_label": "ADL", "subject_id": "S01",
                "present_in_raw": True,
            },
            "old:pending": {
                "source": "old", "video_stem": "pending",
                "source_relative_path": "Fall/pending.mp4",
                "class_label": "Fall", "subject_id": None,
                "present_in_raw": True,
            },
        }}
        selected = filtered_video_ids(
            payload, source="old", unassigned_only=True
        )
        self.assertEqual(selected, ["old:pending"])

    def test_interactive_overlay_preserves_frame_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        display = draw_subject_overlay(
            frame, "old:clip_1", 10, 75, 15.0, False,
            "S02", ["S01", "S02", "S03", "S04"], 1, 10,
        )
        self.assertEqual(display.shape, frame.shape)

    def test_discovery_qualifies_overlapping_stems_by_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_root = root / "old"
            july_root = root / "july"
            (old_root / "ADL_Berjalan").mkdir(parents=True)
            (july_root / "Fall_ke_depan").mkdir(parents=True)
            (old_root / "ADL_Berjalan" / "clip_1.mp4").touch()
            (july_root / "Fall_ke_depan" / "clip_1.mp4").touch()
            datasets = (
                {"name": "old", "raw_root": old_root},
                {"name": "new_july", "raw_root": july_root},
            )
            discovered = discover_training_videos(datasets)

        self.assertEqual(set(discovered), {"old:clip_1", "new_july:clip_1"})
        self.assertEqual(discovered["old:clip_1"]["class_label"], "ADL")
        self.assertEqual(
            discovered["new_july:clip_1"]["class_label"], "Fall"
        )

    def test_sync_preserves_subject_assignment_and_marks_missing_video(self):
        payload = new_subject_payload()
        payload["videos"] = {
            "old:clip_1": {
                "source": "old",
                "video_stem": "clip_1",
                "class_label": "ADL",
                "source_relative_path": "ADL/clip_1.mp4",
                "subject_id": "S01",
                "present_in_raw": True,
            },
            "old:removed": {
                "source": "old",
                "video_stem": "removed",
                "class_label": "Fall",
                "source_relative_path": "Fall/removed.mp4",
                "subject_id": "S02",
                "present_in_raw": True,
            },
        }
        discovered = {
            "old:clip_1": {
                "source": "old",
                "video_stem": "clip_1",
                "class_label": "ADL",
                "source_relative_path": "ADL/clip_1.mp4",
                "subject_id": None,
                "present_in_raw": True,
            }
        }

        merged = sync_subject_metadata(payload, discovered)

        self.assertEqual(merged["videos"]["old:clip_1"]["subject_id"], "S01")
        self.assertFalse(merged["videos"]["old:removed"]["present_in_raw"])

    def test_resolver_rejects_unassigned_subject_when_required(self):
        payload = new_subject_payload()
        payload["videos"] = {
            "old:clip_1": {
                "source": "old",
                "video_stem": "clip_1",
                "class_label": "ADL",
                "source_relative_path": "ADL/clip_1.mp4",
                "subject_id": None,
                "present_in_raw": True,
            }
        }
        self.assertEqual(
            resolve_subject_ids(["old:clip_1"], payload, False), [None]
        )
        with self.assertRaisesRegex(ValueError, "belum memiliki subject_id"):
            resolve_subject_ids(["old:clip_1"], payload, True)

    def test_load_and_summary_validate_subject_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "subjects.json"
            payload = new_subject_payload()
            payload["videos"] = {
                "old:clip_1": {
                    "source": "old",
                    "video_stem": "clip_1",
                    "class_label": "ADL",
                    "source_relative_path": "ADL/clip_1.mp4",
                    "subject_id": "S01",
                    "present_in_raw": True,
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_subject_metadata(path)

        self.assertEqual(subject_summary(loaded)["subjects"], {"S01": 1})
        self.assertEqual(validate_subject_id("subject-02"), "subject-02")
        with self.assertRaises(ValueError):
            validate_subject_id("person name")


if __name__ == "__main__":
    unittest.main()
