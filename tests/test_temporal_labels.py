"""Tests for frame-interval annotations and temporal window labels."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.temporal_labels import (
    AMBIGUOUS_LABEL,
    label_window_starts,
    load_fall_annotations,
    validate_fall_annotation,
    window_end_pose_mask,
    window_start_indices,
)


class TemporalLabelTests(unittest.TestCase):
    def test_window_starts_include_the_final_frame_without_duplicates(self):
        self.assertEqual(
            window_start_indices(42, window_size=30, stride=5),
            [0, 5, 10, 12],
        )
        self.assertEqual(
            window_start_indices(30, window_size=30, stride=5),
            [0],
        )

    def test_only_windows_with_enough_annotated_fall_frames_are_fall(self):
        starts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        annotation = {
            "video_stem": "fall_1",
            "source_frame_count": 75,
            "fall_start_frame": 30,
            "fall_end_frame": 44,
        }
        labels, details = label_window_starts(
            num_frames=75,
            starts=starts,
            window_size=30,
            annotation=annotation,
            min_fall_overlap_frames=6,
        )

        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[1], AMBIGUOUS_LABEL)
        self.assertTrue(np.all(labels[2:8] == 1))
        self.assertEqual(labels[8], AMBIGUOUS_LABEL)
        self.assertEqual(labels[9], 0)
        self.assertEqual(details["fall_frames"], 15)
        self.assertEqual(details["fall_windows"], 6)

    def test_annotation_rejects_a_changed_source_frame_count(self):
        annotation = {
            "fall_start_frame": 10,
            "fall_end_frame": 20,
            "source_frame_count": 80,
        }
        with self.assertRaisesRegex(ValueError, "Frame count"):
            validate_fall_annotation("fall_1", annotation, num_frames=75)

    def test_missing_pose_frames_cannot_make_a_fall_window_positive(self):
        annotation = {
            "video_stem": "fall_1",
            "source_frame_count": 40,
            "fall_start_frame": 10,
            "fall_end_frame": 19,
        }
        pose_present = np.ones(40, dtype=bool)
        pose_present[10:17] = False
        labels, details = label_window_starts(
            num_frames=40,
            starts=[0, 10],
            window_size=30,
            annotation=annotation,
            min_fall_overlap_frames=6,
            pose_present=pose_present,
        )
        self.assertTrue(np.all(labels == AMBIGUOUS_LABEL))
        self.assertEqual(details["window_visible_fall_overlaps"], [3, 3])

    def test_window_must_end_with_a_visible_pose_to_match_runtime(self):
        pose_present = np.ones(40, dtype=bool)
        pose_present[29] = False
        self.assertEqual(
            window_end_pose_mask(pose_present, [0, 10], 30).tolist(),
            [False, True],
        )

    def test_annotation_loader_normalizes_filename_to_stem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "annotations.json"
            path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "videos": {
                        "folder/fall_1.mp4": {
                            "fall_start_frame": 10,
                            "fall_end_frame": 20,
                        }
                    },
                }),
                encoding="utf-8",
            )
            annotations = load_fall_annotations(path)
        self.assertIn("fall_1", annotations)


if __name__ == "__main__":
    unittest.main()
