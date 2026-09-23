"""Tests for leakage-safe hard-negative selection."""

import unittest

import numpy as np

from src.hard_negative_mining import (
    build_sample_weights,
    collect_windows_with_provenance,
    select_hard_negative_indices,
    validate_mining_subject_scope,
)
from src.train_hard_negative import aggregate_metric_dicts


class HardNegativeMiningTests(unittest.TestCase):
    def test_fold_confusion_matrices_aggregate_exactly(self):
        metrics = aggregate_metric_dicts([
            {"confusion_matrix": [[3, 1], [2, 4]]},
            {"confusion_matrix": [[2, 0], [1, 3]]},
        ])
        self.assertEqual(metrics["confusion_matrix"], [[5, 1], [3, 7]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["fall_precision"], 0.875)
        self.assertAlmostEqual(metrics["fall_recall"], 0.7)

    def test_selection_excludes_fall_videos_and_temporal_fall_labels(self):
        selected = select_hard_negative_indices(
            fall_probabilities=[0.9, 0.8, 0.99, 0.95],
            window_labels=[0, 1, 0, 0],
            source_event_labels=[0, 0, 1, 0],
            source_video_indices=[10, 10, 11, 12],
            source_window_indices=[0, 3, 0, 0],
            min_probability=0.5,
            max_per_video=10,
            min_gap_windows=0,
        )
        np.testing.assert_array_equal(selected, [3, 0])

    def test_selection_deduplicates_and_caps_each_video(self):
        selected = select_hard_negative_indices(
            fall_probabilities=[0.99, 0.98, 0.97, 0.96, 0.95],
            window_labels=[0, 0, 0, 0, 0],
            source_event_labels=[0, 0, 0, 0, 0],
            source_video_indices=[2, 2, 2, 2, 3],
            source_window_indices=[10, 11, 14, 20, 0],
            min_probability=0.5,
            max_per_video=2,
            min_gap_windows=3,
        )
        # Video 2 keeps windows 10 and 14; 11 is suppressed and 20 exceeds cap.
        np.testing.assert_array_equal(selected, [0, 2, 4])

    def test_sampler_weights_only_selected_indices(self):
        weights = build_sample_weights(5, [1, 4], multiplier=3.5)
        np.testing.assert_array_equal(weights, [1.0, 3.5, 1.0, 1.0, 3.5])
        with self.assertRaises(ValueError):
            build_sample_weights(5, [5], multiplier=3.5)

    def test_subject_scope_rejects_validation_or_test_leakage(self):
        source_videos = np.asarray([0, 1, 2])
        subjects = ["S01", "S02", "S03"]
        self.assertEqual(
            validate_mining_subject_scope(
                [0, 1], source_videos, subjects, ["S01", "S02"]
            ),
            {"S01", "S02"},
        )
        with self.assertRaisesRegex(ValueError, "non-training"):
            validate_mining_subject_scope(
                [0, 2], source_videos, subjects, ["S01", "S02"]
            )

    def test_collection_retains_only_requested_video_provenance(self):
        video_windows = [
            [np.zeros((2, 3), dtype=np.float32)],
            [np.ones((2, 3), dtype=np.float32)] * 2,
            [np.full((2, 3), 2.0, dtype=np.float32)],
        ]
        video_labels = [[0], [0, 1], [1]]
        result = collect_windows_with_provenance(
            video_windows, video_labels, [0, 1, 1], [1]
        )
        values, labels, events, videos, windows = result
        self.assertEqual(values.shape, (2, 2, 3))
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(events, [1, 1])
        np.testing.assert_array_equal(videos, [1, 1])
        np.testing.assert_array_equal(windows, [0, 1])


if __name__ == "__main__":
    unittest.main()
