"""Regression tests for body-relative features and raw-space augmentation."""

import unittest
from unittest import mock

import numpy as np

from src.config import (
    HYBRID_FEATURE_PIPELINE,
    HYBRID_GLOBAL_FEATURES,
    NUM_SELECTED_LANDMARKS,
    SELECTED_FEATURES,
)
from src.dataset import (
    augment_contiguous_occlusion,
    augment_horizontal_flip,
    apply_legacy_standardized_augmentation,
    body_relative_features,
    engineer_selected_features,
    feature_dimensions,
    prepare_base_features,
)


def visible_pose(frames=6):
    """Build a non-degenerate synthetic selected-landmark pose."""
    values = np.zeros((frames, NUM_SELECTED_LANDMARKS, 4), dtype=np.float32)
    for frame_index in range(frames):
        for landmark_index in range(NUM_SELECTED_LANDMARKS):
            values[frame_index, landmark_index, 0] = 0.2 + landmark_index * 0.025
            values[frame_index, landmark_index, 1] = (
                0.15 + landmark_index * 0.04 + frame_index * 0.002
            )
            values[frame_index, landmark_index, 2] = landmark_index * 0.01
            values[frame_index, landmark_index, 3] = 0.9
    # Make shoulder and hip midpoints spatially distinct.
    values[:, 1:3, 1] = 0.3
    values[:, 7:9, 1] = 0.6
    return values.reshape(frames, SELECTED_FEATURES)


class FeaturePipelineTests(unittest.TestCase):
    def test_legacy_augmentation_profile_is_reproducible(self):
        values = np.linspace(-2.0, 2.0, 30 * 130, dtype=np.float32).reshape(
            30, 130
        )

        def reference(window):
            result = window.copy()
            result += np.random.normal(0.0, 0.05, result.shape).astype(np.float32)
            if np.random.random() < 0.5:
                result *= float(np.random.uniform(0.9, 1.1))
            if np.random.random() < 0.5:
                for index in range(NUM_SELECTED_LANDMARKS):
                    result[:, index * 4] *= -1.0
            if np.random.random() < 0.3:
                seq_len = len(result)
                factor = float(np.random.uniform(0.8, 1.2))
                new_len = max(int(seq_len * factor), seq_len)
                stretched = np.zeros((new_len, result.shape[1]), dtype=np.float32)
                for feature_index in range(result.shape[1]):
                    stretched[:, feature_index] = np.interp(
                        np.arange(new_len),
                        np.linspace(0, new_len - 1, seq_len),
                        result[:, feature_index],
                    )
                start = (new_len - seq_len) // 2
                result = stretched[start:start + seq_len]
            if np.random.random() < 0.3:
                result[np.random.random(result.shape) < 0.05] = 0.0
            return result

        np.random.seed(123)
        expected = reference(values)
        np.random.seed(123)
        actual = apply_legacy_standardized_augmentation(values)
        np.testing.assert_array_equal(actual, expected)

    def test_body_relative_landmarks_are_translation_scale_invariant(self):
        selected = visible_pose()
        transformed = selected.reshape(-1, NUM_SELECTED_LANDMARKS, 4).copy()
        transformed[:, :, :3] *= 1.7
        transformed[:, :, 0] += 0.11
        transformed[:, :, 1] -= 0.08
        transformed[:, :, 2] += 0.03
        transformed = transformed.reshape(selected.shape)

        original_features = body_relative_features(selected)
        transformed_features = body_relative_features(transformed)
        np.testing.assert_allclose(
            original_features[:, :SELECTED_FEATURES],
            transformed_features[:, :SELECTED_FEATURES],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertFalse(np.allclose(
            original_features[:, SELECTED_FEATURES:],
            transformed_features[:, SELECTED_FEATURES:],
        ))

    def test_missing_pose_frames_create_no_derivative_spikes(self):
        selected = visible_pose(frames=5)
        selected[2] = 0.0
        engineered = engineer_selected_features(selected, HYBRID_FEATURE_PIPELINE)
        base_count, derivative_count, total_count = feature_dimensions(
            HYBRID_FEATURE_PIPELINE
        )
        self.assertEqual(engineered.shape, (5, total_count))
        np.testing.assert_array_equal(engineered[2], 0.0)
        velocity = engineered[:, base_count:base_count + derivative_count]
        acceleration = engineered[:, base_count + derivative_count:]
        np.testing.assert_array_equal(velocity[2:4], 0.0)
        np.testing.assert_array_equal(acceleration[1:5], 0.0)

    def test_horizontal_flip_swaps_sides_and_is_reversible(self):
        base = prepare_base_features(visible_pose(), HYBRID_FEATURE_PIPELINE)
        flipped = augment_horizontal_flip(base, HYBRID_FEATURE_PIPELINE)
        original_landmarks = base[:, :SELECTED_FEATURES].reshape(
            -1, NUM_SELECTED_LANDMARKS, 4
        )
        flipped_landmarks = flipped[:, :SELECTED_FEATURES].reshape(
            -1, NUM_SELECTED_LANDMARKS, 4
        )
        np.testing.assert_allclose(
            flipped_landmarks[:, 1, 0], -original_landmarks[:, 2, 0]
        )
        restored = augment_horizontal_flip(flipped, HYBRID_FEATURE_PIPELINE)
        np.testing.assert_allclose(restored, base)

    def test_contiguous_occlusion_never_removes_label_endpoint(self):
        base = prepare_base_features(visible_pose(frames=30), HYBRID_FEATURE_PIPELINE)
        with mock.patch("numpy.random.randint", side_effect=[5, 24]):
            occluded = augment_contiguous_occlusion(base, max_frames=5)
        np.testing.assert_array_equal(occluded[24:29], 0.0)
        np.testing.assert_array_equal(occluded[-1], base[-1])

    def test_all_missing_window_stays_exactly_zero(self):
        selected = np.zeros((30, SELECTED_FEATURES), dtype=np.float32)
        base = prepare_base_features(selected, HYBRID_FEATURE_PIPELINE)
        self.assertEqual(base.shape[1], SELECTED_FEATURES + HYBRID_GLOBAL_FEATURES)
        np.testing.assert_array_equal(base, 0.0)
        np.testing.assert_array_equal(
            engineer_selected_features(selected, HYBRID_FEATURE_PIPELINE), 0.0
        )


if __name__ == "__main__":
    unittest.main()
