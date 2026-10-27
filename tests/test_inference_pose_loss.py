"""Regression tests for fall handling when MediaPipe temporarily loses a pose."""

import sys
import types
import unittest
from collections import deque
from unittest import mock

import numpy as np


# The state-machine tests do not create a MediaPipe detector. Allow them to run
# on development machines where the Jetson-specific MediaPipe wheel is absent.
try:
    import mediapipe  # noqa: F401
except ImportError:
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from src.inference import (  # noqa: E402
    FallDetector,
    INPUT_FEATURES,
    SELECTED_FEATURES,
    SEQUENCE_LENGTH,
)
from src.dataset import engineer_selected_features, filter_landmarks, sliding_window  # noqa: E402
from src.config import BASE_FEATURES  # noqa: E402


class _Landmark:
    x = 0.5
    y = 0.5
    z = 0.0
    visibility = 1.0


LANDMARKS = [_Landmark() for _ in range(33)]


def make_detector(prediction=(0, 0.9, np.array([0.9, 0.1], dtype=np.float32))):
    """Build a detector state without loading model files or MediaPipe."""
    detector = FallDetector.__new__(FallDetector)
    detector.confidence_threshold = 0.6
    detector.pose_loss_grace_sec = 2.0
    detector.fall_hold_sec = 5.0
    detector.pose_reacquire_confirmation_frames = 3
    detector.enable_posture_filter = True
    detector.min_upright_torso_angle = 50.0
    detector.min_upright_aspect_ratio = 1.25
    detector.min_dy_torso = 0.04
    detector.upright_streak = 0
    detector.last_hip_center = None
    detector.trajectory_history = deque(maxlen=30)
    detector.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    detector.current_prediction = "Menunggu..."
    detector.current_confidence = 0.0
    detector.current_landmarks = None
    detector.is_fall = False
    detector.mute = True
    detector.pose_missing_frames = 0
    detector.pose_detected_streak = 0
    detector.last_pose_monotonic = None
    detector.fall_hold_until = 0.0
    detector.last_alert_time = 0.0
    detector.flash_start_time = 0.0
    detector.scaler_mean = np.zeros(INPUT_FEATURES, dtype=np.float32)
    detector.scaler_std = np.ones(INPUT_FEATURES, dtype=np.float32)
    detector._predict = mock.Mock(return_value=prediction)
    return detector


class PoseLossHandlingTests(unittest.TestCase):
    def fill_buffer(self, detector, count, timestamp=10.0):
        with mock.patch("src.inference.time.monotonic", return_value=timestamp):
            for _ in range(count):
                detector.process_frame(LANDMARKS)

    def test_recent_pose_loss_is_neutral_and_skips_model(self):
        detector = make_detector(
            prediction=(1, 0.9, np.array([0.1, 0.9], dtype=np.float32))
        )
        self.fill_buffer(detector, SEQUENCE_LENGTH - 1, timestamp=10.0)
        detector._predict.reset_mock()

        with mock.patch("src.inference.time.monotonic", return_value=10.5):
            result = detector.process_frame(None)

        detector._predict.assert_not_called()
        self.assertFalse(result["model_evaluated"])
        self.assertFalse(result["model_is_fall"])
        self.assertFalse(result["is_fall"])
        self.assertTrue(result["pose_grace_active"])
        self.assertEqual(result["prediction"], "Tidak Terdeteksi")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["fall_prob"], 0.0)
        self.assertEqual(result["pose_missing_frames"], 1)
        np.testing.assert_array_equal(
            detector.frame_buffer[-1],
            np.zeros(SELECTED_FEATURES, dtype=np.float32),
        )

    def test_no_pose_since_start_cannot_create_a_fall_event(self):
        detector = make_detector(
            prediction=(1, 0.95, np.array([0.05, 0.95], dtype=np.float32))
        )

        with mock.patch("src.inference.time.monotonic", return_value=20.0):
            for _ in range(SEQUENCE_LENGTH):
                result = detector.process_frame(None)

        detector._predict.assert_not_called()
        self.assertFalse(result["model_is_fall"])
        self.assertFalse(result["pose_grace_active"])
        self.assertFalse(result["is_fall"])
        self.assertEqual(result["prediction"], "Tidak Terdeteksi")
        self.assertEqual(result["fall_prob"], 0.0)

    def test_scaler_is_loaded_from_the_same_checkpoint(self):
        detector = FallDetector.__new__(FallDetector)
        expected_mean = np.linspace(0.0, 1.0, INPUT_FEATURES, dtype=np.float32)
        expected_std = np.linspace(1.0, 2.0, INPUT_FEATURES, dtype=np.float32)
        detector.checkpoint = {
            "scaler_mean": expected_mean,
            "scaler_std": expected_std,
        }

        detector._load_scaler()

        np.testing.assert_array_equal(detector.scaler_mean, expected_mean)
        np.testing.assert_array_equal(detector.scaler_std, expected_std)

    def test_training_window_feature_order_matches_runtime(self):
        rng = np.random.default_rng(42)
        raw_video = rng.normal(
            size=(SEQUENCE_LENGTH + 5, 33 * 4)
        ).astype(np.float32)
        selected = filter_landmarks(raw_video)
        base_window = sliding_window(
            selected, SEQUENCE_LENGTH, stride=5
        )[0]

        detector = FallDetector.__new__(FallDetector)
        runtime_features = detector._engineer_features(base_window)
        training_features = engineer_selected_features(base_window)

        np.testing.assert_array_equal(training_features, runtime_features)
        np.testing.assert_array_equal(training_features[0, BASE_FEATURES:], 0.0)

    def test_confirmed_fall_is_held_across_pose_loss(self):
        detector = make_detector(
            prediction=(1, 0.9, np.array([0.1, 0.9], dtype=np.float32))
        )
        self.fill_buffer(detector, SEQUENCE_LENGTH, timestamp=30.0)
        self.assertTrue(detector.is_fall)

        detector._predict.return_value = (
            0, 0.9, np.array([0.9, 0.1], dtype=np.float32)
        )

        with mock.patch("src.inference.time.monotonic", return_value=33.0):
            held_result = detector.process_frame(None)
        self.assertTrue(held_result["is_fall"])
        self.assertTrue(held_result["fall_latched"])

        with mock.patch("src.inference.time.monotonic", return_value=35.1):
            expired_result = detector.process_frame(None)
        self.assertFalse(expired_result["is_fall"])
        self.assertFalse(expired_result["fall_latched"])

    def test_pose_loss_after_grace_period_cannot_start_new_fall(self):
        detector = make_detector(
            prediction=(1, 0.9, np.array([0.1, 0.9], dtype=np.float32))
        )
        self.fill_buffer(detector, SEQUENCE_LENGTH - 1, timestamp=40.0)

        with mock.patch("src.inference.time.monotonic", return_value=42.1):
            result = detector.process_frame(None)

        self.assertFalse(result["model_is_fall"])
        self.assertFalse(result["pose_grace_active"])
        self.assertFalse(result["is_fall"])

    def test_long_pose_loss_reprime_is_instant_and_requires_stable_pose(self):
        detector = make_detector()
        self.fill_buffer(detector, SEQUENCE_LENGTH, timestamp=50.0)
        detector._predict.return_value = (
            1, 0.99, np.array([0.01, 0.99], dtype=np.float32)
        )

        with mock.patch("src.inference.time.monotonic", return_value=52.1):
            detector.process_frame(None)

        with mock.patch("src.inference.time.monotonic", return_value=52.2):
            first = detector.process_frame(LANDMARKS)

        self.assertTrue(first["buffer_reprimed"])
        self.assertEqual(first["buffer_filled"], SEQUENCE_LENGTH)
        self.assertTrue(first["model_evaluated"])
        self.assertTrue(first["model_is_fall"])
        self.assertFalse(first["fall_eligible"])
        self.assertFalse(first["is_fall"])
        self.assertEqual(first["fall_prob"], 0.0)
        self.assertEqual(first["prediction"], "Menstabilkan pose...")
        expected = detector._extract_frame_features(LANDMARKS)
        for frame_features in detector.frame_buffer:
            np.testing.assert_array_equal(frame_features, expected)

        with mock.patch("src.inference.time.monotonic", return_value=52.3):
            second = detector.process_frame(LANDMARKS)
        with mock.patch("src.inference.time.monotonic", return_value=52.4):
            third = detector.process_frame(LANDMARKS)

        self.assertFalse(second["fall_eligible"])
        self.assertFalse(second["is_fall"])
        self.assertTrue(third["fall_eligible"])
        self.assertTrue(third["is_fall"])

    def test_upright_posture_suppresses_false_fall(self):
        # Buat pose orang berdiri tegak
        upright_pose = [_Landmark() for _ in range(33)]
        upright_pose[0].y = 0.10
        upright_pose[11].y = 0.25
        upright_pose[12].y = 0.25
        upright_pose[23].y = 0.55
        upright_pose[24].y = 0.55
        upright_pose[27].y = 0.95
        upright_pose[28].y = 0.95

        detector = make_detector(
            prediction=(1, 0.98, np.array([0.02, 0.98], dtype=np.float32))
        )
        detector.enable_posture_filter = True

        with mock.patch('src.inference.time.monotonic', return_value=60.0):
            for _ in range(SEQUENCE_LENGTH):
                result = detector.process_frame(upright_pose)

        self.assertTrue(result['model_evaluated'])
        self.assertFalse(result['model_is_fall'])
        self.assertTrue(result['posture_gated'])
        self.assertFalse(result['is_fall'])
        self.assertEqual(result['prediction'], 'ADL (Postur Tegak)')

    def test_low_keypoint_count_is_rejected_at_entry_gate(self):
        # Pose with only 2 keypoints visible (under minimum 3)
        sparse_pose = [_Landmark() for _ in range(33)]
        for lm in sparse_pose:
            lm.visibility = 0.1
        sparse_pose[0].visibility = 0.9
        sparse_pose[11].visibility = 0.9

        detector = make_detector(
            prediction=(1, 0.99, np.array([0.01, 0.99], dtype=np.float32))
        )
        result = detector.process_frame(sparse_pose)
        self.assertFalse(result['pose_valid'])
        self.assertFalse(result['pose_detected'])
        self.assertFalse(result['model_evaluated'])
        self.assertFalse(result['is_fall'])
        self.assertIn('too_few_valid_keypoints', result['pose_gate_reason'])

    def test_missing_torso_anchors_rejected_at_entry_gate(self):
        # Pose with 8 keypoints visible, but hips missing
        no_hip_pose = [_Landmark() for _ in range(33)]
        for lm in no_hip_pose:
            lm.visibility = 0.9
        no_hip_pose[23].visibility = 0.1
        no_hip_pose[24].visibility = 0.1

        detector = make_detector(
            prediction=(1, 0.99, np.array([0.01, 0.99], dtype=np.float32))
        )
        result = detector.process_frame(no_hip_pose)
        self.assertFalse(result['pose_valid'])
        self.assertFalse(result['pose_detected'])
        self.assertFalse(result['model_evaluated'])
        self.assertFalse(result['is_fall'])
        self.assertEqual(result['pose_gate_reason'], 'missing_torso_anchors')

    def test_unnatural_torso_shoulder_ratio_rejected_at_entry_gate(self):
        # Torso height extremely disproportionate to shoulder width
        abnormal_pose = [_Landmark() for _ in range(33)]
        for lm in abnormal_pose:
            lm.visibility = 0.9
        # Shoulders width 0.03 * 640 = 19.2 px, hips far (dy=0.8 * 480 = 384 px) -> ratio = 20.0 > 8.5
        abnormal_pose[11].x = 0.500
        abnormal_pose[11].y = 0.100
        abnormal_pose[12].x = 0.530
        abnormal_pose[12].y = 0.100
        abnormal_pose[23].x = 0.500
        abnormal_pose[23].y = 0.900
        abnormal_pose[24].x = 0.530
        abnormal_pose[24].y = 0.900

        detector = make_detector(
            prediction=(1, 0.99, np.array([0.01, 0.99], dtype=np.float32))
        )
        result = detector.process_frame(abnormal_pose)
        self.assertFalse(result['pose_valid'])
        self.assertFalse(result['pose_detected'])
        self.assertFalse(result['model_evaluated'])
        self.assertFalse(result['is_fall'])
        self.assertIn('unnatural_torso_ratio', result['pose_gate_reason'])


if __name__ == "__main__":
    unittest.main()
