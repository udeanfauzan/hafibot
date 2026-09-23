import math
import unittest

from src.hafibot_protocol import (
    ESTOP,
    FOLLOW,
    RESET_ESTOP,
    STOP,
    FallEStopLatch,
    encode_command,
)
from src.human_following import ShoulderTargetFilter


class Landmark:
    def __init__(self, x=0.5, y=0.5, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility
        self.presence = presence


def landmarks_with_shoulders(
    left_x=0.4,
    right_x=0.6,
    left_y=0.4,
    right_y=0.4,
    visibility=1.0,
    left_hip_x=None,
    right_hip_x=None,
    left_hip_y=None,
    right_hip_y=None,
    hip_visibility=1.0,
):
    landmarks = [Landmark() for _ in range(33)]
    landmarks[11] = Landmark(left_x, left_y, visibility)
    landmarks[12] = Landmark(right_x, right_y, visibility)
    if left_hip_x is not None:
        landmarks[23] = Landmark(left_hip_x, left_hip_y, hip_visibility)
    if right_hip_x is not None:
        landmarks[24] = Landmark(right_hip_x, right_hip_y, hip_visibility)
    return landmarks


def make_filter(**overrides):
    settings = {
        "coefficients": [0.0942, -18.5704, 1165.8],
        "confirmation_frames": 3,
    }
    settings.update(overrides)
    return ShoulderTargetFilter(**settings)


class HumanFollowingSafetyTests(unittest.TestCase):
    def test_protocol_v2_encodes_follow_and_safety_commands(self):
        self.assertEqual(encode_command(FOLLOW, 420, 0.625, True), b"<F,420,0.625,1>\n")
        self.assertEqual(encode_command(STOP, 420, 0.2, True), b"<S,0,0.500,0>\n")
        self.assertEqual(encode_command(ESTOP), b"<E,0,0.500,0>\n")
        self.assertEqual(encode_command(RESET_ESTOP), b"<R,0,0.500,0>\n")

    def test_protocol_rejects_invalid_or_non_finite_values(self):
        with self.assertRaises(ValueError):
            encode_command("X")
        with self.assertRaises(ValueError):
            encode_command(FOLLOW, math.nan, 0.5, True)
        with self.assertRaises(ValueError):
            encode_command(FOLLOW, 300, math.inf, True)

    def test_fall_estop_stays_latched_after_alert_clears(self):
        latch = FallEStopLatch()
        self.assertFalse(latch.observe(False))
        self.assertTrue(latch.observe(True))
        self.assertTrue(latch.observe(False))
        self.assertTrue(latch.request_reset())
        self.assertFalse(latch.latched)

    def test_fall_estop_cannot_reset_while_fall_signal_is_active(self):
        latch = FallEStopLatch()
        latch.observe(True)
        self.assertFalse(latch.request_reset())
        self.assertTrue(latch.latched)

    def test_target_requires_consecutive_valid_frames_before_detection(self):
        target_filter = make_filter()
        pose = landmarks_with_shoulders(left_x=0.43, right_x=0.57)
        first = target_filter.update(pose)
        second = target_filter.update(pose)
        third = target_filter.update(pose)
        self.assertFalse(first["detected"])
        self.assertFalse(second["detected"])
        self.assertTrue(third["detected"])
        self.assertGreater(third["distance_cm"], 0)

    def test_low_visibility_fails_closed_and_resets_confirmation(self):
        target_filter = make_filter(confirmation_frames=2)
        visible = landmarks_with_shoulders(left_x=0.43, right_x=0.57)
        hidden = landmarks_with_shoulders(
            left_x=0.43, right_x=0.57, visibility=0.3
        )
        self.assertFalse(target_filter.update(visible)["detected"])
        rejected = target_filter.update(hidden)
        self.assertFalse(rejected["detected"])
        self.assertEqual(rejected["reason"], "shoulder_confidence")
        self.assertFalse(target_filter.update(visible)["detected"])

    def test_unset_legacy_presence_does_not_reject_visible_shoulders(self):
        target_filter = make_filter(confirmation_frames=1)
        pose = landmarks_with_shoulders(left_x=0.43, right_x=0.57)
        pose[11].presence = 0.0
        pose[12].presence = 0.0
        result = target_filter.update(pose)
        self.assertTrue(result["detected"])
        self.assertEqual(result["reason"], "valid")

    def test_large_target_jump_is_rejected_instead_of_commanded(self):
        target_filter = make_filter(
            confirmation_frames=1,
            max_center_jump=0.1,
            max_distance_jump_cm=50,
        )
        initial = landmarks_with_shoulders(left_x=0.42, right_x=0.58)
        jumped = landmarks_with_shoulders(left_x=0.72, right_x=0.88)
        self.assertTrue(target_filter.update(initial)["detected"])
        rejected = target_filter.update(jumped)
        self.assertFalse(rejected["detected"])
        self.assertEqual(rejected["reason"], "target_discontinuity")

    def test_current_calibration_saturates_near_251_cm(self):
        target_filter = make_filter(confirmation_frames=1, enable_torso_fusion=False)
        very_wide_shoulders = landmarks_with_shoulders(
            left_x=0.25, right_x=0.75
        )
        result = target_filter.update(very_wide_shoulders)
        self.assertTrue(result["detected"])
        self.assertTrue(250 <= result["distance_cm"] <= 252)

    def test_torso_fusion_facing_front(self):
        target_filter = make_filter(confirmation_frames=1, enable_torso_fusion=True)
        # Person facing front: shoulders width = 0.10 (64 px), torso height = 0.1667 (80 px)
        # Torso equivalent = 0.80 * 80 = 64 px -> Fused = 64 px
        pose = landmarks_with_shoulders(
            left_x=0.45, right_x=0.55, left_y=0.30, right_y=0.30,
            left_hip_x=0.45, right_hip_x=0.55, left_hip_y=0.4667, right_hip_y=0.4667,
        )
        result = target_filter.update(pose)
        self.assertTrue(result["detected"])
        self.assertEqual(result["fusion_mode"], "fused_facing")
        self.assertAlmostEqual(result["shoulder_px"], 64.0, delta=0.5)
        self.assertAlmostEqual(result["effective_px"], 64.0, delta=0.5)

    def test_torso_fusion_sideways_turn_invariance(self):
        target_filter = make_filter(confirmation_frames=1, enable_torso_fusion=True)
        # Person turns 70 degrees sideways: apparent shoulder width drops to only 20 px (0.031)
        # But torso height remains constant at 80 px (0.1667) -> torso equivalent = 64 px
        pose_turned = landmarks_with_shoulders(
            left_x=0.485, right_x=0.515, left_y=0.30, right_y=0.30,
            left_hip_x=0.485, right_hip_x=0.515, left_hip_y=0.4667, right_hip_y=0.4667,
        )
        result = target_filter.update(pose_turned)
        self.assertTrue(result["detected"])
        self.assertEqual(result["fusion_mode"], "torso_rotation_invariant")
        # Effective scale is rescued by torso height to 64 px instead of dropping to 19.2 px
        self.assertAlmostEqual(result["effective_px"], 64.0, delta=0.5)
        # Distance remains in ~350-370 cm range instead of blowing up to >800 cm!
        self.assertTrue(340 <= result["distance_cm"] <= 380)

    def test_torso_fusion_fallback_on_hip_occlusion(self):
        target_filter = make_filter(confirmation_frames=1, enable_torso_fusion=True, min_hip_visibility=0.5)
        # Hips are occluded (visibility = 0.2)
        pose_occluded = landmarks_with_shoulders(
            left_x=0.45, right_x=0.55, left_y=0.30, right_y=0.30,
            left_hip_x=0.45, right_hip_x=0.55, left_hip_y=0.50, right_hip_y=0.50,
            hip_visibility=0.2
        )
        result = target_filter.update(pose_occluded)
        self.assertTrue(result["detected"])
        self.assertEqual(result["fusion_mode"], "shoulder_only")
        self.assertAlmostEqual(result["effective_px"], 64.0, delta=0.5)

    def test_torso_fusion_yolo_13_joint_format(self):
        target_filter = make_filter(confirmation_frames=1, enable_torso_fusion=True)
        # YOLO 13 format: 1=L-Shoulder, 2=R-Shoulder, 7=L-Hip, 8=R-Hip
        yolo_landmarks = [Landmark() for _ in range(13)]
        yolo_landmarks[1] = Landmark(0.45, 0.30, 0.9)
        yolo_landmarks[2] = Landmark(0.55, 0.30, 0.9)
        yolo_landmarks[7] = Landmark(0.45, 0.4667, 0.9)
        yolo_landmarks[8] = Landmark(0.55, 0.4667, 0.9)

        result = target_filter.update(yolo_landmarks)
        self.assertTrue(result["detected"])
        self.assertEqual(result["fusion_mode"], "fused_facing")
        self.assertAlmostEqual(result["effective_px"], 64.0, delta=0.5)


if __name__ == '__main__':
    unittest.main()

