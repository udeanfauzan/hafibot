"""Experimental runtime regressions without cameras or trained weights."""

from contextlib import redirect_stdout
import io
from unittest import mock

import numpy as np
import pytest

from src.config import INPUT_FEATURES, SEQUENCE_LENGTH
from src.dataset import engineer_selected_features
from src.inference import FallDetector
from src.runtime_candidate import CandidateFallDetector, candidate_posture
from src.yolo_detector import PoseKeypoint


def pose(lying=False, offset=0.0):
    joints = np.zeros((13, 4), dtype=np.float32)
    joints[:, 0] = [0.5, .45, .55, .42, .58, .40, .60, .46, .54, .46, .54, .46, .54]
    joints[:, 1] = [.10, .25, .25, .38, .38, .5, .5, .55, .55, .72, .72, .90, .90]
    joints[:, 3] = 1
    if lying:
        joints[:, 0], joints[:, 1] = joints[:, 1].copy(), joints[:, 0].copy()
    joints[:, 1] += offset
    return joints.flatten()


def detector(profile="temporal_v2", probability=.99):
    with mock.patch.object(FallDetector, "_load_model"), mock.patch.object(FallDetector, "_load_scaler"):
        with redirect_stdout(io.StringIO()):
            result = CandidateFallDetector(runtime_profile=profile)
    result.scaler_mean = np.zeros(INPUT_FEATURES, dtype=np.float32)
    result.scaler_std = np.ones(INPUT_FEATURES, dtype=np.float32)
    result._predict = mock.Mock(return_value=(1, probability, np.array([1 - probability, probability], dtype=np.float32)))
    result._trigger_alert = mock.Mock()
    return result


def frame(d, landmarks, index, size=(640, 480), **kwargs):
    with mock.patch("src.inference.time.monotonic", return_value=index / 15):
        return d.process_frame(landmarks, frame_size=size, **kwargs)


def fill(d, landmarks, count=30):
    for i in range(count):
        result = frame(d, landmarks, i)
    return result


def test_geometry_uses_pixels_and_all_configured_thresholds():
    p = pose()
    p.reshape(13, 4)[7:9, 0] += .2
    info = candidate_posture(p, (640, 480))
    assert info["torso_angle_deg"] == pytest.approx(np.degrees(np.arctan2(.30 * 480, .20 * 640)), abs=1e-4)
    assert not info["is_upright"]  # legacy normalized angle would be 56.3 degrees
    assert candidate_posture(p, (640, 480), min_angle=45)["is_upright"]
    assert not candidate_posture(p, (640, 480), min_angle=45, min_aspect=100)["is_upright"]
    assert not candidate_posture(p, (640, 480), min_angle=45, min_dy=.4)["is_upright"]


def test_geometry_array_object_parity_and_low_confidence():
    p = pose()
    p.reshape(13, 4)[1:3, 3] = .12
    objects = [PoseKeypoint(*joint) for joint in p.reshape(13, 4)]
    assert candidate_posture(p, (640, 480)) == candidate_posture(objects, (640, 480))
    assert not candidate_posture(p, (640, 480))["valid"]  # no nose-as-shoulder fallback


@pytest.mark.parametrize("size", [None, (0, 480), (640, float("nan")), (640,), (-1, 480)])
def test_geometry_rejects_unknown_or_invalid_frame_size(size):
    with pytest.raises(ValueError):
        candidate_posture(pose(), size)


def test_geometry_degenerate_anchors_are_unknown_not_upright():
    p = pose()
    p.reshape(13, 4)[:, :2] = .5
    assert not candidate_posture(p, (640, 480))["valid"]
    assert not candidate_posture(None, (640, 480))["is_upright"]


def test_stationary_upright_high_score_remains_gated():
    d = detector()
    result = fill(d, pose(), 40)
    assert result["positive_streak"] >= 3
    assert result["posture_gated"] and not result["is_fall"]
    assert not result["descending"]
    d._trigger_alert.assert_not_called()


def test_sustained_downward_motion_can_override_upright_veto():
    d = detector()
    fill(d, pose())
    for i in range(30, 36):
        result = frame(d, pose(offset=(i - 29) * .03), i)
    assert result["posture"]["is_upright"] and result["descending"]
    assert result["is_fall"] and not result["posture_gated"]
    assert not result["buffer_reset"]
    assert d._trigger_alert.call_count == 1


def test_low_visibility_motion_does_not_override_upright_veto():
    d = detector()
    p = pose()
    p.reshape(13, 4)[0, 3] = .2
    fill(d, p)
    for i in range(30, 36):
        shifted = p.copy()
        shifted.reshape(13, 4)[:, 1] += (i - 29) * .03
        result = frame(d, shifted, i)
    assert not result["descending"] and not result["is_fall"]


def test_one_positive_cannot_confirm_and_missing_pose_resets_vote():
    d = detector(probability=.1)
    fill(d, pose(lying=True))
    d._predict.return_value = (1, .99, np.array([.01, .99], dtype=np.float32))
    assert not frame(d, pose(lying=True), 30)["is_fall"]
    assert not frame(d, None, 31)["is_fall"]
    assert d.positive_streak == 0


def test_short_gap_preserves_raw_history_and_exact_training_features():
    d = detector(probability=.1)
    fill(d, pose())
    before = np.array(d.frame_buffer)
    d._predict.reset_mock()
    missing = frame(d, None, 30)
    assert not missing["model_evaluated"]
    d._predict.assert_not_called()
    result = frame(d, pose(), 31)
    assert result["short_gap_preserved"] and not result["buffer_reset"]
    assert not result["fall_eligible"]
    expected = np.vstack([before[2:], np.zeros((1, 52), dtype=np.float32), pose()[None]])
    np.testing.assert_array_equal(np.array(d.frame_buffer), expected)
    np.testing.assert_array_equal(d._predict.call_args.args[0], engineer_selected_features(expected))
    assert not frame(d, pose(), 32)["fall_eligible"]
    assert frame(d, pose(), 33)["fall_eligible"]


def test_short_gap_requires_reliable_previous_and_current_hips():
    d = detector(probability=.1)
    p = pose()
    p.reshape(13, 4)[7:9, 3] = .2
    fill(d, p)
    frame(d, None, 30)
    result = frame(d, pose(), 31)
    assert result["buffer_reset"] and not result["short_gap_preserved"]
    assert result["buffer_filled"] == 1


def test_long_gap_rewarms_real_frames_not_repeated_return_pose():
    d = detector(probability=.1)
    fill(d, pose())
    for i in range(30, 34):
        frame(d, None, i)
    d._predict.reset_mock()
    result = frame(d, pose(lying=True), 34)
    assert result["buffer_reset"] and result["buffer_filled"] == 1
    assert not result["model_evaluated"] and not result["buffer_reprimed"]
    d._predict.assert_not_called()


@pytest.mark.parametrize("interruption", ["stream", "resolution", "pose_gap"])
def test_confirmed_hold_survives_visual_discontinuity(interruption):
    d = detector()
    assert fill(d, pose(lying=True), 32)["is_fall"]
    prior_until = d.fall_hold_until
    if interruption == "stream":
        result = frame(d, None, 42)
    elif interruption == "resolution":
        result = frame(d, None, 32, size=(1280, 720))
    else:
        for i in range(32, 36):
            frame(d, None, i)
        result = frame(d, pose(lying=True), 36)
    assert result["buffer_reset"] and result["is_fall"]
    assert d.fall_hold_until == prior_until
    assert d._trigger_alert.call_count == 1


def test_target_change_clears_stale_evidence_and_hold():
    d = detector()
    assert fill(d, pose(lying=True), 32)["is_fall"]
    result = frame(d, pose(lying=True), 32, target_changed=True)
    assert result["buffer_reset"] and result["buffer_filled"] == 1
    assert not result["is_fall"]


def test_rapid_downward_motion_is_not_a_target_jump_reset():
    d = detector(probability=.1)
    fill(d, pose())
    result = frame(d, pose(offset=.2), 30)
    assert not result["buffer_reset"] and result["buffer_filled"] == SEQUENCE_LENGTH


def test_single_upright_frame_cannot_erase_confirmed_fall():
    d = detector()
    fill(d, pose(lying=True), 32)
    d._predict.return_value = (0, .9, np.array([.9, .1], dtype=np.float32))
    assert frame(d, pose(), 32)["is_fall"]
    assert d.recovery_streak == 1
    for i in range(33, 37):
        result = frame(d, pose(), i)
    assert not result["is_fall"] and d.recovery_streak == 5


def test_missing_or_zero_visibility_pose_never_starts_fall():
    for missing in (None, np.zeros(52, dtype=np.float32)):
        d = detector()
        result = fill(d, missing, 40)
        assert not result["is_fall"] and result["fall_prob"] == 0
        d._predict.assert_not_called()


def test_nonfinite_pose_and_backward_time_fail_loudly():
    d = detector()
    p = pose()
    p[0] = float("nan")
    with pytest.raises(ValueError, match="Invalid observed pose"):
        frame(d, p, 0)
    frame(d, pose(), 10)
    with pytest.raises(ValueError, match="monotonic"):
        frame(d, pose(), 9)


def test_gap_ablation_keeps_legacy_hip_only_rising_veto():
    d = detector(profile="gap_v2")
    fill(d, pose(lying=True))
    for i in range(30, 37):
        p = pose(lying=True)
        p.reshape(13, 4)[7:9, 1] -= (i - 29) * .01
        result = frame(d, p, i)
    assert result["rising"] and result["posture_gated"]
