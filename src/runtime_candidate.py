"""Opt-in runtime experiments. The server's legacy FallDetector is unchanged.

Geometry is measured in equal-scale image coordinates, not 3-D/world angles.
Short gaps keep training's zero rows: no hallucinated or interpolated skeletons.
All settings below are fixed before validation; only the score threshold is swept.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from src.config import SEQUENCE_LENGTH, SELECTED_FEATURES
from src.inference import FallDetector
from src.posture import parse_pose_landmarks

RUNTIME_PROFILES = ("legacy_v1", "geometry_v2", "gap_v2", "temporal_v2")
RUNTIME_CODE_FILES = (
    "src/runtime_candidate.py", "src/runtime_loader.py", "src/inference.py",
    "src/posture.py", "src/config.py", "src/dataset.py", "src/model.py",
    "src/temporal_labels.py", "scripts/evaluate_runtime_replay.py",
    "scripts/calibrate_runtime.py",
)


@dataclass(frozen=True)
class RuntimeSettings:
    max_short_gap_frames: int = 3
    max_short_gap_sec: float = 0.30
    max_stream_interval_sec: float = 0.50
    max_reacquire_center_distance: float = 0.25
    confirmation_frames: int = 3
    upright_recovery_frames: int = 5
    motion_horizon_sec: float = 0.40
    minimum_motion_span_sec: float = 0.20
    motion_torso_fraction: float = 0.35
    minimum_torso_height: float = 0.04
    motion_anchor_visibility: float = 0.35


def candidate_posture(landmarks, frame_size, min_angle=55.0,
                      min_aspect=1.25, min_dy=0.02):
    """Pixel-aspect-aware geometry; unknown anchors are not upright evidence."""
    if frame_size is None or len(frame_size) != 2:
        raise ValueError("Candidate geometry requires actual (width, height)")
    width, height = map(float, frame_size)
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Invalid frame dimensions")
    # Canonicalize input so arrays and live PoseKeypoints use the same confidence
    # cutoff (the legacy parser has different cutoffs for different formats).
    if isinstance(landmarks, np.ndarray) and landmarks.size in (52, 132):
        landmarks = landmarks.reshape(-1, 4).tolist()
    parsed = parse_pose_landmarks(landmarks)
    pts = parsed["all_valid_points"]
    unknown = {"valid": False, "torso_angle_deg": 0.0, "aspect_ratio": 0.0,
               "dy_torso": 0.0, "is_upright": False, "is_lying": False,
               "is_fall_consistent": True, "posture_label": "unknown"}
    shoulders = [parsed[k] for k in ("left_shoulder", "right_shoulder") if parsed[k]]
    hips = [parsed[k] for k in ("left_hip", "right_hip") if parsed[k]]
    if len(pts) < 3 or not shoulders or not hips:
        return unknown
    shoulder = np.mean([p[:2] for p in shoulders], axis=0)
    hip = np.mean([p[:2] for p in hips], axis=0)
    dx, dy = np.abs((hip - shoulder) * [width, height])
    angle = math.degrees(math.atan2(dy, dx))
    xs, ys = np.array([p[:2] for p in pts]).T
    box_width = np.ptp(xs) * width
    box_height = np.ptp(ys) * height
    if box_width <= 1e-6 or np.linalg.norm((hip - shoulder) * [width, height]) <= 1e-6:
        return unknown
    aspect = box_height / box_width
    dy_normalized = float(hip[1] - shoulder[1])
    upright = bool(angle >= min_angle and aspect >= min_aspect and dy_normalized >= min_dy)
    lying = bool(angle <= 35.0 or aspect <= 0.85)
    return {"valid": True, "torso_angle_deg": float(angle),
            "aspect_ratio": float(aspect), "dy_torso": dy_normalized,
            "is_upright": upright, "is_lying": lying,
            "is_fall_consistent": not upright,
            "posture_label": "upright" if upright else "lying" if lying else "transition"}


class GeometryFallDetector(FallDetector):
    """Geometry-only ablation: legacy buffer, gates, and latches are retained."""
    runtime_profile = "geometry_v2"

    def _analyze_posture(self, landmarks, frame_size=None):
        return candidate_posture(landmarks, frame_size, self.min_upright_torso_angle,
                                 self.min_upright_aspect_ratio, self.min_dy_torso)


class CandidateFallDetector(GeometryFallDetector):
    """Bounded-gap buffer, with optional temporal voting and motion-aware gates."""

    def __init__(self, *args, runtime_profile="temporal_v2", settings=None, **kwargs):
        if runtime_profile not in ("gap_v2", "temporal_v2"):
            raise ValueError("Unknown candidate profile")
        self.runtime_profile = runtime_profile
        self.settings = settings or RuntimeSettings()
        super().__init__(*args, **kwargs)
        self._clear_candidate_state()

    def _clear_candidate_state(self):
        self.last_frame_time = None
        self.last_visible_features = None
        self.positive_streak = 0
        self.recovery_streak = 0
        self.motion_history = deque(maxlen=30)
        self.last_frame_size = None

    def reset_buffer(self):
        super().reset_buffer()
        self._clear_candidate_state()

    def _motion(self, features, now):
        joints = features.reshape(13, 4)
        s = self.settings
        # Require reliable nose, both shoulders and both hips; otherwise motion
        # is unknown, not evidence permitting an upright-gate override.
        if np.any(joints[[0, 1, 2, 7, 8], 3] < s.motion_anchor_visibility):
            self.motion_history.clear()
            return False, False
        hip_y = float(joints[7:9, 1].mean())
        nose_y = float(joints[0, 1])
        torso_height = max(abs(hip_y - float(joints[1:3, 1].mean())), s.minimum_torso_height)
        self.motion_history.append((now, hip_y, nose_y, torso_height))
        while self.motion_history and now - self.motion_history[0][0] > s.motion_horizon_sec:
            self.motion_history.popleft()
        first = self.motion_history[0]
        if now - first[0] < s.minimum_motion_span_sec:
            return False, False
        scale = max(first[3], torso_height, s.minimum_torso_height)
        hip_delta, nose_delta = (hip_y - first[1]) / scale, (nose_y - first[2]) / scale
        return (hip_delta >= s.motion_torso_fraction and nose_delta >= s.motion_torso_fraction,
                hip_delta <= -s.motion_torso_fraction and nose_delta <= -s.motion_torso_fraction)

    def process_frame(self, landmarks, frame_size=None, *, target_changed=False):
        now = time.monotonic()
        if self.last_frame_time is not None and now < self.last_frame_time:
            raise ValueError("Runtime timestamps must be monotonic; reset between clips")
        # Validate dimensions even on missing frames. Resolution changes are a
        # discontinuity, not a real skeleton trajectory.
        candidate_posture(None, frame_size)
        frame_size = tuple(frame_size)
        observed = landmarks is not None
        if observed:
            features = self._extract_frame_features(landmarks)
            if features.shape != (SELECTED_FEATURES,) or not np.isfinite(features).all():
                raise ValueError("Invalid observed pose")
            observed = bool(np.any(features.reshape(13, 4)[:, 3] > 0.15))
        if not observed:
            features = np.zeros(SELECTED_FEATURES, dtype=np.float32)
        s = self.settings
        temporal = self.runtime_profile == "temporal_v2"
        stream_gap = (self.last_frame_time is not None
                      and now - self.last_frame_time > s.max_stream_interval_sec)
        size_changed = self.last_frame_size is not None and frame_size != self.last_frame_size
        short_gap = False
        reset_reason = None
        if target_changed or size_changed or stream_gap:
            reset_reason = "target_changed" if target_changed else "resolution_changed" if size_changed else "stream_gap"
        elif observed and self.pose_missing_frames:
            elapsed = now - self.last_pose_monotonic if self.last_pose_monotonic is not None else float("inf")
            compatible = False
            if self.last_visible_features is not None:
                previous_hips = self.last_visible_features.reshape(13, 4)[7:9]
                current_hips = features.reshape(13, 4)[7:9]
                previous = previous_hips[:, :2].mean(axis=0)
                current = current_hips[:, :2].mean(axis=0)
                reliable = (np.all(current_hips[:, 3] >= s.motion_anchor_visibility)
                            and np.all(previous_hips[:, 3] >= s.motion_anchor_visibility))
                compatible = reliable and np.linalg.norm(current - previous) <= s.max_reacquire_center_distance
            short_gap = (self.pose_missing_frames <= s.max_short_gap_frames
                         and elapsed <= s.max_short_gap_sec and compatible)
            if not short_gap:
                reset_reason = "long_or_incompatible_pose_gap"
        if reset_reason:
            # Losing visual continuity invalidates motion evidence, not an
            # already confirmed alert. Only an explicit new identity clears it.
            held_until, held_fall = self.fall_hold_until, self.is_fall
            self.reset_buffer()
            if not target_changed:
                self.fall_hold_until, self.is_fall = held_until, held_fall
        was_fall = self.is_fall
        self.last_frame_time = now
        self.last_frame_size = frame_size
        self.frame_buffer.append(features)
        posture = None
        descending = rising = False
        if observed:
            self.pose_missing_frames = 0
            self.pose_detected_streak += 1
            self.last_pose_monotonic = now
            self.last_visible_features = features.copy()
            posture = self._analyze_posture(landmarks, frame_size)
            if temporal:
                descending, rising = self._motion(features, now)
            self.trajectory_history.append((float(features[7 * 4 + 1:9 * 4:4].mean()), float(features[1])))
        else:
            self.pose_missing_frames += 1
            self.pose_detected_streak = 0
            self.positive_streak = 0
            self.recovery_streak = 0
            self.motion_history.clear()
            self.trajectory_history.append((None, None))
        if not temporal and len(self.trajectory_history) >= 15:
            # Preserve legacy rising veto in the buffer-only ablation.
            valid_hips = [h[0] for h in list(self.trajectory_history)[-15:] if h[0] is not None]
            rising = len(valid_hips) >= 8 and valid_hips[-1] - valid_hips[0] < -0.05
        model_evaluated = observed and len(self.frame_buffer) == SEQUENCE_LENGTH
        raw_prob = 0.0
        probs = np.array([1.0, 0.0], dtype=np.float32)
        if model_evaluated:
            window = np.asarray(self.frame_buffer, dtype=np.float32)
            _, _, probs = self._predict(self._normalize(self._engineer_features(window)))
            raw_prob = float(probs[1])
        eligible = model_evaluated and self.pose_detected_streak >= self.pose_reacquire_confirmation_frames
        upright = bool(posture and posture["valid"] and posture["is_upright"])
        raw_positive = eligible and raw_prob >= self.confidence_threshold
        self.positive_streak = self.positive_streak + 1 if raw_positive else 0
        gate = bool(raw_positive and self.enable_posture_filter
                    and ((upright and not (temporal and descending)) or rising))
        required = s.confirmation_frames if temporal else 1
        confirmed = raw_positive and self.positive_streak >= required and not gate
        if confirmed:
            self.fall_hold_until = max(self.fall_hold_until, now + self.fall_hold_sec)
        # A single noisy upright landmark cannot clear a confirmed fall. Recovery
        # needs observed, stable upright posture and a low model score.
        recovery = bool(observed and upright and not descending
                        and (not temporal or (model_evaluated and raw_prob < self.confidence_threshold)))
        self.recovery_streak = self.recovery_streak + 1 if recovery else 0
        if self.enable_posture_filter and self.recovery_streak >= (s.upright_recovery_frames if temporal else 1):
            self.fall_hold_until = 0.0
            confirmed = False
        self.is_fall = bool(confirmed or now < self.fall_hold_until)
        if self.is_fall and not was_fall:
            self._trigger_alert()
        self.current_prediction = ("Fall" if self.is_fall else "Tidak Terdeteksi" if not observed
                                   else "ADL (Postur Tegak)" if gate and upright
                                   else "ADL (Bangkit)" if gate else "Menstabilkan pose..." if not eligible
                                   else "Fall suspected" if raw_positive else "ADL")
        self.current_confidence = float(probs.max()) if model_evaluated else 0.0
        return {
            "prediction": self.current_prediction, "confidence": self.current_confidence,
            "is_fall": self.is_fall, "buffer_filled": len(self.frame_buffer),
            "buffer_total": SEQUENCE_LENGTH, "fall_prob": raw_prob if eligible else 0.0,
            "raw_fall_prob": raw_prob, "model_is_fall": raw_positive and not gate,
            "model_evaluated": model_evaluated, "fall_eligible": eligible,
            "pose_detected": observed, "pose_missing_frames": self.pose_missing_frames,
            "pose_detected_streak": self.pose_detected_streak,
            "pose_grace_active": observed or (self.last_pose_monotonic is not None
                                             and now - self.last_pose_monotonic <= self.pose_loss_grace_sec),
            "buffer_reprimed": False, "buffer_reset": bool(reset_reason),
            "buffer_reset_reason": reset_reason, "short_gap_preserved": bool(short_gap),
            "fall_latched": self.is_fall and not confirmed,
            "posture": posture, "posture_gated": gate,
            "descending": bool(descending), "rising": bool(rising),
            "positive_streak": self.positive_streak, "recovery_streak": self.recovery_streak,
        }


def make_runtime_detector(profile, **kwargs):
    if profile == "legacy_v1":
        return FallDetector(**kwargs)
    if profile == "geometry_v2":
        return GeometryFallDetector(**kwargs)
    if profile in ("gap_v2", "temporal_v2"):
        return CandidateFallDetector(runtime_profile=profile, **kwargs)
    raise ValueError(f"Unknown runtime profile: {profile}")


def runtime_settings():
    return asdict(RuntimeSettings())
