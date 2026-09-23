"""Pure target-estimation helpers shared by runtime code and tests.

This module deliberately has no ROS or MediaPipe imports.  The ROS node passes
MediaPipe landmark-like objects with `x`, `y` and `visibility`
attributes.  The legacy MediaPipe Pose API used on Jetson leaves `presence`
unset, so target validation intentionally relies on `visibility` only.
"""

import math


class ShoulderTargetFilter:
    """Validate and smooth shoulder-derived following targets.

    Invalid or discontinuous observations fail closed: the filter immediately
    reports no target and requires a short run of valid frames before motion is
    allowed again.
    """

    def __init__(
        self,
        coefficients,
        frame_width=640,
        frame_height=480,
        min_visibility=0.6,
        confirmation_frames=3,
        distance_alpha=0.35,
        center_alpha=0.4,
        max_distance_jump_cm=90.0,
        max_center_jump=0.38,
        min_distance_cm=30.0,
        max_distance_cm=500.0,
        enable_torso_fusion=True,
        torso_shoulder_ratio=0.80,
        min_hip_visibility=0.5,
    ):
        if len(coefficients) not in (1, 3):
            raise ValueError("shoulder coefficients must contain [A, B, C] or [focal_constant]")

        self.coefficients = tuple(float(value) for value in coefficients)
        if not all(math.isfinite(value) for value in self.coefficients):
            raise ValueError("shoulder coefficients must be finite")

        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.min_visibility = min(1.0, max(0.0, float(min_visibility)))
        self.confirmation_frames = max(1, int(confirmation_frames))
        self.distance_alpha = min(1.0, max(0.01, float(distance_alpha)))
        self.center_alpha = min(1.0, max(0.01, float(center_alpha)))
        self.max_distance_jump_cm = max(0.0, float(max_distance_jump_cm))
        self.max_center_jump = max(0.0, float(max_center_jump))
        self.min_distance_cm = max(0.0, float(min_distance_cm))
        self.max_distance_cm = max(self.min_distance_cm, float(max_distance_cm))
        self.enable_torso_fusion = bool(enable_torso_fusion)
        self.torso_shoulder_ratio = max(0.1, float(torso_shoulder_ratio))
        self.min_hip_visibility = min(1.0, max(0.0, float(min_hip_visibility)))

        self.reset()

    def reset(self):
        self._filtered_distance = None
        self._filtered_center = None
        self._valid_streak = 0

    @staticmethod
    def _confidence(landmark):
        # MediaPipe's legacy Solutions Pose (the 0.8.x build used by the
        # Jetson image) clears the optional protobuf `presence` field before
        # returning landmarks.  Reading that unset field yields 0.0 and would
        # reject every otherwise-visible shoulder.  Visibility is the stable,
        # documented confidence output across both legacy and Tasks APIs.
        if landmark is None:
            return 0.0
        if hasattr(landmark, "visibility"):
            return float(landmark.visibility)
        if isinstance(landmark, dict):
            return float(landmark.get("v", landmark.get("visibility", 1.0)))
        if isinstance(landmark, (list, tuple)) and len(landmark) >= 4:
            return float(landmark[3])
        return float(getattr(landmark, "visibility", 1.0))

    @staticmethod
    def _coords(landmark):
        if landmark is None:
            return None, None
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            return float(landmark.x), float(landmark.y)
        if isinstance(landmark, dict):
            return float(landmark.get("x", 0.0)), float(landmark.get("y", 0.0))
        if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
            return float(landmark[0]), float(landmark[1])
        return None, None

    def _observe(self, landmarks):
        if landmarks is None or len(landmarks) < 13:
            return None, "pose_missing"

        if len(landmarks) == 13:
            left = landmarks[1]   # YOLO 13-joint: left shoulder
            right = landmarks[2]  # YOLO 13-joint: right shoulder
            left_h = landmarks[7]   # YOLO 13-joint: left hip
            right_h = landmarks[8]  # YOLO 13-joint: right hip
        else:
            left = landmarks[11]  # MediaPipe 33-joint: left shoulder
            right = landmarks[12] # MediaPipe 33-joint: right shoulder
            left_h = landmarks[23] if len(landmarks) > 23 else None
            right_h = landmarks[24] if len(landmarks) > 24 else None

        ls_x, ls_y = self._coords(left)
        rs_x, rs_y = self._coords(right)
        if ls_x is None or rs_x is None:
            return None, "non_finite_landmark"
        if not (
            math.isfinite(ls_x)
            and math.isfinite(ls_y)
            and math.isfinite(rs_x)
            and math.isfinite(rs_y)
        ):
            return None, "non_finite_landmark"

        ls_conf = self._confidence(left)
        rs_conf = self._confidence(right)
        if ls_conf < self.min_visibility or rs_conf < self.min_visibility:
            return None, "shoulder_confidence"

        mid_sx = (ls_x + rs_x) / 2.0
        mid_sy = (ls_y + rs_y) / 2.0
        if mid_sx < 0.0 or mid_sx > 1.0:
            return None, "center_out_of_frame"

        dx = (rs_x - ls_x) * self.frame_width
        dy = (rs_y - ls_y) * self.frame_height
        shoulder_px = math.hypot(dx, dy)
        if not math.isfinite(shoulder_px) or shoulder_px <= 5.0:
            return None, "shoulder_width"

        torso_px = 0.0
        effective_px = shoulder_px
        fusion_mode = "shoulder_only"
        center_x = mid_sx

        # Multi-joint Torso + Shoulder Fusion:
        # Mitigates individual body-width differences (broad vs narrow shoulders)
        # and handles sideways turns where shoulder width foreshortens by cos(yaw)
        # while vertical torso height (mid-shoulder to mid-hip) remains invariant.
        if self.enable_torso_fusion and left_h is not None and right_h is not None:
            lh_x, lh_y = self._coords(left_h)
            rh_x, rh_y = self._coords(right_h)
            lh_conf = self._confidence(left_h)
            rh_conf = self._confidence(right_h)

            if (
                lh_x is not None
                and rh_x is not None
                and math.isfinite(lh_x)
                and math.isfinite(lh_y)
                and math.isfinite(rh_x)
                and math.isfinite(rh_y)
                and lh_conf >= self.min_hip_visibility
                and rh_conf >= self.min_hip_visibility
            ):
                mid_hx = (lh_x + rh_x) / 2.0
                mid_hy = (lh_y + rh_y) / 2.0
                dx_t = (mid_hx - mid_sx) * self.frame_width
                dy_t = (mid_hy - mid_sy) * self.frame_height
                h_torso = math.hypot(dx_t, dy_t)

                if math.isfinite(h_torso) and h_torso > 5.0 and mid_hy > mid_sy:
                    torso_px = h_torso
                    s_torso_equiv = self.torso_shoulder_ratio * h_torso
                    if shoulder_px < s_torso_equiv * 0.85:
                        effective_px = s_torso_equiv
                        fusion_mode = "torso_rotation_invariant"
                    else:
                        effective_px = 0.5 * shoulder_px + 0.5 * s_torso_equiv
                        fusion_mode = "fused_facing"
                    center_x = (mid_sx + mid_hx) / 2.0

        if len(self.coefficients) == 1:
            distance_cm = self.coefficients[0] / max(1.0, effective_px)
        else:
            a_value, b_value, c_value = self.coefficients
            clamped_px = effective_px
            if a_value > 0.0 and b_value < 0.0:
                vertex_px = -b_value / (2.0 * a_value)
                clamped_px = min(effective_px, vertex_px)
            distance_cm = (
                a_value * clamped_px * clamped_px
                + b_value * clamped_px
                + c_value
            )
        if not math.isfinite(distance_cm):
            return None, "non_finite_distance"

        distance_cm = min(
            self.max_distance_cm, max(self.min_distance_cm, distance_cm)
        )
        return {
            "center_x": center_x,
            "distance_cm": distance_cm,
            "shoulder_px": shoulder_px,
            "torso_px": torso_px,
            "effective_px": effective_px,
            "fusion_mode": fusion_mode,
        }, "valid"

    def update(self, landmarks):
        observation, reason = self._observe(landmarks)
        if observation is None:
            self.reset()
            return {
                "detected": False,
                "center_x": 0.5,
                "distance_cm": 0,
                "valid_streak": 0,
                "reason": reason,
                "shoulder_px": 0.0,
                "torso_px": 0.0,
                "effective_px": 0.0,
                "fusion_mode": "no_pose",
            }

        center_x = observation["center_x"]
        distance_cm = observation["distance_cm"]
        if self._filtered_distance is not None:
            if (
                abs(distance_cm - self._filtered_distance)
                > self.max_distance_jump_cm
                or abs(center_x - self._filtered_center) > self.max_center_jump
            ):
                self.reset()
                return {
                    "detected": False,
                    "center_x": 0.5,
                    "distance_cm": 0,
                    "valid_streak": 0,
                    "reason": "target_discontinuity",
                    "shoulder_px": observation["shoulder_px"],
                    "torso_px": observation.get("torso_px", 0.0),
                    "effective_px": observation.get("effective_px", observation["shoulder_px"]),
                    "fusion_mode": observation.get("fusion_mode", "shoulder_only"),
                }

            self._filtered_distance += self.distance_alpha * (
                distance_cm - self._filtered_distance
            )
            self._filtered_center += self.center_alpha * (
                center_x - self._filtered_center
            )
        else:
            self._filtered_distance = distance_cm
            self._filtered_center = center_x

        self._valid_streak += 1
        confirmed = self._valid_streak >= self.confirmation_frames
        return {
            "detected": confirmed,
            "center_x": float(self._filtered_center),
            "distance_cm": int(round(self._filtered_distance)) if confirmed else 0,
            "valid_streak": self._valid_streak,
            "reason": "valid" if confirmed else "confirming",
            "shoulder_px": observation["shoulder_px"],
            "torso_px": observation.get("torso_px", 0.0),
            "effective_px": observation.get("effective_px", observation["shoulder_px"]),
            "fusion_mode": observation.get("fusion_mode", "shoulder_only"),
        }

