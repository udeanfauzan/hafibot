"""
posture.py - Geometrical Posture Analysis and Verification for Fall Detection.

Modul ini mengevaluasi geometri dan orientasi tubuh manusia dari MediaPipe
landmarks untuk memverifikasi apakah seseorang dalam posisi tegak (upright)
atau berbaring/runtuh di lantai (lying/collapsed).

Fitur utama:
1. Torso Inclination Angle: Sudut kemiringan vektor bahu-pinggul terhadap
   bidang horizontal lantai (0 deg = horizontal/tidur, 90 deg = vertikal/berdiri).
2. Aspect Ratio (H/W): Rasio tinggi terhadap lebar bounding box kerangka.
3. Vertical Torso Displacement: Jarak vertikal bahu terhadap pinggul (dy_torso).
4. Posture Gating: Memblokir false alarm jatuh saat pengguna bergerak menjauh
   dari robot selama postur tubuh masih terverifikasi tegak berdiri.
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Landmark indices MediaPipe Pose standar (33 landmarks)
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

# Selected 13 landmarks yang digunakan oleh CNN-LSTM
SELECTED_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Default geometric threshold
DEFAULT_MIN_UPRIGHT_TORSO_ANGLE = 55.0  # derajat dari horizontal
DEFAULT_MIN_UPRIGHT_ASPECT_RATIO = 1.25  # H / W bounding box
DEFAULT_MIN_DY_TORSO = 0.02             # y_hip - y_shoulder (shoulder di atas hip)
DEFAULT_MAX_LYING_TORSO_ANGLE = 35.0    # derajat dari horizontal
DEFAULT_MAX_LYING_ASPECT_RATIO = 0.85   # H / W bounding box


def _extract_point(lm: Any) -> Optional[Tuple[float, float, float, float]]:
    """Ekstrak (x, y, z, visibility) dari landmark MediaPipe atau dict."""
    if lm is None:
        return None
    if hasattr(lm, 'x') and hasattr(lm, 'y'):
        x = float(lm.x)
        y = float(lm.y)
        z = float(getattr(lm, 'z', 0.0))
        v = float(getattr(lm, 'visibility', 1.0))
    elif isinstance(lm, dict):
        x = float(lm.get('x', 0.0))
        y = float(lm.get('y', 0.0))
        z = float(lm.get('z', 0.0))
        v = float(lm.get('v', lm.get('visibility', 1.0)))
    elif isinstance(lm, (list, tuple, np.ndarray)) and len(lm) >= 2:
        x = float(lm[0])
        y = float(lm[1])
        z = float(lm[2]) if len(lm) > 2 else 0.0
        v = float(lm[3]) if len(lm) > 3 else 1.0
    else:
        return None

    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y, z, v


def parse_pose_landmarks(landmarks: Any) -> Dict[str, Any]:
    """Mengurai berbagai format input landmarks menjadi dictionary koordinat kunci."""
    points = {
        'nose': None,
        'left_shoulder': None,
        'right_shoulder': None,
        'left_hip': None,
        'right_hip': None,
        'left_knee': None,
        'right_knee': None,
        'left_ankle': None,
        'right_ankle': None,
        'all_valid_points': [],
    }

    if landmarks is None:
        return points

    # Jika input berupa flat numpy array (52 fitur untuk 13 landmarks)
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 1 and landmarks.size == 52:
        reshaped = landmarks.reshape(13, 4)
        mapping = {
            0: 'nose',
            1: 'left_shoulder',
            2: 'right_shoulder',
            7: 'left_hip',
            8: 'right_hip',
            9: 'left_knee',
            10: 'right_knee',
            11: 'left_ankle',
            12: 'right_ankle',
        }
        valid_pts = []
        for idx, name in mapping.items():
            pt = (float(reshaped[idx, 0]), float(reshaped[idx, 1]), float(reshaped[idx, 2]), float(reshaped[idx, 3]))
            if math.isfinite(pt[0]) and math.isfinite(pt[1]) and pt[3] > 0.1:
                points[name] = pt
                valid_pts.append(pt)
        points['all_valid_points'] = valid_pts
        return points

    # Jika input berupa flat numpy array (132 fitur untuk 33 landmarks)
    if isinstance(landmarks, np.ndarray) and landmarks.ndim == 1 and landmarks.size == 132:
        reshaped = landmarks.reshape(33, 4)
        landmarks = [reshaped[i] for i in range(33)]

    # Jika input berupa list/sequence landmarks (MediaPipe format 33 landmarks)
    if hasattr(landmarks, '__len__') and len(landmarks) == 13:
        target_map_13 = {
            0: 'nose',
            1: 'left_shoulder',
            2: 'right_shoulder',
            7: 'left_hip',
            8: 'right_hip',
            9: 'left_knee',
            10: 'right_knee',
            11: 'left_ankle',
            12: 'right_ankle',
        }
        valid_pts = []
        for idx, name in target_map_13.items():
            pt = _extract_point(landmarks[idx])
            if pt is not None and pt[3] > 0.15:
                points[name] = pt
                valid_pts.append(pt)
        points['all_valid_points'] = valid_pts
        return points

    if hasattr(landmarks, '__len__') and len(landmarks) >= 25:
        target_map = {
            MP_NOSE: 'nose',
            MP_LEFT_SHOULDER: 'left_shoulder',
            MP_RIGHT_SHOULDER: 'right_shoulder',
            MP_LEFT_HIP: 'left_hip',
            MP_RIGHT_HIP: 'right_hip',
            MP_LEFT_KNEE: 'left_knee',
            MP_RIGHT_KNEE: 'right_knee',
        }
        if len(landmarks) > MP_RIGHT_ANKLE:
            target_map[MP_LEFT_ANKLE] = 'left_ankle'
            target_map[MP_RIGHT_ANKLE] = 'right_ankle'

        valid_pts = []
        for idx, name in target_map.items():
            pt = _extract_point(landmarks[idx])
            if pt is not None and pt[3] > 0.15:
                points[name] = pt
                valid_pts.append(pt)
        points['all_valid_points'] = valid_pts
        return points

    return points


def analyze_posture(
    landmarks: Any,
    min_upright_torso_angle: float = DEFAULT_MIN_UPRIGHT_TORSO_ANGLE,
    min_upright_aspect_ratio: float = DEFAULT_MIN_UPRIGHT_ASPECT_RATIO,
    min_dy_torso: float = DEFAULT_MIN_DY_TORSO,
    max_lying_torso_angle: float = DEFAULT_MAX_LYING_TORSO_ANGLE,
    max_lying_aspect_ratio: float = DEFAULT_MAX_LYING_ASPECT_RATIO,
) -> Dict[str, Any]:
    """Analisis geometris postur tubuh manusia."""
    parsed = parse_pose_landmarks(landmarks)
    valid_points = parsed['all_valid_points']

    if len(valid_points) < 3:
        return {
            'valid': False,
            'torso_angle_deg': 0.0,
            'aspect_ratio': 0.0,
            'dy_torso': 0.0,
            'is_upright': False,
            'is_lying': False,
            'is_fall_consistent': True,
            'posture_label': 'unknown',
        }

    # 1. Bounding Box & Aspect Ratio (H / W)
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    aspect_ratio = float(h / (w + 1e-6))

    # 2. Mid-Shoulder & Mid-Hip Calculation
    ls, rs = parsed['left_shoulder'], parsed['right_shoulder']
    lh, rh = parsed['left_hip'], parsed['right_hip']

    if ls and rs:
        mid_sx, mid_sy = (ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5
    elif ls:
        mid_sx, mid_sy = ls[0], ls[1]
    elif rs:
        mid_sx, mid_sy = rs[0], rs[1]
    elif parsed['nose']:
        mid_sx, mid_sy = parsed['nose'][0], parsed['nose'][1]
    else:
        mid_sx, mid_sy = None, None

    if lh and rh:
        mid_hx, mid_hy = (lh[0] + rh[0]) * 0.5, (lh[1] + rh[1]) * 0.5
    elif lh:
        mid_hx, mid_hy = lh[0], lh[1]
    elif rh:
        mid_hx, mid_hy = rh[0], rh[1]
    else:
        mid_hx, mid_hy = None, None

    # 3. Torso Angle & dy_torso
    if mid_sx is not None and mid_hx is not None:
        dx = abs(mid_sx - mid_hx)
        dy = abs(mid_sy - mid_hy)
        torso_angle_deg = float(math.degrees(math.atan2(dy, dx + 1e-6)))
        dy_torso = float(mid_hy - mid_sy)
    else:
        torso_angle_deg = 45.0
        dy_torso = 0.0

    # 4. Posture Classification
    is_upright = bool(
        torso_angle_deg >= 45.0 and dy_torso >= min_dy_torso
    )

    is_lying = bool(
        torso_angle_deg <= max_lying_torso_angle
        or aspect_ratio <= max_lying_aspect_ratio
    )

    is_fall_consistent = bool(not is_upright)

    if is_upright:
        posture_label = 'upright'
    elif is_lying:
        posture_label = 'lying'
    elif aspect_ratio < 1.1 or torso_angle_deg < 40.0:
        posture_label = 'horizontal'
    else:
        posture_label = 'transition'

    return {
        'valid': True,
        'torso_angle_deg': round(torso_angle_deg, 1),
        'aspect_ratio': round(aspect_ratio, 2),
        'dy_torso': round(dy_torso, 3),
        'is_upright': is_upright,
        'is_lying': is_lying,
        'is_fall_consistent': is_fall_consistent,
        'posture_label': posture_label,
    }


def is_fall_posture_consistent(
    landmarks: Any,
    min_upright_torso_angle: float = DEFAULT_MIN_UPRIGHT_TORSO_ANGLE,
    min_upright_aspect_ratio: float = DEFAULT_MIN_UPRIGHT_ASPECT_RATIO,
    min_dy_torso: float = DEFAULT_MIN_DY_TORSO,
) -> bool:
    """Fungsi cepat untuk memeriksa apakah postur konsisten dengan kondisi jatuh."""
    if landmarks is None:
        return True
    analysis = analyze_posture(
        landmarks,
        min_upright_torso_angle=min_upright_torso_angle,
        min_upright_aspect_ratio=min_upright_aspect_ratio,
        min_dy_torso=min_dy_torso,
    )
    return bool(analysis['is_fall_consistent'])
