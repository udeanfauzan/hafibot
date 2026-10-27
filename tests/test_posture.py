'''Unit tests for geometric posture analysis and false alarm suppression.'''

import math
import unittest
import numpy as np

from src.posture import (
    analyze_posture,
    is_fall_posture_consistent,
    parse_pose_landmarks,
)


class DummyLandmark:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


def make_upright_landmarks(scale=1.0, offset_y=0.0):
    '''Buat kerangka orang berdiri tegak.'''
    landmarks = [DummyLandmark(0.5, 0.5) for _ in range(33)]
    landmarks[0] = DummyLandmark(0.5, 0.10 * scale + offset_y)       # nose
    landmarks[11] = DummyLandmark(0.45, 0.25 * scale + offset_y)     # left shoulder
    landmarks[12] = DummyLandmark(0.55, 0.25 * scale + offset_y)     # right shoulder
    landmarks[23] = DummyLandmark(0.47, 0.55 * scale + offset_y)     # left hip
    landmarks[24] = DummyLandmark(0.53, 0.55 * scale + offset_y)     # right hip
    landmarks[25] = DummyLandmark(0.47, 0.75 * scale + offset_y)     # left knee
    landmarks[26] = DummyLandmark(0.53, 0.75 * scale + offset_y)     # right knee
    landmarks[27] = DummyLandmark(0.47, 0.95 * scale + offset_y)     # left ankle
    landmarks[28] = DummyLandmark(0.53, 0.95 * scale + offset_y)     # right ankle
    return landmarks


def make_lying_landmarks():
    '''Buat kerangka orang terbaring di lantai.'''
    landmarks = [DummyLandmark(0.5, 0.85) for _ in range(33)]
    landmarks[0] = DummyLandmark(0.15, 0.85)     # nose
    landmarks[11] = DummyLandmark(0.28, 0.82)    # left shoulder
    landmarks[12] = DummyLandmark(0.28, 0.88)    # right shoulder
    landmarks[23] = DummyLandmark(0.55, 0.83)    # left hip
    landmarks[24] = DummyLandmark(0.55, 0.87)    # right hip
    landmarks[25] = DummyLandmark(0.72, 0.84)    # left knee
    landmarks[26] = DummyLandmark(0.72, 0.86)    # right knee
    landmarks[27] = DummyLandmark(0.90, 0.85)    # left ankle
    landmarks[28] = DummyLandmark(0.90, 0.85)    # right ankle
    return landmarks


class PostureAnalysisTests(unittest.TestCase):
    def test_upright_standing_is_detected_as_upright(self):
        pose = make_upright_landmarks()
        analysis = analyze_posture(pose)

        self.assertTrue(analysis['valid'])
        self.assertTrue(analysis['is_upright'])
        self.assertFalse(analysis['is_lying'])
        self.assertFalse(analysis['is_fall_consistent'])
        self.assertEqual(analysis['posture_label'], 'upright')
        self.assertGreaterEqual(analysis['torso_angle_deg'], 80.0)
        self.assertGreaterEqual(analysis['aspect_ratio'], 1.5)

    def test_walking_away_small_scale_stays_upright_and_blocks_fall(self):
        pose = make_upright_landmarks(scale=0.35, offset_y=0.40)
        analysis = analyze_posture(pose)

        self.assertTrue(analysis['valid'])
        self.assertTrue(analysis['is_upright'])
        self.assertFalse(analysis['is_lying'])
        self.assertFalse(analysis['is_fall_consistent'])
        self.assertFalse(is_fall_posture_consistent(pose))

    def test_lying_down_is_detected_as_fall_consistent(self):
        pose = make_lying_landmarks()
        analysis = analyze_posture(pose)

        self.assertTrue(analysis['valid'])
        self.assertFalse(analysis['is_upright'])
        self.assertTrue(analysis['is_lying'])
        self.assertTrue(analysis['is_fall_consistent'])
        self.assertEqual(analysis['posture_label'], 'lying')
        self.assertLessEqual(analysis['torso_angle_deg'], 25.0)
        self.assertLessEqual(analysis['aspect_ratio'], 0.5)
        self.assertTrue(is_fall_posture_consistent(pose))

    def test_52_feature_numpy_array_input(self):
        arr = np.zeros(52, dtype=np.float32)
        arr[0], arr[1], arr[3] = 0.5, 0.1, 1.0
        arr[4], arr[5], arr[7] = 0.45, 0.25, 1.0
        arr[8], arr[9], arr[11] = 0.55, 0.25, 1.0
        arr[28], arr[29], arr[31] = 0.47, 0.55, 1.0
        arr[32], arr[33], arr[35] = 0.53, 0.55, 1.0
        arr[36], arr[37], arr[39] = 0.47, 0.75, 1.0
        arr[40], arr[41], arr[43] = 0.53, 0.75, 1.0
        arr[44], arr[45], arr[47] = 0.47, 0.95, 1.0
        arr[48], arr[49], arr[51] = 0.53, 0.95, 1.0

        analysis = analyze_posture(arr)
        self.assertTrue(analysis['valid'])
        self.assertTrue(analysis['is_upright'])

    def test_none_or_missing_pose_handled_safely(self):
        self.assertTrue(is_fall_posture_consistent(None))
        analysis = analyze_posture(None)
        self.assertFalse(analysis['valid'])
        self.assertFalse(analysis['is_upright'])

    def test_non_finite_landmarks_do_not_crash(self):
        pose = [DummyLandmark(math.nan, math.inf) for _ in range(33)]
        analysis = analyze_posture(pose)
        self.assertFalse(analysis['valid'])
        self.assertFalse(analysis['is_upright'])

    def test_missing_torso_anchors_is_unknown_and_not_upright_nor_lying(self):
        # Pose with valid nose, knee, and ankle, but NO shoulder and NO hip
        landmarks = [DummyLandmark(0.5, 0.5, visibility=0.0) for _ in range(33)]
        landmarks[0] = DummyLandmark(0.5, 0.10, visibility=1.0)    # nose
        landmarks[25] = DummyLandmark(0.47, 0.75, visibility=1.0)  # left knee
        landmarks[27] = DummyLandmark(0.47, 0.95, visibility=1.0)  # left ankle
        analysis = analyze_posture(landmarks)
        self.assertTrue(analysis['valid'])
        self.assertFalse(analysis['is_torso_reliable'])
        self.assertFalse(analysis['is_upright'])
        self.assertFalse(analysis['is_lying'])
        self.assertEqual(analysis['posture_label'], 'unknown')


if __name__ == '__main__':
    unittest.main()
