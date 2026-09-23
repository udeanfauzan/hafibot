"""Tests for deterministic slow-motion frame/annotation repair."""

import unittest

from scripts.repair_slow_motion_dataset import (
    remap_inclusive_interval,
    retained_frame_count,
)


class TimingRepairTests(unittest.TestCase):
    def test_factor_two_frame_count_keeps_zero_and_every_second_frame(self):
        self.assertEqual(retained_frame_count(180, 2), 90)
        self.assertEqual(retained_frame_count(181, 2), 91)

    def test_inclusive_interval_maps_only_retained_source_frames(self):
        self.assertEqual(remap_inclusive_interval(113, 178, 2), (57, 89))
        self.assertEqual(remap_inclusive_interval(90, 157, 2), (45, 78))

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            remap_inclusive_interval(5, 4, 2)


if __name__ == "__main__":
    unittest.main()
