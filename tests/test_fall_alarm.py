import tempfile
import unittest
import wave
from pathlib import Path

from src.fall_alarm import AlertRisingEdgeGate, write_beep_pattern


class FallAlarmTests(unittest.TestCase):
    def test_only_rising_edges_trigger(self):
        gate = AlertRisingEdgeGate()
        self.assertFalse(gate.observe(False, now=0.0))
        self.assertTrue(gate.observe(True, now=1.0))
        self.assertFalse(gate.observe(True, now=2.0))
        self.assertFalse(gate.observe(False, now=3.0))
        self.assertTrue(gate.observe(True, now=4.0))

    def test_cooldown_suppresses_rapid_second_alert(self):
        gate = AlertRisingEdgeGate(cooldown_sec=5.0)
        self.assertTrue(gate.observe(True, now=10.0))
        gate.observe(False, now=11.0)
        self.assertFalse(gate.observe(True, now=12.0))
        gate.observe(False, now=13.0)
        self.assertTrue(gate.observe(True, now=15.0))

    def test_generated_wave_has_expected_format_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alarm.wav"
            samples = write_beep_pattern(
                path,
                frequency_hz=1000,
                beep_duration_sec=0.1,
                beep_count=3,
                gap_sec=0.05,
                volume=0.5,
                sample_rate=8000,
            )
            self.assertEqual(samples, 3200)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), 8000)
                self.assertEqual(handle.getnframes(), 3200)

    def test_invalid_pattern_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_beep_pattern(Path(directory) / "bad.wav", beep_count=0)


if __name__ == "__main__":
    unittest.main()
