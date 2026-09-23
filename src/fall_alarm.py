"""Small, dependency-free helpers for the Jetson audible fall alarm."""

import math
import struct
import time
import wave


class AlertRisingEdgeGate:
    """Trigger once when an alert changes from inactive to active."""

    def __init__(self, cooldown_sec=0.0, clock=None):
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._clock = clock or time.monotonic
        self._active = False
        self._last_trigger = None

    def observe(self, active, now=None):
        active = bool(active)
        rising_edge = active and not self._active
        self._active = active
        if not rising_edge:
            return False

        observed_at = self._clock() if now is None else float(now)
        if (
            self._last_trigger is not None
            and observed_at - self._last_trigger < self.cooldown_sec
        ):
            return False
        self._last_trigger = observed_at
        return True


def write_beep_pattern(
    path,
    frequency_hz=880.0,
    beep_duration_sec=0.25,
    beep_count=3,
    gap_sec=0.15,
    volume=0.8,
    sample_rate=44100,
):
    """Write a mono 16-bit PCM WAV containing a repeated sine-wave beep."""
    frequency_hz = float(frequency_hz)
    beep_duration_sec = float(beep_duration_sec)
    beep_count = int(beep_count)
    gap_sec = float(gap_sec)
    volume = float(volume)
    sample_rate = int(sample_rate)
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be greater than zero")
    if beep_duration_sec <= 0:
        raise ValueError("beep_duration_sec must be greater than zero")
    if beep_count < 1:
        raise ValueError("beep_count must be at least one")
    if gap_sec < 0:
        raise ValueError("gap_sec cannot be negative")
    if not 0.0 <= volume <= 1.0:
        raise ValueError("volume must be between zero and one")
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000")

    tone_samples = max(1, int(round(beep_duration_sec * sample_rate)))
    gap_samples = max(0, int(round(gap_sec * sample_rate)))
    amplitude = int(round(32767 * volume))
    frames = bytearray()
    for beep_index in range(beep_count):
        for sample_index in range(tone_samples):
            # A short fade prevents a loud click at each edge of the tone.
            fade_samples = min(int(sample_rate * 0.01), tone_samples // 2)
            envelope = 1.0
            if fade_samples:
                if sample_index < fade_samples:
                    envelope = sample_index / float(fade_samples)
                elif sample_index >= tone_samples - fade_samples:
                    envelope = (tone_samples - sample_index - 1) / float(
                        fade_samples
                    )
            value = int(
                amplitude
                * envelope
                * math.sin(2.0 * math.pi * frequency_hz * sample_index / sample_rate)
            )
            frames.extend(struct.pack("<h", value))
        if beep_index < beep_count - 1:
            frames.extend(b"\x00\x00" * gap_samples)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))
    return len(frames) // 2
