"""Versioned serial commands for the HAFIBOT motor controller."""

import math


FOLLOW = "F"
STOP = "S"
ESTOP = "E"
RESET_ESTOP = "R"
VALID_MODES = frozenset((FOLLOW, STOP, ESTOP, RESET_ESTOP))


class FallEStopLatch:
    """Small, testable state machine for a manually reset fall stop."""

    def __init__(self):
        self.fall_active = False
        self.latched = False

    def observe(self, fall_active, pause_on_fall=True):
        self.fall_active = bool(fall_active)
        if pause_on_fall and self.fall_active:
            self.latched = True
        return self.latched

    def request_reset(self):
        if self.fall_active:
            return False
        self.latched = False
        return True


def encode_command(mode, distance_cm=0, center_x=0.5, detected=False):
    """Return one strict ASCII protocol-v2 command.

    Protocol: ``<mode,distance_cm,center_x,detected>``.  Non-following modes
    intentionally discard target values so stale target data cannot be acted on.
    """

    if mode not in VALID_MODES:
        raise ValueError("invalid HAFIBOT command mode: {!r}".format(mode))

    distance_value = float(distance_cm)
    center_value = float(center_x)
    if not math.isfinite(distance_value) or not math.isfinite(center_value):
        raise ValueError("target values must be finite")

    if mode != FOLLOW or not detected:
        distance_value, center_value, detected_value = 0, 0.5, 0
    else:
        distance_value = int(max(0, min(500, distance_value)))
        center_value = max(0.0, min(1.0, center_value))
        detected_value = 1 if distance_value > 0 else 0

    packet = "<{},{},{:.3f},{}>\n".format(
        mode, int(distance_value), center_value, detected_value
    )
    return packet.encode("ascii")
