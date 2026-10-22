"""Explicit opt-in only. No setting means the unchanged active legacy detector."""

import json
import math
from pathlib import Path

from src.config import FALL_CONFIDENCE_THRESHOLD
from src.inference import FallDetector
from src.pose_provenance import pose_extraction_signature
from src.runtime_candidate import RUNTIME_CODE_FILES, RUNTIME_PROFILES, make_runtime_detector, runtime_settings
from src.temporal_labels import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_runtime_calibration(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or not report.get("validation_manifest"):
        raise ValueError("Not a frozen runtime validation calibration")
    selection = report["selected"]
    threshold = float(selection["threshold"])
    if selection["profile"] not in RUNTIME_PROFILES or not math.isfinite(threshold) or not .5 <= threshold <= .99:
        raise ValueError("Invalid frozen operating point")
    current_hashes = {name: file_sha256(PROJECT_ROOT / name) for name in RUNTIME_CODE_FILES}
    if report.get("code_sha256") != current_hashes or report.get("fixed_settings") != runtime_settings():
        raise ValueError("Runtime code/settings changed; recalibrate on validation first")
    checkpoint = Path(report["checkpoint"])
    if file_sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("Calibrated classifier checkpoint changed")
    if report.get("pose_signature") != pose_extraction_signature():
        raise ValueError("Calibrated pose extractor changed")
    return report


def create_server_fall_detector(calibration_path=None):
    if not calibration_path:
        return FallDetector(confidence_threshold=FALL_CONFIDENCE_THRESHOLD, enable_posture_filter=True)
    report = verify_runtime_calibration(calibration_path)
    selected = report["selected"]
    print(f"[EXPERIMENTAL] Opt-in runtime {selected['profile']} / threshold {selected['threshold']:.2f}. "
          "Validation replay is not a field-reliability guarantee.")
    return make_runtime_detector(selected["profile"], checkpoint_path=report["checkpoint"],
                                 confidence_threshold=selected["threshold"], enable_posture_filter=True)
