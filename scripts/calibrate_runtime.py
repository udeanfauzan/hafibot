"""Validation-only calibration of full runtime profiles, never classifier maxima.

Fixed profiles and score grid; all alert decisions are replayed at each point.
Model predictions are reused only after checking their exact input fingerprint.
The output freezes the selected operating point before any regression/test run.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_runtime_replay import (  # noqa: E402
    PredictionMemo, load_replay_records, replay_records, select_replay_indices, summarize_events,
)
from src.config import LOGS_DIR, MODEL_DIR, YOLO11_FEATURE_ROOT  # noqa: E402
from src.pose_provenance import pose_extraction_signature  # noqa: E402
from src.runtime_candidate import RUNTIME_PROFILES, RUNTIME_CODE_FILES, make_runtime_detector, runtime_settings  # noqa: E402
from src.temporal_labels import file_sha256  # noqa: E402

THRESHOLDS = tuple(round(i / 100, 2) for i in range(50, 100))


def code_fingerprints():
    return {name: file_sha256(PROJECT_ROOT / name) for name in RUNTIME_CODE_FILES}


def operating_point_key(row):
    """Predeclared recall-first objective; no threshold/profile selection on test.

    Avoid rewarding a false pre-fall latch as a detected fall. Among equal
    onset recall, minimize false ADL episodes/videos and premature episodes;
    prefer lower onset latency, then the higher threshold and simpler profile.
    """
    return (row["onset_fall_recall"], -row["false_adl_videos"],
            -row["false_adl_alert_episodes"], -row["pre_fall_alert_episodes"],
            -(row["median_onset_delay_sec"] if row["median_onset_delay_sec"] is not None else float("inf")),
            row["threshold"], -RUNTIME_PROFILES.index(row["profile"]))


def score_row(profile, threshold, details):
    import numpy as np
    delays = [d["onset_detection_delay_sec"] for d in details if d.get("onset_detection_delay_sec") is not None]
    return {"profile": profile, "threshold": float(threshold), **summarize_events(details),
            "median_onset_delay_sec": float(np.median(delays)) if delays else None}


def calibrate(checkpoint_path, feature_root, output, profiles=RUNTIME_PROFILES):
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in RUNTIME_PROFILES for p in profiles):
        raise ValueError("Need unique, known runtime profiles")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration output already exists; use a new path to retain the frozen audit")
    checkpoint_path = Path(checkpoint_path)
    before_code = code_fingerprints()
    before_model = file_sha256(checkpoint_path)
    rows, best_details, profile_bests = [], None, {}
    validation_manifest = None
    for profile in profiles:
        with redirect_stdout(io.StringIO()):
            d = make_runtime_detector(profile, checkpoint_path=checkpoint_path)
        d.mute = True
        d._trigger_alert = lambda: None
        records, actual_root = load_replay_records(d.checkpoint, feature_root, require_provenance=True)
        indices = select_replay_indices(records, d.checkpoint, "validation")
        if len(indices) == 0:
            raise ValueError("No validation videos")
        manifest = [{"video": records[i]["video"], "label": records[i]["label"],
                     "array_sha256": records[i]["array_sha256"],
                     "source_sha256": records[i]["source_sha256"],
                     "annotation": records[i]["annotation"]} for i in indices]
        if validation_manifest is not None and validation_manifest != manifest:
            raise ValueError("Validation data changed between profiles")
        validation_manifest = manifest
        memo = PredictionMemo()  # never shared across checkpoints or profiles
        profile_rows = []
        for threshold in THRESHOLDS:
            d.confidence_threshold = threshold
            with redirect_stdout(io.StringIO()):
                details = replay_records(d, records, indices, profile, memo=memo, quiet=True)
            row = score_row(profile, threshold, details)
            if not rows or operating_point_key(row) > operating_point_key(max(rows, key=operating_point_key)):
                best_details = details
            rows.append(row)
            profile_rows.append(row)
            print(f"{profile} / {threshold:.2f}: fresh-onset recall={row['onset_fall_recall']:.3f}, "
                  f"false ADL={row['false_adl_videos']}, pre-fall episodes={row['pre_fall_alert_episodes']}", flush=True)
        profile_bests[profile] = {"selected": max(profile_rows, key=operating_point_key), "model_forward_calls": memo.model_calls}
    if before_code != code_fingerprints() or before_model != file_sha256(checkpoint_path):
        raise ValueError("Model/runtime code changed during calibration")
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "VALIDATION ONLY; previously inspected test videos are regression data, not selection data",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": before_model,
        "feature_root": str(Path(actual_root).resolve()),
        "pose_signature": pose_extraction_signature(), "code_sha256": before_code,
        "fixed_settings": runtime_settings(), "threshold_grid": list(THRESHOLDS),
        "selection_rule": "max fresh alert-onset interval recall, min false ADL videos, min false ADL episodes, "
                          "min pre-fall episodes, min median onset delay, max threshold, simpler profile",
        "selected": max(rows, key=operating_point_key), "profile_bests": profile_bests,
        "validation_manifest": validation_manifest, "selected_validation_videos": best_details,
        "sweep": rows,
        "limitations": ["No clinical/field guarantee or 100% recall guarantee", "Video-disjoint, not subject-disjoint",
                        "False alarms/hour is a short-clip proxy, not continuous ADL measurement",
                        "No camera/network latency, ego-motion compensation, motor/notification delivery evaluation"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently overwriting a frozen selection.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Frozen validation selection: {report['selected']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "candidates/yolo11m/best_model.pt")
    parser.add_argument("--feature-root", type=Path, default=YOLO11_FEATURE_ROOT)
    parser.add_argument("--profiles", nargs="+", choices=RUNTIME_PROFILES, default=list(RUNTIME_PROFILES))
    parser.add_argument("--output", type=Path, default=LOGS_DIR / "runtime_v2/calibration.json")
    args = parser.parse_args()
    calibrate(args.checkpoint, args.feature_root, args.output, args.profiles)


if __name__ == "__main__":
    main()
