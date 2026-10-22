"""Regression replay of a frozen validation choice; never selects using test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, RESULTS_DIR  # noqa: E402
from src.runtime_candidate import RUNTIME_PROFILES  # noqa: E402
from src.runtime_loader import verify_runtime_calibration  # noqa: E402
from src.temporal_labels import file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "runtime_v2_comparison.json")
    args = parser.parse_args()
    args.calibration = args.calibration.resolve()
    args.output = args.output.resolve()
    frozen = verify_runtime_calibration(args.calibration)
    active = MODEL_DIR / "best_model.pt"
    scaler = MODEL_DIR / "scaler_params.npz"
    before = {str(p): file_sha256(p) for p in (active, scaler)}
    selected = frozen["selected"]
    scenarios = [("active_legacy_090", active, "legacy_v1", .90)]
    scenarios += [(p + "_090", Path(frozen["checkpoint"]), p, .90) for p in RUNTIME_PROFILES]
    scenarios += [("validation_selected", Path(frozen["checkpoint"]), selected["profile"], selected["threshold"])]
    reports, videos = {}, None
    for name, checkpoint, profile, threshold in scenarios:
        out = args.calibration.parent / (name + "_regression.json")
        with out.with_suffix(".txt").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", "scripts/evaluate_runtime_replay.py",
                            "--checkpoint", str(checkpoint), "--feature-root", frozen["feature_root"],
                            "--profile", profile, "--threshold", str(threshold), "--output", str(out)],
                           cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(out.read_text(encoding="utf-8"))
        names = [(d["video"], d["label"]) for d in report["videos"]]
        if videos is not None and names != videos:
            raise ValueError("Scenarios must use exactly the same regression videos")
        if set(n for n, _ in names) & {r["video"] for r in frozen["validation_manifest"]}:
            raise ValueError("Validation and regression splits overlap")
        videos = names
        reports[name] = {"profile": profile, "threshold": threshold, "summary": report["summary"],
                         "missed_onset_falls": [d["video"] for d in report["videos"] if d["label"] and not d["onset_detected"]],
                         "false_adl_videos": [d["video"] for d in report["videos"] if not d["label"] and d["detected"]],
                         "details_path": str(out.relative_to(PROJECT_ROOT))}
        print(f"{name}: {report['summary']}", flush=True)
    verify_runtime_calibration(args.calibration)
    if before != {str(p): file_sha256(p) for p in (active, scaler)}:
        raise ValueError("Active checkpoint/scaler changed")
    result = {
        "scope": "REGRESSION on 50 previously inspected clips, NOT a pristine unseen final test. No test-based selection.",
        "calibration_sha256": file_sha256(args.calibration),
        "checkpoint_sha256": frozen["checkpoint_sha256"], "active_artifacts_sha256": before,
        "code_sha256": frozen["code_sha256"], "settings": frozen["fixed_settings"],
        "validation_selected": selected, "validation_profile_bests": frozen["profile_bests"],
        "validation_video_count": len(frozen["validation_manifest"]), "regression_video_count": len(videos),
        "regression_scenarios": reports, "candidate_activated": False,
        "limitations": frozen["limitations"] + ["Image-plane geometry is not calibrated 3D/world geometry",
                         "Motion gate can be confounded by mobile-camera ego-motion",
                         "Gap center check is not person identity tracking; explicit target_changed is required for known switches",
                         "Unknown/partial posture is not upright evidence; it may still pass model confirmation",
                         "External server ESTOP/notifications are not measured by detector replay"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {args.output}", flush=True)


if __name__ == "__main__":
    main()
