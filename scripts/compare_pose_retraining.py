"""Compare the baseline and YOLO11m candidate without changing active weights.

Calibrates on validation videos first, then evaluates the same held-out videos
with both offline classification and the live fall state machine.
"""

import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FALL_CONFIDENCE_THRESHOLD, LOGS_DIR, MODEL_DIR, RESULTS_DIR,
    YOLO11_FEATURE_ROOT, training_dataset_sources,
)
from src.pose_provenance import pose_extraction_signature, validate_pose_metadata  # noqa: E402
from src.temporal_labels import file_sha256  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def run_stage(name, arguments, logs_dir):
    print(f"Starting {name}", flush=True)
    log_path = logs_dir / f"{name}.txt"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", *map(str, arguments)],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    print(f"Completed {name}; log: {log_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path,
                        default=MODEL_DIR / "candidates" / "yolo11m" / "best_model.pt")
    parser.add_argument("--baseline", type=Path, default=MODEL_DIR / "best_model.pt")
    parser.add_argument("--feature-root", type=Path, default=YOLO11_FEATURE_ROOT)
    parser.add_argument("--threshold", type=float, default=FALL_CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "yolo11m_retraining_comparison.json")
    args = parser.parse_args()
    logs_dir = LOGS_DIR / "yolo11m_retrain"
    logs_dir.mkdir(parents=True, exist_ok=True)
    active_hash_before = file_sha256(args.baseline)
    active_scaler_hash_before = file_sha256(MODEL_DIR / "scaler_params.npz")
    signature = pose_extraction_signature()
    source_counts, video_hashes = {}, {}
    for source in training_dataset_sources(args.feature_root):
        metadata = read_json(source["processed_adl_dir"].parent / "extraction_metadata.json")
        validate_pose_metadata(metadata, signature)
        source_counts[source["name"]] = metadata["categories"]
        for key, entry in metadata["videos"].items():
            video_hashes.setdefault(entry["source_sha256"], []).append(f"{source['name']}:{key}")
    duplicates = [names for names in video_hashes.values() if len(names) > 1]
    if duplicates:
        raise ValueError(f"Duplicate source videos require a grouped split: {duplicates}")

    calibration_path = logs_dir / "threshold_calibration.json"
    run_stage("validation_calibration", [
        "scripts/calibrate_threshold.py", "--checkpoint", args.candidate,
        "--feature-root", args.feature_root, "--output", calibration_path,
    ], logs_dir)
    calibration = read_json(calibration_path)
    selected_threshold = calibration["selected"]["threshold"]
    print(f"Validation-selected threshold: {selected_threshold:.2f}", flush=True)
    scenarios = [
        ("baseline", args.baseline, args.threshold),
        ("candidate_same_threshold", args.candidate, args.threshold),
    ]
    if selected_threshold != args.threshold:
        scenarios.append(("candidate_validation_threshold", args.candidate, selected_threshold))
    comparisons = {}
    held_out = None
    for name, checkpoint, threshold in scenarios:
        offline_path = logs_dir / f"{name}_offline.json"
        runtime_path = logs_dir / f"{name}_runtime.json"
        run_stage(f"{name}_offline", [
            "src/evaluate.py", "--checkpoint", checkpoint,
            "--feature-root", args.feature_root, "--threshold", threshold,
            "--output", offline_path,
            "--figures-dir", logs_dir / "figures" / name,
            "--history", logs_dir / "training_history.npy"
            if name != "baseline" else LOGS_DIR / "training_history.npy",
        ], logs_dir)
        run_stage(f"{name}_runtime", [
            "scripts/evaluate_runtime_replay.py", "--checkpoint", checkpoint,
            "--feature-root", args.feature_root, "--threshold", threshold,
            "--output", runtime_path,
        ], logs_dir)
        offline, runtime = read_json(offline_path), read_json(runtime_path)
        names = [row["video"] for row in runtime["videos"]]
        if names != [row["video"] for row in offline["event_level"]["videos"]]:
            raise ValueError("Offline and live replay used different test videos")
        if held_out is None:
            held_out = names
        elif names != held_out:
            raise ValueError("Baseline and candidate used different test videos")
        comparisons[name] = {
            "threshold": threshold,
            "window_level": offline["window_level"],
            "classifier_video_max": {
                key: value for key, value in offline["event_level"].items() if key != "videos"
            },
            "runtime_replay": runtime["summary"],
            "detailed_offline_report": str(offline_path.relative_to(PROJECT_ROOT)),
            "detailed_runtime_report": str(runtime_path.relative_to(PROJECT_ROOT)),
        }
        print(f"{name}: {json.dumps(runtime['summary'])}", flush=True)

    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    history = np.load(logs_dir / "training_history.npy", allow_pickle=True).item()
    if active_hash_before != file_sha256(args.baseline):
        raise RuntimeError("The baseline checkpoint changed during comparison")
    if active_scaler_hash_before != file_sha256(MODEL_DIR / "scaler_params.npz"):
        raise RuntimeError("The active scaler changed during comparison")
    report = {
        "protocol": "same video-stratified split, seed 42; validation-only threshold selection; "
                    "both models receive the same YOLO11m pose features",
        "pose_signature": signature,
        "software": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "ultralytics": version("ultralytics"), "numpy": str(np.__version__),
        },
        "source_counts": source_counts,
        "duplicate_source_video_hashes": duplicates,
        "training": {
            "best_epoch": candidate["epoch"], "epochs_run": len(history["train_loss"]),
            "best_val_loss": candidate["val_loss"], "best_val_accuracy": candidate["val_acc"],
            "configuration": candidate["training_config"],
            "video_split_sizes": {
                split: len(candidate["data_split"][split])
                for split in ("train_videos", "validation_videos", "test_videos")
            },
        },
        "artifacts": {
            "baseline": display_path(args.baseline),
            "baseline_sha256": active_hash_before,
            "baseline_training_extractors": {
                name: metadata.get("extractor")
                for name, metadata in baseline.get("extraction_metadata", {}).items()
            },
            "active_scaler_sha256": active_scaler_hash_before,
            "candidate": display_path(args.candidate),
            "candidate_sha256": file_sha256(args.candidate),
            "candidate_activated": False,
        },
        "validation_calibration": calibration["selected"],
        "held_out_test_videos": held_out,
        "comparisons": comparisons,
        "limitations": [
            "Single training seed and a small test sample; observed differences are not significance claims.",
            "Video-disjoint, not subject/room/session-disjoint; augmented windows are correlated.",
            "Classifier video maxima can count pre-fall positives; live replay requires interval overlap.",
            "Replay uses recorded poses and synthetic 15 FPS time; camera/network latency is excluded.",
            "False alerts are counted per short ADL video, not per hour of continuous monitoring.",
            "Validation calibration uses classifier maxima, not the full runtime gates.",
            "Candidate remains separate pending physical robot and continuous-ADL testing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Comparison saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
