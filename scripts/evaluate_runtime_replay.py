"""Replay held-out cached poses through the complete live fall state machine.

Uses synthetic 15 FPS time, posture gates, reacquisition, and alert latching.
No camera, audio, motor, or server connection is needed.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    EXPECTED_VIDEO_FPS, FALL_CONFIDENCE_THRESHOLD, MODEL_DIR,
    training_dataset_sources,
)
from src.dataset import split_video_indices  # noqa: E402
from src.evaluate import checkpoint_split_config  # noqa: E402
from src.inference import FallDetector  # noqa: E402
from src.subject_metadata import load_subject_metadata, resolve_subject_ids  # noqa: E402
from src.temporal_labels import load_fall_annotations, validate_fall_annotation  # noqa: E402
from src.yolo_detector import PoseKeypoint  # noqa: E402


def summarize_events(details):
    tn = sum(row["label"] == 0 and not row["detected"] for row in details)
    fp = sum(row["label"] == 0 and row["detected"] for row in details)
    fn = sum(row["label"] == 1 and not row["detected"] for row in details)
    tp = sum(row["label"] == 1 and row["detected"] for row in details)
    delays = [row["detection_delay_sec"] for row in details
              if row.get("detection_delay_sec") is not None]
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "fall_recall": tp / (tp + fn) if tp + fn else 0.0,
        "fall_precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_adl_videos": fp,
        "fall_videos": tp + fn,
        "detected_falls": tp,
        "median_detection_delay_sec": float(np.median(delays)) if delays else None,
        "p95_detection_delay_sec": float(np.percentile(delays, 95)) if delays else None,
        "pre_fall_alert_videos": sum(bool(row.get("pre_fall_alert_frames")) for row in details),
    }


def replay_checkpoint(checkpoint_path, feature_root=None,
                      threshold=FALL_CONFIDENCE_THRESHOLD):
    detector = FallDetector(checkpoint_path=checkpoint_path, confidence_threshold=threshold)
    detector.mute = True
    detector._trigger_alert = lambda: None
    checkpoint = detector.checkpoint
    feature_root = feature_root or checkpoint.get("training_config", {}).get("feature_root")
    records = []
    for source in training_dataset_sources(feature_root):
        annotations = load_fall_annotations(source["fall_annotations_path"])
        for label, field in ((0, "processed_adl_dir"), (1, "processed_fall_dir")):
            for feature_file in sorted(source[field].glob("*.npy")):
                records.append({
                    "video": f"{source['name']}:{feature_file.stem}",
                    "label": label, "path": feature_file,
                    "annotation": annotations.get(feature_file.stem) if label else None,
                })
    split = checkpoint_split_config(checkpoint)
    subjects = None
    if split["strategy"] == "subject":
        subjects = resolve_subject_ids(
            [record["video"] for record in records], load_subject_metadata(),
        )
    _, _, test_indices = split_video_indices(
        [record["label"] for record in records], video_subject_ids=subjects,
        split_strategy=split["strategy"], train_subject_ids=split["train_subject_ids"],
        val_subject_ids=split["val_subject_ids"], test_subject_ids=split["test_subject_ids"],
    )
    names = [records[index]["video"] for index in test_indices]
    recorded_test = checkpoint.get("data_split", {}).get("test_videos")
    if recorded_test is not None and names != recorded_test:
        raise ValueError("Replay dataset does not reproduce the checkpoint's held-out videos")
    details = []
    for index in test_indices:
        record = records[index]
        features = np.load(record["path"], allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 52:
            raise ValueError(f"Replay requires selected YOLO features: {record['path']}")
        if record["label"]:
            if record["annotation"] is None:
                raise ValueError(f"Missing Fall annotation: {record['video']}")
            start, end = validate_fall_annotation(record["video"], record["annotation"], len(features))
        else:
            start, end = 0, len(features) - 1
        detector.reset_buffer()
        alerts = []
        evaluated = gated = missing = reprimed = 0
        for frame_index, row in enumerate(features):
            landmarks = [PoseKeypoint(*joint) for joint in row.reshape(13, 4)] if np.any(row) else None
            with mock.patch("src.inference.time.monotonic", return_value=frame_index / EXPECTED_VIDEO_FPS):
                result = detector.process_frame(landmarks)
            evaluated += int(result["model_evaluated"])
            gated += int(result["posture_gated"])
            missing += int(not result["pose_detected"])
            reprimed += int(result["buffer_reprimed"])
            if result["is_fall"]:
                alerts.append(frame_index)
        relevant_alerts = [frame for frame in alerts if start <= frame <= end]
        detail = {
            "video": record["video"], "label": record["label"],
            "detected": bool(relevant_alerts),
            "first_alert_frame": alerts[0] if alerts else None,
            "first_fall_interval_alert_frame": relevant_alerts[0] if relevant_alerts and record["label"] else None,
            "detection_delay_sec": (relevant_alerts[0] - start) / EXPECTED_VIDEO_FPS
            if relevant_alerts and record["label"] else None,
            "pre_fall_alert_frames": sum(frame < start for frame in alerts) if record["label"] else 0,
            "model_evaluated_frames": evaluated, "posture_gated_frames": gated,
            "missing_pose_frames": missing, "buffer_reprimed_frames": reprimed,
        }
        details.append(detail)
        print(f"[{len(details)}/{len(test_indices)}] {record['video']}: "
              f"label={record['label']} detected={detail['detected']}", flush=True)
    return {
        "checkpoint": str(checkpoint_path), "threshold": float(threshold),
        "feature_root": str(feature_root) if feature_root else None,
        "scope": "held-out video pose replay with live gates/latches and synthetic source time; "
                 "Fall detection must overlap the annotated Fall interval; network/capture latency excluded",
        "summary": summarize_events(details), "videos": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt")
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--threshold", type=float, default=FALL_CONFIDENCE_THRESHOLD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay_checkpoint(args.checkpoint, args.feature_root, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
