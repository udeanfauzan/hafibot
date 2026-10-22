"""Replay held-out cached poses through the complete live fall state machine.

Uses synthetic 15 FPS time, posture gates, reacquisition, and alert latching.
No camera, audio, motor, or server connection is needed.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
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
from src.pose_provenance import pose_extraction_signature, validate_pose_metadata  # noqa: E402
from src.runtime_candidate import make_runtime_detector, RUNTIME_PROFILES  # noqa: E402
from src.temporal_labels import file_sha256  # noqa: E402


def summarize_events(details):
    tn = sum(row["label"] == 0 and not row["detected"] for row in details)
    fp = sum(row["label"] == 0 and row["detected"] for row in details)
    fn = sum(row["label"] == 1 and not row["detected"] for row in details)
    tp = sum(row["label"] == 1 and row["detected"] for row in details)
    delays = [row["detection_delay_sec"] for row in details
              if row.get("detection_delay_sec") is not None]
    onset_tn = sum(row["label"] == 0 and not row.get("onset_detected", row["detected"]) for row in details)
    onset_fp = sum(row["label"] == 0 and row.get("onset_detected", row["detected"]) for row in details)
    onset_fn = sum(row["label"] == 1 and not row.get("onset_detected", row["detected"]) for row in details)
    onset_tp = sum(row["label"] == 1 and row.get("onset_detected", row["detected"]) for row in details)
    exposure = sum(row.get("duration_sec", 0.0) for row in details if row["label"] == 0)
    false_episodes = sum(row.get("alert_episodes", int(row["detected"])) for row in details if row["label"] == 0)
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
        "onset_confusion_matrix": [[onset_tn, onset_fp], [onset_fn, onset_tp]],
        "onset_fall_recall": onset_tp / (onset_tp + onset_fn) if onset_tp + onset_fn else 0.0,
        "false_adl_alert_episodes": false_episodes,
        "adl_exposure_sec": exposure,
        "false_alerts_per_hour_short_clip_proxy": false_episodes * 3600 / exposure if exposure else None,
        "late_only_fall_videos": sum(row["label"] == 1 and row.get("late_only", False) for row in details),
        "pre_fall_alert_episodes": sum(row.get("pre_fall_alert_episodes", 0) for row in details),
    }


def load_replay_records(checkpoint, feature_root=None, require_provenance=False):
    feature_root = feature_root or checkpoint.get("training_config", {}).get("feature_root")
    records = []
    signature = pose_extraction_signature() if require_provenance else None
    for source in training_dataset_sources(feature_root):
        annotations = load_fall_annotations(source["fall_annotations_path"])
        metadata_path = source["processed_adl_dir"].parent / "extraction_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if require_provenance:
            validate_pose_metadata(metadata, signature)
        for label, field in ((0, "processed_adl_dir"), (1, "processed_fall_dir")):
            feature_files = sorted(source[field].glob("*.npy"))
            if require_provenance:
                category = 'Fall' if label else 'ADL'
                expected = {key.split(':', 1)[1] for key in metadata.get('videos', {}) if key.startswith(category + ':')}
                if not feature_files or {p.stem for p in feature_files} != expected:
                    raise ValueError(f"Incomplete replay cache: {source[field]}")
            for feature_file in feature_files:
                entry = metadata.get("videos", {}).get(f"{'Fall' if label else 'ADL'}:{feature_file.stem}", {})
                if require_provenance:
                    if not entry or file_sha256(feature_file) != entry.get("array_sha256"):
                        raise ValueError(f"Unverified replay features: {feature_file}")
                    if abs(entry.get("fps", 0) - EXPECTED_VIDEO_FPS) > 0.1:
                        raise ValueError("Replay needs source-time resampling, not a changed FPS label")
                fps = float(entry.get("fps", EXPECTED_VIDEO_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Replay FPS must be finite and positive")
                records.append({
                    "video": f"{source['name']}:{feature_file.stem}",
                    "label": label, "path": feature_file,
                    "annotation": annotations.get(feature_file.stem) if label else None,
                    "fps": fps,
                    "frame_size": (entry["width"], entry["height"]) if "width" in entry and "height" in entry else None,
                    "array_sha256": entry.get("array_sha256"),
                    "source_sha256": entry.get("source_sha256"),
                    "expected_frames": entry.get("frames"),
                })
    return records, feature_root


def select_replay_indices(records, checkpoint, split_name="test"):
    if split_name not in ("validation", "test"):
        raise ValueError("Replay split must be validation or test")
    split = checkpoint_split_config(checkpoint)
    subjects = None
    if split["strategy"] == "subject":
        subjects = resolve_subject_ids(
            [record["video"] for record in records], load_subject_metadata(),
        )
    _, val_indices, test_indices = split_video_indices(
        [record["label"] for record in records], video_subject_ids=subjects,
        split_strategy=split["strategy"], train_subject_ids=split["train_subject_ids"],
        val_subject_ids=split["val_subject_ids"], test_subject_ids=split["test_subject_ids"],
    )
    indices = val_indices if split_name == "validation" else test_indices
    names = [records[index]["video"] for index in indices]
    recorded = checkpoint.get("data_split", {}).get(f"{split_name}_videos")
    if recorded is not None and names != recorded:
        raise ValueError("Replay dataset does not reproduce the checkpoint's split manifest")
    return indices


class PredictionMemo:
    """Reuse only identical model windows, never cache alert decisions.

Replays all gates/latches at every threshold. A window hash mismatch fails
instead of assuming runtime history is independent of the alert threshold.
"""
    def __init__(self):
        self.entries = {}
        self.model_calls = 0

    def predict(self, key, sequence, predictor):
        fingerprint = hashlib.sha256(sequence.tobytes()).digest()
        if key in self.entries:
            stored_hash, result = self.entries[key]
            if stored_hash != fingerprint:
                raise ValueError("Threshold changed model inputs; prediction cache is invalid")
            return result
        result = predictor(sequence)
        self.entries[key] = fingerprint, result
        self.model_calls += 1
        return result


def replay_records(detector, records, indices, profile="legacy_v1", memo=None, quiet=False):
    details = []
    for index in indices:
        record = records[index]
        features = np.load(record["path"], allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 52 or not np.isfinite(features).all() or not len(features):
            raise ValueError(f"Replay requires selected YOLO features: {record['path']}")
        if record.get("expected_frames") is not None and len(features) != record["expected_frames"]:
            raise ValueError("Replay frame count differs from extraction metadata")
        if record["label"]:
            if record["annotation"] is None:
                raise ValueError(f"Missing Fall annotation: {record['video']}")
            start, end = validate_fall_annotation(record["video"], record["annotation"], len(features))
        else:
            start, end = 0, len(features) - 1
        if quiet:
            with redirect_stdout(io.StringIO()):
                detector.reset_buffer()
        else:
            detector.reset_buffer()
        alerts = []
        onsets = []
        evaluated = gated = missing = reprimed = resets = preserved = 0
        was_alert = False
        original_predict = getattr(detector, "_predict", None)
        if memo is not None:
            detector._predict = lambda sequence: memo.predict((record["video"], frame_index), sequence, original_predict)
        try:
            for frame_index, row in enumerate(features):
                landmarks = [PoseKeypoint(*joint) for joint in row.reshape(13, 4)] if np.any(row) else None
                with mock.patch("src.inference.time.monotonic", return_value=frame_index / record["fps"]):
                    if profile == "legacy_v1":
                        result = detector.process_frame(landmarks)
                    else:
                        result = detector.process_frame(landmarks, frame_size=record["frame_size"])
                evaluated += int(result["model_evaluated"])
                gated += int(result["posture_gated"])
                missing += int(not result["pose_detected"])
                reprimed += int(result["buffer_reprimed"])
                resets += int(result.get("buffer_reset", False))
                preserved += int(result.get("short_gap_preserved", False))
                if result["is_fall"]:
                    alerts.append(frame_index)
                    if not was_alert:
                        onsets.append(frame_index)
                was_alert = bool(result["is_fall"])
        finally:
            if memo is not None:
                detector._predict = original_predict
        relevant_alerts = [frame for frame in alerts if start <= frame <= end]
        relevant_onsets = [frame for frame in onsets if start <= frame <= end]
        detail = {
            "video": record["video"], "label": record["label"],
            "detected": bool(relevant_alerts),
            "first_alert_frame": alerts[0] if alerts else None,
            "first_fall_interval_alert_frame": relevant_alerts[0] if relevant_alerts and record["label"] else None,
            "detection_delay_sec": (relevant_alerts[0] - start) / record["fps"]
            if relevant_alerts and record["label"] else None,
            "pre_fall_alert_frames": sum(frame < start for frame in alerts) if record["label"] else 0,
            "model_evaluated_frames": evaluated, "posture_gated_frames": gated,
            "missing_pose_frames": missing, "buffer_reprimed_frames": reprimed,
            "onset_detected": bool(relevant_onsets), "alert_onset_frames": onsets,
            "onset_detection_delay_sec": (relevant_onsets[0] - start) / record["fps"]
            if relevant_onsets and record["label"] else None,
            "alert_episodes": len(onsets), "pre_fall_alert_episodes": sum(frame < start for frame in onsets) if record["label"] else 0,
            "late_only": bool(record["label"] and not relevant_onsets and any(frame > end for frame in onsets)),
            "duration_sec": len(features) / record["fps"],
            "buffer_resets": resets, "short_gaps_preserved": preserved,
        }
        details.append(detail)
        if not quiet:
            print(f"[{len(details)}/{len(indices)}] {record['video']}: "
                  f"label={record['label']} detected={detail['detected']}", flush=True)
    return details


def replay_checkpoint(checkpoint_path, feature_root=None,
                      threshold=FALL_CONFIDENCE_THRESHOLD, profile="legacy_v1", split_name="test"):
    if profile == "legacy_v1":
        detector = FallDetector(checkpoint_path=checkpoint_path, confidence_threshold=threshold)
    else:
        detector = make_runtime_detector(profile, checkpoint_path=checkpoint_path, confidence_threshold=threshold)
    detector.mute = True
    detector._trigger_alert = lambda: None
    records, feature_root = load_replay_records(detector.checkpoint, feature_root, require_provenance=profile != "legacy_v1")
    indices = select_replay_indices(records, detector.checkpoint, split_name)
    details = replay_records(detector, records, indices, profile)
    return {
        "checkpoint": str(checkpoint_path), "threshold": float(threshold),
        "feature_root": str(feature_root) if feature_root else None,
        "runtime_profile": profile, "split": split_name,
        "scope": "held-out video pose replay with live gates/latches and synthetic source time; "
                 "Fall detection must overlap the annotated Fall interval; network/capture latency excluded",
        "summary": summarize_events(details), "videos": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt")
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--threshold", type=float, default=FALL_CONFIDENCE_THRESHOLD)
    parser.add_argument("--profile", choices=RUNTIME_PROFILES, default="legacy_v1")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay_checkpoint(args.checkpoint, args.feature_root, args.threshold, args.profile, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
