"""Extract YOLO11m pose features using the deployment detector's policy.

    python src/extract_features_yolo.py --all --batch-size 8

New features are kept in data/processed_yolo11m/{old,new_july,new_aug}.
The historical YOLOv8n caches remain available for baseline evaluation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    EXPECTED_VIDEO_FPS, YOLO11_FEATURE_ROOT, YOLO_POSE_MODEL_PATH,
    training_dataset_sources,
)
from src.pose_provenance import (  # noqa: E402
    pose_extraction_signature, validate_pose_metadata,
)
from src.temporal_labels import file_sha256  # noqa: E402
from src.yolo_detector import YOLOPoseDetector  # noqa: E402

TOTAL_FEATURES = 52


def landmarks_to_features(landmarks):
    """Match the live classifier's [x,y,z,confidence] selected-joint layout."""
    if landmarks is None:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)
    return np.asarray([
        value for joint in landmarks
        for value in (joint.x, joint.y, joint.z, joint.visibility)
    ], dtype=np.float32)


def extract_landmarks_from_video(video_path, detector, batch_size=8):
    """Bound memory to one batch, retaining one feature row per source frame."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open source video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or abs(fps - EXPECTED_VIDEO_FPS) > 0.1:
            raise ValueError(
                f"{video_path}: {fps} FPS, expected {EXPECTED_VIDEO_FPS}; "
                "resample real motion before extraction"
            )
        info = {
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        detector.reset_tracking()
        rows = []
        batch = []
        detected = 0
        while True:
            ok, frame = cap.read()
            if ok:
                batch.append(frame)
            if batch and (len(batch) == batch_size or not ok):
                for landmarks in detector.detect_batch(batch):
                    rows.append(landmarks_to_features(landmarks))
                    detected += int(landmarks is not None)
                batch = []
            if not ok:
                break
        if not rows:
            raise ValueError(f"Source video has no decoded frames: {video_path}")
        values = np.asarray(rows, dtype=np.float32)
        if values.shape != (len(rows), TOTAL_FEATURES) or not np.isfinite(values).all():
            raise ValueError(f"Invalid pose features: {video_path}")
        return values, {**info, "frames": len(rows), "detected_frames": detected}
    finally:
        cap.release()


def write_metadata(path, metadata):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Windows readers/antivirus can briefly deny atomic replacement. Keep the
    # existing manifest intact and retry; never publish a partially written JSON.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.1 * (2 ** attempt), 1.0))


def process_dataset(raw_dir, processed_dir, detector, batch_size=8, overwrite=False):
    raw_dir, processed_dir = Path(raw_dir).resolve(), Path(processed_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {raw_dir}")
    signature = pose_extraction_signature(
        detector.model_path, detector.conf_threshold, detector.input_size,
    )
    videos = []
    seen = set()
    for video in sorted(raw_dir.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            continue
        parts = [part.lower() for part in video.relative_to(raw_dir).parts]
        classes = [name for name in ("ADL", "Fall")
                   if any(name.lower() in part for part in parts)]
        if len(classes) != 1:
            raise ValueError(f"Ambiguous or absent source class: {video}")
        key = f"{classes[0]}:{video.stem}"
        if key in seen:
            raise ValueError(f"Duplicate feature output stem: {key}")
        seen.add(key)
        videos.append((classes[0], video, key))
    if not videos or {category for category, _, _ in videos} != {"ADL", "Fall"}:
        raise ValueError(f"Dataset must contain both ADL and Fall videos: {raw_dir}")

    metadata_path = processed_dir / "extraction_metadata.json"
    previous = {}
    if metadata_path.exists() and not overwrite:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_pose_metadata(previous, signature, require_completed=False)
        if previous.get("source_root") != str(raw_dir):
            raise ValueError(f"Feature cache belongs to another source: {processed_dir}")
    elif any(processed_dir.glob("*/*.npy")) and not overwrite:
        raise ValueError(f"Existing features have no provenance: {processed_dir}")

    expected_outputs = {processed_dir / category / f"{video.stem}.npy"
                        for category, video, _ in videos}
    stale = set(processed_dir.glob("*/*.npy")) - expected_outputs
    if stale:
        raise ValueError(f"Stale feature files in {processed_dir}: {sorted(map(str, stale))}")
    for category in ("ADL", "Fall"):
        (processed_dir / category).mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 2, **signature, "completed": False,
        "source_root": str(raw_dir), "processed_root": str(processed_dir),
        "videos": {}, "categories": {},
    }
    write_metadata(metadata_path, metadata)
    print(f"Extracting {len(videos)} videos: {raw_dir.name} -> {processed_dir}", flush=True)
    for index, (category, video, key) in enumerate(videos, 1):
        output = processed_dir / category / f"{video.stem}.npy"
        source_hash = file_sha256(video)
        entry = previous.get("videos", {}).get(key)
        cached = (not overwrite and entry and output.exists()
                  and entry.get("source_sha256") == source_hash
                  and entry.get("array_sha256") == file_sha256(output))
        if cached:
            values = np.load(output, allow_pickle=False)
            cached = (values.shape == (entry["frames"], TOTAL_FEATURES)
                      and np.isfinite(values).all())
        if not cached:
            values, info = extract_landmarks_from_video(video, detector, batch_size)
            temporary = output.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, values)
            temporary.replace(output)
            entry = {
                **info, "source_relative_path": str(video.relative_to(raw_dir)),
                "source_sha256": source_hash, "array_sha256": file_sha256(output),
            }
        metadata["videos"][key] = entry
        write_metadata(metadata_path, metadata)
        print(f"[{index}/{len(videos)}] {video.name}: "
              f"{entry['detected_frames']}/{entry['frames']} poses"
              f"{' (cached)' if cached else ''}", flush=True)
    for category in ("ADL", "Fall"):
        entries = [entry for key, entry in metadata["videos"].items()
                   if key.startswith(category + ":")]
        metadata["categories"][category] = {
            "total": len(entries), "success": len(entries),
            "frames": sum(entry["frames"] for entry in entries),
            "detected_frames": sum(entry["detected_frames"] for entry in entries),
        }
    metadata["completed"] = True
    write_metadata(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Extract deployment-compatible YOLO11m features")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--all", action="store_true", help="Extract all three local datasets")
    sources.add_argument("--dataset-dir", type=Path, help="Custom raw dataset")
    parser.add_argument("--processed-dir", type=Path, help="Custom output with --dataset-dir")
    parser.add_argument("--feature-root", type=Path, default=YOLO11_FEATURE_ROOT)
    parser.add_argument("--model", type=Path, default=YOLO_POSE_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.processed_dir and not args.dataset_dir:
        parser.error("--processed-dir requires --dataset-dir")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLOPoseDetector(model_path=args.model, device=device)
    if args.dataset_dir:
        output = args.processed_dir or args.feature_root / args.dataset_dir.name
        process_dataset(args.dataset_dir, output, detector, args.batch_size, args.overwrite)
    else:
        for source in training_dataset_sources(args.feature_root):
            process_dataset(source["raw_root"], source["processed_adl_dir"].parent,
                            detector, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
