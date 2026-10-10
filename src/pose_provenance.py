"""Provenance checks for model-specific pose feature caches."""

from pathlib import Path

from src.config import (
    EXPECTED_VIDEO_FPS, YOLO_POSE_CONFIDENCE, YOLO_POSE_INPUT_SIZE,
    YOLO_POSE_MODEL_PATH,
)
from src.temporal_labels import file_sha256


def pose_extraction_signature(model_path=YOLO_POSE_MODEL_PATH,
                              confidence=YOLO_POSE_CONFIDENCE,
                              input_size=YOLO_POSE_INPUT_SIZE):
    """Bind cached features to weights and the shared live decoding policy."""
    model_path = Path(model_path)
    return {
        "extractor": model_path.stem,
        "model_sha256": file_sha256(model_path),
        "confidence_threshold": float(confidence),
        "input_size": list(input_size),
        "expected_video_fps": float(EXPECTED_VIDEO_FPS),
        "pose_policy_sha256": file_sha256(Path(__file__).with_name("yolo_detector.py")),
        "features_per_frame": 52,
        "selected_joints": 13,
    }


def validate_pose_metadata(metadata, signature, require_completed=True):
    """Never silently reuse v8 features or another model's detection policy."""
    if metadata.get("schema_version") != 2:
        raise ValueError("Pose cache metadata must use schema version 2; re-extract features")
    for key, value in signature.items():
        if metadata.get(key) != value:
            raise ValueError(f"Pose cache mismatch for {key}; use a new cache or --overwrite")
    if require_completed and not metadata.get("completed"):
        raise ValueError("Pose feature extraction is incomplete")
