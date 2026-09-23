"""Utilities for frame-level Fall annotations and window-level labels."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


AMBIGUOUS_LABEL = -1


def file_sha256(path: Path) -> str:
    """Return a stable fingerprint for an annotation file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_start_indices(
    num_frames: int, window_size: int, stride: int
) -> List[int]:
    """Return starts matching the project's sliding-window edge behavior."""
    if num_frames < 0:
        raise ValueError("num_frames tidak boleh negatif")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size dan stride harus lebih besar dari nol")
    if num_frames <= window_size:
        return [0]

    starts = list(range(0, num_frames - window_size + 1, stride))
    last_start = num_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def window_end_pose_mask(
    pose_present: np.ndarray,
    starts: Iterable[int],
    window_size: int,
) -> np.ndarray:
    """Return whether each window ends on a frame eligible for inference."""
    pose_present = np.asarray(pose_present, dtype=bool)
    if pose_present.ndim != 1 or len(pose_present) == 0:
        raise ValueError("pose_present harus array 1D yang tidak kosong")

    result = []
    for window_start in starts:
        window_end = min(int(window_start) + window_size, len(pose_present))
        result.append(bool(pose_present[window_end - 1]))
    return np.asarray(result, dtype=bool)


def load_fall_annotations(path: Path) -> Dict[str, dict]:
    """Load annotations keyed by processed-video stem."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Fall annotation file tidak ditemukan: {}. Jalankan "
            "python scripts/annotate_falls.py".format(path)
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != 1:
        raise ValueError(
            "Schema fall annotation harus versi 1, ditemukan: {}".format(
                payload.get("schema_version")
            )
        )
    videos = payload.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("Fall annotation harus memiliki object 'videos'")

    normalized = {}
    for key, annotation in videos.items():
        stem = Path(str(key)).stem
        if stem in normalized:
            raise ValueError("Duplikat annotation untuk video stem: " + stem)
        if not isinstance(annotation, dict):
            raise ValueError("Annotation {} harus berupa object".format(key))
        normalized[stem] = annotation
    return normalized


def validate_fall_annotation(
    video_stem: str, annotation: dict, num_frames: int
) -> Tuple[int, int]:
    """Validate and return an inclusive [start_frame, end_frame] interval."""
    try:
        start_frame = int(annotation["fall_start_frame"])
        end_frame = int(annotation["fall_end_frame"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Annotation {} membutuhkan fall_start_frame dan fall_end_frame"
            .format(video_stem)
        )

    if start_frame < 0 or end_frame < start_frame or end_frame >= num_frames:
        raise ValueError(
            "Interval {} tidak valid: {}..{} untuk {} frames".format(
                video_stem, start_frame, end_frame, num_frames
            )
        )

    source_frame_count = annotation.get("source_frame_count")
    if (
        source_frame_count is not None
        and int(source_frame_count) != int(num_frames)
    ):
        raise ValueError(
            "Frame count {} berubah: annotation={}, processed={}. "
            "Periksa video/annotation sebelum training.".format(
                video_stem, source_frame_count, num_frames
            )
        )
    return start_frame, end_frame


def label_window_starts(
    num_frames: int,
    starts: Iterable[int],
    window_size: int,
    annotation: dict,
    min_fall_overlap_frames: int,
    pose_present: np.ndarray = None,
) -> Tuple[np.ndarray, dict]:
    """Label windows from a frame interval.

    A window is Fall when it contains at least ``min_fall_overlap_frames``
    annotated Fall frames with a detected pose, ADL when it does not overlap
    the annotation, and ambiguous otherwise. The Fall interval is inclusive at
    both ends. When ``pose_present`` is omitted, every frame is treated as
    visible for backwards compatibility.
    """
    if min_fall_overlap_frames <= 0 or min_fall_overlap_frames > window_size:
        raise ValueError("min_fall_overlap_frames tidak valid")

    video_stem = str(annotation.get("video_stem", "unknown"))
    start_frame, end_frame = validate_fall_annotation(
        video_stem, annotation, num_frames
    )
    frame_labels = np.zeros(num_frames, dtype=np.uint8)
    frame_labels[start_frame:end_frame + 1] = 1
    if pose_present is None:
        pose_present = np.ones(num_frames, dtype=bool)
    else:
        pose_present = np.asarray(pose_present, dtype=bool)
        if pose_present.shape != (num_frames,):
            raise ValueError(
                "pose_present shape tidak cocok: {} expected ({},)".format(
                    pose_present.shape, num_frames
                )
            )

    labels = []
    overlaps = []
    visible_overlaps = []
    for window_start in starts:
        window_end = min(int(window_start) + window_size, num_frames)
        overlap = int(frame_labels[int(window_start):window_end].sum())
        visible_overlap = int(np.sum(
            frame_labels[int(window_start):window_end]
            * pose_present[int(window_start):window_end]
        ))
        overlaps.append(overlap)
        visible_overlaps.append(visible_overlap)
        if overlap == 0:
            labels.append(0)
        elif visible_overlap >= min_fall_overlap_frames:
            labels.append(1)
        else:
            labels.append(AMBIGUOUS_LABEL)

    labels_array = np.asarray(labels, dtype=np.int64)
    details = {
        "fall_start_frame": start_frame,
        "fall_end_frame": end_frame,
        "fall_frames": end_frame - start_frame + 1,
        "adl_windows": int(np.sum(labels_array == 0)),
        "fall_windows": int(np.sum(labels_array == 1)),
        "ambiguous_windows": int(np.sum(labels_array == AMBIGUOUS_LABEL)),
        "window_fall_overlaps": overlaps,
        "window_visible_fall_overlaps": visible_overlaps,
    }
    return labels_array, details
