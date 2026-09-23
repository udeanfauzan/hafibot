"""Leakage-safe utilities for mining difficult ADL windows.

Hard negatives are windows whose temporal label is ADL, whose source video is
also ADL, and whose Fall probability exceeds a configurable boundary. Selection
is capped and temporally de-duplicated per video so one false-positive burst
cannot dominate fine-tuning.
"""

from collections import defaultdict

import numpy as np


def collect_windows_with_provenance(
    video_windows, video_labels, video_event_labels, video_indices
):
    """Flatten selected videos while retaining source video/window indices."""
    values = []
    labels = []
    event_labels = []
    source_videos = []
    source_windows = []
    allowed_videos = {int(index) for index in video_indices}

    for video_index in video_indices:
        video_index = int(video_index)
        windows = video_windows[video_index]
        window_labels = video_labels[video_index]
        if len(windows) != len(window_labels):
            raise ValueError(
                "Jumlah window dan label berbeda untuk video {}".format(
                    video_index
                )
            )
        for window_index, (window, label) in enumerate(
            zip(windows, window_labels)
        ):
            values.append(window)
            labels.append(label)
            event_labels.append(video_event_labels[video_index])
            source_videos.append(video_index)
            source_windows.append(window_index)

    if not values:
        raise ValueError("Split tidak memiliki window eligible")
    if not set(source_videos).issubset(allowed_videos):
        raise AssertionError("Window di luar scope video training")
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(event_labels, dtype=np.int64),
        np.asarray(source_videos, dtype=np.int64),
        np.asarray(source_windows, dtype=np.int64),
    )


def select_hard_negative_indices(
    fall_probabilities,
    window_labels,
    source_event_labels,
    source_video_indices,
    source_window_indices,
    min_probability=0.50,
    max_per_video=12,
    min_gap_windows=3,
):
    """Select high-scoring ADL windows with per-video temporal NMS.

    Only windows from source videos labelled ADL are eligible. This avoids
    mining settled or pre-fall sections from Fall recordings whose context can
    be less reliable. Results are returned in descending probability order.
    """
    arrays = [
        np.asarray(fall_probabilities, dtype=np.float64),
        np.asarray(window_labels, dtype=np.int64),
        np.asarray(source_event_labels, dtype=np.int64),
        np.asarray(source_video_indices, dtype=np.int64),
        np.asarray(source_window_indices, dtype=np.int64),
    ]
    sample_count = len(arrays[0])
    if any(array.shape != (sample_count,) for array in arrays):
        raise ValueError("Semua input mining harus berupa vector dengan shape sama")
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability harus berada pada rentang 0..1")
    if max_per_video <= 0:
        raise ValueError("max_per_video harus lebih besar dari nol")
    if min_gap_windows < 0:
        raise ValueError("min_gap_windows tidak boleh negatif")

    probabilities, labels, event_labels, videos, window_indices = arrays
    eligible = np.flatnonzero(
        (labels == 0)
        & (event_labels == 0)
        & (probabilities >= min_probability)
    )
    candidates_by_video = defaultdict(list)
    for sample_index in eligible:
        candidates_by_video[int(videos[sample_index])].append(int(sample_index))

    selected = []
    for video_index in sorted(candidates_by_video):
        candidates = sorted(
            candidates_by_video[video_index],
            key=lambda index: (-probabilities[index], window_indices[index]),
        )
        kept_window_indices = []
        for sample_index in candidates:
            window_index = int(window_indices[sample_index])
            if any(
                abs(window_index - kept) < min_gap_windows
                for kept in kept_window_indices
            ):
                continue
            selected.append(sample_index)
            kept_window_indices.append(window_index)
            if len(kept_window_indices) >= max_per_video:
                break

    return np.asarray(
        sorted(selected, key=lambda index: -probabilities[index]),
        dtype=np.int64,
    )


def build_sample_weights(sample_count, hard_negative_indices, multiplier=4.0):
    """Return sampler weights that emphasize selected hard negatives."""
    if sample_count <= 0:
        raise ValueError("sample_count harus lebih besar dari nol")
    if multiplier < 1.0:
        raise ValueError("Hard-negative multiplier minimal 1.0")
    indices = np.asarray(hard_negative_indices, dtype=np.int64)
    if len(indices) and (indices.min() < 0 or indices.max() >= sample_count):
        raise ValueError("Hard-negative index berada di luar dataset")
    weights = np.ones(sample_count, dtype=np.float64)
    weights[indices] = float(multiplier)
    return weights


def validate_mining_subject_scope(
    selected_indices, source_video_indices, subjects, allowed_subjects
):
    """Reject any selected window not owned by an allowed training subject."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    videos = np.asarray(source_video_indices, dtype=np.int64)
    allowed = set(allowed_subjects)
    observed = {subjects[int(videos[index])] for index in selected}
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(
            "Hard negative bocor dari subject non-training: {}".format(
                sorted(unexpected)
            )
        )
    return observed
