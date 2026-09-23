"""
dataset.py - Load, preprocess, augmentasi, dan split data untuk training.

Pipeline:
    - Checkpoint-compatible legacy and experimental body-relative features
    - Reproducible legacy plus physically constrained augmentation profiles
    - Velocity/acceleration recomputation after raw-space augmentation
    - Grouped split per video or subject (prevents data leakage)

Penggunaan:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROCESSED_ADL_DIR, PROCESSED_FALL_DIR,
    FALL_ANNOTATIONS_PATH, FALL_MIN_OVERLAP_FRAMES,
    TRAINING_DATASETS, SUBJECT_METADATA_PATH,
    LABEL_MAP, MP_TOTAL_FEATURES, MP_NUM_LANDMARKS, MP_FEATURES_PER_LANDMARK,
    SELECTED_LANDMARKS, NUM_SELECTED_LANDMARKS, SELECTED_FEATURES,
    SEQUENCE_LENGTH, STRIDE, INPUT_FEATURES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    BATCH_SIZE, RANDOM_SEED, SPLIT_STRATEGY,
    SUBJECT_TRAIN_IDS, SUBJECT_VAL_IDS, SUBJECT_TEST_IDS,
    USE_VELOCITY, USE_ACCELERATION, VELOCITY_FEATURES,
    FEATURE_PIPELINE, LEGACY_FEATURE_PIPELINE, HYBRID_FEATURE_PIPELINE,
    HYBRID_GLOBAL_FEATURES, BODY_RELATIVE_SCALE_MIN,
    BODY_RELATIVE_COORD_CLIP,
    USE_AUGMENTATION, AUG_JITTER_STD, AUG_SCALE_RANGE,
    AUG_TIME_WARP_PROB, AUG_DROPOUT_PROB,
    AUG_HORIZONTAL_FLIP_PROB, AUG_CAMERA_MOTION_PROB,
    AUG_CAMERA_TRANSLATE, AUG_CAMERA_ROTATE_DEG,
    AUG_CONTIGUOUS_OCCLUSION_PROB,
    AUG_CONTIGUOUS_OCCLUSION_MAX_FRAMES, AUGMENTATION_PROFILE,
)
from src.temporal_labels import (
    AMBIGUOUS_LABEL,
    label_window_starts,
    load_fall_annotations,
    window_end_pose_mask,
    window_start_indices,
)
from src.subject_metadata import load_subject_metadata, resolve_subject_ids


# ============================================================
# LANDMARK FILTERING
# ============================================================
def filter_landmarks(data: np.ndarray) -> np.ndarray:
    """
    Filter 33 landmarks menjadi 13 landmark kunci untuk fall detection.

    Menghilangkan landmark wajah (mata, telinga, mulut), jari tangan,
    dan jari kaki yang menambah noise tanpa manfaat untuk deteksi jatuh.

    Args:
        data: shape (num_frames, 132)  [33 landmarks × 4(x,y,z,vis)]

    Returns:
        filtered: shape (num_frames, 52)  [13 landmarks × 4(x,y,z,vis)]
    """
    # Buat indices untuk mengambil 4 fitur (x,y,z,vis) dari setiap landmark terpilih
    if data.shape[1] == SELECTED_FEATURES:
        return data.copy()

    selected_indices = []
    for lm_idx in SELECTED_LANDMARKS:
        base = lm_idx * MP_FEATURES_PER_LANDMARK
        selected_indices.extend([base, base + 1, base + 2, base + 3])

    return data[:, selected_indices]


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_velocity_features(data: np.ndarray) -> np.ndarray:
    """
    Tambahkan fitur velocity (kecepatan) ke data.

    Velocity = perbedaan koordinat antar frame berturutan.
    Hanya untuk x, y, z (bukan visibility).

    Args:
        data: shape (num_frames, 52)  [13 selected landmarks × 4(x,y,z,vis)]

    Returns:
        data dengan velocity: shape (num_frames, 52 + 39)
    """
    num_frames, num_features = data.shape
    num_landmarks = num_features // MP_FEATURES_PER_LANDMARK

    # Ekstrak hanya x, y, z (tanpa visibility) untuk setiap landmark
    # Layout: [x0, y0, z0, vis0, x1, y1, z1, vis1, ...]
    xyz_indices = []
    for i in range(num_landmarks):
        base = i * MP_FEATURES_PER_LANDMARK
        xyz_indices.extend([base, base + 1, base + 2])  # x, y, z

    xyz = data[:, xyz_indices]  # (num_frames, 39)

    # Velocity = frame[t] - frame[t-1]
    velocity = np.zeros_like(xyz)
    velocity[1:] = xyz[1:] - xyz[:-1]  # Frame pertama = 0

    return np.concatenate([data, velocity], axis=1)


def add_acceleration_features(data_with_velocity: np.ndarray) -> np.ndarray:
    """
    Tambahkan fitur acceleration (percepatan) ke data yang sudah punya velocity.

    Acceleration = perbedaan velocity antar frame.

    Args:
        data_with_velocity: shape (num_frames, 52 + 39)

    Returns:
        data dengan acceleration: shape (num_frames, 52 + 39 + 39)
    """
    # Velocity ada di kolom terakhir
    velocity = data_with_velocity[:, -VELOCITY_FEATURES:]

    # Acceleration = velocity[t] - velocity[t-1]
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]

    return np.concatenate([data_with_velocity, acceleration], axis=1)


def engineer_features(data: np.ndarray) -> np.ndarray:
    """
    Apply landmark filtering + feature engineering berdasarkan config.

    Pipeline: raw (132) → filter (52) → + velocity (91) → + acceleration (130)
    """
    # Step 1: Filter ke 13 landmark kunci
    data = filter_landmarks(data)

    return engineer_selected_features(data)


def engineer_selected_features(data: np.ndarray) -> np.ndarray:
    """Tambah derivative features pada data 52 fitur yang sudah difilter."""
    # Harus dipanggil SETELAH window dibuat. Runtime menghitung derivative di
    # dalam buffer 30 frame, sehingga training wajib memakai boundary yang sama.
    if USE_VELOCITY:
        data = add_velocity_features(data)
    if USE_ACCELERATION:
        data = add_acceleration_features(data)
    return data


def feature_dimensions(feature_pipeline=FEATURE_PIPELINE):
    """Return (base, derivative, total) dimensions for a feature pipeline."""
    if feature_pipeline == LEGACY_FEATURE_PIPELINE:
        base_features = SELECTED_FEATURES
        derivative_features = NUM_SELECTED_LANDMARKS * 3
    elif feature_pipeline == HYBRID_FEATURE_PIPELINE:
        base_features = SELECTED_FEATURES + HYBRID_GLOBAL_FEATURES
        derivative_features = NUM_SELECTED_LANDMARKS * 3 + HYBRID_GLOBAL_FEATURES
    else:
        raise ValueError("Feature pipeline tidak dikenal: " + str(feature_pipeline))

    total_features = base_features
    if USE_VELOCITY:
        total_features += derivative_features
    if USE_ACCELERATION:
        total_features += derivative_features
    return base_features, derivative_features, total_features


def pose_frame_mask(selected_data: np.ndarray) -> np.ndarray:
    """Identify frames that contain a selected MediaPipe pose."""
    if selected_data.shape[1] < SELECTED_FEATURES:
        raise ValueError("Selected landmark data harus memiliki minimal 52 fitur")
    return np.any(selected_data[:, :SELECTED_FEATURES] != 0.0, axis=1)


def body_relative_features(selected_data: np.ndarray) -> np.ndarray:
    """Create body-relative landmarks plus seven compact global pose cues."""
    if selected_data.ndim != 2 or selected_data.shape[1] != SELECTED_FEATURES:
        raise ValueError(
            "Expected selected pose shape (frames, {}), got {}".format(
                SELECTED_FEATURES, selected_data.shape
            )
        )

    frames = selected_data.reshape(-1, NUM_SELECTED_LANDMARKS, 4)
    result = np.zeros(
        (len(frames), SELECTED_FEATURES + HYBRID_GLOBAL_FEATURES),
        dtype=np.float32,
    )
    present = pose_frame_mask(selected_data)
    if not np.any(present):
        return result

    visible_frames = frames[present].astype(np.float32, copy=True)
    xyz = visible_frames[:, :, :3]
    visibility = visible_frames[:, :, 3]
    hip_center = (xyz[:, 7] + xyz[:, 8]) * 0.5
    shoulder_center = (xyz[:, 1] + xyz[:, 2]) * 0.5
    torso_scale = np.linalg.norm(
        shoulder_center[:, :2] - hip_center[:, :2], axis=1
    )
    torso_scale = np.maximum(torso_scale, BODY_RELATIVE_SCALE_MIN)

    relative_xyz = (xyz - hip_center[:, None, :]) / torso_scale[:, None, None]
    relative_xyz = np.clip(
        relative_xyz, -BODY_RELATIVE_COORD_CLIP, BODY_RELATIVE_COORD_CLIP
    )
    relative_landmarks = np.concatenate(
        [relative_xyz, visibility[:, :, None]], axis=2
    ).reshape(-1, SELECTED_FEATURES)

    x_values = xyz[:, :, 0]
    y_values = xyz[:, :, 1]
    bbox_width = np.maximum(x_values.max(axis=1) - x_values.min(axis=1), 0.0)
    bbox_height = np.maximum(y_values.max(axis=1) - y_values.min(axis=1), 0.0)
    bbox_aspect = bbox_width / np.maximum(bbox_height, BODY_RELATIVE_SCALE_MIN)
    globals_ = np.stack(
        [
            hip_center[:, 1], shoulder_center[:, 1],
            (y_values.max(axis=1) + y_values.min(axis=1)) * 0.5,
            bbox_width, bbox_height, np.clip(bbox_aspect, 0.0, 5.0),
            torso_scale,
        ],
        axis=1,
    ).astype(np.float32)
    result[present, :SELECTED_FEATURES] = relative_landmarks
    result[present, SELECTED_FEATURES:] = globals_
    return result


def prepare_base_features(selected_data: np.ndarray, feature_pipeline=FEATURE_PIPELINE):
    """Convert filtered 52-feature poses into a pipeline-specific base."""
    selected_data = np.asarray(selected_data, dtype=np.float32)
    if feature_pipeline == LEGACY_FEATURE_PIPELINE:
        return selected_data.copy()
    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        return body_relative_features(selected_data)
    raise ValueError("Feature pipeline tidak dikenal: " + str(feature_pipeline))


def derivative_signal(base_data: np.ndarray, feature_pipeline=FEATURE_PIPELINE):
    """Return the coordinate/global signals used for temporal derivatives."""
    xyz_indices = []
    for i in range(NUM_SELECTED_LANDMARKS):
        base = i * MP_FEATURES_PER_LANDMARK
        xyz_indices.extend([base, base + 1, base + 2])
    signal = base_data[:, xyz_indices]
    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        signal = np.concatenate([signal, base_data[:, SELECTED_FEATURES:]], axis=1)
    return signal


def engineer_base_features(base_data: np.ndarray, feature_pipeline=FEATURE_PIPELINE):
    """Add velocity/acceleration to an already prepared feature window."""
    base_data = np.asarray(base_data, dtype=np.float32)
    expected_base, derivative_features, expected_total = feature_dimensions(
        feature_pipeline
    )
    if base_data.ndim != 2 or base_data.shape[1] != expected_base:
        raise ValueError(
            "Base shape {} tidak cocok dengan pipeline {} (expected {})".format(
                base_data.shape, feature_pipeline, expected_base
            )
        )

    signal = derivative_signal(base_data, feature_pipeline)
    velocity = np.zeros((len(base_data), derivative_features), dtype=np.float32)
    velocity[1:] = signal[1:] - signal[:-1]
    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        present = pose_frame_mask(base_data[:, :SELECTED_FEATURES])
        velocity[1:][~(present[1:] & present[:-1])] = 0.0

    parts = [base_data]
    if USE_VELOCITY:
        parts.append(velocity)
    if USE_ACCELERATION:
        acceleration = np.zeros_like(velocity)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        if feature_pipeline == HYBRID_FEATURE_PIPELINE:
            present = pose_frame_mask(base_data[:, :SELECTED_FEATURES])
            valid_acceleration = present[2:] & present[1:-1] & present[:-2]
            acceleration[1:] = 0.0
            acceleration[2:][valid_acceleration] = (
                velocity[2:][valid_acceleration]
                - velocity[1:-1][valid_acceleration]
            )
        parts.append(acceleration)
    engineered = np.concatenate(parts, axis=1)
    if engineered.shape[1] != expected_total:
        raise AssertionError("Internal feature dimension mismatch")
    return engineered


def engineer_features(data: np.ndarray, feature_pipeline=FEATURE_PIPELINE) -> np.ndarray:
    """Filter raw MediaPipe landmarks and engineer one complete sequence."""
    return engineer_selected_features(filter_landmarks(data), feature_pipeline)


def engineer_selected_features(data: np.ndarray, feature_pipeline=FEATURE_PIPELINE):
    """Engineer one window of already-filtered 52-feature pose data."""
    return engineer_base_features(
        prepare_base_features(data, feature_pipeline), feature_pipeline
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================
def _deprecated_standardized_horizontal_flip(window: np.ndarray) -> np.ndarray:
    """
    Mirror pose secara horizontal: flip koordinat x.

    Efektif menggandakan variasi jatuh samping (kiri ↔ kanan).
    Hanya flip fitur base (52 pertama dari 13 selected landmarks).

    Note: Augmentasi ini diterapkan SETELAH normalisasi.
    Negate x = mirror horizontal.
    """
    result = window.copy()
    # Flip koordinat x untuk setiap selected landmark (index 0, 4, 8, ..., 48)
    for i in range(NUM_SELECTED_LANDMARKS):
        x_idx = i * MP_FEATURES_PER_LANDMARK  # x coordinate index
        if x_idx < result.shape[1]:
            result[:, x_idx] = -result[:, x_idx]
    return result


def _deprecated_standardized_augmentation(window: np.ndarray) -> np.ndarray:
    """Apply random combination of augmentations."""
    # Jitter (selalu)
    window = augment_jitter(window)

    # Scale (50% chance)
    if np.random.random() < 0.5:
        window = augment_scale(window)

    # Horizontal flip (50% chance) — efektif untuk variasi jatuh samping
    if np.random.random() < 0.5:
        window = augment_horizontal_flip(window)

    # Time warp
    if np.random.random() < AUG_TIME_WARP_PROB:
        window = augment_time_warp(window)

    # Landmark dropout
    if np.random.random() < 0.3:
        window = augment_landmark_dropout(window)

    return window


_LEFT_RIGHT_SELECTED_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)
)


def _landmark_view(base_window):
    return base_window[:, :SELECTED_FEATURES].reshape(
        len(base_window), NUM_SELECTED_LANDMARKS, MP_FEATURES_PER_LANDMARK
    )


def augment_jitter(window: np.ndarray, std: float = AUG_JITTER_STD) -> np.ndarray:
    """Jitter only visible landmark xyz values in unstandardized space."""
    result = window.copy()
    landmarks = _landmark_view(result)
    present = pose_frame_mask(result[:, :SELECTED_FEATURES])
    noise = np.random.normal(
        0.0, std, (len(result), NUM_SELECTED_LANDMARKS, 3)
    ).astype(np.float32)
    landmarks[present, :, :3] += noise[present]
    return result


def augment_scale(
    window: np.ndarray,
    scale_range: tuple = AUG_SCALE_RANGE,
    feature_pipeline=FEATURE_PIPELINE,
) -> np.ndarray:
    """Apply a mild body-size variation without changing missing frames."""
    result = window.copy()
    landmarks = _landmark_view(result)
    present = pose_frame_mask(result[:, :SELECTED_FEATURES])
    scale = float(np.random.uniform(scale_range[0], scale_range[1]))
    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        landmarks[present, :, :3] *= scale
        result[present, SELECTED_FEATURES + 3] *= scale
        result[present, SELECTED_FEATURES + 4] *= scale
        result[present, SELECTED_FEATURES + 6] *= scale
    else:
        for frame_index in np.flatnonzero(present):
            center = landmarks[frame_index, 7:9, :3].mean(axis=0)
            landmarks[frame_index, :, :3] = (
                center + (landmarks[frame_index, :, :3] - center) * scale
            )
    return result


def augment_time_warp(window: np.ndarray) -> np.ndarray:
    """Resample motion speed while retaining pose-loss frames and endpoint."""
    seq_len = len(window)
    if seq_len < 3:
        return window.copy()
    factor = float(np.random.uniform(0.8, 1.2))
    center = (seq_len - 1) * 0.5
    source_positions = center + (np.arange(seq_len) - center) * factor
    source_positions = np.clip(source_positions, 0.0, seq_len - 1.0)
    timeline = np.arange(seq_len, dtype=np.float32)
    result = np.empty_like(window)
    for feature_index in range(window.shape[1]):
        result[:, feature_index] = np.interp(
            source_positions, timeline, window[:, feature_index]
        )

    nearest = np.rint(source_positions).astype(np.int64)
    source_present = pose_frame_mask(window[:, :SELECTED_FEATURES])
    missing = ~source_present[nearest]
    result[missing] = 0.0
    # The window label is tied to the final frame; never move that endpoint.
    result[-1] = window[-1]
    return result


def augment_landmark_dropout(
    window: np.ndarray, prob: float = AUG_DROPOUT_PROB
) -> np.ndarray:
    """Deprecated independent dropout retained as an explicit no-op by default."""
    if prob <= 0.0:
        return window.copy()
    result = window.copy()
    landmarks = _landmark_view(result)
    mask = np.random.random(landmarks.shape[:2]) < prob
    landmarks[mask] = 0.0
    return result


def augment_horizontal_flip(
    window: np.ndarray, feature_pipeline=FEATURE_PIPELINE
) -> np.ndarray:
    """Mirror x coordinates and swap all anatomical left/right landmarks."""
    result = window.copy()
    landmarks = _landmark_view(result)
    present = pose_frame_mask(result[:, :SELECTED_FEATURES])
    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        landmarks[present, :, 0] *= -1.0
    else:
        landmarks[present, :, 0] = 1.0 - landmarks[present, :, 0]
    original = landmarks.copy()
    for left_index, right_index in _LEFT_RIGHT_SELECTED_PAIRS:
        landmarks[:, left_index] = original[:, right_index]
        landmarks[:, right_index] = original[:, left_index]
    return result


def augment_camera_motion(
    window: np.ndarray, feature_pipeline=FEATURE_PIPELINE
) -> np.ndarray:
    """Apply mild camera translation/roll/zoom in physically meaningful axes."""
    result = window.copy()
    landmarks = _landmark_view(result)
    present = pose_frame_mask(result[:, :SELECTED_FEATURES])
    if not np.any(present):
        return result
    translate_x = float(np.random.uniform(-AUG_CAMERA_TRANSLATE, AUG_CAMERA_TRANSLATE))
    translate_y = float(np.random.uniform(-AUG_CAMERA_TRANSLATE, AUG_CAMERA_TRANSLATE))
    angle = np.deg2rad(np.random.uniform(-AUG_CAMERA_ROTATE_DEG, AUG_CAMERA_ROTATE_DEG))
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    zoom = float(np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1]))

    if feature_pipeline == HYBRID_FEATURE_PIPELINE:
        x_values = landmarks[present, :, 0].copy()
        y_values = landmarks[present, :, 1].copy()
        landmarks[present, :, 0] = cos_angle * x_values - sin_angle * y_values
        landmarks[present, :, 1] = sin_angle * x_values + cos_angle * y_values
        result[present, SELECTED_FEATURES:SELECTED_FEATURES + 3] += translate_y
        result[present, SELECTED_FEATURES + 3] *= zoom
        result[present, SELECTED_FEATURES + 4] *= zoom
        result[present, SELECTED_FEATURES + 6] *= zoom
    else:
        xy = landmarks[present, :, :2]
        center = np.array([0.5, 0.5], dtype=np.float32)
        centered = (xy - center) * zoom
        rotated_x = cos_angle * centered[:, :, 0] - sin_angle * centered[:, :, 1]
        rotated_y = sin_angle * centered[:, :, 0] + cos_angle * centered[:, :, 1]
        landmarks[present, :, 0] = rotated_x + center[0] + translate_x
        landmarks[present, :, 1] = rotated_y + center[1] + translate_y
    return result


def augment_contiguous_occlusion(
    window: np.ndarray, max_frames=AUG_CONTIGUOUS_OCCLUSION_MAX_FRAMES
) -> np.ndarray:
    """Hide a short contiguous internal block; preserve the labelled endpoint."""
    result = window.copy()
    if len(result) <= 2 or max_frames <= 0:
        return result
    max_length = min(int(max_frames), len(result) - 1)
    length = int(np.random.randint(1, max_length + 1))
    start = int(np.random.randint(0, len(result) - length))
    result[start:start + length] = 0.0
    return result


def apply_legacy_standardized_augmentation(window: np.ndarray) -> np.ndarray:
    """Reproduce the augmentation used by the retained baseline checkpoint.

    This profile intentionally operates after StandardScaler. It is kept for
    reproducibility because it gave the strongest held-out Fall recall; newer
    physically constrained profiles must be selected explicitly.
    """
    result = window.copy()
    result += np.random.normal(0.0, 0.05, result.shape).astype(np.float32)
    if np.random.random() < 0.5:
        result *= float(np.random.uniform(0.9, 1.1))
    if np.random.random() < 0.5:
        for landmark_index in range(NUM_SELECTED_LANDMARKS):
            result[:, landmark_index * MP_FEATURES_PER_LANDMARK] *= -1.0
    if np.random.random() < 0.3:
        seq_len = len(result)
        warp_factor = float(np.random.uniform(0.8, 1.2))
        new_len = max(int(seq_len * warp_factor), seq_len)
        stretched = np.zeros((new_len, result.shape[1]), dtype=np.float32)
        destination = np.arange(new_len)
        source = np.linspace(0, new_len - 1, seq_len)
        for feature_index in range(result.shape[1]):
            stretched[:, feature_index] = np.interp(
                destination, source, result[:, feature_index]
            )
        start = (new_len - seq_len) // 2
        result = stretched[start:start + seq_len]
    if np.random.random() < 0.3:
        result[np.random.random(result.shape) < 0.05] = 0.0
    return result


def apply_augmentation(
    window: np.ndarray,
    feature_pipeline=FEATURE_PIPELINE,
    augmentation_profile=AUGMENTATION_PROFILE,
):
    """Augment an unstandardized base window, then rebuild derivatives."""
    if augmentation_profile not in ("minimal_v2", "full_v2"):
        raise ValueError(
            "Augmentation profile tidak dikenal: " + str(augmentation_profile)
        )
    result = augment_jitter(window)
    if np.random.random() < AUG_HORIZONTAL_FLIP_PROB:
        result = augment_horizontal_flip(result, feature_pipeline)
    if augmentation_profile == "full_v2":
        if np.random.random() < 0.5:
            result = augment_scale(result, feature_pipeline=feature_pipeline)
        if np.random.random() < AUG_CAMERA_MOTION_PROB:
            result = augment_camera_motion(result, feature_pipeline)
        if np.random.random() < AUG_TIME_WARP_PROB:
            result = augment_time_warp(result)
        if np.random.random() < AUG_CONTIGUOUS_OCCLUSION_PROB:
            result = augment_contiguous_occlusion(result)
    return result


# ============================================================
# SLIDING WINDOW
# ============================================================
def sliding_window(data: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    """
    Potong sequence panjang menjadi windows berukuran tetap.

    Args:
        data: np.ndarray shape (num_frames, num_features)
        window_size: panjang tiap window (SEQUENCE_LENGTH)
        stride: langkah geser antar window

    Returns:
        List of np.ndarray, masing-masing shape (window_size, num_features)
    """
    num_frames = data.shape[0]

    if num_frames < window_size:
        # Jika video lebih pendek dari window: pad dengan zeros di akhir
        padded = np.zeros((window_size, data.shape[1]), dtype=data.dtype)
        padded[:num_frames] = data
        return [padded]

    return [
        data[start:start + window_size]
        for start in window_start_indices(num_frames, window_size, stride)
    ]


# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================
def load_all_data(
    window_stride: int = STRIDE,
    include_ambiguous: bool = False,
    return_subject_ids: bool = False,
    require_subject_metadata: bool = False,
    feature_pipeline: str = FEATURE_PIPELINE,
) -> tuple:
    """
    Load semua file .npy, apply feature engineering, dan buat sliding windows.

    PENTING: Split dilakukan per VIDEO, bukan per window.

    Returns:
        video_windows: list of list of np.ndarray (windows per video)
        video_window_labels: list of list[int] (label per retained window)
        video_event_labels: list[int] (source-video class for grouped split)
        video_names: list of str (nama file per video)
        video_subject_ids: optional list[str|None], returned only when
            return_subject_ids=True
    """
    video_windows = []
    video_window_labels = []
    video_event_labels = []
    video_names = []
    label_summary = {
        "ADL": 0,
        "Fall": 0,
        "ambiguous_skipped": 0,
        "no_pose_end_skipped": 0,
    }
    source_summary = {}

    for source in TRAINING_DATASETS:
        source_name = source["name"]
        source_summary[source_name] = {"ADL": 0, "Fall": 0}
        for label_name, label_id in LABEL_MAP.items():
            if label_name == "ADL":
                processed_dir = source["processed_adl_dir"]
            else:
                processed_dir = source["processed_fall_dir"]

            if not processed_dir.is_dir():
                raise FileNotFoundError(
                    "Processed dataset '{}' tidak ditemukan: {}. Jalankan "
                    "feature extraction untuk source ini.".format(
                        source_name, processed_dir
                    )
                )
            npy_files = sorted(processed_dir.glob("*.npy"))
            if not npy_files:
                raise ValueError(
                    "Tidak ada .npy untuk dataset '{}' class {} di {}".format(
                        source_name, label_name, processed_dir
                    )
                )
            fall_annotations = None
            if label_name == "Fall":
                fall_annotations = load_fall_annotations(
                    source["fall_annotations_path"]
                )

            for npy_path in npy_files:
                data = np.load(str(npy_path))  # shape: (num_frames, 132)
                pose_present = np.any(data != 0.0, axis=1)

                # Samakan urutan dengan inference: filter base landmarks -> buat
                # window 30 frame -> hitung velocity/acceleration di dalam window.
                # Dengan begitu timestep pertama selalu memiliki derivative nol.
                selected_data = filter_landmarks(data)
                starts = window_start_indices(
                    len(selected_data), SEQUENCE_LENGTH, window_stride
                )
                base_windows = sliding_window(
                    selected_data, SEQUENCE_LENGTH, window_stride
                )

                if label_name == "ADL":
                    window_labels = np.zeros(len(base_windows), dtype=np.int64)
                else:
                    annotation = fall_annotations.get(npy_path.stem)
                    if annotation is None:
                        raise ValueError(
                            "Fall video '{}:{}' belum memiliki interval di {}"
                            .format(
                                source_name, npy_path.stem,
                                source["fall_annotations_path"],
                            )
                        )
                    annotation = dict(annotation)
                    annotation["video_stem"] = npy_path.stem
                    window_labels, details = label_window_starts(
                        num_frames=len(selected_data),
                        starts=starts,
                        window_size=SEQUENCE_LENGTH,
                        annotation=annotation,
                        min_fall_overlap_frames=FALL_MIN_OVERLAP_FRAMES,
                        pose_present=pose_present,
                    )
                    if details["fall_windows"] == 0:
                        print(
                            "  [WARNING] '{}:{}' tidak memiliki Fall window "
                            "dengan pose terlihat; hanya window ADL yang "
                            "eligible akan dipakai.".format(
                                source_name, npy_path.stem
                            )
                        )

                label_keep_mask = (
                    np.ones(len(window_labels), dtype=bool)
                    if include_ambiguous
                    else window_labels != AMBIGUOUS_LABEL
                )
                end_pose_present = window_end_pose_mask(
                    pose_present, starts, SEQUENCE_LENGTH
                )
                keep_mask = label_keep_mask & end_pose_present
                retained_base_windows = [
                    window for window, keep in zip(base_windows, keep_mask) if keep
                ]
                retained_labels = window_labels[keep_mask]
                windows = [
                    engineer_selected_features(window, feature_pipeline)
                    for window in retained_base_windows
                ]

                video_windows.append(windows)
                video_window_labels.append(retained_labels.tolist())
                video_event_labels.append(label_id)
                video_names.append("{}:{}".format(source_name, npy_path.stem))
                source_summary[source_name][label_name] += 1
                label_summary["ADL"] += int(np.sum(retained_labels == 0))
                label_summary["Fall"] += int(np.sum(retained_labels == 1))
                if not include_ambiguous:
                    label_summary["ambiguous_skipped"] += int(
                        np.sum(window_labels == AMBIGUOUS_LABEL)
                    )
                label_summary["no_pose_end_skipped"] += int(
                    np.sum(label_keep_mask & ~end_pose_present)
                )

    print(
        "  Dataset sources: {}".format(
            ", ".join(
                "{}(ADL={}, Fall={})".format(
                    name, counts["ADL"], counts["Fall"]
                )
                for name, counts in source_summary.items()
            )
        )
    )

    print(
        "  Temporal window labels: ADL={}, Fall={}, ambiguous skipped={}, "
        "no-pose-end skipped={}".format(
            label_summary["ADL"], label_summary["Fall"],
            label_summary["ambiguous_skipped"],
            label_summary["no_pose_end_skipped"],
        )
    )
    result = (
        video_windows,
        video_window_labels,
        video_event_labels,
        video_names,
    )
    if return_subject_ids:
        subject_payload = load_subject_metadata(
            SUBJECT_METADATA_PATH,
            allow_missing=not require_subject_metadata,
        )
        video_subject_ids = resolve_subject_ids(
            video_names,
            subject_payload,
            require_complete=require_subject_metadata,
        )
        return result + (video_subject_ids,)
    return result


def split_video_indices(
    video_event_labels,
    video_subject_ids=None,
    split_strategy=SPLIT_STRATEGY,
    train_subject_ids=SUBJECT_TRAIN_IDS,
    val_subject_ids=SUBJECT_VAL_IDS,
    test_subject_ids=SUBJECT_TEST_IDS,
):
    """Return deterministic train/validation/test video indices."""
    event_labels_array = np.asarray(video_event_labels, dtype=np.int64)
    indices = np.arange(len(event_labels_array))

    if split_strategy == "video_stratified":
        from sklearn.model_selection import train_test_split

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=event_labels_array,
        )
        val_relative_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_relative_size,
            random_state=RANDOM_SEED,
            stratify=event_labels_array[train_val_idx],
        )
        return train_idx, val_idx, test_idx

    if split_strategy != "subject":
        raise ValueError("Split strategy tidak dikenal: " + str(split_strategy))
    if video_subject_ids is None:
        raise ValueError("Subject split memerlukan video_subject_ids")
    if len(video_subject_ids) != len(event_labels_array):
        raise ValueError("Jumlah subject ID tidak cocok dengan jumlah video")

    train_subjects = set(train_subject_ids)
    val_subjects = set(val_subject_ids)
    test_subjects = set(test_subject_ids)
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("Train, validation, dan test subject tidak boleh kosong")
    if (
        train_subjects & val_subjects
        or train_subjects & test_subjects
        or val_subjects & test_subjects
    ):
        raise ValueError("Subject tidak boleh muncul di lebih dari satu split")

    subjects = np.asarray(video_subject_ids, dtype=object)
    observed_subjects = set(subjects.tolist())
    configured_subjects = train_subjects | val_subjects | test_subjects
    unassigned = {None, ""} & observed_subjects
    if unassigned:
        raise ValueError("Subject metadata belum lengkap")
    missing_config = configured_subjects - observed_subjects
    extra_subjects = observed_subjects - configured_subjects
    if missing_config:
        raise ValueError(
            "Configured subject tidak ditemukan: {}".format(
                sorted(missing_config)
            )
        )
    if extra_subjects:
        raise ValueError(
            "Subject belum ditempatkan ke split: {}".format(
                sorted(extra_subjects)
            )
        )

    train_idx = np.flatnonzero(np.isin(subjects, list(train_subjects)))
    val_idx = np.flatnonzero(np.isin(subjects, list(val_subjects)))
    test_idx = np.flatnonzero(np.isin(subjects, list(test_subjects)))
    for split_name, split_idx in (
        ("train", train_idx), ("validation", val_idx), ("test", test_idx)
    ):
        if len(split_idx) == 0:
            raise ValueError("Split {} tidak memiliki video".format(split_name))
        if set(event_labels_array[split_idx].tolist()) != {0, 1}:
            raise ValueError(
                "Split {} harus memiliki video ADL dan Fall".format(split_name)
            )
    return train_idx, val_idx, test_idx


def split_data(
    video_windows,
    video_window_labels,
    video_event_labels,
    video_names,
    video_subject_ids=None,
    split_strategy=SPLIT_STRATEGY,
    train_subject_ids=SUBJECT_TRAIN_IDS,
    val_subject_ids=SUBJECT_VAL_IDS,
    test_subject_ids=SUBJECT_TEST_IDS,
    feature_pipeline: str = FEATURE_PIPELINE,
):
    """
    Split data per VIDEO menjadi train/val/test.

    Menggunakan subject split atau fallback stratified per-video.

    Returns:
        Tuple of (X, y) untuk train, val, test
    """
    n_videos = len(video_event_labels)
    event_labels_array = np.array(video_event_labels)
    train_idx, val_idx, test_idx = split_video_indices(
        video_event_labels,
        video_subject_ids=video_subject_ids,
        split_strategy=split_strategy,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=test_subject_ids,
    )

    def collect_windows(indices):
        X_list = []
        y_list = []
        names = []
        for idx in indices:
            for window, window_label in zip(
                video_windows[idx], video_window_labels[idx]
            ):
                X_list.append(window)
                y_list.append(window_label)
            names.append(video_names[idx])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), names

    X_train, y_train, train_names = collect_windows(train_idx)
    X_val, y_val, val_names = collect_windows(val_idx)
    X_test, y_test, test_names = collect_windows(test_idx)

    # Log
    print(f"\n{'='*60}")
    print(f"  DATA SPLIT ({split_strategy})")
    print(f"{'='*60}")
    print(f"  Train : {len(train_idx)} video -> {len(X_train)} windows "
          f"(ADL={sum(y_train==0)}, Fall={sum(y_train==1)})")
    print(f"  Val   : {len(val_idx)} video -> {len(X_val)} windows "
          f"(ADL={sum(y_val==0)}, Fall={sum(y_val==1)})")
    print(f"  Test  : {len(test_idx)} video -> {len(X_test)} windows "
          f"(ADL={sum(y_test==0)}, Fall={sum(y_test==1)})")
    print(f"  Total : {n_videos} video -> {len(X_train)+len(X_val)+len(X_test)} windows")
    print(
        "  Video groups: ADL={}, Fall={} (mixed labels stay in one split)".format(
            int(np.sum(event_labels_array == 0)),
            int(np.sum(event_labels_array == 1)),
        )
    )
    base_count, derivative_count, _ = feature_dimensions(feature_pipeline)
    print(f"  Features per frame: {X_train.shape[2]} "
          f"(base={base_count}"
          f"{' +vel=' + str(derivative_count) if USE_VELOCITY else ''}"
          f"{' +acc=' + str(derivative_count) if USE_ACCELERATION else ''})")
    if split_strategy == "subject":
        print("  Train subjects: {}".format(sorted(set(train_subject_ids))))
        print("  Val subjects  : {}".format(sorted(set(val_subject_ids))))
        print("  Test subjects : {}".format(sorted(set(test_subject_ids))))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# PYTORCH DATASET
# ============================================================
class FallDetectionDataset(Dataset):
    """Dataset that augments raw features before applying StandardScaler."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        normalization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_pipeline: str = FEATURE_PIPELINE,
        augmentation_profile: str = AUGMENTATION_PROFILE,
    ):
        """
        Args:
            X: np.ndarray shape (num_samples, SEQUENCE_LENGTH, INPUT_FEATURES)
            y: np.ndarray shape (num_samples,)
            augment: Apakah apply data augmentation (hanya untuk training)
        """
        self.X = np.asarray(X, dtype=np.float32)
        self.y = torch.LongTensor(y)
        self.augment = augment and USE_AUGMENTATION
        self.feature_pipeline = feature_pipeline
        self.augmentation_profile = augmentation_profile
        self.base_features, _, expected_features = feature_dimensions(
            feature_pipeline
        )
        if self.X.shape[2] != expected_features:
            raise ValueError(
                "Dataset memiliki {} fitur, expected {} untuk {}".format(
                    self.X.shape[2], expected_features, feature_pipeline
                )
            )
        if normalization is None:
            self.mean = np.zeros(expected_features, dtype=np.float32)
            self.std = np.ones(expected_features, dtype=np.float32)
        else:
            self.mean = np.asarray(normalization[0], dtype=np.float32)
            self.std = np.asarray(normalization[1], dtype=np.float32)
            if self.mean.shape != (expected_features,) or self.std.shape != (
                expected_features,
            ):
                raise ValueError("Normalization shape tidak cocok dengan dataset")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        if self.augment and self.augmentation_profile == "legacy_v1":
            x = (x - self.mean) / self.std
            x = apply_legacy_standardized_augmentation(x)
            return torch.FloatTensor(x), self.y[idx]

        if self.augment:
            base_window = x[:, :self.base_features]
            base_window = apply_augmentation(
                base_window,
                feature_pipeline=self.feature_pipeline,
                augmentation_profile=self.augmentation_profile,
            )
            x = engineer_base_features(base_window, self.feature_pipeline)

        x = (x - self.mean) / self.std

        return torch.FloatTensor(x), self.y[idx]


# ============================================================
# DATALOADER FACTORY
# ============================================================
def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    normalization: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    save_scaler: bool = True,
    return_scaler: bool = False,
    split_strategy: str = SPLIT_STRATEGY,
    train_subject_ids=SUBJECT_TRAIN_IDS,
    val_subject_ids=SUBJECT_VAL_IDS,
    test_subject_ids=SUBJECT_TEST_IDS,
    feature_pipeline: str = FEATURE_PIPELINE,
):
    """
    Entry point utama: load data, split, normalize, buat DataLoader.

    Args:
        batch_size: Jumlah sequence per batch.
        normalization: Optional tuple (mean, std) dari checkpoint. Jika None,
            scaler di-fit hanya pada training split.
        save_scaler: Simpan scaler eksternal untuk kompatibilitas deployment lama.
        return_scaler: Tambahkan (mean, std) pada nilai return.

    Returns:
        train_loader, val_loader, test_loader, dan optional scaler tuple.
    """
    # Load dan window semua data
    data = load_all_data(
        return_subject_ids=(split_strategy == "subject"),
        require_subject_metadata=(split_strategy == "subject"),
        feature_pipeline=feature_pipeline,
    )
    video_windows, video_window_labels, video_event_labels, video_names = data[:4]
    video_subject_ids = data[4] if len(data) == 5 else None

    # Split per video
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        video_windows,
        video_window_labels,
        video_event_labels,
        video_names,
        video_subject_ids=video_subject_ids,
        split_strategy=split_strategy,
        train_subject_ids=train_subject_ids,
        val_subject_ids=val_subject_ids,
        test_subject_ids=test_subject_ids,
        feature_pipeline=feature_pipeline,
    )

    # ========================================
    # NORMALIZATION (StandardScaler)
    # ========================================
    # Fit scaler HANYA pada training data (mencegah data leakage)
    # Reshape: (num_samples, seq_len, features) -> (num_samples*seq_len, features)
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)

    # Hitung mean/std dari train atau gunakan scaler yang terikat checkpoint.
    if normalization is None:
        train_mean = X_train_flat.mean(axis=0).astype(np.float32)
        train_std = X_train_flat.std(axis=0).astype(np.float32)
        train_std[train_std == 0] = 1.0  # Cegah division by zero
        scaler_source = "fit on current train split"
    else:
        train_mean = np.asarray(normalization[0], dtype=np.float32)
        train_std = np.asarray(normalization[1], dtype=np.float32)
        if train_mean.shape != (n_feat,) or train_std.shape != (n_feat,):
            raise ValueError(
                f"Scaler shape tidak cocok: mean={train_mean.shape}, "
                f"std={train_std.shape}, expected=({n_feat},)"
            )
        if not np.isfinite(train_mean).all() or not np.isfinite(train_std).all():
            raise ValueError("Scaler mengandung nilai non-finite")
        if np.any(train_std <= 0):
            raise ValueError("Scaler std harus lebih besar dari nol")
        scaler_source = "checkpoint"

    # Preview only. Dataset applies this scaler after any raw-space augmentation.
    X_train_preview = (X_train - train_mean) / train_std
    X_val_preview = (X_val - train_mean) / train_std
    X_test_preview = (X_test - train_mean) / train_std

    print(f"\n  Normalization applied (StandardScaler, source={scaler_source})")
    print(f"    Train range: [{X_train_preview.min():.2f}, {X_train_preview.max():.2f}]")
    print(f"    Val   range: [{X_val_preview.min():.2f}, {X_val_preview.max():.2f}]")
    print(f"    Test  range: [{X_test_preview.min():.2f}, {X_test_preview.max():.2f}]")

    # Simpan scaler untuk inference nanti
    if save_scaler:
        scaler_path = Path(__file__).resolve().parent.parent / "models" / "scaler_params.npz"
        np.savez(str(scaler_path), mean=train_mean, std=train_std)
        print(f"    Scaler saved: {scaler_path.name}")

    # Buat Dataset (augmentation HANYA untuk training)
    scaler = (train_mean, train_std)
    train_dataset = FallDetectionDataset(
        X_train, y_train, augment=True, normalization=scaler,
        feature_pipeline=feature_pipeline,
    )
    val_dataset = FallDetectionDataset(
        X_val, y_val, augment=False, normalization=scaler,
        feature_pipeline=feature_pipeline,
    )
    test_dataset = FallDetectionDataset(
        X_test, y_test, augment=False, normalization=scaler,
        feature_pipeline=feature_pipeline,
    )

    # Buat DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    aug_status = "ON" if USE_AUGMENTATION else "OFF"
    print(f"\n  DataLoader created (augmentation={aug_status}):")
    print(f"    Train: {len(train_loader)} batches (batch_size={batch_size})")
    print(f"    Val  : {len(val_loader)} batches")
    print(f"    Test : {len(test_loader)} batches")

    if return_scaler:
        return train_loader, val_loader, test_loader, (train_mean, train_std)
    return train_loader, val_loader, test_loader


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    # Quick check
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n  Sample batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"  X dtype: {X_batch.dtype}, y dtype: {y_batch.dtype}")
    print(f"  Labels in batch: {y_batch.unique(return_counts=True)}")
    print(f"  X range: min={X_batch.min():.4f}, max={X_batch.max():.4f}")
