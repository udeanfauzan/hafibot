"""
extract_features.py - Ekstraksi fitur pose dari video menggunakan MediaPipe Tasks API.

Pipeline:
    1. Baca setiap video dari data/raw/{ADL,Fall}/
    2. Jalankan MediaPipe PoseLandmarker per frame (Tasks API)
    3. Ambil 33 landmark × 4 fitur (x, y, z, visibility) = 132 fitur per frame
    4. Simpan sebagai .npy di data/processed/{ADL,Fall}/

Penggunaan:
    python src/extract_features.py --overwrite
"""

import argparse
import hashlib
import json
import re
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# Tambahkan root project ke sys.path agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_ADL_DIR, RAW_FALL_DIR,
    PROCESSED_ADL_DIR, PROCESSED_FALL_DIR,
    MP_NUM_LANDMARKS, MP_TOTAL_FEATURES,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_POSE_PRESENCE_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE, MP_MODEL_PATH, MP_MODEL_VARIANT,
    EXPECTED_VIDEO_FPS, PROCESSED_DIR,
    DATA_DIR,
)

# MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def file_sha256(path: Path) -> str:
    """Hitung SHA-256 model supaya provenance hasil ekstraksi dapat diverifikasi."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_landmarks_from_video(
    video_path: str | Path, model_path: Path = MP_MODEL_PATH
) -> tuple[np.ndarray, dict]:
    """
    Ekstrak landmark pose dari satu video menggunakan MediaPipe Tasks API.

    Args:
        video_path: Path ke file video (.mp4)

    Returns:
        np.ndarray dengan shape (num_frames, 132)
        - num_frames: jumlah total frame dari video
        - 132: 33 landmarks × 4 (x, y, z, visibility)
        Tuple landmarks dan metadata video. Jika video gagal dibuka, landmarks
        berupa array kosong (0, 132).
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Tidak bisa membuka video: {video_path.name}")
        return np.empty((0, MP_TOTAL_FEATURES), dtype=np.float32), {
            "fps": 0.0,
            "frames": 0,
            "detected_frames": 0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Konfigurasi PoseLandmarker dengan mode VIDEO
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MP_MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )

    all_landmarks = []
    frame_count = 0
    detected_count = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Hitung timestamp dalam milliseconds
            timestamp_ms = int((frame_count / fps) * 1000)

            # Konversi BGR (OpenCV) → RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Buat MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Deteksi pose
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                # Ambil pose pertama (kita hanya deteksi 1 orang)
                pose = result.pose_landmarks[0]

                frame_features = []
                for landmark in pose:
                    frame_features.extend([
                        landmark.x,           # Normalized [0, 1]
                        landmark.y,           # Normalized [0, 1]
                        landmark.z,           # Depth (relatif)
                        landmark.visibility,  # Confidence [0, 1]
                    ])
                all_landmarks.append(frame_features)
                detected_count += 1
            else:
                # Pose tidak terdeteksi: isi dengan zeros
                # (agar jumlah frame tetap konsisten dengan video asli)
                all_landmarks.append([0.0] * MP_TOTAL_FEATURES)

    cap.release()

    landmarks_array = np.array(all_landmarks, dtype=np.float32)

    print(f"  {video_path.name}: {frame_count} frames, "
          f"{detected_count} detected ({detected_count/max(frame_count,1)*100:.1f}%), "
          f"FPS={fps:.1f}")

    return landmarks_array, {
        "fps": float(fps),
        "frames": int(frame_count),
        "detected_frames": int(detected_count),
    }


def process_directory(
    raw_dir: Path,
    processed_dir: Path,
    label_name: str,
    model_path: Path,
    overwrite: bool = False,
    video_files=None,
) -> dict:
    """
    Proses semua video dalam satu direktori.

    Args:
        raw_dir: Path ke folder video mentah (e.g., data/raw/ADL/)
        processed_dir: Path ke folder output (e.g., data/processed/ADL/)
        label_name: Nama label untuk logging ("ADL" atau "Fall")

    Returns:
        dict berisi statistik: jumlah video, berhasil, gagal, total frame
    """
    # Kumpulkan semua file video (termasuk dari subfolder)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    if video_files is None:
        video_files = sorted([
            f for f in raw_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in video_extensions
        ])
    else:
        video_files = sorted(video_files)

    if not video_files:
        print(f"  [WARNING] Tidak ada video ditemukan di {raw_dir}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    print(f"\n{'='*60}")
    print(f"  Memproses {label_name}: {len(video_files)} video")
    print(f"  Input:  {raw_dir}")
    print(f"  Output: {processed_dir}")
    print(f"{'='*60}")

    stats = {
        "total": len(video_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "frames": 0,
        "detected_frames": 0,
        "source_fps": [],
    }

    for i, video_path in enumerate(video_files, 1):
        output_path = processed_dir / f"{video_path.stem}.npy"

        # Skip jika sudah pernah diproses
        if output_path.exists() and not overwrite:
            print(f"  [{i}/{len(video_files)}] SKIP (sudah ada): {video_path.name}")
            stats["skipped"] += 1
            continue

        print(f"  [{i}/{len(video_files)}] Memproses: {video_path.name}")

        landmarks, video_stats = extract_landmarks_from_video(video_path, model_path)

        if landmarks.shape[0] == 0:
            print(f"  [WARNING] Tidak ada landmark terdeteksi: {video_path.name}")
            stats["failed"] += 1
            continue

        # Simpan sebagai .npy
        np.save(str(output_path), landmarks)
        stats["success"] += 1
        stats["frames"] += video_stats["frames"]
        stats["detected_frames"] += video_stats["detected_frames"]
        stats["source_fps"].append(video_stats["fps"])

    return stats


def main():
    """
    Entry point: ekstrak fitur dari semua video ADL dan Fall.
    """
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks for fall-detection training"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing .npy files (required after changing pose model)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MP_MODEL_PATH,
        help=f"MediaPipe .task model (default: {MP_MODEL_PATH})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Custom dataset root containing ADL* and Fall* folders/files; "
            "the original data/raw dataset is used when omitted"
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help=(
            "Output root for a custom dataset; defaults to "
            "data/processed_<dataset-name>"
        ),
    )
    args = parser.parse_args()
    model_path = args.model_path.resolve()
    custom_dataset = args.dataset_dir is not None

    if custom_dataset:
        source_root = args.dataset_dir.resolve()
        slug = re.sub(
            r"[^A-Za-z0-9_-]+", "_", source_root.name
        ).strip("_").lower() or "custom_dataset"
        processed_root = (
            args.processed_dir.resolve()
            if args.processed_dir is not None
            else (DATA_DIR / ("processed_" + slug)).resolve()
        )
        processed_adl_dir = processed_root / "ADL"
        processed_fall_dir = processed_root / "Fall"
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        all_video_files = sorted(
            path for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in video_extensions
        )
        adl_video_files = [
            path for path in all_video_files
            if path.stem.lower().startswith("adl")
            or any(
                part.lower().startswith("adl")
                for part in path.relative_to(source_root).parts[:-1]
            )
        ]
        fall_video_files = [
            path for path in all_video_files
            if path.stem.lower().startswith("fall")
            or any(
                part.lower().startswith("fall")
                for part in path.relative_to(source_root).parts[:-1]
            )
        ]
        classified = set(adl_video_files) | set(fall_video_files)
        unclassified = [path for path in all_video_files if path not in classified]
        if unclassified:
            raise ValueError(
                "Video tanpa prefix ADL/Fall: {}".format(
                    ", ".join(path.name for path in unclassified)
                )
            )
        raw_adl_dir = source_root
        raw_fall_dir = source_root
    else:
        if args.processed_dir is not None:
            raise ValueError(
                "--processed-dir hanya digunakan bersama --dataset-dir"
            )
        source_root = RAW_ADL_DIR.parent
        processed_root = PROCESSED_DIR
        raw_adl_dir = RAW_ADL_DIR
        raw_fall_dir = RAW_FALL_DIR
        processed_adl_dir = PROCESSED_ADL_DIR
        processed_fall_dir = PROCESSED_FALL_DIR
        adl_video_files = None
        fall_video_files = None

    print("=" * 60)
    print("  FALL DETECTION - Feature Extraction (MediaPipe Pose)")
    print("=" * 60)
    print(f"  MediaPipe Tasks API config:")
    print(f"    - Model          : {model_path.name}")
    print(f"    - Model variant  : {MP_MODEL_VARIANT}")
    print(f"    - Detection conf.: {MP_MIN_DETECTION_CONFIDENCE}")
    print(f"    - Presence conf. : {MP_MIN_POSE_PRESENCE_CONFIDENCE}")
    print(f"    - Tracking conf. : {MP_MIN_TRACKING_CONFIDENCE}")
    print(f"    - Features/frame : {MP_TOTAL_FEATURES} (33 landmarks x 4)")
    print()

    # Validasi model file
    if not model_path.exists():
        print(f"[ERROR] Model file tidak ditemukan: {model_path}")
        print(f"  Download dari: https://storage.googleapis.com/mediapipe-models/"
              f"pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        sys.exit(1)

    existing_arrays = list(processed_root.rglob("*.npy"))
    if existing_arrays and not args.overwrite:
        print(
            f"[ERROR] Ditemukan {len(existing_arrays)} processed arrays. "
            "Gunakan --overwrite agar fitur lama tidak tercampur dengan model baru."
        )
        sys.exit(1)

    # Validasi folder input
    for dir_path, name in [(raw_adl_dir, "ADL"), (raw_fall_dir, "Fall")]:
        if not dir_path.exists():
            print(f"[ERROR] Folder tidak ditemukan: {dir_path}")
            print(f"  Pastikan video {name} sudah ada di {dir_path}")
            sys.exit(1)

    start_time = time.time()
    processed_adl_dir.mkdir(parents=True, exist_ok=True)
    processed_fall_dir.mkdir(parents=True, exist_ok=True)

    # Proses ADL
    adl_stats = process_directory(
        raw_adl_dir, processed_adl_dir, "ADL", model_path, args.overwrite,
        video_files=adl_video_files,
    )

    # Proses Fall
    fall_stats = process_directory(
        raw_fall_dir, processed_fall_dir, "Fall", model_path, args.overwrite,
        video_files=fall_video_files,
    )

    all_fps = adl_stats["source_fps"] + fall_stats["source_fps"]
    fps_values = sorted({round(value, 3) for value in all_fps})
    unexpected_fps = [
        value for value in all_fps
        if abs(value - EXPECTED_VIDEO_FPS) > 0.01
    ]
    if unexpected_fps:
        print(
            f"  [WARNING] {len(unexpected_fps)} video tidak sesuai "
            f"EXPECTED_VIDEO_FPS={EXPECTED_VIDEO_FPS}"
        )

    metadata = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "mediapipe_version": mp.__version__,
        "api": "tasks_pose_landmarker_video",
        "model_variant": MP_MODEL_VARIANT,
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "expected_video_fps": EXPECTED_VIDEO_FPS,
        "observed_video_fps": fps_values,
        "features_per_frame": MP_TOTAL_FEATURES,
        "source_root": str(source_root),
        "processed_root": str(processed_root),
        "categories": {"ADL": adl_stats, "Fall": fall_stats},
    }
    metadata_path = processed_root / "extraction_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    elapsed = time.time() - start_time

    # Ringkasan
    print(f"\n{'='*60}")
    print(f"  SELESAI! (waktu: {elapsed:.1f} detik)")
    print(f"{'='*60}")
    print(f"  ADL  : {adl_stats['success']} berhasil, "
          f"{adl_stats['skipped']} skip, {adl_stats['failed']} gagal "
          f"(dari {adl_stats['total']} video)")
    print(f"  Fall : {fall_stats['success']} berhasil, "
          f"{fall_stats['skipped']} skip, {fall_stats['failed']} gagal "
          f"(dari {fall_stats['total']} video)")
    print(f"  Output tersimpan di: {processed_root}")
    print(f"  Metadata: {metadata_path}")
    print()


if __name__ == "__main__":
    main()
