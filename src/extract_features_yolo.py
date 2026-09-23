'''
extract_features_yolo.py - Ekstraksi fitur pose dari video menggunakan YOLOv8n-Pose.

Pipeline:
    1. Baca setiap video dari dataset raw
    2. Jalankan YOLOv8n-pose per frame (GPU/CUDA)
    3. Ambil 13 keypoint utama (COCO: nose, shoulders, elbows, wrists, hips, knees, ankles)
    4. Simpan sebagai .npy shape (num_frames, 52) dengan layout [x, y, z, conf] per sendi

Penggunaan:
    python src/extract_features_yolo.py --all
'''

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import (
    DATA_DIR,
    MODEL_DIR,
    EXPECTED_VIDEO_FPS,
    RAW_DIR,
    NEW_JULY_RAW_DIR,
    NEW_AUG_RAW_DIR,
)

# COCO 17 Keypoints -> 13 Selected Keypoints mapping
# 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist,
# 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
COCO_SELECTED_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_SELECTED_JOINTS = len(COCO_SELECTED_KEYPOINTS)  # 13
FEATURES_PER_JOINT = 4  # x, y, z, v
TOTAL_FEATURES = NUM_SELECTED_JOINTS * FEATURES_PER_JOINT  # 52

YOLO_MODEL_PATH = MODEL_DIR / 'yolov8n-pose.pt'


def extract_landmarks_from_video(video_path: Path, model: YOLO, device: str = 'cuda', conf_thresh: float = 0.25):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.empty((0, TOTAL_FEATURES), dtype=np.float32), {'fps': 0.0, 'frames': 0, 'detected_frames': 0}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or EXPECTED_VIDEO_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return np.empty((0, TOTAL_FEATURES), dtype=np.float32), {'fps': fps, 'frames': 0, 'detected_frames': 0}

    # Predict in batches on GPU
    batch_size = 32
    all_features = []
    detected_count = 0

    for b_start in range(0, len(frames), batch_size):
        batch_frames = frames[b_start:b_start + batch_size]
        results = model(batch_frames, device=device, verbose=False, conf=conf_thresh)
        for res in results:
            if res.keypoints is not None and len(res.keypoints.data) > 0:
                boxes = res.boxes
                if len(boxes) > 1:
                    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
                    best_idx = int(torch.argmax(areas).item())
                else:
                    best_idx = 0

                xyn = res.keypoints.xyn[best_idx].cpu().numpy()
                conf = res.keypoints.conf[best_idx].cpu().numpy()

                frame_feat = []
                for kpt_idx in COCO_SELECTED_KEYPOINTS:
                    x = float(xyn[kpt_idx, 0])
                    y = float(xyn[kpt_idx, 1])
                    z = 0.0
                    v = float(conf[kpt_idx])
                    frame_feat.extend([x, y, z, v])
                all_features.append(frame_feat)
                detected_count += 1
            else:
                all_features.append([0.0] * TOTAL_FEATURES)

    landmarks_arr = np.array(all_features, dtype=np.float32)
    return landmarks_arr, {
        'fps': fps,
        'frames': len(frames),
        'detected_frames': detected_count,
    }


def process_dataset(raw_dir: Path, processed_dir: Path, model: YOLO, device: str = 'cuda', overwrite: bool = False):
    processed_adl_dir = processed_dir / 'ADL'
    processed_fall_dir = processed_dir / 'Fall'
    processed_adl_dir.mkdir(parents=True, exist_ok=True)
    processed_fall_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    all_videos = sorted([f for f in raw_dir.rglob('*') if f.is_file() and f.suffix.lower() in video_exts])

    adl_videos = [v for v in all_videos if 'adl' in v.stem.lower() or any('adl' in p.lower() for p in v.relative_to(raw_dir).parts)]
    fall_videos = [v for v in all_videos if 'fall' in v.stem.lower() or any('fall' in p.lower() for p in v.relative_to(raw_dir).parts)]

    print('=' * 60)
    print(f'  Processing dataset: {raw_dir.name}')
    print(f'  Input:  {raw_dir}')
    print(f'  Output: {processed_dir}')
    print(f'  Found : {len(adl_videos)} ADL, {len(fall_videos)} Fall ({len(all_videos)} total)')
    print('=' * 60)

    stats = {'ADL': {'total': len(adl_videos), 'success': 0, 'detected_frames': 0, 'frames': 0},
             'Fall': {'total': len(fall_videos), 'success': 0, 'detected_frames': 0, 'frames': 0}}

    for category, video_list, out_dir in [('ADL', adl_videos, processed_adl_dir), ('Fall', fall_videos, processed_fall_dir)]:
        for i, v_path in enumerate(video_list, 1):
            out_npy = out_dir / f'{v_path.stem}.npy'
            if out_npy.exists() and not overwrite:
                print(f'  [{i}/{len(video_list)}] SKIP (exists): {v_path.name}')
                stats[category]['success'] += 1
                continue

            landmarks, v_stats = extract_landmarks_from_video(v_path, model, device=device)
            if landmarks.shape[0] == 0:
                print(f'  [WARNING] Empty landmarks for {v_path.name}')
                continue

            np.save(str(out_npy), landmarks)
            stats[category]['success'] += 1
            stats[category]['frames'] += v_stats['frames']
            stats[category]['detected_frames'] += v_stats['detected_frames']
            det_pct = (v_stats['detected_frames'] / max(1, v_stats['frames'])) * 100.0
            print(f'  [{i}/{len(video_list)}] {v_path.name}: {v_stats["frames"]} fr, {v_stats["detected_frames"]} det ({det_pct:.1f}%)')

    # Write extraction metadata
    meta = {
        'schema_version': 1,
        'extractor': 'yolov8n-pose',
        'features_per_frame': TOTAL_FEATURES,
        'selected_joints': NUM_SELECTED_JOINTS,
        'source_root': str(raw_dir),
        'processed_root': str(processed_dir),
        'categories': stats,
    }
    with (processed_dir / 'extraction_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract YOLOv8n-pose landmarks from videos')
    parser.add_argument('--all', action='store_true', help='Extract all 3 datasets')
    parser.add_argument('--dataset-dir', type=Path, help='Custom dataset raw dir')
    parser.add_argument('--processed-dir', type=Path, help='Custom dataset processed dir')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing .npy')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading YOLOv8n-pose on {device}...')
    model = YOLO(str(YOLO_MODEL_PATH))

    if args.all or not args.dataset_dir:
        datasets = [
            (RAW_DIR, DATA_DIR / 'processed_yolo'),
            (NEW_JULY_RAW_DIR, DATA_DIR / 'processed_yolo_new_july_realtime'),
            (NEW_AUG_RAW_DIR, DATA_DIR / 'processed_yolo_new_dataset_20_aug'),
        ]
        for raw_path, proc_path in datasets:
            if raw_path.exists():
                process_dataset(raw_path, proc_path, model, device=device, overwrite=args.overwrite)
    else:
        raw_dir = args.dataset_dir.resolve()
        proc_dir = args.processed_dir.resolve() if args.processed_dir else DATA_DIR / f'processed_yolo_{raw_dir.stem}'
        process_dataset(raw_dir, proc_dir, model, device=device, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
