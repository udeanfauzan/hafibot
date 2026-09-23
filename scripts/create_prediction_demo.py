#!/usr/bin/env python3
"""Create an MP4 demo with one correctly predicted video per scenario.

The script uses the saved landmark arrays for deterministic, fast inference,
selects only videos whose event-level prediction matches their source label,
balances the selection across subject IDs, and renders the raw videos with a
pose skeleton and CNN-LSTM Fall-probability overlay.

Examples:
    python scripts/create_prediction_demo.py
    python scripts/create_prediction_demo.py --device cpu --dry-run
    python scripts/create_prediction_demo.py --output results/my_demo.mp4 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FALL_CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    SEQUENCE_LENGTH,
    SUBJECT_METADATA_PATH,
    TRAINING_DATASETS,
)
from src.dataset import (  # noqa: E402
    engineer_selected_features,
    filter_landmarks,
)
from src.model import (  # noqa: E402
    FallDetectionModel,
    checkpoint_feature_pipeline,
    checkpoint_input_features,
    checkpoint_rnn_type,
)


VIDEO_NUMBER_RE = re.compile(r"_\d+$")
COLOR_BG = (18, 22, 30)
COLOR_PANEL = (26, 31, 40)
COLOR_WHITE = (245, 247, 250)
COLOR_MUTED = (174, 183, 194)
COLOR_GREEN = (63, 205, 118)
COLOR_RED = (65, 65, 235)
COLOR_ORANGE = (40, 165, 245)
COLOR_CYAN = (235, 196, 70)
COLOR_YELLOW = (65, 220, 245)

# Connections used by the desktop detector. Landmark arrays contain all 33
# MediaPipe pose points, so the demo can draw exactly the extracted pose.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
)

SCENARIO_TITLES = {
    "ADL_Bangun_dari_lantai": "ADL - Standing Up from Floor",
    "ADL_Berbaring_controlled": "ADL - Controlled Lying Down",
    "ADL_Berdiri_diam": "ADL - Standing Still",
    "ADL_Berjalan": "ADL - Walking",
    "ADL_Duduk_di_kursi": "ADL - Sitting in a Chair",
    "ADL_Jongkok": "ADL - Squatting",
    "ADL_Lompat": "ADL - Jumping",
    "ADL_Membungkuk": "ADL - Bending",
    "ADL_Plank": "ADL - Plank",
    "ADL_Stretching": "ADL - Stretching",
    "ADL_Sujud": "ADL - Prostration",
    "Fall_dari_duduk": "FALL - Falling from Sitting",
    "Fall_ke_belakang": "FALL - Backward Fall",
    "Fall_ke_depan": "FALL - Forward Fall",
    "Fall_saat_berbalik": "FALL - Fall While Turning",
    "Fall_saat_berjalan": "FALL - Fall While Walking",
    "Fall_samping_kanan": "FALL - Right-Side Fall",
    "Fall_samping_kiri": "FALL - Left-Side Fall",
}


@dataclass
class Candidate:
    key: str
    source: str
    stem: str
    scenario: str
    label: str
    subject_id: str
    video_path: Path
    landmarks_path: Path
    prediction: str = ""
    max_fall_probability: float = 0.0
    positive_windows: int = 0
    total_windows: int = 0
    frame_probabilities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32), repr=False
    )

    @property
    def correct(self) -> bool:
        return self.prediction == self.label

    @property
    def quality_score(self) -> float:
        """Higher is better while staying within the correct class."""
        if self.label == "Fall":
            positive_bonus = min(self.positive_windows / 10.0, 1.0) * 0.01
            return self.max_fall_probability + positive_bonus
        return 1.0 - self.max_fall_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one correctly predicted raw video for every ADL/Fall "
            "scenario and render an annotated MP4 demo."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "fall_detection_demo.mp4",
        help="Output MP4 path (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Selection JSON path (default: same name as output, .json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "best_model.pt",
        help="CNN-LSTM checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FALL_CONFIDENCE_THRESHOLD,
        help="Fall event threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--title-seconds", type=float, default=0.8)
    parser.add_argument("--end-seconds", type=float, default=1.5)
    parser.add_argument(
        "--fall-hold-seconds",
        type=float,
        default=5.0,
        help="Keep a triggered Fall alert visible for this long",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Render only this exact scenario name; repeat for multiple",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true", help="Do not draw pose landmarks"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select clips and write the manifest without rendering video",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs"
    )
    return parser.parse_args()


def scenario_from_stem(stem: str) -> str:
    return VIDEO_NUMBER_RE.sub("", stem)


def scenario_title(scenario: str) -> str:
    if scenario in SCENARIO_TITLES:
        return SCENARIO_TITLES[scenario]
    return scenario.replace("_", " ")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_runtime(
    checkpoint_path: Path, device: torch.device
) -> Tuple[FallDetectionModel, dict, np.ndarray, np.ndarray, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Model checkpoint not found: {}".format(checkpoint_path))
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    feature_pipeline = checkpoint_feature_pipeline(checkpoint)
    input_features = checkpoint_input_features(checkpoint)
    rnn_type = checkpoint_rnn_type(checkpoint)
    model = FallDetectionModel(
        input_size=input_features, rnn_type=rnn_type
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "scaler_mean" not in checkpoint or "scaler_std" not in checkpoint:
        scaler_path = MODEL_DIR / "scaler_params.npz"
        scaler = np.load(str(scaler_path))
        mean = scaler["mean"].astype(np.float32)
        std = scaler["std"].astype(np.float32)
    else:
        mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["scaler_std"], dtype=np.float32)
    if mean.shape != (input_features,) or std.shape != (input_features,):
        raise ValueError(
            "Scaler shape does not match checkpoint input width {}".format(
                input_features
            )
        )
    return model, checkpoint, mean, std, feature_pipeline


def discover_candidates(metadata_path: Path) -> List[Candidate]:
    if not metadata_path.is_file():
        raise FileNotFoundError("Subject metadata not found: {}".format(metadata_path))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", {})
    source_config = {item["name"]: item for item in TRAINING_DATASETS}
    candidates: List[Candidate] = []

    for key, item in sorted(videos.items()):
        if not item.get("present_in_raw", True):
            continue
        source = str(item["source"])
        label = str(item["class_label"])
        stem = str(item["video_stem"])
        subject_id = str(item.get("subject_id") or "UNKNOWN")
        config = source_config.get(source)
        if config is None:
            raise ValueError("Unknown dataset source in metadata: {}".format(source))

        video_path = Path(config["raw_root"]) / item["source_relative_path"]
        processed_dir = (
            Path(config["processed_adl_dir"])
            if label == "ADL"
            else Path(config["processed_fall_dir"])
        )
        landmarks_path = processed_dir / (stem + ".npy")
        if not video_path.is_file():
            raise FileNotFoundError("Raw video not found for {}: {}".format(key, video_path))
        if not landmarks_path.is_file():
            raise FileNotFoundError(
                "Processed landmarks not found for {}: {}".format(key, landmarks_path)
            )
        candidates.append(
            Candidate(
                key=key,
                source=source,
                stem=stem,
                scenario=scenario_from_stem(stem),
                label=label,
                subject_id=subject_id,
                video_path=video_path,
                landmarks_path=landmarks_path,
            )
        )
    return candidates


@torch.no_grad()
def predict_landmarks(
    raw_landmarks: np.ndarray,
    model: FallDetectionModel,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    feature_pipeline: str,
    threshold: float,
    batch_size: int,
) -> Tuple[np.ndarray, float, int, int]:
    if raw_landmarks.ndim != 2 or raw_landmarks.shape[1] != 132:
        raise ValueError(
            "Expected landmark array shaped (frames, 132), got {}".format(
                raw_landmarks.shape
            )
        )
    frame_probabilities = np.full(len(raw_landmarks), np.nan, dtype=np.float32)
    if len(raw_landmarks) < SEQUENCE_LENGTH:
        return frame_probabilities, 0.0, 0, 0

    pose_present = np.any(raw_landmarks != 0.0, axis=1)
    selected = filter_landmarks(raw_landmarks)
    valid_starts = [
        start
        for start in range(len(selected) - SEQUENCE_LENGTH + 1)
        if pose_present[start + SEQUENCE_LENGTH - 1]
    ]
    for batch_start in range(0, len(valid_starts), batch_size):
        starts = valid_starts[batch_start:batch_start + batch_size]
        windows = np.stack(
            [
                engineer_selected_features(
                    selected[start:start + SEQUENCE_LENGTH], feature_pipeline
                )
                for start in starts
            ],
            axis=0,
        ).astype(np.float32)
        windows = (windows - mean) / std
        tensor = torch.from_numpy(windows).to(device)
        logits, _ = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        for start, probability in zip(starts, probabilities):
            frame_probabilities[start + SEQUENCE_LENGTH - 1] = probability

    finite = frame_probabilities[np.isfinite(frame_probabilities)]
    maximum = float(finite.max()) if len(finite) else 0.0
    positive = int(np.sum(finite >= threshold))
    return frame_probabilities, maximum, positive, int(len(finite))


def predict_candidates(
    candidates: Sequence[Candidate],
    model: FallDetectionModel,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    feature_pipeline: str,
    threshold: float,
    batch_size: int,
) -> None:
    total = len(candidates)
    for index, candidate in enumerate(candidates, 1):
        raw_landmarks = np.load(str(candidate.landmarks_path)).astype(np.float32)
        probabilities, maximum, positive, window_count = predict_landmarks(
            raw_landmarks,
            model,
            device,
            mean,
            std,
            feature_pipeline,
            threshold,
            batch_size,
        )
        candidate.frame_probabilities = probabilities
        candidate.max_fall_probability = maximum
        candidate.positive_windows = positive
        candidate.total_windows = window_count
        candidate.prediction = "Fall" if maximum >= threshold else "ADL"
        if index == 1 or index % 20 == 0 or index == total:
            print("  Predicted {:>3}/{} videos".format(index, total), flush=True)


def select_balanced_candidates(
    candidates: Sequence[Candidate], scenarios: Optional[Iterable[str]] = None
) -> List[Candidate]:
    by_scenario: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[candidate.scenario].append(candidate)

    requested = set(scenarios or by_scenario.keys())
    unknown = requested.difference(by_scenario)
    if unknown:
        raise ValueError("Unknown scenario(s): {}".format(", ".join(sorted(unknown))))

    eligible: Dict[str, List[Candidate]] = {}
    unavailable: List[str] = []
    for scenario in requested:
        correct = [item for item in by_scenario[scenario] if item.correct]
        if not correct:
            unavailable.append(scenario)
        eligible[scenario] = correct
    if unavailable:
        raise RuntimeError(
            "No correctly predicted video is available for: {}".format(
                ", ".join(sorted(unavailable))
            )
        )

    # Handle constrained scenarios first, then choose the least-used available
    # subject. Within that subject, keep the strongest correct prediction.
    selection_order = sorted(
        requested,
        key=lambda scenario: (
            len({item.subject_id for item in eligible[scenario]}),
            len(eligible[scenario]),
            scenario,
        ),
    )
    subject_use: Counter = Counter()
    selected: Dict[str, Candidate] = {}
    for scenario in selection_order:
        subject_groups: Dict[str, List[Candidate]] = defaultdict(list)
        for item in eligible[scenario]:
            subject_groups[item.subject_id].append(item)
        best_per_subject = {
            subject: sorted(
                items,
                key=lambda item: (-item.quality_score, item.key),
            )[0]
            for subject, items in subject_groups.items()
        }
        chosen_subject = min(
            best_per_subject,
            key=lambda subject: (
                subject_use[subject],
                -best_per_subject[subject].quality_score,
                subject,
            ),
        )
        selected[scenario] = best_per_subject[chosen_subject]
        subject_use[chosen_subject] += 1

    return sorted(
        selected.values(),
        key=lambda item: (0 if item.label == "ADL" else 1, item.scenario),
    )


def load_fall_interval(candidate: Candidate) -> Optional[Tuple[int, int]]:
    if candidate.label != "Fall":
        return None
    source = next(
        item for item in TRAINING_DATASETS if item["name"] == candidate.source
    )
    annotation_path = Path(source["fall_annotations_path"])
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    items = payload.get("videos", payload.get("annotations", payload))
    item = items.get(candidate.stem)
    if not item:
        return None
    start = item.get("fall_start_frame", item.get("start_frame"))
    end = item.get("fall_end_frame", item.get("end_frame"))
    if start is None or end is None:
        return None
    return int(start), int(end)


def put_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    scale: float = 0.65,
    color: Tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = max(10, (frame.shape[1] - size[0]) // 2)
    put_text(frame, text, (x, y), scale, color, thickness)


def title_card(
    width: int,
    height: int,
    title: str,
    subtitle: str,
    detail: str,
) -> np.ndarray:
    frame = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (18, height), COLOR_CYAN, -1)
    centered_text(frame, title, int(height * 0.43), 1.25, COLOR_WHITE, 2)
    centered_text(frame, subtitle, int(height * 0.53), 0.72, COLOR_CYAN, 2)
    centered_text(frame, detail, int(height * 0.62), 0.55, COLOR_MUTED, 1)
    return frame


def draw_skeleton(frame: np.ndarray, raw_landmarks: np.ndarray) -> None:
    if raw_landmarks.shape != (132,) or not np.any(raw_landmarks):
        return
    landmarks = raw_landmarks.reshape(33, 4)
    height, width = frame.shape[:2]
    points = [
        (int(item[0] * width), int(item[1] * height)) for item in landmarks
    ]
    for start, end in POSE_CONNECTIONS:
        if landmarks[start, 3] > 0.3 and landmarks[end, 3] > 0.3:
            cv2.line(
                frame, points[start], points[end], COLOR_CYAN, 2, cv2.LINE_AA
            )
    for point, landmark in zip(points, landmarks):
        if landmark[3] > 0.3:
            cv2.circle(frame, point, 3, COLOR_YELLOW, -1, cv2.LINE_AA)
            cv2.circle(frame, point, 4, (15, 15, 15), 1, cv2.LINE_AA)


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (target_w, target_h), interpolation=interpolation)
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    x = (width - target_w) // 2
    y = (height - target_h) // 2
    canvas[y:y + target_h, x:x + target_w] = resized
    return canvas


def draw_detection_overlay(
    frame: np.ndarray,
    candidate: Candidate,
    frame_index: int,
    probability: float,
    threshold: float,
    alert_active: bool,
    interval: Optional[Tuple[int, int]],
    clip_number: int,
    clip_count: int,
) -> None:
    height, width = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (width, 112), COLOR_PANEL, -1)
    cv2.rectangle(panel, (0, height - 80), (width, height), COLOR_PANEL, -1)
    frame[:] = cv2.addWeighted(panel, 0.82, frame, 0.18, 0)

    ready = frame_index >= SEQUENCE_LENGTH - 1 and math.isfinite(probability)
    pose_visible = math.isfinite(probability) or frame_index < SEQUENCE_LENGTH - 1
    if alert_active:
        status = "FALL DETECTED"
        status_color = COLOR_RED
    elif ready:
        status = "ADL / NORMAL"
        status_color = COLOR_GREEN
    elif frame_index < SEQUENCE_LENGTH - 1:
        status = "BUFFERING"
        status_color = COLOR_ORANGE
    else:
        status = "POSE LOST"
        status_color = COLOR_ORANGE

    put_text(frame, "CNN-LSTM FALL DETECTION", (24, 32), 0.62, COLOR_CYAN, 2)
    put_text(frame, scenario_title(candidate.scenario), (24, 66), 0.83, COLOR_WHITE, 2)
    put_text(
        frame,
        "Subject {}  |  Ground truth {}  |  Clip {}/{}".format(
            candidate.subject_id, candidate.label, clip_number, clip_count
        ),
        (24, 96),
        0.52,
        COLOR_MUTED,
        1,
    )

    status_size, _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2
    )
    status_x = width - status_size[0] - 34
    cv2.rectangle(
        frame,
        (status_x - 12, 28),
        (width - 20, 72),
        status_color,
        -1,
    )
    put_text(frame, status, (status_x, 59), 0.78, COLOR_WHITE, 2)

    shown_probability = float(probability) if math.isfinite(probability) else 0.0
    bar_x, bar_y = 24, height - 43
    bar_w, bar_h = width - 48, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 75, 82), -1)
    fill_color = COLOR_RED if shown_probability >= threshold else COLOR_GREEN
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * shown_probability), bar_y + bar_h),
        fill_color,
        -1,
    )
    threshold_x = bar_x + int(bar_w * threshold)
    cv2.line(
        frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_h + 5),
        COLOR_WHITE, 2,
    )
    probability_text = (
        "Fall probability: {:.1%}   Threshold: {:.0%}".format(
            shown_probability, threshold
        )
        if ready
        else "Fall probability: waiting for a valid {}-frame window".format(
            SEQUENCE_LENGTH
        )
    )
    put_text(frame, probability_text, (bar_x, height - 53), 0.49, COLOR_WHITE, 1)

    if interval is not None and interval[0] <= frame_index <= interval[1]:
        label = "ANNOTATED FALL MOTION"
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        x = (width - size[0]) // 2
        cv2.rectangle(frame, (x - 12, 122), (x + size[0] + 12, 158), COLOR_ORANGE, -1)
        put_text(frame, label, (x, 148), 0.60, COLOR_WHITE, 2)


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create MP4 writer: {}".format(path))
    return writer


def write_repeated(writer: cv2.VideoWriter, frame: np.ndarray, count: int) -> None:
    for _ in range(max(0, count)):
        writer.write(frame)


def render_demo(
    selected: Sequence[Candidate],
    working_path: Path,
    width: int,
    height: int,
    fps: float,
    title_seconds: float,
    end_seconds: float,
    fall_hold_seconds: float,
    draw_pose: bool,
    threshold: float,
) -> None:
    writer = open_writer(working_path, fps, width, height)
    title_frames = int(round(title_seconds * fps))
    hold_frames = int(round(fall_hold_seconds * fps))
    try:
        opening = title_card(
            width,
            height,
            "Fall Detection System Demo",
            "1D CNN-LSTM + MediaPipe Pose",
            "One correctly predicted video from every ADL and Fall scenario",
        )
        write_repeated(writer, opening, max(title_frames * 2, 1))

        for clip_number, candidate in enumerate(selected, 1):
            print(
                "  Rendering {:>2}/{}: {} ({}, {})".format(
                    clip_number,
                    len(selected),
                    candidate.scenario,
                    candidate.subject_id,
                    candidate.key,
                ),
                flush=True,
            )
            card = title_card(
                width,
                height,
                scenario_title(candidate.scenario),
                "Subject {} | Correct prediction: {}".format(
                    candidate.subject_id, candidate.prediction
                ),
                "Peak Fall probability {:.1%}".format(
                    candidate.max_fall_probability
                ),
            )
            write_repeated(writer, card, title_frames)

            landmarks = np.load(str(candidate.landmarks_path)).astype(np.float32)
            capture = cv2.VideoCapture(str(candidate.video_path))
            if not capture.isOpened():
                raise RuntimeError("Could not open video: {}".format(candidate.video_path))
            interval = load_fall_interval(candidate)
            last_trigger = -10**9
            frame_index = 0
            try:
                while True:
                    ok, source_frame = capture.read()
                    if not ok:
                        break
                    probability = (
                        float(candidate.frame_probabilities[frame_index])
                        if frame_index < len(candidate.frame_probabilities)
                        else float("nan")
                    )
                    if math.isfinite(probability) and probability >= threshold:
                        last_trigger = frame_index
                    alert_active = frame_index - last_trigger <= hold_frames
                    if alert_active:
                        tint = np.full_like(source_frame, COLOR_RED)
                        source_frame = cv2.addWeighted(
                            tint, 0.10, source_frame, 0.90, 0
                        )
                    if draw_pose and frame_index < len(landmarks):
                        draw_skeleton(source_frame, landmarks[frame_index])
                    frame = letterbox(source_frame, width, height)
                    draw_detection_overlay(
                        frame,
                        candidate,
                        frame_index,
                        probability,
                        threshold,
                        alert_active,
                        interval,
                        clip_number,
                        len(selected),
                    )
                    writer.write(frame)
                    frame_index += 1
            finally:
                capture.release()
            if frame_index == 0:
                raise RuntimeError("Video has no readable frames: {}".format(candidate.video_path))

        counts = Counter(item.subject_id for item in selected)
        closing = title_card(
            width,
            height,
            "Demo Complete",
            "{} scenarios | {} subjects".format(len(selected), len(counts)),
            "Selection: " + ", ".join(
                "{}={} clips".format(subject, count)
                for subject, count in sorted(counts.items())
            ),
        )
        write_repeated(writer, closing, max(1, int(round(end_seconds * fps))))
    finally:
        writer.release()


def finalize_mp4(working_path: Path, output_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(working_path),
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            working_path.unlink()
            return "H.264 (ffmpeg)"
        print("  ffmpeg transcode failed; keeping OpenCV MPEG-4 output")
    os.replace(str(working_path), str(output_path))
    return "MPEG-4 Part 2 (OpenCV)"


def candidate_manifest(candidate: Candidate) -> dict:
    return {
        "scenario": candidate.scenario,
        "title": scenario_title(candidate.scenario),
        "dataset_key": candidate.key,
        "source": candidate.source,
        "video_stem": candidate.stem,
        "subject_id": candidate.subject_id,
        "ground_truth": candidate.label,
        "prediction": candidate.prediction,
        "correct": candidate.correct,
        "max_fall_probability": candidate.max_fall_probability,
        "positive_windows": candidate.positive_windows,
        "total_windows": candidate.total_windows,
        "video_path": str(candidate.video_path.relative_to(PROJECT_ROOT)),
        "landmarks_path": str(candidate.landmarks_path.relative_to(PROJECT_ROOT)),
    }


def write_manifest(
    path: Path,
    selected: Sequence[Candidate],
    model_path: Path,
    threshold: float,
    output_path: Path,
    codec: Optional[str],
) -> None:
    counts = Counter(item.subject_id for item in selected)
    payload = {
        "description": (
            "One event-level correctly predicted video per ADL/Fall scenario, "
            "balanced across available subject IDs"
        ),
        "model": str(model_path),
        "threshold": threshold,
        "selection_scope": "all raw dataset videos (demonstration, not evaluation)",
        "output_video": None if codec is None else str(output_path),
        "codec": codec,
        "scenario_count": len(selected),
        "subject_counts": dict(sorted(counts.items())),
        "selected_videos": [candidate_manifest(item) for item in selected],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_options(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.width < 320 or args.height < 240:
        raise ValueError("Output resolution must be at least 320x240")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 output width and height must be even")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.title_seconds < 0 or args.end_seconds < 0 or args.fall_hold_seconds < 0:
        raise ValueError("Durations cannot be negative")


def main() -> int:
    args = parse_args()
    validate_options(args)
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else args.output.with_suffix(".json")
    )
    targets = [manifest_path] if args.dry_run else [args.output, manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output exists (pass --overwrite to replace): {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    device = resolve_device(args.device)
    print("Loading model on {}...".format(device))
    model, _checkpoint, mean, std, feature_pipeline = load_runtime(
        args.model, device
    )
    candidates = discover_candidates(SUBJECT_METADATA_PATH)
    if args.scenario:
        wanted = set(args.scenario)
        candidates_to_predict = [
            item for item in candidates if item.scenario in wanted
        ]
    else:
        wanted = None
        candidates_to_predict = candidates
    print(
        "Running event inference for {} candidate videos...".format(
            len(candidates_to_predict)
        )
    )
    predict_candidates(
        candidates_to_predict,
        model,
        device,
        mean,
        std,
        feature_pipeline,
        args.threshold,
        args.batch_size,
    )
    selected = select_balanced_candidates(candidates_to_predict, wanted)
    counts = Counter(item.subject_id for item in selected)
    print("Selected {} scenarios:".format(len(selected)))
    for item in selected:
        print(
            "  {:<28} {:<3} {:<4} max={:>6.1%} {}".format(
                item.scenario,
                item.subject_id,
                item.prediction,
                item.max_fall_probability,
                item.key,
            )
        )
    print(
        "Subject balance: " + ", ".join(
            "{}={}".format(subject, count)
            for subject, count in sorted(counts.items())
        )
    )

    if args.dry_run:
        write_manifest(
            manifest_path,
            selected,
            args.model,
            args.threshold,
            args.output,
            codec=None,
        )
        print("Dry run complete. Manifest: {}".format(manifest_path))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    working_path = args.output.with_name(args.output.stem + ".working.mp4")
    if working_path.exists():
        working_path.unlink()
    try:
        print("Rendering MP4...")
        render_demo(
            selected,
            working_path,
            args.width,
            args.height,
            args.fps,
            args.title_seconds,
            args.end_seconds,
            args.fall_hold_seconds,
            not args.no_skeleton,
            args.threshold,
        )
        codec = finalize_mp4(working_path, args.output)
    except Exception:
        if working_path.exists():
            working_path.unlink()
        raise
    write_manifest(
        manifest_path,
        selected,
        args.model,
        args.threshold,
        args.output,
        codec,
    )
    print("Created: {}".format(args.output))
    print("Manifest: {}".format(manifest_path))
    print("Codec: {}".format(codec))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
