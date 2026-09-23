#!/usr/bin/env python3
"""Create one correctly predicted MP4 per ADL/Fall scenario.

Unlike ``create_prediction_demo.py``, this script does not concatenate clips.
Each output video uses the original detector overlay from ``src/inference.py``
(``FallDetector.draw_overlay``), including its pose skeleton, buffer state,
prediction panel, Fall-probability bar, and alert UI.

Examples:
    python scripts/create_scenario_videos.py
    python scripts/create_scenario_videos.py --overwrite
    python scripts/create_scenario_videos.py --scenario Fall_ke_depan --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_prediction_demo import (  # noqa: E402
    Candidate,
    candidate_manifest,
    discover_candidates,
    load_runtime,
    predict_candidates,
    resolve_device,
    select_balanced_candidates,
)
from src.config import (  # noqa: E402
    FALL_CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    SEQUENCE_LENGTH,
    SUBJECT_METADATA_PATH,
)
from src.inference import FallDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one correctly predicted video per ADL/Fall scenario and "
            "write each scenario as a separate MP4 using the runtime UI."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "scenario_videos",
        help="Directory for separate MP4 files (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest (default: <output-dir>/manifest.json)",
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
        help="Fall alert threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--fall-hold-seconds",
        type=float,
        default=5.0,
        help="Keep a detected Fall alert active for this many video seconds",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Render only this exact scenario; repeat for multiple scenarios",
    )
    parser.add_argument(
        "--no-skeleton", action="store_true", help="Hide the pose skeleton"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select videos and write the manifest without rendering MP4 files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.fall_hold_seconds < 0:
        raise ValueError("--fall-hold-seconds cannot be negative")


def landmarks_for_ui(raw_frame: np.ndarray) -> Optional[List[SimpleNamespace]]:
    """Convert the saved 132-value pose into the objects expected by the UI."""
    if raw_frame.shape != (132,) or not np.any(raw_frame):
        return None
    points = raw_frame.reshape(33, 4)
    return [
        SimpleNamespace(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            visibility=float(point[3]),
        )
        for point in points
    ]


def create_ui_renderer(threshold: float) -> FallDetector:
    """Create only the drawing portion of FallDetector without reloading a model."""
    renderer = object.__new__(FallDetector)
    renderer.confidence_threshold = float(threshold)
    renderer.flash_start_time = 0.0
    renderer.mute = True
    return renderer


def runtime_result(
    candidate: Candidate,
    raw_landmarks: np.ndarray,
    frame_index: int,
    threshold: float,
    hold_frames: int,
    pose_streak: int,
    last_alert_frame: int,
    previous_prediction: str,
    previous_confidence: float,
) -> tuple[dict, int, str, float]:
    """Build the same result fields consumed by FallDetector.draw_overlay()."""
    pose_detected = bool(np.any(raw_landmarks))
    buffer_filled = min(frame_index + 1, SEQUENCE_LENGTH)
    probability = (
        float(candidate.frame_probabilities[frame_index])
        if frame_index < len(candidate.frame_probabilities)
        else float("nan")
    )
    evaluated = pose_detected and math.isfinite(probability)
    fall_eligible = bool(
        evaluated
        and buffer_filled >= SEQUENCE_LENGTH
        and pose_streak >= 3
    )
    model_is_fall = bool(
        evaluated and probability >= threshold
    )
    if model_is_fall and fall_eligible:
        last_alert_frame = frame_index
    is_fall = bool(
        last_alert_frame >= 0
        and frame_index - last_alert_frame <= hold_frames
    )

    prediction = previous_prediction
    confidence = previous_confidence
    if evaluated:
        prediction = "Fall" if probability >= 0.5 else "ADL"
        confidence = max(probability, 1.0 - probability)

    if pose_detected:
        if buffer_filled >= SEQUENCE_LENGTH and not fall_eligible:
            display_prediction = "Menstabilkan pose..."
            display_confidence = 0.0
        else:
            display_prediction = prediction
            display_confidence = confidence
    elif is_fall:
        display_prediction = "Fall (pose hilang)"
        display_confidence = 0.0
    else:
        display_prediction = "Tidak Terdeteksi"
        display_confidence = 0.0

    result = {
        "prediction": display_prediction,
        "confidence": display_confidence,
        "is_fall": is_fall,
        "buffer_filled": buffer_filled,
        "buffer_total": SEQUENCE_LENGTH,
        "fall_prob": probability if fall_eligible else 0.0,
        "model_is_fall": model_is_fall,
        "model_evaluated": evaluated,
        "fall_eligible": fall_eligible,
        "pose_detected": pose_detected,
        "pose_missing_frames": 0 if pose_detected else 1,
        "pose_detected_streak": pose_streak,
        "pose_grace_active": pose_detected,
        "buffer_reprimed": False,
        "fall_latched": bool(is_fall and not (model_is_fall and fall_eligible)),
    }
    return result, last_alert_frame, prediction, confidence


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create video writer: {}".format(path))
    return writer


def transcode_h264(working_path: Path, output_path: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        completed = subprocess.run(
            [
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
            ],
            check=False,
        )
        if completed.returncode == 0:
            working_path.unlink()
            return "H.264 (ffmpeg)"
        print("    ffmpeg failed; retaining OpenCV MPEG-4 output")
    working_path.replace(output_path)
    return "MPEG-4 Part 2 (OpenCV)"


def render_scenario(
    candidate: Candidate,
    output_path: Path,
    threshold: float,
    fall_hold_seconds: float,
    draw_skeleton: bool,
) -> dict:
    landmarks = np.load(str(candidate.landmarks_path)).astype(np.float32)
    capture = cv2.VideoCapture(str(candidate.video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open source video: {}".format(candidate.video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video resolution: {}".format(candidate.video_path))

    working_path = output_path.with_name("." + output_path.stem + ".working.mp4")
    if working_path.exists():
        working_path.unlink()
    writer = open_writer(working_path, fps, width, height)
    ui = create_ui_renderer(threshold)
    hold_frames = int(round(fall_hold_seconds * fps))
    pose_streak = 0
    last_alert_frame = -1
    flash_start_frame = -1
    previous_is_fall = False
    previous_prediction = "Menunggu..."
    previous_confidence = 0.0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            raw_frame = (
                landmarks[frame_index]
                if frame_index < len(landmarks)
                else np.zeros(132, dtype=np.float32)
            )
            pose = landmarks_for_ui(raw_frame)
            pose_streak = pose_streak + 1 if pose is not None else 0
            result, new_alert_frame, previous_prediction, previous_confidence = (
                runtime_result(
                    candidate,
                    raw_frame,
                    frame_index,
                    threshold,
                    hold_frames,
                    pose_streak,
                    last_alert_frame,
                    previous_prediction,
                    previous_confidence,
                )
            )
            if new_alert_frame != last_alert_frame:
                last_alert_frame = new_alert_frame
            if result["is_fall"] and not previous_is_fall:
                flash_start_frame = frame_index
            previous_is_fall = bool(result["is_fall"])
            if result["is_fall"] and flash_start_frame >= 0:
                # draw_overlay uses wall-clock time for its flash. Convert the
                # video-frame age to wall time so offline output has the same
                # half-second red flash duration as live inference.
                ui.flash_start_time = time.time() - (
                    (frame_index - flash_start_frame) / fps
                )
            frame = ui.draw_overlay(
                frame,
                result,
                landmarks=pose if draw_skeleton else None,
                fps=fps,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        if working_path.exists():
            working_path.unlink()
        raise RuntimeError("Source video contains no readable frames: {}".format(candidate.video_path))
    codec = transcode_h264(working_path, output_path)
    return {
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "codec": codec,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_index,
        "duration_seconds": frame_index / fps,
    }


def write_manifest(
    path: Path,
    selected: Sequence[Candidate],
    rendered: dict[str, dict],
    model_path: Path,
    threshold: float,
    output_dir: Path,
) -> None:
    subject_counts = Counter(item.subject_id for item in selected)
    items = []
    for candidate in selected:
        item = candidate_manifest(candidate)
        item["rendered_video"] = rendered.get(candidate.scenario)
        items.append(item)
    payload = {
        "description": (
            "Separate correctly predicted videos for every ADL/Fall scenario, "
            "rendered with src.inference.FallDetector.draw_overlay"
        ),
        "model": str(model_path),
        "threshold": threshold,
        "output_directory": str(output_dir),
        "scenario_count": len(selected),
        "subject_counts": dict(sorted(subject_counts.items())),
        "selected_videos": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    model_path = args.model.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_dir / "manifest.json"
    )

    device = resolve_device(args.device)
    print("Loading model on {}...".format(device))
    model, _checkpoint, mean, std, feature_pipeline = load_runtime(
        model_path, device
    )
    candidates = discover_candidates(SUBJECT_METADATA_PATH)
    wanted = set(args.scenario) if args.scenario else None
    candidates_to_predict = (
        [item for item in candidates if item.scenario in wanted]
        if wanted is not None
        else candidates
    )
    if not candidates_to_predict:
        raise ValueError("No videos matched the requested scenario(s)")
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

    output_paths = {
        item.scenario: output_dir / (item.scenario + ".mp4")
        for item in selected
    }
    targets = [manifest_path]
    if not args.dry_run:
        targets.extend(output_paths.values())
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output exists (pass --overwrite to replace): {}".format(
                ", ".join(str(path) for path in existing)
            )
        )

    print("Selected {} correctly predicted scenarios:".format(len(selected)))
    for candidate in selected:
        print(
            "  {:<28} {:<3} {:<4} max={:>6.1%} {}".format(
                candidate.scenario,
                candidate.subject_id,
                candidate.prediction,
                candidate.max_fall_probability,
                candidate.key,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, dict] = {}
    if not args.dry_run:
        for index, candidate in enumerate(selected, 1):
            output_path = output_paths[candidate.scenario]
            print(
                "Rendering {:>2}/{}: {} -> {}".format(
                    index, len(selected), candidate.scenario, output_path.name
                ),
                flush=True,
            )
            rendered[candidate.scenario] = render_scenario(
                candidate,
                output_path,
                args.threshold,
                args.fall_hold_seconds,
                not args.no_skeleton,
            )

    write_manifest(
        manifest_path,
        selected,
        rendered,
        model_path,
        args.threshold,
        output_dir,
    )
    print("Created {} separate MP4 files in {}".format(len(rendered), output_dir))
    print("Manifest: {}".format(manifest_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
