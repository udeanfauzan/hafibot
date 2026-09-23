#!/usr/bin/env python3
"""Interactively mark the true falling interval in every Fall video.

Controls:
    SPACE       play / pause
    , and .     previous / next frame
    j and l     jump backward / forward five frames
    [ and ]     mark Fall start / end
    s           save annotation and continue
    r           clear current marks
    n           skip this video
    q or ESC    quit
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_DIR,
    FALL_ANNOTATIONS_PATH,
    RAW_FALL_DIR,
)
from src.temporal_labels import validate_fall_annotation  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_videos(source_root, filter_fall_names=False):
    videos = sorted(
        path for path in source_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and (
                not filter_fall_names
                or path.stem.lower().startswith("fall")
                or any(
                    part.lower().startswith("fall")
                    for part in path.relative_to(source_root).parts[:-1]
                )
            )
        )
    )
    stems = {}
    for path in videos:
        if path.stem in stems:
            raise ValueError(
                "Nama video stem harus unik: {} dan {}".format(
                    stems[path.stem], path
                )
            )
        stems[path.stem] = path
    return videos, stems


def portable_source_root(source_root):
    try:
        return source_root.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source_root.as_posix()


def default_annotation_path(source_root, custom_dataset):
    if not custom_dataset:
        return FALL_ANNOTATIONS_PATH
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", source_root.name).strip("_")
    if not slug:
        slug = "custom_dataset"
    return DATA_DIR / ("fall_annotations_{}.json".format(slug.lower()))


def load_payload(path, source_root):
    expected_source_root = portable_source_root(source_root)
    if not path.exists():
        return {
            "schema_version": 1,
            "description": (
                "Inclusive frame interval where the person is actively falling"
            ),
            "source_root": expected_source_root,
            "videos": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("videos"), dict
    ):
        raise ValueError("Format annotation tidak valid: " + str(path))
    stored_source_root = payload.get("source_root")
    if stored_source_root and stored_source_root != expected_source_root:
        raise ValueError(
            "Annotation file dibuat untuk source_root '{}', bukan '{}'."
            .format(stored_source_root, expected_source_root)
        )
    payload["source_root"] = expected_source_root
    return payload


def save_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def video_info(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Tidak bisa membuka video: " + str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0.0 or frame_count <= 0:
        raise ValueError("FPS/frame count tidak valid: " + str(video_path))
    return fps, frame_count


def make_annotation(video_path, source_root, fps, frame_count,
                    start_frame, end_frame):
    annotation = {
        "video_stem": video_path.stem,
        "source_relative_path": video_path.relative_to(source_root).as_posix(),
        "source_fps": float(fps),
        "source_frame_count": int(frame_count),
        "fall_start_frame": int(start_frame),
        "fall_end_frame": int(end_frame),
        "fall_start_sec": float(start_frame / fps),
        "fall_end_sec": float(end_frame / fps),
        "updated_at_unix": time.time(),
    }
    validate_fall_annotation(video_path.stem, annotation, frame_count)
    return annotation


def draw_annotation_overlay(frame, video_path, frame_index, fps, frame_count,
                            start_frame, end_frame, playing):
    canvas = frame.copy()
    lines = [
        video_path.name,
        "Frame {}/{} | {:.2f}s / {:.2f}s | {}".format(
            frame_index, frame_count - 1, frame_index / fps,
            (frame_count - 1) / fps, "PLAY" if playing else "PAUSE"
        ),
        "Fall start: {} | Fall end: {}".format(
            "--" if start_frame is None else "{} ({:.2f}s)".format(
                start_frame, start_frame / fps
            ),
            "--" if end_frame is None else "{} ({:.2f}s)".format(
                end_frame, end_frame / fps
            ),
        ),
        "SPACE play | ,/. step | j/l jump | [ start | ] end | s save | r reset | n skip | q quit",
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 112), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line, (12, 24 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    timeline_y = canvas.shape[0] - 20
    timeline_left = 15
    timeline_right = canvas.shape[1] - 15
    cv2.line(
        canvas, (timeline_left, timeline_y), (timeline_right, timeline_y),
        (180, 180, 180), 3,
    )
    denominator = max(frame_count - 1, 1)
    if start_frame is not None:
        start_x = timeline_left + int(
            (timeline_right - timeline_left) * start_frame / denominator
        )
        cv2.line(canvas, (start_x, timeline_y - 12),
                 (start_x, timeline_y + 5), (0, 255, 255), 3)
    if end_frame is not None:
        end_x = timeline_left + int(
            (timeline_right - timeline_left) * end_frame / denominator
        )
        cv2.line(canvas, (end_x, timeline_y - 12),
                 (end_x, timeline_y + 5), (0, 80, 255), 3)
    current_x = timeline_left + int(
        (timeline_right - timeline_left) * frame_index / denominator
    )
    cv2.circle(canvas, (current_x, timeline_y), 6, (0, 255, 0), -1)
    return canvas


def annotate_video(video_path, source_root, existing=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print("[ERROR] Tidak bisa membuka: " + str(video_path))
        return "skip", None

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = existing.get("fall_start_frame") if existing else None
    end_frame = existing.get("fall_end_frame") if existing else None
    frame_index = int(start_frame or 0)
    playing = False
    window_name = "Fall interval annotation"

    while True:
        frame_index = max(0, min(frame_count - 1, frame_index))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            playing = False
            frame_index = max(0, frame_index - 1)
            continue

        display = draw_annotation_overlay(
            frame, video_path, frame_index, fps, frame_count,
            start_frame, end_frame, playing,
        )
        cv2.imshow(window_name, display)
        delay_ms = max(1, int(round(1000.0 / fps))) if playing else 0
        key = cv2.waitKey(delay_ms) & 0xFF

        if playing and key == 255:
            if frame_index >= frame_count - 1:
                playing = False
            else:
                frame_index += 1
            continue

        if key == ord(" "):
            playing = not playing
        elif key == ord(","):
            playing = False
            frame_index -= 1
        elif key == ord("."):
            playing = False
            frame_index += 1
        elif key == ord("j"):
            playing = False
            frame_index -= 5
        elif key == ord("l"):
            playing = False
            frame_index += 5
        elif key == ord("["):
            start_frame = frame_index
        elif key == ord("]"):
            end_frame = frame_index
        elif key == ord("r"):
            start_frame = None
            end_frame = None
        elif key == ord("s"):
            if start_frame is None or end_frame is None:
                print("  Tandai start '[' dan end ']' sebelum save.")
                continue
            if end_frame < start_frame:
                print("  End Fall tidak boleh sebelum start Fall.")
                continue
            annotation = make_annotation(
                video_path, source_root, fps, frame_count,
                start_frame, end_frame
            )
            capture.release()
            cv2.destroyWindow(window_name)
            return "save", annotation
        elif key == ord("n"):
            capture.release()
            cv2.destroyWindow(window_name)
            return "skip", None
        elif key in (ord("q"), 27):
            capture.release()
            cv2.destroyAllWindows()
            return "quit", None


def find_video(query, stems):
    query_stem = Path(query).stem
    if query_stem in stems:
        return stems[query_stem]
    matches = [path for stem, path in stems.items() if query.lower() in stem.lower()]
    if len(matches) != 1:
        raise ValueError(
            "Video query '{}' cocok dengan {} file; gunakan stem lengkap."
            .format(query, len(matches))
        )
    return matches[0]


def validate_all(videos, payload):
    missing = []
    invalid = []
    for video_path in videos:
        annotation = payload["videos"].get(video_path.stem)
        if annotation is None:
            missing.append(video_path.stem)
            continue
        try:
            _, frame_count = video_info(video_path)
            validate_fall_annotation(video_path.stem, annotation, frame_count)
        except Exception as error:
            invalid.append((video_path.stem, str(error)))

    print("Annotated: {}/{}".format(len(videos) - len(missing), len(videos)))
    if missing:
        print("Missing ({}):".format(len(missing)))
        for stem in missing:
            print("  " + stem)
    if invalid:
        print("Invalid ({}):".format(len(invalid)))
        for stem, error in invalid:
            print("  {}: {}".format(stem, error))
    return not missing and not invalid


def main():
    parser = argparse.ArgumentParser(
        description="Mark the true falling interval in Fall videos"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Custom dataset root; only videos in Fall* folders or with Fall* "
            "filenames are selected"
        ),
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        help="Output JSON; defaults to a dataset-specific file under data/",
    )
    parser.add_argument("--video", help="Annotate one video stem")
    parser.add_argument("--all", action="store_true",
                        help="Review annotated videos too")
    parser.add_argument("--list", action="store_true",
                        help="List annotation status and exit")
    parser.add_argument("--validate", action="store_true",
                        help="Fail unless every Fall video is validly annotated")
    parser.add_argument(
        "--set", nargs=3, metavar=("VIDEO", "START_SEC", "END_SEC"),
        dest="set_interval",
        help="Set one interval non-interactively using seconds",
    )
    args = parser.parse_args()

    custom_dataset = args.dataset_dir is not None
    source_root = (
        args.dataset_dir.resolve() if custom_dataset else RAW_FALL_DIR.resolve()
    )
    if not source_root.is_dir():
        raise ValueError("Dataset directory tidak ditemukan: " + str(source_root))
    annotation_path = (
        args.annotation_file.resolve()
        if args.annotation_file is not None
        else default_annotation_path(source_root, custom_dataset).resolve()
    )
    videos, stems = discover_videos(
        source_root, filter_fall_names=custom_dataset
    )
    if not videos:
        raise ValueError("Tidak ada Fall video ditemukan di: " + str(source_root))
    payload = load_payload(annotation_path, source_root)

    if args.list or args.validate:
        valid = validate_all(videos, payload)
        if args.validate and not valid:
            return 1
        if args.list:
            return 0

    if args.set_interval:
        query, start_text, end_text = args.set_interval
        video_path = find_video(query, stems)
        fps, frame_count = video_info(video_path)
        start_frame = int(round(float(start_text) * fps))
        end_frame = int(round(float(end_text) * fps))
        annotation = make_annotation(
            video_path, source_root, fps, frame_count, start_frame, end_frame
        )
        payload["videos"][video_path.stem] = annotation
        save_payload(annotation_path, payload)
        print(
            "Saved {}: frames {}..{} ({:.2f}s..{:.2f}s)".format(
                video_path.stem, start_frame, end_frame,
                start_frame / fps, end_frame / fps,
            )
        )
        return 0

    if args.video:
        selected_videos = [find_video(args.video, stems)]
    elif args.all:
        selected_videos = videos
    else:
        selected_videos = [
            path for path in videos if path.stem not in payload["videos"]
        ]

    if not selected_videos:
        print("Semua Fall video sudah dianotasi. Jalankan --validate untuk cek.")
        return 0

    print("Annotation file: " + str(annotation_path))
    print("Dataset root: " + str(source_root))
    print("Videos to review: {}".format(len(selected_videos)))
    for index, video_path in enumerate(selected_videos, 1):
        print("[{}/{}] {}".format(index, len(selected_videos), video_path.name))
        action, annotation = annotate_video(
            video_path, source_root,
            payload["videos"].get(video_path.stem)
        )
        if action == "quit":
            break
        if action == "save":
            payload["videos"][video_path.stem] = annotation
            save_payload(annotation_path, payload)
            print(
                "  Saved: frame {}..{}".format(
                    annotation["fall_start_frame"],
                    annotation["fall_end_frame"],
                )
            )

    cv2.destroyAllWindows()
    validate_all(videos, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
