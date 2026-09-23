#!/usr/bin/env python3
"""Initialize, assign, inspect, and validate video subject metadata."""

import argparse
import fnmatch
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUBJECT_METADATA_PATH, TRAINING_DATASETS  # noqa: E402
from src.subject_metadata import (  # noqa: E402
    discover_training_videos,
    load_subject_metadata,
    resolve_subject_ids,
    save_subject_metadata,
    subject_summary,
    sync_subject_metadata,
    validate_subject_id,
)


def filtered_video_ids(payload, source=None, pattern=None, videos=None,
                       unassigned_only=False):
    entries = payload["videos"]
    if videos:
        unknown = [video_id for video_id in videos if video_id not in entries]
        if unknown:
            raise ValueError(
                "Video ID tidak ditemukan: " + ", ".join(unknown)
            )
        candidates = list(videos)
    else:
        candidates = sorted(entries)
    selected = []
    for video_id in candidates:
        entry = entries[video_id]
        if not entry.get("present_in_raw", True):
            continue
        if source and entry["source"] != source:
            continue
        if pattern and not any(fnmatch.fnmatch(value, pattern) for value in (
            video_id, entry["video_stem"], entry["source_relative_path"]
        )):
            continue
        if unassigned_only and entry.get("subject_id"):
            continue
        selected.append(video_id)
    return selected


def print_entries(payload, video_ids):
    for video_id in video_ids:
        entry = payload["videos"][video_id]
        print("{} | {:4s} | {:12s} | {}".format(
            video_id,
            entry["class_label"],
            entry.get("subject_id") or "UNASSIGNED",
            entry["source_relative_path"],
        ))
    print("Total: {}".format(len(video_ids)))


def print_summary(payload):
    summary = subject_summary(payload)
    print("Subject metadata: {}".format(SUBJECT_METADATA_PATH))
    print("  Videos     : {}".format(summary["total"]))
    print("  Assigned   : {}".format(summary["assigned"]))
    print("  Unassigned : {}".format(summary["unassigned"]))
    print("  Stale      : {}".format(summary["stale"]))
    print("  Sources    : {}".format(summary["sources"]))
    print("  Subjects   : {}".format(summary["subjects"]))


def raw_video_path(entry):
    """Resolve one manifest entry back to its raw video file."""
    raw_roots = {
        source["name"]: Path(source["raw_root"])
        for source in TRAINING_DATASETS
    }
    raw_root = raw_roots.get(entry["source"])
    if raw_root is None:
        raise ValueError("Source tidak dikenal: " + entry["source"])
    video_path = raw_root / Path(entry["source_relative_path"])
    if not video_path.is_file():
        raise FileNotFoundError("Video tidak ditemukan: " + str(video_path))
    return video_path


def draw_subject_overlay(frame, video_id, frame_index, frame_count, fps,
                         playing, selected_subject, subject_choices,
                         item_index, item_count):
    """Draw assignment controls without modifying the source video."""
    canvas = frame.copy()
    max_width, max_height = 1280, 800
    scale = min(
        1.0,
        max_width / float(max(canvas.shape[1], 1)),
        max_height / float(max(canvas.shape[0], 1)),
    )
    if scale < 1.0:
        canvas = cv2.resize(
            canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    shortcut_text = " | ".join(
        "{}={}".format(index + 1, subject_id)
        for index, subject_id in enumerate(subject_choices)
    )
    lines = [
        "Video {}/{} | {}".format(item_index, item_count, video_id),
        "Frame {}/{} | {:.2f}s / {:.2f}s | {}".format(
            frame_index, frame_count - 1, frame_index / fps,
            (frame_count - 1) / fps, "PLAY" if playing else "PAUSE",
        ),
        "Selected subject: {}".format(selected_subject or "UNASSIGNED"),
        shortcut_text,
        "SPACE play | ,/. step | j/l jump | s save+next | n skip | b previous | q quit",
    ]
    overlay = canvas.copy()
    overlay_height = min(canvas.shape[0], 140)
    cv2.rectangle(
        overlay, (0, 0), (canvas.shape[1], overlay_height), (0, 0, 0), -1
    )
    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
    for index, line in enumerate(lines):
        color = (0, 255, 255) if index == 2 else (255, 255, 255)
        cv2.putText(
            canvas, line, (12, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
        )

    timeline_y = canvas.shape[0] - 18
    timeline_left = 15
    timeline_right = canvas.shape[1] - 15
    cv2.line(
        canvas, (timeline_left, timeline_y), (timeline_right, timeline_y),
        (180, 180, 180), 3,
    )
    current_x = timeline_left + int(
        (timeline_right - timeline_left)
        * frame_index / max(frame_count - 1, 1)
    )
    cv2.circle(canvas, (current_x, timeline_y), 6, (0, 255, 0), -1)
    return canvas


def annotate_subject_video(video_id, entry, subject_choices,
                           item_index, item_count):
    """Interactively inspect one video and return navigation/save action."""
    video_path = raw_video_path(entry)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print("[ERROR] Tidak bisa membuka: " + str(video_path))
        return "skip", entry.get("subject_id")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0.0 or frame_count <= 0:
        capture.release()
        print("[ERROR] FPS/frame count tidak valid: " + str(video_path))
        return "skip", entry.get("subject_id")

    frame_index = 0
    playing = False
    selected_subject = entry.get("subject_id")
    window_name = "Subject assignment"

    while True:
        frame_index = max(0, min(frame_count - 1, frame_index))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            playing = False
            frame_index = max(0, frame_index - 1)
            continue

        display = draw_subject_overlay(
            frame, video_id, frame_index, frame_count, fps, playing,
            selected_subject, subject_choices, item_index, item_count,
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
        elif ord("1") <= key <= ord("9"):
            choice_index = key - ord("1")
            if choice_index < len(subject_choices):
                selected_subject = subject_choices[choice_index]
        elif key == ord("s"):
            if not selected_subject:
                print("  Pilih subject dengan tombol angka sebelum save.")
                continue
            capture.release()
            cv2.destroyWindow(window_name)
            return "save", selected_subject
        elif key == ord("n"):
            capture.release()
            cv2.destroyWindow(window_name)
            return "skip", selected_subject
        elif key == ord("b"):
            capture.release()
            cv2.destroyWindow(window_name)
            return "previous", selected_subject
        elif key in (ord("q"), 27):
            capture.release()
            cv2.destroyAllWindows()
            return "quit", selected_subject


def run_interactive_annotation(payload, video_ids, subject_choices):
    """Review selected videos, saving atomically after every assignment."""
    if not video_ids:
        print("Tidak ada video yang perlu diberi subject ID.")
        return
    print("Subject choices: {}".format(
        ", ".join(
            "{}={}".format(index + 1, subject_id)
            for index, subject_id in enumerate(subject_choices)
        )
    ))
    print("Videos to review: {}".format(len(video_ids)))
    index = 0
    while 0 <= index < len(video_ids):
        video_id = video_ids[index]
        entry = payload["videos"][video_id]
        print("[{}/{}] {}".format(index + 1, len(video_ids), video_id))
        action, subject_id = annotate_subject_video(
            video_id, entry, subject_choices, index + 1, len(video_ids)
        )
        if action == "quit":
            break
        if action == "previous":
            index = max(0, index - 1)
            continue
        if action == "save":
            entry["subject_id"] = validate_subject_id(subject_id)
            entry["subject_updated_at_unix"] = time.time()
            payload["updated_at_unix"] = time.time()
            save_subject_metadata(payload)
            print("  Saved: {} -> {}".format(video_id, subject_id))
        index += 1
    cv2.destroyAllWindows()
    print_summary(payload)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Manage pseudonymous subject IDs for all Fall/ADL source videos"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--init", action="store_true",
        help="Discover videos and create/update the manifest",
    )
    action.add_argument("--summary", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--list-unassigned", action="store_true")
    action.add_argument(
        "--annotate", action="store_true",
        help="Open videos interactively and assign subjects with number keys",
    )
    action.add_argument(
        "--set", metavar="SUBJECT_ID",
        help="Assign one pseudonymous subject ID to the selected videos",
    )
    action.add_argument(
        "--clear", action="store_true",
        help="Clear subject IDs for the selected videos",
    )
    parser.add_argument(
        "--source", choices=[source["name"] for source in TRAINING_DATASETS],
        help="Limit selection to one dataset source",
    )
    parser.add_argument(
        "--pattern",
        help="Glob matched against qualified ID, stem, or relative path",
    )
    parser.add_argument(
        "--video", action="append", default=[],
        help="Exact qualified video ID; repeat for multiple videos",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=["S01", "S02", "S03", "S04"],
        help="Ordered subject choices for --annotate (default: S01 S02 S03 S04)",
    )
    parser.add_argument(
        "--include-assigned", action="store_true",
        help="With --annotate, review already assigned videos too",
    )
    return parser


def require_selector(args):
    if not args.video and not args.source:
        raise ValueError(
            "Gunakan --video atau --source agar assignment tidak mengenai "
            "seluruh dataset secara tidak sengaja."
        )


def main():
    args = build_parser().parse_args()

    if args.init:
        payload = load_subject_metadata(
            SUBJECT_METADATA_PATH, allow_missing=True
        )
        discovered = discover_training_videos()
        payload = sync_subject_metadata(payload, discovered)
        payload["updated_at_unix"] = time.time()
        save_subject_metadata(payload)
        print("Manifest dibuat/diperbarui dari {} video raw.".format(
            len(discovered)
        ))
        print_summary(payload)
        return 0

    payload = load_subject_metadata()

    if args.summary:
        print_summary(payload)
        return 0

    if args.list or args.list_unassigned:
        selected = filtered_video_ids(
            payload,
            source=args.source,
            pattern=args.pattern,
            videos=args.video,
            unassigned_only=args.list_unassigned,
        )
        print_entries(payload, selected)
        return 0

    if args.annotate:
        subject_choices = [
            validate_subject_id(subject_id) for subject_id in args.subjects
        ]
        if len(subject_choices) != len(set(subject_choices)):
            raise ValueError("Daftar --subjects memiliki ID duplikat")
        if not 1 <= len(subject_choices) <= 9:
            raise ValueError("Mode interaktif mendukung 1 sampai 9 subject")
        selected = filtered_video_ids(
            payload,
            source=args.source,
            pattern=args.pattern,
            videos=args.video,
            unassigned_only=not args.include_assigned,
        )
        run_interactive_annotation(payload, selected, subject_choices)
        return 0

    if args.set or args.clear:
        require_selector(args)
        selected = filtered_video_ids(
            payload,
            source=args.source,
            pattern=args.pattern,
            videos=args.video,
        )
        if not selected:
            raise ValueError("Selector tidak menemukan video")
        subject_id = validate_subject_id(args.set) if args.set else None
        for video_id in selected:
            payload["videos"][video_id]["subject_id"] = subject_id
        payload["updated_at_unix"] = time.time()
        save_subject_metadata(payload)
        print("{} video diperbarui -> {}".format(
            len(selected), subject_id or "UNASSIGNED"
        ))
        print_entries(payload, selected[:20])
        if len(selected) > 20:
            print("... dan {} video lainnya".format(len(selected) - 20))
        return 0

    if args.validate:
        discovered = discover_training_videos()
        active_ids = sorted(
            video_id for video_id, entry in payload["videos"].items()
            if entry.get("present_in_raw", True)
        )
        missing_manifest = sorted(set(discovered) - set(active_ids))
        stale = sorted(set(active_ids) - set(discovered))
        errors = []
        if missing_manifest:
            errors.append(
                "{} raw video belum ada di manifest; jalankan --init".format(
                    len(missing_manifest)
                )
            )
        if stale:
            errors.append(
                "{} entry aktif tidak ditemukan di raw dataset; jalankan "
                "--init".format(len(stale))
            )
        try:
            resolve_subject_ids(active_ids, payload, require_complete=True)
        except ValueError as exc:
            errors.append(str(exc))
        print_summary(payload)
        if errors:
            for error in errors:
                print("[ERROR] " + error)
            return 1
        if len(subject_summary(payload)["subjects"]) < 3:
            print(
                "[WARNING] Metadata lengkap, tetapi kurang dari 3 subject. "
                "Train/validation/test subject-independent belum memungkinkan."
            )
        print("Subject metadata valid dan lengkap.")
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print("[ERROR] {}".format(exc))
        sys.exit(1)
