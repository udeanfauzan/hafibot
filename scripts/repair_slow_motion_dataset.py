#!/usr/bin/env python3
"""Repair videos whose real-time frames were written at an underspecified FPS.

For the July dataset, approximately 30 real frames/second were written into a
15 FPS file. Repairing with factor 2 retains source frames 0, 2, 4, ... and
writes those frames at 15 FPS. The result is real-time 15 FPS data rather than
slow-motion 15 FPS data.

The operation is non-destructive: source videos and annotations are never
modified. Without --execute, the command only prints and validates its plan.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def portable_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_videos(root):
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def retained_frame_count(source_frame_count, speed_factor):
    """Count indices 0, factor, 2*factor, ... below source_frame_count."""
    if source_frame_count <= 0:
        raise ValueError("source_frame_count harus lebih besar dari nol")
    if speed_factor < 2:
        raise ValueError("speed_factor integer minimal 2")
    return (int(source_frame_count) + int(speed_factor) - 1) // int(speed_factor)


def remap_inclusive_interval(start_frame, end_frame, speed_factor):
    """Map an inclusive interval onto retained source frames."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Interval frame tidak valid")
    start = int(math.ceil(start_frame / float(speed_factor)))
    end = int(math.floor(end_frame / float(speed_factor)))
    if end < start:
        # A sub-factor annotation has no retained source frame. Preserve one
        # nearest output frame rather than silently deleting the Fall interval.
        end = start
    return start, end


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Tidak dapat membuka video: {}".format(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("Metadata video tidak valid: {}".format(path))
    return fps, frame_count, width, height


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_video(source, destination, speed_factor, output_fps):
    source_fps, source_count, width, height = video_info(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps), (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Tidak dapat membuat video: {}".format(destination))

    read_count = 0
    written_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_count % speed_factor == 0:
            writer.write(frame)
            written_count += 1
        read_count += 1
    capture.release()
    writer.release()

    expected = retained_frame_count(source_count, speed_factor)
    repaired_fps, repaired_count, repaired_width, repaired_height = video_info(
        destination
    )
    if read_count != source_count:
        raise ValueError(
            "Decode count {} berbeda dari metadata {}: {}".format(
                read_count, source_count, source
            )
        )
    if written_count != expected or repaired_count != expected:
        raise ValueError(
            "Output frame count salah untuk {}: write={}, metadata={}, expected={}"
            .format(destination, written_count, repaired_count, expected)
        )
    if abs(repaired_fps - output_fps) > 0.01:
        raise ValueError("Output FPS tidak sesuai: {}".format(destination))
    if (repaired_width, repaired_height) != (width, height):
        raise ValueError("Resolusi output berubah: {}".format(destination))
    return {
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "source_duration_sec": source_count / source_fps,
        "output_fps": repaired_fps,
        "output_frame_count": repaired_count,
        "output_duration_sec": repaired_count / repaired_fps,
        "width": width,
        "height": height,
        "source_sha256": sha256(source),
        "output_sha256": sha256(destination),
    }


def repair_annotations(
    source_path, output_path, source_root, output_root, speed_factor, output_fps,
    repaired_by_stem,
):
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = dict(payload)
    result["source_root"] = portable_path(output_root)
    result["timing_repair"] = {
        "method": "integer_frame_decimation",
        "speed_factor": speed_factor,
        "retained_source_frame_rule": "source_frame_index % speed_factor == 0",
        "output_fps": output_fps,
        "original_annotation_file": source_path.name,
        "created_at_unix": time.time(),
    }
    result["videos"] = {}
    for stem, annotation in payload.get("videos", {}).items():
        if stem not in repaired_by_stem:
            raise ValueError("Annotation tanpa repaired video: {}".format(stem))
        info = repaired_by_stem[stem]
        if int(annotation["source_frame_count"]) != info["source_frame_count"]:
            raise ValueError("Annotation source frame count berubah: {}".format(stem))
        start, end = remap_inclusive_interval(
            int(annotation["fall_start_frame"]),
            int(annotation["fall_end_frame"]),
            speed_factor,
        )
        if end >= info["output_frame_count"]:
            raise ValueError("Remapped annotation melewati output: {}".format(stem))
        updated = dict(annotation)
        updated.update({
            "fall_start_frame": start,
            "fall_end_frame": end,
            "fall_start_sec": start / float(output_fps),
            "fall_end_sec": end / float(output_fps),
            "source_fps": float(output_fps),
            "source_frame_count": info["output_frame_count"],
            "original_fall_start_frame": int(annotation["fall_start_frame"]),
            "original_fall_end_frame": int(annotation["fall_end_frame"]),
            "timing_speed_factor": speed_factor,
        })
        result["videos"][stem] = updated
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/dataset new july")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/dataset new july realtime"),
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("data/fall_annotations_dataset_new_july.json"),
    )
    parser.add_argument(
        "--output-annotations", type=Path,
        default=Path("data/fall_annotations_dataset_new_july_realtime.json"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/dataset_new_july_timing_repair.json"),
    )
    parser.add_argument("--speed-factor", type=int, default=2)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument(
        "--execute", action="store_true",
        help="Create repaired videos; default is a read-only dry run",
    )
    args = parser.parse_args()
    source_root = args.source_dir.resolve()
    output_root = args.output_dir.resolve()
    videos = discover_videos(source_root)
    if not videos:
        raise ValueError("Tidak ada video di {}".format(source_root))
    if args.speed_factor < 2:
        raise ValueError("--speed-factor harus integer minimal 2")
    if args.output_fps <= 0:
        raise ValueError("--output-fps harus lebih besar dari nol")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output tidak boleh sama/berada di dalam source dataset")
    existing_outputs = discover_videos(output_root) if output_root.exists() else []
    if args.execute and existing_outputs:
        raise ValueError(
            "Output sudah berisi video. Gunakan folder kosong agar repair atomik."
        )

    plan = []
    source_stems = set()
    for source in videos:
        relative = source.relative_to(source_root)
        fps, frame_count, width, height = video_info(source)
        if source.stem in source_stems:
            raise ValueError("Video stem duplikat: {}".format(source.stem))
        source_stems.add(source.stem)
        plan.append({
            "source": str(source),
            "relative_path": str(relative).replace("\\", "/"),
            "destination": str(output_root / relative),
            "source_fps": fps,
            "source_frame_count": frame_count,
            "source_duration_sec": frame_count / fps,
            "expected_output_frame_count": retained_frame_count(
                frame_count, args.speed_factor
            ),
            "expected_output_duration_sec": (
                retained_frame_count(frame_count, args.speed_factor)
                / args.output_fps
            ),
            "width": width,
            "height": height,
        })
    print("Videos: {}".format(len(plan)))
    print("Source duration: {:.1f}s -> expected output: {:.1f}s".format(
        sum(item["source_duration_sec"] for item in plan),
        sum(item["expected_output_duration_sec"] for item in plan),
    ))
    print("Speed factor: {}x; output FPS: {}".format(
        args.speed_factor, args.output_fps
    ))
    if not args.execute:
        print("DRY RUN only. Add --execute to create the repaired dataset.")
        return 0

    repaired = []
    repaired_by_stem = {}
    started = time.time()
    for index, item in enumerate(plan, 1):
        source = Path(item["source"])
        destination = Path(item["destination"])
        info = repair_video(
            source, destination, args.speed_factor, args.output_fps
        )
        record = dict(item)
        record.update(info)
        repaired.append(record)
        repaired_by_stem[source.stem] = info
        print("[{}/{}] {}: {} -> {} frames".format(
            index, len(plan), item["relative_path"],
            info["source_frame_count"], info["output_frame_count"],
        ))

    annotations = repair_annotations(
        args.annotations.resolve(), args.output_annotations.resolve(),
        source_root, output_root, args.speed_factor, args.output_fps,
        repaired_by_stem,
    )
    manifest = {
        "schema_version": 1,
        "method": "integer_frame_decimation",
        "speed_factor": args.speed_factor,
        "output_fps": args.output_fps,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_annotations": str(args.annotations.resolve()),
        "output_annotations": str(args.output_annotations.resolve()),
        "video_count": len(repaired),
        "annotation_count": len(annotations["videos"]),
        "elapsed_sec": time.time() - started,
        "videos": repaired,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("Repaired dataset: {}".format(output_root))
    print("Annotations: {}".format(args.output_annotations))
    print("Manifest: {}".format(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
