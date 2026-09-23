"""Subject metadata helpers for subject-independent model evaluation.

Subject IDs are stored separately from Fall interval annotations because every
ADL and Fall source video needs a subject assignment. Qualified video IDs use
``<dataset-source>:<video-stem>`` so overlapping stems remain unambiguous.
"""

import json
import re
from collections import Counter
from pathlib import Path

from src.config import SUBJECT_METADATA_PATH, TRAINING_DATASETS


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SUBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_subject_id(subject_id):
    """Return a normalized pseudonymous subject ID or raise ValueError."""
    value = str(subject_id).strip()
    if not SUBJECT_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Subject ID harus diawali huruf/angka dan hanya boleh berisi "
            "huruf, angka, underscore, atau hyphen: {!r}".format(subject_id)
        )
    return value


def classify_video(video_path, raw_root):
    """Classify a raw video as ADL/Fall from its stem or parent folders."""
    relative = video_path.relative_to(raw_root)
    stem = video_path.stem.lower()
    parent_parts = [part.lower() for part in relative.parts[:-1]]
    is_adl = stem.startswith("adl") or any(
        part.startswith("adl") for part in parent_parts
    )
    is_fall = stem.startswith("fall") or any(
        part.startswith("fall") for part in parent_parts
    )
    if is_adl == is_fall:
        raise ValueError(
            "Video harus dapat diklasifikasikan tepat sebagai ADL atau Fall: "
            + str(video_path)
        )
    return "ADL" if is_adl else "Fall"


def discover_training_videos(training_datasets=TRAINING_DATASETS):
    """Discover all raw videos and return manifest entries by qualified ID."""
    discovered = {}
    for source in training_datasets:
        source_name = source["name"]
        raw_root = Path(source["raw_root"])
        if not raw_root.is_dir():
            raise FileNotFoundError(
                "Raw root dataset '{}' tidak ditemukan: {}".format(
                    source_name, raw_root
                )
            )
        paths = sorted(
            path for path in raw_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not paths:
            raise ValueError(
                "Dataset '{}' tidak memiliki video di {}".format(
                    source_name, raw_root
                )
            )
        stems = set()
        for path in paths:
            if path.stem in stems:
                raise ValueError(
                    "Video stem harus unik di source '{}': {}".format(
                        source_name, path.stem
                    )
                )
            stems.add(path.stem)
            video_id = "{}:{}".format(source_name, path.stem)
            discovered[video_id] = {
                "source": source_name,
                "video_stem": path.stem,
                "class_label": classify_video(path, raw_root),
                "source_relative_path": path.relative_to(raw_root).as_posix(),
                "subject_id": None,
                "present_in_raw": True,
            }
    return discovered


def new_subject_payload():
    return {
        "schema_version": 1,
        "description": (
            "Pseudonymous subject ID for every raw ADL and Fall video"
        ),
        "videos": {},
    }


def load_subject_metadata(path=SUBJECT_METADATA_PATH, allow_missing=False):
    """Load and structurally validate a subject manifest."""
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return new_subject_payload()
        raise FileNotFoundError(
            "Subject metadata tidak ditemukan: {}. Jalankan "
            "python scripts/manage_subject_metadata.py --init".format(path)
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version subject metadata harus 1")
    videos = payload.get("videos")
    if not isinstance(videos, dict):
        raise ValueError("Field 'videos' subject metadata harus object/dict")
    for video_id, entry in videos.items():
        if not isinstance(entry, dict):
            raise ValueError("Entry subject tidak valid: " + video_id)
        expected_id = "{}:{}".format(
            entry.get("source"), entry.get("video_stem")
        )
        if video_id != expected_id:
            raise ValueError(
                "Qualified video ID tidak cocok: {} != {}".format(
                    video_id, expected_id
                )
            )
        if entry.get("class_label") not in ("ADL", "Fall"):
            raise ValueError("class_label tidak valid: " + video_id)
        subject_id = entry.get("subject_id")
        if subject_id not in (None, ""):
            entry["subject_id"] = validate_subject_id(subject_id)
        else:
            entry["subject_id"] = None
    return payload


def save_subject_metadata(payload, path=SUBJECT_METADATA_PATH):
    """Atomically write a normalized subject manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def sync_subject_metadata(payload, discovered):
    """Merge discovery into a manifest while preserving manual assignments."""
    previous = payload.get("videos", {})
    merged = {}
    for video_id, discovered_entry in discovered.items():
        entry = dict(discovered_entry)
        old_entry = previous.get(video_id, {})
        old_subject = old_entry.get("subject_id")
        entry["subject_id"] = (
            validate_subject_id(old_subject)
            if old_subject not in (None, "")
            else None
        )
        for optional_field in ("session_id", "environment_id", "notes"):
            if optional_field in old_entry:
                entry[optional_field] = old_entry[optional_field]
        merged[video_id] = entry

    # Keep assignments for temporarily missing files, but mark them inactive so
    # an interrupted/mis-mounted dataset cannot silently erase manual work.
    for video_id, old_entry in previous.items():
        if video_id not in merged:
            entry = dict(old_entry)
            entry["present_in_raw"] = False
            merged[video_id] = entry

    output = dict(payload)
    output["schema_version"] = 1
    output["description"] = new_subject_payload()["description"]
    output["videos"] = merged
    return output


def resolve_subject_ids(video_names, payload, require_complete=True):
    """Resolve loader video names to subject IDs in identical order."""
    videos = payload.get("videos", {})
    subject_ids = []
    missing = []
    for video_name in video_names:
        entry = videos.get(video_name)
        subject_id = entry.get("subject_id") if entry else None
        if not subject_id:
            missing.append(video_name)
            subject_ids.append(None)
        else:
            subject_ids.append(validate_subject_id(subject_id))
    if require_complete and missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            "{} video belum memiliki subject_id: {}{}. Jalankan "
            "manage_subject_metadata.py --list-unassigned.".format(
                len(missing), preview, suffix
            )
        )
    return subject_ids


def subject_summary(payload):
    """Return assignment counts for display and validation."""
    active = {
        key: value for key, value in payload.get("videos", {}).items()
        if value.get("present_in_raw", True)
    }
    assigned = {
        key: value for key, value in active.items()
        if value.get("subject_id")
    }
    return {
        "total": len(active),
        "assigned": len(assigned),
        "unassigned": len(active) - len(assigned),
        "stale": sum(
            not value.get("present_in_raw", True)
            for value in payload.get("videos", {}).values()
        ),
        "subjects": dict(sorted(Counter(
            value["subject_id"] for value in assigned.values()
        ).items())),
        "sources": dict(sorted(Counter(
            value["source"] for value in active.values()
        ).items())),
    }
