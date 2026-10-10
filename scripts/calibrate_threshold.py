#!/usr/bin/env python3
"""Calibrate a checkpoint threshold on its validation video events only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, LOGS_DIR, MODEL_DIR  # noqa: E402
from src.dataset import load_all_data, split_video_indices  # noqa: E402
from src.evaluate import checkpoint_split_config  # noqa: E402
from src.event_calibration import (  # noqa: E402
    event_threshold_rows,
    select_safety_first_threshold,
)
from src.model import (  # noqa: E402
    FallDetectionModel,
    checkpoint_feature_pipeline,
    checkpoint_input_features,
    checkpoint_rnn_type,
)


@torch.no_grad()
def video_max_probability(model, windows, mean, std, device):
    if not windows:
        return 0.0
    values = (np.asarray(windows, dtype=np.float32) - mean) / std
    probabilities = []
    for start in range(0, len(values), 256):
        tensor = torch.from_numpy(values[start:start + 256]).to(device)
        logits, _ = model(tensor)
        probabilities.append(
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        )
    return float(np.concatenate(probabilities).max())


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Fall threshold using validation videos only"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt"
    )
    parser.add_argument(
        "--feature-root", type=Path,
        help="Override the checkpoint-bound cache for an explicit comparison",
    )
    parser.add_argument(
        "--output", type=Path,
        default=LOGS_DIR / "threshold_calibration.json",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(
            str(args.checkpoint), map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(args.checkpoint), map_location=device)
    feature_pipeline = checkpoint_feature_pipeline(checkpoint)
    model = FallDetectionModel(
        input_size=checkpoint_input_features(checkpoint),
        rnn_type=checkpoint_rnn_type(checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler_std"], dtype=np.float32)

    split = checkpoint_split_config(checkpoint)
    feature_root = args.feature_root or checkpoint.get("training_config", {}).get("feature_root")
    data = load_all_data(
        window_stride=1,
        include_ambiguous=True,
        return_subject_ids=(split["strategy"] == "subject"),
        require_subject_metadata=(split["strategy"] == "subject"),
        feature_pipeline=feature_pipeline,
        feature_root=feature_root,
    )
    windows, _, labels, names = data[:4]
    subjects = data[4] if len(data) == 5 else None
    _, val_idx, _ = split_video_indices(
        labels,
        video_subject_ids=subjects,
        split_strategy=split["strategy"],
        train_subject_ids=split["train_subject_ids"],
        val_subject_ids=split["val_subject_ids"],
        test_subject_ids=split["test_subject_ids"],
    )

    y_true = np.asarray([labels[index] for index in val_idx], dtype=np.int64)
    max_probabilities = np.asarray([
        video_max_probability(model, windows[index], mean, std, device)
        for index in val_idx
    ])
    rows = event_threshold_rows(y_true, max_probabilities)
    selected = select_safety_first_threshold(rows)
    details = [
        {
            "video": names[index],
            "subject_id": subjects[index] if subjects is not None else None,
            "label": int(labels[index]),
            "max_fall_probability": float(probability),
        }
        for index, probability in zip(val_idx, max_probabilities)
    ]
    result = {
        "checkpoint": str(args.checkpoint),
        "feature_root": str(feature_root) if feature_root else None,
        "evaluation_scope": "validation classifier video maxima; no runtime posture gates",
        "checkpoint_epoch": checkpoint.get("epoch", checkpoint.get("fold")),
        "data_split": checkpoint.get("data_split", {
            "strategy": "video_stratified"
        }),
        "selection_rule": (
            "maximize Fall recall, then minimize false ADL events, then use "
            "the highest threshold"
        ),
        "selected": selected,
        "candidates": rows,
        "validation_videos": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Selected threshold: {:.2f}".format(selected["threshold"]))
    print(json.dumps(selected, indent=2))
    print("Saved: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
