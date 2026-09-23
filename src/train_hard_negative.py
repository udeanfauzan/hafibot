"""Mine training-only ADL errors and fine-tune retained LOSO checkpoints.

This command never writes models/best_model.pt or models/loso/. Each fold uses
only its training subjects for mining, its validation subject for early stopping
and threshold calibration, and its untouched test subject for final reporting.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    AUGMENTATION_PROFILE,
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    LOGS_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    STRIDE,
    WEIGHT_DECAY,
)
from src.dataset import FallDetectionDataset, load_all_data  # noqa: E402
from src.event_calibration import (  # noqa: E402
    event_threshold_rows,
    select_safety_first_threshold,
)
from src.hard_negative_mining import (  # noqa: E402
    build_sample_weights,
    collect_windows_with_provenance,
    select_hard_negative_indices,
    validate_mining_subject_scope,
)
from src.model import (  # noqa: E402
    FallDetectionModel,
    checkpoint_feature_pipeline,
    checkpoint_input_features,
    checkpoint_rnn_type,
)
from src.temporal_labels import file_sha256  # noqa: E402
from src.train import evaluate, set_seed, train_one_epoch  # noqa: E402
from src.train_loso import (  # noqa: E402
    binary_metrics,
    collect_windows,
    predict_window_probabilities,
    video_max_probability,
)


def load_checkpoint(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def subject_indices(subjects, selected_subjects):
    selected = set(selected_subjects)
    return np.asarray([
        index for index, subject in enumerate(subjects) if subject in selected
    ], dtype=np.int64)


def checkpoint_subject_split(checkpoint, observed_subjects, expected_test_subject):
    split = checkpoint.get("data_split")
    if not split or split.get("strategy") != "subject":
        raise ValueError("Baseline checkpoint harus memiliki subject split")
    train_subjects = list(split.get("train_subject_ids", []))
    val_subjects = list(split.get("val_subject_ids", []))
    test_subjects = list(split.get("test_subject_ids", []))
    if test_subjects != [expected_test_subject]:
        raise ValueError(
            "Checkpoint {} bukan fold test {}".format(
                test_subjects, expected_test_subject
            )
        )
    groups = [set(train_subjects), set(val_subjects), set(test_subjects)]
    if any(not group for group in groups):
        raise ValueError("Checkpoint split memiliki group kosong")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Subject checkpoint muncul pada lebih dari satu split")
    if set.union(*groups) != set(observed_subjects):
        raise ValueError("Checkpoint split tidak mencakup semua subject dataset")
    return train_subjects, val_subjects, test_subjects


@torch.no_grad()
def predict_video_events(
    model, event_windows, event_labels, video_names, subjects, indices,
    mean, std, device, threshold,
):
    y_true, y_pred, details = [], [], []
    for index in indices:
        maximum, window_count = video_max_probability(
            model, event_windows[int(index)], mean, std, device
        )
        prediction = int(maximum >= threshold)
        label = int(event_labels[int(index)])
        y_true.append(label)
        y_pred.append(prediction)
        details.append({
            "video": video_names[int(index)],
            "subject_id": subjects[int(index)],
            "label": CLASS_NAMES[label],
            "prediction": CLASS_NAMES[prediction],
            "max_fall_probability": maximum,
            "total_windows": window_count,
        })
    return np.asarray(y_true), np.asarray(y_pred), details


def mining_manifest(
    selected_indices, probabilities, source_videos, source_windows,
    video_names, subjects,
):
    records = []
    for rank, sample_index in enumerate(selected_indices, 1):
        video_index = int(source_videos[sample_index])
        window_index = int(source_windows[sample_index])
        records.append({
            "rank": rank,
            "sample_index": int(sample_index),
            "video": video_names[video_index],
            "subject_id": subjects[video_index],
            "training_window_index": window_index,
            "approx_start_frame": window_index * STRIDE,
            "fall_probability": float(probabilities[sample_index]),
        })
    return records


def aggregate_metric_dicts(metric_dicts):
    """Aggregate fold metrics exactly by summing their confusion matrices."""
    matrix = np.sum(
        [np.asarray(metrics["confusion_matrix"], dtype=np.int64)
         for metrics in metric_dicts],
        axis=0,
    )
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = int(matrix.sum())
    precision = float(tp / float(tp + fp)) if tp + fp else 0.0
    recall = float(tp / float(tp + fn)) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "accuracy": float((tn + tp) / float(total)) if total else 0.0,
        "fall_precision": precision,
        "fall_recall": recall,
        "fall_f1": f1,
        "confusion_matrix": matrix.tolist(),
    }


def merge_hard_negative_results(result_paths, output_path):
    """Merge independently executed fold files after CUDA-safe isolated runs."""
    payloads = []
    for path in result_paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("mine_only"):
            raise ValueError("Mine-only result tidak dapat digabung sebagai evaluasi")
        if len(payload.get("folds", [])) != 1:
            raise ValueError("Setiap merge input harus berisi tepat satu fold")
        payloads.append(payload)
    test_subjects = [payload["folds"][0]["test_subject"] for payload in payloads]
    if len(test_subjects) != len(set(test_subjects)):
        raise ValueError("Test subject duplikat pada merge input")
    first = payloads[0]
    for payload in payloads[1:]:
        if payload["feature_pipeline"] != first["feature_pipeline"]:
            raise ValueError("Feature pipeline merge input berbeda")
        if payload["config"] != first["config"]:
            raise ValueError("Konfigurasi mining merge input berbeda")

    folds = sorted(
        [payload["folds"][0] for payload in payloads],
        key=lambda fold: fold["test_subject"],
    )
    baseline_window = aggregate_metric_dicts([
        fold["baseline"]["window_level"] for fold in folds
    ])
    candidate_window = aggregate_metric_dicts([
        fold["candidate"]["window_level"] for fold in folds
    ])
    baseline_event = aggregate_metric_dicts([
        fold["baseline"]["event_level"] for fold in folds
    ])
    candidate_event = aggregate_metric_dicts([
        fold["candidate"]["event_level"] for fold in folds
    ])
    inherited_true, inherited_pred = [], []
    for fold in folds:
        inherited_threshold = float(fold["baseline"]["threshold"])
        fold_true, fold_pred = [], []
        for video in fold["candidate"]["event_level"]["videos"]:
            label = CLASS_NAMES.index(video["label"])
            prediction = int(
                video["max_fall_probability"] >= inherited_threshold
            )
            fold_true.append(label)
            fold_pred.append(prediction)
        inherited_metrics = binary_metrics(fold_true, fold_pred)
        fold["candidate_at_inherited_baseline_threshold"] = {
            "threshold": inherited_threshold,
            "event_level": inherited_metrics,
        }
        inherited_true.extend(fold_true)
        inherited_pred.extend(fold_pred)
    candidate_inherited_event = binary_metrics(inherited_true, inherited_pred)
    baseline_fp = baseline_event["confusion_matrix"][0][1]
    candidate_fp = candidate_event["confusion_matrix"][0][1]
    promotion_passed = (
        candidate_event["fall_recall"] >= baseline_event["fall_recall"]
        and candidate_fp <= baseline_fp
        and candidate_event["fall_f1"] >= baseline_event["fall_f1"]
    )
    result = {
        "method": "training-only-hard-negative-mining",
        "merged_isolated_folds": True,
        "mine_only": False,
        "feature_pipeline": first["feature_pipeline"],
        "baseline_checkpoint_dir": first["baseline_checkpoint_dir"],
        "candidate_checkpoint_dir": first["candidate_checkpoint_dir"],
        "config": first["config"],
        "folds": folds,
        "aggregate_baseline_window_level": baseline_window,
        "aggregate_candidate_window_level": candidate_window,
        "aggregate_baseline_event_level": baseline_event,
        "aggregate_candidate_event_level": candidate_event,
        "aggregate_candidate_event_at_inherited_baseline_threshold": (
            candidate_inherited_event
        ),
        "promotion_rule": (
            "candidate Fall recall >= baseline, event false positives <= "
            "baseline, and candidate Fall F1 >= baseline"
        ),
        "promotion_passed": bool(promotion_passed),
        "active_deployment_modified": False,
        "source_result_files": [str(Path(path)) for path in result_paths],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Merged: {}".format(output_path))
    print(json.dumps({
        "baseline_event": baseline_event,
        "candidate_event": candidate_event,
        "promotion_passed": promotion_passed,
    }, indent=2))
    return result


def train_hard_negative(
    baseline_dir=MODEL_DIR / "loso",
    checkpoint_dir=MODEL_DIR / "loso_hard_negative",
    output_path=LOGS_DIR / "loso_hard_negative_results.json",
    selected_test_subjects=None,
    min_probability=0.50,
    max_per_video=12,
    min_gap_windows=3,
    weight_multiplier=4.0,
    learning_rate=1e-4,
    max_epochs=20,
    patience=7,
    mine_only=False,
):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    baseline_dir = Path(baseline_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_path = Path(output_path)
    checkpoint_paths = sorted(baseline_dir.glob("test_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            "Tidak ada baseline LOSO checkpoint di {}".format(baseline_dir)
        )
    available_subjects = [path.stem.replace("test_", "", 1) for path in checkpoint_paths]
    test_subjects = (
        list(selected_test_subjects)
        if selected_test_subjects else available_subjects
    )
    unknown = set(test_subjects) - set(available_subjects)
    if unknown:
        raise ValueError("Baseline fold tidak ditemukan: {}".format(sorted(unknown)))

    first_checkpoint = load_checkpoint(
        baseline_dir / "test_{}.pt".format(test_subjects[0]), device
    )
    feature_pipeline = checkpoint_feature_pipeline(first_checkpoint)
    input_features = checkpoint_input_features(first_checkpoint)
    rnn_type = checkpoint_rnn_type(first_checkpoint)
    data = load_all_data(
        return_subject_ids=True,
        require_subject_metadata=True,
        feature_pipeline=feature_pipeline,
    )
    video_windows, video_labels, event_labels, video_names, subjects = data
    event_data = load_all_data(
        window_stride=1,
        include_ambiguous=True,
        return_subject_ids=True,
        require_subject_metadata=True,
        feature_pipeline=feature_pipeline,
    )
    event_windows, _, event_labels_stride1, event_names, event_subjects = event_data
    if video_names != event_names or subjects != event_subjects:
        raise ValueError("Urutan training/event video tidak cocok")
    if event_labels != event_labels_stride1:
        raise ValueError("Event labels berubah antara stride")
    observed_subjects = sorted(set(subjects))

    print("Device: {}".format(device))
    print("Baseline: {}".format(baseline_dir))
    print("Pipeline: {} ({} features)".format(
        feature_pipeline, input_features
    ))
    print("Recurrent architecture: {}".format(rnn_type))
    print(
        "Mining: p>={:.2f}, cap={}/video, gap={} windows, weight={:.1f}x"
        .format(
            min_probability, max_per_video, min_gap_windows,
            weight_multiplier,
        )
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    aggregate_baseline_window_true, aggregate_baseline_window_pred = [], []
    aggregate_candidate_window_true, aggregate_candidate_window_pred = [], []
    aggregate_baseline_event_true, aggregate_baseline_event_pred = [], []
    aggregate_candidate_event_true, aggregate_candidate_event_pred = [], []
    started = time.time()

    for fold_number, test_subject in enumerate(test_subjects, 1):
        baseline_path = baseline_dir / "test_{}.pt".format(test_subject)
        checkpoint = load_checkpoint(baseline_path, device)
        if checkpoint_feature_pipeline(checkpoint) != feature_pipeline:
            raise ValueError("Semua baseline fold harus memakai pipeline sama")
        if checkpoint_input_features(checkpoint) != input_features:
            raise ValueError("Semua baseline fold harus memiliki input sama")
        if checkpoint_rnn_type(checkpoint) != rnn_type:
            raise ValueError("Semua baseline fold harus memiliki RNN sama")
        train_subjects, val_subjects, _ = checkpoint_subject_split(
            checkpoint, observed_subjects, test_subject
        )
        train_idx = subject_indices(subjects, train_subjects)
        val_idx = subject_indices(subjects, val_subjects)
        test_idx = subject_indices(subjects, [test_subject])
        if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx):
            raise AssertionError("Video index leakage pada fold")

        print("\n{}".format("=" * 72))
        print("HNM fold {}/{} | test={} | val={} | train={}".format(
            fold_number, len(test_subjects), test_subject,
            val_subjects, train_subjects,
        ))
        print("{}".format("=" * 72))
        set_seed(RANDOM_SEED + 100 + fold_number)

        (
            x_train, y_train, train_event_labels,
            source_videos, source_windows,
        ) = collect_windows_with_provenance(
            video_windows, video_labels, event_labels, train_idx
        )
        x_val, y_val = collect_windows(video_windows, video_labels, val_idx)
        x_test, y_test = collect_windows(video_windows, video_labels, test_idx)
        mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["scaler_std"], dtype=np.float32)
        if mean.shape != (input_features,) or std.shape != (input_features,):
            raise ValueError("Baseline scaler tidak cocok dengan model")
        x_train_normalized = (x_train - mean) / std
        x_test_normalized = (x_test - mean) / std

        baseline_model = FallDetectionModel(
            input_size=input_features, rnn_type=rnn_type
        ).to(device)
        baseline_model.load_state_dict(checkpoint["model_state_dict"])
        baseline_model.eval()
        train_probabilities = predict_window_probabilities(
            baseline_model, x_train_normalized, device
        )
        selected = select_hard_negative_indices(
            train_probabilities,
            y_train,
            train_event_labels,
            source_videos,
            source_windows,
            min_probability=min_probability,
            max_per_video=max_per_video,
            min_gap_windows=min_gap_windows,
        )
        mined_subjects = validate_mining_subject_scope(
            selected, source_videos, subjects, train_subjects
        )
        eligible_count = int(np.sum(
            (y_train == 0)
            & (train_event_labels == 0)
            & (train_probabilities >= min_probability)
        ))
        records = mining_manifest(
            selected, train_probabilities, source_videos, source_windows,
            video_names, subjects,
        )
        mining = {
            "min_probability": float(min_probability),
            "eligible_before_deduplication": eligible_count,
            "selected_windows": int(len(selected)),
            "selected_videos": int(len({
                int(source_videos[index]) for index in selected
            })),
            "selected_subjects": sorted(mined_subjects),
            "max_per_video": int(max_per_video),
            "min_gap_windows": int(min_gap_windows),
            "weight_multiplier": float(weight_multiplier),
            "windows": records,
        }
        print(
            "Mined {} windows from {} ADL videos ({} eligible before NMS)"
            .format(
                mining["selected_windows"], mining["selected_videos"],
                eligible_count,
            )
        )
        if records:
            print("Top hard negatives:")
            for record in records[:5]:
                print("  {:.1%}  {}  window={}".format(
                    record["fall_probability"], record["video"],
                    record["training_window_index"],
                ))

        fold_result = {
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "validation_subjects": val_subjects,
            "baseline_checkpoint": str(baseline_path),
            "mining": mining,
        }
        if mine_only:
            fold_results.append(fold_result)
            del baseline_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if not len(selected):
            raise ValueError(
                "Tidak ada hard negative untuk fold {}. Turunkan "
                "--min-probability secara terkontrol.".format(test_subject)
            )

        baseline_threshold = float(checkpoint["calibrated_threshold"])
        baseline_window_prob = predict_window_probabilities(
            baseline_model, x_test_normalized, device
        )
        baseline_window_pred = (
            baseline_window_prob >= baseline_threshold
        ).astype(np.int64)
        baseline_window_metrics = binary_metrics(y_test, baseline_window_pred)
        baseline_event_true, baseline_event_pred, baseline_event_details = (
            predict_video_events(
                baseline_model, event_windows, event_labels, video_names,
                subjects, test_idx, mean, std, device, baseline_threshold,
            )
        )
        baseline_event_metrics = binary_metrics(
            baseline_event_true, baseline_event_pred
        )
        # The baseline is no longer needed. Release it before creating the
        # fine-tuning model so small Windows GPUs start each stage cleanly.
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        augmentation_profile = checkpoint.get("training_config", {}).get(
            "augmentation_profile", AUGMENTATION_PROFILE
        )
        sampler_weights = build_sample_weights(
            len(y_train), selected, weight_multiplier
        )
        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED + 1000 + fold_number)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=len(y_train),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            FallDetectionDataset(
                x_train, y_train, augment=True,
                normalization=(mean, std),
                feature_pipeline=feature_pipeline,
                augmentation_profile=augmentation_profile,
            ),
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            FallDetectionDataset(
                x_val, y_val, augment=False,
                normalization=(mean, std),
                feature_pipeline=feature_pipeline,
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        counts = np.bincount(y_train, minlength=2)
        class_weights = torch.FloatTensor(
            len(y_train) / (2.0 * counts.astype(np.float64))
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model = FallDetectionModel(
            input_size=input_features, rnn_type=rnn_type
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        baseline_val_loss, baseline_val_acc = evaluate(
            model, val_loader, criterion, device
        )
        # Select among actual hard-negative fine-tuning epochs. The untouched
        # baseline is evaluated separately and remains available for rejection.
        best_state = None
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [],
        }
        print("Baseline validation: loss={:.4f}, accuracy={:.1%}".format(
            baseline_val_loss, baseline_val_acc
        ))

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            improved = val_loss < best_val_loss
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            print(
                "Epoch {:>2}: train={:.4f}/{:.1%} val={:.4f}/{:.1%}{}"
                .format(
                    epoch, train_loss, train_acc, val_loss, val_acc,
                    " *" if improved else "",
                )
            )
            if patience_counter >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break

        if best_state is None:
            raise RuntimeError("Fine-tuning tidak menghasilkan candidate")
        model.load_state_dict(best_state)
        model.eval()
        val_event_true = np.asarray(
            [event_labels[int(index)] for index in val_idx], dtype=np.int64
        )
        val_event_max = np.asarray([
            video_max_probability(
                model, event_windows[int(index)], mean, std, device
            )[0]
            for index in val_idx
        ])
        threshold_rows = event_threshold_rows(val_event_true, val_event_max)
        threshold_selection = select_safety_first_threshold(threshold_rows)
        candidate_threshold = float(threshold_selection["threshold"])

        candidate_window_prob = predict_window_probabilities(
            model, x_test_normalized, device
        )
        candidate_window_pred = (
            candidate_window_prob >= candidate_threshold
        ).astype(np.int64)
        candidate_window_metrics = binary_metrics(y_test, candidate_window_pred)
        candidate_event_true, candidate_event_pred, candidate_event_details = (
            predict_video_events(
                model, event_windows, event_labels, video_names, subjects,
                test_idx, mean, std, device, candidate_threshold,
            )
        )
        candidate_event_metrics = binary_metrics(
            candidate_event_true, candidate_event_pred
        )

        candidate_path = checkpoint_dir / "test_{}.pt".format(test_subject)
        candidate_checkpoint = {
            "checkpoint_version": 3,
            "model_state_dict": best_state,
            "fine_tune_epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "scaler_mean": mean,
            "scaler_std": std,
            "calibrated_threshold": candidate_threshold,
            "history": history,
            "training_config": {
                "input_features": input_features,
                "feature_pipeline": feature_pipeline,
                "augmentation_profile": augmentation_profile,
                "rnn_type": rnn_type,
                "hard_negative_mining": True,
            },
            "data_split": checkpoint["data_split"],
            "parent_checkpoint": {
                "path": str(baseline_path),
                "sha256": file_sha256(baseline_path),
            },
            "hard_negative_mining": mining,
        }
        torch.save(candidate_checkpoint, str(candidate_path))

        fold_result.update({
            "candidate_checkpoint": str(candidate_path),
            "fine_tune_best_epoch": best_epoch,
            "fine_tune_best_val_loss": best_val_loss,
            "threshold_calibration": threshold_selection,
            "baseline": {
                "threshold": baseline_threshold,
                "window_level": baseline_window_metrics,
                "event_level": dict(
                    baseline_event_metrics, videos=baseline_event_details
                ),
            },
            "candidate": {
                "threshold": candidate_threshold,
                "window_level": candidate_window_metrics,
                "event_level": dict(
                    candidate_event_metrics, videos=candidate_event_details
                ),
            },
        })
        fold_results.append(fold_result)
        aggregate_baseline_window_true.extend(y_test.tolist())
        aggregate_baseline_window_pred.extend(baseline_window_pred.tolist())
        aggregate_candidate_window_true.extend(y_test.tolist())
        aggregate_candidate_window_pred.extend(candidate_window_pred.tolist())
        aggregate_baseline_event_true.extend(baseline_event_true.tolist())
        aggregate_baseline_event_pred.extend(baseline_event_pred.tolist())
        aggregate_candidate_event_true.extend(candidate_event_true.tolist())
        aggregate_candidate_event_pred.extend(candidate_event_pred.tolist())
        print(
            "Result: baseline event P/R/F1={:.1%}/{:.1%}/{:.1%}; "
            "candidate={:.1%}/{:.1%}/{:.1%}"
            .format(
                baseline_event_metrics["fall_precision"],
                baseline_event_metrics["fall_recall"],
                baseline_event_metrics["fall_f1"],
                candidate_event_metrics["fall_precision"],
                candidate_event_metrics["fall_recall"],
                candidate_event_metrics["fall_f1"],
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "method": "training-only-hard-negative-mining",
        "mine_only": bool(mine_only),
        "feature_pipeline": feature_pipeline,
        "rnn_type": rnn_type,
        "baseline_checkpoint_dir": str(baseline_dir),
        "candidate_checkpoint_dir": str(checkpoint_dir),
        "elapsed_sec": time.time() - started,
        "config": {
            "min_probability": float(min_probability),
            "max_per_video": int(max_per_video),
            "min_gap_windows": int(min_gap_windows),
            "weight_multiplier": float(weight_multiplier),
            "learning_rate": float(learning_rate),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
        },
        "folds": fold_results,
    }
    if not mine_only:
        baseline_window = binary_metrics(
            aggregate_baseline_window_true, aggregate_baseline_window_pred
        )
        candidate_window = binary_metrics(
            aggregate_candidate_window_true, aggregate_candidate_window_pred
        )
        baseline_event = binary_metrics(
            aggregate_baseline_event_true, aggregate_baseline_event_pred
        )
        candidate_event = binary_metrics(
            aggregate_candidate_event_true, aggregate_candidate_event_pred
        )
        baseline_fp = baseline_event["confusion_matrix"][0][1]
        candidate_fp = candidate_event["confusion_matrix"][0][1]
        promotion_passed = (
            candidate_event["fall_recall"] >= baseline_event["fall_recall"]
            and candidate_fp <= baseline_fp
            and candidate_event["fall_f1"] >= baseline_event["fall_f1"]
        )
        result.update({
            "aggregate_baseline_window_level": baseline_window,
            "aggregate_candidate_window_level": candidate_window,
            "aggregate_baseline_event_level": baseline_event,
            "aggregate_candidate_event_level": candidate_event,
            "promotion_rule": (
                "candidate Fall recall >= baseline, event false positives <= "
                "baseline, and candidate Fall F1 >= baseline"
            ),
            "promotion_passed": bool(promotion_passed),
            "active_deployment_modified": False,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("\nSaved: {}".format(output_path))
    if not mine_only:
        print(json.dumps({
            "baseline_event": result["aggregate_baseline_event_level"],
            "candidate_event": result["aggregate_candidate_event_level"],
            "promotion_passed": result["promotion_passed"],
        }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe LOSO hard-negative mining experiment"
    )
    parser.add_argument("--baseline-dir", type=Path, default=MODEL_DIR / "loso")
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=MODEL_DIR / "loso_hard_negative",
    )
    parser.add_argument(
        "--output", type=Path,
        default=LOGS_DIR / "loso_hard_negative_results.json",
    )
    parser.add_argument("--test-subject", action="append")
    parser.add_argument("--min-probability", type=float, default=0.50)
    parser.add_argument("--max-per-video", type=int, default=12)
    parser.add_argument("--min-gap-windows", type=int, default=3)
    parser.add_argument("--weight-multiplier", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument(
        "--mine-only", action="store_true",
        help="Inspect and save selected training hard negatives without training",
    )
    parser.add_argument(
        "--merge-result", action="append", type=Path,
        help="Merge one-fold result files; repeat once per subject",
    )
    args = parser.parse_args()
    if args.merge_result:
        merge_hard_negative_results(args.merge_result, args.output)
        return
    train_hard_negative(
        baseline_dir=args.baseline_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
        selected_test_subjects=args.test_subject,
        min_probability=args.min_probability,
        max_per_video=args.max_per_video,
        min_gap_windows=args.min_gap_windows,
        weight_multiplier=args.weight_multiplier,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        mine_only=args.mine_only,
    )


if __name__ == "__main__":
    main()
