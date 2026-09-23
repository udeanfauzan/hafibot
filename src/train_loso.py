"""Leave-one-subject-out training and evaluation.

Each participant is used as the untouched test subject exactly once. One of
the remaining subjects is selected for validation; threshold calibration uses
only that validation subject. Fold checkpoints never replace best_model.pt.
"""

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LOGS_DIR,
    MODEL_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    SUBJECT_METADATA_PATH,
    WEIGHT_DECAY,
    FEATURE_PIPELINE,
    BASE_FEATURES,
    INPUT_FEATURES,
    LEGACY_FEATURE_PIPELINE,
    HYBRID_FEATURE_PIPELINE,
    AUGMENTATION_PROFILE,
    RNN_TYPE,
)
from src.dataset import FallDetectionDataset, feature_dimensions, load_all_data
from src.event_calibration import (
    event_threshold_rows,
    select_safety_first_threshold,
)
from src.model import FallDetectionModel
from src.temporal_labels import file_sha256
from src.train import evaluate, set_seed, train_one_epoch


def choose_validation_subject(test_subject, all_subjects, video_counts):
    """Keep the smallest remaining subject for validation to maximize train."""
    candidates = [
        subject for subject in all_subjects if subject != test_subject
    ]
    return min(candidates, key=lambda subject: (video_counts[subject], subject))


def collect_windows(video_windows, video_labels, indices):
    values, labels = [], []
    for index in indices:
        values.extend(video_windows[index])
        labels.extend(video_labels[index])
    if not values:
        raise ValueError("Split tidak memiliki window eligible")
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def normalize_splits(x_train, x_val, x_test):
    feature_count = x_train.shape[2]
    flat = x_train.reshape(-1, feature_count)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return (
        x_train,
        x_val,
        (x_test - mean) / std,
        mean,
        std,
    )


@torch.no_grad()
def predict_window_probabilities(model, values, device, batch_size=256):
    model.eval()
    probabilities = []
    for start in range(0, len(values), batch_size):
        tensor = torch.from_numpy(values[start:start + batch_size]).to(device)
        logits, _ = model(tensor)
        probabilities.append(
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        )
    return np.concatenate(probabilities)


@torch.no_grad()
def video_max_probability(model, windows, mean, std, device):
    if not windows:
        return 0.0, 0
    values = (np.asarray(windows, dtype=np.float32) - mean) / std
    probabilities = predict_window_probabilities(model, values, device)
    return float(probabilities.max()), int(len(probabilities))


def binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fall_precision": float(precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "fall_recall": float(recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "fall_f1": float(f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def train_loso(max_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
               selected_test_subjects=None,
               feature_pipeline=FEATURE_PIPELINE,
               augmentation_profile=AUGMENTATION_PROFILE,
               rnn_type=RNN_TYPE,
               checkpoint_dir=None,
               output_path=None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    base_features, _, input_features = feature_dimensions(feature_pipeline)
    print("Feature pipeline: {} ({} inputs)".format(
        feature_pipeline, input_features
    ))
    use_augmentation = augmentation_profile != "none"
    print("Augmentation profile: {}".format(augmentation_profile))
    rnn_type = str(rnn_type).upper()
    if rnn_type not in ("LSTM", "GRU"):
        raise ValueError("rnn_type harus LSTM atau GRU")
    print("Recurrent architecture: {}".format(rnn_type))

    default_data = load_all_data(
        return_subject_ids=True, require_subject_metadata=True,
        feature_pipeline=feature_pipeline,
    )
    video_windows, video_labels, event_labels, video_names, subjects = default_data
    event_data = load_all_data(
        window_stride=1,
        include_ambiguous=True,
        return_subject_ids=True,
        require_subject_metadata=True,
        feature_pipeline=feature_pipeline,
    )
    event_windows, _, event_labels_stride1, event_names, event_subjects = event_data
    if video_names != event_names or subjects != event_subjects:
        raise ValueError("Urutan video stride training dan event tidak cocok")
    if event_labels != event_labels_stride1:
        raise ValueError("Event labels berubah antara stride")

    all_subjects = sorted(set(subjects))
    if len(all_subjects) < 3:
        raise ValueError("LOSO memerlukan minimal tiga subject")
    if selected_test_subjects:
        unknown = set(selected_test_subjects) - set(all_subjects)
        if unknown:
            raise ValueError("Test subject tidak dikenal: {}".format(sorted(unknown)))
        test_subjects = list(selected_test_subjects)
    else:
        test_subjects = all_subjects
    video_counts = Counter(subjects)

    checkpoint_dir = Path(checkpoint_dir or (MODEL_DIR / "loso"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    aggregate_window_true, aggregate_window_pred = [], []
    aggregate_event_true, aggregate_event_pred = [], []
    start_time = time.time()

    for fold_index, test_subject in enumerate(test_subjects, 1):
        validation_subject = choose_validation_subject(
            test_subject, all_subjects, video_counts
        )
        train_subjects = [
            subject for subject in all_subjects
            if subject not in (test_subject, validation_subject)
        ]
        train_idx = np.asarray([
            index for index, subject in enumerate(subjects)
            if subject in train_subjects
        ])
        val_idx = np.asarray([
            index for index, subject in enumerate(subjects)
            if subject == validation_subject
        ])
        test_idx = np.asarray([
            index for index, subject in enumerate(subjects)
            if subject == test_subject
        ])

        print("\n{}".format("=" * 70))
        print("LOSO fold {}/{} | test={} | val={} | train={}".format(
            fold_index, len(test_subjects), test_subject,
            validation_subject, train_subjects,
        ))
        print("{}".format("=" * 70))
        set_seed(RANDOM_SEED + fold_index)

        x_train, y_train = collect_windows(video_windows, video_labels, train_idx)
        x_val, y_val = collect_windows(video_windows, video_labels, val_idx)
        x_test, y_test = collect_windows(video_windows, video_labels, test_idx)
        x_train, x_val, x_test, mean, std = normalize_splits(
            x_train, x_val, x_test
        )
        print("Windows: train={} val={} test={}".format(
            len(y_train), len(y_val), len(y_test)
        ))

        train_loader = DataLoader(
            FallDetectionDataset(
                x_train, y_train, augment=use_augmentation,
                normalization=(mean, std),
                feature_pipeline=feature_pipeline,
                augmentation_profile=(
                    AUGMENTATION_PROFILE
                    if augmentation_profile == "none" else augmentation_profile
                ),
            ),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            FallDetectionDataset(
                x_val, y_val, augment=False, normalization=(mean, std),
                feature_pipeline=feature_pipeline,
            ),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
        )

        label_counts = np.bincount(y_train, minlength=2)
        if np.any(label_counts == 0):
            raise ValueError("Training fold harus memiliki kedua class")
        class_weights = torch.FloatTensor(
            len(y_train) / (2.0 * label_counts.astype(np.float64))
        ).to(device)
        model = FallDetectionModel(
            input_size=input_features, rnn_type=rnn_type
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7
        )
        best_state = None
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [],
        }

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device
            )
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch <= 5 or epoch % 5 == 0 or improved:
                print(
                    "Epoch {:>3}: train={:.4f}/{:.1%} val={:.4f}/{:.1%}{}"
                    .format(
                        epoch, train_loss, train_acc, val_loss, val_acc,
                        " *" if improved else "",
                    )
                )
            if patience_counter >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break

        if best_state is None:
            raise RuntimeError("Fold tidak menghasilkan checkpoint")
        model.load_state_dict(best_state)
        model.eval()

        val_event_true = np.asarray(
            [event_labels[index] for index in val_idx], dtype=np.int64
        )
        val_event_max = np.asarray([
            video_max_probability(
                model, event_windows[index], mean, std, device
            )[0]
            for index in val_idx
        ])
        threshold_rows = event_threshold_rows(val_event_true, val_event_max)
        selected_threshold = select_safety_first_threshold(threshold_rows)
        threshold = selected_threshold["threshold"]

        window_probabilities = predict_window_probabilities(model, x_test, device)
        window_prediction = (window_probabilities >= threshold).astype(np.int64)
        window_metrics = binary_metrics(y_test, window_prediction)

        event_true, event_prediction, event_details = [], [], []
        for index in test_idx:
            maximum, window_count = video_max_probability(
                model, event_windows[index], mean, std, device
            )
            prediction = int(maximum >= threshold)
            event_true.append(int(event_labels[index]))
            event_prediction.append(prediction)
            event_details.append({
                "video": video_names[index],
                "subject_id": subjects[index],
                "label": CLASS_NAMES[int(event_labels[index])],
                "prediction": CLASS_NAMES[prediction],
                "max_fall_probability": maximum,
                "total_windows": window_count,
            })
        event_metrics = binary_metrics(event_true, event_prediction)

        checkpoint_path = checkpoint_dir / (
            "test_{}.pt".format(test_subject)
        )
        checkpoint = {
            "checkpoint_version": 2,
            "loso_fold": fold_index,
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "scaler_mean": mean,
            "scaler_std": std,
            "calibrated_threshold": threshold,
            "history": history,
            "training_config": {
                "input_features": input_features,
                "base_features": base_features,
                "feature_pipeline": feature_pipeline,
                "augmentation_profile": augmentation_profile,
                "rnn_type": rnn_type,
            },
            "data_split": {
                "strategy": "subject",
                "evaluation_protocol": "leave-one-subject-out",
                "train_subject_ids": train_subjects,
                "val_subject_ids": [validation_subject],
                "test_subject_ids": [test_subject],
                "subject_metadata_file": SUBJECT_METADATA_PATH.name,
                "subject_metadata_sha256": file_sha256(
                    SUBJECT_METADATA_PATH
                ),
            },
        }
        torch.save(checkpoint, str(checkpoint_path))

        fold_result = {
            "fold": fold_index,
            "train_subjects": train_subjects,
            "validation_subject": validation_subject,
            "test_subject": test_subject,
            "train_videos": int(len(train_idx)),
            "validation_videos": int(len(val_idx)),
            "test_videos": int(len(test_idx)),
            "train_windows": int(len(y_train)),
            "validation_windows": int(len(y_val)),
            "test_windows": int(len(y_test)),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "threshold_calibration": selected_threshold,
            "window_level": window_metrics,
            "event_level": dict(event_metrics, videos=event_details),
            "checkpoint": str(checkpoint_path),
        }
        fold_results.append(fold_result)
        aggregate_window_true.extend(y_test.tolist())
        aggregate_window_pred.extend(window_prediction.tolist())
        aggregate_event_true.extend(event_true)
        aggregate_event_pred.extend(event_prediction)
        print(
            "Fold result: window_acc={:.1%}, event_acc={:.1%}, "
            "event_fall_recall={:.1%}, event_fall_precision={:.1%}, thr={:.2f}"
            .format(
                window_metrics["accuracy"], event_metrics["accuracy"],
                event_metrics["fall_recall"], event_metrics["fall_precision"],
                threshold,
            )
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "method": "leave-one-subject-out",
        "feature_pipeline": feature_pipeline,
        "augmentation_profile": augmentation_profile,
        "rnn_type": rnn_type,
        "subjects": all_subjects,
        "validation_policy": (
            "smallest remaining subject; maximizes training video count"
        ),
        "elapsed_sec": time.time() - start_time,
        "folds": fold_results,
        "aggregate_window_level": binary_metrics(
            aggregate_window_true, aggregate_window_pred
        ),
        "aggregate_event_level": binary_metrics(
            aggregate_event_true, aggregate_event_pred
        ),
    }
    if output_path is None:
        output_path = LOGS_DIR / "loso_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("\nLOSO results saved: {}".format(output_path))
    print(json.dumps({
        "aggregate_window_level": result["aggregate_window_level"],
        "aggregate_event_level": result["aggregate_event_level"],
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-subject-out Fall detection evaluation"
    )
    parser.add_argument("--max-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--test-subject", action="append",
        help="Run only one/repeated test subject(s); default runs all",
    )
    parser.add_argument(
        "--feature-pipeline",
        choices=[LEGACY_FEATURE_PIPELINE, HYBRID_FEATURE_PIPELINE],
        default=FEATURE_PIPELINE,
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=["none", "legacy_v1", "minimal_v2", "full_v2"],
        default=AUGMENTATION_PROFILE,
    )
    parser.add_argument(
        "--rnn-type", choices=["LSTM", "GRU"], default=RNN_TYPE.upper()
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    train_loso(
        max_epochs=args.max_epochs,
        patience=args.patience,
        selected_test_subjects=args.test_subject,
        feature_pipeline=args.feature_pipeline,
        augmentation_profile=args.augmentation_profile,
        rnn_type=args.rnn_type,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
