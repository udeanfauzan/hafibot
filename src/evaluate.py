"""
evaluate.py - Evaluasi model Fall Detection pada test set.

Output:
    - Classification report (precision, recall, F1-score)
    - Confusion matrix (heatmap)
    - Training curves (loss & accuracy)
    - Semua grafik disimpan ke results/figures/

Penggunaan:
    python src/evaluate.py
"""

import argparse
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, MODEL_DIR, FIGURES_DIR, LOGS_DIR,
    CLASS_NAMES, NUM_CLASSES,
    TEST_RATIO, RANDOM_SEED,
    FALL_CONFIDENCE_THRESHOLD,
)
from src.dataset import get_dataloaders, load_all_data, split_video_indices
from src.model import (
    FallDetectionModel,
    checkpoint_feature_pipeline,
    checkpoint_input_features,
    checkpoint_rnn_type,
)


# ============================================================
# PREDICTION
# ============================================================
@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Dapatkan semua prediksi dan label dari DataLoader.

    Returns:
        y_true: np.ndarray — label sebenarnya
        y_pred: np.ndarray — label prediksi
        y_probs: np.ndarray — probabilitas prediksi
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits, _ = model(X_batch)
        probs = torch.softmax(logits, dim=1)
        _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def checkpoint_split_config(checkpoint):
    """Use checkpoint-bound split metadata; old checkpoints keep old split."""
    split = checkpoint.get("data_split")
    if not split:
        return {
            "strategy": "video_stratified",
            "train_subject_ids": (),
            "val_subject_ids": (),
            "test_subject_ids": (),
        }
    return {
        "strategy": split["strategy"],
        "train_subject_ids": tuple(split.get("train_subject_ids", ())),
        "val_subject_ids": tuple(split.get("val_subject_ids", ())),
        "test_subject_ids": tuple(split.get("test_subject_ids", ())),
    }


@torch.no_grad()
def get_video_event_predictions(model, device, checkpoint, threshold, feature_root=None):
    """Agregasi window menjadi event: satu video positif jika ada window Fall."""
    # Runtime memprediksi setiap pose frame setelah buffer penuh (stride=1),
    # bukan hanya setiap STRIDE training window.
    split_config = checkpoint_split_config(checkpoint)
    feature_pipeline = checkpoint_feature_pipeline(checkpoint)
    feature_root = feature_root or checkpoint.get("training_config", {}).get("feature_root")
    data = load_all_data(
        window_stride=1,
        include_ambiguous=True,
        return_subject_ids=(split_config["strategy"] == "subject"),
        require_subject_metadata=(split_config["strategy"] == "subject"),
        feature_pipeline=feature_pipeline,
        feature_root=feature_root,
    )
    video_windows, _video_window_labels, video_event_labels, video_names = data[:4]
    video_subject_ids = data[4] if len(data) == 5 else None
    labels = np.asarray(video_event_labels)
    _, _, test_idx = split_video_indices(
        labels,
        video_subject_ids=video_subject_ids,
        split_strategy=split_config["strategy"],
        train_subject_ids=split_config["train_subject_ids"],
        val_subject_ids=split_config["val_subject_ids"],
        test_subject_ids=split_config["test_subject_ids"],
    )

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["scaler_std"], dtype=np.float32)
    y_true, y_pred, details = [], [], []

    model.eval()
    for idx in test_idx:
        windows = np.asarray(video_windows[idx], dtype=np.float32)
        if len(windows) == 0:
            fall_probs = np.empty(0, dtype=np.float32)
            event_prediction = 0
            max_fall_probability = 0.0
        else:
            windows = (windows - mean) / std
            tensor = torch.FloatTensor(windows).to(device)
            logits, _ = model(tensor)
            fall_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            event_prediction = int(np.any(fall_probs >= threshold))
            max_fall_probability = float(fall_probs.max())

        y_true.append(int(labels[idx]))
        y_pred.append(event_prediction)
        details.append({
            "video": video_names[idx],
            "subject_id": (
                video_subject_ids[idx] if video_subject_ids is not None else None
            ),
            "label": CLASS_NAMES[int(labels[idx])],
            "prediction": CLASS_NAMES[event_prediction],
            "max_fall_probability": max_fall_probability,
            "positive_windows": int(np.sum(fall_probs >= threshold)),
            "total_windows": int(len(fall_probs)),
        })

    return np.asarray(y_true), np.asarray(y_pred), details


# ============================================================
# PLOT: CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """Plot dan simpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0], cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Confusion Matrix (Count)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)

    # Normalized (percentage)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt=".1%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[1], cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


# ============================================================
# PLOT: TRAINING CURVES
# ============================================================
def plot_training_curves(history: dict, save_path: Path):
    """Plot loss dan accuracy curves selama training."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    # Learning rate
    axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {save_path}")


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate_model(
    threshold: float = FALL_CONFIDENCE_THRESHOLD,
    checkpoint_path=None,
    results_path=None,
    figures_dir=None,
    history_path=None,
    feature_root=None,
):
    """Entry point: evaluasi model terbaik pada test set."""

    checkpoint_path = Path(
        checkpoint_path or (MODEL_DIR / "best_model.pt")
    )
    results_path = Path(
        results_path or (LOGS_DIR / "evaluation_results.json")
    )
    figures_dir = Path(figures_dir or FIGURES_DIR)
    history_path = Path(
        history_path or (LOGS_DIR / "training_history.npy")
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  FALL DETECTION - Evaluation")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load model
    best_model_path = checkpoint_path
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {best_model_path}")
    try:
        checkpoint = torch.load(str(best_model_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(best_model_path), map_location=device)
    feature_pipeline = checkpoint_feature_pipeline(checkpoint)
    feature_root = feature_root or checkpoint.get("training_config", {}).get("feature_root")
    input_features = checkpoint_input_features(checkpoint)
    model = FallDetectionModel(
        input_size=input_features,
        rnn_type=checkpoint_rnn_type(checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Model loaded (epoch/fold: {checkpoint.get('epoch', checkpoint.get('fold', 'unknown'))})")

    # Data harus dinormalisasi dengan scaler dari checkpoint yang sama.
    if "scaler_mean" not in checkpoint or "scaler_std" not in checkpoint:
        raise ValueError(
            "Checkpoint tidak memiliki embedded scaler. Retrain model dengan "
            "pipeline terbaru sebelum evaluasi."
        )
    split_config = checkpoint_split_config(checkpoint)
    _, val_loader, test_loader = get_dataloaders(
        normalization=(checkpoint["scaler_mean"], checkpoint["scaler_std"]),
        save_scaler=False,
        split_strategy=split_config["strategy"],
        train_subject_ids=split_config["train_subject_ids"],
        val_subject_ids=split_config["val_subject_ids"],
        test_subject_ids=split_config["test_subject_ids"],
        feature_pipeline=feature_pipeline,
        feature_root=feature_root,
    )

    # ========================================
    # TEST SET EVALUATION
    # ========================================
    y_true, _, y_probs = get_predictions(model, test_loader, device)
    y_pred = (y_probs[:, 1] >= threshold).astype(np.int64)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"  TEST WINDOW RESULTS (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy:.1%})")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # ========================================
    # EVENT-LEVEL EVALUATION
    # ========================================
    event_true, event_pred, event_details = get_video_event_predictions(
        model, device, checkpoint, threshold, feature_root=feature_root,
    )
    event_accuracy = accuracy_score(event_true, event_pred)
    event_precision = precision_score(event_true, event_pred, pos_label=1)
    event_recall = recall_score(event_true, event_pred, pos_label=1)
    event_f1 = f1_score(event_true, event_pred, pos_label=1)
    event_cm = confusion_matrix(event_true, event_pred)

    print(f"\n{'='*60}")
    print("  TEST VIDEO EVENT RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy       : {event_accuracy:.4f} ({event_accuracy:.1%})")
    print(f"  Fall precision : {event_precision:.4f}")
    print(f"  Fall recall    : {event_recall:.4f}")
    print(f"  Fall F1        : {event_f1:.4f}")
    print(f"  Confusion      : {event_cm.tolist()}")

    false_events = [
        item for item in event_details
        if item["label"] != item["prediction"]
    ]
    if false_events:
        print("\n  Misclassified videos:")
        for item in false_events:
            print(
                f"    {item['video']}: {item['label']} -> {item['prediction']} "
                f"(max={item['max_fall_probability']:.1%}, "
                f"positive={item['positive_windows']}/{item['total_windows']})"
            )

    # ========================================
    # PLOTS
    # ========================================
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")

    # Training curves (dari history yang tersimpan)
    if history_path.exists():
        history = np.load(str(history_path), allow_pickle=True).item()
        plot_training_curves(history, figures_dir / "training_curves.png")
    else:
        print(f"  [WARNING] Training history tidak ditemukan: {history_path}")

    evaluation_results = {
        "threshold": threshold,
        "feature_root": str(feature_root) if feature_root else None,
        "evaluation_scope": "offline classifier windows and video maxima; no runtime posture gates",
        "data_split": checkpoint.get("data_split", {
            "strategy": "video_stratified"
        }),
        "window_level": {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "fall_precision": precision_score(y_true, y_pred, pos_label=1),
            "fall_recall": recall_score(y_true, y_pred, pos_label=1),
            "fall_f1": f1_score(y_true, y_pred, pos_label=1),
        },
        "event_level": {
            "accuracy": event_accuracy,
            "fall_precision": event_precision,
            "fall_recall": event_recall,
            "fall_f1": event_f1,
            "confusion_matrix": event_cm.tolist(),
            "videos": event_details,
        },
    }
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_results, handle, indent=2)
    print(f"  Evaluation JSON saved: {results_path}")

    print(f"\n  Semua hasil tersimpan di: {figures_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fall detector")
    parser.add_argument(
        "--threshold", type=float, default=FALL_CONFIDENCE_THRESHOLD,
        help="Deployment Fall probability threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt",
        help="Checkpoint to evaluate",
    )
    parser.add_argument(
        "--feature-root", type=Path,
        help="Override the checkpoint cache for an explicit comparison",
    )
    parser.add_argument(
        "--output", type=Path,
        default=LOGS_DIR / "evaluation_results.json",
        help="Evaluation JSON output path",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR,
        help="Directory for confusion matrix and training curves",
    )
    parser.add_argument(
        "--history", type=Path, default=LOGS_DIR / "training_history.npy",
        help="Training history used for the curve plot",
    )
    args = parser.parse_args()
    evaluate_model(
        threshold=args.threshold,
        checkpoint_path=args.checkpoint,
        results_path=args.output,
        figures_dir=args.figures_dir,
        history_path=args.history,
        feature_root=args.feature_root,
    )
