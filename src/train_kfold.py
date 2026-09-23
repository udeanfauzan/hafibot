"""
train_kfold.py - Training dengan Stratified K-Fold Cross-Validation.

Mengapa K-Fold?
    - Dataset kecil (70 video) → single split sangat noisy
    - K-Fold menggunakan SEMUA data untuk training & testing
    - Hasil lebih robust dan reliable untuk laporan tugas akhir

Pipeline:
    1. Load semua video → feature engineering → sliding windows
    2. Split per VIDEO menggunakan StratifiedKFold (5 fold)
    3. Untuk setiap fold: normalize → train → evaluate
    4. Agregasi hasil dari semua fold → mean +/- std

Penggunaan:
    python src/train_kfold.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, MODEL_DIR, FIGURES_DIR, LOGS_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, RANDOM_SEED,
    BATCH_SIZE, N_FOLDS, CLASS_NAMES,
    SEQUENCE_LENGTH, STRIDE, INPUT_FEATURES,
    FEATURE_PIPELINE, BASE_FEATURES, AUGMENTATION_PROFILE,
    RNN_TYPE,
    USE_AUGMENTATION, USE_VELOCITY, USE_ACCELERATION,
    FALL_MIN_OVERLAP_FRAMES, TRAINING_DATASETS,
)
from src.dataset import (
    load_all_data, sliding_window,
    FallDetectionDataset, apply_augmentation,
)
from src.model import FallDetectionModel, count_parameters
from src.temporal_labels import file_sha256
from torch.utils.data import DataLoader


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = RANDOM_SEED):
    """Set random seed untuk reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# TRAINING & EVALUATION (per fold)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model untuk satu epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluasi model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    """Dapatkan semua prediksi."""
    model.eval()
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits, _ = model(X_batch)
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)


def normalize_data(X_train, X_val):
    """Normalize berdasarkan training data."""
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)

    train_mean = X_train_flat.mean(axis=0)
    train_std = X_train_flat.std(axis=0)
    train_std[train_std == 0] = 1.0

    return X_train, X_val, train_mean, train_std


# ============================================================
# K-FOLD TRAINING
# ============================================================
def train_kfold():
    """Training dengan Stratified K-Fold Cross-Validation."""
    set_seed()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FALL DETECTION - K-Fold Cross-Validation Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  K-Folds     : {N_FOLDS}")
    print(f"  Seq Length   : {SEQUENCE_LENGTH}")
    print(f"  Stride       : {STRIDE}")
    print(f"  Features     : {INPUT_FEATURES}")
    print(f"  Augmentation : {'ON' if USE_AUGMENTATION else 'OFF'}")

    # Load semua data
    (
        video_windows,
        video_window_labels,
        video_event_labels,
        video_names,
    ) = load_all_data()
    n_videos = len(video_event_labels)
    labels_array = np.array(video_event_labels)
    fall_labeling_metadata = {
        "method": "visible_pose_interval_overlap",
        "annotation_schema_version": 1,
        "annotation_files": {
            source["name"]: {
                "file": source["fall_annotations_path"].name,
                "sha256": file_sha256(source["fall_annotations_path"]),
            }
            for source in TRAINING_DATASETS
        },
        "minimum_fall_overlap_frames": FALL_MIN_OVERLAP_FRAMES,
        "ambiguous_boundary_policy": "exclude",
        "fall_overlap_requires_visible_pose": True,
        "window_end_requires_visible_pose": True,
    }

    print(f"\n  Total videos : {n_videos}")
    print(f"  ADL videos   : {sum(labels_array == 0)}")
    print(f"  Fall videos  : {sum(labels_array == 1)}")

    # K-Fold splitter (split per VIDEO)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Storage untuk hasil per fold
    fold_results = []
    all_y_true = []
    all_y_pred = []
    best_fold_acc = 0
    best_fold_idx = -1

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(n_videos), labels_array), 1):
        # Bersihkan VRAM sebelum fold baru
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{N_FOLDS}")
        print(f"{'='*60}")

        # Collect windows per split
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []

        train_videos = []
        val_videos = []

        for idx in train_idx:
            for window, window_label in zip(
                video_windows[idx], video_window_labels[idx]
            ):
                X_train_list.append(window)
                y_train_list.append(window_label)
            train_videos.append(video_names[idx])

        for idx in val_idx:
            for window, window_label in zip(
                video_windows[idx], video_window_labels[idx]
            ):
                X_val_list.append(window)
                y_val_list.append(window_label)
            val_videos.append(video_names[idx])

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int64)
        X_val = np.array(X_val_list, dtype=np.float32)
        y_val = np.array(y_val_list, dtype=np.int64)

        # Normalize
        X_train, X_val, train_mean, train_std = normalize_data(X_train, X_val)

        print(f"  Train: {len(train_idx)} video -> {len(X_train)} windows "
              f"(ADL={sum(y_train==0)}, Fall={sum(y_train==1)})")
        print(f"  Val  : {len(val_idx)} video -> {len(X_val)} windows "
              f"(ADL={sum(y_val==0)}, Fall={sum(y_val==1)})")

        # DataLoaders
        scaler = (train_mean, train_std)
        train_dataset = FallDetectionDataset(
            X_train, y_train, augment=True, normalization=scaler,
            feature_pipeline=FEATURE_PIPELINE,
        )
        val_dataset = FallDetectionDataset(
            X_val, y_val, augment=False, normalization=scaler,
            feature_pipeline=FEATURE_PIPELINE,
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # Class weights
        label_counts = Counter(y_train.tolist())
        total_samples = sum(label_counts.values())
        class_weights = torch.FloatTensor([
            total_samples / (len(label_counts) * label_counts[i])
            for i in range(len(label_counts))
        ]).to(device)

        # Model (fresh untuk setiap fold)
        set_seed(RANDOM_SEED + fold_idx)
        model = FallDetectionModel().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

        # Training loop
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                status = "*"
            else:
                patience_counter += 1

            if epoch <= 5 or epoch % 5 == 0 or status == "*" or patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Epoch {epoch:>3}: train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.1%} | lr={current_lr:.6f} {status}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best model dan evaluasi
        model.load_state_dict(best_state)
        y_true, y_pred = get_predictions(model, val_loader, device)

        fold_acc = accuracy_score(y_true, y_pred)
        fold_f1 = f1_score(y_true, y_pred, average="weighted")
        fold_precision = precision_score(y_true, y_pred, average="weighted")
        fold_recall = recall_score(y_true, y_pred, average="weighted")

        # Per-class metrics
        fall_recall = recall_score(y_true, y_pred, pos_label=1)
        fall_precision = precision_score(y_true, y_pred, pos_label=1)

        fold_results.append({
            "fold": fold_idx,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "precision": fold_precision,
            "recall": fold_recall,
            "fall_recall": fall_recall,
            "fall_precision": fall_precision,
            "best_epoch": epoch - patience_counter,
            "val_loss": best_val_loss,
        })

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(f"\n  Fold {fold_idx} Results: Acc={fold_acc:.1%}, F1={fold_f1:.4f}, "
              f"Fall Recall={fall_recall:.1%}, Fall Precision={fall_precision:.1%}")

        # Simpan model fold terbaik
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            best_fold_idx = fold_idx
            torch.save({
                "checkpoint_version": 2,
                "fold": fold_idx,
                "model_state_dict": best_state,
                "val_acc": fold_acc,
                "scaler_mean": train_mean,
                "scaler_std": train_std,
                "training_config": {
                    "sequence_length": SEQUENCE_LENGTH,
                    "stride": STRIDE,
                    "input_features": INPUT_FEATURES,
                    "base_features": BASE_FEATURES,
                    "feature_pipeline": FEATURE_PIPELINE,
                    "augmentation_profile": AUGMENTATION_PROFILE,
                    "rnn_type": RNN_TYPE,
                    "class_names": CLASS_NAMES,
                    "random_seed": RANDOM_SEED,
                },
                "fall_labeling_metadata": fall_labeling_metadata,
            }, str(MODEL_DIR / "best_model_kfold.pt"))
            # Simpan scaler terbaik
            np.savez(
                str(MODEL_DIR / "scaler_params_kfold.npz"),
                mean=train_mean,
                std=train_std,
            )

    elapsed = time.time() - start_time

    # ========================================
    # AGGREGATE RESULTS
    # ========================================
    print(f"\n{'='*60}")
    print(f"  K-FOLD CROSS-VALIDATION RESULTS ({N_FOLDS} folds)")
    print(f"{'='*60}")
    print(f"  Waktu total: {elapsed:.1f} detik")

    # Table header
    print(f"\n  {'Fold':>4} | {'Acc':>7} | {'F1':>7} | {'Precision':>9} | {'Recall':>7} | "
          f"{'Fall Rec':>8} | {'Fall Prec':>9} | {'Best Ep':>7}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}")

    for r in fold_results:
        print(f"  {r['fold']:>4} | {r['accuracy']:>6.1%} | {r['f1']:>7.4f} | {r['precision']:>8.1%} | "
              f"{r['recall']:>6.1%} | {r['fall_recall']:>7.1%} | {r['fall_precision']:>8.1%} | "
              f"{r['best_epoch']:>7}")

    # Mean +/- Std
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    fall_recalls = [r["fall_recall"] for r in fold_results]
    fall_precs = [r["fall_precision"] for r in fold_results]

    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}")
    print(f"  Mean | {np.mean(accs):>6.1%} | {np.mean(f1s):>7.4f} | "
          f"         |        | {np.mean(fall_recalls):>7.1%} | {np.mean(fall_precs):>8.1%} |")
    print(f"  Std  | {np.std(accs):>6.1%} | {np.std(f1s):>7.4f} | "
          f"         |        | {np.std(fall_recalls):>7.1%} | {np.std(fall_precs):>8.1%} |")

    print(f"\n  Best fold: {best_fold_idx} (Acc={best_fold_acc:.1%})")
    print(f"  Best model saved to: {MODEL_DIR / 'best_model_kfold.pt'}")

    # ========================================
    # AGGREGATED CONFUSION MATRIX
    # ========================================
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Overall classification report
    print(f"\n  Aggregated Classification Report (all folds combined):")
    print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES, digits=4))

    # Plot aggregated confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title(f"Confusion Matrix - {N_FOLDS}-Fold CV (Count)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title(f"Confusion Matrix - {N_FOLDS}-Fold CV (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)

    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix_kfold.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    # Plot per-fold accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [r["fold"] for r in fold_results]
    ax.bar(folds, accs, color="#4C72B0", alpha=0.8, label="Accuracy")
    ax.bar([f + 0.3 for f in folds], fall_recalls, width=0.3, color="#DD8452", alpha=0.8, label="Fall Recall")
    ax.axhline(y=np.mean(accs), color="#4C72B0", linestyle="--", linewidth=1.5, label=f"Mean Acc ({np.mean(accs):.1%})")
    ax.axhline(y=np.mean(fall_recalls), color="#DD8452", linestyle="--", linewidth=1.5, label=f"Mean Fall Recall ({np.mean(fall_recalls):.1%})")
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Fold Performance ({N_FOLDS}-Fold CV)", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fold_path = FIGURES_DIR / "kfold_performance.png"
    plt.savefig(str(fold_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-fold plot saved: {fold_path}")

    # Simpan results
    np.save(str(LOGS_DIR / "kfold_results.npy"), fold_results)
    print(f"\n  Semua hasil tersimpan!")
    print()

    return fold_results


if __name__ == "__main__":
    train_kfold()
