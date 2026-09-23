"""
train.py - Training loop untuk model Fall Detection.

Fitur:
    - Training dengan GPU (CUDA) jika tersedia
    - Early stopping berdasarkan validation loss
    - Learning rate scheduler (ReduceLROnPlateau)
    - Class weighting untuk menangani imbalance (ADL > Fall)
    - Logging loss & accuracy per epoch
    - Simpan best model ke models/

Penggunaan:
    python src/train.py
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, MODEL_DIR, LOGS_DIR, PROCESSED_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, RANDOM_SEED,
    SEQUENCE_LENGTH, STRIDE, INPUT_FEATURES, CLASS_NAMES,
    FEATURE_PIPELINE, BASE_FEATURES, AUGMENTATION_PROFILE,
    RNN_TYPE,
    MP_MODEL_VARIANT, EXPECTED_VIDEO_FPS,
    FALL_MIN_OVERLAP_FRAMES, TRAINING_DATASETS, SUBJECT_METADATA_PATH,
    SPLIT_STRATEGY, SUBJECT_TRAIN_IDS, SUBJECT_VAL_IDS, SUBJECT_TEST_IDS,
)
from src.dataset import get_dataloaders
from src.model import FallDetectionModel, count_parameters
from src.temporal_labels import file_sha256


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
# TRAINING FUNCTIONS
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

        # Forward pass
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (mencegah exploding gradients pada RNN)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistik
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluasi model pada validation/test set."""
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train(
    checkpoint_path=None,
    scaler_path=None,
    history_path=None,
    split_strategy=SPLIT_STRATEGY,
):
    """Entry point: training loop lengkap."""
    set_seed()

    checkpoint_path = Path(
        checkpoint_path or (MODEL_DIR / "best_model.pt")
    )
    scaler_path = Path(
        scaler_path or (MODEL_DIR / "scaler_params.npz")
    )
    history_path = Path(
        history_path or (LOGS_DIR / "training_history.npy")
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FALL DETECTION - Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        return_scaler=True,
        save_scaler=False,
        split_strategy=split_strategy,
    )
    scaler_mean, scaler_std = scaler
    np.savez(str(scaler_path), mean=scaler_mean, std=scaler_std)
    print(f"    Scaler saved: {scaler_path}")
    data_split_metadata = {
        "strategy": split_strategy,
        "train_subject_ids": list(SUBJECT_TRAIN_IDS),
        "val_subject_ids": list(SUBJECT_VAL_IDS),
        "test_subject_ids": list(SUBJECT_TEST_IDS),
        "subject_metadata_file": SUBJECT_METADATA_PATH.name,
        "subject_metadata_sha256": file_sha256(SUBJECT_METADATA_PATH),
    }
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

    extraction_metadata = {}
    for source in TRAINING_DATASETS:
        extraction_metadata_path = (
            source["processed_adl_dir"].parent / "extraction_metadata.json"
        )
        if extraction_metadata_path.exists():
            with extraction_metadata_path.open("r", encoding="utf-8") as handle:
                extraction_metadata[source["name"]] = json.load(handle)
        else:
            print(
                f"  [WARNING] Metadata ekstraksi tidak ditemukan: "
                f"{extraction_metadata_path}"
            )

    # Hitung class weights (menangani imbalance ADL vs Fall)
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.extend(y_batch.numpy())
    label_counts = Counter(all_labels)
    total_samples = sum(label_counts.values())
    class_weights = torch.FloatTensor([
        total_samples / (len(label_counts) * label_counts[i])
        for i in range(len(label_counts))
    ]).to(device)
    print(f"\n  Class distribution (train): {dict(label_counts)}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    # Model
    model = FallDetectionModel().to(device)
    print(f"  Model parameters: {count_parameters(model):,}")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7,
    )

    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_model_path = checkpoint_path

    print(f"\n{'='*60}")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} | {'LR':>10} | Status")
    print(f"  {'-'*5}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-+-------")

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Best model check
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "checkpoint_version": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history,
                "scaler_mean": scaler_mean,
                "scaler_std": scaler_std,
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
                    "pose_model_variant": MP_MODEL_VARIANT,
                    "expected_video_fps": EXPECTED_VIDEO_FPS,
                },
                "extraction_metadata": extraction_metadata,
                "fall_labeling_metadata": fall_labeling_metadata,
                "data_split": data_split_metadata,
            }, str(best_model_path))
            status = "* BEST"
        else:
            patience_counter += 1

        # Log
        print(f"  {epoch:>5} | {train_loss:>10.4f} {train_acc:>9.1%} | "
              f"{val_loss:>10.4f} {val_acc:>9.1%} | {current_lr:>10.6f} | {status}")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    elapsed = time.time() - start_time

    # ========================================
    # FINAL EVALUATION ON TEST SET
    # ========================================
    # Load best model
    try:
        checkpoint = torch.load(str(best_model_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  TRAINING SELESAI! (waktu: {elapsed:.1f} detik)")
    print(f"{'='*60}")
    print(f"  Best epoch       : {checkpoint['epoch']}")
    print(f"  Best val loss    : {best_val_loss:.4f}")
    print(f"  Best val accuracy: {best_val_acc:.1%}")
    print(f"  Test loss        : {test_loss:.4f}")
    print(f"  Test accuracy    : {test_acc:.1%}")
    print(f"  Model saved to   : {best_model_path}")
    print()

    # Simpan history
    np.save(str(history_path), history)
    print(f"  History saved to  : {history_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fall detector")
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_DIR / "best_model.pt",
        help="Best-checkpoint output path",
    )
    parser.add_argument(
        "--scaler", type=Path, default=MODEL_DIR / "scaler_params.npz",
        help="Legacy scaler output path",
    )
    parser.add_argument(
        "--history", type=Path, default=LOGS_DIR / "training_history.npy",
        help="Training-history output path",
    )
    parser.add_argument(
        "--split-strategy", choices=("video_stratified", "subject"),
        default=SPLIT_STRATEGY,
        help="Video-level split strategy (default: %(default)s)",
    )
    args = parser.parse_args()
    model, history = train(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        history_path=args.history,
        split_strategy=args.split_strategy,
    )
