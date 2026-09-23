"""
model.py - Arsitektur 1D CNN + LSTM/GRU + Attention untuk Fall Detection.

Arsitektur:
    Input (batch, seq_len, 130)
        → 1D CNN layers (ekstraksi fitur spasial lokal)
        → LSTM/GRU layers (tangkap dependensi temporal)
        → Attention mechanism (fokus pada frame penting)
        → Fully Connected → Output (2 kelas: ADL vs Fall)

Penggunaan:
    from src.model import FallDetectionModel
    model = FallDetectionModel()
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    INPUT_FEATURES, SEQUENCE_LENGTH, NUM_CLASSES,
    CNN_FILTERS, CNN_KERNEL_SIZE, CNN_DROPOUT,
    RNN_TYPE, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, RNN_DROPOUT, RNN_BIDIRECTIONAL,
    USE_ATTENTION, FC_HIDDEN_SIZE, FC_DROPOUT,
    LEGACY_FEATURE_PIPELINE,
)


def checkpoint_input_features(checkpoint):
    """Read model input width from the checkpoint, including legacy files."""
    state_dict = checkpoint.get("model_state_dict", {})
    first_conv = state_dict.get("cnn.0.weight")
    inferred = int(first_conv.shape[1]) if first_conv is not None else None
    configured = checkpoint.get("training_config", {}).get("input_features")
    if configured is not None and inferred is not None and int(configured) != inferred:
        raise ValueError(
            "Checkpoint input_features={} tetapi CNN mengharapkan {}".format(
                configured, inferred
            )
        )
    if configured is not None:
        return int(configured)
    if inferred is not None:
        return inferred
    raise ValueError("Tidak dapat menentukan jumlah input feature checkpoint")


def checkpoint_feature_pipeline(checkpoint):
    """Checkpoints created before pipeline metadata use exact legacy features."""
    return checkpoint.get("training_config", {}).get(
        "feature_pipeline", LEGACY_FEATURE_PIPELINE
    )


def checkpoint_rnn_type(checkpoint):
    """Return LSTM/GRU architecture from metadata or recurrent gate shape."""
    configured = checkpoint.get("training_config", {}).get("rnn_type")
    recurrent_weight = checkpoint.get("model_state_dict", {}).get(
        "rnn.weight_ih_l0"
    )
    inferred = None
    if recurrent_weight is not None:
        hidden_size = int(
            checkpoint["model_state_dict"]["rnn.weight_hh_l0"].shape[1]
        )
        gate_count = int(recurrent_weight.shape[0]) // hidden_size
        if gate_count == 4:
            inferred = "LSTM"
        elif gate_count == 3:
            inferred = "GRU"
    if configured is not None:
        configured = str(configured).upper()
        if configured not in ("LSTM", "GRU"):
            raise ValueError("Checkpoint rnn_type tidak dikenal: " + configured)
        if inferred is not None and configured != inferred:
            raise ValueError(
                "Checkpoint rnn_type={} tetapi weight menunjukkan {}".format(
                    configured, inferred
                )
            )
        return configured
    return inferred or "LSTM"


# ============================================================
# ATTENTION MECHANISM
# ============================================================
class Attention(nn.Module):
    """
    Additive Attention (Bahdanau-style).

    Menghitung bobot perhatian untuk setiap timestep dari output RNN,
    sehingga model bisa fokus pada frame-frame yang paling informatif
    (misal: saat tubuh mulai jatuh).

    Input:  (batch, seq_len, hidden_size)
    Output: (batch, hidden_size) — weighted sum dari semua timestep
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, rnn_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_output: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — vektor konteks
            weights: (batch, seq_len) — bobot attention per timestep
        """
        # Hitung skor attention untuk setiap timestep
        scores = self.attention(rnn_output)          # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)       # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(weights * rnn_output, dim=1)  # (batch, hidden_size)
        weights = weights.squeeze(-1)                      # (batch, seq_len)

        return context, weights


# ============================================================
# MODEL UTAMA: 1D CNN + LSTM/GRU + ATTENTION
# ============================================================
class FallDetectionModel(nn.Module):
    """
    Model Fall Detection: 1D CNN + LSTM/GRU + Attention.

    Alur data:
        1. Conv1D blocks: ekstrak pola lokal antar landmark
        2. LSTM/GRU: tangkap pola temporal antar frame
        3. Attention: bobot untuk frame-frame penting
        4. FC layers: klasifikasi akhir
    """

    def __init__(
        self,
        input_size: int = INPUT_FEATURES,
        cnn_filters: List[int] = CNN_FILTERS,
        cnn_kernel_size: int = CNN_KERNEL_SIZE,
        cnn_dropout: float = CNN_DROPOUT,
        rnn_type: str = RNN_TYPE,
        rnn_hidden_size: int = RNN_HIDDEN_SIZE,
        rnn_num_layers: int = RNN_NUM_LAYERS,
        rnn_dropout: float = RNN_DROPOUT,
        rnn_bidirectional: bool = RNN_BIDIRECTIONAL,
        use_attention: bool = USE_ATTENTION,
        fc_hidden_size: int = FC_HIDDEN_SIZE,
        fc_dropout: float = FC_DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        rnn_type = rnn_type.upper()
        if rnn_type not in ("LSTM", "GRU"):
            raise ValueError("rnn_type harus LSTM atau GRU")
        self.rnn_type = rnn_type
        self.use_attention = use_attention
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_hidden_size = rnn_hidden_size

        # ========================================
        # 1D CNN BLOCK
        # ========================================
        # Input: (batch, seq_len, input_size)
        # Conv1D expects: (batch, channels, seq_len)
        cnn_layers = []
        in_channels = input_size

        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=cnn_kernel_size, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # ========================================
        # LSTM / GRU BLOCK
        # ========================================
        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU

        self.rnn = rnn_class(
            input_size=cnn_filters[-1],       # Output dari CNN terakhir
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0,
            bidirectional=rnn_bidirectional,
        )

        # Ukuran output RNN (doubled jika bidirectional)
        rnn_output_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)

        # ========================================
        # ATTENTION BLOCK
        # ========================================
        if use_attention:
            self.attention = Attention(rnn_output_size)

        # ========================================
        # FULLY CONNECTED (CLASSIFIER)
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor shape (batch, seq_len, 130)

        Returns:
            logits: (batch, num_classes) — raw logits (sebelum softmax)
            attention_weights: (batch, seq_len) atau None jika attention off
        """
        # === CNN ===
        # Transpose: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # Transpose kembali: (batch, channels, seq_len) → (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # === RNN ===
        rnn_output, _ = self.rnn(x)  # (batch, seq_len, rnn_output_size)

        # === Attention atau Last Hidden ===
        attention_weights = None
        if self.use_attention:
            context, attention_weights = self.attention(rnn_output)
        else:
            # Ambil output timestep terakhir
            context = rnn_output[:, -1, :]  # (batch, rnn_output_size)

        # === Classifier ===
        logits = self.classifier(context)  # (batch, num_classes)

        return logits, attention_weights


# ============================================================
# UTILITY: Model Summary
# ============================================================
def count_parameters(model: nn.Module) -> int:
    """Hitung jumlah parameter yang trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """Print ringkasan model."""
    print(f"\n{'='*60}")
    print(f"  MODEL SUMMARY")
    print(f"{'='*60}")
    print(model)
    print(f"\n  Total parameters    : {count_parameters(model):,}")
    print(f"  Input shape         : (batch, {SEQUENCE_LENGTH}, {INPUT_FEATURES})")
    print(f"  Output shape        : (batch, {NUM_CLASSES})")

    # Test forward pass
    device = "cpu"
    model = model.to(device)
    dummy = torch.randn(2, SEQUENCE_LENGTH, INPUT_FEATURES).to(device)
    logits, attn_w = model(dummy)
    print(f"  Test forward pass   : logits={logits.shape}", end="")
    if attn_w is not None:
        print(f", attention={attn_w.shape}")
    else:
        print()
    print()


if __name__ == "__main__":
    model = FallDetectionModel()
    print_model_summary(model)
