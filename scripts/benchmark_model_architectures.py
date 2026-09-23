#!/usr/bin/env python3
"""Benchmark checkpoint-bound LSTM and GRU model-only batch-1 latency."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import LOGS_DIR, SEQUENCE_LENGTH  # noqa: E402
from src.model import (  # noqa: E402
    FallDetectionModel,
    checkpoint_input_features,
    checkpoint_rnn_type,
    count_parameters,
)


def load_checkpoint(path, device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def benchmark_checkpoint(path, device, warmup, iterations):
    checkpoint = load_checkpoint(path, device)
    input_features = checkpoint_input_features(checkpoint)
    rnn_type = checkpoint_rnn_type(checkpoint)
    model = FallDetectionModel(
        input_size=input_features, rnn_type=rnn_type
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    values = torch.randn(1, SEQUENCE_LENGTH, input_features, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(values)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples_ms = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(values)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples_ms.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    result = {
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "rnn_type": rnn_type,
        "input_shape": [1, SEQUENCE_LENGTH, input_features],
        "trainable_parameters": count_parameters(model),
        "latency_ms_mean": statistics.mean(samples_ms),
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_p95": ordered[p95_index],
        "iterations": iterations,
        "warmup": warmup,
    }
    del model, values
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lstm-checkpoint", type=Path,
        default=PROJECT_ROOT / "models" / "loso" / "test_S02.pt",
    )
    parser.add_argument(
        "--gru-checkpoint", type=Path,
        default=PROJECT_ROOT / "models" / "loso_gru" / "test_S02.pt",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--output", type=Path,
        default=LOGS_DIR / "rnn_architecture_benchmark.json",
    )
    args = parser.parse_args()
    for path in (args.lstm_checkpoint, args.gru_checkpoint):
        if not path.is_file():
            raise FileNotFoundError("Checkpoint tidak ditemukan: {}".format(path))
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup >= 0 dan iterations > 0 diperlukan")

    requested_devices = [args.device] if args.device != "all" else ["cpu", "cuda"]
    devices = []
    for name in requested_devices:
        if name == "cuda" and not torch.cuda.is_available():
            print("CUDA tidak tersedia; benchmark CUDA dilewati")
            continue
        devices.append(torch.device(name))

    result = {
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "results": {},
    }
    for device in devices:
        print("Benchmark device: {}".format(device))
        rows = [
            benchmark_checkpoint(path, device, args.warmup, args.iterations)
            for path in (args.lstm_checkpoint, args.gru_checkpoint)
        ]
        by_type = {row["rnn_type"]: row for row in rows}
        if set(by_type) != {"LSTM", "GRU"}:
            raise ValueError("Benchmark memerlukan satu LSTM dan satu GRU")
        lstm_latency = by_type["LSTM"]["latency_ms_median"]
        gru_latency = by_type["GRU"]["latency_ms_median"]
        comparison = {
            "parameter_reduction_fraction": (
                1.0
                - by_type["GRU"]["trainable_parameters"]
                / float(by_type["LSTM"]["trainable_parameters"])
            ),
            "checkpoint_size_reduction_fraction": (
                1.0
                - by_type["GRU"]["checkpoint_bytes"]
                / float(by_type["LSTM"]["checkpoint_bytes"])
            ),
            "median_latency_reduction_fraction": (
                1.0 - gru_latency / float(lstm_latency)
            ),
            "median_speedup": lstm_latency / float(gru_latency),
        }
        result["results"][str(device)] = {
            "models": by_type,
            "comparison": comparison,
        }
        print(json.dumps(result["results"][str(device)], indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Saved: {}".format(args.output))


if __name__ == "__main__":
    main()
