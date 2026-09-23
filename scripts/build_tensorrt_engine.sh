#!/usr/bin/env bash
# =============================================================================
# build_tensorrt_engine.sh - Compile YOLOv8n-Pose ONNX to TensorRT Engine (FP16)
# Run this script on NVIDIA Jetson Nano inside or outside Docker container.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_PATH="$PROJECT_ROOT/models/yolov8n-pose.onnx"
ENGINE_PATH="$PROJECT_ROOT/models/yolov8n-pose.engine"

TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
if ! command -v trtexec &>/dev/null; then
  if [[ -x "$TRTEXEC_BIN" ]]; then
    TRTEXEC="$TRTEXEC_BIN"
  else
    echo "[ERROR] trtexec tidak ditemukan di PATH atau $TRTEXEC_BIN."
    echo "Pastikan TensorRT sudah terinstall di Jetson Nano (JetPack 4.6)."
    exit 1
  fi
else
  TRTEXEC="trtexec"
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "[ERROR] ONNX model tidak ditemukan di $ONNX_PATH."
  exit 1
fi

echo "============================================================"
echo " Compiling YOLOv8n-Pose to TensorRT FP16 Engine on Jetson"
echo " Input  : $ONNX_PATH"
echo " Output : $ENGINE_PATH"
echo "============================================================"

$TRTEXEC \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --workspace=512 \
  --verbose

echo "============================================================"
echo " [SUCCESS] TensorRT Engine berhasil dibuat di: $ENGINE_PATH"
echo "============================================================"
