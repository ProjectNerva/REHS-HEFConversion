#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash automation.sh <model.pt> <calib_image_dir> [options]

Options:
  --pipeline A|B   Pipeline A (raw HEF) or B (NMS-baked HEF). Default: A
  --height H       Model input height in pixels. Default: 416
  --width  W       Model input width in pixels.  Default: 608
  --gpu            Pass --gpus all to Docker (Linux + NVIDIA Container Toolkit only)

Examples:
  bash automation.sh yolo26n.pt images/
  bash automation.sh yolo26n.pt images/ --pipeline B
  bash automation.sh yolo26n.pt images/ --pipeline B --gpu
  bash automation.sh yolo11n.pt images/ --height 640 --width 640
EOF
    exit 1
}

# ── Defaults ─────────────────────────────────────────────────────────────────
PIPELINE="A"
HEIGHT=416
WIDTH=608
USE_GPU=false

# ── Required positionals ──────────────────────────────────────────────────────
if [ $# -lt 2 ]; then usage; fi
MODEL_PT="$1"
CALIB_DIR="$2"
shift 2

# ── Optional flags ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --height)   HEIGHT="$2";   shift 2 ;;
        --width)    WIDTH="$2";    shift 2 ;;
        --gpu)      USE_GPU=true;  shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

PIPELINE=$(echo "$PIPELINE" | tr '[:lower:]' '[:upper:]')
if [[ "$PIPELINE" != "A" && "$PIPELINE" != "B" ]]; then
    echo "ERROR: --pipeline must be A or B"; exit 1
fi
if [ ! -f "$MODEL_PT" ]; then
    echo "ERROR: Model file not found: $MODEL_PT"; exit 1
fi
if [ ! -d "$CALIB_DIR" ]; then
    echo "ERROR: Calibration directory not found: $CALIB_DIR"; exit 1
fi

MODEL_STEM=$(basename "$MODEL_PT" .pt)
MODEL_DIR=$(cd "$(dirname "$MODEL_PT")" && pwd)
ONNX_SRC="${MODEL_DIR}/${MODEL_STEM}.onnx"
CALIB_FILE="${MODEL_STEM}_calib.npy"

if [[ "$PIPELINE" == "B" ]]; then
    SCRIPT="HEFConversion-pB.py"
    HEF_OUTPUT="${MODEL_STEM}_nms.hef"
else
    SCRIPT="HEFConversion-pA.py"
    HEF_OUTPUT="${MODEL_STEM}.hef"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Pipeline ${PIPELINE} | Model: ${MODEL_STEM} | ${HEIGHT}x${WIDTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── [1/4] Export and simplify ONNX ───────────────────────────────────────────
echo ""
echo "[1/4] Exporting and simplifying ONNX..."
python main.py "$MODEL_PT"

if [ ! -f "$ONNX_SRC" ]; then
    echo "ERROR: Expected ONNX at ${ONNX_SRC} — not found after export."
    exit 1
fi
echo "      OK: ${ONNX_SRC}"

# ── [2/4] Build calibration dataset ──────────────────────────────────────────
echo ""
echo "[2/4] Building calibration dataset (${HEIGHT}x${WIDTH})..."
python build_calib_set.py "$CALIB_DIR" "$CALIB_FILE" "$HEIGHT" "$WIDTH"
echo "      OK: ${CALIB_FILE}"

# ── [3/4] Stage files into shared_data/ ──────────────────────────────────────
echo ""
echo "[3/4] Staging files into shared_data/..."
cp "$ONNX_SRC"  "shared_data/${MODEL_STEM}.onnx"
mv "$CALIB_FILE" "shared_data/${CALIB_FILE}"
echo "      OK: shared_data/${MODEL_STEM}.onnx"
echo "      OK: shared_data/${CALIB_FILE}"

# ── [4/4] Docker conversion ───────────────────────────────────────────────────
echo ""
echo "[4/4] Running Docker conversion (this may take a while)..."

DOCKER_ARGS=(--rm -v "$(pwd)/shared_data:/app/shared_data")
if [ "$USE_GPU" = true ]; then
    DOCKER_ARGS+=(--gpus all)
    echo "      GPU passthrough enabled"
fi

docker run "${DOCKER_ARGS[@]}" hef-conversion \
    bash -c "cd /app/shared_data && python3 ${SCRIPT} ${MODEL_STEM} ${MODEL_STEM}.onnx ${CALIB_FILE}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! Output: shared_data/${HEF_OUTPUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
