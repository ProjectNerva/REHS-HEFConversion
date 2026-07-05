#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash automation.sh <model.pt|model.onnx> <calib_image_dir> [options]

The model may be an Ultralytics '.pt' (exported to ONNX for you) or a '.onnx'
file you bring yourself (export step is skipped). Bringing your own ONNX is how
to convert non-YOLO models until framework-specific export adapters exist.

Options:
  --config FILE    Model profile (YAML). Supplies input dims, hw_arch, and
                   start/end nodes. Optional; without it the YOLO26 defaults apply.
  --pipeline A|B   Pipeline A (raw HEF) or B (NMS-baked HEF). Default: A
  --height H       Model input height in pixels. Overrides the profile.
  --width  W       Model input width in pixels.  Overrides the profile.
  --gpu            Pass --gpus all to Docker (Linux + NVIDIA Container Toolkit only)

Value precedence for pipeline/height/width: CLI flag > --config > built-in default.

Teardown:
  bash automation.sh --clean
                   Remove the Docker image + build cache and delete generated
                   intermediates from shared_data/ (ONNX, calib .npy, HARs, logs,
                   rendered .alls, staged config). KEEPS .hef outputs, source
                   scripts, .alls templates, yaml-config/, and model weights.

Examples:
  bash automation.sh yolo26n.pt images/
  bash automation.sh yolo26n.pt images/ --pipeline B --config model_config.yaml
  bash automation.sh model.onnx images/ --config model_config.yaml
  bash automation.sh yolo11n.pt images/ --height 640 --width 640
  bash automation.sh --clean
EOF
    exit 1
}

# ── --clean: tear down build artifacts and reclaim disk (standalone mode) ──────
if [[ "${1:-}" == "--clean" ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Clean — reclaim disk from the HEF build pipeline"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "This will remove:"
    echo "  • Docker image 'hef-conversion' (~30 GB) and the Docker build cache"
    echo "  • shared_data/ intermediates: *.onnx, *_calib.npy, *.har, *.log,"
    echo "    *.rendered.alls, yolo_nms_config.json, staged model_config.yaml"
    echo "  • repo-root *_calib.npy left by interrupted runs"
    echo "It KEEPS: .hef outputs, *.py scripts, *.alls templates, yaml-config/,"
    echo "and model weights (*.pt / *.pth)."
    echo "Note: the next conversion will rebuild the image cold (~minutes)."
    echo ""
    read -r -p "Continue? [y/N] " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then echo "Aborted — nothing removed."; exit 0; fi

    # 1. Docker image
    if docker image inspect hef-conversion >/dev/null 2>&1; then
        echo "Removing Docker image 'hef-conversion'..."
        docker rmi hef-conversion || echo "  (image in use or already gone — skipped)"
    else
        echo "Docker image 'hef-conversion' not present — skipping."
    fi

    # 2. Docker build cache
    echo "Pruning Docker build cache..."
    docker builder prune -f

    # 3. Generated intermediates in shared_data/ (explicit patterns; sources kept)
    echo "Removing generated intermediates from shared_data/..."
    rm -f shared_data/*.onnx \
          shared_data/*_calib.npy \
          shared_data/*.har \
          shared_data/*.log \
          shared_data/*.rendered.alls \
          shared_data/yolo_nms_config.json \
          shared_data/model_config.yaml

    # 4. Repo-root calib arrays from interrupted runs (before the mv into shared_data)
    rm -f ./*_calib.npy

    echo ""
    echo "Done. Remaining .hef outputs in shared_data/:"
    ls -1 shared_data/*.hef 2>/dev/null || echo "  (none)"
    exit 0
fi

# ── Defaults / CLI sentinels (empty = "not passed on the CLI") ─────────────────
PIPELINE_CLI=""
HEIGHT_CLI=""
WIDTH_CLI=""
USE_GPU=false
CONFIG=""

# ── Required positionals ──────────────────────────────────────────────────────
if [ $# -lt 2 ]; then usage; fi
MODEL_INPUT="$1"
CALIB_DIR="$2"
shift 2

# ── Optional flags ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";       shift 2 ;;
        --pipeline) PIPELINE_CLI="$2"; shift 2 ;;
        --height)   HEIGHT_CLI="$2";   shift 2 ;;
        --width)    WIDTH_CLI="$2";    shift 2 ;;
        --gpu)      USE_GPU=true;      shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────────────
if [ ! -f "$MODEL_INPUT" ]; then echo "ERROR: Model file not found: $MODEL_INPUT"; exit 1; fi
if [ ! -d "$CALIB_DIR" ];   then echo "ERROR: Calibration directory not found: $CALIB_DIR"; exit 1; fi
if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then echo "ERROR: Config file not found: $CONFIG"; exit 1; fi

# ── Read profile values on the host (empty when no config / key missing) ───────
H_CFG=""
W_CFG=""
P_CFG=""
if [ -n "$CONFIG" ]; then
    H_CFG=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')) or {}; print((c.get('input') or {}).get('height',''))")
    W_CFG=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')) or {}; print((c.get('input') or {}).get('width',''))")
    P_CFG=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')) or {}; print(c.get('pipeline','') or '')")
fi

# Resolve with precedence: CLI flag > config > built-in default.
PIPELINE="${PIPELINE_CLI:-${P_CFG:-A}}"
HEIGHT="${HEIGHT_CLI:-${H_CFG:-416}}"
WIDTH="${WIDTH_CLI:-${W_CFG:-608}}"

PIPELINE=$(echo "$PIPELINE" | tr '[:lower:]' '[:upper:]')
if [[ "$PIPELINE" != "A" && "$PIPELINE" != "B" ]]; then
    echo "ERROR: pipeline must be A or B (got '$PIPELINE')"; exit 1
fi

# ── Resolve model naming / input format ───────────────────────────────────────
MODEL_BASE=$(basename "$MODEL_INPUT")
MODEL_EXT=$(echo "${MODEL_BASE##*.}" | tr '[:upper:]' '[:lower:]')
MODEL_STEM="${MODEL_BASE%.*}"
MODEL_DIR=$(cd "$(dirname "$MODEL_INPUT")" && pwd)

case "$MODEL_EXT" in
    pt|onnx) ;;
    pth) echo "ERROR: generic '.pth' export is not supported yet. Export it to ONNX"
         echo "       first (with your model's own script), then pass the '.onnx' here."; exit 1 ;;
    *) echo "ERROR: unsupported input '.$MODEL_EXT'. Use a '.pt' or '.onnx' file."; exit 1 ;;
esac

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
[ -n "$CONFIG" ] && echo "  Profile: ${CONFIG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── [1/4] Obtain the ONNX (export from .pt, or use the provided .onnx) ─────────
echo ""
if [ "$MODEL_EXT" = "onnx" ]; then
    echo "[1/4] Using provided ONNX (skipping export)..."
    ONNX_SRC="${MODEL_DIR}/${MODEL_BASE}"
else
    echo "[1/4] Exporting and simplifying ONNX..."
    python main.py "$MODEL_INPUT" "$HEIGHT" "$WIDTH"
    ONNX_SRC="${MODEL_DIR}/${MODEL_STEM}.onnx"
fi

if [ ! -f "$ONNX_SRC" ]; then
    echo "ERROR: Expected ONNX at ${ONNX_SRC} — not found."
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

# Stage the profile so the in-container conversion script can read it. Passed as
# an extra positional arg (empty string when no --config was given).
CONFIG_ARG=""
if [ -n "$CONFIG" ]; then
    cp "$CONFIG" "shared_data/model_config.yaml"
    CONFIG_ARG="model_config.yaml"
    echo "      OK: shared_data/model_config.yaml"
fi

# ── [4/4] Docker conversion ───────────────────────────────────────────────────
echo ""
echo "[4/4] Running Docker conversion (this may take a while)..."

DOCKER_ARGS=(--rm -v "$(pwd)/shared_data:/app/shared_data")
if [ "$USE_GPU" = true ]; then
    DOCKER_ARGS+=(--gpus all)
    echo "      GPU passthrough enabled"
fi

docker run "${DOCKER_ARGS[@]}" hef-conversion \
    bash -c "cd /app/shared_data && python3 ${SCRIPT} ${MODEL_STEM} ${MODEL_STEM}.onnx ${CALIB_FILE} ${CONFIG_ARG}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done! Output: shared_data/${HEF_OUTPUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
