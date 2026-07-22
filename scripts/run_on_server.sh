#!/usr/bin/env bash
# SignLink CSLT pipeline — run on the remote Linux GPU server.
# Usage: bash run_on_server.sh [--skip-preprocess] [--epochs 50]

set -euo pipefail

PROJECT_ROOT="/workspace/SignLink"
PREPROCESS_DIR="${PROJECT_ROOT}/preprocessing"
TRAINING_DIR="${PROJECT_ROOT}/training"
NPZ_DIR="${PROJECT_ROOT}/preprocessed"
MODELS_DIR="${PROJECT_ROOT}/models"
LOGS_DIR="${PROJECT_ROOT}/logs"
VIDEO_ROOT="/workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/Videos_Sentence_Level"
CSV_PATH="/workspace/datasets/ISL_CSLRT/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv"

SKIP_PREPROCESS=0
EPOCHS=50
BATCH_SIZE=16
WORKERS=16

for arg in "$@"; do
  case "$arg" in
    --skip-preprocess) SKIP_PREPROCESS=1 ;;
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
    --workers=*) WORKERS="${arg#*=}" ;;
  esac
done

echo "============================================================"
echo "SignLink CSLT Pipeline"
echo "============================================================"

echo ""
echo "[1/6] Verifying CUDA and GPU..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi

echo ""
echo "[2/6] Verifying preprocessing scripts..."
cd "${PREPROCESS_DIR}"
python -m py_compile mediapipe_utils.py save_npz.py extract_landmarks.py
echo "Preprocessing syntax: OK"

echo ""
echo "[3/6] Verifying training scripts..."
cd "${TRAINING_DIR}"
python -m py_compile dataset.py dataloader.py model.py train.py
echo "Training syntax: OK"

NPZ_COUNT=$(find "${NPZ_DIR}" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
VIDEO_COUNT=$(find "${VIDEO_ROOT}" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "NPZ files: ${NPZ_COUNT} | Videos discovered: ${VIDEO_COUNT}"

if [[ "${SKIP_PREPROCESS}" -eq 0 && "${NPZ_COUNT}" -lt "${VIDEO_COUNT}" ]]; then
  echo ""
  echo "[4/6] Running preprocessing (resume enabled)..."
  cd "${PREPROCESS_DIR}"
  python extract_landmarks.py \
    --video-root "${VIDEO_ROOT}" \
    --csv-path "${CSV_PATH}" \
    --output-dir "${NPZ_DIR}" \
    --workers "${WORKERS}"
else
  echo ""
  echo "[4/6] Skipping preprocessing (NPZ count sufficient or --skip-preprocess set)."
fi

NPZ_COUNT=$(find "${NPZ_DIR}" -maxdepth 1 -name '*.npz' -type f 2>/dev/null | wc -l | tr -d ' ')
if [[ "${NPZ_COUNT}" -eq 0 ]]; then
  echo "ERROR: No NPZ files found in ${NPZ_DIR}. Preprocessing failed or dataset missing."
  exit 1
fi

echo ""
echo "[5/6] Verifying a sample NPZ..."
python - <<'PY'
import numpy as np
from pathlib import Path

npz_dir = Path("/workspace/SignLink/preprocessed")
sample = next(sorted(npz_dir.glob("*.npz")))
with np.load(sample, allow_pickle=False) as data:
    print("Sample:", sample.name)
    print("  keys:", data.files)
    print("  landmarks shape:", data["landmarks"].shape, "dtype:", data["landmarks"].dtype)
    print("  sentence:", str(data["sentence"])[:80])
    print("  gloss:", str(data["gloss"])[:80])
    print("  fps:", float(data["fps"]))
    print("  frame_count:", int(data["frame_count"]))
    print("  feature_dim:", int(data["feature_dim"]))
PY

echo ""
echo "[6/6] Starting Transformer training on GPU..."
mkdir -p "${MODELS_DIR}" "${LOGS_DIR}"
cd "${TRAINING_DIR}"
python train.py \
  --npz-dir "${NPZ_DIR}" \
  --save-dir "${MODELS_DIR}" \
  --log-dir "${LOGS_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers 4

echo ""
echo "Done. Checkpoints: ${MODELS_DIR} | Logs: ${LOGS_DIR}/train.log"
