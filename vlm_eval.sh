#!/bin/bash 
#SBATCH --job-name=vlm_eval
#SBATCH --account=MST114205
#SBATCH --partition=normal2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

# ==============================
# 1. 環境設定（沿用 cobra_1115 設定）
# ==============================
module load cuda/12.4

# conda activate under nounset-safe wrapper
set +u
source /work/asdf1234/miniconda3/etc/profile.d/conda.sh
conda activate cobra
set -u

# 避免 nounset 打壞某些 script
export ADDR2LINE=${ADDR2LINE:-$(command -v addr2line || true)}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ===== Quantized Cobra 必備 =====
# cobra_1115 專案位置
export COBRA_1115_ROOT="${COBRA_1115_ROOT:-/work/asdf1234/cobra_1115}"
export PYTHONPATH="${COBRA_1115_ROOT}:${PYTHONPATH:-}"

# base float model（給 cobra.load(base_id) 用）
export COBRA_MODEL_BASE_ID="${COBRA_MODEL_BASE_ID:-cobra+3b}"

# BITS：控制 W_bits / A_bits，會灌進 MODEL_ID
# 例：BITS=W8A8 → "cobra+3b-ptq-w8a8-fake"
export BITS="${BITS:-W8A8}"

# BACKEND：
#   float = 不量化，直接用 CobraVLM（MODEL_ID=cobra+3b）
#   fake  = fake-quant backend（low-bit semantics simulated in float）
export BACKEND="${BACKEND:-fake}"

# -----------------------------
# Preflight checks (fail-fast)
# -----------------------------
if [[ "${BACKEND}" == "fake" ]]; then
  PCT_HI_LO_PATH="${COBRA_1115_ROOT}/outputs/quantize/pct_hi_lo_${BITS}.pt"
  if [[ ! -f "${PCT_HI_LO_PATH}" ]]; then
    echo "[ERROR] Missing pct_hi_lo file for BITS=${BITS}: ${PCT_HI_LO_PATH}"
    echo "Run cobra_1115_ptq.sh first with the same BITS, e.g.:"
    echo "BITS=${BITS} SMOKE=0 MODE=calibrate ./cobra_1115_ptq.sh"
    exit 2
  fi
fi
# ==============================
# ROTATION_MODE：控制 LLM output projector 的旋轉模式
#   hk       -> KLT + Hadamard
#   hadamard -> 只有 Hadamard
#   none     -> 完全不旋轉
export ROTATION_MODE="${ROTATION_MODE:-hk}"

case "${ROTATION_MODE}" in
  hk|hadamard|none)
    ;;
  *)
    echo "[WARN] Unknown ROTATION_MODE='${ROTATION_MODE}', falling back to 'hk'." >&2
    ROTATION_MODE="hk"
    export ROTATION_MODE
    ;;
esac

# 統一給 cobra_1115 runtime 使用的 env key
export COBRA_PROJECTOR_ROTATION_MODE="${ROTATION_MODE}"

# 僅允許 float/fake；int 在此 repo 與 cobra_1115 refactor 中皆不支援
BACKEND_LOWER="$(echo "${BACKEND}" | tr 'A-Z' 'a-z')"
case "${BACKEND_LOWER}" in
  float|fake)
    ;;
  *)
    echo "[ERROR] Unsupported BACKEND='${BACKEND}'. Only BACKEND=float|fake is supported." >&2
    exit 1
    ;;
esac

# ==============================
# 2. 專案位置
# ==============================

VLM_ROOT="${VLM_ROOT:-/work/asdf1234/vlm-evaluation}"
cd "${VLM_ROOT}"

mkdir -p outputs/slurm
mkdir -p results

# ==============================
# 3. 控制參數
# ==============================

# MODE:
#   prepare  = 只下載 / 準備 dataset
#   evaluate = 只跑 scripts/evaluate.py
#   score    = 只跑 scripts/score.py
#   full     = 依序 prepare + evaluate + score
MODE="${MODE:-evaluate}"

DATA_ROOT="${DATA_ROOT:-/work/asdf1234/datasets}"
# Dataset family（給 scripts/datasets/prepare.py 用）
DATASET_FAMILY="${DATASET_FAMILY:-text-vqa}"

# DatasetRegistry 的 enum key（給 evaluate/score 用）
DATASET_KEY="${DATASET_KEY:-TEXTVQA_SLIM}"

# Model family（目前只用 cobra）
MODEL_FAMILY="${MODEL_FAMILY:-cobra}"

# 根據 BACKEND / BITS 產生 MODEL_ID
#   float: cobra+3b
#   fake : cobra+3b-ptq-w8a8-fake
BITS_LOWER="$(echo "${BITS}" | tr 'A-Z' 'a-z')"

case "${BACKEND_LOWER}" in
  float)
    MODEL_ID="${MODEL_ID:-${COBRA_MODEL_BASE_ID}}"
    ;;
  fake)
    MODEL_ID="${MODEL_ID:-${COBRA_MODEL_BASE_ID}-ptq-${BITS_LOWER}-${BACKEND_LOWER}}"
    ;;
esac

# run_dir / model_dir:
#   - fake：給 QuantizedCobraVLM 用來找 outputs/quantize/*（pct_hi_lo）
#   - float：不需要，讓 cobra.load() 直接用 HF model_id
case "${BACKEND_LOWER}" in
  float)
    MODEL_DIR=""
    ;;
  fake)
    MODEL_DIR="${MODEL_DIR:-${COBRA_1115_ROOT}}"
    ;;
esac

# 結果輸出根目錄
RESULTS_DIR="${RESULTS_DIR:-results}"

# HF token 文字檔位置（內容為一行 HF access token）
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${VLM_ROOT}/.hf_token}"

COBRA_FUSION_ROTATION_MODE="${COBRA_FUSION_ROTATION_MODE:-none}"
COBRA_FUSION_ROTATION_ABSORB="${COBRA_FUSION_ROTATION_ABSORB:-0}"
COBRA_DEBUG_FLOW="${COBRA_DEBUG_FLOW:-0}"

export MODE DATA_ROOT DATASET_FAMILY DATASET_KEY
export MODEL_FAMILY MODEL_ID MODEL_DIR
export RESULTS_DIR HF_TOKEN_PATH
export BITS BACKEND COBRA_MODEL_BASE_ID COBRA_1115_ROOT
export COBRA_FUSION_ROTATION_MODE=hk
export COBRA_FUSION_ROTATION_ABSORB=1
export COBRA_DEBUG_FLOW

echo "[INFO] MODE=${MODE}"
echo "[INFO] DATASET_FAMILY=${DATASET_FAMILY}, DATASET_KEY=${DATASET_KEY}"
echo "[INFO] MODEL_FAMILY=${MODEL_FAMILY}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] MODEL_DIR(run_dir)=${MODEL_DIR}"
echo "[INFO] RESULTS_DIR=${RESULTS_DIR}"
echo "[INFO] HF_TOKEN_PATH=${HF_TOKEN_PATH}"
echo "[INFO] VLM_ROOT=${VLM_ROOT}"
echo "[INFO] COBRA_1115_ROOT=${COBRA_1115_ROOT}"
echo "[INFO] BITS=${BITS}, BACKEND=${BACKEND}, ROTATION_MODE=${ROTATION_MODE}"

# ==============================
# 4. Dataset 準備（scripts/datasets/prepare.py）
# ==============================
if [[ "${MODE}" == "prepare" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation dataset prepare (family=${DATASET_FAMILY})..."

  python - << 'PY'
import os
from pathlib import Path

from scripts.datasets.prepare import DatasetPreparationConfig, prepare

dataset_family = os.environ.get("DATASET_FAMILY", "vqa-v2")
data_root = Path(os.environ.get("DATA_ROOT", "/work/asdf1234/datasets"))
hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))

# 防呆：token 檔必須存在
if not hf_token_path.is_file():
    raise SystemExit(
        f"[ERROR] HF token file not found: {hf_token_path}\n"
        f"        Please create it (one-line token) or set HF_TOKEN_PATH correctly."
    )

cfg = DatasetPreparationConfig(
    dataset_family=dataset_family,
    root_dir=data_root,
    hf_token=hf_token_path,
)

prepare(cfg)
PY

  echo "[STEP] Dataset prepare finished for family=${DATASET_FAMILY}"
fi

# ==============================
# 5. Evaluate（scripts/evaluate.py）
# ==============================
if [[ "${MODE}" == "evaluate" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation evaluate.py ..."

  python - << 'PY'
import os
from pathlib import Path

from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry
from scripts.evaluate import EvaluationConfig, evaluate

dataset_key = os.environ.get("DATASET_KEY", "VQAv2_FULL")
data_root = Path(os.environ.get("DATA_ROOT", "/work/asdf1234/datasets"))

model_family = os.environ.get("MODEL_FAMILY", "cobra")
model_id = os.environ.get("MODEL_ID", "cobra+3b")
model_dir_str = os.environ.get("MODEL_DIR", "").strip()
results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))

# 這裡不要讀檔，直接把 Path 丟給 EvaluationConfig
hf_token = hf_token_path

# 1) 建立 DatasetConfig
try:
    ds_enum = DatasetRegistry[dataset_key]
except KeyError as e:
    raise SystemExit(
        f"[ERROR] Unknown DATASET_KEY={dataset_key!r}. "
        f"請確認 vlm_eval/conf/datasets.py 裡的 DatasetRegistry。"
    ) from e

ds_cls = DatasetConfig.get_choice_class(ds_enum.dataset_id)
dataset_cfg = ds_cls()
dataset_cfg.root_dir = data_root

# 2) 建立 EvaluationConfig
model_dir = Path(model_dir_str) if model_dir_str else None

cfg = EvaluationConfig(
    dataset=dataset_cfg,
    model_family=model_family,
    model_id=model_id,
    model_dir=model_dir,
    results_dir=results_dir,
    hf_token=hf_token,  # 這裡是 Path，不是字串 token
)

evaluate(cfg)
PY

  echo "[STEP] Evaluate finished for MODEL_FAMILY=${MODEL_FAMILY}, MODEL_ID=${MODEL_ID}, DATASET_KEY=${DATASET_KEY}"
fi

# ==============================
# 6. Score（scripts/score.py）
# ==============================
if [[ "${MODE}" == "score" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running vlm-evaluation score.py ..."

  python - << 'PY'
import os
from pathlib import Path

from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry
from scripts.score import ScoreConfig, score

dataset_key = os.environ.get("DATASET_KEY", "VQAv2_FULL")
data_root = Path(os.environ.get("DATA_ROOT", "/work/asdf1234/datasets"))
model_id = os.environ.get("MODEL_ID", "cobra+3b")
results_dir = Path(os.environ.get("RESULTS_DIR", "results"))

try:
    ds_enum = DatasetRegistry[dataset_key]
except KeyError as e:
    raise SystemExit(
        f"[ERROR] Unknown DATASET_KEY={dataset_key!r}. "
        f"請確認 vlm_eval/conf/datasets.py 裡的 DatasetRegistry。"
    ) from e

ds_cls = DatasetConfig.get_choice_class(ds_enum.dataset_id)
dataset_cfg = ds_cls()
dataset_cfg.root_dir = data_root

cfg = ScoreConfig(
    dataset=dataset_cfg,
    model_id=model_id,
    results_dir=results_dir,
)

score(cfg)
PY

  echo "[STEP] Score finished for MODEL_ID=${MODEL_ID}, DATASET_KEY=${DATASET_KEY}"
fi

echo "[DONE] vlm-evaluation pipeline complete (MODE=${MODE}, MODEL_ID=${MODEL_ID}, BACKEND=${BACKEND}, BITS=${BITS})."

