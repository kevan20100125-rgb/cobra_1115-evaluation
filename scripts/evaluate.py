"""
evaluate.py

Entry point for all VLM-Evaluation evaluations; specify model and dataset, get results.

Run with `accelerate` from repository root (for naive parallelization):
    =>> [Single-GPU] CUDA_VISIBLE_DEVICES={0-7} accelerate launch --num_processes=1 scripts/evaluate.py < args >
    =>> [Multi-GPU]  accelerate launch --num_processes={>1} scripts/evaluate.py < args >

Design note (Cobra PTQ integration):
    - This script is intentionally agnostic to float vs. quantized models.
    - `vlm_eval.models.load_vlm` is responsible for routing:
        * FLOAT Cobra  → CobraVLM
        * PTQ / Fake-Quant Cobra (e.g., "cobra+3b-ptq-w8a8-fake") → QuantizedCobraVLM
    - No direct import or special handling of QuantizedCobraVLM is needed here.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import draccus
from accelerate.utils import set_seed

from vlm_eval.conf import DatasetConfig, DatasetRegistry
from vlm_eval.models import load_vlm
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import get_task_runner

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch => Wraps `logging.Logger` and (internally) accelerate.PartialState
overwatch = initialize_overwatch(__name__)


@dataclass
class EvaluationConfig:
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with:
    #   --dataset.type DatasetRegistry.<DATASET>.dataset_id
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # === Model Parameters (Cobra default) ===
    # model_family is dispatched via `vlm_eval/models/__init__.py`:
    #   * "cobra"       → CobraVLM (float)
    #   * "cobra-quant" → QuantizedCobraVLM (explicit quant family)
    #   In addition, any model_id containing "-ptq-" / "fake" / "w4a4"/"w8a8" etc.
    #   will be routed to QuantizedCobraVLM automatically.
    model_family: str = "cobra"
    model_id: Optional[str] = "cobra+3b"
    model_dir: Optional[Path] = None  # Path to model checkpoint root (or repo root for Cobra PTQ)

    # Example overrides for other families:
    # --- Official LLaVa ---
    # model_family: str = "llava-v15"
    # model_id: str = "llava-v1.5-7b"
    # model_dir: Path = Path("liuhaotian/llava-v1.5-7b")
    #
    # --- Official InstructBLIP ---
    # model_family: str = "instruct-blip"
    # model_id: str = "instructblip-vicuna-7b"
    # model_dir: Path = Path("Salesforce/instructblip-vicuna-7b")

    # Inference Parameters
    device_batch_size: int = 1   # Keep 1 until all backends are verified for larger batches
    num_workers: int = 2         # Number of DataLoader workers (per process)

    # Artifact Parameters
    results_dir: Path = Path("results")

    # HF Hub Credentials (for gated backbones)
    hf_token: Union[str, Path] = Path(".hf_token")  # Env var name or path to token file

    # Randomness
    seed: int = 21

    def __post_init__(self) -> None:
        # For historical reasons `run_dir` is used by loaders; we map model_dir -> run_dir.
        self.run_dir = self.model_dir

    # fmt: on


@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    overwatch.info(
        "Starting Evaluation for Dataset `%s` w/ Model `%s` (family=%s)",
        cfg.dataset.dataset_id,
        cfg.model_id,
        cfg.model_family,
    )
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Short-circuit if metrics already exist
    # ------------------------------------------------------------------
    task_results_dir = (
        cfg.results_dir / cfg.dataset.dataset_family / cfg.dataset.dataset_id / cfg.model_id
    )

    metrics_path = task_results_dir / "metrics.json"
    if metrics_path.exists():
        overwatch.info(
            "Metrics for dataset=`%s` model=`%s` already exist at `%s` => exiting.",
            cfg.dataset.dataset_id,
            cfg.model_id,
            str(metrics_path),
        )
        return

    # ------------------------------------------------------------------
    # Build the VLM (float or quant) via unified loader
    # ------------------------------------------------------------------
    overwatch.info("Initializing VLM => Bundling model, image processor, and tokenizer")

    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = os.environ[cfg.hf_token]

    vlm = load_vlm(
        cfg.model_family,
        cfg.model_id,
        cfg.run_dir,
        hf_token=hf_token,
        ocr=cfg.dataset.ocr,
    )

    # ------------------------------------------------------------------
    # Build Task Runner
    # ------------------------------------------------------------------
    overwatch.info(
        "Building Evaluation Runner for Dataset `%s` (family=%s)",
        cfg.dataset.dataset_id,
        cfg.dataset.dataset_family,
    )

    task_runner = get_task_runner(
        cfg.dataset.dataset_family,
        cfg.dataset.root_dir,
        cfg.dataset.index_file,
        task_results_dir,
        cfg.model_id,
        prompt_fn=vlm.get_prompt_fn(cfg.dataset.dataset_family),
        image_processor=vlm.image_processor,
    )

    # ------------------------------------------------------------------
    # Run Evaluation
    # ------------------------------------------------------------------
    overwatch.info("Starting (Distributed) Evaluation Loop")
    task_runner.evaluate(vlm, cfg.device_batch_size, cfg.num_workers)


if __name__ == "__main__":
    evaluate()
    print("Evaluation Complete!")
