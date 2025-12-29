from pathlib import Path
from typing import Optional

from vlm_eval.util.interfaces import VLM

from .cobra import CobraVLM
from .quant_cobra import QuantizedCobraVLM  # Quantized / Fake-Quant Cobra

from vlm_eval.overwatch import initialize_overwatch

# Initialize logger
overwatch = initialize_overwatch(__name__)


# ---------------------------------------------------------------------
# Float Model Dispatch
# ---------------------------------------------------------------------
FAMILY2INITIALIZER = {
    "cobra": CobraVLM,
    "cobra-float": CobraVLM,   # alias for explicit float
}


# ---------------------------------------------------------------------
# Quant Model Dispatch (explicit)
# ---------------------------------------------------------------------
QUANT_FAMILY2INITIALIZER = {
    "cobra-quant": QuantizedCobraVLM,
}


# ---------------------------------------------------------------------
# Additional model_id keyword triggers
# ---------------------------------------------------------------------
QUANT_MODEL_ID_KEYWORDS = [
    "quant",
    "ptq",
    "fake",
    "int",
    # bit patterns
    "w8a8", "w4a4", "w4a8", "w8a4",
    "w2a2", "w2a4", "w2a8", "w2a16",
    "w4a2", "w8a2",
    "w16a2", "w16a4", "w16a8", "w16a16",
]


def _is_quant_model(model_family: str, model_id: str) -> bool:
    """
    Unified detection rule for PTQ models.

    Conditions that trigger QuantizedCobraVLM:
        1) Explicit quant family: model_family == 'cobra-quant'
        2) model_id contains '-ptq-'
        3) model_id ends with '-fake'
        4) model_id ends with '-int'
        5) model_id contains any legacy bit keyword (w4a4, w8a8, ...)
    """
    family_lower = model_family.lower()
    mid = model_id.lower()

    # 1) explicit quant family
    if family_lower in QUANT_FAMILY2INITIALIZER:
        return True

    # 2) standard quant flag
    if "-ptq-" in mid:
        return True

    # 3–4) backend flag for fake / int
    if mid.endswith("-fake") or mid.endswith("-int"):
        return True

    # 5) bit keywords
    return any(k in mid for k in QUANT_MODEL_ID_KEYWORDS)


# ---------------------------------------------------------------------
# load_vlm — unified entry point
# ---------------------------------------------------------------------
def load_vlm(
    model_family: str,
    model_id: str,
    run_dir: Optional[Path],
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length: int = 128,
    temperature: float = 1.0,
) -> VLM:

    # ===== Quantized Routing =====
    if _is_quant_model(model_family, model_id):
        overwatch.info(
            "[load_vlm] Routing to QuantizedCobraVLM | family=%s model_id=%s",
            model_family,
            model_id,
        )
        return QuantizedCobraVLM(
            model_family="cobra",      # always treat quant as cobra backbone
            model_id=model_id,
            run_dir=run_dir,
            hf_token=hf_token,
            load_precision=load_precision,
            max_length=max_length,
            temperature=temperature,
            ocr=ocr,
        )

    # ===== Float Routing =====
    assert (
        model_family in FAMILY2INITIALIZER
    ), f"Model family `{model_family}` not supported!"

    overwatch.info(
        "[load_vlm] Routing FLOAT model | family=%s model_id=%s",
        model_family,
        model_id,
    )

    return FAMILY2INITIALIZER[model_family](
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )
