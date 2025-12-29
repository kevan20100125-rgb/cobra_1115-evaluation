"""
quant_cobra.py — Quantized Cobra Adapter (Fake-Quant Backend Only for Accuracy Study)

本檔負責在 vlm-evaluation 中對接「已完成 PTQ 的 Cobra」：

Design（對應目前 W4/W8 Fake Quant Accuracy Study）：
    - 僅啟用「Fake Quant Backend」：
        * 使用 cobra_1115 的 wrap_model_for_quantization + calibrator 流程，
          所有 kernel 仍是 PyTorch float kernel，前後包 fake quant。
    - 量化路徑集中在 cobra.quantize.runtime.load_quantized_vlm.load_quantized_cobra_vlm。

支援的 model_id 形式（僅針對 quant case）：
    - "cobra+3b-ptq-w8a8-fake"
    - "cobra+3b-ptq-w4a4-fake"

約定：
    - base_id 例如 "cobra+3b" 由 cobra.load(...) 負責載入 float checkpoint。
    - run_dir 指向 cobra_1115 repo root（預設 /work/asdf1234/cobra_1115）：
        * pct_hi_lo 路徑：  run_dir/outputs/quantize/pct_hi_lo_W8A8.pt
"""

from pathlib import Path
from typing import Any, Optional, Tuple
import os
import re

import torch
import torch.nn as nn
from PIL.Image import Image  # noqa: F401  (kept for interface parity)

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.util.interfaces import ImageProcessor, Tokenizer

from .cobra import CobraVLM
from cobra.quantize.runtime.load_quantized_vlm import load_quantized_cobra_vlm

overwatch = initialize_overwatch(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_model_id_backend(model_id: str) -> Tuple[str, Optional[str], str]:
    """
    回傳：
        base_id: "cobra+3b"
        bits:    "W8A8" / "W4A4" / None
        backend: "fake" | "float"

    支援格式：
        cobra+3b
        cobra+3b-ptq-w8a8
        cobra+3b-ptq-w8a8-fake

    說明：
        - 若不含 "-ptq-"，視為 float base_id，bits=None。
        - backend 預設為 "float"，但目前 accuracy study 僅實作 "fake" backend；
          "int" 會在 runtime 顯示明確錯誤。
    """
    lower = model_id.lower()

    backend = "float"  # 預設 backend = float；目前僅 fake 有實作
    base_for_bits = model_id

    # Step 1: parse backend flag
    if lower.endswith("-fake"):
        backend = "fake"
        base_for_bits = model_id[:-5]  # remove "-fake"

    # Step 2: extract bits
    lower2 = base_for_bits.lower()
    if "-ptq-" not in lower2:
        # 無 quant 標記 → 視為 float
        return base_for_bits, None, "float"

    # 保留原始大小寫，只用 lower2 尋找 split 位置
    idx = lower2.index("-ptq-")
    base = base_for_bits[:idx]
    suffix = base_for_bits[idx + len("-ptq-") :]  # e.g. "w8a8"
    bits = suffix.upper()  # e.g. "W8A8"

    return base, bits, backend

def _validate_bits(bits: Optional[str]) -> Optional[str]:
    """
    bits="W8A8" → return "W8A8"
    bits=None  → return None（float fallback）
    """
    if bits is None:
        return None

    m = re.fullmatch(r"[Ww](\d+)[Aa](\d+)", bits.strip())
    if not m:
        raise ValueError(
            f"Invalid bits spec {bits!r}; expected W8A8/W4A4/W2A2/W16A16 etc."
        )

    w_bits = int(m.group(1))
    a_bits = int(m.group(2))
    valid_bits = (2, 4, 8, 16)

    if w_bits not in valid_bits or a_bits not in valid_bits:
        raise ValueError(
            f"Unsupported bitwidth W{w_bits}A{a_bits}; must be in {valid_bits}."
        )
    return f"W{w_bits}A{a_bits}"

def _resolve_pct_hi_lo_path(run_dir: Path, bits: str) -> Path:
    """
    Resolve pct_hi_lo path robustly.

    Priority:
      1) outputs/quantize/pct_hi_lo_{bits}.pt  (new convention in vlm-eval)
      2) outputs/quantize/pct_hi_lo.pt         (cobra default)
    """
    cand1 = run_dir / "outputs" / "quantize" / f"pct_hi_lo_{bits}.pt"
    if cand1.is_file():
        return cand1

    cand2 = run_dir / "outputs" / "quantize" / "pct_hi_lo.pt"
    if cand2.is_file():
        return cand2

    raise FileNotFoundError(
        f"[QuantizedCobraVLM] Cannot find pct_hi_lo file. Tried:\n"
        f"  - {str(cand1)}\n"
        f"  - {str(cand2)}"
    )

# -----------------------------------------------------------------------------
# QuantizedCobraVLM
# -----------------------------------------------------------------------------

class QuantizedCobraVLM(CobraVLM):
    """
    Quantized Cobra VLM adapter（目前僅 fake-quant backend 實作）。

    行為：
        - 解析 model_id → base_model_id / bits / backend。
        - 若 bits=None 或 backend="float" → fallback 到 FLOAT（等同 CobraVLM）。
        - 若 backend="fake" 且 bits!=None →
            使用 cobra.quantize.runtime.load_quantized_vlm.load_quantized_cobra_vlm
            建立「wrap + calibrate」過後的 fake-quant 模型。
    """

    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Optional[Path],
        hf_token: str,
        *,
        load_precision: str = "bf16",
        ocr: bool = False,
        max_length: int = 128,
        temperature: float = 1.0,
        quant_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        self.raw_model_id = model_id

        # Parse: base_id / bits / backend
        base_id, parsed_bits, backend = _parse_model_id_backend(model_id)
        validated_bits = _validate_bits(parsed_bits)

        self.base_model_id = base_id          # e.g. "cobra+3b"
        self.bits = validated_bits            # "W8A8" / "W4A4" / None
        self.quant_backend = backend          # "fake" / "float"

        # quant_mode 只在有 bits 時代表真實量化模式；否則可以標示為 "float"
        if validated_bits is not None:
            self.quant_mode = validated_bits
        else:
            self.quant_mode = quant_mode or "float"

        overwatch.info(
            "QuantizedCobraVLM | raw_id=%s | base_id=%s | bits=%s | backend=%s | run_dir=%s",
            model_id,
            self.base_model_id,
            self.bits,
            self.quant_backend,
            str(run_dir) if run_dir else "<none>",
        )

        # 呼叫 FLOAT 版本的 CobraVLM.__init__，但 load() 已被 override 成 quant aware
        super().__init__(
            model_family=model_family,
            model_id=self.base_model_id,
            run_dir=run_dir,
            hf_token=hf_token,
            load_precision=load_precision,
            ocr=ocr,
            max_length=max_length,
            temperature=temperature,
            **kwargs,
        )

        self._post_load_quant_sanity_check()

    # -------------------------------------------------------------------------
    # Load (override parent)
    # -------------------------------------------------------------------------

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Load base model，然後依 quant_backend / bits 決定是否套用 fake quant。

        Case:
            - bits is None or backend=="float":
                → 等同 CobraVLM：直接走 super().load()（FLOAT pipeline）。
            - backend=="fake" 且 bits!=None:
                → 使用 load_quantized_cobra_vlm（wrap + calibrate，fake quant）。
        """
        base_id = self.base_model_id
        bits = self.bits

        # 分支 1：沒有 quant bits 或顯式 float backend → 視為 FLOAT
        if bits is None or self.quant_backend == "float":
            overwatch.info(
                "[QuantizedCobraVLM] bits=None or backend=float → delegating to FLOAT CobraVLM.load()"
            )
            # 完全沿用原本 CobraVLM 的行為，避免手動重寫 float loading 流程。
            return super().load()

        # 分支 3：Fake backend（目前主要路徑）
        if self.quant_backend != "fake":
            raise ValueError(f"[QuantizedCobraVLM] Unknown backend: {self.quant_backend}")

        # 確保 runtime loader 真的使用 FAKE backend，而不是被外部殘留的
        # BACKEND=float 之類的設定干擾。
        os.environ["BACKEND"] = "fake"

        # run_dir 用於定位 pct_hi_lo 檔案
        if self.run_dir is None:
            default_dir = Path("/work/asdf1234/cobra_1115")
            overwatch.warning(
                "QuantizedCobraVLM: run_dir=None; using default=%s",
                str(default_dir),
            )
            self.run_dir = default_dir

        run_dir = Path(self.run_dir)

        # pct_hi_lo 路徑
        pct_hi_lo_path = _resolve_pct_hi_lo_path(run_dir, bits)

        # 確保 COBRA_MODEL_ID_OR_PATH 指向 float base id（給 cobra.load 使用）
        os.environ.setdefault("COBRA_MODEL_ID_OR_PATH", base_id)

        overwatch.info(
            "[QuantizedCobraVLM] Using fake-quant backend: base_id=%s, bits=%s, pct_hi_lo=%s",
            base_id,
            bits,
            str(pct_hi_lo_path),
        )

        # 呼叫 cobra_1115 runtime loader（wrap + calibrate，fake quant）
        vlm = load_quantized_cobra_vlm(
            bits=bits,
            pct_hi_lo_path=pct_hi_lo_path,
            hf_token=self.hf_token,
            base_dtype=self.dtype,
            device=self.distributed_state.device,
        )

        tokenizer = vlm.llm_backbone.tokenizer
        img_proc = vlm.vision_backbone.image_transform
        return vlm, tokenizer, img_proc

    # -------------------------------------------------------------------------
    # Sanity
    # -------------------------------------------------------------------------

    def _post_load_quant_sanity_check(self) -> None:
        overwatch.debug(
            "[post_load] raw_id=%s | base_id=%s | bits=%s | backend=%s | quant_mode=%s",
            self.raw_model_id,
            self.base_model_id,
            self.bits,
            self.quant_backend,
            self.quant_mode,
        )
