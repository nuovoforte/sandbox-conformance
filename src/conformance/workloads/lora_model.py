"""Model and tokenizer loading, isolated so training and evaluation agree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lora_config import RunConfig


def _quant_config() -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(cfg: RunConfig) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left padding is required for correct batched generation; right padding
    # silently produces garbage completions and is a classic time sink.
    tok.padding_side = "left"
    return tok


def load_model(cfg: RunConfig, adapter_path: str | None = None) -> Any:
    """Load the base model, optionally with a trained LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if cfg.device == "cuda" else torch.float32,
        "device_map": "auto" if cfg.device == "cuda" else None,
    }
    if cfg.load_in_4bit and cfg.device == "cuda":
        kwargs["quantization_config"] = _quant_config()

    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    except TypeError:
        # transformers < 4.56 named this argument torch_dtype.
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model


def lora_config(cfg: RunConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
