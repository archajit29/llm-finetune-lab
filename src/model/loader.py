"""Model and tokenizer loading.

Loads a HuggingFace causal LM and tokenizer from a model config dict,
applying 4-bit quantization when the method config indicates QLoRA (see
configs/model/*.yaml and configs/method/qlora.yaml).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.method.qlora import get_bnb_config


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _is_qlora(method_config):
    """Detect whether method_config specifies QLoRA (i.e. it carries 4-bit
    bitsandbytes quantization settings, as in configs/method/qlora.yaml)."""
    return bool(method_config) and method_config.get("load_in_4bit", False)


def load_model_and_tokenizer(model_config, method_config):
    """Load a causal LM and its tokenizer according to `model_config`,
    applying QLoRA quantization if indicated by `method_config`.

    Args:
        model_config: Dict with model settings (see configs/model/*.yaml).
            Expected keys: model_name_or_path, tokenizer_name_or_path,
            torch_dtype, attn_implementation, trust_remote_code.
        method_config: Dict with method settings (see configs/method/*.yaml).
            If it contains QLoRA quantization fields (load_in_4bit=True plus
            bnb_4bit_* settings), the model is loaded 4-bit quantized via
            BitsAndBytesConfig.

    Returns:
        A tuple (model, tokenizer).
    """
    torch_dtype = _DTYPE_MAP[model_config["torch_dtype"]]
    trust_remote_code = model_config.get("trust_remote_code", False)

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation=model_config["attn_implementation"],
        trust_remote_code=trust_remote_code,
    )

    if _is_qlora(method_config):
        load_kwargs["quantization_config"] = get_bnb_config(method_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        **load_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_name_or_path"],
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer