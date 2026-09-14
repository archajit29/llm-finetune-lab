"""QLoRA method wrapper.

Prepares a 4-bit quantized causal LM for k-bit training and applies HuggingFace
PEFT LoRA adapters, using settings read from a config dict (see
configs/method/qlora.yaml for the expected fields).

Note: get_bnb_config() builds the BitsAndBytesConfig used to load the model in
4-bit precision. It must be called during model *loading*
(e.g. AutoModelForCausalLM.from_pretrained(..., quantization_config=get_bnb_config(config))),
not here — apply() only prepares and wraps an already-quantized model.
"""

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def get_bnb_config(config):
    """Build a BitsAndBytesConfig for 4-bit model loading.

    Args:
        config: Dict with quantization settings. Expected keys:
            - load_in_4bit (bool)
            - bnb_4bit_quant_type (str): e.g. "nf4".
            - bnb_4bit_compute_dtype (str): e.g. "bfloat16".
            - bnb_4bit_use_double_quant (bool)

    Returns:
        A transformers.BitsAndBytesConfig instance to pass as
        `quantization_config` when loading the model.
    """
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])

    return BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )


def apply(model, config):
    """Prepare a 4-bit quantized `model` for k-bit training and wrap with LoRA.

    Args:
        model: A causal LM already loaded in 4-bit precision (loaded with a
            quantization_config built from get_bnb_config()).
        config: Dict with LoRA settings. Expected keys:
            - r (int): LoRA rank.
            - lora_alpha (int): LoRA scaling factor.
            - lora_dropout (float, optional): Dropout probability for LoRA layers.
            - target_modules (list[str]): Module names to attach adapters to.
            - bias (str, optional): Bias training mode. Defaults to "none".
            - task_type (str, optional): PEFT task type name. Defaults to
              "CAUSAL_LM".

    Returns:
        The PEFT-wrapped model with LoRA adapters attached, ready for k-bit training.
    """
    model = prepare_model_for_kbit_training(model)

    task_type_name = config.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_name)

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.0),
        target_modules=config["target_modules"],
        bias=config.get("bias", "none"),
        task_type=task_type,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    pct = 100 * trainable_params / total_params if total_params else 0.0
    print(
        f"trainable params: {trainable_params:,} || "
        f"total params: {total_params:,} || "
        f"trainable%: {pct:.4f}"
    )

    return model