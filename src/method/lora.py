"""LoRA method wrapper.

Applies HuggingFace PEFT LoRA adapters to a loaded causal LM using settings
read from a config dict (see configs/method/lora.yaml for the expected fields).
"""

from peft import LoraConfig, get_peft_model, TaskType


def apply(model, config):
    """Wrap `model` with LoRA adapters according to `config`.

    Args:
        model: A loaded HuggingFace causal LM (e.g. from
            AutoModelForCausalLM.from_pretrained(...)).
        config: Dict with LoRA settings. Expected keys:
            - r (int): LoRA rank.
            - lora_alpha (int): LoRA scaling factor.
            - lora_dropout (float): Dropout probability for LoRA layers.
            - target_modules (list[str]): Module names to attach adapters to.
            - bias (str, optional): Bias training mode. Defaults to "none".
            - task_type (str, optional): PEFT task type name. Defaults to
              "CAUSAL_LM".

    Returns:
        The PEFT-wrapped model with LoRA adapters attached.
    """
    task_type_name = config.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_name)

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
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