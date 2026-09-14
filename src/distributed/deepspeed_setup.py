"""DeepSpeed setup.

Initializes a DeepSpeed engine from a JSON config file (see
configs/distributed/deepspeed_zero2.json / deepspeed_zero3.json).
"""

import deepspeed


def init_deepspeed(model, optimizer, config_path):
    """Initialize a DeepSpeed engine wrapping `model` and `optimizer`.

    Args:
        model: A loaded (and, if applicable, FSDP/PEFT-wrapped) PyTorch model.
        optimizer: The optimizer instance whose parameter groups should be
            managed by the DeepSpeed engine.
        config_path: Path to a DeepSpeed JSON config file (e.g.
            "configs/distributed/deepspeed_zero2.json").

    Returns:
        A tuple (engine, optimizer, lr_scheduler) as produced by
        deepspeed.initialize:
            - engine: The DeepSpeed engine wrapping the model, used for
              forward/backward/step calls during training.
            - optimizer: The (possibly DeepSpeed-wrapped, e.g. ZeRO-partitioned)
              optimizer.
            - lr_scheduler: The learning rate scheduler, if one was configured
              in the DeepSpeed JSON config, otherwise None.
    """
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=config_path,
    )

    return engine, optimizer, lr_scheduler