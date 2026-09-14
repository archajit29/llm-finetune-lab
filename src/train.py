"""
src/train.py

Single training entrypoint for LLM fine-tuning.

Reads model + method + distributed + cluster + data config files, merges them
into one run config, reads the distributed environment torchrun exports
(RANK / WORLD_SIZE / LOCAL_RANK), dynamically dispatches to the selected
fine-tuning method (full_ft / lora / qlora) under src/method/ and the selected
distributed strategy (fsdp / deepspeed) under src/distributed/, loads the
model + tokenizer, builds a HuggingFace Trainer, trains, and saves a checkpoint.

Launch examples:
    # single GPU
    python src/train.py \
        --model configs/model/llama3_8b.yaml \
        --method configs/method/lora.yaml \
        --distributed configs/distributed/fsdp.yaml \
        --cluster configs/cluster/single_gpu.yaml \
        --data configs/data.yaml

    # multi-GPU / multi-node — torchrun sets RANK / WORLD_SIZE / LOCAL_RANK
    torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC \
        src/train.py --model ... --method ... --distributed ... --cluster ... --data ...

Expected interfaces of collaborating modules (implemented elsewhere in the repo):
    src/utils/config.py
        load_config(path: str) -> dict
        merge_configs(model_cfg, method_cfg, distributed_cfg, cluster_cfg, data_cfg) -> dict
    src/utils/seed.py
        set_seed(seed: int) -> None
    src/utils/checkpoint.py
        save_checkpoint(model, tokenizer, output_dir: str, config: dict) -> None
    src/model/loader.py
        load_model_and_tokenizer(model_cfg: dict, method_cfg: dict) -> (model, tokenizer)
    src/method/{full_ft,lora,qlora}.py
        apply(model, method_cfg: dict) -> model
    src/distributed/{fsdp_setup,deepspeed_setup}.py
        setup(model, distributed_cfg: dict, dist_env: dict) -> (model, extra_trainer_kwargs: dict)
    src/data/dataset.py
        load_packed_dataset(data_cfg: dict) -> datasets.Dataset
    src/data/collator.py
        get_collator(tokenizer) -> Callable
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is importable as `src.*` when this file is run
# directly as `python src/train.py` (rather than `python -m src.train`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import Trainer, TrainingArguments  # noqa: E402

from src.utils.config import load_config, merge_configs  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.checkpoint import save_checkpoint  # noqa: E402
from src.model.loader import load_model_and_tokenizer  # noqa: E402
from src.data.dataset import load_packed_dataset  # noqa: E402
from src.data.collator import get_collator  # noqa: E402


SUPPORTED_METHODS = {"full_ft", "lora", "qlora"}
SUPPORTED_DISTRIBUTED = {
    "fsdp": "fsdp_setup",
    "deepspeed": "deepspeed_setup",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM fine-tuning entrypoint")
    parser.add_argument("--model", required=True, help="Path to model config (configs/model/*.yaml)")
    parser.add_argument("--method", required=True, help="Path to method config (configs/method/*.yaml)")
    parser.add_argument("--distributed", required=True, help="Path to distributed config (configs/distributed/*)")
    parser.add_argument("--cluster", required=True, help="Path to cluster config (configs/cluster/*.yaml)")
    parser.add_argument("--data", required=True, help="Path to data config (configs/data.yaml)")
    parser.add_argument("--output_dir", default=None, help="Override output_dir from config")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to a checkpoint to resume from")
    return parser.parse_args()


def get_dist_env() -> dict:
    """Read distributed env vars exported by torchrun. Defaults to a single
    process run (rank 0, world size 1) when not launched via torchrun."""
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def setup_logging(rank: int) -> logging.Logger:
    """Configure logging so only rank 0 prints INFO logs to the console.
    Non-zero ranks are gated up to WARNING to avoid interleaved, duplicate
    log spam across processes in multi-GPU / multi-node runs."""
    logging.basicConfig(
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO if rank == 0 else logging.WARNING,
        stream=sys.stdout,
    )
    return logging.getLogger("train")


def apply_method(model, method_name: str, method_cfg: dict, logger: logging.Logger):
    """Dynamically import and apply the fine-tuning method module matching
    method_name (e.g. 'lora' -> src/method/lora.py), which must expose
    an apply(model, method_cfg) -> model function."""
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown method '{method_name}'. Must be one of {sorted(SUPPORTED_METHODS)}."
        )

    module_path = f"src.method.{method_name}"
    logger.info(f"Importing method module: {module_path}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import '{module_path}'. Does the file exist?") from e

    if not hasattr(module, "apply"):
        raise AttributeError(f"{module_path} must define an apply(model, method_cfg) function")

    return module.apply(model, method_cfg)


def apply_distributed(model, distributed_name: str, distributed_cfg: dict, dist_env: dict, logger: logging.Logger):
    """Dynamically import and apply the distributed wrapper matching
    distributed_name (e.g. 'fsdp' -> src/distributed/fsdp_setup.py), which
    must expose a setup(model, distributed_cfg, dist_env) -> (model, extra_trainer_kwargs)
    function. extra_trainer_kwargs is merged into TrainingArguments so each
    strategy can inject its own required flags (e.g. fsdp policy, deepspeed
    config path)."""
    if distributed_name not in SUPPORTED_DISTRIBUTED:
        raise ValueError(
            f"Unknown distributed strategy '{distributed_name}'. "
            f"Must be one of {sorted(SUPPORTED_DISTRIBUTED)}."
        )

    module_name = SUPPORTED_DISTRIBUTED[distributed_name]
    module_path = f"src.distributed.{module_name}"
    logger.info(f"Importing distributed module: {module_path}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import '{module_path}'. Does the file exist?") from e

    if not hasattr(module, "setup"):
        raise AttributeError(f"{module_path} must define a setup(model, distributed_cfg, dist_env) function")

    return module.setup(model, distributed_cfg, dist_env)


def build_training_arguments(config: dict, dist_env: dict, extra_trainer_kwargs: dict, args: argparse.Namespace) -> TrainingArguments:
    """Build TrainingArguments from the merged config's `training` section,
    with CLI overrides and distributed-strategy-specific kwargs layered on top."""
    training_cfg = dict(config.get("training", {}))

    output_dir = args.output_dir or training_cfg.pop("output_dir", "outputs/run")

    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg.pop("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.pop("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.pop("learning_rate", 2e-5),
        num_train_epochs=training_cfg.pop("num_train_epochs", 1),
        max_steps=training_cfg.pop("max_steps", -1),
        warmup_ratio=training_cfg.pop("warmup_ratio", 0.03),
        lr_scheduler_type=training_cfg.pop("lr_scheduler_type", "cosine"),
        logging_steps=training_cfg.pop("logging_steps", 10),
        save_steps=training_cfg.pop("save_steps", 500),
        save_total_limit=training_cfg.pop("save_total_limit", 3),
        bf16=training_cfg.pop("bf16", True),
        gradient_checkpointing=training_cfg.pop("gradient_checkpointing", True),
        report_to=training_cfg.pop("report_to", []),
        # Only rank 0 writes to disk / prints progress bars.
        disable_tqdm=dist_env["rank"] != 0,
        log_level="info" if dist_env["rank"] == 0 else "warning",
    )

    # Any remaining training_cfg keys the user set explicitly pass straight through.
    kwargs.update(training_cfg)

    # Distributed-strategy-specific kwargs (e.g. fsdp / fsdp_config, deepspeed json path)
    # take precedence since they encode requirements of the chosen strategy.
    kwargs.update(extra_trainer_kwargs or {})

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    dist_env = get_dist_env()
    logger = setup_logging(dist_env["rank"])

    logger.info("=" * 60)
    logger.info("LLM Fine-Tuning — starting run")
    logger.info(
        f"world_size={dist_env['world_size']} rank={dist_env['rank']} local_rank={dist_env['local_rank']}"
    )

    logger.info("Loading configs: model / method / distributed / cluster / data")
    model_cfg = load_config(args.model)
    method_cfg = load_config(args.method)
    distributed_cfg = load_config(args.distributed)
    cluster_cfg = load_config(args.cluster)
    data_cfg = load_config(args.data)

    config = merge_configs(model_cfg, method_cfg, distributed_cfg, cluster_cfg, data_cfg)
    config["dist_env"] = dist_env

    set_seed(config.get("seed", 42))

    method_name = method_cfg.get("method", method_cfg.get("name"))
    distributed_name = distributed_cfg.get("distributed", distributed_cfg.get("name"))
    logger.info(f"Method: {method_name} | Distributed strategy: {distributed_name}")

    logger.info(f"Loading model + tokenizer from model config: {args.model}")
    model, tokenizer = load_model_and_tokenizer(model_cfg, method_cfg)

    logger.info(f"Applying fine-tuning method: {method_name}")
    model = apply_method(model, method_name, method_cfg, logger)

    logger.info(f"Applying distributed strategy: {distributed_name}")
    model, extra_trainer_kwargs = apply_distributed(model, distributed_name, distributed_cfg, dist_env, logger)

    logger.info(f"Loading packed training dataset from data config: {args.data}")
    train_dataset = load_packed_dataset(data_cfg)
    collator = get_collator(tokenizer)
    logger.info(f"Training examples: {len(train_dataset):,}")

    training_args = build_training_arguments(config, dist_env, extra_trainer_kwargs, args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("Training finished.")

    # Only rank 0 persists artifacts to avoid every process racing to write
    # the same files.
    if dist_env["rank"] == 0:
        logger.info(f"Saving final checkpoint to: {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        save_checkpoint(model, tokenizer, training_args.output_dir, config)
        logger.info("Checkpoint saved.")

    logger.info("Run complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()