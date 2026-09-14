"""
Loads and merges the per-tier config files (model / method / distributed /
cluster / data) into a single run config used by `src/train.py`.

Each fork of the pipeline contributes one file:
    model_path        configs/model/{llama3_8b,qwen3_30b,llama3_70b}.yaml
    method_path        configs/method/{full_ft,lora,qlora}.yaml
    distributed_path   configs/distributed/fsdp.yaml
                       OR configs/distributed/deepspeed_zero{2,3}.json
    cluster_path       configs/cluster/{single_gpu,single_node,multi_node}.yaml
    data_path          configs/data.yaml

`distributed_path` is special-cased: FSDP configs are YAML, DeepSpeed configs
are JSON (DeepSpeed's native format), so the loader picks a parser by file
extension rather than assuming YAML for every file.

Usage:
    from src.utils.config import load_and_merge_configs

    run_config = load_and_merge_configs(
        model_path="configs/model/llama3_8b.yaml",
        method_path="configs/method/qlora.yaml",
        distributed_path="configs/distributed/deepspeed_zero3.json",
        cluster_path="configs/cluster/single_node.yaml",
        data_path="configs/data.yaml",
    )
    run_config.model["name"]
    run_config.distributed["zero_optimization"]["stage"]
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Raised when a config file is missing, malformed, or missing required fields."""


# Fields each section MUST contain for training to proceed safely.
# Dotted paths address nested keys, e.g. "optimizer.lr".
REQUIRED_FIELDS = {
    "model": ["name", "path"],
    "method": ["type"],  # one of: full_ft | lora | qlora
    "distributed": ["backend"],  # one of: fsdp | deepspeed
    "cluster": ["num_nodes", "gpus_per_node"],
    "data": ["train_path"],
}


@dataclass
class RunConfig:
    model: Dict[str, Any] = field(default_factory=dict)
    method: Dict[str, Any] = field(default_factory=dict)
    distributed: Dict[str, Any] = field(default_factory=dict)
    cluster: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "method": self.method,
            "distributed": self.distributed,
            "cluster": self.cluster,
            "data": self.data,
        }


def _load_file(path: str) -> Dict[str, Any]:
    """Parse a single config file as YAML or JSON based on its extension."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r") as f:
            if ext == ".json":
                content = json.load(f)
            elif ext in (".yaml", ".yml"):
                content = yaml.safe_load(f)
            else:
                raise ConfigError(
                    f"Unsupported config file extension '{ext}' for '{path}'. "
                    "Expected .yaml, .yml, or .json."
                )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to parse '{path}': {e}") from e

    if content is None:
        raise ConfigError(f"Config file '{path}' is empty.")
    if not isinstance(content, dict):
        raise ConfigError(
            f"Config file '{path}' must contain a top-level mapping/object, "
            f"got {type(content).__name__}."
        )
    return content


def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    node = d
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_key)
        node = node[part]
    return node


def _validate_section(section_name: str, section: Dict[str, Any], source_path: str) -> None:
    missing = []
    for required_key in REQUIRED_FIELDS.get(section_name, []):
        try:
            _get_nested(section, required_key)
        except KeyError:
            missing.append(required_key)

    if missing:
        raise ConfigError(
            f"Missing required field(s) {missing} in '{section_name}' config "
            f"loaded from '{source_path}'."
        )


def load_and_merge_configs(
    model_path: str,
    method_path: str,
    distributed_path: str,
    cluster_path: str,
    data_path: str,
) -> RunConfig:
    """
    Reads the five per-tier config files, validates required fields, and
    merges them into a single `RunConfig`.

    Raises:
        ConfigError: if any file is missing/unparsable, or any section is
            missing one of its required fields. The error message names the
            offending section, file path, and missing field(s).
    """
    sources = {
        "model": model_path,
        "method": method_path,
        "distributed": distributed_path,
        "cluster": cluster_path,
        "data": data_path,
    }

    sections: Dict[str, Dict[str, Any]] = {}
    errors = []

    for name, path in sources.items():
        try:
            sections[name] = _load_file(path)
            _validate_section(name, sections[name], path)
        except ConfigError as e:
            errors.append(str(e))

    if errors:
        raise ConfigError(
            "Failed to build run config due to the following issue(s):\n  - "
            + "\n  - ".join(errors)
        )

    return RunConfig(
        model=sections["model"],
        method=sections["method"],
        distributed=sections["distributed"],
        cluster=sections["cluster"],
        data=sections["data"],
    )