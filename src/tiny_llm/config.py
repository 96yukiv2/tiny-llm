"""Read and validate the intentionally small training configuration."""
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    data_path: str
    output_path: str
    seed: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    batch_size: int
    steps: int
    learning_rate: float
    eval_interval: int

    def __post_init__(self):
        for name in ("block_size", "n_embd", "n_head", "n_layer", "batch_size", "steps", "eval_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    expected = {field.name for field in fields(TrainConfig)}
    if set(data) != expected:
        raise ValueError(f"Configuration keys: missing={sorted(expected - set(data))}, "
                         f"unknown={sorted(set(data) - expected)}")
    return TrainConfig(**data)
