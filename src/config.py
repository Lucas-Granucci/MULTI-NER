import torch
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Training params
    num_tags: int
    batch_size: int
    patience: int
    epochs: int
    device: torch.device

    # Model params
    bert_learning_rate: float
    lstm_learning_rate: float
    crf_learning_rate: float

    # Other params
    seed: int
    low_resource_base_count: int
    results_dir: str
    model_dir: str
    logging_dir: str
