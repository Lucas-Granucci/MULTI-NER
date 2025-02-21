import torch
import pandas as pd
from typing import Tuple

from config import ExperimentConfig
from utils import train_val_test_split
from training.train_utils import train_model, evaluate_epoch, create_dataloaders


def train_evaluate(model_class, dataframe: pd.DataFrame, config: ExperimentConfig, save_model: str) -> Tuple[float, float]:
    """
    Train and evaluate model for NER
    """

    # Split data into train/val/test
    train_df, val_df, test_df = train_val_test_split(dataframe, random_state=42)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config.batch_size
    )

    # --------------- TRAINING --------------- #

    model = model_class(config.num_tags).to(config.device)
    best_val_f1 = train_model(model, config, train_loader, val_loader, save_model)

    # --------------- EVALUATION --------------- #

    eval_model = model_class(config.num_tags).to(config.device)
    eval_model.load_state_dict(torch.load(save_model, weights_only=True))
    _, test_f1 = evaluate_epoch(test_loader, eval_model, config.device)

    # Return best train f1 and test f1
    return best_val_f1, test_f1