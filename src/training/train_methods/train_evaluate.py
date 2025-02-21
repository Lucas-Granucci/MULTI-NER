import torch
import pandas as pd
from typing import Tuple

from config import ExperimentConfig
from utils import train_val_test_split
from training.train_utils import train_model, evaluate_epoch, create_dataloaders


def train_evaluate(model_class, dataframe: pd.DataFrame, config: ExperimentConfig, save_model_path: str) -> Tuple[float, float]:
    """
    Train and evaluate model for NER
    """

    # Split data into train/val/test
    train_df, val_df, test_df = train_val_test_split(dataframe, random_state=42)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config.batch_size
    )

    # --------------- TRAINING --------------- #

    # Initialize model and move to device
    model = model_class(config.num_tags).to(config.device)
    # Train model and get best validation F1 score
    best_val_f1 = train_model(model, config, train_loader, val_loader, save_model_path)

    # --------------- EVALUATION --------------- #

    # Load the best model for evaluation
    eval_model = model_class(config.num_tags).to(config.device)
    eval_model.load_state_dict(torch.load(save_model_path, weights_only=True))
    # Evaluate model on test data and get test F1 score
    _, test_f1 = evaluate_epoch(test_loader, eval_model, config.device)

    # Return best validation F1 and test F1 scores
    return best_val_f1, test_f1