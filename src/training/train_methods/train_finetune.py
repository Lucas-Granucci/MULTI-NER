import torch
import pandas as pd
from typing import Tuple, Any
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from utils import train_val_test_split
from training.train_methods.train_evaluate import evaluate_epoch
from training.train_utils import create_dataloaders, train_model


def train_finetune(
    model_class,
    low_resource_dataframe: pd.DataFrame,
    high_resource_dataframe: pd.DataFrame,
    save_model_path: str,
    save_high_resource_model_path: str,
    config: ExperimentConfig,
) -> Tuple[float, float]:
    """
    Train model with transfer learning data and evaluate results
    """

    # Split high-resource data into training and validation sets
    high_resource_train_df, high_resource_val_df = train_test_split(
        high_resource_dataframe, test_size=0.2, random_state=config.seed
    )
    
    # Split low-resource data into training, validation, and test sets
    low_resource_train_df, low_resource_val_df, low_resource_test_df = (
        train_val_test_split(low_resource_dataframe, random_state=config.seed)
    )

    # Create dataloaders for high-resource data
    high_resource_train_loader, high_resource_val_loader = create_dataloaders(
        high_resource_train_df, high_resource_val_df, None, config.batch_size
    )
    
    # Create dataloaders for low-resource data
    low_resource_train_loader, low_resource_val_loader, low_resource_test_loader = create_dataloaders(
        low_resource_train_df, low_resource_val_df, low_resource_test_df, config.batch_size
    )

    # --------------- TRAINING (HIGH-RESOURCE) --------------- #

    # Initialize and train the high-resource model
    high_resource_model = model_class(config.num_tags).to(config.device)
    best_high_resource_val_f1 = train_model(
        high_resource_model, config, high_resource_train_loader, high_resource_val_loader, save_high_resource_model_path
    )

    # --------------- TRAINING (LOW-RESOURCE) --------------- #

    # Load model from pretrained high-resource model
    low_resource_model = model_class(config.num_tags).to(config.device)
    low_resource_model.load_state_dict(torch.load(save_high_resource_model_path, weights_only=True))

    # Train the low-resource model
    best_low_resource_val_f1 = train_model(
        low_resource_model, config, low_resource_train_loader, low_resource_val_loader, save_model_path
    )

    # --------------- EVALUATION --------------- #

    # Load the trained low-resource model for evaluation
    eval_model = model_class(config.num_tags).to(config.device)
    eval_model.load_state_dict(torch.load(save_model_path, weights_only=True))

    # Evaluate the model on the test set
    _, test_f1 = evaluate_epoch(low_resource_test_loader, eval_model, config.device)

    # Return best validation F1 score and test F1 score
    return best_low_resource_val_f1, test_f1
