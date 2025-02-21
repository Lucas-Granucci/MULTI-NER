import torch
import pandas as pd
from typing import Tuple, Any

from utils import train_val_test_split
from training.train_methods.train_evaluate import train_model, evaluate_epoch
from training.train_utils import create_dataloaders


def transfer_train(
    model_class,
    low_resource_dataframe: pd.DataFrame,
    high_resource_dataframe: pd.DataFrame,
    config: Any,
    save_model: str
    ) -> Tuple[float, float]:
    """
    Train model on with transfer learning data and evaluate results
    """

    # Split low-resource data into train/val/test
    low_resource_train_df, low_resource_val_df, low_resource_test_df = (
        train_val_test_split(low_resource_dataframe, random_state=config.seed)
    )

    # Combine low-resource train data with high-resource data for training
    train_df = pd.concat([high_resource_dataframe, low_resource_train_df])

    # Use only low-resource data for validation and testing
    val_df = low_resource_val_df
    test_df = low_resource_test_df

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config.batch_size
    )

    # --------------- TRAINING --------------- #

    model = model_class(config.num_tags).to(config.device)
    best_val_f1 = train_model(model, config, train_loader, val_loader, save_model)

    # --------------- EVALUATION --------------- #

    # Instantiate model and move to device
    model = model_class(config.num_tags).to(config.device)
    model.load_state_dict(torch.load(save_model, weights_only=True))
    _, test_f1 = evaluate_epoch(test_loader, model, config.device)

    # Return best train f1 and test f1
    return best_val_f1, test_f1
