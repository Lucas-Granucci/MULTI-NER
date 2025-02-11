import torch
import pandas as pd
from training.train_utils import (
    TrainConfig,
    setup_optimizer,
    create_dataloaders,
    create_weighted_dataloaders,
)
from training.train_evaluate import train_epoch, evaluate_epoch
from typing import Tuple
from utils import train_val_test_split


def transfer_train(
    ModelClass,
    low_resource_dataframe: pd.DataFrame,
    high_resource_dataframe: pd.DataFrame,
    config: TrainConfig,
    save_model: str,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Train model on with transfer learning data and evaluate results
    """

    log_results = {}

    # Split low-resource data into train/val/test
    low_resource_train_df, low_resource_val_df, low_resource_test_df = (
        train_val_test_split(low_resource_dataframe, random_state=42)
    )

    # New weighting implementation (UNTESTED)
    train_loader, val_loader, test_loader = create_weighted_dataloaders(
        low_resource_train_df,
        high_resource_dataframe,
        low_resource_val_df,
        low_resource_test_df,
        config.BATCH_SIZE,
    )

    # Combine low-resource train data with high-resource data for training
    # train_df = pd.concat([low_resource_train_df, high_resource_dataframe])
    # train_df = train_df.sample(frac=1, random_state=42).reset_index(
    #     drop=True
    # )  # Shuffle train data

    # # Use only low-resource data for validation and testing
    # val_df = low_resource_val_df
    # test_df = low_resource_test_df

    # # Create dataloaders
    # train_loader, val_loader, test_loader = create_dataloaders(
    #     train_df, val_df, test_df, config.BATCH_SIZE
    # )

    # Initialize logging if verbose
    if verbose:
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []

    # --------------- TRAINING --------------- #
    # Instantiate model and move to device
    model = ModelClass(config.NUM_TAGS).to(config.DEVICE)

    # Setup optimizer and learning rate scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=0.3,
        total_iters=config.EPOCHS,
    )

    best_val_f1 = -float("inf")  # Track best validation F1-score
    patience_counter = config.PATIENCE  # Early stopping counter

    # Train model over epochs
    for _ in range(config.EPOCHS):
        # Train single epoch
        train_loss, train_f1 = train_epoch(
            train_loader, model, optimizer, scheduler, config.DEVICE
        )
        # Get validation scores for single epoch
        val_loss, val_f1 = evaluate_epoch(val_loader, model, config.DEVICE)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = config.PATIENCE  # Reset patience counter
            torch.save(model.state_dict(), save_model)
        else:
            patience_counter -= 1

        if patience_counter == 0:
            break  # Stop training if model doesn't improve

        if verbose:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

    # --------------- EVALUATION --------------- #
    # Instantiate model and move to device
    model = ModelClass(config.NUM_TAGS).to(config.DEVICE)
    model.load_state_dict(torch.load(save_model, weights_only=True))
    _, test_f1 = evaluate_epoch(test_loader, model, config.DEVICE)

    if verbose:
        log_results.update(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_f1s": train_f1s,
                "val_f1s": val_f1s,
            }
        )

    # Return average F1 scores across folds
    return best_val_f1, test_f1, log_results
