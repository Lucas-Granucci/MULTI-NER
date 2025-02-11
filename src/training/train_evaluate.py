import torch
import pandas as pd

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from training.train_utils import (
    TrainConfig,
    setup_optimizer,
    create_dataloaders,
    calculate_f1_score,
)

from utils import train_val_test_split

from typing import Tuple


def train_evaluate(
    ModelClass,
    dataframe: pd.DataFrame,
    config: TrainConfig,
    save_model: str,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Train and evaluate model for NER
    """

    log_results = {}

    # Split data into train/val/test
    train_df, val_df, test_df = train_val_test_split(dataframe, random_state=42)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config.BATCH_SIZE
    )

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


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train model for sequence tagging ; return loss and F1 score
    """
    model.train()
    total_loss, total_f1 = 0, 0

    for batch in dataloader:
        # Move data to device
        batch = {key: value.to(device) for key, value in batch.items()}

        # Backward propagation
        optimizer.zero_grad()
        emissions, loss = model(**batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        # Decode and get F1 score
        pred_tags = model.decode(emissions, batch["mask"])
        f1_score = calculate_f1_score(batch["target_tags"], pred_tags, batch["mask"])

        total_f1 += f1_score

    # Return average loss and F1-score
    return total_loss / len(dataloader), total_f1 / len(dataloader)


def evaluate_epoch(
    dataloader: DataLoader, model: torch.nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model for sequence tagging ; return loss and F1-score
    """
    model.eval()
    total_loss, total_f1 = 0, 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}

        emissions, loss = model(**batch)
        total_loss += loss.item()

        # Decode predictions and calculate F1-score
        pred_tags = model.decode(emissions, batch["mask"])
        f1_score = calculate_f1_score(batch["target_tags"], pred_tags, batch["mask"])
        total_f1 += f1_score

    # Return average loss and F1-score
    return total_loss / len(dataloader), total_f1 / len(dataloader)
