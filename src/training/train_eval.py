import torch
from torch.utils.data import DataLoader
from training.utils import calculate_f1_score

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Tuple


def train_fn(
    train_dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train model for sequence tagging ; return loss and F1 score
    """
    model.train()

    loss_ = 0
    f1_score_ = 0

    for data in train_dataloader:
        # Move data to device
        for i, j in data.items():
            data[i] = j.to(device)

        # Backward propagation
        optimizer.zero_grad()
        emissions, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ += loss.item()

        # Decode and get F1 score
        pred_tags = model.decode(emissions, data["mask"])
        f1_score = calculate_f1_score(data["target_tags"], pred_tags, data["mask"])

        f1_score_ += f1_score

    # Return average loss and F1-score
    return loss_ / len(train_dataloader), f1_score_ / len(train_dataloader)


def evaluate_fn(
    val_dataloader: DataLoader, model: torch.nn.Module, device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model for sequence tagging ; return loss and F1-score
    """
    model.eval()

    loss_ = 0
    f1_score_ = 0

    for data in val_dataloader:
        # Move data to device
        for i, j in data.items():
            data[i] = j.to(device)

        emissions, loss = model(**data)
        loss_ += loss.item()

        # Decode predictions and calculate F1-score
        pred_tags = model.decode(emissions, data["mask"])
        f1_score = calculate_f1_score(data["target_tags"], pred_tags, data["mask"])
        f1_score_ += f1_score

    # Return average loss and F1-score
    return loss_ / len(val_dataloader), f1_score_ / len(val_dataloader)
