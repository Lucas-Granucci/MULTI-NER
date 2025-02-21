import torch
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from torch.optim import Optimizer
from preprocessing.dataset import NERDataset
from torch.utils.data import DataLoader
from typing import List, Tuple

def train_model(model, config, train_loader, val_loader, save_model_path):
    # Setup optimizer and learning rate scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    best_val_f1 = -float("inf")  # Track best validation F1-score
    patience_counter = config.patience  # Early stopping counter

    # Train model over epochs
    pbar = tqdm(total=config.epochs, desc="Train F1: 0.0    Val F1: 0.0", leave=False)
    for _ in range(config.epochs):
        # Train single epoch
        train_loss, train_f1 = train_epoch(train_loader, model, optimizer, config.device)
        
        # Get validation scores for single epoch
        val_loss, val_f1 = evaluate_epoch(val_loader, model, config.device)

        # Reduce learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = config.patience  # Reset patience counter
            torch.save(model.state_dict(), save_model_path)
        else:
            patience_counter -= 1

        if patience_counter == 0:
            break  # Stop training if model doesn't improve

        # Update progress bar
        pbar.set_description(f"Train F1: {train_f1:.4f}    Val F1: {val_f1:.4f}")
        pbar.update()

    return best_val_f1

def train_epoch(dataloader: DataLoader, model: torch.nn.Module, optimizer: Optimizer, device: torch.device) -> Tuple[float, float]:
    """
    Train model for sequence tagging; return loss and F1 score
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
        total_loss += loss.item()

        # Decode and get F1 score
        pred_tags = model.decode(emissions, batch["mask"])
        f1 = calculate_f1_score(batch["target_tags"], pred_tags, batch["mask"])
        total_f1 += f1

    # Return average loss and F1-score
    return total_loss / len(dataloader), total_f1 / len(dataloader)

def evaluate_epoch(dataloader: DataLoader, model: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model for sequence tagging; return loss and F1-score
    """
    model.eval()
    total_loss, total_f1 = 0, 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}

        emissions, loss = model(**batch)
        total_loss += loss.item()

        # Decode predictions and calculate F1-score
        pred_tags = model.decode(emissions, batch["mask"])
        f1 = calculate_f1_score(batch["target_tags"], pred_tags, batch["mask"])
        total_f1 += f1

    # Return average loss and F1-score
    return total_loss / len(dataloader), total_f1 / len(dataloader)

def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int):
    train_dataset = NERDataset(texts=train_df["tokens"].to_list(), tags=train_df["ner_tags"].to_list())
    val_dataset = NERDataset(texts=val_df["tokens"].to_list(), tags=val_df["ner_tags"].to_list())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    if test_df is not None:
        test_dataset = NERDataset(texts=test_df["tokens"].to_list(), tags=test_df["ner_tags"].to_list())
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
    
    return train_dataloader, val_dataloader

def calculate_f1_score(target_tags: torch.Tensor, pred_tags: torch.Tensor | List[List[int]], attention_mask: torch.Tensor) -> float:
    """
    Calculate the micro F1-score for sequence tagging from decoded emissions
    """
    # Handle CRF return type (list)
    if isinstance(pred_tags, list):
        pred_tags = [sequence + [0] * (101 - len(sequence)) for sequence in pred_tags]
        pred_tags = torch.tensor(pred_tags).to(target_tags.device)

    # Flatten batch results
    target_tags = target_tags.view(-1)
    pred_tags = pred_tags.view(-1)
    attention_mask = attention_mask.view(-1)

    # Filter out padding and special tokens
    target_tags = target_tags[attention_mask == 1]
    pred_tags = pred_tags[attention_mask == 1]

    f1_micro = f1_score(target_tags.cpu(), pred_tags.cpu(), average="micro")
    return f1_micro

def setup_optimizer(model: torch.nn.Module, config) -> optim.Optimizer:
    """
    Set up optimizer for different models with different learning rates for each part
    """
    param_groups = []
    if hasattr(model, "bert"):
        param_groups.append({"params": model.bert.parameters(), "lr": config.bert_learning_rate})

    if hasattr(model, "lstm"):
        param_groups.append({"params": model.lstm.parameters(), "lr": config.lstm_learning_rate})

    if hasattr(model, "crf"):
        param_groups.append({"params": model.crf.parameters(), "lr": config.crf_learning_rate})

    optimizer = optim.Adam(param_groups, weight_decay=0.01)
    return optimizer
