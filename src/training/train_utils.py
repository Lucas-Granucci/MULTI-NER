import torch
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from preprocessing.dataset import NERDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

from typing import List


def create_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, batch_size: int
):
    train_dataset = NERDataset(
        texts=train_df["tokens"].to_list(), tags=train_df["ner_tags"].to_list()
    )
    val_dataset = NERDataset(
        texts=val_df["tokens"].to_list(), tags=val_df["ner_tags"].to_list()
    )
    test_dataset = NERDataset(
        texts=test_df["tokens"].to_list(), tags=test_df["ner_tags"].to_list()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def create_weighted_dataloaders(
    low_resource_train_df, high_resource_df, val_df, test_df, batch_size: int
):
    # Assign  weighting to low and high resource datasets
    low_resource_train_dataset = NERDataset(
        texts=low_resource_train_df["tokens"].to_list(),
        tags=low_resource_train_df["ner_tags"].to_list(),
    )
    high_resource_train_dataset = NERDataset(
        texts=high_resource_df["tokens"].to_list(),
        tags=high_resource_df["ner_tags"].to_list(),
    )

    # Combine datasets
    combined_texts = (
        low_resource_train_dataset.texts + high_resource_train_dataset.texts
    )
    combined_tags = low_resource_train_dataset.tags + high_resource_train_dataset.tags
    combined_dataset = NERDataset(combined_texts, combined_tags)

    # Assign weights inversely proportional to dataset size
    num_low = len(low_resource_train_dataset)
    num_high = len(high_resource_train_dataset)
    total_samples = num_low + num_high

    weights_low = [
        num_high / total_samples
    ] * num_low  # Higher weight for low-resource samples
    weights_high = [
        num_low / total_samples
    ] * num_high  # Lower weight for high-resource samples
    all_weights = weights_low + weights_high

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=all_weights, num_samples=total_samples, replacement=True
    )

    # Create dataloader
    train_dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

    val_dataset = NERDataset(
        texts=val_df["tokens"].to_list(), tags=val_df["ner_tags"].to_list()
    )
    test_dataset = NERDataset(
        texts=test_df["tokens"].to_list(), tags=test_df["ner_tags"].to_list()
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def calculate_f1_score(
    target_tags: torch.Tensor,
    pred_tags: torch.Tensor | List[List[int]],
    attention_mask: torch.Tensor,
) -> float:
    """
    Calculate the micro F1-score for sequence tagging from decoded emissions
    """
    # Handle CRF return type (list)
    if type(pred_tags) is list:
        pred_tags = [sequence + [0] * (128 - len(sequence)) for sequence in pred_tags]
        pred_tags = torch.tensor(pred_tags).to(target_tags.get_device())

    # Flatten batch results
    target_tags = target_tags.view(-1)
    pred_tags = pred_tags.view(-1)
    attention_mask = attention_mask.view(-1)

    # Filter out padding and special tokens
    target_tags = target_tags[attention_mask == 1]
    pred_tags = pred_tags[attention_mask == 1]

    f1_micro = f1_score(target_tags.cpu(), pred_tags.cpu(), average="micro")
    return f1_micro


class TrainConfig:
    """
    Training configuration class
    """

    def __init__(
        self,
        num_tags: int,
        batch_size: int,
        bert_learning_rate: float,
        lstm_learning_rate: float,
        crf_learning_rate: float,
        patience: int,
        epochs: int,
        device: torch.device,
    ):
        self.NUM_TAGS = num_tags
        self.BATCH_SIZE = batch_size
        self.BERT_LEARNING_RATE = bert_learning_rate
        self.LSTM_LEARNING_RATE = lstm_learning_rate
        self.CRF_LEARNING_RATE = crf_learning_rate
        self.PATIENCE = patience
        self.EPOCHS = epochs
        self.DEVICE = device


def setup_optimizer(model: torch.nn.Module, config: TrainConfig) -> optim.Optimizer:
    """
    Setups up optimizer for different models with different learning rates for each part
    """
    param_groups = []
    if hasattr(model, "bert"):
        param_groups.append(
            {
                "params": model.bert.parameters(),
                "lr": config.BERT_LEARNING_RATE,
            }
        )

    if hasattr(model, "lstm"):
        param_groups.append(
            {
                "params": model.lstm.parameters(),
                "lr": config.LSTM_LEARNING_RATE,
            }
        )

    if hasattr(model, "crf"):
        param_groups.append(
            {
                "params": model.crf.parameters(),
                "lr": config.CRF_LEARNING_RATE,
            }
        )

    optimizer = optim.Adam(param_groups, weight_decay=0.01)
    return optimizer
