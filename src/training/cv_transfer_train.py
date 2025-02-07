import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from training.utils import TrainConfig, setup_optimizer
from training.train_eval import train_fn, evaluate_fn
from preprocessing.dataloader import create_dataloaders
from preprocessing.dataset import NERDataset
from typing import Tuple


def cv_transfer_train(
    Model,
    low_resource_dataframe: pd.DataFrame,
    high_resource_dataframe: pd.DataFrame,
    k_splits: int,
    config: TrainConfig,
    verbose: bool = False,
) -> Tuple[float, float]:
    """
    Train model on cross-validation splits and return results
    """

    logging_results = {}

    # Create NER dataset for low-resource language
    low_resource_dataset = NERDataset(
        texts=low_resource_dataframe["tokens"].to_list(),
        tags=low_resource_dataframe["ner_tags"].to_list(),
    )

    # Prepare Kfold
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    # Precompute splits for the low-resource dataset
    splits = list(kf.split(low_resource_dataset))

    train_f1_score_ = 0
    val_f1_score_ = 0

    # Set up logging
    if verbose:
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []

    # Iterate over folds in dataset
    for train_idx, test_idx in tqdm(splits, desc="Fold: "):
        # Modify train_idx to include added transfer learning data
        added_indices = np.arange(
            len(low_resource_dataset),
            len(low_resource_dataset) + len(high_resource_dataframe),
        )
        train_idx = np.append(train_idx, added_indices)

        # Add transfer learning data
        low_resource_dataset.add_texts(
            new_texts=high_resource_dataframe["tokens"].to_list(),
            new_tags=high_resource_dataframe["ner_tags"].to_list(),
        )

        # Setup dataloaders with fold train/test split
        train_dataloader, test_dataloader = create_dataloaders(
            dataset=low_resource_dataset,
            batch_size=config.BATCH_SIZE,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        # Instantiate model and move to device
        model = Model(num_tags=config.NUM_TAGS)
        model.to(config.DEVICE)

        # Setup optimizer and learning rate scheduler
        optimizer = setup_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.3,
            total_iters=config.EPOCHS,
        )

        # Save scores from the best epoch
        best_val_f1 = -float("inf")
        best_train_f1 = -1

        # Implement early stopping
        patience = config.PATIENCE

        # Set up fold-level logging
        if verbose:
            fold_train_losses, fold_val_losses = [], []
            fold_train_f1s, fold_val_f1s = [], []

        # Train and evaluate model
        for _ in range(config.EPOCHS):
            # Train loop
            train_loss, train_f1 = train_fn(
                train_dataloader, model, optimizer, scheduler, config.DEVICE
            )
            # Evaluate loop
            val_loss, val_f1 = evaluate_fn(test_dataloader, model, config.DEVICE)

            # Store best f1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_train_f1 = train_f1

                # Reset patience counter
                patience = config.PATIENCE
            else:
                patience -= 1

            # Stop training if model doesn't improve
            if patience == 0:
                break

            # Fold-level logging
            if verbose:
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)
                fold_train_f1s.append(train_f1)
                fold_val_f1s.append(val_f1)

        # Save best F1 from fold
        val_f1_score_ += best_val_f1
        train_f1_score_ += best_train_f1

        # Global level logging
        if verbose:
            train_losses.append(fold_train_losses)
            val_losses.append(fold_val_losses)
            train_f1s.append(fold_train_f1s)
            val_f1s.append(fold_val_f1s)

    if verbose:
        logging_results["train_losses"] = train_losses
        logging_results["val_losses"] = val_losses
        logging_results["train_f1s"] = train_f1s
        logging_results["val_f1s"] = val_f1s

    # Return average F1 scores across folds
    return val_f1_score_ / k_splits, train_f1_score_ / k_splits, logging_results
