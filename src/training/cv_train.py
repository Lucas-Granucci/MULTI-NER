import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

from training.utils import TrainConfig, setup_optimizer
from training.train_eval import train_fn, evaluate_fn
from preprocessing.dataloader import create_dataloaders
from preprocessing.dataset import NERDataset
from typing import Tuple


def cv_train(
    Model, dataset: NERDataset, k_splits: int, config: TrainConfig
) -> Tuple[float, float]:
    """
    Train model on cross-validation splits and return results
    """
    # Prepare Kfold
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)

    train_f1_score_ = 0
    val_f1_score_ = 0

    # Iterate over folds in dataset
    for train_idx, test_idx in tqdm((kf.split(dataset)), desc="Fold: "):
        # Setup dataloaders with fold train/test split
        train_dataloader, test_dataloader = create_dataloaders(
            dataset=dataset,
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

        # Train and evaluate model
        for _ in range(config.EPOCHS):
            # Train loop
            _, train_f1 = train_fn(
                train_dataloader, model, optimizer, scheduler, config.DEVICE
            )
            # Evaluate loop
            _, val_f1 = evaluate_fn(test_dataloader, model, config.DEVICE)

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

        # Save best F1 from fold
        val_f1_score_ += best_val_f1
        train_f1_score_ += best_train_f1

    # Return average F1 scores across folds
    return val_f1_score_ / k_splits, train_f1_score_ / k_splits
