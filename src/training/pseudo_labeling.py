import torch
import pandas as pd

from training.train_evaluate import train_epoch, evaluate_epoch
from training.train_utils import create_dataloaders, TrainConfig, setup_optimizer
from utils import train_val_test_split
from preprocessing.dataset import U_NERDataset
from torch.utils.data import DataLoader
from typing import Tuple


def pseudo_labeling_train(
    teacher_model,
    student_model_type,
    unlabeled_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    save_model: str,
    config: TrainConfig,
) -> Tuple[float, float]:
    """
    Generate predictions for model
    """

    # Create unlabeled NER dataset
    dataset = U_NERDataset(texts=unlabeled_df["sentences"].to_list())

    unlabeled_dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    teacher_model.to(config.DEVICE)
    teacher_model.eval()

    # Generate pseudo-labels
    pseudo_sents = []
    pseudo_tags = []

    for data in unlabeled_dataloader:
        # Remove information before passing through model
        sentences = data["sentence"]
        word_counts = data["sentence_length"]

        del data["sentence_length"]
        del data["sentence"]

        # Move data to device
        for i, j in data.items():
            data[i] = j.to(config.DEVICE)

        emissions, _ = teacher_model(**data)

        # Decode and get F1 score
        pred_tags = teacher_model.decode(emissions, data["mask"])

        pred_tags = [
            sequence + [0] * (dataset.MAX_LEN - len(sequence)) for sequence in pred_tags
        ]

        # Trim pred tags
        trimmed_pred_tags = []
        for tag_seq, word_count in zip(pred_tags, word_counts):
            tag_seq = tag_seq[1:-1]
            trimmed_pred_tags.append(tag_seq[:word_count])

        pseudo_sents.extend([sent.split(" ") for sent in sentences])
        pseudo_tags.extend(trimmed_pred_tags)

    # Create new dataset combining original data and pseudo-labels
    pseudo_df = pd.DataFrame({"tokens": pseudo_sents, "ner_tags": pseudo_tags})

    # Split data into train/val/test
    train_df, val_df, test_df = train_val_test_split(labeled_df, random_state=42)

    # Combine train dataframe with pseudo labels
    combined_df = pd.concat([train_df, pseudo_df])

    train_loader, val_loader, test_loader = create_dataloaders(
        combined_df, val_df, test_df, config.BATCH_SIZE
    )

    # --------------- TRAINING --------------- #

    student_model = student_model_type(num_tags=config.NUM_TAGS)

    # Setup optimizer and learning rate scheduler
    optimizer = setup_optimizer(student_model, config)
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
            train_loader, student_model, optimizer, scheduler, config.DEVICE
        )
        # Get validation scores for single epoch
        val_loss, val_f1 = evaluate_epoch(val_loader, student_model, config.DEVICE)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = config.PATIENCE  # Reset patience counter
            torch.save(student_model.state_dict(), save_model)
        else:
            patience_counter -= 1

        if patience_counter == 0:
            break  # Stop training if model doesn't improve

    # --------------- EVALUATION --------------- #
    # Instantiate model and move to device
    eval_model = student_model_type(config.NUM_TAGS).to(config.DEVICE)
    eval_model.load_state_dict(torch.load(save_model, weights_only=True))
    _, test_f1 = evaluate_epoch(test_loader, eval_model, config.DEVICE)

    # Return average F1 scores
    return best_val_f1, test_f1

    return pseudo_sents
