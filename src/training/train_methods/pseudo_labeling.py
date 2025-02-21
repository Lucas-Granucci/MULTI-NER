import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import ExperimentConfig
from utils import train_val_test_split
from preprocessing.dataset import U_NERDataset
from training.train_utils import create_dataloaders, evaluate_epoch, train_model


def pseudo_labeling_train(
    teacher_model,
    student_model_type,
    pseudo_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    save_model_path: str,
    save_pseudo_model_path: str,
    config: ExperimentConfig,
) -> Tuple[float, float]:
    """
    Train a model using pseudo-labeling technique.
    """

    # Dynamic confidence threshold
    confidence_threshold = pseudo_df["confidence_score"].quantile(0.7)
    filtered_pseudo_df = pseudo_df[pseudo_df["confidence_score"] > confidence_threshold]

    # Split data into train/val/test
    labeled_train_df, labeled_val_df, labeled_test_df = train_val_test_split(labeled_df, random_state=config.seed)
    pseudo_train_df, pseudo_val_df = train_test_split(filtered_pseudo_df, test_size=0.2, random_state=config.seed)

    print(f"PSEUDO-DATASET LEN: {len(filtered_pseudo_df)}     THRESH: {confidence_threshold}")

    # Create dataloaders
    labeled_train_loader, labeled_val_loader, labeled_test_loader = create_dataloaders(
        labeled_train_df, labeled_val_df, labeled_test_df, config.batch_size
    )
    pseudo_train_loader, pseudo_val_loader = create_dataloaders(
        pseudo_train_df, pseudo_val_df, None, config.batch_size
    )

    # --------------- TRAINING (PSEUDO-LABELS) --------------- #

    pseudo_model = student_model_type(config.num_tags).to(config.device)
    pseudo_model.load_state_dict(teacher_model.state_dict())
    best_pseudo_val_f1 = train_model(pseudo_model, config, pseudo_train_loader, pseudo_val_loader, save_pseudo_model_path)

    # --------------- TRAINING (LOW-RESOURCE) --------------- #

    # Reduce learning rate 
    config.bert_learning_rate *= 0.5
    config.lstm_learning_rate *= 0.5
    config.crf_learning_rate *= 0.5

    # Load model from pretrained high-resource model
    low_resource_model = student_model_type(config.num_tags).to(config.device)
    low_resource_model.load_state_dict(torch.load(save_pseudo_model_path, weights_only=True))

    best_labeled_val_f1 = train_model(low_resource_model, config, labeled_train_loader, labeled_val_loader, save_model_path)

    # --------------- EVALUATION --------------- #
    eval_model = student_model_type(config.num_tags).to(config.device)
    eval_model.load_state_dict(torch.load(save_model_path, weights_only=True))
    
    _, test_f1 = evaluate_epoch(labeled_test_loader, eval_model, config.device)

    # Return average F1 scores
    return best_labeled_val_f1, test_f1
