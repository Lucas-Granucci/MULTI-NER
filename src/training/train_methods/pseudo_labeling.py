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
    save_model: str,
    save_pseudo_model: str,
    config: ExperimentConfig,
) -> Tuple[float, float]:
    """
    Generate predictions for model
    """

    # Dynamic confidence threshold
    confidence_threshold = pseudo_df["confidence_score"].quantile(0.7)
    filtered_pseudo_df = pseudo_df[pseudo_df["confidence_score"] > confidence_threshold]

    # Split data into train/val/test
    lr_train_df, lr_val_df, lr_test_df = train_val_test_split(labeled_df, random_state=config.seed)
    pseduo_train_df, pseduo_val_df = train_test_split(filtered_pseudo_df, test_size=0.2, random_state=config.seed)

    print(f"PSEUDO-DATASET LEN: {len(filtered_pseudo_df)}     THRESH: {confidence_threshold}")

    # Create dataloaders
    lr_train_loader, lr_val_loader, lr_test_loader = create_dataloaders(
        lr_train_df, lr_val_df, lr_test_df, config.batch_size
    )
    pseudo_train_loader, pseudo_val_loader = create_dataloaders(
        pseduo_train_df, pseduo_val_df, None, config.batch_size
    )

    # --------------- TRAINING (PSEUDO-LABELS) --------------- #

    pseudo_model = student_model_type(config.num_tags).to(config.device)
    pseudo_model.load_state_dict(teacher_model.state_dict())
    best_val_f1 = train_model(pseudo_model, config, pseudo_train_loader, pseudo_val_loader, save_pseudo_model)

    # --------------- TRAINING (LOW-RESOURCE) --------------- #

    # Reduce learning rate 
    config.bert_learning_rate *= 0.5
    config.lstm_learning_rate *= 0.5
    config.crf_learning_rate *= 0.5

    # Load model from pretrained high-resource model
    low_resource_model = student_model_type(config.num_tags).to(config.device)
    low_resource_model.load_state_dict(torch.load(save_pseudo_model, weights_only=True))

    best_val_f1 = train_model(low_resource_model, config, lr_train_loader, lr_val_loader, save_model)

    # --------------- EVALUATION --------------- #
    eval_model = student_model_type(config.num_tags).to(config.device)
    eval_model.load_state_dict(torch.load(save_model, weights_only=True))
    
    _, test_f1 = evaluate_epoch(lr_test_loader, eval_model, config.device)

    # Return average F1 scores
    return best_val_f1, test_f1



