import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
    data: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
):
    train_df, temp_df = train_test_split(
        data, test_size=(1 - train_size), random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (val_size + test_size)),
        random_state=random_state,
    )
    return train_df, val_df, test_df


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
