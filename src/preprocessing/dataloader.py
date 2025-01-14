import ast
import torch
import pandas as pd
from torch.utils.data import DataLoader
from preprocessing.dataset import NERDataset
from typing import Tuple


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load language data from csv into dataframe
    """
    data = pd.read_csv(data_dir)

    def convert_to_list(string_representation: str) -> list:
        return ast.literal_eval(string_representation)

    data["tokens"] = data["tokens"].apply(convert_to_list)
    data["ner_tags"] = data["ner_tags"].apply(convert_to_list)

    return data


def create_dataloaders(
    dataset: NERDataset, batch_size: int, train_idx: list, test_idx: list
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders with idx's from cross-validation split
    """
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )

    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    return train_dataloader, test_dataloader
