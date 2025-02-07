import torch
import pandas as pd

from preprocessing.dataset import U_NERDataset
from torch.utils.data import DataLoader
from typing import Tuple


def pseudo_labeling_train(
    teacher_model,
    student_model,
    dataframe: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Generate predictions for model
    """

    # Create NER dataset for language
    dataset = U_NERDataset(texts=dataframe["sentences"].to_list())

    dataloader = DataLoader(dataset, batch_size=batch_size)

    teacher_model.to(device)
    teacher_model.eval()

    for data in dataloader:
        # Move data to device
        for i, j in data.items():
            data[i] = j.to(device)

        emissions, _ = teacher_model(**data)

        # Decode and get F1 score
        pred_tags = teacher_model.decode(emissions, data["mask"])
        pred_tags = [sequence + [0] * (256 - len(sequence)) for sequence in pred_tags]
    return pred_tags
