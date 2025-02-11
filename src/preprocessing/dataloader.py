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


def load_unlabeled_data(data_dir: str) -> pd.DataFrame:
    """
    Load unlabeled data from txt files into dataframe
    """
    with open(data_dir, "r", encoding="utf8") as file:
        sentences = file.readlines()

    def convert_to_tokens_list(sentence: str) -> list:
        return sentence.split(" ")

    data = pd.DataFrame({"sentences": sentences})
    data["sentences"] = data["sentences"].apply(convert_to_tokens_list)
    return data
