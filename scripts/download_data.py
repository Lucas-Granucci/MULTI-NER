import json
import pandas as pd
from typing import DefaultDict
from datasets import load_dataset


def load_language_data(language_json_dir: str, data_dir: str) -> None:
    """
    Load lanugage config and download required language data
    """
    with open(language_json_dir, "r") as file:
        language_groups = json.load(file)

    for lang_group in language_groups:
        low_resource_code = lang_group["low_resource_code"]
        high_resource_code = lang_group["high_resource_code"]

        save_wikiann_datasets(lang_code=low_resource_code, data_dir=data_dir)

        if high_resource_code != "":
            save_wikiann_datasets(lang_code=high_resource_code, data_dir=data_dir)

    print("Download complete")


def save_wikiann_datasets(lang_code: str, data_dir: str) -> None:
    """
    Combine wikiann dataset splits and save dataframe
    """
    lang_data = get_wikiann_dataset(lang_code=lang_code)

    train_df = pd.DataFrame(lang_data["train"])
    val_df = pd.DataFrame(lang_data["validation"])
    test_df = pd.DataFrame(lang_data["test"])

    complete_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    save_dir = f"{data_dir}/{lang_code}_data.csv"
    complete_df.to_csv(save_dir)


def get_wikiann_dataset(lang_code: str) -> DefaultDict:
    """
    Get wikiann dataset from HuggingFace datasets
    """
    lang_dataset = load_dataset("unimelb-nlp/wikiann", name=lang_code)
    return lang_dataset


load_language_data(language_json_dir="data/languages.json", data_dir="data/raw")
