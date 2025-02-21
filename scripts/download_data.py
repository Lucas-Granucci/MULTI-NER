import json
import pandas as pd
from typing import DefaultDict
from datasets import load_dataset

def load_language_data(language_json_path: str, output_dir: str) -> None:
    """
    Load language configuration and download required language data.
    """
    with open(language_json_path, "r") as file:
        language_groups = json.load(file)

    for lang_group in language_groups:
        low_resource_code = lang_group["low_resource_code"]
        high_resource_code = lang_group["high_resource_code"]

        # Save low resource language dataset
        save_wikiann_datasets(lang_code=low_resource_code, output_dir=output_dir)

        # Save high resource language dataset if available
        if high_resource_code:
            save_wikiann_datasets(lang_code=high_resource_code, output_dir=output_dir)

    print("Download complete")

def save_wikiann_datasets(lang_code: str, output_dir: str) -> None:
    """
    Combine wikiann dataset splits and save as a single dataframe.
    """
    lang_data = get_wikiann_dataset(lang_code=lang_code)

    train_df = pd.DataFrame(lang_data["train"])
    val_df = pd.DataFrame(lang_data["validation"])
    test_df = pd.DataFrame(lang_data["test"])

    # Combine all splits into a single dataframe
    complete_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Save the combined dataframe to a CSV file
    save_path = f"{output_dir}/{lang_code}_data.csv"
    complete_df.to_csv(save_path)

def get_wikiann_dataset(lang_code: str) -> DefaultDict:
    """
    Get wikiann dataset from HuggingFace datasets.
    """
    lang_dataset = load_dataset("unimelb-nlp/wikiann", name=lang_code)
    return lang_dataset

# Load language data based on the configuration file
load_language_data(language_json_path="data/languages.json", output_dir="data/labeled")
