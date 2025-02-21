import ast
import pandas as pd
from typing import Dict


class LanguageDataManager:
    def __init__(self, base_path: str = "data/labeled"):
        self.base_path = base_path

    def load_languages(self, languages: list):
        language_data = {
            lang: self.load_data(f"{self.base_path}/{lang}_data.csv")
            for lang in languages
        }
        return language_data

    def load_language_pairs(
        self,
        low_resource_langs: list,
        high_resource_langs: list,
        low_resource_count: int,
        high_resource_count: int,
    ) -> Dict:
        low_resource_data = {
            lang: self.load_data(f"{self.base_path}/{lang}_data.csv").head(
                low_resource_count
            )
            for lang in low_resource_langs
        }

        high_resource_data = {
            lang: self.load_data(f"{self.base_path}/{lang}_data.csv").head(
                high_resource_count
            )
            for lang in high_resource_langs
        }

        return low_resource_data, high_resource_data

    def load_data(self, data_dir: str) -> pd.DataFrame:
        """
        Load language data from csv into dataframe
        """
        data = pd.read_csv(data_dir)

        def convert_to_list(string_representation: str) -> list:
            return ast.literal_eval(string_representation)

        data["tokens"] = data["tokens"].apply(convert_to_list)
        data["ner_tags"] = data["ner_tags"].apply(convert_to_list)

        return data

    def load_unlabeled_data(self, data_dir: str) -> pd.DataFrame:
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
