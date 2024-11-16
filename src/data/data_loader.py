import json
from typing import Dict, Any

from data.wikiann_downloader import WikiANN_Downloader
from utils.logging import print_submessage

class LanguageDataLoader:
    
    def __init__(self, config):
        self.path_to_lang_groups = config["languages"]["path_to_groups"]
        self.wikiann_downloader = WikiANN_Downloader()

    def load_language_groups(self) -> Dict[str, Any]:

        with open(self.path_to_lang_groups, "r") as file:
            language_groups = json.load(file)

        language_data = {}
        for group_name, lang_dict in language_groups.items():

            print_submessage(f"Downloading data for {group_name} languages...")
            language_data[group_name] = self._process_language_group(lang_dict)

        return language_data

    def _process_language_group(self, lang_dict: dict) -> dict:

        low_resource_langs = lang_dict["low_resource"]
        high_resource_langs = lang_dict["high_resource"]

        low_resource_train, low_resource_val, low_resource_test = self.wikiann_downloader.load_split_data(low_resource_langs, shuffle=True)
        high_resource_data = self.wikiann_downloader.load_data(high_resource_langs, shuffle=True)

        return {
            "low_resource": {
                "train": low_resource_train,
                "val": low_resource_val,
                "test": low_resource_test
            },
            "high_resource": {
                "train": high_resource_data
            }
        }

