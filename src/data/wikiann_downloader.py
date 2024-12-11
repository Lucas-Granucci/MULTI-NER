import pandas as pd

from collections import defaultdict
from datasets import load_dataset

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class WikiANN_Downloader:
    def __init__(self):
        self.dataset = None

    def load_data(self, lang: str) -> pd.DataFrame:
        lang_dataset = self.get_multilingual_dataset(lang)

        lang_data = lang_dataset

        train_df = pd.DataFrame(lang_data["train"])
        val_df = pd.DataFrame(lang_data["validation"])
        test_df = pd.DataFrame(lang_data["test"])

        train_df["lang"] = lang
        val_df["lang"] = lang
        test_df["lang"] = lang

        return train_df, val_df, test_df

    def get_multilingual_dataset(self, lang: str) -> defaultdict:
        lang_dataset = load_dataset("unimelb-nlp/wikiann", name=lang, download_mode="reuse_dataset_if_exists")

        return lang_dataset
