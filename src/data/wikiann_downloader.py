import pandas as pd

from collections import defaultdict
from datasets import load_dataset

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class WikiANN_Downloader:
    def __init__(self):
        self.dataset = None

    def load_data(self, lang: str, shuffle: bool = False) -> pd.DataFrame:
        lang_dataset = self.get_multilingual_dataset(lang)

        lang_data = lang_dataset

        train_df = pd.DataFrame(lang_data["train"])
        val_df = pd.DataFrame(lang_data["validation"])
        test_df = pd.DataFrame(lang_data["test"])

        complete_lang_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        complete_lang_df["lang"] = lang

        if shuffle:
            complete_lang_df = complete_lang_df.sample(frac=1)

        return complete_lang_df

    def load_split_data(
        self, lang: str, train_ratio=0.8, val_ratio=0.1, shuffle: bool = False
    ):
        lang_df = self.load_data(lang, shuffle)

        # Calculate the indices for splitting
        total_len = len(lang_df)
        train_end = int(train_ratio * total_len)
        val_end = train_end + int(val_ratio * total_len)

        # Perform the split
        df_train = lang_df.iloc[:train_end]
        df_val = lang_df.iloc[train_end:val_end]
        df_test = lang_df.iloc[val_end:]

        return df_train, df_val, df_test

    def get_multilingual_dataset(self, lang: str) -> defaultdict:
        lang_dataset = load_dataset("unimelb-nlp/wikiann", name=lang)

        return lang_dataset
