import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from datasets import DatasetDict
from datasets import load_dataset

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class WikiANN_Dataloader:

    def __init__(self):
        self.dataset = None

    def load_data(self, langs: list, shuffle: bool = False) -> pd.DataFrame:

        wikiann_dataset = self.get_multilingual_dataset(langs)
        lang_dataframes = []

        for lang in langs:
            lang_data = wikiann_dataset[lang]

            train_df = pd.DataFrame(lang_data["train"])
            val_df = pd.DataFrame(lang_data["validation"])
            test_df = pd.DataFrame(lang_data["test"])

            complete_lang_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            complete_lang_df["lang"] = lang

            lang_dataframes.append(complete_lang_df)

        multilingual_df = pd.concat(lang_dataframes)

        if shuffle:
            multilingual_df = multilingual_df.sample(frac=1)

        return multilingual_df

    def load_split_data(self, langs: list, train_ratio=0.8, val_ratio=0.1, shuffle: bool = False):

        multilingual_df = self.load_data(langs, shuffle)
        multilingual_df = multilingual_df.sample(frac=1, random_state=42)

        # Calculate the indices for splitting
        total_len = len(multilingual_df)
        train_end = int(train_ratio * total_len)
        val_end = train_end + int(val_ratio * total_len)

        # Perform the split
        df_train, df_val, df_test = np.split(multilingual_df, [train_end, val_end])

        return df_train, df_val, df_test

    def get_multilingual_dataset(self, langs: list) -> defaultdict:

        wikiann_dataset = defaultdict(DatasetDict)

        for lang in langs:
            ds = load_dataset("unimelb-nlp/wikiann", name=lang)
            wikiann_dataset[lang] = ds

        return wikiann_dataset
