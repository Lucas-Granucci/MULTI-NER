import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset


class NER_Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["bert_model"], do_lower_case=True)
        self.label2id = config["data"]["label2id"]
        self.id2label = {_id : _label for _label, _id in self.label2id.items()}
        self.dataset = self.preprocess(dataframe)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config["model"]["device"]

    def preprocess(self, dataframe: pd.DataFrame):
        data = []
        sentences = []
        labels = []

        # Process each row in dataframe
        for _, row in dataframe.iterrows():
            words = []
            word_lens = []

            # Tokenize each word and calculate token length
            for token in row["tokens"]:
                tokenized = self.tokenizer.tokenize(token)
                words.append(tokenized)
                word_lens.append(len(token))

            # Flatten tokenized words and add [CLS] at the beginning
            tokens = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])

            # Convert tokens to token ids
            sentence_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            sentences.append((sentence_ids, token_start_idxs))

            # Get ner tags and add to labels
            label_ids = row["ner_tags"]
            labels.append(label_ids)

        # Add all processed sentences and labels to data
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))

        return data

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        pass
