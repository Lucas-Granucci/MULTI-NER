import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset


class NER_Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["bert_model"], do_lower_case=True)
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
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # Batch length
        batch_len = len(sentences)

        # Compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0
        
        # Padding data
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # Padding and aligning
        for j in range(batch_len):
            current_len = len(sentences[j][0])
            batch_data[j][:current_len] = sentences[j][0]

            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)

            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # Padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            current_tags_len = len(labels[j])
            batch_labels[j][:current_tags_len] = labels[j]

        # Convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # Shift tensors to GPU if available
        batch_data = batch_data.to(self.device)
        batch_label_starts = batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]