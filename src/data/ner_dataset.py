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

        origin_sentences = dataframe["tokens"].tolist()
        origin_labels = dataframe["ner_tags"].tolist()

        for line in origin_sentences:
            words = []
            word_lens = []

            for token in line:
                word_tokens = self.tokenizer.tokenize(token)
                words.append(word_tokens)
                word_lens.append(len(word_tokens))

            words = ['[CLS]'] + [item for sublist in words for item in sublist]

            # Calculate the starting index for each word
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])

            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))

        for tag in origin_labels:
            labels.append(tag) # Tags are already in ID format

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

        print("batch_data.shape: ", batch_data.shape)
        print("batch_label_starts.shape: ", batch_label_starts.shape)
        print("batch_labels.shape: ", batch_labels.shape)

        # Shift tensors to GPU if available
        batch_data = batch_data.to(self.device)
        batch_label_starts = batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]