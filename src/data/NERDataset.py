import torch
import numpy as np
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

class NERDataset(Dataset):
    def __init__(self, df, config, word_pad_idx=0, label_pad_idx=-100):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            config["tokenizer"]["bert_model"], 
            do_lower_case=True,
            truncation=True,
            max_length=512
        )
        self.dataset = self.preprocess(df)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config["model"]["device"]

    def preprocess(self, df):
        data = []
        sentences = []
        labels = []

        origin_sentences = df["tokens"].tolist()
        origin_labels = df["ner_tags"].tolist()

        for line in origin_sentences:
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        for tag in origin_labels:
            labels.append(tag) # Tags already in ID form
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

        batch_len = len(sentences)
        max_len = min(512, max([len(s[0]) for s in sentences]))  # Set max_len to 512

        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_attention_mask = []

        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0][:max_len]  # Truncate to max_len
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            #batch_label_starts.append(label_starts)

            # Fixed attention mask?
            attention_mask = np.zeros(max_len)
            attention_mask[:cur_len] = 1
            batch_attention_mask.append(attention_mask)

        #batch_label_starts = np.array(batch_label_starts)
        #batch_label_starts[:, 0] = 1

        batch_attention_mask = np.array(batch_attention_mask)

        batch_labels = self.label_pad_idx * np.ones((batch_len, max_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j][:max_len]  # Truncate to max_len

        batch_data = torch.tensor(batch_data, dtype=torch.long)
        #batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)

        batch_attention_mask = (batch_labels != -100).long()

        batch_data = batch_data.to(self.device)
        #batch_label_starts = batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)

        batch_attention_mask = batch_attention_mask.to(self.device)

        return [batch_data, batch_attention_mask, batch_labels]


# Usage example:
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
# dataset = NERDataset(words, labels, config)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)