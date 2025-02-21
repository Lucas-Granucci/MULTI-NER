import torch
from transformers import BertTokenizerFast
from typing import List, Dict


class NERDataset:
    """
    Dataset for named entity recognition
    """

    def __init__(self, texts: List[List[str]], tags: List[List[int]]):
        self.texts = texts
        self.tags = tags

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-multilingual-cased", do_lower_case=True
        )

        self.CLS_TOKEN = [101]
        self.SEP_TOKEN = [102]
        self.PAD_TOKEN = [0]
        self.MAX_LEN = 101

    def add_texts(self, new_texts: List[List[str]], new_tags: List[List[int]]):
        self.texts.extend(new_texts)
        self.tags.extend(new_tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Gets items and preprocesses it for training
        """
        text = self.texts[index]
        tags = self.tags[index]

        # Tokenize
        token_ids = []
        target_tags = []

        for i, word in enumerate(text):
            if i >= len(tags):
                break
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(word_ids)
            target_tags.extend(len(word_ids) * [tags[i]])

        # Resize for special tokens
        token_ids = token_ids[: self.MAX_LEN - 2]
        target_tags = target_tags[: self.MAX_LEN - 2]

        # Add special tokens
        token_ids = self.CLS_TOKEN + token_ids + self.SEP_TOKEN
        target_tags = self.PAD_TOKEN + target_tags + self.PAD_TOKEN

        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        # Add padding if the input length is small
        padding_len = self.MAX_LEN - len(token_ids)
        token_ids += [0] * padding_len
        target_tags += [0] * padding_len
        attention_mask += [0] * padding_len
        token_type_ids += [0] * padding_len

        return {
            "ids": torch.tensor(token_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tags": torch.tensor(target_tags, dtype=torch.long),
        }


class UNERDataset:
    """
    Unlabeled dataset for named entity recognition
    """

    def __init__(self, texts: List[List[str]]):
        self.texts = texts

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-multilingual-cased", do_lower_case=True
        )

        self.CLS_TOKEN = [101]
        self.SEP_TOKEN = [102]
        self.PAD_TOKEN = [0]
        self.MAX_LEN = 101

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Gets items and preprocesses it for training
        """
        text = self.texts[index]
        word_count = len(text)

        # Tokenize
        token_ids = []

        for word in text:
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(word_ids)

        # Resize for special tokens
        token_ids = token_ids[: self.MAX_LEN - 2]

        # Add special tokens
        token_ids = self.CLS_TOKEN + token_ids + self.SEP_TOKEN

        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        # Add padding if the input length is small
        padding_len = self.MAX_LEN - len(token_ids)
        token_ids += [0] * padding_len
        attention_mask += [0] * padding_len
        token_type_ids += [0] * padding_len

        # Fake target tags
        target_tags = [0] * len(token_type_ids)

        return {
            "ids": torch.tensor(token_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_tags": torch.tensor(target_tags, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "sentence_length": word_count,
            "sentence": " ".join(text),
        }
