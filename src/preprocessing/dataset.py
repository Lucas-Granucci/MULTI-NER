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

        self.CLS = [101]
        self.SEP = [102]
        self.VALUE_TOKEN = [0]
        self.MAX_LEN = 256

    def add_texts(self, new_texts: List[List[str]], new_tags: List[List[int]]):
        self.texts.extend(new_texts)
        self.tags.extend(new_tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Gets items and preprocesses it for training
        """
        texts = self.texts[index]
        tags = self.tags[index]

        # Tokenize
        ids = []
        target_tag = []

        for i, word in enumerate(texts):
            inputs = self.tokenizer.encode(word, add_special_tokens=False)

            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend(input_len * [tags[i]])

        # Resize for special tokens
        ids = ids[: self.MAX_LEN - 2]
        target_tag = target_tag[: self.MAX_LEN - 2]

        # Add special tokens
        ids = self.CLS + ids + self.SEP
        target_tags = self.VALUE_TOKEN + target_tag + self.VALUE_TOKEN

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        # Add padding if the input_len is small
        padding_len = self.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_len)
        target_tags = target_tags + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tags": torch.tensor(target_tags, dtype=torch.long),
        }
