import torch
from transformers import BertTokenizerFast
from config import BaseConfig

class NERDataset:
    def __init__(self, texts, tags, include_sentence = False):
        self.texts = texts
        self.tags = tags

        self.tokenizer = BertTokenizerFast.from_pretrained(
            "google-bert/bert-base-multilingual-cased", do_lower_case = True
        )

        self.CLS_TOKEN = [101]
        self.SEP_TOKEN = [102]
        self.PAD_TOKEN = [0]
        self.MAX_LEN = BaseConfig.MAX_SEQ_LEN

        # Determines if the original sentence is returned for each batch
        self.include_sentence = include_sentence

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        tags = self.tags[index]

        token_ids = []
        target_tags = []
        for i, word in enumerate(text):
            word_ids = self.tokenizer.encode(word, add_special_tokens = False)
            token_ids.extend(word_ids)
            target_tags.extend(len(word_ids) * [tags[i]])

        # Resize for special tokens
        token_ids = token_ids[:self.MAX_LEN - 2]
        target_tags = target_tags[:self.MAX_LEN - 2]

        # Add special tokens
        token_ids = self.CLS_TOKEN + token_ids + self.SEP_TOKEN
        target_tags = self.PAD_TOKEN + target_tags + self.PAD_TOKEN

        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        # Add padding to make sure all inputs are the same size
        padding_len = self.MAX_LEN - len(token_ids)
        token_ids += [0] * padding_len
        target_tags += [0] * padding_len
        attention_mask += [0] * padding_len
        token_type_ids += [0] * padding_len

        if self.include_sentence:
            return {
                "input_ids": torch.tensor(token_ids, dtype = torch.long),
                "target_tags": torch.tensor(target_tags, dtype = torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype = torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
                "orginal_text": " ".join(text)
            }

        return {
            "input_ids": torch.tensor(token_ids, dtype = torch.long),
            "target_tags": torch.tensor(target_tags, dtype = torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long)
        }
