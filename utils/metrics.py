import torch
from sklearn.metrics import f1_score

from config import BaseConfig

def calculate_f1(target_tags, pred_tags, attention_mask):

    if isinstance(pred_tags, list):
        pred_tags = [sequence + [0] * (BaseConfig.MAX_SEQ_LEN - len(sequence)) for sequence in pred_tags]
        pred_tags = torch.tensor(pred_tags).to(BaseConfig.DEVICE)

    # Flatten batch results
    target_tags = target_tags.view(-1)
    pred_tags = pred_tags.view(-1)
    attention_mask = attention_mask.view(-1)

    # Filter out padding and special tokens
    target_tags = target_tags[attention_mask == 1]
    pred_tags = pred_tags[attention_mask == 1]

    f1_micro = f1_score(target_tags.cpu(), pred_tags.cpu(), average="micro")
    return f1_micro