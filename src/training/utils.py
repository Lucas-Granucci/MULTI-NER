import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from typing import List


def calculate_f1_score(
    target_tags: torch.Tensor,
    pred_tags: torch.Tensor | List[List[int]],
    attention_mask: torch.Tensor,
) -> float:
    """
    Calculate the micro F1-score for sequence tagging from decoded emissions
    """
    # Handle CRF return type (list)
    if type(pred_tags) is list:
        pred_tags = [sequence + [0] * (256 - len(sequence)) for sequence in pred_tags]
        pred_tags = torch.tensor(pred_tags).to(target_tags.get_device())

    # Flatten batch results
    target_tags = target_tags.view(-1)
    pred_tags = pred_tags.view(-1)
    attention_mask = attention_mask.view(-1)

    # Filter out padding and special tokens
    target_tags = target_tags[attention_mask == 1]
    pred_tags = pred_tags[attention_mask == 1]

    f1_micro = f1_score(target_tags.cpu(), pred_tags.cpu(), average="micro")
    return f1_micro


class TrainConfig:
    """
    Training configuration class
    """

    def __init__(
        self,
        num_tags: int,
        batch_size: int,
        bert_learning_rate: float,
        lstm_learning_rate: float,
        crf_learning_rate: float,
        patience: int,
        epochs: int,
        device: torch.device,
    ):
        self.NUM_TAGS = num_tags
        self.BATCH_SIZE = batch_size
        self.BERT_LEARNING_RATE = bert_learning_rate
        self.LSTM_LEARNING_RATE = lstm_learning_rate
        self.CRF_LEARNING_RATE = crf_learning_rate
        self.PATIENCE = patience
        self.EPOCHS = epochs
        self.DEVICE = device


def setup_optimizer(model: torch.nn.Module, config: TrainConfig) -> optim.Optimizer:
    """
    Setups up optimizer for different models with different learning rates for each part
    """
    param_groups = []
    if hasattr(model, "bert"):
        param_groups.append(
            {
                "params": model.bert.parameters(),
                "lr": config.BERT_LEARNING_RATE,
            }
        )

    if hasattr(model, "lstm"):
        param_groups.append(
            {
                "params": model.lstm.parameters(),
                "lr": config.LSTM_LEARNING_RATE,
            }
        )

    if hasattr(model, "crf"):
        param_groups.append(
            {
                "params": model.crf.parameters(),
                "lr": config.CRF_LEARNING_RATE,
            }
        )

    optimizer = optim.Adam(param_groups, weight_decay=0.01)
    return optimizer
