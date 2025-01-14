import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from torch import Tensor
from typing import Tuple


class XLMRoberta(nn.Module):
    """
    XLM-Roberta model for sequence tagging
    """

    def __init__(self, num_tags: int):
        super(XLMRoberta, self).__init__()
        self.num_tags = num_tags
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, ids: Tensor, mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        output = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = output.last_hidden_state
        emissions = self.fc(sequence_output)

        # Calculate loss
        Critirion_Loss = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = emissions.view(-1, self.num_tags)
        active_labels = torch.where(
            active_loss,
            target_tags.view(-1),
            torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags),
        )
        loss = Critirion_Loss(active_logits, active_labels)
        return emissions, loss

    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """
        Decode the emissions to get the predicted tags
        """
        return emissions.argmax(dim=-1)

    @staticmethod
    def get_model_type() -> str:
        """
        Get the type of the model"""
        return "roberta"
