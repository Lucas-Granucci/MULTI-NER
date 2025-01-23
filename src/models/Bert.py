import torch
import torch.nn as nn
from transformers import BertModel
from torch import Tensor
from typing import Tuple


class Bert(nn.Module):
    """
    BERT model for sequence tagging
    """

    def __init__(self, num_tags: int):
        super(Bert, self).__init__()
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, ids: Tensor, mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
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
