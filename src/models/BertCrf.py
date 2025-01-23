import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from torch import Tensor
from typing import Tuple


class BertCrf(nn.Module):
    """
    BERT model with CRF decoder or sequence tagging
    """

    def __init__(self, num_tags: int):
        super(BertCrf, self).__init__()
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

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
        loss = -self.crf(emissions, target_tags, mask=mask.bool(), reduction="mean")
        return emissions, loss

    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """
        Decode the emissions to get the predicted tags
        """
        return self.crf.decode(emissions, mask=mask.bool())
