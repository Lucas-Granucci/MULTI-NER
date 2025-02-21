import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from torch import Tensor
from typing import Tuple


class BertCrf(nn.Module):
    """
    BERT model with CRF decoder for sequence tagging
    """

    def __init__(self, num_tags: int):
        super(BertCrf, self).__init__()
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        # Get BERT outputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output.last_hidden_state
        
        # Compute emissions
        emissions = self.fc(sequence_output)

        # Calculate loss using CRF
        loss = -self.crf(emissions, target_tags, mask=attention_mask.bool(), reduction="mean")
        return emissions, loss

    def decode(self, emissions: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Decode the emissions to get the predicted tags
        """
        return self.crf.decode(emissions, mask=attention_mask.bool())
