import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from torch import Tensor
from typing import Tuple, List


class BertBilstmCrf(nn.Module):
    """
    BERT model with BiLSTM and CRF for sequence tagging
    """

    def __init__(self, num_tags):
        super(BertBilstmCrf, self).__init__()
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(256, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self, ids: Tensor, mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)

        # Calculate loss
        loss = -self.crf(emissions, target_tags, mask=mask.bool(), reduction="mean")
        return emissions, loss

    def decode(self, emissions: Tensor, mask: Tensor) -> List[List[int]]:
        """
        Decode the emissions to get the predicted tags
        """
        return self.crf.decode(emissions, mask=mask.bool())

    @staticmethod
    def get_model_type() -> str:
        """
        Get the type of the model
        """
        return "bert"
