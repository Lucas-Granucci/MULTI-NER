import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
from torch import Tensor
from typing import Tuple, List


class BertBilstmCrf(nn.Module):
    """
    BERT model with BiLSTM and CRF for sequence tagging
    """

    def __init__(self, num_tags: int):
        super(BertBilstmCrf, self).__init__()
        self.num_tags = num_tags
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,
        )
        
        # Fully connected layer
        self.fc = nn.Linear(in_features=256, out_features=num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        # BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output.last_hidden_state
        
        # BiLSTM output
        lstm_output, _ = self.lstm(sequence_output)
        
        # Emissions from fully connected layer
        emissions = self.fc(lstm_output)

        # Calculate loss using CRF
        loss = -self.crf(emissions, target_tags, mask=attention_mask.bool(), reduction="mean")
        return emissions, loss

    def decode(self, emissions: Tensor, attention_mask: Tensor) -> List[List[int]]:
        """
        Decode the emissions to get the predicted tags
        """
        return self.crf.decode(emissions, mask=attention_mask.bool())
