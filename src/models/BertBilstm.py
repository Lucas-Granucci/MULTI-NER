import torch
import torch.nn as nn
from transformers import BertModel
from torch import Tensor
from typing import Tuple


class BertBilstm(nn.Module):
    """
    BERT model with BiLSTM for sequence tagging
    """

    def __init__(self, num_tags: int):
        super(BertBilstm, self).__init__()
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
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, target_tags: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model
        """
        # Get BERT outputs
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output.last_hidden_state
        
        # Pass through BiLSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # Fully connected layer
        emissions = self.fc(lstm_output)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        active_loss = attention_mask.view(-1) == 1
        active_logits = emissions.view(-1, self.num_tags)
        active_labels = torch.where(
            active_loss,
            target_tags.view(-1),
            torch.tensor(criterion.ignore_index).type_as(target_tags),
        )
        loss = criterion(active_logits, active_labels)
        
        return emissions, loss

    def decode(self, emissions: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Decode the emissions to get the predicted tags
        """
        return emissions.argmax(dim=-1)
