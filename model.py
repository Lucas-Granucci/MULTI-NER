import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

class BertBilstmCrf(nn.Module):
    def __init__(self, num_tags):
        super(BertBilstmCrf, self).__init__()

        # Define model layers
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.lstm = nn.LSTM(
            input_size = self.bert.config.hidden_size,
            hidden_size = 128,
            num_layers = 2,
            bidirectional = True,
            batch_first = True,
            dropout = 0.3
        )
        self.fc = nn.Linear(in_features = 256, out_features = num_tags)
        self.crf = CRF(num_tags, batch_first = True)

    @torch.autocast(device_type="cuda")
    def forward(self, input_ids, target_tags, attention_mask, token_type_ids):
        # Pass inputs through layers
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)

        loss = -self.crf(emissions, target_tags, mask = attention_mask.bool(), reduction = "mean")
        return emissions, loss

    def decode(self, emissions, attention_mask):
        return self.crf.decode(emissions, mask = attention_mask.bool())