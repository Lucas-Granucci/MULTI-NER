import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BERTBiLSTMCRF(nn.Module):
    def __init__(self, num_tags, label_pad_idx=-100):
        super(BERTBiLSTMCRF, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(256, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.num_tags = num_tags
        self.label_pad_idx = label_pad_idx

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)
        return emissions

    def loss(self, emissions, tags, mask):

        adjusted_tags = torch.where(tags == -100, torch.tensor(0, device=tags.device), tags)

        return -self.crf(emissions, adjusted_tags, mask=mask)

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)
