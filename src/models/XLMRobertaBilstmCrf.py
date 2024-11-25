import torch.nn as nn
from transformers import XLMRobertaModel
from torchcrf import CRF


class XLMRoBERTaBiLSTMCRF(nn.Module):
    def __init__(self, num_tags, label_pad_idx=-100):
        super(XLMRoBERTaBiLSTMCRF, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.lstm = nn.LSTM(
            self.roberta.config.hidden_size,
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
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask)

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)
