import torch.nn as nn
from transformers import XLMRobertaModel


class XLMRoBERTaBiLSTM(nn.Module):
    def __init__(self, num_tags, label_pad_idx=-100):
        super(XLMRoBERTaBiLSTM, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.lstm = nn.LSTM(
            self.roberta.config.hidden_size,
            128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(256, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=label_pad_idx)
        self.label_pad_idx = label_pad_idx
        self.num_tags = num_tags

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        emissions = self.fc(lstm_output)
        return emissions

    def loss(self, emissions, tags):
        emissions = emissions.view(-1, self.num_tags)
        tags = tags.view(-1)
        return self.loss_fn(emissions, tags)

    def decode(self, emissions):
        return emissions.argmax(dim=-1)

    def get_model_type(self):
        return "roberta"
