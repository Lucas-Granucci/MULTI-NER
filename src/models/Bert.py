import torch.nn as nn
from transformers import BertModel


class BERT(nn.Module):
    def __init__(self, num_tags, label_pad_idx=-100):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=label_pad_idx)
        self.label_pad_idx = label_pad_idx
        self.num_tags = num_tags

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        emissions = self.fc(sequence_output)
        return emissions

    def loss(self, emissions, tags):
        emissions = emissions.view(-1, self.num_tags)
        tags = tags.view(-1)
        return self.loss_fn(emissions, tags)

    def decode(self, emissions):
        return emissions.argmax(dim=-1)
