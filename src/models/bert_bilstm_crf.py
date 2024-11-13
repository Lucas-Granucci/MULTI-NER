from torch import nn
from torchcrf import CRF
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bert.modeling_bert import *

class BertBiLstmCRF(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiLstmCRF, self).__init__(config)
        self.num_labels = config.num_labels

        # BERT layer
        self.bert = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased", config=config, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size = config.lstm_embedding_size, # TODO determine input size
            hidden_size = config.hidden_size // 2, # TODO determine hidden size
            batch_first = True,
            num_layers = 2,
            dropout = config.lstm_dropout_prob,  # TODO determine dropout prob
            bidirectional = True
        )

        # Linear layer for classification
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.crf = CRF(self.num_labels, batch_first=True)

        # Initialize weights
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        # IDs of tokens and indices marking the start of each token
        input_ids, input_token_starts = input_data

        # Padd data through BERT model
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        # Get the last hidden state
        sequence_output = outputs[0]

        # Extract token embeddings based on token start indices to align with original input tokens
        origin_sequence_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(sequence_output, input_token_starts)
        ]
        
        # Pad sequence outputs to the same length for batch processing
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)

        # Apply dropout for regularization
        padded_sequence_output = self.dropout(padded_sequence_output)

        # Pass through BiLSTM layer to capture contextual dependencies
        lstm_output, _ = self.bilstm(padded_sequence_output)

        # Project over label space using classifier layer
        logits = self.classifier(lstm_output)

        outputs = (logits,)

        # If labels are provided, comput CRF loss for training
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs
