from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertSoftmax(BertPreTrainedModel):
    def __init__(self, bert_config, num_labels, hidden_dropout_prob):
        super(BertSoftmax, self).__init__(bert_config)

        bert_model = BertModel(bert_config, add_pooling_layer=False)
        self.bert = bert_model.from_pretrained("third_party_weights/bert_base_chinese/")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]

        return logits
