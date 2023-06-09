from torch import nn
from transformers import BertModel


class BertSoftmax(nn.Module):
    def __init__(self, num_labels, hidden_dropout_prob):
        super(BertSoftmax, self).__init__()

        self.bert = BertModel.from_pretrained("./third_party_weights/bert_base_chinese/")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]

        return logits
