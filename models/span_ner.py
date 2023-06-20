import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn import functional as F


class BertSpan(BertPreTrainedModel):
    def __init__(self, model_config, num_labels, hidden_dropout_prob, soft_label):
        super(BertSpan, self).__init__(model_config)
        self.soft_label = soft_label
        self.num_labels = num_labels

        self.bert = BertModel(model_config, add_pooling_layer=False).from_pretrained("third_party_weights/bert_base_chinese/")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.start_fc = nn.Linear(model_config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = nn.Sequential(
                nn.Linear(model_config.hidden_size + self.num_labels, model_config.hidden_size + self.num_labels),
                nn.Tanh(),
                nn.LayerNorm(model_config.hidden_size + self.num_labels),
                nn.Linear(model_config.hidden_size + self.num_labels, self.num_labels)
            )
        else:
            self.end_fc = nn.Sequential(
                nn.Linear(model_config.hidden_size + 1, model_config.hidden_size + 1),
                nn.Tanh(),
                nn.LayerNorm(model_config.hidden_size + 1),
                nn.Linear(model_config.hidden_size + 1, self.num_labels)
            )

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs[0])
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None:
            if self.soft_label:
                batch_size, seq_len = input_ids.size()
                label_logits = torch.zeros((batch_size, seq_len, self.num_labels))
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)   # batch_size, max_len, 1
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_input = torch.cat([sequence_output, label_logits], dim=-1)
        end_logits = self.end_fc(end_input)
        outputs = (start_logits, end_logits,)

        return outputs
