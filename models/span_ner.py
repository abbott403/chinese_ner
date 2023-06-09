import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from utils.all_loss import FocalLoss, LabelSmoothingCrossEntropy


class BertSpan(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BertSpan, self).__init__(model_config)
        self.soft_label = model_config.soft_label
        self.num_labels = model_config.num_labels
        self.loss_type = model_config.loss_type

        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs[0])
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()

            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs