from torch import nn
from transformers import BertModel, BertPreTrainedModel
from models.layers.crf import CRF


class BertCrf(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BertCrf, self).__init__(model_config)
        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(model_config.hidden_size, model_config.num_labels)
        self.crf = CRF(model_config.num_labels, True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores
