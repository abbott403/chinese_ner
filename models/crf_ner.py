from torch import nn
from transformers import BertModel, BertPreTrainedModel
from models.layers.crf import CRF


class BertCrf(BertPreTrainedModel):
    def __init__(self, model_config, num_tags, dropout_rate):
        super(BertCrf, self).__init__(model_config)
        self.bert = BertModel(model_config, add_pooling_layer=False).from_pretrained("third_party_weights/bert_base_chinese/")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(model_config.hidden_size, num_tags)
        self.crf = CRF(num_tags, True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        else:
            outputs = self.crf.decode(outputs[0], attention_mask)
        return outputs  # (loss), scores
