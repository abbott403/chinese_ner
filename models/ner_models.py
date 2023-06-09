import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from utils.all_loss import FocalLoss, LabelSmoothingCrossEntropy
from models.layers.crf import CRF


class GlobalPoint(nn.Module):
    def __init__(self, ent_type_size, inner_dim, device="cuda", rope=True):
        super(GlobalPoint, self).__init__()
        self.encoder = BertModel.from_pretrained("./third_party_weights/bert_base_chinese/")
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.rope = rope
        self.device = device

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.rope:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            # repeat_interleave是将张量中的元素沿某一维度复制n次，即复制后的张量沿该维度相邻的n个元素是相同的。
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos

            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(b
        # atch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class BertSoftmax(BertPreTrainedModel):
    def __init__(self, model_config):
        super(BertSoftmax, self).__init__(model_config)
        self.num_labels = model_config.num_labels
        self.loss_type = model_config.loss_type

        self.bert = BertModel(model_config)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(model_config.hidden_size, model_config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.loss_type == "lsr":
                loss_fun = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == "focal":
                loss_fun = FocalLoss(ignore_index=0)
            else:
                loss_fun = CrossEntropyLoss(ignore_index=0)

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fun(active_logits, active_labels)
            else:
                loss = loss_fun(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs


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
