from torch import nn
from transformers import BertModel, BertPreTrainedModel
import torch
from torch.nn import functional as F
from models.layers.crf import CRF


class BertCrf(BertPreTrainedModel):
    def __init__(self, model_config, num_tags, dropout_rate):
        super(BertCrf, self).__init__(model_config)
        self.bert = BertModel(model_config, add_pooling_layer=False).from_pretrained("")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(model_config.hidden_size, num_tags)
        self.crf = CRF(num_tags, True)
        # self.init_weights()

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


# -------------------Softmax-----------------------------
class BertSoftmax(BertPreTrainedModel):
    def __init__(self, bert_config, num_labels, hidden_dropout_prob):
        super(BertSoftmax, self).__init__(bert_config)

        self.bert = BertModel(bert_config, add_pooling_layer=False).from_pretrained("")
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        # sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        # outputs = (logits,) + outputs[2:]

        return logits


# -------------------Global Point-----------------------------
class GlobalPoint(nn.Module):
    def __init__(self, ent_type_size, inner_dim, device="cuda", rope=True):
        super(GlobalPoint, self).__init__()
        self.encoder = BertModel.from_pretrained("./third_party_weights/bert_base_chinese/")
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size*self.inner_dim*2)

        self.rope = rope
        self.device = device

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim//2, dtype=torch.float)
        indices = torch.pow(10000, -2*indices/output_dim)
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
    # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size,
                                 self.ent_type_size * 2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        logits = self.add_mask_tril(logits, mask=attention_mask)
        return logits


# --------------------Span-------------------------------
class BertSpan(BertPreTrainedModel):
    def __init__(self, model_config, num_labels, hidden_dropout_prob, soft_label):
        super(BertSpan, self).__init__(model_config)
        self.soft_label = soft_label
        self.num_labels = num_labels

        self.bert = BertModel(model_config, add_pooling_layer=False).from_pretrained("")
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

        # self.init_weights()

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
