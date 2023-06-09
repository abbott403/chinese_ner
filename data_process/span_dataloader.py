import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from train_config import span_config as configs
from load_ner_data import load_data


class SpanDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataCollate:
    def __init__(self, tokenizer, add_special_tokens=True):
        super(DataCollate, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        ent2token_spans = []
        input_data = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = input_data["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            start, end = ent_span[0], ent_span[1]
            ent = text[start:end+1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 寻找ent的token_span
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]

            token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间
            token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1]-1 == ent_span[1], token_end_indexs))

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)

        return ent2token_spans

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """
        生成喂入模型的数据
        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        all_inputs = []
        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            start_ids = None
            end_ids = None
            if data_type != "predict":
                ent2token_spans = self.get_ent2token_spans(sample["text"], sample["entity_list"])
                start_ids = np.zeros(max_seq_len)
                end_ids = np.zeros(max_seq_len)
                for start, end, label in ent2token_spans:
                    start_ids[start] = ent2id[label]
                    end_ids[end] = ent2id[label]

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if start_ids is not None:
                start_ids = torch.tensor(start_ids).long()
                end_ids = torch.tensor(end_ids).long()

            sample_input = (sample, input_ids, attention_mask, token_type_ids, start_ids, end_ids)

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        start_ids_list = []
        end_ids_list = []

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "predict":
                start_ids_list.append(sample[4])
                end_ids_list.append(sample[5])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_start_ids = torch.stack(start_ids_list, dim=0) if data_type != "predict" else None
        batch_end_ids = torch.stack(end_ids_list, dim=0) if data_type != "predict" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_start_ids, batch_end_ids


def data_generator(tokenizer):
    train_data_path = os.path.join(configs.train_data_path, "train.json")
    dev_data_path = os.path.join(configs.train_data_path, "dev.json")

    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)
    all_data = train_data + dev_data

    max_token_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        max_token_num = max(max_token_num, len(tokens))

    assert max_token_num <= configs.max_len, f'数据文本最大token数量{max_token_num}超过预设{configs.max_len}'
    max_seq_len = min(max_token_num, configs.max_len)

    data_collate = DataCollate(tokenizer)
    train_dataloader = DataLoader(SpanDataset(train_data), batch_size=configs.batch_size, shuffle=True,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, max_seq_len, configs.ent2id))
    valid_dataloader = DataLoader(SpanDataset(dev_data), batch_size=configs.batch_size,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, max_seq_len, configs.ent2id))

    return train_dataloader, valid_dataloader
