import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from configs import global_point_config as configs
from data.load_ner_data import load_data_span_format


class GlobalPointDataset(Dataset):
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

    def generate_inputs(self, datas, ent2id):
        """
        生成喂入模型的数据
        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            ent2id (dict): ent到id的映射

        Returns:
            list: [(input_ids, attention_mask, token_type_ids, labels),(),()...]
        """
        ent_type_size = len(ent2id)
        batch_size = len(datas)

        batch_sentence = []
        for sample in datas:
            batch_sentence.append(sample['text'])
        batch_inputs = self.tokenizer(
            batch_sentence,
            padding=True,
            return_tensors="pt")
        max_seq_len = batch_inputs['input_ids'].shape[1]
        batch_label = np.zeros((batch_size, ent_type_size, max_seq_len, max_seq_len), dtype=int)

        for batch_idx, sample in enumerate(datas):
            input_data = self.tokenizer(sample["text"])

            for start, end, tag in sample["entity_list"]:
                token_start = input_data.char_to_token(start)
                token_end = input_data.char_to_token(end)
                # print(input_data.tokens())
                # print(input_data.tokens()[token_start: token_end+1])

                batch_label[batch_idx, ent2id[tag], token_start, token_end] = 1

        return batch_inputs["input_ids"], batch_inputs["token_type_ids"], batch_inputs["attention_mask"], \
            torch.tensor(batch_label)

    def generate_batch(self, batch_data, ent2id):
        input_ids, token_type_ids, attention_mask, batch_label = self.generate_inputs(batch_data, ent2id)
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []

        for sample in zip(input_ids, token_type_ids, attention_mask, batch_label):
            input_ids_list.append(sample[0])
            attention_mask_list.append(sample[1])
            token_type_ids_list.append(sample[2])
            labels_list.append(sample[3])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0)

        return batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels


def data_generator(tokenizer):
    train_data_path = os.path.join(configs.train_data_path, "train.json")
    dev_data_path = os.path.join(configs.train_data_path, "dev.json")

    train_data = load_data_span_format(train_data_path)
    dev_data = load_data_span_format(dev_data_path)
    all_data = train_data + dev_data

    max_token_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        max_token_num = max(max_token_num, len(tokens))

    assert max_token_num <= configs.max_len, f'数据文本最大token数量{max_token_num}超过预设{configs.max_len}'

    data_collate = DataCollate(tokenizer)

    train_dataset = GlobalPointDataset(train_data)
    dev_dataset = GlobalPointDataset(dev_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=configs.batch_size,
                                num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id),
                                persistent_workers=True, prefetch_factor=configs.batch_size // configs.num_work_load)

    if configs.is_ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, sampler=train_sampler,
                                      num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                      collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id),
                                      persistent_workers=True,
                                      prefetch_factor=configs.batch_size // configs.num_work_load)
        return train_dataloader, dev_dataloader, train_sampler
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                                      num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                      collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id),
                                      persistent_workers=True,
                                      prefetch_factor=configs.batch_size // configs.num_work_load)
        return train_dataloader, dev_dataloader


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("../third_party_weights/bert_base_chinese/",add_special_tokens=True,
                                                  do_lower_case=True)

    t_data, v_data = data_generator(tokenizer)
    batch_X = next(iter(t_data))
