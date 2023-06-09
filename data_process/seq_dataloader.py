import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from train_config import seq_config as configs
from data_process.load_ner_data import load_data


class SeqDataset(Dataset):
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
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """
        batch_sentence = []
        for sample in datas:
            batch_sentence.append(sample['text'])
        batch_inputs = self.tokenizer(
            batch_sentence,
            padding=True,
            return_tensors="pt")
        batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)

        for batch_idx, sample in enumerate(datas):
            input_data = self.tokenizer(sample["text"])
            batch_label[batch_idx][0] = -100
            batch_label[batch_idx][len(input_data.tokens()) - 1:] = -100

            for start, end, tag, _ in sample["entity_list"]:
                token_start = input_data.char_to_token(start)
                token_end = input_data.char_to_token(end)
                # print(input_data.tokens())
                # print(input_data.tokens()[token_start: token_end+1])

                batch_label[batch_idx][token_start] = ent2id[tag]
                batch_label[batch_idx][token_start + 1: token_end + 1] = ent2id[tag] + 1

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
    train_data_path = os.path.join(configs.train_data_path, "example.train")
    dev_data_path = os.path.join(configs.train_data_path, "example.dev")

    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)

    data_collate = DataCollate(tokenizer)
    train_dataloader = DataLoader(SeqDataset(train_data), batch_size=configs.batch_size, shuffle=True,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))
    valid_dataloader = DataLoader(SeqDataset(dev_data), batch_size=configs.batch_size,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    return train_dataloader, valid_dataloader


def data_generator_ddp(tokenizer):
    train_data_path = os.path.join(configs.train_data_path, "example.train")
    dev_data_path = os.path.join(configs.train_data_path, "example.dev")

    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)

    data_collate = DataCollate(tokenizer)

    train_dataset = SeqDataset(train_data)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, sampler=train_sampler,
                                  num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    dev_dataset = SeqDataset(dev_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=configs.batch_size,
                                num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    return train_dataloader, dev_dataloader, train_sampler


if __name__ == "__main__":
    from transformers import BertTokenizerFast
    train_data_path = os.path.join("../", configs.train_data_path, "example.train")
    train_data = load_data(train_data_path)

    tokenizer = BertTokenizerFast.from_pretrained("../third_party_weights/bert_base_chinese/", add_special_tokens=True,
                                                  do_lower_case=False)
    data_collate = DataCollate(tokenizer)
    train_dataloader = DataLoader(SeqDataset(train_data), batch_size=2, shuffle=True,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    batch_X = next(iter(train_dataloader))
    print(batch_X)
