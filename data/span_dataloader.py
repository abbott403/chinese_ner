import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from configs import span_config as configs
from data.load_ner_data import load_data
from utils.utils import filter_data


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

    def generate_inputs(self, datas, ent2id):
        """
        生成喂入模型的数据
        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            ent2id (dict): ent到id的映射

        Returns:
            list: [(input_ids, attention_mask, token_type_ids, labels),(),()...]
        """
        batch_size = len(datas)
        entity_list = []

        batch_sentence = []
        for sample in datas:
            batch_sentence.append(sample['text'])

        batch_inputs = self.tokenizer(batch_sentence, padding=True, return_tensors="pt")
        max_seq_len = batch_inputs['input_ids'].shape[1]
        start_ids = np.zeros((batch_size, max_seq_len))
        end_ids = np.zeros((batch_size, max_seq_len))

        for batch_idx, sample in enumerate(datas):
            input_data = self.tokenizer(sample["text"])
            start_ids[batch_idx][0] = -100
            start_ids[batch_idx][len(input_data.tokens()) - 1:] = -100
            end_ids[batch_idx][0] = -100
            end_ids[batch_idx][len(input_data.tokens()) - 1:] = -100
            sample_entity_list = []

            for start, end, tag, _ in sample["entity_list"]:
                token_start = input_data.char_to_token(start)
                token_end = input_data.char_to_token(end)

                start_ids[batch_idx][token_start] = ent2id[tag[2:]]
                end_ids[batch_idx][token_end] = ent2id[tag[2:]]
                sample_entity_list.append([tag[2:], start, end])

            entity_list.append(sample_entity_list)

        return batch_inputs["input_ids"], batch_inputs["token_type_ids"], batch_inputs["attention_mask"], \
               torch.tensor(start_ids, dtype=torch.int64), torch.tensor(end_ids, dtype=torch.int64), entity_list

    def generate_batch(self, batch_data, ent2id):
        input_ids, token_type_ids, attention_mask, start_ids, end_ids, entity_list = self.generate_inputs(batch_data, ent2id)
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        start_ids_list = []
        end_ids_list = []

        for sample in zip(input_ids, token_type_ids, attention_mask, start_ids, end_ids):
            input_ids_list.append(sample[0])
            token_type_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            start_ids_list.append(sample[3])
            end_ids_list.append(sample[4])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_start_ids = torch.stack(start_ids_list, dim=0)
        batch_end_ids = torch.stack(end_ids_list, dim=0)

        return batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_start_ids, batch_end_ids, entity_list


def data_generator(tokenizer):
    train_data_path = os.path.join(configs.train_data_path, "train.txt")
    dev_data_path = os.path.join(configs.train_data_path, "test.txt")

    train_data = load_data(train_data_path)
    train_data = filter_data(train_data)
    dev_data = load_data(dev_data_path)
    # all_data = train_data + dev_data
    #
    # max_token_num = 0
    # for sample in all_data:
    #     tokens = tokenizer(sample["text"])["input_ids"]
    #     max_token_num = max(max_token_num, len(tokens))
    #
    # assert max_token_num <= configs.max_len, f'数据文本最大token数量{max_token_num}超过预设{configs.max_len}'

    data_collate = DataCollate(tokenizer)
    train_dataloader = DataLoader(SpanDataset(train_data), batch_size=configs.batch_size, shuffle=True,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id),
                                  persistent_workers=True)
    valid_dataloader = DataLoader(SpanDataset(dev_data), batch_size=configs.batch_size,
                                  num_workers=configs.num_work_load, drop_last=False,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id),
                                  persistent_workers=True)

    return train_dataloader, valid_dataloader


def data_generator_ddp(tokenizer):
    train_data_path = os.path.join(configs.train_data_path, "train.txt")
    dev_data_path = os.path.join(configs.train_data_path, "test.txt")

    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)

    data_collate = DataCollate(tokenizer)

    train_dataset = SpanDataset(train_data)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, sampler=train_sampler,
                                  num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                  collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    dev_dataset = SpanDataset(dev_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=configs.batch_size,
                                num_workers=configs.num_work_load, drop_last=False, pin_memory=True,
                                collate_fn=lambda x: data_collate.generate_batch(x, configs.ent2id))

    return train_dataloader, dev_dataloader, train_sampler


if __name__ == "__main__":
    from transformers import BertTokenizerFast

    train_path = os.path.join("../", configs.train_data_path, "train.txt")
    train_datas = load_data(train_path)

    test_tokenizer = BertTokenizerFast.from_pretrained("../third_party_weights/bert_base_chinese/",
                                                       add_special_tokens=True,
                                                       do_lower_case=False)
    data_coll = DataCollate(test_tokenizer)
    train_loader = DataLoader(SpanDataset(train_datas), batch_size=2, shuffle=True,
                              num_workers=1, drop_last=False,
                              collate_fn=lambda x: data_coll.generate_batch(x, configs.ent2id))

    batch_X = next(iter(train_loader))
    print(batch_X)
