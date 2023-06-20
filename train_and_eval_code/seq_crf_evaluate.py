import os
from train_config import seq_config as configs
import torch
import json
from transformers import BertTokenizerFast, BertConfig
from models.crf_ner import BertCrf
from data_process.seq_dataloader import SeqDataset
from torch.utils.data import DataLoader
from utils.utils import crf_decode_ent, load_data


class DataCollate:
    def __init__(self, tokenizer, add_special_tokens=True):
        super(DataCollate, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def generate_batch(self, batch_data):
        batch_inputs = self.tokenizer(batch_data, padding=True, return_tensors="pt")
        input_ids, token_type_ids, attention_mask = batch_inputs["input_ids"], batch_inputs["token_type_ids"], \
                                                    batch_inputs["attention_mask"]
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        for sample in zip(input_ids, token_type_ids, attention_mask):
            input_ids_list.append(sample[0])
            attention_mask_list.append(sample[1])
            token_type_ids_list.append(sample[2])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        return batch_data, batch_input_ids, batch_attention_mask, batch_token_type_ids


def data_generator(tokenizer):
    """
    读取数据，生成DataLoader。
    """
    predict_data_path = os.path.join("data/", "test.json")
    predict_data = load_data(predict_data_path)

    data_collate = DataCollate(tokenizer)

    predict_dataloader = DataLoader(SeqDataset(predict_data), batch_size=32, shuffle=False, drop_last=False,
                                    collate_fn=lambda x: data_collate.generate_batch(x))
    return predict_dataloader


def predict(dataloader, model, device, tokenizer):
    predict_res = []

    model.eval()
    for batch_data in dataloader:
        batch_sample, batch_input_ids, batch_attention_mask, batch_token_type_ids = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                       batch_attention_mask.to(device),
                                                                       batch_token_type_ids.to(device))
        with torch.no_grad():
            batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        for ind in range(len(batch_sample)):
            text = batch_sample[ind]
            pred_matrix = batch_logits[ind]
            labels = crf_decode_ent(text, pred_matrix, tokenizer)
            predict_res.append({"text": text, "label": labels})
    return predict_res


def main():
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained("third_party_weights/bert_base_chinese/", add_special_tokens=True,
                                                  do_lower_case=False)
    test_dataloader = data_generator(tokenizer)

    bert_config = BertConfig.from_pretrained(configs.pretrained_model_path)
    model = BertCrf(bert_config, ent_type_size, configs.dropout_rate)
    model.load_state_dict(torch.load("model_weights/XXX"))
    model = model.to(device)

    predict_res = predict(test_dataloader, model, device, tokenizer)

    save_path = "./predict_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        for item in predict_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
