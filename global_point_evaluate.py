import os
from train_config import global_point_config as configs
import torch
import json
from transformers import BertTokenizerFast
from models.global_point import GlobalPoint
from data_process.global_point_dataloader import DataCollate, GlobalPointDataset
from torch.utils.data import DataLoader
from utils.utils import decode_ent


def load_data(data_path):
    datas = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            datas.append(line)
    return datas


def data_generator(tokenizer):
    """
    读取数据，生成DataLoader。
    """
    predict_data_path = os.path.join("data/", "test.json")
    predict_data = load_data(predict_data_path)

    all_data = predict_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= 128, f'数据文本最大token数量{max_tok_num}超过预设128'
    max_seq_len = min(max_tok_num, 128)

    data_collate = DataCollate(tokenizer)

    predict_dataloader = DataLoader(GlobalPointDataset(predict_data), batch_size=32, shuffle=False, drop_last=False,
                                    collate_fn=lambda x: data_collate.generate_batch(x, max_seq_len, configs.ent2id,
                                                                                     data_type="predict"))
    return predict_dataloader


def predict(dataloader, model, device, tokenizer):
    predict_res = []

    model.eval()
    for batch_data in dataloader:
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
        batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                       batch_attention_mask.to(device),
                                                                       batch_token_type_ids.to(device))
        with torch.no_grad():
            batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        for ind in range(len(batch_samples)):
            gold_sample = batch_samples[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            pred_matrix = batch_logits[ind]
            labels = decode_ent(text, pred_matrix, tokenizer)
            predict_res.append({"id": text_id, "text": text, "label": labels})
    return predict_res


def main():
    ent_type_size = len(configs.ent2id)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained("third_party_weights/bert_base_chinese/", add_special_tokens=True,
                                                  do_lower_case=False)
    test_dataloader = data_generator(tokenizer)

    model = GlobalPoint(ent_type_size, 64)
    model.load_state_dict(torch.load("model_weights/gp_792.pt"))
    model = model.to(device)

    predict_res = predict(test_dataloader, model, device, tokenizer)

    save_path = "./predict_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        for item in predict_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
