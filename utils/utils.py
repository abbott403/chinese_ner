import torch
from torch import distributed as dist
import numpy as np
import random
import json
import argparse
from train_config import global_point_config as g_configs
from train_config import seq_config as s_configs
from train_config import span_config as p_configs


def set_random_seed(seed_value=0):
    # 1. 设置PyTorch随机数种子
    torch.manual_seed(seed_value)

    # 2. 如果有可用的GPU，则为其设置随机数种子
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 3. 设置Numpy随机数种子
    np.random.seed(seed_value)

    print("all seed is done")


def load_data(data_path):
    datas = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            datas.append(line["text"])
    return datas


def get_entity_bios(seq):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def global_decode_ent(text, pred_matrix, tokenizer, threshold=0):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}

    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = g_configs.id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]: ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})

    return ent_list


def softmax_decode_ent(text, pred_matrix, tokenizer):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    probabilities = torch.nn.functional.softmax(pred_matrix, dim=-1)[0].cpu().numpy().tolist()
    predictions = pred_matrix.argmax(dim=-1)[0].cpu().numpy().tolist()
    pred_label = []

    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = s_configs.id2ent[pred]
        if label != "O":
            label = label[2:]  # Remove the B- or I-
            start, end = token2char_span_mapping[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while idx + 1 < len(predictions) and s_configs.id2ent[predictions[idx + 1]] == f"I-{label}":
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = token2char_span_mapping[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            word = text[start:end]
            pred_label.append(
                {
                    "entity_label": label,
                    "score": score,
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )
        idx += 1

    return pred_label


def crf_decode_ent(text, pred_matrix, tokenizer):
    pred_label = []
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    labels = get_entity_bio(pred_matrix)
    for tag, token_start, token_end in labels:
        start, _ = token2char_span_mapping[token_start]
        _, end = token2char_span_mapping[token_end]

        word = text[start:end]
        pred_label.append(
            {
                "entity_label": tag,
                "word": word,
                "start": start,
                "end": end,
            }
        )


def span_extract_item(start_logits, end_logits):
    res = []
    start_pred = torch.argmax(start_logits, -1).cpu().tolist()   # batch_size, maxlen
    end_pred = torch.argmax(end_logits, -1).cpu().tolist()

    for batch_idx in range(len(start_pred)):
        start_sample = start_pred[batch_idx]
        end_sample = end_pred[batch_idx]
        batch_res = []

        for i, s_l in enumerate(start_sample):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_sample[i:]):
                if s_l == e_l:
                    batch_res.append([p_configs.id2ent[s_l], i, i + j])
                    break

        res.append(batch_res)
    return res


def get_parse_args():
    parser = argparse.ArgumentParser(description="training global pointer model")
    parser.add_argument("--local_rank", default=-1)

    args = parser.parse_args()
    return args


def ddp_reduce_mean(loss_data, nprocs):
    rt = loss_data.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def freeze_weight(model, unfreeze_layer):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layer:
            if ele in name:
                param.requires_grad = True
                break


def filter_data(all_input_data):
    res = []
    empty_data = []
    for item in all_input_data:
        if item["entity_list"]:
            res.append(item)
        else:
            empty_data.append(item)

    sample_num = 1000
    sampled_list = random.sample(empty_data, sample_num)
    res.extend(sampled_list)

    return res
