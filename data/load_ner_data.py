import json
import re


def load_data(data_path):
    datas = []
    with open(data_path, encoding="utf8") as f:
        for line in f.read().split("\n\n"):
            if not line:
                break

            sentence, labels = '', []
            for i, item in enumerate(line.split('\n')):
                char, tag = item.split()
                sentence += char
                if tag.startswith('B'):
                    labels.append([i, i, tag, char])
                elif tag.startswith('I'):
                    labels[-1][1] = i
                    labels[-1][3] += char

            if len(sentence) > 200:
                continue

            datas.append({"text": sentence, "entity_list": labels})
    return datas


def load_data_span_format(data_path):
    datas = []
    with open(data_path, encoding="utf8") as f:
        for line in f.readlines():
            line = json.loads(line)
            item = {"text": line["text"], "entity_list": []}
            for k, v in line["label"].items():
                for spans in v.values():
                    for start, end in spans:
                        item["entity_list"].append((start, end, k))
            datas.append(item)

    return datas


# if __name__ == "__main__":
