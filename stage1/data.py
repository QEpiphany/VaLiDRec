import os
from collections import defaultdict

from utils import clean_text, load_json


def load_data(args):
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    return load_json(item2feature_path)


def generate_text(item2feature, features):
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text_parts = []

        for meta_key in features:
            if meta_key in data and data[meta_key] is not None:
                meta_value = clean_text(data[meta_key])
                if len(meta_value.strip()) > 0:
                    text_parts.append(meta_value.strip())

        # â­ æ‹¼æˆä¸€ä¸ªå®Œæ•´æ–‡æœ¬
        full_text = " ".join(text_parts)

        item_text_list.append([int(item), [full_text]])

    return item_text_list


def preprocess_text(args):
    print('Process text data:')
    print(' Dataset:', args.dataset)
    item2feature = load_data(args)
    return generate_text(item2feature, ['title', 'description'])


def load_item_frequency(args):
    inter_path = os.path.join(args.root, f"{args.dataset}.train.inter")
    freq = defaultdict(int)

    if not os.path.exists(inter_path):
        return defaultdict(lambda: 1)

    with open(inter_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            try:
                item_id = int(parts[2])
                freq[item_id] += 1
            except:
                pass

    return freq
