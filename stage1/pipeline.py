import json
import os

import numpy as np
from tqdm import tqdm

from .builder import build_sid_from_text_withPrompt, build_token_idf
from .collision import (
    analyze_length_vs_collision,
    analyze_sid_collision,
    assign_frequency_suffix_ranks,
    expand_colliding_sids,
)
from .data import load_item_frequency


def generate_variable_length_sid(args, item_text_list, tokenizer, model):
    items, texts = zip(*item_text_list)
    order_texts = [None] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text

    all_sids = {}
    debug_info = {}
    quality_metrics = {}
    idf_dict = build_token_idf(item_text_list, tokenizer)

    for item_id, fields in tqdm(list(enumerate(order_texts))):
        title = fields[0]

        sid, weights, contrib_words, quality_info = build_sid_from_text_withPrompt(
            title,
            tokenizer,
            model,
            idf_dict=idf_dict,  # â­ æ–°å¢ž
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            max_len=args.max_sent_len,
            semantic_quality_threshold=args.semantic_quality_threshold,
            pruning_pool_size=args.pruning_pool_size,
            distinctiveness_lambda=args.distinctiveness_lambda,
            candidate_prefix_tokens=args.candidate_prefix_tokens,
            layer_idx=args.functional_layer_idx,
            return_debug=True
        )

        all_sids[item_id] = {
            "tokens": sid,
            "weights": [float(x) for x in weights]
        }
        quality_metrics[item_id] = quality_info

        if item_id < 5:
            debug_info[item_id] = {
                "title": title,
                "contrib_words": contrib_words,
                "quality_info": quality_info
            }

    item_freq = load_item_frequency(args)

    semantic_sids_before_expansion = {
        item_id: {
            "tokens": list(sid_info["tokens"]),
            "weights": [float(x) for x in sid_info["weights"]]
        }
        for item_id, sid_info in all_sids.items()
    }

    if args.max_collision_expand_rounds > 0:
        all_sids = expand_colliding_sids(
            all_sids,
            quality_metrics,
            max_tokens=args.max_tokens,
            max_rounds=args.max_collision_expand_rounds
        )

    semantic_sids = {
        item_id: {
            "tokens": list(sid_info["tokens"]),
            "weights": [float(x) for x in sid_info["weights"]]
        }
        for item_id, sid_info in all_sids.items()
    }

    all_sids = assign_frequency_suffix_ranks(all_sids, item_freq)

    print("\n[Before collision-aware expansion]")
    analyze_sid_collision(
        semantic_sids_before_expansion,
        assign_frequency_suffix_ranks(semantic_sids_before_expansion, item_freq)
    )
    print("\n[After collision-aware expansion]")
    analyze_sid_collision(semantic_sids, all_sids)
    analyze_length_vs_collision(semantic_sids)

    print("\n========== Debug: First 5 Items ==========")
    for item_id in range(5):
        print(f"\nItem {item_id}")
        print("Title:", debug_info[item_id]["title"])
        print("SID:", all_sids[item_id])
        print("Contrib tokens:", debug_info[item_id]["contrib_words"])
        print("Quality:", debug_info[item_id]["quality_info"])
    print("==========================================\n")

    save_dir = os.path.join("data", "sids", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # ðŸ”§ ç»Ÿä¸€è½¬æˆ python int
    all_sids = {
        int(item_id): {
            "semantic_tokens": [
                int(token) for token in sid["semantic_tokens"]
            ],
            "semantic_weights": [
                float(weight) for weight in sid["semantic_weights"]
            ],
            "suffix_rank": int(sid["suffix_rank"]),
        }
        for item_id, sid in all_sids.items()
    }

    sid_path = os.path.join(
        save_dir,
        f"{args.dataset}.sid-set-token-weight-stage1.json",
    )
    with open(sid_path, "w", encoding="utf-8") as file:
        json.dump(all_sids, file)

    quality_metrics = {
        int(k): v
        for k, v in quality_metrics.items()
    }
    quality_path = os.path.join(
        save_dir,
        f"{args.dataset}.sid-set-token-quality-stage1.json",
    )
    with open(quality_path, "w", encoding="utf-8") as file:
        json.dump(quality_metrics, file, indent=2)

    # æœ€ç»ˆ SID é•¿åº¦ç»Ÿè®¡
    lengths = [len(v["semantic_tokens"]) for v in all_sids.values()]
    print("\nFinal SID length stats:")
    print("mean:", np.mean(lengths))
    print("max:", np.max(lengths))
    print("min:", np.min(lengths))
