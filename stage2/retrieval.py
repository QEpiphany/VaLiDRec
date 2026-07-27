from collections import defaultdict
from typing import Dict, List

import torch

from .graph import forward_with_item_graph_prompt


def build_inverted_index(item2sid: Dict[int, List[int]]):
    """
    token -> items containing token
    also store token_counts per item for length normalization
    """
    token2items = defaultdict(list)  # token -> list[item_id]
    item_token_count = {}            # item_id -> |SID|
    for it, sid in item2sid.items():
        uniq = set(int(t) for t in sid)
        item_token_count[it] = max(1, len(uniq))
        for t in uniq:
            token2items[t].append(it)
    return token2items, item_token_count


def apply_suffix_tie_break(ranked, item2sid, item2suffix):
    """
    Apply Eq. (30): only items with identical semantic token sets are
    reordered, and the smaller suffix rank is preferred.

    Positions occupied by different semantic-SID groups are left unchanged,
    so suffix ranks cannot alter a strict semantic-score ordering.
    """
    group_positions = defaultdict(list)
    for position, (item_id, _) in enumerate(ranked):
        semantic_sid = tuple(sorted(set(item2sid[item_id])))
        group_positions[semantic_sid].append(position)

    result = list(ranked)
    for positions in group_positions.values():
        if len(positions) < 2:
            continue

        group_entries = [ranked[position] for position in positions]
        group_entries.sort(
            key=lambda entry: (
                item2suffix[entry[0]],
                entry[0],
            )
        )
        for position, entry in zip(positions, group_entries):
            result[position] = entry

    return result


def predict_batch(
    model,
    graph_prompt,
    item2sid,
    input_ids,
    attention_mask,
    history_items_batch,
    itemid2row,
    item_sid_tokens,
    item_sid_mask,
    all_sid_tokens_t,
    token2items,
    item_token_count,
    item2suffix,
    topk_items=20,
    topm_tokens=0,
):
    device = input_ids.device

    # ===== forward (LLM + graph prompt) =====
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        outputs, item_emb_table = forward_with_item_graph_prompt(
            model=model,
            graph_prompt=graph_prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_items_batch=history_items_batch,
            itemid2row=itemid2row,
            item_sid_tokens=item_sid_tokens,
            item_sid_mask=item_sid_mask,
        )

        # last hidden
        h = outputs.hidden_states[-1][:, -1, :]   # [B, D]

        # token scoring
        emb = model.get_input_embeddings().weight
        sid_tok = all_sid_tokens_t
        W = emb[sid_tok]   # [V_sid, D]

        token_scores = torch.sigmoid(h @ W.t())   # [B, V_sid]

    # ===== 后处理（CPU / float32 更安全）=====
    token_scores = token_scores.float()

    results = []

    for b in range(input_ids.size(0)):
        scores = token_scores[b]

        # ===== top-M tokens; topm_tokens <= 0 means use every SID token =====
        if topm_tokens is None or int(topm_tokens) <= 0:
            top_indices = torch.arange(scores.size(0), device=scores.device)
            top_values = scores
        else:
            top = torch.topk(scores, min(int(topm_tokens), scores.size(0)))
            top_indices = top.indices
            top_values = top.values

        item_scores = defaultdict(float)

        # ===== inverted index aggregation =====
        for idx, s in zip(top_indices.tolist(), top_values.tolist()):
            tok = int(sid_tok[idx].item())

            for it in token2items.get(tok, []):
                item_scores[it] += s

        # ===== length normalization =====
        ranked = [
            (it, sc / item_token_count[it])
            for it, sc in item_scores.items()
        ]

        ranked.sort(key=lambda x: -x[1])

        ranked = apply_suffix_tie_break(ranked, item2sid, item2suffix)

        # ===== top-K items =====
        pred = [it for it, _ in ranked[:topk_items]]

        while len(pred) < topk_items:
            pred.append(-1)

        results.append(pred)

    return results
