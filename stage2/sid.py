import json
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


def load_sid_bundle(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "item2sid" in data:
        raw = data["item2sid"]
    else:
        raw = data

    item2sid = {}
    item2suffix = {}
    legacy_suffix_tokens = {}

    for k, sid_record in raw.items():
        suffix_rank = None
        if isinstance(sid_record, dict):
            suffix_rank = sid_record.get("suffix_rank")
            if "semantic_tokens" in sid_record:
                semantic_sid = list(map(int, sid_record["semantic_tokens"]))
                if suffix_rank is None:
                    raise ValueError(
                        f"Item {k} uses semantic_tokens but has no suffix_rank."
                    )
                item2sid[int(k)] = semantic_sid
                item2suffix[int(k)] = int(suffix_rank)
                continue
            sid = sid_record.get("tokens", sid_record.get("sid", []))
        else:
            sid = sid_record

        sid = list(map(int, sid))
        if not sid:
            raise ValueError(f"Item {k} has an empty stored SID.")

        # The complete stored identifier ends with the non-semantic suffix.
        semantic_sid = sid[:-1]
        suffix_token = sid[-1]

        item2sid[int(k)] = semantic_sid
        if suffix_rank is None:
            legacy_suffix_tokens[int(k)] = suffix_token
        else:
            item2suffix[int(k)] = int(suffix_rank)

    if legacy_suffix_tokens:
        # Backward compatibility for Stage-1 artifacts created before
        # suffix_rank was stored explicitly. Stage 1 adds <SUF_0>, <SUF_1>, ...
        # consecutively, and every non-collided item uses <SUF_0>.
        suffix_zero_token = min(legacy_suffix_tokens.values())
        for item_id, suffix_token in legacy_suffix_tokens.items():
            item2suffix[item_id] = suffix_token - suffix_zero_token

    return item2sid, item2suffix


def flatten_history_sid(history_items: List[int], item2sid: Dict[int, List[int]], max_history_items: Optional[int]):
    if max_history_items is not None:
        history_items = history_items[-max_history_items:]
    tokens: List[int] = []
    for it in history_items:
        sid = item2sid.get(it, None)
        if sid is None:
            continue
        tokens.extend(sid)
    return tokens


def build_training_tokenizer(model_name_or_path: str, cache_dir: str) -> AutoTokenizer:
    """Load the base tokenizer used by the recommendation model."""
    tok = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = tok.unk_token

    return tok


def build_item_sid_tensors(item2sid: Dict[int, List[int]], device: torch.device):
    item_ids = sorted(item2sid.keys())
    sid_lens = [len(item2sid[it]) for it in item_ids]
    if len(item_ids) == 0:
        return {}, torch.zeros((0, 1), dtype=torch.long, device=device), torch.zeros((0, 1), dtype=torch.float32, device=device)

    max_len = max(1, max(sid_lens))
    token_mat = torch.zeros((len(item_ids), max_len), dtype=torch.long, device=device)
    mask_mat = torch.zeros((len(item_ids), max_len), dtype=torch.float32, device=device)

    for row, it in enumerate(item_ids):
        sid = item2sid[it]
        if len(sid) == 0:
            continue
        sid_t = torch.tensor(sid, dtype=torch.long, device=device)
        token_mat[row, :sid_t.numel()] = sid_t
        mask_mat[row, :sid_t.numel()] = 1.0

    itemid2row = {it: i for i, it in enumerate(item_ids)}
    return itemid2row, token_mat, mask_mat
