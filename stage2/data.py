from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .sid import flatten_history_sid


def read_inter_file(file_path: str):
    """Read user, item-history, and target-item records."""
    records = []
    skipped = 0

    def is_int(value: str):
        value = value.strip()
        return value.isdigit() or (
            value.startswith("-") and value[1:].isdigit()
        )

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                skipped += 1
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                skipped += 1
                continue

            user, history, target = (part.strip() for part in parts)
            history_tokens = history.split() if history else []
            if (
                not is_int(user)
                or not is_int(target)
                or not all(is_int(token) for token in history_tokens)
            ):
                skipped += 1
                continue

            records.append(
                (int(user), [int(token) for token in history_tokens], int(target))
            )

    if not records:
        raise ValueError(f"No valid data lines found in {file_path}.")
    if skipped:
        print(f"[read_inter_file] skipped {skipped} non-data lines.")
    return records


class TokenSetSIDDataset(Dataset):
    """
    Each record: (uid, history_items, target_item)
    Input  : history SID tokens (flattened)
    Target : semantic token set of the target item's SID
    """
    def __init__(
        self,
        records: List[Tuple[int, List[int], int]],
        item2sid: Dict[int, List[int]],
        max_history_items: Optional[int] = None,
        min_history_len: int = 1,
    ):
        self.records = records
        self.item2sid = item2sid
        self.max_history_items = max_history_items
        self.min_history_len = min_history_len

        # filter invalid
        cleaned = []
        for uid, hist, tgt in self.records:
            if tgt not in self.item2sid:
                continue
            if hist is None:
                hist = []
            if len(hist) < self.min_history_len:
                continue
            cleaned.append((uid, hist, tgt))
        self.records = cleaned

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        uid, hist_items, tgt = self.records[idx]

        if self.max_history_items is not None:
            hist_items = hist_items[-self.max_history_items:]

        x = flatten_history_sid(hist_items, self.item2sid, max_history_items=None)
        if len(x) == 0:
            x = [0]

        y_tokens = self.item2sid[tgt]
        y_set = set(int(t) for t in y_tokens)

        return {
            "uid": int(uid),
            "history_items": list(hist_items),   # 这里�?input_tokens 对齐�?
            "input_tokens": x,
            "target_item": int(tgt),
            "target_set": y_set,
        }


def collate_token_set_batch(batch, pad_id: int):
    B = len(batch)
    lens = [len(b["input_tokens"]) for b in batch]
    L = max(lens)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    target_items = torch.tensor([b["target_item"] for b in batch], dtype=torch.long)

    target_sets = []
    history_items = []

    for i, b in enumerate(batch):
        toks = b["input_tokens"]
        input_ids[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        attention_mask[i, :len(toks)] = 1
        target_sets.append(b["target_set"])
        history_items.append(b["history_items"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_sets": target_sets,
        "target_items": target_items,
        "history_items": history_items,
    }
