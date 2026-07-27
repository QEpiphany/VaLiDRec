from typing import List, Set

import torch
import torch.nn.functional as F


def item_contrastive_loss(
    h,
    item_emb_table,
    target_items,
    itemid2row,
    temp=0.2,   # �?新增
):
    B = h.size(0)
    if B <= 1:
        return h.new_zeros(())

    rows = []
    for it in target_items.tolist():
        row = itemid2row.get(int(it), None)
        if row is None:
            return h.new_zeros(())
        rows.append(row)

    rows = torch.tensor(rows, device=h.device, dtype=torch.long)

    batch_item_emb = item_emb_table[rows]   # [B,D]

    # ===== �?NEW 1: normalize =====
    h = F.normalize(h, dim=-1)
    batch_item_emb = F.normalize(batch_item_emb, dim=-1)

    # ===== �?NEW 2: temperature =====
    logits = torch.matmul(h, batch_item_emb.t()) / temp

    labels = torch.arange(B, device=h.device)

    loss = F.cross_entropy(logits, labels)

    return loss


def item_rank_loss(
    h,
    emb_weight,
    target_items,
    itemid2row,
    item_sid_tokens,
    item_sid_mask,
    num_neg_items=64,
):
    """
    Align training with final unweighted item scoring:
      score(item) = mean(sigmoid(h dot token_emb)) over SID tokens.
    The positive item is ranked against sampled negative items by CE.
    """
    B = h.size(0)
    I = item_sid_tokens.size(0)
    if B == 0 or I <= 1 or num_neg_items <= 0:
        return h.new_zeros(())

    pos_rows = []
    for it in target_items.tolist():
        row = itemid2row.get(int(it), None)
        if row is None:
            return h.new_zeros(())
        pos_rows.append(row)
    pos_rows = torch.tensor(pos_rows, dtype=torch.long, device=h.device)

    neg_rows = torch.randint(0, I, (B, num_neg_items), device=h.device)
    same = neg_rows.eq(pos_rows.unsqueeze(1))
    if same.any():
        neg_rows = torch.where(same, (neg_rows + 1) % I, neg_rows)

    cand_rows = torch.cat([pos_rows.unsqueeze(1), neg_rows], dim=1)  # [B, 1+N]
    cand_tokens = item_sid_tokens[cand_rows]                         # [B, C, S]
    cand_mask = item_sid_mask[cand_rows].to(h.dtype)                 # [B, C, S]
    cand_emb = emb_weight[cand_tokens]                               # [B, C, S, D]

    token_logits = torch.einsum("bd,bcsd->bcs", h, cand_emb)
    token_scores = torch.sigmoid(token_logits) * cand_mask
    item_scores = token_scores.sum(dim=-1) / cand_mask.sum(dim=-1).clamp(min=1.0)

    labels = torch.zeros(B, dtype=torch.long, device=h.device)
    return F.cross_entropy(item_scores, labels)


def build_batch_candidate_token_list(
    batch_target_sets: List[Set[int]],
    device: torch.device,
):
    """
    Batch negative sampling:
    candidate tokens = union of all positive tokens in batch
    """

    B = len(batch_target_sets)

    # union of batch tokens
    cand_set = set()
    for s in batch_target_sets:
        cand_set.update(s)

    cand_list = sorted(list(cand_set))
    candidate_ids = torch.tensor(cand_list, dtype=torch.long, device=device)

    C = len(cand_list)

    label_mat = torch.zeros((B, C), dtype=torch.float32, device=device)

    idx = {t: i for i, t in enumerate(cand_list)}

    for b, s in enumerate(batch_target_sets):
        for t in s:
            if t in idx:
                label_mat[b, idx[t]] = 1.0

    return candidate_ids, label_mat
