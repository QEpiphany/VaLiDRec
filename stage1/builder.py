import math
from collections import defaultdict

import numpy as np
import torch

from .pruning import compute_functional_weights, greedy_functional_prune


def build_token_idf(item_text_list, tokenizer):
    df = defaultdict(int)
    N = len(item_text_list)

    for item_id, fields in item_text_list:
        text = fields[0]

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        )

        token_ids = set(encoded["input_ids"])  # â­ åŽ»é‡ï¼ˆå…³é”®ï¼ï¼‰

        for tid in token_ids:
            df[tid] += 1

    idf = {}
    for tid, freq in df.items():
        idf[tid] = math.log((N + 1) / (freq + 1))

    return idf


def build_sid_from_text_withPrompt(
        text,
        tokenizer,
        model,
        idf_dict,                     # â­ æ–°å¢ž
        min_tokens=2,
        max_tokens=16,
        max_len=128,
        semantic_quality_threshold=0.95,
        pruning_pool_size=64,
        distinctiveness_lambda=0.05,
        candidate_prefix_tokens=8,
        layer_idx=-1,
        return_debug=False
):
    assert (text is not None and len(text) != 0)

    # â—å»ºè®®åŽ»æŽ‰ promptï¼ˆæ›´ç¨³å®šï¼‰
    full_text = text

    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            use_cache=False
        )

    input_ids = encoded["input_ids"].squeeze(0).cpu().numpy()
    attention_mask = encoded["attention_mask"].squeeze(0).cpu().numpy()

    # =========================
    # ðŸ”¥ importanceï¼ˆåŽ»æŽ‰ whitenï¼Œæ›´ç¨³ï¼‰
    # =========================
    h = outputs.hidden_states[layer_idx][0]   # [seq, dim]
    attention_mask_np = encoded["attention_mask"].squeeze(0).cpu().numpy()
    target_idx = np.where(attention_mask_np == 1)[0]
    target_emb = h[target_idx].mean(dim=0)
    importance = torch.norm(h, dim=1)  # â­ ç›´æŽ¥ç”¨ norm
    importance = importance.detach().cpu().numpy()

    # =========================
    # ðŸ”¥ valid tokenç­›é€‰
    # =========================
    valid_idx = np.where(attention_mask == 1)[0]

    special_ids = set(tokenizer.all_special_ids)
    valid_idx = [i for i in valid_idx if int(input_ids[i]) not in special_ids]

    token_candidates = []
    contrib_tokens = []

    for i in valid_idx:
        tid = int(input_ids[i])

        tok_text = tokenizer.decode([tid]).lower().strip()

        # ===== åŸºç¡€è¿‡æ»¤ï¼ˆä¿ç•™ä½ çš„ token-level æ€è·¯ï¼‰=====
        if len(tok_text) <= 2:
            continue
        if not tok_text.isascii():
            continue

        # =========================
        # ðŸ”¥ IDF åŠ æƒï¼ˆæ ¸å¿ƒï¼‰
        # =========================
        idf_score = idf_dict.get(tid, 0.0)

        # clipï¼ˆé˜²æ­¢æžç«¯tokenï¼‰
        idf_score = min(idf_score, 5.0)

        # å¹³æ»‘ï¼ˆé¿å…idf=0ï¼‰
        score = importance[i] * (idf_score + 1.0)

        token_candidates.append((
            i,
            tid,
            float(score),
            tok_text,
            float(idf_score),
            float(importance[i]),
            h[i].detach()
        ))

    # fallback
    if len(token_candidates) == 0:
        unk = tokenizer.unk_token_id or tokenizer.pad_token_id
        sid = [int(unk)]
        weights = [1.0]
        if return_debug:
            return sid, weights, [], {
                "semantic_quality": 0.0,
                "base_quality": 0.0,
                "candidate_count": 0,
                "pool_size": 0,
                "sid_tokens": len(sid),
                "compression_ratio": 1.0,
                "deletion_steps": 0,
                "semantic_quality_threshold": float(semantic_quality_threshold),
                "deleted_tokens": []
            }
        return sid, weights

    kept, final_quality, deletion_trace, base_quality = greedy_functional_prune(
        token_candidates,
        target_emb,
        model,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        semantic_quality_threshold=semantic_quality_threshold,
        pruning_pool_size=pruning_pool_size,
        distinctiveness_lambda=distinctiveness_lambda,
        candidate_prefix_tokens=candidate_prefix_tokens,
        layer_idx=layer_idx
    )

    weights = compute_functional_weights(
        kept,
        final_quality,
        target_emb,
        model,
        layer_idx=layer_idx
    )

    ranked = sorted(
        zip(kept, weights),
        key=lambda x: (-float(x[1]), -float(x[0][2]))
    )

    sid = [int(x[0][1]) for x in ranked]
    weights = [float(x[1]) for x in ranked]
    contrib_tokens = []

    for (idx, tid, score, tok_text, idf_score, hidden_norm, _), w in ranked:
        contrib_tokens.append({
            "token": tok_text,
            "raw_score": float(score),
            "idf": float(idf_score),
            "hidden_norm": float(hidden_norm),
            "functional_weight": float(w),
            "position": int(idx)
        })

    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-12)

    quality_info = {
        "semantic_quality": float(final_quality),
        "base_quality": float(base_quality),
        "candidate_count": int(len(token_candidates)),
        "pool_size": int(min(len(token_candidates), pruning_pool_size)),
        "sid_tokens": int(len(sid)),
        "compression_ratio": float(len(sid) / max(1, len(valid_idx))),
        "deletion_steps": int(len(deletion_trace)),
        "semantic_quality_threshold": float(semantic_quality_threshold),
        "distinctiveness_lambda": float(distinctiveness_lambda),
        "functional_layer_idx": int(layer_idx),
        "deleted_tokens": deletion_trace[:10],
        "candidate_backlog": [
            {
                "position": int(idx),
                "token_id": int(tid),
                "token": tok_text,
                "raw_score": float(score),
                "idf": float(idf_score),
                "hidden_norm": float(hidden_norm)
            }
            for idx, tid, score, tok_text, idf_score, hidden_norm, _
            in sorted(token_candidates, key=lambda x: (-x[4], -x[2]))[:max_tokens * 4]
        ]
    }

    if return_debug:
        return sid, weights, contrib_tokens, quality_info

    return sid, weights
