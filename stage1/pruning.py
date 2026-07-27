import numpy as np
import torch
import torch.nn.functional as F


def cosine_candidates_to_target(candidates, target_emb):
    if len(candidates) == 0:
        return -1.0

    emb = torch.stack([candidate[6] for candidate in candidates], dim=0).mean(dim=0)
    return F.cosine_similarity(
        target_emb.float().unsqueeze(0),
        emb.float().unsqueeze(0)
    ).item()


def cosine_embedding_to_target(emb, target_emb):
    return F.cosine_similarity(
        target_emb.float().unsqueeze(0),
        emb.float().unsqueeze(0)
    ).item()


def greedy_functional_prune(
        token_candidates,
        target_emb,
        model,
        min_tokens,
        max_tokens,
        semantic_quality_threshold,
        pruning_pool_size,
        distinctiveness_lambda,
        candidate_prefix_tokens,
        layer_idx=-1
):
    if pruning_pool_size <= 0 or pruning_pool_size >= len(token_candidates):
        pooled = list(token_candidates)
    else:
        pool = {}

        for candidate in sorted(token_candidates, key=lambda x: -x[2])[:pruning_pool_size]:
            pool[candidate[0]] = candidate
        for candidate in sorted(token_candidates, key=lambda x: -x[4])[:max(1, pruning_pool_size // 2)]:
            pool[candidate[0]] = candidate
        for candidate in sorted(token_candidates, key=lambda x: -x[5])[:max(1, pruning_pool_size // 2)]:
            pool[candidate[0]] = candidate
        for candidate in sorted(token_candidates, key=lambda x: x[0])[:candidate_prefix_tokens]:
            pool[candidate[0]] = candidate

        pooled = list(pool.values())

    pooled = sorted(pooled, key=lambda x: x[0])

    if len(pooled) <= max_tokens:
        quality = cosine_candidates_to_target(pooled, target_emb)
        return pooled, quality, [], quality

    keep = set(range(len(pooled)))
    deletion_trace = []
    step = 1

    while len(keep) > min_tokens:
        best_idx = None
        best_quality = -1.0
        best_score = -1e9
        total_idf = sum(max(0.0, pooled[i][4]) for i in keep) + 1e-12
        current_sum = torch.stack([pooled[i][6] for i in keep], dim=0).sum(dim=0)
        next_len = len(keep) - 1

        for idx in keep:
            trial_emb = (current_sum - pooled[idx][6]) / max(1, next_len)
            quality = cosine_embedding_to_target(trial_emb, target_emb)
            distinctiveness_loss = max(0.0, pooled[idx][4]) / total_idf
            delete_score = quality - distinctiveness_lambda * distinctiveness_loss

            if delete_score > best_score:
                best_score = delete_score
                best_quality = quality
                best_idx = idx

        if best_idx is None:
            break

        must_prune_for_length = len(keep) > max_tokens
        preserves_semantics = best_quality >= semantic_quality_threshold

        if not must_prune_for_length and not preserves_semantics:
            break

        deleted = pooled[best_idx]
        deletion_trace.append({
            "step": step,
            "position": int(deleted[0]),
            "token_id": int(deleted[1]),
            "token": deleted[3],
            "idf": float(deleted[4]),
            "quality_after_delete": float(best_quality)
        })
        keep.remove(best_idx)
        step += 1

    kept = [pooled[i] for i in sorted(keep, key=lambda x: pooled[x][0])]
    final_quality = cosine_candidates_to_target(kept, target_emb)
    base_quality = cosine_candidates_to_target(pooled, target_emb)

    return kept, final_quality, deletion_trace, base_quality


def compute_functional_weights(kept, final_quality, target_emb, model, layer_idx=-1):
    if len(kept) == 0:
        return np.array([], dtype=np.float32)

    if len(kept) == 1:
        return np.array([1.0], dtype=np.float32)

    losses = []
    current_sum = torch.stack([candidate[6] for candidate in kept], dim=0).sum(dim=0)
    next_len = len(kept) - 1
    for idx in range(len(kept)):
        trial_emb = (current_sum - kept[idx][6]) / max(1, next_len)
        quality_without = cosine_embedding_to_target(trial_emb, target_emb)
        losses.append(max(0.0, final_quality - quality_without))

    losses = np.array(losses, dtype=np.float32)
    if float(losses.sum()) <= 1e-12:
        saliency = np.array([x[2] for x in kept], dtype=np.float32)
        losses = np.clip(saliency, 1e-8, None)

    return losses / (losses.sum() + 1e-12)
