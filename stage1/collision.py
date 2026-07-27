from collections import defaultdict

import numpy as np


def analyze_sid_collision(all_sids_before_suffix, all_sids_after_suffix):
    from collections import defaultdict
    import numpy as np

    # ---------- 1ï¸âƒ£ semantic SID ç¢°æ’ž ----------
    semantic_buckets = defaultdict(list)

    for item_id, sid in all_sids_before_suffix.items():
        tokens = sid.get("semantic_tokens", sid.get("tokens", []))
        semantic_buckets[tuple(tokens)].append(item_id)

    total_items = len(all_sids_before_suffix)
    unique_semantic = len(semantic_buckets)

    semantic_collision_rate = 1 - unique_semantic / total_items

    cluster_sizes = [len(v) for v in semantic_buckets.values()]
    max_cluster = max(cluster_sizes) if cluster_sizes else 0
    avg_cluster = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0

    # ---------- 2ï¸âƒ£ æœ€ç»ˆ SID ç¢°æ’ž ----------
    final_buckets = defaultdict(list)

    for item_id, sid in all_sids_after_suffix.items():
        tokens = sid.get("semantic_tokens", sid.get("tokens", []))
        suffix_rank = sid.get("suffix_rank")
        complete_identifier = (
            tuple(tokens),
            int(suffix_rank) if suffix_rank is not None else None,
        )
        final_buckets[complete_identifier].append(item_id)

    unique_final = len(final_buckets)
    final_collision_rate = 1 - unique_final / total_items

    # ---------- æ‰“å° ----------
    print("\n========== SID Collision Analysis ==========")

    print("\n[Semantic SID]")
    print("Total items:", total_items)
    print("Unique semantic SID:", unique_semantic)
    print("Semantic collision rate:", round(semantic_collision_rate, 4))
    print("Avg items per semantic SID:", round(avg_cluster, 3))
    print("Max items sharing one semantic SID:", max_cluster)

    print("\n[Final SID (with suffix)]")
    print("Unique final SID:", unique_final)
    print("Final collision rate:", round(final_collision_rate, 6))

    if final_collision_rate == 0:
        print("âœ“ All items uniquely identified after suffix")
    else:
        print("âœ— Still collisions after suffix!")

    print("============================================\n")


def analyze_length_vs_collision(all_sids_before_suffix):
    from collections import defaultdict
    import numpy as np

    # length -> list of cluster sizes
    length_buckets = defaultdict(list)

    # å…ˆå»º semantic bucket
    semantic_buckets = defaultdict(list)
    for item_id, sid in all_sids_before_suffix.items():
        tokens = sid["tokens"]
        semantic_buckets[tuple(tokens)].append(item_id)

    # å†æŒ‰é•¿åº¦ç»Ÿè®¡
    for tokens, items in semantic_buckets.items():
        L = len(tokens)
        length_buckets[L].append(len(items))

    print("\n========== Length vs Collision ==========")

    for L in sorted(length_buckets.keys()):
        sizes = length_buckets[L]

        avg_cluster = np.mean(sizes)
        max_cluster = np.max(sizes)
        collision_rate = np.mean([1 if s > 1 else 0 for s in sizes])

        print(f"\nLength = {L}")
        print(f"  #SID groups: {len(sizes)}")
        print(f"  Avg cluster size: {avg_cluster:.3f}")
        print(f"  Max cluster size: {max_cluster}")
        print(f"  Collision rate: {collision_rate:.3f}")

    print("=========================================\n")


def assign_frequency_suffix_ranks(all_sids, item_freq):
    """
    all_sids:
        item_id -> {
            "tokens": [...],
            "weights": [...]
        }
    """

    sid_buckets = defaultdict(list)

    # âœ… bucketé‡Œä¿ç•™å®Œæ•´ä¿¡æ¯
    for item_id, sid_info in all_sids.items():
        tokens = list(sid_info["tokens"])
        weights = list(sid_info["weights"])
        if len(tokens) != len(weights):
            raise ValueError(
                f"Item {item_id} has {len(tokens)} semantic tokens but "
                f"{len(weights)} semantic weights."
            )

        sid_buckets[tuple(tokens)].append((item_id, tokens, weights))

    new_sids = {}

    for sid_tuple, items in sid_buckets.items():

        # items = [(item_id, weights), ...]

        # ===== case 1: no collision =====
        if len(items) == 1:
            item_id, tokens, weights = items[0]

            new_sids[item_id] = {
                "semantic_tokens": tokens,
                "semantic_weights": weights,
                "suffix_rank": 0,
            }
            continue

        # ===== case 2: collision =====
        items_sorted = sorted(
            items,
            key=lambda x: (-item_freq.get(x[0], 0), x[0])
        )

        for rank, (item_id, tokens, weights) in enumerate(items_sorted):
            new_sids[item_id] = {
                "semantic_tokens": tokens,
                "semantic_weights": weights,
                "suffix_rank": rank,
            }

    return new_sids


def normalize_weights(weights):
    weights = np.array(weights, dtype=np.float32)
    total = float(weights.sum())
    if total <= 1e-12:
        if len(weights) == 0:
            return []
        return (np.ones(len(weights), dtype=np.float32) / len(weights)).tolist()
    return (weights / total).tolist()


def semantic_collision_buckets(all_sids):
    buckets = defaultdict(list)
    for item_id, sid_info in all_sids.items():
        buckets[tuple(sid_info["tokens"])].append(item_id)
    return buckets


def expand_colliding_sids(all_sids, quality_metrics, max_tokens, max_rounds):
    expanded = {
        item_id: {
            "tokens": list(sid_info["tokens"]),
            "weights": [float(x) for x in sid_info["weights"]]
        }
        for item_id, sid_info in all_sids.items()
    }

    expanded_items = defaultdict(int)
    replaced_items = defaultdict(int)

    for _ in range(max_rounds):
        buckets = semantic_collision_buckets(expanded)
        colliding_groups = [items for items in buckets.values() if len(items) > 1]
        if len(colliding_groups) == 0:
            break

        changed = False
        for items in colliding_groups:
            for item_id in items:
                sid_info = expanded[item_id]

                used = set(int(x) for x in sid_info["tokens"])
                backlog = quality_metrics.get(item_id, {}).get("candidate_backlog", [])

                chosen = None
                for candidate in backlog:
                    tid = int(candidate["token_id"])
                    if tid not in used:
                        chosen = candidate
                        break

                if chosen is None:
                    continue

                chosen_token = int(chosen["token_id"])
                chosen_weight = float(chosen.get("idf", 0.0)) + 1e-6

                if len(sid_info["tokens"]) < max_tokens:
                    sid_info["tokens"].append(chosen_token)
                    sid_info["weights"].append(chosen_weight)
                    expanded_items[item_id] += 1
                else:
                    replace_idx = int(np.argmin(np.array(sid_info["weights"], dtype=np.float32)))
                    sid_info["tokens"][replace_idx] = chosen_token
                    sid_info["weights"][replace_idx] = chosen_weight
                    replaced_items[item_id] += 1

                sid_info["weights"] = normalize_weights(sid_info["weights"])
                changed = True

        if not changed:
            break

    before_collision_groups = sum(
        1 for items in semantic_collision_buckets(all_sids).values()
        if len(items) > 1
    )
    after_collision_groups = sum(
        1 for items in semantic_collision_buckets(expanded).values()
        if len(items) > 1
    )

    print("\n========== Collision-aware Expansion ==========")
    print("Collision groups before:", before_collision_groups)
    print("Collision groups after:", after_collision_groups)
    print("Items expanded:", len(expanded_items))
    print("Items replaced:", len(replaced_items))
    if len(expanded_items) > 0:
        print("Avg added tokens:", round(float(np.mean(list(expanded_items.values()))), 3))
        print("Max added tokens:", int(max(expanded_items.values())))
    if len(replaced_items) > 0:
        print("Avg replaced tokens:", round(float(np.mean(list(replaced_items.values()))), 3))
        print("Max replaced tokens:", int(max(replaced_items.values())))
    print("==============================================\n")

    for item_id, added in expanded_items.items():
        quality_metrics[item_id]["collision_added_tokens"] = int(added)
    for item_id, replaced in replaced_items.items():
        quality_metrics[item_id]["collision_replaced_tokens"] = int(replaced)

    return expanded
