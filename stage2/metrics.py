import torch
from tqdm import tqdm

from .retrieval import predict_batch
from .sid import flatten_history_sid


def calculate_pos_index_items(pred_items: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred_items = pred_items.detach().cpu()
    labels = labels.detach().cpu()
    B, K = pred_items.shape
    pos_index = torch.zeros((B, K), dtype=torch.bool)
    for i in range(B):
        gt = int(labels[i].item())
        for j in range(K):
            if int(pred_items[i, j].item()) == gt:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    device = pos_index.device
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=device)
    dcg_weights = 1.0 / torch.log2(ranks + 1)
    dcg_weights = dcg_weights.unsqueeze(0).expand_as(pos_index).to(torch.float)
    dcg = torch.where(pos_index, dcg_weights, torch.zeros_like(dcg_weights))
    return dcg[:, :k].sum(dim=1).float()


def evaluate(
    model,
    graph_prompt,
    records,
    item2sid,
    item2suffix,
    itemid2row,
    item_sid_tokens,
    item_sid_mask,
    all_sid_tokens_t,
    token2items,
    item_token_count,
    device,
    pad_id,
    topk_list,
    topk_items,
    max_history_items,
    infer_batch_size,
    topm_tokens,
):
    model.eval()
    recalls = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcgs = {f"NDCG@{k}": 0.0 for k in topk_list}
    total_samples = 0

    for st in tqdm(range(0, len(records), infer_batch_size), desc="Evaluating"):
        chunk = records[st: st + infer_batch_size]
        pred_mat, label_vec = [], []

        batch_inputs = []
        labels = []
        history_items_batch = []

        for _, hist_items, tgt in chunk:
            hist_trim = hist_items[-max_history_items:] if max_history_items is not None else hist_items
            hist_sid = flatten_history_sid(hist_trim, item2sid, max_history_items=None)
            if len(hist_sid) == 0:
                hist_sid = [pad_id]

            batch_inputs.append(hist_sid)
            labels.append(int(tgt))
            history_items_batch.append(hist_trim)

        # padding
        max_len = max(len(x) for x in batch_inputs)
        input_ids = torch.full((len(batch_inputs), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)

        for i, seq in enumerate(batch_inputs):
            input_ids[i, :len(seq)] = torch.tensor(seq, device=device)
            attention_mask[i, :len(seq)] = 1

        pred_mat = predict_batch(
            model=model,
            graph_prompt=graph_prompt,
            item2sid=item2sid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            history_items_batch=history_items_batch,   # �?新加
            itemid2row=itemid2row,
            item_sid_tokens=item_sid_tokens,
            item_sid_mask=item_sid_mask,
            all_sid_tokens_t=all_sid_tokens_t,
            token2items=token2items,
            item_token_count=item_token_count,
            item2suffix=item2suffix,
            topk_items=topk_items,
            topm_tokens=topm_tokens,
        )

        label_vec = labels

        pred_items_t = torch.tensor(pred_mat, dtype=torch.long)
        labels_t = torch.tensor(label_vec, dtype=torch.long)
        pos_index = calculate_pos_index_items(pred_items_t, labels_t)
        batch_size = pos_index.size(0)
        total_samples += batch_size

        for k in topk_list:
            recalls[f"Recall@{k}"] += recall_at_k(pos_index, k).sum().item()
            ndcgs[f"NDCG@{k}"] += ndcg_at_k(pos_index, k).sum().item()

    if total_samples == 0:
        avg_recalls = {k: 0.0 for k in recalls}
        avg_ndcgs = {k: 0.0 for k in ndcgs}
    else:
        avg_recalls = {k: (v / total_samples) for k, v in recalls.items()}
        avg_ndcgs = {k: (v / total_samples) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs
