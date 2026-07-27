import torch
import torch.nn as nn
import torch.nn.functional as F


def build_batch_item_transition_graph(history_items_batch, itemid2row, device):
    """
    history_items_batch: List[List[int]]
    按点击顺序构图：i_t -> i_{t+1}
    返回:
        node_rows: 当前 batch 中出现的 item row id（对�?item_emb_table 的行号）
        edge_index: [2, E]
        edge_weight: [E]
        sample_node_pos: 每个样本的历�?item �?node_rows 中对应的位置序列
    """
    node_row_set = set()
    sample_row_seq = []

    for hist in history_items_batch:
        row_seq = []
        for it in hist:
            row = itemid2row.get(int(it), None)
            if row is None:
                continue
            row_seq.append(row)
            node_row_set.add(row)
        sample_row_seq.append(row_seq)

    node_rows = sorted(list(node_row_set))
    if len(node_rows) == 0:
        node_rows = [0]   # safety

    row2local = {row: i for i, row in enumerate(node_rows)}

    edge_src, edge_dst, edge_w = [], [], []
    sample_node_pos = []

    for row_seq in sample_row_seq:
        local_seq = [row2local[r] for r in row_seq if r in row2local]
        sample_node_pos.append(local_seq)

        if len(local_seq) == 0:
            continue

        # self-loop
        for x in local_seq:
            edge_src.append(x)
            edge_dst.append(x)
            edge_w.append(1.0)

        # sequential transition: i_t -> i_{t+1}
        for i in range(len(local_seq) - 1):
            u = local_seq[i]
            v = local_seq[i + 1]
            edge_src.append(u)
            edge_dst.append(v)
            edge_w.append(1.0)

    if len(edge_src) == 0:
        N = len(node_rows)
        edge_src = list(range(N))
        edge_dst = list(range(N))
        edge_w = [1.0] * N

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_w, dtype=torch.float32, device=device)
    node_rows = torch.tensor(node_rows, dtype=torch.long, device=device)

    return node_rows, edge_index, edge_weight, sample_node_pos


class GraphSAGELayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim * 2, dim)

    def forward(self, x, edge_index, edge_weight=None):
        src, dst = edge_index
        N, D = x.shape

        if edge_weight is None:
            edge_weight = torch.ones(src.size(0), device=x.device, dtype=x.dtype)
        else:
            edge_weight = edge_weight.to(x.dtype)

        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src] * edge_weight.unsqueeze(-1))

        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, edge_weight)

        neigh_mean = neigh_sum / deg.clamp(min=1e-8).unsqueeze(-1)

        h = torch.cat([x, neigh_mean], dim=-1)
        return F.relu(self.lin(h))


class BatchItemGraphPrompt(nn.Module):
    def __init__(self, hidden_dim, prompt_len=8, num_layers=2):
        super().__init__()
        self.prompt_len = prompt_len
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(hidden_dim, prompt_len * hidden_dim)

    def build_item_embeddings(self, emb_weight, item_sid_tokens, item_sid_mask):
        """
        emb_weight: [V, D]
        item_sid_tokens: [I, S]
        item_sid_mask: [I, S]
        return: item_emb_table [I, D]
        """
        sid_emb = emb_weight[item_sid_tokens]  # [I, S, D]
        mask = item_sid_mask.unsqueeze(-1).to(sid_emb.dtype)
        item_emb = (sid_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return item_emb

    def forward(self, emb_weight, history_items_batch, itemid2row, item_sid_tokens, item_sid_mask):
        item_emb_table = self.build_item_embeddings(emb_weight, item_sid_tokens, item_sid_mask)  # [I, D]

        node_rows, edge_index, edge_weight, sample_node_pos = build_batch_item_transition_graph(
            history_items_batch, itemid2row, emb_weight.device
        )

        x = item_emb_table[node_rows]   # [N, D]

        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        # ===== FIX: align dtype with embedding =====
        x = x.to(item_emb_table.dtype)

        item_emb_table_graph = item_emb_table.clone()
        item_emb_table_graph[node_rows] = x

        # �?graph 更新后的局部节点写回全局 item table
        item_emb_table_graph = item_emb_table.clone()
        item_emb_table_graph[node_rows] = x

        B = len(history_items_batch)
        D = x.size(-1)
        pooled = torch.zeros((B, D), device=x.device, dtype=x.dtype)

        for b in range(B):
            idx_list = sample_node_pos[b]
            if len(idx_list) == 0:
                pooled[b] = torch.zeros(D, device=x.device, dtype=x.dtype)
            else:
                idx = torch.tensor(idx_list, dtype=torch.long, device=x.device)
                xb = x[idx]   # [L, D]

                w = torch.arange(1, xb.size(0) + 1, device=x.device, dtype=x.dtype)
                w = w / w.sum().clamp(min=1e-8)

                pooled[b] = (xb * w.unsqueeze(-1)).sum(dim=0)

        prompt = self.proj(pooled).view(B, self.prompt_len, D)
        return prompt, item_emb_table_graph


def forward_with_item_graph_prompt(
    model,
    graph_prompt,
    input_ids,
    attention_mask,
    history_items_batch,
    itemid2row,
    item_sid_tokens,
    item_sid_mask,
):
    emb_layer = model.get_input_embeddings()
    model_dtype = emb_layer.weight.dtype

    prompt_embeds, item_emb_table = graph_prompt(
        emb_layer.weight,
        history_items_batch,
        itemid2row,
        item_sid_tokens,
        item_sid_mask
    )

    prompt_embeds = prompt_embeds.to(model_dtype)

    tok_embeds = emb_layer(input_ids)
    inputs_embeds = torch.cat([prompt_embeds, tok_embeds], dim=1)

    B = input_ids.size(0)
    P = prompt_embeds.size(1)

    prompt_mask = torch.ones((B, P), device=attention_mask.device, dtype=attention_mask.dtype)
    new_mask = torch.cat([prompt_mask, attention_mask], dim=1)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=new_mask,
        output_hidden_states=True,
        use_cache=False
    )

    return outputs, item_emb_table
