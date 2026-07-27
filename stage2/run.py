import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import parse_args
from .data import TokenSetSIDDataset, collate_token_set_batch, read_inter_file
from .graph import BatchItemGraphPrompt, forward_with_item_graph_prompt
from .metrics import evaluate
from .objectives import (
    build_batch_candidate_token_list,
    item_contrastive_loss,
    item_rank_loss,
)
from .retrieval import build_inverted_index
from .sid import build_item_sid_tensors, build_training_tokenizer, load_sid_bundle
from .utils import freeze_all_params, set_seed


def main():
    args = parse_args()
    cfg = vars(args)

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    dataset_name = os.path.basename(os.path.normpath(cfg["dataset_path"]))
    if cfg["train_file"] == "Luxury.train.inter":
        cfg["train_file"] = f"{dataset_name}.train.inter"
    if cfg["valid_file"] == "Luxury.valid.inter":
        cfg["valid_file"] = f"{dataset_name}.valid.inter"
    if cfg["test_file"] == "Luxury.test.inter":
        cfg["test_file"] = f"{dataset_name}.test.inter"

    os.makedirs(os.path.dirname(cfg["log_path"]), exist_ok=True)
    logging.basicConfig(
        filename=cfg["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # load SID
    item2sid, item2suffix = load_sid_bundle(cfg["sid_json"])
    # 重新构建纯语�?token 集合
    all_sid_tokens = sorted(
        {int(t) for sid in item2sid.values() for t in sid}
    )

    all_sid_tokens_t = torch.tensor(all_sid_tokens, dtype=torch.long, device=device)
    logging.info(f"Loaded item2sid={len(item2sid)}, all_sid_tokens={len(all_sid_tokens)}")

    itemid2row, item_sid_tokens, item_sid_mask = build_item_sid_tensors(item2sid, device)
    logging.info(f"Prepared item SID tensors: items={item_sid_tokens.size(0)}, sid_max_len={item_sid_tokens.size(1)}")

    # tokenizer
    tokenizer = build_training_tokenizer(cfg["model_name_or_path"], cfg["model_cache_dir"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


    # dtype
    dtype = None
    if device.type == "cuda":
        if cfg["use_bf16"] and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    # model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        cache_dir=cfg["model_cache_dir"],
        torch_dtype=dtype,
    ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = pad_id

    # resize for tokenizer (+ suffix tokens)
    model.resize_token_embeddings(len(tokenizer))

    # safety resize for sid token ids
    if len(all_sid_tokens) > 0:
        max_tok = int(max(all_sid_tokens))
        cur_vocab = model.get_input_embeddings().weight.size(0)
        if max_tok >= cur_vocab:
            model.resize_token_embeddings(max_tok + 1)
            logging.info(f"resize vocab -> {max_tok+1} (max sid token {max_tok})")

    # LoRA
    model.config.use_cache = False  # training uses cache=False
    freeze_all_params(model)
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    graph_prompt = BatchItemGraphPrompt(
    hidden_dim=model.get_input_embeddings().weight.size(1),
    prompt_len=8,
    num_layers=2).to(device)
    model = get_peft_model(model, lora_cfg)

    # =====================================================
    # LoRA + 只训�?SID token embedding �?
    # =====================================================
    sid_token_set = set(int(t) for t in all_sid_tokens)
    emb = model.get_input_embeddings()
    for p in emb.parameters():
        p.requires_grad = True
    def mask_embedding_grad(grad):
        mask = torch.zeros_like(grad)

        sid_ids = torch.tensor(
            list(sid_token_set),
            device=grad.device,
            dtype=torch.long
        )

        mask[sid_ids] = 1.0
        return grad * mask
    emb.weight.register_hook(mask_embedding_grad)


    print(f"[INFO] SID embedding training enabled. SID size = {len(sid_token_set)}")

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    print("Final tokenizer size:", len(tokenizer))
    print("Final embedding size:", model.get_input_embeddings().weight.size(0))

    # read records
    train_records = read_inter_file(os.path.join(cfg["dataset_path"], cfg["train_file"]))
    valid_records = read_inter_file(os.path.join(cfg["dataset_path"], cfg["valid_file"]))
    test_records  = read_inter_file(os.path.join(cfg["dataset_path"], cfg["test_file"]))

    # datasets / loaders
    train_ds = TokenSetSIDDataset(train_records, item2sid, max_history_items=cfg["max_history_items"])
    logging.info(f"Train records: {len(train_ds)}")

    def _collate(batch):
        return collate_token_set_batch(batch, pad_id=pad_id)

    g = torch.Generator()
    g.manual_seed(cfg["seed"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,  # 必须
        collate_fn=_collate,
        generator=g
    )

    # retrieval index for eval
    token2items, item_token_count = build_inverted_index(item2sid)
    logging.info(f"Inverted index tokens: {len(token2items)}")

    metric_names = [f"Recall@{k}" for k in cfg["topk_list"]] + [f"NDCG@{k}" for k in cfg["topk_list"]]
    tracker_val = {m: {"val": -1e9, "epoch": -1} for m in metric_names}
    tracker_test = {m: {"val": -1e9, "epoch": -1} for m in metric_names}

    # =====================================================
    # 分开 LoRA �?SID embedding 学习�?
    # =====================================================

    lora_params = []
    embed_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed_tokens" in name:
            embed_params.append(p)
        else:
            lora_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": lora_params, "lr": cfg["lr"]},
            {"params": embed_params, "lr": cfg["lr"] * 0.5},
            {"params": graph_prompt.parameters(), "lr": cfg["lr"]}
        ]
    )
    # ===== collect all trainable params (model + graph_prompt) =====
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad] +
        [p for p in graph_prompt.parameters() if p.requires_grad]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype == torch.float16))
    os.makedirs(cfg["save_path"], exist_ok=True)

    def save_adapter(tag: str):
        model.save_pretrained(cfg["save_path"])
        tokenizer.save_pretrained(cfg["save_path"])
        torch.save(
            {
                "state_dict": graph_prompt.state_dict(),
                "prompt_len": int(graph_prompt.prompt_len),
                "num_layers": int(len(graph_prompt.layers)),
                "hidden_dim": int(graph_prompt.proj.in_features),
            },
            os.path.join(cfg["save_path"], "graph_prompt.pt")
        )
        with open(os.path.join(cfg["save_path"], "eval_config.json"), "w") as f:
            json.dump(
                {
                    "model_name_or_path": cfg["model_name_or_path"],
                    "sid_json": cfg["sid_json"],
                    "dataset_path": cfg["dataset_path"],
                    "valid_file": cfg["valid_file"],
                    "test_file": cfg["test_file"],
                    "max_history_items": cfg["max_history_items"],
                    "infer_batch_size": cfg["infer_batch_size"],
                    "topk_list": cfg["topk_list"],
                    "topk_items": cfg["topk_items"],
                    "topm_tokens": cfg["topm_tokens"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logging.info(f"Saved adapter ({tag}) to {cfg['save_path']}")

    def merge_metrics(rec, ndcg):
        m = {}
        m.update(rec or {})
        m.update(ndcg or {})
        return m

    def update_best(best_dict, merged, epoch):
        improved_any = False
        improved = []
        for k, v in merged.items():
            if v is None or (not np.isfinite(v)):
                continue
            if v > best_dict[k]["val"]:
                best_dict[k]["val"] = float(v)
                best_dict[k]["epoch"] = int(epoch)
                improved_any = True
                improved.append(k)
        return improved_any, improved

    es_counter = 0

    # ============================
    # Train
    # ============================
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        graph_prompt.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_sets = batch["target_sets"]

            # negatives sampled per batch
            candidate_ids, label_mat = build_batch_candidate_token_list(
                target_sets,
                device=device
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs, item_emb_table = forward_with_item_graph_prompt(
                model=model,
                graph_prompt=graph_prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                history_items_batch=batch["history_items"],
                itemid2row=itemid2row,
                item_sid_tokens=item_sid_tokens,
                item_sid_mask=item_sid_mask,
                )

                h = outputs.hidden_states[-1][:, -1, :]
                emb = model.get_input_embeddings().weight

                # ===== token set loss =====
                W = emb[candidate_ids]
                logits = h @ W.t()

                loss_bce = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([cfg["pos_weight"]], device=h.device)
                )(logits, label_mat)


                loss_infonce = item_contrastive_loss(
                    h,
                    item_emb_table,   # 👈 �?graph 输出
                    batch["target_items"].to(device),
                    itemid2row,
                )

                loss_item_rank = item_rank_loss(
                    h,
                    emb,
                    batch["target_items"].to(device),
                    itemid2row,
                    item_sid_tokens,
                    item_sid_mask,
                    num_neg_items=cfg["item_rank_neg_items"],
                )

                loss = loss_bce + 0.1 * loss_infonce + cfg["item_rank_alpha"] * loss_item_rank

            if (loss is None) or (not torch.isfinite(loss)):
                msg = f"[FATAL] NaN/Inf loss at epoch={epoch} step={step}"
                print(msg)
                logging.error(msg)
                return

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        cfg["grad_clip"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        cfg["grad_clip"]
                    )
                optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "bce": f"{float(loss_bce.item()):.4f}",
                "rank": f"{float(loss_item_rank.item()):.4f}",
                "avg": f"{(total_loss/(step+1)):.4f}",
            })

            if step % 50 == 0:
                logging.info(
                    f"Epoch {epoch} Step {step}: "
                    f"loss={loss_val:.6f}, "
                    f"loss_bce={float(loss_bce.item()):.6f}, "
                    f"loss_infonce={float(loss_infonce.item()):.6f}, "
                    f"loss_item_rank={float(loss_item_rank.item()):.6f}"
                )

        train_loss = total_loss / max(1, len(train_loader))
        logging.info(f"Epoch {epoch} train_loss={train_loss:.6f}")
        print(f"Epoch {epoch} train_loss={train_loss:.6f}")

        # ============================
        # Validation
        # ============================
        model.eval()
        graph_prompt.eval()
        val_rec, val_ndcg = evaluate(
        model=model,
        graph_prompt=graph_prompt,
        records=valid_records,
        item2sid=item2sid,
        item2suffix=item2suffix,
        itemid2row=itemid2row,
        item_sid_tokens=item_sid_tokens,
        item_sid_mask=item_sid_mask,
        all_sid_tokens_t=all_sid_tokens_t,
        token2items=token2items,
        item_token_count=item_token_count,
        device=device,
        pad_id=pad_id,
        topk_list=cfg["topk_list"],
        topk_items=cfg["topk_items"],
        max_history_items=cfg["max_history_items"],
        infer_batch_size=cfg["infer_batch_size"],
        topm_tokens=cfg["topm_tokens"],
    )
        logging.info(f"VAL Recall: {val_rec}")
        logging.info(f"VAL NDCG  : {val_ndcg}")

        merged_val = merge_metrics(val_rec, val_ndcg)
        val_improved_any, val_improved = update_best(tracker_val, merged_val, epoch)
        es_counter = 0 if val_improved_any else es_counter + 1

        need_test = (cfg["test_on"] == "always") or val_improved_any
        test_improved_any, test_improved = False, []
        if need_test:
            test_rec, test_ndcg = evaluate(
                model=model,
                graph_prompt=graph_prompt,
                records=test_records,
                item2sid=item2sid,
                item2suffix=item2suffix,
                itemid2row=itemid2row,
                item_sid_tokens=item_sid_tokens,
                item_sid_mask=item_sid_mask,
                all_sid_tokens_t=all_sid_tokens_t,
                token2items=token2items,
                item_token_count=item_token_count,
                device=device,
                pad_id=pad_id,
                topk_list=cfg["topk_list"],
                topk_items=cfg["topk_items"],
                max_history_items=cfg["max_history_items"],
                infer_batch_size=cfg["infer_batch_size"],
                topm_tokens=cfg["topm_tokens"],
            )
            logging.info(f"TEST Recall: {test_rec}")
            logging.info(f"TEST NDCG  : {test_ndcg}")
            merged_test = merge_metrics(test_rec, test_ndcg)
            test_improved_any, test_improved = update_best(tracker_test, merged_test, epoch)

        # 这里清一�?
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # save
        if cfg["save_on"] == "val" and val_improved_any:
            save_adapter(f"VAL improved: {val_improved}")
        if cfg["save_on"] == "test" and test_improved_any:
            save_adapter(f"TEST improved: {test_improved}")

        # print best summary
        def best_lines(name, dct):
            lines = [f"[{name} BEST]"]
            for k in metric_names:
                lines.append(f"{k}: {dct[k]['val']:.6f} @epoch {dct[k]['epoch']}")
            return "\n".join(lines)

        logging.info("\n" + best_lines("VAL", tracker_val))
        logging.info("\n" + best_lines("TEST", tracker_test))

        print(best_lines("VAL", tracker_val))
        print(best_lines("TEST", tracker_test))

        if es_counter >= cfg["early_stop"]:
            logging.info("Early stopping triggered.")
            break

    logging.info("DONE.")


if __name__ == "__main__":
    main()
