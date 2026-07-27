"""Command-line configuration for Stage 2."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser("Parallel token-set SID prediction")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "Luxury"),
    )
    parser.add_argument("--train_file", type=str, default="Luxury.train.inter")
    parser.add_argument("--valid_file", type=str, default="Luxury.valid.inter")
    parser.add_argument("--test_file", type=str, default="Luxury.test.inter")
    parser.add_argument(
        "--sid_json",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "sids"
            / "Luxury"
            / "Luxury.sid-set-token-weight-stage1.json"
        ),
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=str(
            PROJECT_ROOT / "logs" / "LLMSetSID" / "Luxury" / "itemrank.log"
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(
            PROJECT_ROOT / "checkpoints" / "LLMSetSID" / "Luxury" / "itemrank"
        ),
    )

    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use_bf16", type=bool, default=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--max_history_items", type=int, default=20)

    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--item_rank_alpha", type=float, default=0.8)
    parser.add_argument("--item_rank_neg_items", type=int, default=64)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        nargs="+",
        type=str,
        default=["q_proj", "v_proj"],
    )

    parser.add_argument("--infer_batch_size", type=int, default=32)
    parser.add_argument(
        "--topk_list",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
    )
    parser.add_argument("--topk_items", type=int, default=20)
    parser.add_argument(
        "--topm_tokens",
        type=int,
        default=0,
        help="Top-M token approximation; <= 0 scores all semantic SID tokens.",
    )
    parser.add_argument(
        "--test_on",
        type=str,
        choices=["improve", "always"],
        default="improve",
    )
    parser.add_argument(
        "--save_on",
        type=str,
        choices=["val", "test"],
        default="test",
    )

    return parser.parse_args()
