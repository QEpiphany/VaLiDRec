"""Command-line entry point for Stage 1."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_device

from .config import parse_args
from .data import preprocess_text
from .pipeline import generate_variable_length_sid


def main():
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    item_text_list = preprocess_text(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    model.eval()
    generate_variable_length_sid(args, item_text_list, tokenizer, model)


if __name__ == "__main__":
    main()
