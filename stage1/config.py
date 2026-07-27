import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Luxury')
    parser.add_argument(
        '--root',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed'),
        help='Directory containing one subdirectory per processed dataset.',
    )
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default=None,
        help='Optional Hugging Face cache directory.',
    )
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--semantic_quality_threshold', type=float, default=0.95)
    parser.add_argument('--pruning_pool_size', type=int, default=64)
    parser.add_argument('--distinctiveness_lambda', type=float, default=0.05)
    parser.add_argument('--candidate_prefix_tokens', type=int, default=8)
    parser.add_argument('--max_collision_expand_rounds', type=int, default=8)
    parser.add_argument('--functional_layer_idx', type=int, default=-1)
    parser.add_argument('--min_tokens', type=int, default=2)
    parser.add_argument('--max_tokens', type=int, default=16)
    return parser.parse_args()
