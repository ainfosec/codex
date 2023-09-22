from argparse import ArgumentParser
from pathlib import Path

from utils import OUTPUT_DIR


def add_summarization_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset with episode tags.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=OUTPUT_DIR / "semantic_sum.txt",
        type=Path,
        help="Path to save results.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Tokenizer max sequence length.",
    )
    parser.add_argument(
        "--save-vecs",
        action="store_true",
        help="Save episode tags and vectors.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of UMAP dimensions.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Number of UMAP nearest neighbors.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="Minimum HDBSCAN cluster size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum HDBSCAN cluster samples for core points.",
    )
    parser.add_argument(
        "--sum-threshold",
        type=float,
        default=0.6,
        help="Cosine Similarity summary diversity threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Set a random state seed (for reproducibility).",
    )
