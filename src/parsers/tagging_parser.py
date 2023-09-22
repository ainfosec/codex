from argparse import ArgumentParser
from pathlib import Path

from utils import OUTPUT_DIR


def add_tagging_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "env",
        type=str,
        choices=["minigrid", "sc2", "crafter"],
        help="The environment in which episodes were collected",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to data (can be file or directory)",
    )
    parser.add_argument(
        "output_dir",
        default=OUTPUT_DIR / "tags",
        type=Path,
        help='Output directory (relative to "outputs" if not absolute)',
    )
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        help="Glob pattern (ignored if input path is not a directory)",
    )
