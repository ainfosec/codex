import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from data.tagging import gridworld, sc2
from utils import get_episode, get_episode_indices


def main(args: argparse.Namespace) -> None:
    # Extract CLI arguments.
    env_type: str = args.env
    input_path: Path = args.input_path.expanduser()
    output_dir: Path = args.output_dir.expanduser()
    glob_pattern: Optional[str] = args.glob or "*"

    data_paths: List[Path]
    if input_path.is_dir():
        data_paths = []
        for path in input_path.glob(glob_pattern):
            if not path.is_dir():
                data_paths.append(path)
    else:
        data_paths = [input_path]

    output_dir.mkdir(parents=True, exist_ok=True)

    generator: Callable[[Path], List]
    if env_type == "minigrid":
        generator = gen_tags_gridworld
    elif env_type == "sc2":
        generator = gen_tags_sc2
    elif env_type == "crafter":
        print("Tagging for Crafter not implemented.")
        return
    else:
        raise ValueError(f"Unrecognized environment: {env_type}")

    for path in tqdm(data_paths, "Tagging batches", leave=False):
        batch_tags = generator(path)
        with open(output_dir / f"{path.stem}.json", "w") as fp:
            json.dump(batch_tags, fp, indent=2)


def gen_tags_gridworld(path: Path) -> List[Dict]:
    episode_indices: List[int] = get_episode_indices(path)
    batch_tags: List[Dict[str, List[List[str]]]] = []
    for idx in tqdm(sorted(episode_indices), "Tagging episodes", leave=False):
        ep = get_episode(path, idx)
        obs_tags = gridworld.get_annotations(ep["image"], idx)
        ep_tags = {"ep_tags": [[tag[0] for tag in f_tags] for f_tags in obs_tags]}
        batch_tags.append(ep_tags)
    return batch_tags


def gen_tags_sc2(path: Path) -> List[Dict]:
    episode_indices: List[int] = get_episode_indices(path)
    batch_tags: List[Dict[str, List[List[str]]]] = []
    for idx in tqdm(sorted(episode_indices), "Tagging episodes", leave=False):
        ep = get_episode(path, idx)
        feature_units = ep["feature_units"]
        trajectory_walker = sc2.TrajectoryWalker(
            [
                sc2.EntityMoves,
                sc2.EntityMovesRelative,
                sc2.EntityReached,
                sc2.Groups,
            ]
        )
        for ts, data in enumerate(feature_units):
            num_entities = np.count_nonzero(data[:, 0] >= 0)
            data = data[:num_entities].T
            trajectory_walker(ts, data)

        ep_tags: List[str] = trajectory_walker.export()

        batch_tags.append({"ep_tags": ep_tags})
    return batch_tags
