import io
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).absolute().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"


class Tee:
    def __init__(self, fd1: io.TextIOWrapper, fd2: io.TextIOWrapper):
        self._fd1 = fd1
        self._fd2 = fd2

    def __enter__(self) -> "Tee":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._fd1 != sys.stdout and self._fd1 != sys.stderr:
            self._fd1.close()
        if self._fd2 != sys.stdout and self._fd2 != sys.stderr:
            self._fd2.close()

    def write(self, text: str) -> None:
        self._fd1.write(text)
        self._fd2.write(text)
        self.flush()

    def flush(self) -> None:
        self._fd1.flush()
        self._fd2.flush()


def get_episode(path: Path, idx: int) -> Dict[str, np.ndarray]:
    ep: Dict[str, np.ndarray] = {}

    with np.load(path) as npz:
        for key in npz:
            if not key.startswith(f"{idx}/"):
                continue

            new_key = key.split("/")[1]
            ep[new_key] = npz[key]

    return ep


def get_episode_indices(path: Path) -> List[int]:
    """
    Grabs all the episode indices contained in the npz data
    :param path: path to .npz file
    :returns: list of episode indices
    """
    with np.load(path) as npz:
        indices: List[int] = []
        for key in npz:
            ep_index = int(key.split("/")[0])
            if ep_index in indices:
                continue

            indices.append(ep_index)

        return indices
