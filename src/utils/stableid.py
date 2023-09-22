import json
import pathlib
from enum import Enum
from typing import Dict, List, Union

_StableID = Dict[str, List[Dict[str, Union[int, str]]]]

# Abilities: Enum
Buffs: Enum
Effects: Enum
Units: Enum
Upgrades: Enum


def _get_enum(name: str, id_mappings: _StableID) -> Enum:
    return Enum(name, ((d["name"], d["id"]) for d in id_mappings[name]))


def _load_stableid() -> None:
    global Buffs, Effects, Units, Upgrades

    # look for id file
    filename = "stableid.json"
    path = pathlib.Path().resolve()
    if not (path / filename).exists():
        for _p in path.parents:
            if (_p / filename).exists():
                path = _p
                break
        else:
            raise FileNotFoundError(
                f"stableid.json looked for in {_p} parents but not found."
            )

    with open(path / filename, "r") as fp:
        stableid: _StableID = json.load(fp)

    for key, entries in stableid.items():
        # For now, we skip Abilities due to name collisions.
        if key == "Abilities":
            continue
        for d in entries:
            assert (
                len(d["name"].split()) == 1
            ), f"name {d['name']} in {key} has whitespace."

    # Abilities = _get_enum("Abilities", stableid)
    Buffs = _get_enum("Buffs", stableid)
    Effects = _get_enum("Effects", stableid)
    Units = _get_enum("Units", stableid)
    Upgrades = _get_enum("Upgrades", stableid)


_load_stableid()
