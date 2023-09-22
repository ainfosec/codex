from enum import Enum, auto
from itertools import tee
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from utils.cv import gridworld
from utils.cv.gridworld import EntIdx, EntState, EntType

# Since we aren't on python 3.10 yet, use this try/except to define the
# pairwise function
try:
    from itertools import pairwise
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


class AnnoState(Enum):
    """
    Enum indicating whether an annotation corresponds to a VALID or INVALID state/move.
    This enum is used to help color code annotations on plots.
    """

    VALID = auto()
    INVALID = auto()


def get_annotations(
    frames: np.ndarray,
    ep_num: int,
    cell_size: int = 8,
    player_data: Optional[List[Dict]] = None,
    goal_data: Optional[List[Dict]] = None,
    key_door_data: Optional[np.ndarray] = None,
) -> List[List[Tuple[str, AnnoState]]]:
    """
    Given an array of gameplay frames, return a list of annotations for each frame of the video

    Params:
        frames - (np.ndarray) an array of frames from an episode
        ep_num - (int) number of the episode being annotated
        cell_size - (int) pixel size of cell in video frame
        player_data - (Optional[List[Dict]]) data returned from the find_player() function
        goal_data - (Optional[List[Dict]]) data returned from the find_goal() function
        key_door_data - (Optional[np.ndarray]) data returned from the find_key_and_door() function

    Return:
        (List[List[Tuple[str, AnnoState]]]) - List of Lists where each List[i] is a List of tuples of
        the form (annotation string, AnnoState enum) corresponding to the tags for the i-th frame
    """

    dir_dict: Dict[EntState, str] = {
        EntState.UNKNOWN: "a non-cardinal direction",
        EntState.FACING_RIGHT: "right",
        EntState.FACING_UP: "up",
        EntState.FACING_LEFT: "left",
        EntState.FACING_DOWN: "down",
    }
    dir2angle: Dict[EntState, int] = {
        EntState.FACING_RIGHT: 0,
        EntState.FACING_UP: 90,
        EntState.FACING_LEFT: 180,
        EntState.FACING_DOWN: 270,
    }

    tags = []
    try:
        # Grab player/key/goal data from the decoded video if it wasn't provided to the function
        if not player_data:
            player_data = gridworld.find_player(frames, cell_size)
        if not goal_data:
            goal_data = gridworld.find_goal(frames, cell_size)
        if not key_door_data:
            key_door_data = gridworld.find_key_and_door(frames, goal_data, cell_size)

        # Generate tags for this episode.
        prev_loc: Optional[Tuple[int, int]] = None
        prev_direction: Optional[EntState] = None
        # The door always starts closed.
        prev_door_state: EntState = EntState.CLOSED
        # The key always starts on the map.
        prev_key_on_map: bool = True

        # Handle player and goal data because every minigrid game has those two elements
        for step, step_data in enumerate(zip(player_data, goal_data)):
            step_tags = []  # The list to hold annotations for this frame
            player_loc = None
            direction = None
            goal_loc = None

            player, goal = step_data

            # Sanity check assertions. You need as many player locations as your have player directions
            assert EntType(player["EntType"]) == EntType.PLAYER
            player_locs = player["locs"]
            player_directions = player["directions"]
            assert len(player_locs) == len(player_directions)

            # Annotate player actions
            if len(player_locs) == 0:
                # No players found
                step_tags.append(
                    (f"{step}: The player has disappeared.", AnnoState.INVALID)
                )
                prev_loc = None
                prev_direction = None
            elif len(player_locs) == 1:
                # Single player located
                x: int = player_locs[0][EntIdx.x]
                y: int = player_locs[0][EntIdx.y]
                player_loc: Tuple[int, int] = (x, y)
                direction: EntState = EntState(player_directions[0])
                step_tags.append(
                    (f"{step}: The player is at ({x}, {y}).", AnnoState.VALID)
                )
                step_tags.append(
                    (
                        f"{step}: The player is facing {dir_dict[direction]}.",
                        AnnoState.VALID
                        if direction != EntState.UNKNOWN
                        else AnnoState.INVALID,
                    )
                )
            else:
                # Multiple players located
                pos_string = " and ".join([f"({x}, {y})" for x, y in player_locs])
                step_tags.append(
                    (
                        f"{step}: The player is in multiple positions at {pos_string}.",
                        AnnoState.INVALID,
                    )
                )
                for pos, p_dir in zip(player_locs, player_directions):
                    step_tags.append(
                        (
                            f"{step}: The player at ({pos[0]}, {pos[1]}) is facing {dir_dict[p_dir]}.",
                            AnnoState.INVALID,
                        )
                    )
                prev_loc = None
                prev_direction = None

            # Check if the player has moved forward.
            if prev_loc is not None:
                delta_x = player_loc[0] - prev_loc[0]
                delta_y = player_loc[1] - prev_loc[1]

                # The player can only move in one of the cardinal directions,
                # so at most only one delta can be non-zero.
                msg = f"{prev_loc} -> {player_loc}"
                if delta_x != 0 and delta_y != 0:
                    step_tags.append(
                        (f"{step}: Unexpected diagonal move: {msg}", AnnoState.INVALID)
                    )
                # The player can only move one cell over at a time, so if a
                # delta is non-zero, it must be one.
                if any(
                    [
                        delta_x != 0 and abs(delta_x) > 1,
                        delta_y != 0 and abs(delta_y) > 1,
                    ]
                ):
                    step_tags.append(
                        (
                            f"{step}: The player moves more than one cell: {msg}",
                            AnnoState.INVALID,
                        )
                    )

                # Check if player moved forward in a single direction
                if (delta_x != 0 and delta_y == 0) or (delta_x == 0 and delta_y != 0):
                    step_tags.append(
                        (f"{step}: The player moves forward.", AnnoState.VALID)
                    )

            prev_loc = player_loc

            # Check if the player has turned.
            if all(
                [
                    prev_direction is not None,
                    prev_direction != EntState.UNKNOWN,
                    direction != EntState.UNKNOWN,
                    direction != prev_direction,
                ]
            ):
                delta = dir2angle[direction] - dir2angle[prev_direction]
                if abs(delta) == 270:
                    delta *= -1
                elif abs(delta) != 90:
                    step_tags.append(
                        (
                            f"{step}: Unexpected angle delta at step {step}: {delta}",
                            AnnoState.INVALID,
                        )
                    )
                if delta > 0:
                    step_tags.append(
                        (f"{step}: The player turns left.", AnnoState.VALID)
                    )
                else:
                    step_tags.append(
                        (f"{step}: The player turns right.", AnnoState.VALID)
                    )
            prev_direction = direction

            # Annotate goal
            assert EntType(goal["EntType"]) == EntType.GOAL
            goal_locs = goal["locs"]
            if len(goal_locs) == 0:
                # No goal found
                step_tags.append(
                    (f"{step}: The goal has disappeared.", AnnoState.INVALID)
                )
                goal_loc = None
            elif len(goal_locs) == 1:
                # One goal found
                goal_x: int = goal_locs[0][EntIdx.x]
                goal_y: int = goal_locs[0][EntIdx.y]
                goal_loc: Tuple[int, int] = (goal_x, goal_y)
                step_tags.append(
                    (f"{step}: The goal is at ({goal_x}, {goal_y}).", AnnoState.VALID)
                )
            else:
                # Multiple goals found
                goal_string = " and ".join([f"({x}, {y})" for x, y in goal_locs])
                step_tags.append(
                    (
                        f"{step}: The goal is in multiple locations at {goal_string}.",
                        AnnoState.INVALID,
                    )
                )
                goal_loc = None

            # If player has reached the goal
            if goal_loc is not None and player_loc == goal_loc:
                step_tags.append(
                    (f"{step}: The player reached the goal.", AnnoState.VALID)
                )

            # If this is a room-key episode then add key and door annotations
            if key_door_data is not None:
                step_data = key_door_data[step]
                key_on_map: bool = False
                for entity_data in step_data:
                    if np.all(entity_data == -1):
                        continue
                    x: int = entity_data[EntIdx.x]
                    y: int = entity_data[EntIdx.y]
                    unit_type = entity_data[EntIdx.unit_type]

                    assert EntType(unit_type) in [EntType.KEY, EntType.DOOR], unit_type

                    if EntType(unit_type) == EntType.KEY:
                        key_on_map = True
                        step_tags.append(
                            (f"{step}: The key is at ({x}, {y}).", AnnoState.VALID)
                        )
                    else:
                        state = EntState(entity_data[EntIdx.state])
                        step_tags.append(
                            (f"{step}: The door is at ({x}, {y}).", AnnoState.VALID)
                        )
                        if state == EntState.CLOSED:
                            step_tags.append(
                                (f"{step}: The door is closed.", AnnoState.VALID)
                            )
                        elif state == EntState.OPEN:
                            step_tags.append(
                                (f"{step}: The door is open.", AnnoState.VALID)
                            )
                        else:
                            step_tags.append(
                                (
                                    f"{step}: The state of the door is unknown.",
                                    AnnoState.INVALID,
                                )
                            )

                        if (
                            prev_door_state == EntState.CLOSED
                            and state == EntState.OPEN
                        ):
                            step_tags.append(
                                (f"{step}: The player opens the door.", AnnoState.VALID)
                            )
                        elif (
                            prev_door_state == EntState.OPEN
                            and state == EntState.CLOSED
                        ):
                            step_tags.append(
                                (
                                    f"{step}: The player closes the door.",
                                    AnnoState.VALID,
                                )
                            )

                        prev_door_state = state

                if not key_on_map:
                    if prev_key_on_map:
                        step_tags.append(
                            (f"{step}: The player picks up the key.", AnnoState.VALID)
                        )
                    step_tags.append(
                        (f"{step}: The key has been picked up.", AnnoState.VALID)
                    )

                else:
                    if not prev_key_on_map:
                        step_tags.append(
                            (f"{step}: The player drops the key.", AnnoState.VALID)
                        )
                prev_key_on_map = key_on_map

            tags.append(step_tags)

    except:
        tqdm.write(f"Could not extract symbolic data from episode {ep_num}")
        raise

    return tags
