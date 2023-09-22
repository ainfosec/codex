from enum import Enum, IntEnum, auto
from typing import Dict, List, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.cv.common import get_centroid

PLAYER_HSV_FILTER_1 = ((0, 100, 50), (10, 255, 255))
PLAYER_HSV_FILTER_2 = ((170, 100, 50), (180, 255, 255))
GOAL_HSV_FILTER = ((50, 150, 150), (70, 255, 255))
KEY_DOOR_HSV_FILTER = ((27, 220, 100), (32, 255, 255))
ANGULAR_THRESHOLD_DEG = 15


class EntIdx(IntEnum):
    x = 0
    y = 1
    state = 2
    unit_type = 3


class EntType(Enum):
    PLAYER = auto()
    GOAL = auto()
    KEY = auto()
    DOOR = auto()


class EntState(Enum):
    UNKNOWN = auto()
    FACING_RIGHT = auto()
    FACING_UP = auto()
    FACING_LEFT = auto()
    FACING_DOWN = auto()
    OPEN = auto()
    CLOSED = auto()


def find_player(video: np.ndarray, cell_size: int = 8, debug=False) -> List[Dict]:
    """
    Given a video of gameplay, finds the player's location and orientation for each frame
    and returns a list of those annotations for each frame

    Params:
        video - (np.ndarray) an array of frames from an episode
        cell_size - (int) the pixel width of a game cell
        debug - (bool) whether or not to draw contour and bounding box of player on frame

    Return:
        (List[Dict]) - List of Dicts where List[i] contains the player annotations for frame i.
                        The Dict has the following keys:
                        - "locs": List of (x, y) tuples for the location of the player(s) found
                        - "directions": List of EntState Enums for direction of the player(s) found
                        - "EntType": The EntType.PLAYER value

                        The length of lists for the "locs" and "direction" keys are gauaranteed to
                        be the same. If no players are found, the list within a Dict will be empty.
                        If multiple players are found, len(list) within a Dict will be > 1.
    """

    annotated_frames = []
    for frame in video:
        # Create HSV mask and find player contours
        frame_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        mask_1 = cv.inRange(frame_hsv, *PLAYER_HSV_FILTER_1)
        mask_2 = cv.inRange(frame_hsv, *PLAYER_HSV_FILTER_2)
        mask = mask_1 + mask_2
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # For each player contour add centroid and direction to the frames list
        # Each frame can have multiple or no contours because the decoded Dreamer latent state
        # sometimes has zero or multiple players
        frame_locs = []
        frame_directions = []
        for contour in contours:
            if debug:
                # Add bounding rect and contours to image for help in debugging CV issues
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 0)
                cv.drawContours(frame, contour, -1, (255, 0, 255), -1)

            x, y = get_centroid(contour)
            frame_locs.append((x // cell_size, y // cell_size))
            moments = cv.moments(contour)
            frame_directions.append(get_direction(moments))

        # Create dict of player locations and directions and append to list
        frame_dict = {
            "locs": frame_locs,
            "directions": frame_directions,
            "EntType": EntType.PLAYER.value,
        }
        annotated_frames.append(frame_dict)

    return annotated_frames


def find_goal(video: np.ndarray, cell_size: int = 8) -> List[Dict]:
    """
    Given a video of gameplay, finds the goal's location for each frame and returns a list of
    those annotations for each frame

    Params:
        video - (np.ndarray) an array of frames from an episode
        cell_size - (int) the pixel width of a game cell

    Return:
        (List[Dict]) - List of Dicts where each List[i] contains the goal annotations for frame i.
                        The Dict has the following keys:
                        - "locs": List of (x, y) tuples for the location of the goal(s) found
                        - "EntType": The EntType.GOAL value

                        If no goals are found, the list for the "locs" key will be empty.
                        If multiple goals are found, len(list) for the "locs" key will be > 1.
    """
    annotated_frames = []
    for frame in video:
        # Create HSV mask and find goal contours
        frame_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        mask = cv.inRange(frame_hsv, *GOAL_HSV_FILTER)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # For each contour append its centroid to this frame list
        frame_locs = []
        for contour in contours:
            x, y = get_centroid(contour)
            frame_locs.append((x // cell_size, y // cell_size))

        # Create frame dict and append to episode list
        frame_dict = {"locs": frame_locs, "EntType": EntType.GOAL.value}
        annotated_frames.append(frame_dict)

    return annotated_frames


def get_direction(moments: Dict[str, float]) -> EntState:
    mu11_prime = moments["mu11"] / (moments["m00"] + 1e-5)
    mu20_prime = moments["mu20"] / (moments["m00"] + 1e-5)
    mu02_prime = moments["mu02"] / (moments["m00"] + 1e-5)
    theta = 0.5 * np.arctan2(2 * mu11_prime, mu20_prime - mu02_prime)

    # Due to central moment invariants, we cannot tell the player's
    # direction from theta alone. We can only tell whether the player
    # is horizontal or vertical. To distinguish left from right and up
    # from down, we look at the sign of the appropriate third-order
    # central moment.
    theta_deg = np.rad2deg(np.abs(theta))
    if -ANGULAR_THRESHOLD_DEG <= theta_deg <= ANGULAR_THRESHOLD_DEG:
        # If the third-order moment is close to zero, then we cannot
        # tell if the player is facing left or right.
        if abs(moments["mu30"]) < 1e-4:
            # print(f'abs(moments["mu30"]) < 1e-4, abs(moments["mu30"]) = {abs(moments["mu30"])}')
            return EntState.UNKNOWN
        # Player is horizontal.
        if moments["mu30"] > 0:
            return EntState.FACING_RIGHT
        else:
            return EntState.FACING_LEFT
    elif 90 - ANGULAR_THRESHOLD_DEG <= theta_deg <= 90 + ANGULAR_THRESHOLD_DEG:
        # If the third-order moment is close to zero, then we cannot
        # tell if the player is facing up or down.
        if abs(moments["mu03"]) < 1e-4:
            # print(f'abs(moments["mu03"]) < 1e-4, abs(moments["mu03"]) = {abs(moments["mu03"])}')
            return EntState.UNKNOWN
        # Player is vertical.
        if moments["mu03"] > 0:
            # Remember that for images, the y-axis points downward.
            return EntState.FACING_DOWN
        else:
            return EntState.FACING_UP

    # If theta_deg is close to neither zero nor 90, then the player is
    # not facing a cardinal direction.
    # print(f'Theta is not close to 0 or 90 degrees, theta = {theta_deg}')
    return EntState.UNKNOWN


def find_key_and_door(
    video: np.ndarray, goal_data: List[Dict], cell_size: int = 8
) -> Optional[np.ndarray]:
    symbols: List[Optional[np.ndarray]] = []
    for i, frame in enumerate(video):
        frame_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
        mask = cv.inRange(frame_hsv, *KEY_DOOR_HSV_FILTER)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            symbols.append(None)
            continue

        # We need to distinguish between a key, a closed door, and an
        # open door. Also, sometimes the player standing on the goal gets caught
        # by the filter because it looks brown-ish when decoded from latent state
        symbols_frame: List[List[int]] = []
        for contour in contours:
            # Handle case where the player on goal is caught by key/door filter
            # by checkiing if the contour is on top of a goal's location
            loc = get_centroid(contour)
            loc_norm = (
                loc[0] // cell_size,
                loc[1] // cell_size,
            )  # convert to grid coords
            if loc_norm in [
                (x // cell_size, y // cell_size) for x, y in goal_data[i]["locs"]
            ]:
                continue

            # Otherwise this is a real key/door so try to decode it
            try:
                symbol: Optional[List[int]] = decode_key_door(contour, cell_size)
                if symbol:
                    symbols_frame.append(symbol)
            except RuntimeError as e:
                plt.ion()
                frame_copy = frame.copy()
                cv.drawContours(frame_copy, contour, -1, (255, 0, 255), 1)
                plt.figure()
                plt.imshow(frame_copy)
                plt.show()
                raise RuntimeError(f"Unexpected contour at frame {i}: {e}")

        # If we only caught contours that were the player on the goal, append None.
        # Otherwise stack the symbols frame and append it
        if len(symbols_frame) == 0:
            symbols.append(None)
        else:
            symbols.append(np.stack(symbols_frame))

    if all(a is None for a in symbols):
        return None

    max_entities = max(a.shape[0] for a in symbols if a is not None)
    symbolic_data = np.full((len(video), max_entities, len(EntIdx)), -1, dtype=np.int64)
    for i, a in enumerate(symbols):
        if a is None:
            continue
        num_entities, num_features = a.shape
        symbolic_data[i, :num_entities, :num_features] = a

    return symbolic_data


def decode_key_door(contour: np.ndarray, cell_size: int) -> Optional[List[int]]:
    x, y, area = get_centroid(contour, include_area=True)
    symbol = [-1] * len(EntIdx)
    symbol[EntIdx.x] = x // cell_size
    symbol[EntIdx.y] = y // cell_size
    # First, count the number of points in the contour. An open
    # door should have four points, and a closed door two.
    if contour.shape == (4, 1, 2):
        # To avoid a false-positive, check that the four points
        # correspond to a square with an area of 49.
        p1, p2, p3, p4 = np.squeeze(contour)
        if (p1[0], p3[0], p1[1], p2[1]) != (p2[0], p3[0], p4[1], p3[1]) or np.around(
            area
        ) != 49.0:
            # If the 4 points don't line up like a square then this is most likely a key
            symbol[EntIdx.state] = EntState.UNKNOWN.value
            symbol[EntIdx.unit_type] = EntType.KEY.value
        else:
            symbol[EntIdx.state] = EntState.CLOSED.value
            symbol[EntIdx.unit_type] = EntType.DOOR.value
    # An open door should have two points
    elif contour.shape == (2, 1, 2):
        # To avoid a false-positive, we check that the two
        # points are aligned vertically.
        p1, p2 = np.squeeze(contour)
        if np.sum(np.square(np.subtract(p2, p1))) == 1 or p1[0] != p2[0]:
            # If the 2 points don't line up then this is most likely a key
            symbol[EntIdx.state] = EntState.UNKNOWN.value
            symbol[EntIdx.unit_type] = EntType.KEY.value
        else:
            symbol[EntIdx.state] = EntState.OPEN.value
            symbol[EntIdx.unit_type] = EntType.DOOR.value
    # Otherwise we assume that this contour corresponds to a key
    else:
        symbol[EntIdx.state] = EntState.UNKNOWN.value
        symbol[EntIdx.unit_type] = EntType.KEY.value
    return symbol
