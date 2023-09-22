from typing import Tuple, Union

import cv2 as cv
import numpy as np


def array2image(image: np.ndarray) -> np.ndarray:
    rescaled = (image + 0.5) * 255
    clipped = np.clip(rescaled, 0, 255)
    converted_to_uint8 = clipped.astype(np.uint8)
    return converted_to_uint8


def get_centroid(
    contour: np.ndarray,
    *,
    include_area: bool = False,
) -> Union[Tuple[int, int], Tuple[int, int, float]]:
    """
    Compute the centroid of the given contour using its moments.

    This approach is more robust than simply taking the mean x- and
    y-coordinates.

    :param contour: A collection of points denoting a detected shape.
    :returns: The centroid of the given contour.
    """
    M = cv.moments(contour)
    area = M["m00"]
    if area == 0:
        cx, cy = np.mean(np.reshape(contour, (-1, 2)), axis=0)
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    cx = int(cx)
    cy = int(cy)
    if include_area:
        return cx, cy, area
    else:
        return cx, cy
