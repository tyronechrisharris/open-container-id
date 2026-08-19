import cv2
import numpy as np


def classify_orientation(
    crop: np.ndarray, horizontal_thresh: float = 1.5, tall_thresh: float = 1.2
) -> str:
    """
    Classifies the orientation based on aspect ratio heuristic.
    """
    if crop is None or crop.size == 0:
        return "near_square"

    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "near_square"

    if w / h >= horizontal_thresh:
        return "horizontal"
    elif h / w >= tall_thresh:
        return "tall"
    else:
        return "near_square"


def generate_transform_candidates(
    crop: np.ndarray, orientation: str
) -> dict[str, np.ndarray]:
    """
    Generates transform candidates based on the layout orientation.
    Returns a dict mapping transform name to the transformed numpy array.
    """
    candidates: dict[str, np.ndarray] = {}

    if crop is None or crop.size == 0:
        return candidates

    # All crops get identity by default
    candidates["identity"] = crop.copy()

    if orientation == "horizontal":
        # Usually horizontal means just reading left to right.
        # But if it's upside down, we can optionally test rotate_180.
        candidates["rotate_180"] = cv2.rotate(crop, cv2.ROTATE_180)

        # We can also add a contrast enhanced version
        # as a candidate for tricky lighting conditions.
        # Simple histogram equalization for contrast enhancement:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        enhanced = cv2.equalizeHist(gray)
        if len(crop.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        candidates["contrast_enhanced"] = enhanced

    elif orientation == "tall":
        candidates["rotate_90"] = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        candidates["rotate_270"] = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # We do NOT include vertical_unstack here since it is deferred to Issue 24.

    elif orientation == "near_square":
        candidates["rotate_90"] = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        candidates["rotate_180"] = cv2.rotate(crop, cv2.ROTATE_180)
        candidates["rotate_270"] = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return candidates
