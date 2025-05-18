from __future__ import annotations

"""Wrapper around the `gaze_tracking` library."""

from typing import Optional, Tuple
import cv2
from gaze_tracking import GazeTracking


class AdvancedGaze:
    """High level interface to the `gaze_tracking` gaze estimator."""

    def __init__(self) -> None:
        self._gaze = GazeTracking()

    def process(self, frame: "np.ndarray") -> Optional[Tuple[float, float]]:
        """Return (x_ratio, y_ratio) if the gaze is detected.

        The returned coordinates are normalized to ``[0, 1]`` in both axes.
        ``None`` is returned if the library fails to detect the gaze.
        """
        self._gaze.refresh(frame)
        x_ratio = self._gaze.horizontal_ratio()
        y_ratio = self._gaze.vertical_ratio()
        if x_ratio is None or y_ratio is None:
            return None
        return float(x_ratio), float(y_ratio)
