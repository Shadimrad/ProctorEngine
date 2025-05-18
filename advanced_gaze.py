from __future__ import annotations

"""Wrapper around open‑source gaze tracking libraries.

This module attempts to use the more accurate `LaserGaze` library if it is
installed. If not, it falls back to ``gaze_tracking``. Both implementations
return normalized ``(x_ratio, y_ratio)`` coordinates in ``[0, 1]``.
"""

from typing import Optional, Tuple
import cv2

try:  # Prefer LaserGaze when available
    from lasergaze import LaserGaze
    _LaserGazeType = LaserGaze  # Alias for type checkers
except Exception:  # pragma: no cover - optional dependency
    _LaserGazeType = None
    LaserGaze = None

try:  # Fallback implementation
    from gaze_tracking import GazeTracking
except Exception:  # pragma: no cover - optional dependency
    GazeTracking = None


class AdvancedGaze:
    """High level gaze estimator using available open‑source libraries."""

    def __init__(self) -> None:
        if _LaserGazeType is not None:
            self._impl = "lasergaze"
            self._gaze = _LaserGazeType()
        elif GazeTracking is not None:
            self._impl = "gaze_tracking"
            self._gaze = GazeTracking()
        else:  # pragma: no cover - should not happen in normal installs
            raise ImportError("No supported gaze tracking library found")

    def process(self, frame: "np.ndarray") -> Optional[Tuple[float, float]]:
        """Return ``(x_ratio, y_ratio)`` if the gaze is detected."""

        if self._impl == "lasergaze":
            # ``LaserGaze`` returns the gaze ratios directly from ``refresh``
            result = self._gaze.refresh(frame)
            if result is None:
                return None
            x_ratio, y_ratio = result
        else:
            self._gaze.refresh(frame)
            x_ratio = self._gaze.horizontal_ratio()
            y_ratio = self._gaze.vertical_ratio()
            if x_ratio is None or y_ratio is None:
                return None

        return float(x_ratio), float(y_ratio) 