# ProctorEngine

This project provides simple utilities for estimating user gaze relative to a computer screen. Calibration is performed with a five‑point procedure after which the system can report where on the display the user is looking. The implementation relies on MediaPipe's face mesh for detecting facial landmarks and provides a small Kalman smoother for stable gaze points.

## Usage

Calibrate a new screen plane:

```
python main.py calibrate --out plane.json --screen-w <width_mm> --screen-h <height_mm>
```

Track gaze using the saved plane:

```
python main.py track --plane plane.json
```

An optional wrapper around the `gaze_tracking` library is available in `advanced_gaze.py` for more sophisticated detection. Install the extra dependency and integrate it as needed.
