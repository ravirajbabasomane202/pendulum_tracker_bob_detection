# Object Trajectory Analysis (Flask + OpenCV)

Video of a swinging bob (any colour, shape or size) → confirm detection → track → analyse path shape, periods, θ, amplitudes. The tracker supports handheld cameras, camera zoom, brief occlusions, and appearance learning.

## Features
- **Tracking**: learns bob appearance (any colour / shape / size) via auto-detect + confirmation dialog or click calibration; HSV histogram + shape/size/prediction tracking; up to 15 interpolated lost frames
- **Moving camera**: estimates global motion from background corners with optical flow + RANSAC; stabilization is on by default
- **Zoom normalization**: uses the ball's apparent radius to normalize the trajectory; enabled by default and independently configurable
- **Diagnostics**: one tracking-check image plus a contact sheet of sampled frames across the clip
- **Coordinates**: raw, camera-stabilized, and zoom-normalized positions are included in both CSV and JSON exports
- **Trajectory**: x(t), y(t), z est., y-vs-x path
- **Shape class**: line · circle · ellipse · precessing ellipse · Lissajous · decaying amplitude
- **Periods / frequencies**: Tx, Ty, fx, fy (peaks + FFT), Δf
- **Orientation**: θ(t), θ drift rate, eccentricity (minor/major)
- **Amplitudes**: Ax, Ay, amplitude decay
- **Theory check**: T = 2π√(L/g) vs measured (optional L, g)
- **Custom params**: cm/px, start/end, mass, length, g, trial ID, notes
- **UI**: single-screen live values, plots, 3-D view synced with video
- **Export**: full **CSV** and **JSON** (parameters + summary + time series + sliding θ windows), plus result video

## Analysis dashboard (`/dashboard`)
Open it with the **Dashboard** button in the lab header (after an analysis), or at `/dashboard`. You can also load a
previously exported `trajectory_full.csv` / `.json` with **Load CSV / JSON export** - no video needed.

- **Data source**: analysis series, stabilised, zoom-normalised or raw coordinates; overlay original vs filtered
- **Filters** (scipy, server-side): moving average, Savitzky-Golay, Gaussian, median, Butterworth low/high/band-pass
  (zero-phase), LOWESS (local linear, tricube). Irrelevant sliders dim automatically; invalid cut-offs are corrected with a warning
- **Outlier removal**: z-score, IQR, Hampel - measured against a robust local cubic trend, so normal swings are never flagged;
  flagged samples are replaced by the trend value
- **13 cards**: signals, periods/frequencies, amplitudes + damping (A0, gamma, tau, Q), shape/orientation, spectral check +
  independent damping fits (envelope, peaks, Hilbert), sliding windows (period + theta), phase space, statistics,
  histograms, correlation, velocity/acceleration, energy proxy, residuals. Each card can be hidden and exported as PNG
- **Exports**: *Download Images* (zip of 13 PNGs), *Download PPT* (title + 13 analysis slides + notes, built with
  python-pptx), *Filtered CSV*. All of them use the current filter settings
- Plotly is bundled in `static/plotly.min.js`, so the dashboard works offline

## Run
```bash
python -m pip install -r requirements.txt
python app.py
```
Open the app preview (or http://127.0.0.1:5000 when running locally) → set optional parameters → **Upload video** → play / export.

## Settings
| Field | Role |
|-------|------|
| Hue min / max | Blue hue range in OpenCV HSV (defaults 100–130) |
| Saturation min / value min | Reject gray-blue and dim background colours (defaults 120 / 50) |
| Camera is moving (stabilise) | Correct camera motion before calculating the analysis trajectory |
| Normalise for zoom | Scale the stabilized coordinates by `r₀ / r(t)` |
| Click to calibrate | Fallback: opens the first frame; click the bob to sample its colour |
| Bob confirmation (primary) | After upload, auto-detects candidates and asks you to confirm or select the bob |
| If mirror image, keep | Tie-breaker when two blue round blobs have similar scores |
| Scale (cm/px) | Pixel → real units (leave empty for px) |
| Start / end (s) | Analyse a segment (default: longest clean run) |
| Mass, Length L, g | Labels + theoretical period |
| Trial ID / notes | Stored in CSV/JSON export |

## Exports
- **CSV**: header comments with all parameters + summary, then time series (t, x, y, z, r, φ) and sliding-window θ table
- **JSON**: params, summary, full time series, sliding windows
- **Result video**: 1920×1080 recording of the full single-screen layout from start→end

## Mirror images / dark corners
The tracker only follows components inside the selected blue HSV range. It does not use background subtraction, darkness, or fixed pixel-area limits. Border-touching balls remain eligible with a lower score. Click **Tracking check** to see the selected ball (green), rejected blue blobs and reasons, and the path. **Tracking samples** shows decisions at points across the clip.

## Tests
```bash
python -m unittest discover -s tests -v
```
The synthetic acceptance clip tests camera-motion compensation, the red chair-leg overlap, changing bob size, click calibration, gap interpolation, and the existing analysis/export routes. `pan3.mp4` was not present in the supplied archive, so its real-video acceptance check still needs to be run when that clip is available.
