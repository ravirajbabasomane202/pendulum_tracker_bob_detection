"""Pure numerical analysis helpers shared by the main app and the dashboard."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def nanmean2(a, b, default=float("nan")):
    """Mean of two values that ignores NaN and never raises 'Mean of empty slice'."""
    vals = [float(v) for v in (a, b) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else default


def _fill_nan(a):
    """Linearly fill NaN inside a 1-D array (edges are held constant)."""
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a)
    if good.all() or not good.any():
        return a
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a



# ---------------------------------------------------------------- analysis
def zc_period(a, t):
    """Mean period from upward zero-crossings of the mean-removed, smoothed signal."""
    if len(a) < 15:
        return np.nan
    a = np.convolve(a, np.ones(5) / 5, "same")
    a = a - a.mean()
    i = np.where((a[:-1] < 0) & (a[1:] >= 0))[0]
    if len(i) < 3:
        return np.nan
    tc = t[i] + (t[i + 1] - t[i]) * (-a[i]) / (a[i + 1] - a[i])
    return (tc[-1] - tc[0]) / (len(tc) - 1)


def peak_period(t, s, fps):
    sm = savgol_filter(s, 9, 3)
    prom, dist = 0.15 * np.ptp(sm), int(fps / 8)
    pmax, _ = find_peaks(sm, prominence=prom, distance=dist)
    pmin, _ = find_peaks(-sm, prominence=prom, distance=dist)
    p = np.concatenate([np.diff(t[pmax]), np.diff(t[pmin])])
    return (p.mean(), p.std(), len(p)) if len(p) else (np.nan, np.nan, 0)


def fft_peak(s, fps, fmin=0.1, pad=16):
    s = s - s.mean()
    n = len(s)
    nfft = int(2 ** np.ceil(np.log2(n * pad)))
    sp = np.abs(np.fft.rfft(s * np.hanning(n), nfft))
    fr = np.fft.rfftfreq(nfft, 1 / fps)
    sp[fr < fmin] = 0
    k = int(np.argmax(sp))
    if 0 < k < len(sp) - 1:
        a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * fps / nfft


def pca(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3:
        return float("nan"), float("nan")
    C = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, 1]
    if vals[1] <= 0:
        return float("nan"), float("nan")
    return np.degrees(np.arctan2(v[1], v[0])), np.sqrt(max(vals[0], 0) / vals[1])


def sliding(t, x, y, fps, win=10.0, step=0.25, warm=4.0):
    rows = []
    for te in np.arange(t[0] + warm, t[-1], step):
        m = (t > te - win) & (t <= te)
        Tx, Ty = zc_period(x[m], t[m]), zc_period(y[m], t[m])
        Tm = nanmean2(Tx, Ty, 2.0)
        q = (t > te - 3 * Tm) & (t <= te)
        th, ecc = pca(x[q], y[q])
        rows.append([te, Tx, Ty, th, ecc])
    if not rows:
        return np.zeros((0, 5)), np.array([]), np.array([])
    r = np.array(rows)
    th2 = np.unwrap(np.radians(2 * r[:, 3])) / 2                # unwrap mod-180 axis angle
    rate = np.full(len(r), np.nan)
    for i in range(7, len(r)):
        rate[i] = np.degrees(np.polyfit(r[i - 7:i + 1, 0], th2[i - 7:i + 1], 1)[0])
    th = (r[:, 3] + 90) % 180 - 90
    return r, th, rate


def L(a, d=3):
    return [None if not np.isfinite(v) else round(float(v), d) for v in a]


def safe(v, d=4):
    """JSON-safe scalar: NaN/Inf -> None so jsonify never emits invalid JSON."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, d)


def amplitude(s):
    """Half peak-to-peak amplitude of a 1-D signal (ignores NaN)."""
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return float("nan")
    return 0.5 * (np.nanmax(s) - np.nanmin(s))


def phase_diff_deg(x, y, fps):
    """Rough phase difference (degrees) between x and y via cross-correlation lag at dominant period."""
    x = np.asarray(x, float) - np.nanmean(x)
    y = np.asarray(y, float) - np.nanmean(y)
    n = min(len(x), len(y))
    if n < 30:
        return float("nan")
    x, y = x[:n], y[:n]
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    corr = np.correlate(x, y, mode="full")
    lag = int(np.argmax(corr) - (n - 1))
    # estimate period from x
    Tx, _, _ = peak_period(np.arange(n) / fps, x, fps)
    if not np.isfinite(Tx) or Tx <= 0:
        return float("nan")
    return float((lag / fps) / Tx * 360.0)


def classify_shape(Ax, Ay, ecc, phase_deg, df, drift, decay):
    """
    Classify trajectory shape from summary metrics.
    Returns (label, detail).
    """
    if not (np.isfinite(Ax) and np.isfinite(Ay)):
        return "unknown", "Insufficient amplitude data"
    ratio = min(Ax, Ay) / max(Ax, Ay) if max(Ax, Ay) > 0 else 0.0
    abs_df = abs(df) if np.isfinite(df) else 0.0
    abs_drift = abs(drift) if np.isfinite(drift) else 0.0
    circ = 1.0 - (ecc if np.isfinite(ecc) else 1.0)  # ecc = minor/major ∈ [0,1]; circ high when near 1 minor/major… wait
    # In this code ecc from pca is sqrt(λ_min/λ_max) = minor/major ∈ [0,1]; near 1 → circular
    minor_over_major = ecc if np.isfinite(ecc) else 0.0
    phase = phase_deg if np.isfinite(phase_deg) else 0.0
    phase_mod = abs(((phase + 180) % 360) - 180)  # 0..180
    near_90 = abs(phase_mod - 90) < 25 or abs(phase_mod - 270) < 25

    if decay is not None and np.isfinite(decay) and decay > 0.25:
        base = "decaying amplitude"
    else:
        base = None

    if ratio < 0.12:
        label = "line"
        detail = "One axis amplitude dominates (nearly linear path)"
    elif minor_over_major > 0.82 and near_90 and ratio > 0.75:
        label = "circle"
        detail = "Similar amplitudes, ~90° phase, high circularity"
    elif abs_df > 0.02 and ratio > 0.25:
        label = "Lissajous"
        detail = f"|Δf| ≈ {abs_df:.4f} Hz → frequency difference produces Lissajous figure"
    elif abs_drift > 0.8 and minor_over_major < 0.85:
        label = "precessing ellipse"
        detail = f"Ellipse with θ drift ≈ {drift:.2f} °/s"
    elif minor_over_major < 0.85 and ratio > 0.15:
        label = "ellipse"
        detail = f"Elongated path (minor/major ≈ {minor_over_major:.2f})"
    else:
        label = "ellipse"
        detail = "Near-elliptical trajectory"

    if base:
        label = f"{label} ({base})"
        detail = detail + f"; amplitude decay ≈ {decay:.0%}"
    return label, detail


def amplitude_decay(t, x, y):
    """Fractional amplitude drop from first third to last third of the segment."""
    r = np.hypot(x, y)
    n = len(r)
    if n < 30:
        return float("nan")
    r0, r1 = r[: n // 3], r[2 * n // 3 :]
    if not (np.isfinite(r0).any() and np.isfinite(r1).any()):
        return float("nan")
    a0 = np.nanmean(r0)
    a1 = np.nanmean(r1)
    if not (np.isfinite(a0) and a0 > 0 and np.isfinite(a1)):
        return float("nan")
    return float(max(0.0, (a0 - a1) / a0))


def envelope_and_damping(t, x, y, fps):
    """
    Sliding RMS amplitude A(t) and exponential damping rate γ from
    A(t) ≈ A0 · exp(−γ t).  Returns (A_t, A_vals, gamma, A0).
    """
    r = np.hypot(np.asarray(x, float), np.asarray(y, float))
    t = np.asarray(t, float)
    n = len(r)
    if n < 40:
        return np.array([]), np.array([]), float("nan"), float("nan")
    Tx, _, _ = peak_period(t, np.nan_to_num(x), fps)
    Ty, _, _ = peak_period(t, np.nan_to_num(y), fps)
    Tm = nanmean2(Tx, Ty, 1.0)
    if not np.isfinite(Tm) or Tm <= 0:
        Tm = 1.0
    win = max(int(0.75 * Tm * fps), 8)
    step = max(win // 4, 1)
    ts, As = [], []
    for i in range(0, n - win, step):
        seg = r[i : i + win]
        seg = seg[np.isfinite(seg)]
        if len(seg) < win // 2:
            continue
        rms = float(np.sqrt(np.mean((seg - seg.mean()) ** 2)))
        ts.append(float(t[i + win // 2]))
        As.append(rms * np.sqrt(2))
    ts, As = np.asarray(ts), np.asarray(As)
    if len(As) < 5:
        return ts, As, float("nan"), float("nan")
    m = As > 1e-9
    if m.sum() < 5:
        return ts, As, float("nan"), float("nan")
    try:
        coef = np.polyfit(ts[m], np.log(As[m]), 1)
        gamma = float(-coef[0])
        A0 = float(np.exp(coef[1]))
        if not np.isfinite(gamma) or gamma < -1 or gamma > 50:
            gamma, A0 = float("nan"), float("nan")
    except Exception:
        gamma, A0 = float("nan"), float("nan")
    return ts, As, gamma, A0


