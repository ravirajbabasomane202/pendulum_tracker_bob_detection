"""Analysis dashboard (Flask blueprint).

Everything numerical happens here on the server with numpy/scipy:
  * dataset loading   - a finished tracking job, or an uploaded CSV / JSON export
  * outlier removal   - z-score / IQR / Hampel (measured against a local median trend)
  * filtering         - moving average, Savitzky-Golay, Gaussian, median, Butterworth
                        (zero-phase), LOWESS (local linear, tricube)
  * analysis          - periods, FFT, amplitude + damping, shape, sliding windows,
                        phase space, statistics, histograms, correlation, energy, residuals
  * exports           - PNG images (zip), PowerPoint report, filtered CSV
The browser (templates/dashboard.html) only draws the returned numbers with Plotly.
"""
import csv
import io
import json
import math
import threading
import zipfile
from collections import OrderedDict

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, render_template, request
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from analysis import (_fill_nan, amplitude, amplitude_decay, classify_shape, envelope_and_damping,
                      fft_peak, nanmean2, peak_period, pca, phase_diff_deg, sliding)

dashboard_bp = Blueprint("dashboard", __name__)

FILTER_NAMES = OrderedDict([
    ("none", "None (raw)"),
    ("ma", "Moving average"),
    ("sg", "Savitzky-Golay"),
    ("gauss", "Gaussian"),
    ("median", "Median"),
    ("butter_lp", "Butterworth low-pass"),
    ("butter_hp", "Butterworth high-pass"),
    ("butter_bp", "Butterworth band-pass"),
    ("lowess", "LOWESS (local linear)"),
])
OUTLIER_METHODS = ("none", "zscore", "iqr", "hampel")
DEFAULTS = dict(series="analysis", filter="sg", window=11, poly=3, cutoff=2.0, cutoff_high=3.0,
                outlier="none", thresh=3.0, downsample=True, show_velocity=False,
                slide_win=10.0, slide_step=0.5)
_LOCK = threading.Lock()
_CACHE = OrderedDict()
_CACHE_MAX = 6


# ----------------------------------------------------------------------------- helpers
def _clean(o):
    """Make any structure JSON-safe (NaN/Inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return float(f"{f:.7g}") if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json(obj, status=200):
    return Response(json.dumps(_clean(obj), allow_nan=False), status=status, mimetype="application/json")


def _err(msg, status=400):
    return _json({"error": msg}, status)


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _decimate(n, max_pts):
    if n <= max_pts:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_pts).astype(int))


def _odd(w, lo=3, hi=None):
    w = int(round(w))
    if w % 2 == 0:
        w += 1
    w = max(lo, w)
    if hi is not None and w > hi:
        w = hi if hi % 2 else hi - 1
    return max(w, 1)


# ----------------------------------------------------------------------------- datasets
def _store():
    return current_app.config.setdefault("DASH_DATA", {})


def _center_xy(x, y, flip_y):
    x = _fill_nan(x)
    y = _fill_nan(-y if flip_y else y)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    return x - x.mean(), y - y.mean()


def _series_block(label, x, y, unit):
    return dict(label=label, x=np.asarray(x, float), y=np.asarray(y, float), unit=unit)


def _dataset_from_arrays(did, name, source, t, fps, unit, analysis_xy, others, params, summary):
    series = OrderedDict()
    if analysis_xy is not None:
        series["analysis"] = _series_block(f"Analysis series ({unit}, as in the main app)", *analysis_xy, unit)
    labels = OrderedDict([("stab", "Camera-stabilised (px)"), ("zoom", "Stabilised + zoom-normalised (px)"),
                          ("raw", "Raw image coordinates (px)")])
    for key, label in labels.items():
        xy = others.get(key)
        if xy is not None:
            series[key] = _series_block(label, xy[0], xy[1], "px")
    if not series:
        raise ValueError("The data contains no usable x/y columns.")
    n = len(t)
    if n < 30:
        raise ValueError("At least 30 samples are needed for the dashboard.")
    cov = summary.get("tracking_coverage") if isinstance(summary, dict) else None
    return dict(id=did, name=name, source=source, t=np.asarray(t, float), fs=float(fps), unit=unit,
                series=series, params=params or {}, summary=summary or {},
                coverage=float(cov) if cov not in (None, "") and math.isfinite(_f(cov)) else None)


def _dataset_from_job(jid, job):
    if job.get("state") != "done" or "x" not in job:
        return None
    i0, i1, fps = int(job["i0"]), int(job["i1"]), float(job["fps"])
    unit = job.get("unit", "px")
    ax = _fill_nan(job["x"][i0:i1])
    ay = _fill_nan(job["y"][i0:i1])
    analysis_xy = (ax - ax.mean(), ay - ay.mean()) if np.isfinite(ax).all() and np.isfinite(ay).all() else None
    others = {}
    for key, field in (("raw", "raw_coordinates"), ("stab", "stabilized_coordinates"), ("zoom", "zoom_coordinates")):
        arr = job.get(field)
        if arr is not None:
            others[key] = _center_xy(arr[i0:i1, 0], arr[i0:i1, 1], True)
    t = np.arange(i1 - i0) / fps
    return _dataset_from_arrays(jid, job.get("name") or jid, "video analysis", t, fps, unit, analysis_xy,
                                others, job.get("params"), job.get("summary"))


def _resample_uniform(t, cols):
    """Return (t_uniform, fps, resampled columns)."""
    t = np.asarray(t, float)
    order = np.argsort(t)
    t = t[order]
    cols = {k: (None if v is None else np.asarray(v, float)[order]) for k, v in cols.items()}
    dt = np.diff(t)
    dt = dt[dt > 0]
    if not len(dt):
        raise ValueError("The time column is not increasing.")
    med = float(np.median(dt))
    fps = 1.0 / med
    uniform = np.abs(np.diff(t) - med) <= 0.05 * med
    if uniform.all():
        return t - t[0], fps, cols
    tu = np.arange(0, t[-1] - t[0], med)
    out = {}
    for k, v in cols.items():
        if v is None:
            out[k] = None
            continue
        good = np.isfinite(v)
        out[k] = np.interp(tu, (t - t[0])[good], v[good]) if good.sum() > 1 else None
    return tu, fps, out


def _dataset_from_csv(did, name, text):
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    meta, head = {}, None
    for i, r in enumerate(rows):
        if not r:
            continue
        if r[0].startswith("#"):
            meta[r[0][1:].strip()] = r[1] if len(r) > 1 else ""
        elif r[0].strip() == "time_s":
            head = i
            break
    if head is None:
        raise ValueError("This CSV has no 'time_s' header row - use the CSV exported by the Pendulum Lab.")
    header = [h.strip() for h in rows[head]]
    data = []
    for r in rows[head + 1:]:
        if not r or r[0].startswith("#") or not r[0].strip():
            break
        data.append(r + [""] * (len(header) - len(r)))
    if len(data) < 30:
        raise ValueError("The CSV contains fewer than 30 data rows.")
    table = {h: np.array([_f(r[j]) for r in data]) for j, h in enumerate(header)}

    def col(prefix):
        for h in header:
            if h.startswith(prefix):
                return table[h]
        return None
    xcol = next((h for h in header if h.startswith("x_")), None)
    ycol = next((h for h in header if h.startswith("y_")), None)
    unit = xcol.split("_", 1)[1] if xcol else "px"
    cols = dict(ax=table.get(xcol), ay=table.get(ycol),
                rx=col("raw_x"), ry=col("raw_y"), sx=col("stabilized_x"), sy=col("stabilized_y"),
                zx=col("zoom_normalized_x"), zy=col("zoom_normalized_y"))
    t, fps, cols = _resample_uniform(table["time_s"], cols)
    summary = {k[len("summary_"):]: v for k, v in meta.items() if k.startswith("summary_")}
    params = {k[len("param_"):]: v for k, v in meta.items() if k.startswith("param_")}
    for k in ("g", "length", "mass"):
        if k in params:
            params[k] = _f(params[k]) if _f(params[k]) == _f(params[k]) else None
    return _finish_loaded(did, name, "CSV export", t, fps, unit, cols, params, summary)


def _dataset_from_json(did, name, text):
    payload = json.loads(text)
    ts = payload.get("time_series") or {}
    if "t" not in ts or "x" not in ts:
        raise ValueError("This JSON has no time_series.t / time_series.x - use the JSON exported by the Pendulum Lab.")

    def arr(k):
        v = ts.get(k)
        return None if v is None else np.array([np.nan if e is None else e for e in v], float)
    t = arr("t")
    cols = dict(ax=arr("x"), ay=arr("y"), rx=arr("raw_x"), ry=arr("raw_y"), sx=arr("stabilized_x"),
                sy=arr("stabilized_y"), zx=arr("zoom_normalized_x"), zy=arr("zoom_normalized_y"))
    keep = np.isfinite(cols["ax"]) & np.isfinite(cols["ay"]) & np.isfinite(t)
    t = t[keep]
    cols = {k: (None if v is None else v[keep]) for k, v in cols.items()}
    if len(t) < 30:
        raise ValueError("The JSON contains fewer than 30 analysed samples.")
    t, fps, cols = _resample_uniform(t, cols)
    return _finish_loaded(did, name, "JSON export", t, fps, payload.get("unit", "px"), cols,
                          payload.get("params"), payload.get("summary"))


def _finish_loaded(did, name, source, t, fps, unit, cols, params, summary):
    if cols.get("ax") is None or cols.get("ay") is None:
        raise ValueError("x/y columns are missing or empty.")
    ax, ay = _fill_nan(cols["ax"]), _fill_nan(cols["ay"])
    analysis_xy = (ax - ax.mean(), ay - ay.mean())
    others = {}
    for key, (a, b) in dict(raw=("rx", "ry"), stab=("sx", "sy"), zoom=("zx", "zy")).items():
        if cols.get(a) is not None and cols.get(b) is not None:
            others[key] = _center_xy(cols[a], cols[b], True)
    return _dataset_from_arrays(did, name, source, t, fps, unit, analysis_xy, others, params, summary)


def get_dataset(did):
    job = current_app.config.get("JOBS", {}).get(did)
    if job is not None:
        return _dataset_from_job(did, job)
    return _store().get(did)


def list_datasets():
    out = []
    for jid, job in list(current_app.config.get("JOBS", {}).items()):
        if job.get("state") == "done" and "x" in job:
            n = int(job["i1"] - job["i0"])
            out.append(dict(id=jid, name=job.get("name") or jid, source="video analysis",
                            duration=n / float(job["fps"])))
    for did, ds in _store().items():
        out.append(dict(id=did, name=ds["name"], source=ds["source"], duration=float(ds["t"][-1])))
    return out


# ----------------------------------------------------------------------------- filters
def _lowess(y, win, chunk=4000):
    """Local-linear regression with tricube weights (LOWESS, one pass, no robustness iterations)."""
    n = len(y)
    half = win // 2
    offs = np.arange(-half, half + 1)
    u = offs / (half + 1.0)
    wt = (1 - np.abs(u) ** 3) ** 3
    out = np.empty(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        idx = np.arange(a, b)[:, None] + offs[None, :]
        ok = (idx >= 0) & (idx < n)
        yy = y[np.clip(idx, 0, n - 1)]
        w = wt[None, :] * ok
        sw, sx, sxx = w.sum(1), (w * u).sum(1), (w * u * u).sum(1)
        sy, sxy = (w * yy).sum(1), (w * u * yy).sum(1)
        den = sw * sxx - sx ** 2
        good = den > 1e-12
        out[a:b] = np.where(good, (sxx * sy - sx * sxy) / np.where(good, den, 1.0), sy / np.maximum(sw, 1e-12))
    return out


def _butter(a, fs, kind, cut, cut_high, warn):
    nyq = fs / 2.0
    top = 0.95 * nyq
    lo, hi = cut, cut_high
    if lo >= top:
        warn.append(f"Cut-off {lo:.2f} Hz is above the Nyquist limit ({nyq:.2f} Hz); using {top:.2f} Hz.")
        lo = top
    if kind == "bp":
        if hi >= top:
            warn.append(f"Band-high {hi:.2f} Hz is above the Nyquist limit; using {top:.2f} Hz.")
            hi = top
        if hi <= lo:
            warn.append("Band-high must be larger than the cut-off; the two values were swapped.")
            lo, hi = min(lo, hi), max(lo, hi)
            if hi - lo < 1e-3:
                hi = min(top, lo + 0.1)
    try:
        if kind == "low":
            sos = butter(4, lo, btype="low", fs=fs, output="sos")
        elif kind == "high":
            sos = butter(4, lo, btype="high", fs=fs, output="sos")
        else:
            sos = butter(2, [lo, hi], btype="band", fs=fs, output="sos")
        return sosfiltfilt(sos, a)
    except ValueError as e:
        warn.append(f"Butterworth filter skipped: {e}")
        return a.copy()


def apply_filter(a, p, fs, warn):
    a = np.asarray(a, float)
    n = len(a)
    kind = p["filter"]
    win = _odd(p["window"], 3, n)
    if kind == "none":
        return a.copy()
    if kind == "ma":
        return uniform_filter1d(a, size=win, mode="nearest")
    if kind == "sg":
        poly = int(p["poly"])
        if win <= poly:
            win = _odd(poly + 2, 3, n)
            warn.append(f"Savitzky-Golay window must exceed the polynomial order; using window {win}.")
        if win <= poly:
            warn.append("Signal too short for Savitzky-Golay; filter skipped.")
            return a.copy()
        from scipy.signal import savgol_filter
        return savgol_filter(a, win, poly, mode="interp")
    if kind == "gauss":
        return gaussian_filter1d(a, sigma=max(win / 6.0, 0.5), mode="nearest")
    if kind == "median":
        return median_filter(a, size=win, mode="nearest")
    if kind == "lowess":
        return _lowess(a, _odd(max(win, 5), 5, n))
    if kind == "butter_lp":
        return _butter(a, fs, "low", p["cutoff"], p["cutoff_high"], warn)
    if kind == "butter_hp":
        return _butter(a, fs, "high", p["cutoff"], p["cutoff_high"], warn)
    if kind == "butter_bp":
        return _butter(a, fs, "bp", p["cutoff"], p["cutoff_high"], warn)
    return a.copy()


def _robust_trend(a, win=11, deg=3, iters=4, chunk=8000):
    """Robust local cubic regression (Cleveland-style bisquare re-weighting).

    A cubic follows a normal oscillation, so only genuine glitches have a large residual, and re-weighting
    stops one spike from dragging the trend (and therefore its neighbours) towards itself.
    """
    n = len(a)
    half = win // 2
    offs = np.arange(-half, half + 1)
    V = np.vander(offs / float(half), deg + 1, increasing=True)          # (win, deg+1)
    floor = 1e-6 * max(1.0, float(np.ptp(a)))
    # initial weights: a 5-sample median exposes isolated glitches, so they never enter the first fit
    med5 = median_filter(a, size=5, mode="nearest")
    d0 = a - med5
    s0 = max(6.0 * float(np.median(np.abs(d0))), floor)
    u0 = d0 / s0
    delta = np.where(np.abs(u0) < 1, (1 - u0 ** 2) ** 2, 0.0)
    fit = a.copy()
    for it in range(iters):
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            raw = np.arange(c0, c1)[:, None] + offs[None, :]
            ok = (raw >= 0) & (raw < n)
            ix = np.clip(raw, 0, n - 1)
            w = ok * delta[ix]
            M = np.einsum("nw,wp,wq->npq", w, V, V)
            M += (1e-9 * np.trace(M, axis1=1, axis2=2)[:, None, None] + 1e-12) * np.eye(deg + 1)[None]
            rhs = np.einsum("nw,wp,nw->np", w, V, a[ix])
            try:
                sol = np.linalg.solve(M, rhs[..., None])[..., 0][:, 0]
            except np.linalg.LinAlgError:
                sol = a[c0:c1].copy()
            starved = w.sum(axis=1) < deg + 1           # everything in this window was down-weighted
            fit[c0:c1] = np.where(starved, med5[c0:c1], sol)
        resid = a - fit
        s = max(6.0 * float(np.median(np.abs(resid))), floor)
        u = resid / s
        delta = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return fit


def remove_outliers(a, method, thresh, warn, label):
    """Flag points far from a robust local trend; replace them by the trend value at that sample."""
    a = np.asarray(a, float)
    if method == "none":
        return a.copy(), 0
    n = len(a)
    win = _odd(11, 5, n)
    if n < 12 or win < 7:
        return a.copy(), 0
    floor = 1e-6 * max(1.0, float(np.ptp(a)))
    trend = _robust_trend(a, win)
    resid = a - trend
    if method == "zscore":
        sd = max(float(resid.std()), floor)
        mask = np.abs(resid - resid.mean()) / sd > thresh
    elif method == "iqr":
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = max(q3 - q1, floor)
        mask = (resid < q1 - thresh * iqr) | (resid > q3 + thresh * iqr)
    else:  # hampel: local MAD over 21 samples
        mad = median_filter(np.abs(resid), size=_odd(21, 3, n), mode="nearest")
        mask = np.abs(resid) > thresh * 1.4826 * np.maximum(mad, floor)
    count = int(mask.sum())
    if count == 0:
        return a.copy(), 0
    if mask.mean() > 0.2:
        warn.append(f"Outlier threshold too tight on {label}: {mask.mean():.0%} of samples flagged - skipped.")
        return a.copy(), 0
    out = a.copy()
    out[mask] = trend[mask]
    return out, count


# ----------------------------------------------------------------------------- parameters
def _truthy(v, default):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_params(args, ds):
    def num(name, lo, hi, cast=float):
        try:
            v = cast(float(args.get(name, DEFAULTS[name])))
        except (TypeError, ValueError):
            v = DEFAULTS[name]
        return min(max(v, lo), hi)
    series = args.get("series") or DEFAULTS["series"]
    if series not in ds["series"]:
        series = next(iter(ds["series"]))
    filt = args.get("filter", DEFAULTS["filter"])
    outlier = args.get("outlier", DEFAULTS["outlier"])
    return dict(
        series=series,
        filter=filt if filt in FILTER_NAMES else DEFAULTS["filter"],
        window=_odd(num("window", 3, 301, int)),
        poly=int(num("poly", 1, 7, int)),
        cutoff=num("cutoff", 0.01, 100.0),
        cutoff_high=num("cutoff_high", 0.02, 100.0),
        outlier=outlier if outlier in OUTLIER_METHODS else "none",
        thresh=num("thresh", 0.5, 10.0),
        downsample=_truthy(args.get("downsample"), True),
        show_velocity=_truthy(args.get("show_velocity"), False),
        slide_win=num("slide_win", 2.0, 120.0),
        slide_step=num("slide_step", 0.1, 30.0),
    )


def describe_filter(p):
    k = p["filter"]
    if k == "none":
        return "no filter"
    if k == "sg":
        return f"Savitzky-Golay (window {p['window']}, order {p['poly']})"
    if k in ("ma", "gauss", "median", "lowess"):
        return f"{FILTER_NAMES[k]} (window {p['window']})"
    if k == "butter_bp":
        return f"Butterworth band-pass {p['cutoff']:.2f}-{p['cutoff_high']:.2f} Hz"
    return f"{FILTER_NAMES[k]} at {p['cutoff']:.2f} Hz"


# ----------------------------------------------------------------------------- analysis
def _stats(a):
    a = np.asarray(a, float)
    return dict(mean=a.mean(), std=a.std(), min=a.min(), max=a.max(), median=np.median(a),
                rms=float(np.sqrt(np.mean(a ** 2))))


def _spectrum(s, fs, fmax):
    s = np.asarray(s, float)
    s = s - s.mean()
    n = len(s)
    w = np.hanning(n)
    nfft = min(int(2 ** np.ceil(np.log2(max(n, 2) * 8))), 2 ** 20)
    sp = np.abs(np.fft.rfft(s * w, nfft)) * 2.0 / max(w.sum(), 1e-12)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = f <= fmax
    f, sp = f[m], sp[m]
    k = max(1, int(math.ceil(len(f) / 3000)))
    if k > 1:
        cut = len(f) // k * k
        f, sp = f[:cut:k], sp[:cut].reshape(-1, k).max(axis=1)
    return f, sp


def _exp_fit(t, a):
    """Least-squares fit of ln A = ln A0 - gamma t.  Returns (gamma, A0, R^2)."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    m = np.isfinite(a) & np.isfinite(t) & (a > 1e-9)
    if m.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    y = np.log(a[m])
    c = np.polyfit(t[m], y, 1)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - np.polyval(c, t[m])) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(-c[0]), float(np.exp(c[1])), r2


def _safe(fn, *args, default=float("nan")):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - analysis must never take the whole dashboard down
        return default


def compute(ds, p):
    t, fs = ds["t"], ds["fs"]
    ser = ds["series"][p["series"]]
    unit = ser["unit"]
    n = len(t)
    warn = []
    xs, ys = ser["x"].copy(), ser["y"].copy()

    x1, out_x = remove_outliers(xs, p["outlier"], p["thresh"], warn, "x")
    y1, out_y = remove_outliers(ys, p["outlier"], p["thresh"], warn, "y")
    x = apply_filter(x1, p, fs, warn)
    y = apply_filter(y1, p, fs, warn)
    dt = 1.0 / fs
    vx, vy = np.gradient(x, dt), np.gradient(y, dt)
    ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)

    # ---- periods & frequencies
    pk = lambda s: _safe(peak_period, t, s, fs, default=(np.nan, np.nan, 0))  # noqa: E731
    Tx, Tx_sd, _ = pk(x)
    Ty, Ty_sd, _ = pk(y)
    fx_fft = _safe(fft_peak, x, fs)
    fy_fft = _safe(fft_peak, y, fs)
    fx_pk = 1 / Tx if np.isfinite(Tx) and Tx > 0 else float("nan")
    fy_pk = 1 / Ty if np.isfinite(Ty) and Ty > 0 else float("nan")
    Tm = nanmean2(Tx, Ty)
    periods = dict(
        Tx=Tx, Ty=Ty, Tx_sd=Tx_sd, Ty_sd=Ty_sd, fx_peak=fx_pk, fy_peak=fy_pk, fx_fft=fx_fft, fy_fft=fy_fft,
        df_peak=fx_pk - fy_pk, df_fft=fx_fft - fy_fft,
        ratio_peak=fx_pk / fy_pk if np.isfinite(fy_pk) and fy_pk else float("nan"),
        ratio_fft=fx_fft / fy_fft if np.isfinite(fy_fft) and fy_fft else float("nan"),
        T_meas=Tm, Tx_fft=1 / fx_fft if np.isfinite(fx_fft) and fx_fft > 0 else float("nan"),
        Ty_fft=1 / fy_fft if np.isfinite(fy_fft) and fy_fft > 0 else float("nan"))
    fmax = min(fs / 2.0, 6.0)
    sf, sx = _spectrum(x, fs, fmax)
    _, sy = _spectrum(y, fs, fmax)
    top = max([v for v in (fx_fft, fy_fft) if np.isfinite(v)] or [0.5])
    spectrum = dict(f=sf, x=sx, y=sy, range_hi=min(fmax, max(2.0, 3.0 * top)))

    # ---- amplitudes & damping
    Ax, Ay = amplitude(x), amplitude(y)
    env_t, env_a, gamma, A0 = _safe(envelope_and_damping, t, x, y, fs,
                                    default=(np.array([]), np.array([]), np.nan, np.nan))
    decay = _safe(amplitude_decay, t, x, y)
    tau = 1 / gamma if np.isfinite(gamma) and gamma > 0 else float("nan")
    Q = math.pi / (gamma * Tm) if np.isfinite(gamma) and gamma > 0 and np.isfinite(Tm) and Tm > 0 else float("nan")
    amp = dict(Ax=Ax, Ay=Ay, ratio=Ax / Ay if Ay else float("nan"), A0=A0, gamma=gamma, tau=tau, Q=Q,
               decay=decay, env_t=env_t, env_a=env_a)

    # ---- sliding windows
    T = float(t[-1] - t[0])
    warm = min(4.0, 0.4 * p["slide_win"])
    rows = np.zeros((0, 5))
    th_w = rate = np.array([])
    step = p["slide_step"]
    if T > warm + 1.0:
        step = max(step, (T - warm) / 1500.0)
        rows, th_w, rate = _safe(sliding, t - t[0], x, y, fs, p["slide_win"], step, warm,
                                 default=(np.zeros((0, 5)), np.array([]), np.array([])))
    drift = float("nan")
    if len(rows) >= 3:
        try:
            drift = float(np.polyfit(rows[:, 0], np.degrees(np.unwrap(np.radians(2 * rows[:, 3])) / 2), 1)[0])
        except Exception:  # noqa: BLE001
            pass

    def med(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else float("nan")

    def sd(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(a.std()) if len(a) else float("nan")
    sl = dict(t=rows[:, 0] if len(rows) else [], Tx=rows[:, 1] if len(rows) else [],
              Ty=rows[:, 2] if len(rows) else [], theta=th_w, ecc=rows[:, 4] if len(rows) else [], rate=rate,
              win=p["slide_win"], step=step, need=warm + 1.0,
              Tx_med=med(rows[:, 1]) if len(rows) else float("nan"),
              Ty_med=med(rows[:, 2]) if len(rows) else float("nan"),
              Tx_sd=sd(rows[:, 1]) if len(rows) else float("nan"),
              Ty_sd=sd(rows[:, 2]) if len(rows) else float("nan"),
              ecc_med=med(rows[:, 4]) if len(rows) else float("nan"), drift=drift)

    # ---- shape / orientation
    theta, ecc = _safe(pca, x, y, default=(np.nan, np.nan))
    sub = slice(0, min(n, 20000))
    phase = _safe(phase_diff_deg, x[sub], y[sub], fs)
    label, detail = classify_shape(Ax, Ay, ecc, phase, periods["df_peak"], drift, decay)
    g = _f(ds["params"].get("g")) if ds["params"].get("g") not in (None, "") else float("nan")
    if not np.isfinite(g) or g <= 0:
        g = 9.81
    length = _f(ds["params"].get("length")) if ds["params"].get("length") not in (None, "") else float("nan")
    L_eq = g * (Tm / (2 * math.pi)) ** 2 if np.isfinite(Tm) else float("nan")
    T_th = float("nan")
    if np.isfinite(length) and length > 0:
        T_th = 2 * math.pi * math.sqrt((length / 100.0 if length > 3 else length) / g)
    shape = dict(theta=theta, ecc=ecc, ecc_e=math.sqrt(max(0.0, 1 - ecc ** 2)) if np.isfinite(ecc) else float("nan"),
                 phase=phase, drift=drift, label=label, detail=detail, L_eq=L_eq, g=g, T_theory=T_th,
                 T_err=(Tm - T_th) / T_th if np.isfinite(T_th) and np.isfinite(Tm) else float("nan"))

    # ---- independent damping fits
    r = np.hypot(x, y)
    dist = max(1, int(0.4 * Tm * fs)) if np.isfinite(Tm) else max(1, int(0.4 * fs))
    pidx, _ = find_peaks(r, distance=dist)
    pg, pA, pR = _exp_fit(t[pidx], r[pidx])
    rows_d = []

    def add(method, gam, a0, r2):
        rows_d.append(dict(method=method, gamma=gam, tau=1 / gam if np.isfinite(gam) and gam > 0 else float("nan"),
                           A0=a0, r2=r2))
    sg_, sa_, sr_ = _exp_fit(env_t, env_a)
    add("Sliding-RMS envelope (card 3)", sg_, sa_, sr_)
    add("Peak envelope of r(t)", pg, pA, pR)
    trim = max(1, int(0.05 * n))
    for lab, sig in (("Hilbert envelope x", x), ("Hilbert envelope y", y)):
        try:
            e = np.abs(hilbert(sig - sig.mean()))
            add(lab, *_exp_fit(t[trim:-trim], e[trim:-trim]))
        except Exception:  # noqa: BLE001
            add(lab, np.nan, np.nan, np.nan)

    # ---- energy proxy (units: (length/s)^2 - PE uses omega0^2 so both terms match)
    w0 = 2 * math.pi / Tm if np.isfinite(Tm) and Tm > 0 else 1.0
    ke = 0.5 * (vx ** 2 + vy ** 2)
    pe = 0.5 * w0 ** 2 * (x ** 2 + y ** 2)
    te = ke + pe
    tes = uniform_filter1d(te, size=max(3, int(round((Tm if np.isfinite(Tm) else 1.0) * fs))), mode="nearest")
    gE, E0fit, _ = _exp_fit(t, tes)
    energy = dict(omega0=w0, E_start=float(tes[:max(1, n // 20)].mean()), E_end=float(tes[-max(1, n // 20):].mean()),
                  gamma_E=gE / 2 if np.isfinite(gE) else float("nan"), rate=gE)

    # ---- residuals
    rx, ry = x - xs, y - ys
    resid = dict(x=_stats(rx), y=_stats(ry), outliers_x=out_x, outliers_y=out_y)

    # ---- histograms / correlation / statistics
    hist = {}
    for key, arr in (("x", x), ("y", y)):
        cnt, edges = np.histogram(arr, bins=60)
        hist[key] = dict(c=(edges[:-1] + edges[1:]) / 2, n=cnt, w=float(edges[1] - edges[0]))
    mat = np.vstack([x, y, vx, vy])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(mat)
    corr = np.where(np.isfinite(corr), corr, np.nan)
    stats = {k: _stats(v) for k, v in (("x", x), ("y", y), ("vx", vx), ("vy", vy), ("ax", ax), ("ay", ay))}
    speed = np.hypot(vx, vy)

    idx = _decimate(n, 4000 if p["downsample"] else 20000)
    full = dict(t=t, x=x, y=y, xs=xs, ys=ys, vx=vx, vy=vy, ax=ax, ay=ay, ke=ke, pe=pe, te=te, tes=tes, rx=rx, ry=ry)
    view = {k: v[idx] for k, v in full.items()}
    return dict(
        id=ds["id"], name=ds["name"], source=ds["source"], series=p["series"], series_label=ser["label"], unit=unit,
        fs=fs, n=n, duration=float(t[-1]), coverage=ds["coverage"], params=p, filter_desc=describe_filter(p),
        warnings=warn, outliers=dict(x=out_x, y=out_y), stats=stats, periods=periods, spectrum=spectrum, amp=amp,
        sliding=sl, shape=shape, damping=dict(rows=rows_d, peaks_t=t[pidx], peaks_a=r[pidx]), energy=energy,
        resid=resid, hist=hist, corr=dict(labels=["x", "y", "vx", "vy"], matrix=corr),
        speed=dict(peak=float(speed.max()), median=float(np.median(speed))),
        view=view, shown=len(idx), full=full)


def get_result(ds, p):
    key = (ds["id"], json.dumps(p, sort_keys=True))
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    result = compute(ds, p)
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return result


# ----------------------------------------------------------------------------- routes
def _resolve():
    did = request.view_args["did"]
    ds = get_dataset(did)
    if ds is None:
        return None, None, _err("Unknown or unfinished dataset (the server may have been restarted).", 404)
    return ds, parse_params(request.args, ds), None


@dashboard_bp.route("/dashboard")
@dashboard_bp.route("/dashboard/<did>")
def page(did=None):
    return render_template("dashboard.html", initial_id=did or "")


@dashboard_bp.route("/api/dashboard/datasets")
def datasets():
    return _json({"datasets": list_datasets()})


@dashboard_bp.route("/api/dashboard/<did>/meta")
def meta(did):
    ds = get_dataset(did)
    if ds is None:
        return _err("Unknown or unfinished dataset (the server may have been restarted).", 404)
    return _json(dict(
        id=ds["id"], name=ds["name"], source=ds["source"], fs=ds["fs"], n=len(ds["t"]),
        duration=float(ds["t"][-1]), unit=ds["unit"], coverage=ds["coverage"],
        series=[dict(key=k, label=v["label"], unit=v["unit"]) for k, v in ds["series"].items()],
        defaults=DEFAULTS, nyquist=ds["fs"] / 2.0))


@dashboard_bp.route("/api/dashboard/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return _err("No file received.")
    try:
        text = f.read().decode("utf-8-sig", errors="replace")
        did = "ds" + __import__("uuid").uuid4().hex[:8]
        if f.filename.lower().endswith(".json"):
            ds = _dataset_from_json(did, f.filename, text)
        else:
            ds = _dataset_from_csv(did, f.filename, text)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return _err(str(e) or "Could not read this file.")
    _store()[did] = ds
    return _json(dict(id=did, name=ds["name"]))


@dashboard_bp.route("/api/dashboard/<did>/analysis")
def analysis_api(did):
    ds, p, err = _resolve()
    if err:
        return err
    try:
        result = get_result(ds, p)
    except Exception as e:  # noqa: BLE001
        current_app.logger.exception("dashboard analysis failed")
        return _err(f"Analysis failed: {e}", 500)
    return _json({k: v for k, v in result.items() if k != "full"})


@dashboard_bp.route("/api/dashboard/<did>/filtered.csv")
def filtered_csv(did):
    ds, p, err = _resolve()
    if err:
        return err
    R = get_result(ds, p)
    F, u = R["full"], R["unit"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"# Pendulum dashboard - filtered series ({R['series_label']}); filter: {R['filter_desc']}; "
                f"outliers: {p['outlier']}"])
    w.writerow(["time_s", f"x_original_{u}", f"y_original_{u}", f"x_filtered_{u}", f"y_filtered_{u}",
                f"vx_{u}_per_s", f"vy_{u}_per_s", f"ax_{u}_per_s2", f"ay_{u}_per_s2"])
    for i in range(R["n"]):
        w.writerow([f"{F['t'][i]:.5f}"] + [f"{F[k][i]:.5f}" for k in ("xs", "ys", "x", "y", "vx", "vy", "ax", "ay")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=pendulum_filtered.csv"})


@dashboard_bp.route("/api/dashboard/<did>/images.zip")
def images_zip(did):
    from dashboard_render import render_all
    ds, p, err = _resolve()
    if err:
        return err
    R = get_result(ds, p)
    figs = render_all(R, dark=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (cid, png) in enumerate(figs.items(), 1):
            z.writestr(f"{i:02d}_{cid}.png", png)
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=pendulum_dashboard_images.zip"})


@dashboard_bp.route("/api/dashboard/<did>/report.pptx")
def report_pptx(did):
    from dashboard_render import build_pptx
    ds, p, err = _resolve()
    if err:
        return err
    R = get_result(ds, p)
    return Response(build_pptx(R), mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    headers={"Content-Disposition": "attachment; filename=Pendulum_Trajectory_Analysis_Dashboard.pptx"})
