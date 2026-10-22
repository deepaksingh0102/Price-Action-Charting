"""Price-Action Charting app — the techniques from the "Price Action Trading"
book (@FORRNEERO), drawn on the chart for any US stock:

  * Support / resistance ZONES — price clusters touched >= 2-3 times
    (the book: a level only counts after repeated tests; broken resistance
    flips to support and vice versa)
  * TRENDLINES from swing highs / lows — drawn only when tested >= 2 times;
    the upper line is the book's "selling area", the lower the buying area
  * CHART PATTERN RECOGNITION — ranges/consolidations, triangles, wedges,
    flags/pennants after a fierce pole, and staircase push-pause sequences;
    each detected pattern is outlined and labeled FORMING, ACTIVE, or
    BREAKOUT / BREAKDOWN CONFIRMED with its trigger and invalidation levels
  * CANDLESTICK FORMATIONS marked where they occur — highlighted ONLY when
    they form AT a level (the book's core rule: a candle mid-chart means
    nothing); mid-chart formations are drawn faint
  * MOVING AVERAGES — EMA 9 / 21 (the book's crossover pair) + session VWAP
  * Prior-day levels (the book's AHPMB levels): PDH / PDC / PDL, and the
    30-minute opening range
  * STOCHASTIC %K / %D — the book's momentum-shift gauge (bottom panel)
  * TIME CYCLES — the book names the technique but gives no method; this app
    draws the intraday session phases (opening drive / first hour, midday,
    final hour) as the cycle guide
  * A confluence checklist — the book's "Premium" stack counted at the last
    bar (fresh 9/21 cross, price at a level, pattern at the level, stochastic
    confirmation, volume confirmation). Display only — no verdict, no gates.

Data: Alpaca REST (same Secrets as the other apps). Self-contained file.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except Exception:  # PCR is an add-on; the app must still run without it
    yf = None

ET = ZoneInfo("America/New_York")
ALPACA_BASE = "https://data.alpaca.markets/v2"
LOAD_DAYS = 7            # calendar days of 1-min bars (covers ~5 sessions)
ZONE_TOL = 0.0015        # 0.15% — prices within this band are "the same level"
LEVEL_TOL = 0.0020       # 0.20% — a bar is "at" a level within this band

# Sidebar "Chart timeframe" labels -> trailing minutes shown on the chart
# (None = full session since market open) — same control as the
# stock-analyzer's. It sets the visible window, not the bar size; zones,
# trendlines and formations are computed on the full session.
CHART_TF_TRAILING = {"5 minutes": 5, "15 minutes": 15, "1 hour": 60,
                     "2 hours": 120, "3 hours": 180, "4 hours": 240,
                     "Since market open": None}


# ---------------------------------------------------------------- Alpaca
def _secrets() -> tuple[str, str, str]:
    def _get(name, default=""):
        try:
            v = st.secrets.get(name, default)
        except Exception:
            v = default
        import os
        return str(os.environ.get(name, v) or default)
    return (_get("APCA_API_KEY_ID"), _get("APCA_API_SECRET_KEY"),
            _get("ALPACA_FEED", "sip") or "sip")


def alpaca_get(path: str, params: dict, tries: int = 4) -> dict:
    key, secret, _ = _secrets()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    delay = 1.0
    for _ in range(tries):
        r = requests.get(f"{ALPACA_BASE}{path}", params=params,
                         headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, 16.0)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _bucket(live: bool, refresh_secs: int,
            bucket_secs: int | None = None) -> int:
    """Time bucket (live tracking) + manual-refresh nonce — same scheme as
    the stock-analyzer's. The bucket is part of the fetch's cache key, so a
    new bucket (interval elapsed, or 🔄 Refresh now bumped the nonce) forces
    a fresh Alpaca fetch; within a bucket the cached bars are reused.
    bucket_secs lets slow-moving data (PCR) refresh on its own slower
    bucket instead of re-fetching every cycle."""
    base = int(time.time() // (bucket_secs or refresh_secs)) if live else 0
    try:
        nonce = st.session_state.get("refresh_nonce", 0)
    except Exception:
        nonce = 0
    return base + nonce


@st.cache_data(show_spinner=False)
def fetch_bars_1m(symbol: str, bucket: int = 0,
                  days: int = LOAD_DAYS) -> pd.DataFrame:
    """Last `days` calendar days of 1-minute bars, regular hours only
    (09:30-16:00 ET), paginated (Alpaca's limit is per page)."""
    _, _, feed = _secrets()
    start = (datetime.now(ET) - timedelta(days=days)).strftime(
        "%Y-%m-%dT00:00:00Z")
    rows, token = [], None
    while True:
        params = {"timeframe": "1Min", "start": start, "limit": 10000,
                  "feed": feed, "sort": "asc", "adjustment": "split"}
        if token:
            params["page_token"] = token
        data = alpaca_get(f"/stocks/{symbol}/bars", params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "t": "ts", "o": "Open", "h": "High", "l": "Low", "c": "Close",
        "v": "Volume"})
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df["ts"], utc=True)).tz_convert(ET)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df.sort_index()
    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= 570) & (mins < 960)]          # 09:30 <= t < 16:00 ET
    return df[~df.index.duplicated(keep="last")]


# ---------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series,
             k: int = 14, d: int = 3):
    """Stochastic %K / %D — identical math to the screener and backtester."""
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    pk = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    return pk, pk.rolling(d).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """EMA 9/21, session-anchored VWAP, stochastic — computed per session on
    the frame as given (for 1-min frames the VWAP is the exact session VWAP;
    on resampled frames it is the standard resampled approximation)."""
    df = df.copy()
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    dates = df.index.date
    df["_d"] = dates
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = (tp * df["Volume"]).groupby(df["_d"]).cumsum()
    vv = df["Volume"].groupby(df["_d"]).cumsum()
    df["VWAP"] = pv / vv.replace(0, np.nan)
    df = df.drop(columns=["_d"])
    df["StochK"], df["StochD"] = stoch_kd(df["High"], df["Low"], df["Close"])
    return df


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes <= 1:
        return df
    out = df.resample(f"{minutes}min", origin="start_day").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}).dropna(subset=["Close"])
    return out


# ---------------------------------------------------------------- levels
def prior_day_levels(df_1m: pd.DataFrame) -> dict | None:
    """PDH / PDC / PDL from the last completed session in the 1-min frame
    (the session before the most recent one present)."""
    if df_1m.empty:
        return None
    dates = sorted(set(df_1m.index.date))
    if len(dates) < 2:
        return None
    prior = df_1m[df_1m.index.date == dates[-2]]
    if prior.empty:
        return None
    return {"PDH": float(prior["High"].max()),
            "PDC": float(prior["Close"].iloc[-1]),
            "PDL": float(prior["Low"].min())}


def opening_range(df_day: pd.DataFrame, minutes: int = 30) -> dict | None:
    if df_day.empty:
        return None
    start = df_day.index[0]
    orb = df_day[df_day.index < start + pd.Timedelta(minutes=minutes)]
    if orb.empty:
        return None
    return {"high": float(orb["High"].max()), "low": float(orb["Low"].min())}


# ---------------------------------------------------------------- swings, zones, trendlines
def swing_points(df: pd.DataFrame, w: int = 5) -> tuple[list[int], list[int]]:
    """Positional indices of swing highs / lows (fractal: extreme within
    +/- w bars)."""
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    highs, lows = [], []
    for i in range(w, len(df) - w):
        if H[i] >= H[i - w:i + w + 1].max():
            highs.append(i)
        if L[i] <= L[i - w:i + w + 1].min():
            lows.append(i)
    return highs, lows


def sr_zones(df: pd.DataFrame, tol: float = ZONE_TOL) -> list[dict]:
    """Cluster swing-point prices into zones. Each zone: center, low, high,
    touches (swing count), has_highs / has_lows, hi_idx / lo_idx (bar
    positions of the tests — their order shows a flip: resistance tests
    before support tests = the book's resistance-turned-support). Only
    clusters with >= 2 touches qualify — the book's rule."""
    highs, lows = swing_points(df)
    pts = [(float(df["High"].iloc[i]), "H", i) for i in highs]
    pts += [(float(df["Low"].iloc[i]), "L", i) for i in lows]
    if not pts:
        return []
    pts.sort()
    zones, cur = [], [pts[0]]
    for p, kind, i in pts[1:]:
        if abs(p - cur[-1][0]) / max(cur[-1][0], 1e-9) <= tol:
            cur.append((p, kind, i))
        else:
            zones.append(cur)
            cur = [(p, kind, i)]
    zones.append(cur)
    out = []
    for cl in zones:
        if len(cl) < 2:
            continue
        prices = [p for p, _, _ in cl]
        out.append({
            "center": float(np.median(prices)),
            "low": float(min(prices)), "high": float(max(prices)),
            "touches": len(cl),
            "has_highs": any(k == "H" for _, k, _ in cl),
            "has_lows": any(k == "L" for _, k, _ in cl),
            "hi_idx": [i for _, k, i in cl if k == "H"],
            "lo_idx": [i for _, k, i in cl if k == "L"],
        })
    return out


def _line_touches(df: pd.DataFrame, x0: int, y0: float, x1: int, y1: float,
                  tol: float = ZONE_TOL) -> int:
    """Bars whose High (resistance line) or Low (support line) comes within
    tol of the line — including the two anchors."""
    n = len(df)
    slope = (y1 - y0) / max(x1 - x0, 1)
    xs = np.arange(x0, n)
    line = y0 + slope * (xs - x0)
    near_hi = np.abs(df["High"].to_numpy()[x0:] - line) / line <= tol
    near_lo = np.abs(df["Low"].to_numpy()[x0:] - line) / line <= tol
    return int(max(near_hi.sum(), near_lo.sum()))


def trendlines(df: pd.DataFrame) -> list[dict]:
    """The book's trendlines: through the two most recent swing highs
    (resistance / selling area) and the two most recent swing lows
    (support / buying area), kept only when the line has been tested
    >= 2 times. Extended from the first anchor to the last bar."""
    highs, lows = swing_points(df)
    n = len(df)
    out = []
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        ya, yb = float(df["High"].iloc[a]), float(df["High"].iloc[b])
        t = _line_touches(df, a, ya, b, yb)
        if t >= 2:
            slope = (yb - ya) / max(b - a, 1)
            out.append({"kind": "resistance", "x0": a, "y0": ya,
                        "x1": n - 1, "y1": ya + slope * (n - 1 - a),
                        "touches": t})
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        ya, yb = float(df["Low"].iloc[a]), float(df["Low"].iloc[b])
        t = _line_touches(df, a, ya, b, yb)
        if t >= 2:
            slope = (yb - ya) / max(b - a, 1)
            out.append({"kind": "support", "x0": a, "y0": ya,
                        "x1": n - 1, "y1": ya + slope * (n - 1 - a),
                        "touches": t})
    return out


# ---------------------------------------------------------------- chart patterns
def _make_chart_pattern(name: str, bias: str, status: str, start: int,
                        end: int, segments: list[dict], label_x: int,
                        label_y: float, trigger: float | None,
                        invalidation: float | None, detail: str,
                        score: float, family: str,
                        box: dict | None = None) -> dict:
    return {"name": name, "bias": bias, "status": status,
            "start": int(start), "end": int(end), "segments": segments,
            "label_x": int(label_x), "label_y": float(label_y),
            "trigger": trigger, "invalidation": invalidation,
            "detail": detail, "score": float(score), "family": family,
            "box": box}


def _fit_line(xs, ys) -> tuple[float, float] | None:
    if len(xs) < 2 or len(set(xs)) < 2:
        return None
    slope, intercept = np.polyfit(np.asarray(xs, dtype=float),
                                  np.asarray(ys, dtype=float), 1)
    return float(slope), float(intercept)


def _line_at(fit: tuple[float, float], x: int) -> float:
    return fit[0] * x + fit[1]


def _detect_range_pattern(df: pd.DataFrame) -> dict | None:
    """A tradeable sideways range: repeated tests of two horizontal
    boundaries, little net drift, plus a confirmed break when the close has
    left the range within the last 15 bars."""
    n = len(df)
    if n < 18:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    all_highs, all_lows = swing_points(df, w=3)
    best = None
    for end in range(n - 1, max(11, n - 31), -1):
        for start in range(max(0, end - 89), end - 11):
            length = end - start + 1
            hi = float(H[start:end + 1].max())
            lo = float(L[start:end + 1].min())
            mid = (hi + lo) / 2.0
            if mid <= 0:
                continue
            width = (hi - lo) / mid * 100.0
            if not 0.25 <= width <= 4.0:
                continue
            slope = np.polyfit(np.arange(length), C[start:end + 1], 1)[0]
            drift = abs(float(slope) * (length - 1) / mid * 100.0)
            if drift > max(0.45, width * 0.70):
                continue
            tol = max(mid * 0.0012, (hi - lo) * 0.18)
            # Count swing tests, not ordinary bars: in a noisy range almost
            # every candle can brush a boundary, which proves nothing.
            upper_tests = sum(1 for i in all_highs
                              if start <= i <= end and H[i] >= hi - tol)
            lower_tests = sum(1 for i in all_lows
                              if start <= i <= end and L[i] <= lo + tol)
            if upper_tests < 2 or lower_tests < 2:
                continue
            status, bias = "FORMING", "neutral"
            if end < n - 1:
                tail = C[end + 1:]
                above = np.where(tail > hi * 1.0003)[0]
                below = np.where(tail < lo * 0.9997)[0]
                first_above = int(above[0]) if len(above) else 999
                first_below = int(below[0]) if len(below) else 999
                if first_above < first_below:
                    event_i = end + 1 + first_above
                    if first_above > 5 or n - 1 - event_i > 8:
                        continue
                    if C[-1] < lo:
                        continue
                    status = ("BREAKOUT CONFIRMED" if C[-1] >= hi
                              else "BREAKOUT / BACK IN RANGE")
                    bias = "bull"
                elif first_below < first_above:
                    event_i = end + 1 + first_below
                    if first_below > 5 or n - 1 - event_i > 8:
                        continue
                    if C[-1] > hi:
                        continue
                    status = ("BREAKDOWN CONFIRMED" if C[-1] <= lo
                              else "BREAKDOWN / BACK IN RANGE")
                    bias = "bear"
                elif end < n - 4:
                    continue
            trigger = hi if bias != "bear" else lo
            invalidation = lo if bias != "bear" else hi
            score = (upper_tests + lower_tests) * 10 + min(length, 60) \
                + (30 if status != "FORMING" else 12) - width * 3 \
                - abs(n - 1 - end)
            cand = _make_chart_pattern(
                "Range / Consolidation", bias, status, start, n - 1,
                [{"x0": start, "y0": hi, "x1": n - 1, "y1": hi},
                 {"x0": start, "y0": lo, "x1": n - 1, "y1": lo}],
                start, hi, trigger, invalidation,
                f"{length} bars · {upper_tests} upper tests / "
                f"{lower_tests} lower tests · width {width:.2f}%",
                score, "consolidation",
                box={"x0": start, "x1": n - 1, "y0": lo, "y1": hi})
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def _detect_triangle_pattern(df: pd.DataFrame) -> dict | None:
    """Triangles and wedges from fitted swing-high / swing-low boundary
    lines. A pattern qualifies only when the two lines converge and price
    is mostly contained between them."""
    n = len(df)
    if n < 24:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    all_highs, all_lows = swing_points(df, w=3)
    best = None
    for end in range(n - 1, max(14, n - 31), -1):
        for start in range(max(0, end - 89), end - 14):
            highs = [i for i in all_highs if start <= i <= end]
            lows = [i for i in all_lows if start <= i <= end]
            if len(highs) < 2 or len(lows) < 2 \
                    or len(highs) + len(lows) < 5:
                continue
            first = min(highs[0], lows[0])
            last_swing = max(highs[-1], lows[-1])
            if last_swing - first < 12:
                continue
            if last_swing < end - 15 or highs[-1] < end - 25 \
                    or lows[-1] < end - 25:
                continue
            upper = _fit_line(highs, [H[i] for i in highs])
            lower = _fit_line(lows, [L[i] for i in lows])
            if upper is None or lower is None:
                continue
            u0, u1 = _line_at(upper, first), _line_at(upper, end)
            l0, l1 = _line_at(lower, first), _line_at(lower, end)
            start_width, end_width = u0 - l0, u1 - l1
            mid = (u0 + l0) / 2.0
            if mid <= 0 or start_width <= 0 or end_width <= 0:
                continue
            start_width_pct = start_width / mid * 100.0
            if not 0.35 <= start_width_pct <= 6.0:
                continue
            compression = end_width / start_width
            if not 0.22 <= compression <= 0.82:
                continue
            upper_resid = np.mean([abs(H[i] - _line_at(upper, i))
                                   for i in highs]) / start_width
            lower_resid = np.mean([abs(L[i] - _line_at(lower, i))
                                   for i in lows]) / start_width
            if upper_resid > 0.22 or lower_resid > 0.22:
                continue
            upper_change = (u1 - u0) / mid * 100.0
            lower_change = (l1 - l0) / mid * 100.0
            flat = max(0.18, start_width_pct * 0.18)
            if upper_change <= -0.25 and lower_change >= 0.25:
                name = "Symmetrical Triangle"
            elif abs(upper_change) <= flat and lower_change >= 0.30:
                name = "Ascending Triangle"
            elif abs(lower_change) <= flat and upper_change <= -0.30:
                name = "Descending Triangle"
            elif upper_change > 0.20 and lower_change > upper_change + 0.20:
                name = "Rising Wedge"
            elif upper_change < -0.20 and upper_change < lower_change - 0.20:
                name = "Falling Wedge"
            else:
                continue
            xs = np.arange(first, end + 1)
            tol = max(mid * 0.0015, start_width * 0.12)
            inside = ((C[first:end + 1] <= _line_at(upper, xs) + tol)
                      & (C[first:end + 1] >= _line_at(lower, xs) - tol))
            containment = float(inside.mean())
            if containment < 0.90:
                continue
            u_last = _line_at(upper, n - 1)
            l_last = _line_at(lower, n - 1)
            buffer = max(mid * 0.0003, max(end_width, 0.0) * 0.03)

            def side_at(j: int) -> int:
                if C[j] > _line_at(upper, j) + buffer:
                    return 1
                if C[j] < _line_at(lower, j) - buffer:
                    return -1
                return 0

            last_side = side_at(n - 1)
            if last_side:
                # "Confirmed" means the crossing itself is fresh. A pattern
                # whose boundary was crossed long ago is history, not a
                # current recognition result.
                cross_i = None
                for j in range(max(first + 1, n - 7), n):
                    if side_at(j) == last_side and side_at(j - 1) != last_side:
                        cross_i = j
                        break
                if cross_i is None:
                    continue
                if last_side == 1:
                    status, bias = "BREAKOUT CONFIRMED", "bull"
                    trigger, invalidation = u_last, l_last
                else:
                    status, bias = "BREAKDOWN CONFIRMED", "bear"
                    trigger, invalidation = l_last, u_last
            else:
                if end < n - 4:
                    continue
                status, bias = "FORMING", "neutral"
                if name == "Ascending Triangle":
                    bias = "bull"
                elif name == "Descending Triangle":
                    bias = "bear"
                trigger, invalidation = u_last, l_last
            score = len(highs) + len(lows) + containment * 20 \
                + (1.0 - compression) * 25 + min(end - first, 60) * 0.25 \
                + (25 if status != "FORMING" else 8)
            cand = _make_chart_pattern(
                name, bias, status, first, n - 1,
                [{"x0": first, "y0": u0, "x1": n - 1, "y1": u_last},
                 {"x0": first, "y0": l0, "x1": n - 1, "y1": l_last}],
                max(first, n - 12), (u_last + l_last) / 2.0,
                trigger, invalidation,
                f"{len(highs)} swing highs / {len(lows)} swing lows · "
                f"range compressed to {compression:.0%} · "
                f"{containment:.0%} contained",
                score, "consolidation")
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def _detect_flag_pattern(df: pd.DataFrame) -> dict | None:
    """Flags and pennants, using the book's sequence: a fierce pole on
    strong movement, then a small contained pause. Parallel counter-trend
    boundaries = flag; converging boundaries = pennant."""
    n = len(df)
    if n < 22:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    best = None
    for direction in (1, -1):
        for pole_end in range(max(10, n - 90), n - 5):
            for pole_len in range(4, 21):
                pole_start = pole_end - pole_len
                if pole_start < 0:
                    continue
                if direction == 1:
                    base = float(L[pole_start])
                    extreme = float(H[pole_end])
                else:
                    base = float(H[pole_start])
                    extreme = float(L[pole_end])
                move = direction * (extreme / base - 1.0) * 100.0
                if not 0.75 <= move <= 8.0:
                    continue
                pole_slope = np.polyfit(
                    np.arange(pole_len + 1),
                    C[pole_start:pole_end + 1], 1)[0]
                if direction * pole_slope <= 0:
                    continue
                pole_vol = float(V[pole_start:pole_end + 1].mean())
                prior_vol = float(V[max(0, pole_start - 20):pole_start].mean()) \
                    if pole_start >= 5 else 0.0
                vol_ratio = pole_vol / prior_vol if prior_vol > 0 else 1.0
                # The book requires the pole to be a fierce move on heavy
                # volume. Without that evidence, an ordinary oscillation can
                # masquerade as a flagpole.
                if prior_vol <= 0 or vol_ratio < 1.30:
                    continue
                move_abs = abs(extreme - base)
                for cons_len in range(6, 36):
                    end = pole_end + cons_len
                    if end >= n:
                        break
                    seg = slice(pole_end, end + 1)
                    xs = np.arange(pole_end, end + 1)
                    upper = _fit_line(xs, H[seg])
                    lower = _fit_line(xs, L[seg])
                    if upper is None or lower is None:
                        continue
                    width_abs = float(H[seg].max() - L[seg].min())
                    if width_abs > move_abs * 0.75:
                        continue
                    if direction == 1:
                        if H[seg].max() > extreme * 1.0015:
                            continue
                        if L[seg].min() < extreme - move_abs * 0.58:
                            continue
                    else:
                        if L[seg].min() < extreme * 0.9985:
                            continue
                        if H[seg].max() > extreme + move_abs * 0.58:
                            continue
                    u_change = (_line_at(upper, end)
                                - _line_at(upper, pole_end)) / extreme * 100
                    l_change = (_line_at(lower, end)
                                - _line_at(lower, pole_end)) / extreme * 100
                    pennant = (direction == 1 and u_change < -0.08
                               and l_change > 0.08) or \
                        (direction == -1 and u_change < -0.08
                         and l_change > 0.08)
                    flag = False
                    if direction == 1:
                        flag = u_change < 0.08 and l_change < 0.08 \
                            and u_change > -1.0 and l_change > -1.0 \
                            and abs(u_change - l_change) < 0.45
                    else:
                        flag = u_change > -0.08 and l_change > -0.08 \
                            and u_change < 1.0 and l_change < 1.0 \
                            and abs(u_change - l_change) < 0.45
                    if not pennant and not flag:
                        continue
                    tol = max(extreme * 0.0015, width_abs * 0.12)
                    inside = ((C[seg] <= _line_at(upper, xs) + tol)
                              & (C[seg] >= _line_at(lower, xs) - tol))
                    if float(inside.mean()) < 0.82:
                        continue
                    u_last = _line_at(upper, n - 1)
                    l_last = _line_at(lower, n - 1)
                    if direction == 1:
                        if C[-1] > u_last:
                            status, bias = "BREAKOUT CONFIRMED", "bull"
                            trigger, invalidation = u_last, l_last
                        elif C[-1] < l_last:
                            continue
                        else:
                            if end < n - 4:
                                continue
                            status, bias = "FORMING", "bull"
                            trigger, invalidation = u_last, l_last
                    else:
                        if C[-1] < l_last:
                            status, bias = "BREAKDOWN CONFIRMED", "bear"
                            trigger, invalidation = l_last, u_last
                        elif C[-1] > u_last:
                            continue
                        else:
                            if end < n - 4:
                                continue
                            status, bias = "FORMING", "bear"
                            trigger, invalidation = l_last, u_last
                    kind = "Pennant" if pennant else "Flag"
                    name = ("Bull " if direction == 1 else "Bear ") + kind
                    pole_y0 = base
                    pole_y1 = extreme
                    score = move * 8 + min(vol_ratio, 2.0) * 8 \
                        + float(inside.mean()) * 15 + min(cons_len, 30) \
                        + (25 if status != "FORMING" else 8)
                    cand = _make_chart_pattern(
                        name, bias, status, pole_start, n - 1,
                        [{"x0": pole_start, "y0": pole_y0,
                          "x1": pole_end, "y1": pole_y1},
                         {"x0": pole_end, "y0": _line_at(upper, pole_end),
                          "x1": n - 1, "y1": u_last},
                         {"x0": pole_end, "y0": _line_at(lower, pole_end),
                          "x1": n - 1, "y1": l_last}],
                        max(pole_end, n - 12),
                        u_last if direction == 1 else l_last,
                        trigger, invalidation,
                        f"pole {move:.2f}% in {pole_len + 1} bars · "
                        f"pole volume {vol_ratio:.2f}x prior · "
                        f"pause {cons_len + 1} bars",
                        score, "consolidation")
                    if best is None or cand["score"] > best["score"]:
                        best = cand
    return best


def _detect_staircase_pattern(df: pd.DataFrame) -> dict | None:
    """A staircase is a sequence of pushes separated by pauses: successive
    swing highs and swing lows both step in the trend direction."""
    n = len(df)
    if n < 35:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    highs, lows = swing_points(df, w=3)
    raw = [(i, "H", float(H[i])) for i in highs]
    raw += [(i, "L", float(L[i])) for i in lows]
    raw.sort()
    points: list[tuple[int, str, float]] = []
    for point in raw:
        if points and points[-1][1] == point[1]:
            prev = points[-1]
            if (point[1] == "H" and point[2] > prev[2]) or \
                    (point[1] == "L" and point[2] < prev[2]):
                points[-1] = point
        else:
            points.append(point)
    if len(points) < 5:
        return None
    best = None
    for end_i in range(len(points) - 1, 3, -1):
        if n - 1 - points[end_i][0] > 30:
            continue
        for start_i in range(max(0, end_i - 8), end_i - 3):
            seq = points[start_i:end_i + 1]
            if len(seq) < 5:
                continue
            hs = [p for p in seq if p[1] == "H"]
            ls = [p for p in seq if p[1] == "L"]
            if len(hs) < 3 or len(ls) < 3:
                continue
            bull_steps = all(hs[j][2] > hs[j - 1][2] * 1.0012
                             for j in range(1, len(hs))) and \
                all(ls[j][2] > ls[j - 1][2] * 1.0012
                    for j in range(1, len(ls)))
            bear_steps = all(hs[j][2] < hs[j - 1][2] * 0.9988
                             for j in range(1, len(hs))) and \
                all(ls[j][2] < ls[j - 1][2] * 0.9988
                    for j in range(1, len(ls)))
            if not bull_steps and not bear_steps:
                continue
            direction = 1 if bull_steps else -1
            last_h, last_l = hs[-1], ls[-1]
            if direction == 1:
                if C[-1] < last_l[2]:
                    continue
                trigger, invalidation = last_h[2], last_l[2]
                bias, name = "bull", "Bullish Staircase"
            else:
                if C[-1] > last_h[2]:
                    continue
                trigger, invalidation = last_l[2], last_h[2]
                bias, name = "bear", "Bearish Staircase"
            segments = [{"x0": seq[j][0], "y0": seq[j][2],
                         "x1": seq[j + 1][0], "y1": seq[j + 1][2]}
                        for j in range(len(seq) - 1)]
            segments.append({"x0": seq[-1][0], "y0": trigger,
                             "x1": n - 1, "y1": trigger})
            score = len(seq) * 8 + (len(hs) + len(ls)) * 2 \
                - (n - 1 - seq[-1][0]) * 0.25
            cand = _make_chart_pattern(
                name, bias, "ACTIVE", seq[0][0], n - 1, segments,
                seq[-1][0], trigger, trigger, invalidation,
                f"{len(hs)} stepping highs / {len(ls)} stepping lows · "
                "push-pause sequence still intact",
                score, "staircase")
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def detect_chart_patterns(df: pd.DataFrame) -> list[dict]:
    """Current chart-pattern recognition for the displayed session. Returns
    at most three non-duplicative patterns, confirmed breaks first."""
    found = []
    for detector in (_detect_range_pattern, _detect_triangle_pattern,
                     _detect_flag_pattern, _detect_staircase_pattern):
        try:
            pattern = detector(df)
        except Exception:
            pattern = None
        if pattern is not None:
            found.append(pattern)
    if not found:
        return []

    def overlap(a: dict, b: dict) -> float:
        lo = max(a["start"], b["start"])
        hi = min(a["end"], b["end"])
        span = max(a["end"] - a["start"], b["end"] - b["start"], 1)
        return max(0, hi - lo) / span

    def specificity(pattern: dict) -> int:
        # When geometries overlap, a converging or pole-based pattern is a
        # more specific description than the generic range that contains it.
        if pattern["name"].startswith("Range"):
            return 1
        if "Staircase" in pattern["name"]:
            return 2
        return 3

    found.sort(key=lambda p: (p["status"] != "FORMING"
                              and p["status"] != "ACTIVE",
                              specificity(p), p["score"]), reverse=True)
    kept: list[dict] = []
    for pattern in found:
        duplicate = False
        for other in kept:
            if pattern["family"] == other["family"] == "consolidation" \
                    and overlap(pattern, other) >= 0.60:
                duplicate = True
                break
        if not duplicate:
            kept.append(pattern)
        if len(kept) >= 3:
            break
    return kept

# ---------------------------------------------------------------- candle formations
def candle_patterns(df: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Positional index -> (pattern name, bias 'bull' | 'bear' | 'neutral').
    Pure candle geometry; whether a formation matters (at a level) is judged
    by the caller — the book's rule lives in the display, not the detector."""
    O, H, L, C = (df[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    out: dict[int, tuple[str, str]] = {}
    for i in range(len(df)):
        rng = H[i] - L[i]
        if rng <= 0:
            continue
        body = C[i] - O[i]
        abody = abs(body)
        upper = H[i] - max(O[i], C[i])
        lower = min(O[i], C[i]) - L[i]
        if i >= 1:
            pbody = C[i - 1] - O[i - 1]
            if (pbody < 0 and body > 0 and O[i] <= C[i - 1]
                    and C[i] >= O[i - 1] and abody > abs(pbody)):
                out[i] = ("Bullish engulfing", "bull")
                continue
            if (pbody > 0 and body < 0 and O[i] >= C[i - 1]
                    and C[i] <= O[i - 1] and abody > abs(pbody)):
                out[i] = ("Bearish engulfing", "bear")
                continue
        if abody <= 0.10 * rng and upper <= 0.35 * rng and lower <= 0.35 * rng:
            out[i] = ("Doji", "neutral")
        elif lower >= 2 * max(abody, 1e-9) and upper <= 0.30 * rng:
            out[i] = ("Hammer", "bull")
        elif upper >= 2 * max(abody, 1e-9) and lower <= 0.30 * rng:
            out[i] = ("Shooting star", "bear")
        elif abody >= 0.80 * rng:
            out[i] = ("Momentum candle", "bull" if body > 0 else "bear")
    return out


def at_level(lo: float, hi: float, levels: list[float],
             tol: float = LEVEL_TOL) -> bool:
    """Does the bar's [lo, hi] range come within tol of any level?"""
    for v in levels:
        if v <= 0:
            continue
        if lo - tol * v <= v <= hi + tol * v:
            return True
    return False


# ---------------------------------------------------------------- confluence
def confluence(df: pd.DataFrame, zones: list[dict],
               levels: list[float]) -> list[tuple[str, bool]]:
    """The book's Premium stack, evaluated at the last bar. Display only."""
    checks: list[tuple[str, bool]] = []
    if len(df) < 25:
        return checks
    e9, e21 = df["EMA9"], df["EMA21"]
    cross_up = cross_dn = False
    for i in range(len(df) - 10, len(df)):
        if e9.iloc[i - 1] <= e21.iloc[i - 1] and e9.iloc[i] > e21.iloc[i]:
            cross_up, cross_dn = True, False
        if e9.iloc[i - 1] >= e21.iloc[i - 1] and e9.iloc[i] < e21.iloc[i]:
            cross_dn, cross_up = True, False
    checks.append(("Fresh 9/21 EMA cross in the last 10 bars "
                   f"({'up' if cross_up else 'down' if cross_dn else 'none'})",
                   cross_up or cross_dn))
    last = df.iloc[-1]
    price = float(last["Close"])
    all_levels = levels + [z["center"] for z in zones]
    checks.append(("Price at a key level (zone / PDH / PDC / PDL / OR edge)",
                   at_level(float(last["Low"]), float(last["High"]),
                            all_levels)))
    pats = candle_patterns(df)
    pat_ok, pat_txt = False, "none in the last 3 bars"
    for i in range(len(df) - 3, len(df)):
        if i in pats:
            b = df.iloc[i]
            if at_level(float(b["Low"]), float(b["High"]), all_levels):
                pat_ok, pat_txt = True, f"{pats[i][0]} at a level"
            else:
                pat_txt = f"{pats[i][0]} (not at a level)"
    checks.append((f"Candlestick formation at a level — {pat_txt}", pat_ok))
    k, d = df["StochK"], df["StochD"]
    scross = False
    for i in range(len(df) - 5, len(df)):
        if (k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i]) or \
           (k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i]):
            scross = True
    checks.append(("Stochastic %K/%D cross in the last 5 bars", scross))
    vol = df["Volume"]
    recent = float(vol.iloc[-5:].mean())
    base = float(vol.iloc[-25:-5].mean()) if len(vol) >= 25 else 0.0
    checks.append((f"Volume confirming (last 5 bars "
                   f"{recent / base:.2f}x the prior 20)" if base > 0
                   else "Volume confirming", base > 0 and recent >= 1.3 * base))
    return checks


# ---------------------------------------------------------------- PCR
# Zone bands exactly as the user specified (the sideways band is 0.9-1.1,
# centered on PCR = 1):
#   PCR < 0.7          Oversold zone — reversal on the cards
#   0.7 <= PCR < 0.9   Bearish zone
#   0.9 <= PCR <= 1.1  Sideways
#   1.1 <  PCR <= 1.6  Bullish zone
#   PCR > 1.6          Overbought zone — reversal on the cards
PCR_BANDS = [(0.0, 0.7, "Oversold zone — reversal on the cards", "#d97706"),
             (0.7, 0.9, "Bearish zone", "#dc2626"),
             (0.9, 1.1, "Sideways", "#6b7280"),
             (1.1, 1.6, "Bullish zone", "#16a34a"),
             (1.6, 99.0, "Overbought zone — reversal on the cards", "#7c3aed")]


def pcr_zone(pcr: float) -> tuple[str, str]:
    for lo, hi, label, color in PCR_BANDS:
        if lo <= pcr < hi or (label.startswith("Sideways") and pcr == hi) \
                or (label.startswith("Bullish") and pcr == hi):
            return label, color
    return PCR_BANDS[-1][2], PCR_BANDS[-1][3]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_pcr_alpaca(symbol: str, bucket: int = 0) -> dict | None:
    """PCR from Alpaca options snapshots (data.alpaca.markets
    /v1beta1/options/snapshots/{symbol}) — every contract's dailyBar volume
    and openInterest, summed over expiries within ~60 days. Tries the OPRA
    feed first, then the free indicative feed; entitlement decides which
    answers. Returns the standard PCR dict plus 'source', or None."""
    key, secret, _ = _secrets()
    if not key or not secret:
        return None
    import re
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    today = datetime.now(ET).date()
    occ = re.compile(r"^[A-Z]+\d{6}([CP])\d{8}$")
    for feed in ("opra", "indicative"):
        try:
            pv = cv = poi = coi = 0.0
            token, pages = None, 0
            while pages < 20:
                params = {"feed": feed, "limit": 1000,
                          "expiration_date_gte": today.isoformat(),
                          "expiration_date_lte":
                              (today + timedelta(days=60)).isoformat()}
                if token:
                    params["page_token"] = token
                r = requests.get(
                    "https://data.alpaca.markets/v1beta1/options"
                    f"/snapshots/{symbol}", params=params, headers=headers,
                    timeout=30)
                if r.status_code in (401, 403):
                    break  # not entitled to this feed — try the next one
                r.raise_for_status()
                data = r.json()
                for csym, snap in (data.get("snapshots") or {}).items():
                    m = occ.match(csym)
                    if not m:
                        continue
                    vol = float((snap.get("dailyBar") or {}).get("v") or 0)
                    oi = float(snap.get("openInterest") or 0)
                    if m.group(1) == "P":
                        pv += vol
                        poi += oi
                    else:
                        cv += vol
                        coi += oi
                token = data.get("next_page_token")
                pages += 1
                if not token:
                    break
            if cv <= 0 and coi <= 0:
                continue  # empty chain on this feed — try the next
            pcr_vol = pv / cv if cv > 0 else None
            pcr_oi = poi / coi if coi > 0 else None
            src = f"Alpaca ({feed})"
            asof = datetime.now(ET).strftime("%H:%M")
            if pcr_vol is not None:
                return {"pcr": pcr_vol, "kind": "volume",
                        "pcr_vol": pcr_vol, "pcr_oi": pcr_oi, "source": src,
                        "asof": asof, "pv": pv, "cv": cv}
            if pcr_oi is not None:
                return {"pcr": pcr_oi, "kind": "open interest",
                        "pcr_vol": None, "pcr_oi": pcr_oi, "source": src,
                        "asof": asof, "pv": pv, "cv": cv}
        except Exception:
            continue
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_pcr_yahoo(symbol: str, bucket: int = 0) -> dict | None:
    """PCR from Yahoo's options chain (nearest 10 expiries) — the fallback
    when Alpaca options data is unavailable. Same return shape."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        exps = list(t.options or [])[:10]
        if not exps:
            return None
        pv = cv = poi = coi = 0.0
        for e in exps:
            ch = t.option_chain(e)
            pv += float(ch.puts["volume"].fillna(0).sum())
            cv += float(ch.calls["volume"].fillna(0).sum())
            poi += float(ch.puts["openInterest"].fillna(0).sum())
            coi += float(ch.calls["openInterest"].fillna(0).sum())
        pcr_vol = pv / cv if cv > 0 else None
        pcr_oi = poi / coi if coi > 0 else None
        asof = datetime.now(ET).strftime("%H:%M")
        if pcr_vol is not None:
            return {"pcr": pcr_vol, "kind": "volume",
                    "pcr_vol": pcr_vol, "pcr_oi": pcr_oi, "source": "Yahoo",
                    "asof": asof, "pv": pv, "cv": cv}
        if pcr_oi is not None:
            return {"pcr": pcr_oi, "kind": "open interest",
                    "pcr_vol": None, "pcr_oi": pcr_oi, "source": "Yahoo",
                    "asof": asof, "pv": pv, "cv": cv}
        return None
    except Exception:
        return None


def fetch_pcr(symbol: str, bucket: int = 0) -> dict | None:
    """PCR source preference: a VOLUME-based PCR always wins — it's the one
    that moves intraday. Alpaca first; if Alpaca serves only open interest
    (its volume fields come back empty on some feeds), Yahoo's volume PCR
    takes over; open interest is the last resort."""
    a = fetch_pcr_alpaca(symbol, bucket)
    if a is not None and a["kind"] == "volume":
        return a
    y = fetch_pcr_yahoo(symbol, bucket)
    if y is not None and y["kind"] == "volume":
        return y
    return a or y


def build_pcr_gauge(pcr: float) -> go.Figure:
    """Horizontal band strip 0.3-2.0 with the five PCR zones colored and a
    marker at the current PCR (clipped to the strip for display)."""
    fig = go.Figure()
    edges = [(0.3, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.6), (1.6, 2.0)]
    for (lo, hi), (_, _, label, color) in zip(edges, PCR_BANDS):
        fig.add_trace(go.Bar(
            x=[hi - lo], y=["PCR"], orientation="h", base=lo,
            marker_color=color, opacity=0.75, hoverinfo="skip",
            text=label.split(" — ")[0], textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=10), showlegend=False))
    fig.add_trace(go.Scatter(
        x=[min(max(pcr, 0.3), 2.0)], y=["PCR"], mode="markers",
        marker=dict(symbol="triangle-down", size=16, color="black"),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        barmode="stack", height=110,
        margin=dict(l=10, r=10, t=18, b=10),
        title=dict(text=f"PCR {pcr:.2f}", font=dict(size=12), x=0.0),
        xaxis=dict(range=[0.3, 2.0], tickvals=[0.7, 0.9, 1.0, 1.1, 1.6],
                   ticktext=["0.7", "0.9", "1.0", "1.1", "1.6"],
                   fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        dragmode=False)
    return fig
# ---------------------------------------------------------------- chart
PHASES = [(570, 630, "Opening drive", "rgba(37, 99, 235, 0.05)"),
          (750, 780, "Midday", "rgba(107, 114, 128, 0.05)"),
          (900, 960, "Final hour", "rgba(249, 115, 22, 0.06)")]


def build_chart(df: pd.DataFrame, symbol: str, zones: list[dict],
                lines: list[dict], pats: dict[int, tuple[str, str]],
                levels: list[float], pa: dict | None, orng: dict | None,
                show: dict, window_start=None,
                chart_patterns: list[dict] | None = None) -> go.Figure:
    """3 panels: price (candles, EMA 9/21, VWAP, zones, trendlines, PDH/PDC/
    PDL, opening range, pattern markers, time-cycle phases), volume,
    stochastic."""
    price = float(df["Close"].iloc[-1])
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.66, 0.10, 0.24], vertical_spacing=0.05,
        subplot_titles=(f"{symbol} — price action", "Volume",
                        "Stochastic (momentum shift)"))
    x = df.index
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], name="Price",
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["EMA9"], name="EMA 9",
                             line=dict(width=1.4, color="#f59e0b")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["EMA21"], name="EMA 21",
                             line=dict(width=1.4, color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["VWAP"], name="VWAP",
                             line=dict(width=1.2, color="#7c3aed", dash="dash")),
                  row=1, col=1)

    # Time-cycle phases: shaded session bands on the price panel
    if show.get("phases"):
        dates = sorted(set(df.index.date))
        for day in dates:
            base = pd.Timestamp(day, tz=ET)
            for m0, m1, label, color in PHASES:
                fig.add_vrect(
                    x0=base + pd.Timedelta(minutes=m0),
                    x1=base + pd.Timedelta(minutes=m1),
                    fillcolor=color, line_width=0, row=1, col=1,
                    annotation_text=label if day == dates[-1] else None,
                    annotation_position="top left")

    # S/R zones: nearest 3 above + nearest 3 below the last price
    drawn_zones: list[dict] = []
    if show.get("zones"):
        above = sorted([z for z in zones if z["center"] >= price],
                       key=lambda z: z["center"])[:3]
        below = sorted([z for z in zones if z["center"] < price],
                       key=lambda z: -z["center"])[:3]
        drawn_zones = above + below
        for z in drawn_zones:
            # The book's zone language: a zone above price, rejected >= 2
            # times, is a SELLING ZONE (red). A zone below price is a
            # BUYING ZONE (green) — an area price used to trade at and now
            # trades above because demand stepped in; strongest when it is
            # old resistance that flipped (resistance tests came before
            # support tests in the cluster).
            if z["center"] >= price:
                label = f"SELLING ZONE x{z['touches']}"
                flipped = (z["has_highs"] and z["has_lows"]
                           and min(z["lo_idx"]) < max(z["hi_idx"]))
                if flipped:
                    label += " (flipped support)"
                fill = "rgba(220, 38, 38, 0.10)"
                line_c = "rgba(220, 38, 38, 0.45)"
            else:
                label = f"BUYING ZONE x{z['touches']}"
                flipped = (z["has_highs"] and z["has_lows"]
                           and min(z["hi_idx"]) < max(z["lo_idx"]))
                if flipped:
                    label += " (flipped resistance)"
                fill = "rgba(22, 163, 74, 0.10)"
                line_c = "rgba(22, 163, 74, 0.45)"
            fig.add_hrect(
                y0=z["low"], y1=z["high"], row=1, col=1,
                fillcolor=fill, line=dict(color=line_c, width=1),
                annotation_text=label, annotation_position="top right")

    # Prior-day levels + opening range
    if pa:
        for key, pos in (("PDH", "top right"), ("PDC", "bottom right"),
                         ("PDL", "bottom right")):
            fig.add_hline(y=pa[key], line_dash="dash", line_color="#b45309",
                          line_width=1.5, row=1, col=1,
                          annotation_text=f"{key} {pa[key]:,.2f}",
                          annotation_position=pos)
    if orng:
        fig.add_hline(y=orng["high"], line_dash="dash", line_color="#16a34a",
                      line_width=1.2, row=1, col=1,
                      annotation_text=f"OR high {orng['high']:,.2f}",
                      annotation_position="top left")
        fig.add_hline(y=orng["low"], line_dash="dash", line_color="#dc2626",
                      line_width=1.2, row=1, col=1,
                      annotation_text=f"OR low {orng['low']:,.2f}",
                      annotation_position="bottom left")

    # Trendlines
    if show.get("lines"):
        for ln in lines:
            col = "#dc2626" if ln["kind"] == "resistance" else "#16a34a"
            fig.add_trace(go.Scatter(
                x=[x[ln["x0"]], x[ln["x1"]]], y=[ln["y0"], ln["y1"]],
                name=(f"{ln['kind'].title()} trendline x{ln['touches']}"),
                mode="lines", line=dict(width=1.6, color=col, dash="dot")),
                row=1, col=1)

    # Chart-pattern recognition: ranges, triangles/wedges, flags/pennants
    # and staircases. Boundary lines are drawn in a heavier style than the
    # ordinary trendlines, with the pattern name and status on the chart.
    if show.get("chart_patterns") and chart_patterns:
        pattern_colors = {"bull": "#16a34a", "bear": "#dc2626",
                          "neutral": "#7c3aed"}
        for pat in chart_patterns:
            color = pattern_colors.get(pat["bias"], "#7c3aed")
            if pat.get("box"):
                box = pat["box"]
                fig.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=x[box["x0"]], x1=x[box["x1"]],
                    y0=box["y0"], y1=box["y1"],
                    fillcolor="rgba(124, 58, 237, 0.055)",
                    line=dict(color="rgba(124, 58, 237, 0.25)", width=1),
                    row=1, col=1)
            for j, seg in enumerate(pat["segments"]):
                fig.add_trace(go.Scatter(
                    x=[x[seg["x0"]], x[seg["x1"]]],
                    y=[seg["y0"], seg["y1"]],
                    mode="lines", name=pat["name"] if j == 0 else pat["name"],
                    legendgroup=pat["name"], showlegend=(j == 0),
                    line=dict(width=2.4, color=color,
                              dash="solid" if pat["family"] != "staircase"
                              else "dashdot")),
                    row=1, col=1)
            label = f"{pat['name']}<br>{pat['status']}"
            fig.add_trace(go.Scatter(
                x=[x[pat["label_x"]]], y=[pat["label_y"]],
                mode="markers+text", text=[label],
                textposition="top center",
                textfont=dict(size=10, color=color),
                marker=dict(size=8, color=color, symbol="diamond"),
                name=f"{pat['name']} label", showlegend=False),
                row=1, col=1)

    # Candlestick formations: highlighted at a level, faint elsewhere
    if show.get("patterns"):
        strong = [(i, p) for i, p in pats.items()
                  if at_level(float(df["Low"].iloc[i]),
                              float(df["High"].iloc[i]), levels)]
        faint = [(i, p) for i, p in pats.items()
                 if (i, p) not in strong]
        for group, opacity, size, labels in ((strong, 1.0, 11, True),
                                             (faint, 0.35, 8, False)):
            if not group:
                continue
            xs, ys, syms, cols, txts = [], [], [], [], []
            last_labeled = -10
            for i, (name, bias) in group:
                xs.append(x[i])
                if bias == "bear":
                    ys.append(float(df["High"].iloc[i]) * 1.001)
                    syms.append("triangle-down")
                    cols.append("#dc2626")
                elif bias == "bull":
                    ys.append(float(df["Low"].iloc[i]) * 0.999)
                    syms.append("triangle-up")
                    cols.append("#16a34a")
                else:
                    ys.append(float(df["High"].iloc[i]) * 1.001)
                    syms.append("diamond")
                    cols.append("#6b7280")
                # Label at most one formation per 4 bars — in a choppy open
                # the labels otherwise stack into an unreadable blob. The
                # markers themselves are always drawn.
                if labels and i - last_labeled >= 4:
                    txts.append(name)
                    last_labeled = i
                else:
                    txts.append("")
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text" if labels else "markers",
                text=txts, textposition="top center",
                textfont=dict(size=9),
                marker=dict(symbol=syms, size=size, color=cols, opacity=opacity),
                name="Formation at level" if labels else "Formation (mid-chart)"),
                row=1, col=1)

    # Volume
    colors = np.where(df["Close"] >= df["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=x, y=df["Volume"], marker_color=colors,
                         name="Volume", showlegend=False), row=2, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(x=x, y=df["StochK"], name="%K",
                             line=dict(width=1.5, color="#2563eb")), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["StochD"], name="%D",
                             line=dict(width=1.2, color="#f97316")), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#9ca3af", row=3, col=1,
                  annotation_text="Overbought 80", annotation_position="top left")
    fig.add_hline(y=20, line_dash="dot", line_color="#9ca3af", row=3, col=1,
                  annotation_text="Oversold 20", annotation_position="bottom left")

    fig.update_layout(
        height=950, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    if window_start is not None:
        fig.update_xaxes(range=[window_start, df.index[-1]])

    # Fit the price axis to the visible bars. The always-drawn level lines
    # (PDH/PDC/PDL, far zones) otherwise stretch the axis on gap days and
    # squash the candles into a sliver. A level extends the range only when
    # it sits within one visible-range of the action (tested against the
    # base range, so inclusions can't cascade); farther levels stay drawn
    # but off-view until price approaches them.
    vis = df[df.index >= window_start] if window_start is not None else df
    if len(vis):
        pad = 0.004 * price
        base_lo = float(vis["Low"].min())
        base_hi = float(vis["High"].max())
        ylo, yhi = base_lo - pad, base_hi + pad
        span = base_hi - base_lo
        cands = [z["center"] for z in drawn_zones]
        if pa:
            cands += [pa["PDH"], pa["PDC"], pa["PDL"]]
        if orng:
            cands += [orng["high"], orng["low"]]
        for v in cands:
            if base_lo - span <= v <= base_hi + span:
                ylo = min(ylo, v - pad * 0.5)
                yhi = max(yhi, v + pad * 0.5)
        fig.update_yaxes(range=[ylo, yhi], row=1, col=1)
    return fig


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Price-Action Charting", layout="wide")
st.title("📐 Price-Action Charting")
st.caption("The charting techniques from the *Price Action Trading* book — "
           "zones, trendlines, chart-pattern recognition, formations at "
           "levels, EMA 9/21, stochastic — drawn on any US stock. "
           "Display only: no verdicts, no signals.")

_key, _secret, _feed = _secrets()
if not _key or not _secret:
    st.error("Alpaca keys are not set for this app. Add APCA_API_KEY_ID and "
             "APCA_API_SECRET_KEY (same values as the stock-analyzer) under "
             "Settings → Secrets, then reboot the app.")
    st.stop()

with st.sidebar:
    st.header("Chart")
    symbol = st.text_input("Stock symbol", value="AAPL").strip().upper() or "AAPL"
    tf_label = st.selectbox("Bar size", ["1 min", "5 min", "15 min"], index=0)
    chart_tf = st.selectbox(
        "Chart timeframe",
        ["5 minutes", "15 minutes", "1 hour", "2 hours", "3 hours",
         "4 hours", "Since market open"],
        index=6,
        help="How much recent history the chart shows — same control as the "
             "stock-analyzer's. Zones, trendlines and formations are "
             "computed on the full session and stay drawn across the window.")
    live = st.toggle("Live tracking (auto-refresh)", value=True,
                     help="When on, the chart re-runs automatically at the "
                          "interval below.")
    refresh_secs = st.number_input(
        "Refresh every (seconds)", min_value=30, max_value=3600, value=30,
        step=5,
        help="How often live tracking re-runs the chart. Below 30s the "
             "server runs out of threads/memory.")
    if st.button("🔄 Refresh now"):
        # The click itself reruns the app; bumping the nonce changes the
        # fetch bucket, so the loaders pull fresh bars.
        st.session_state["refresh_nonce"] = \
            st.session_state.get("refresh_nonce", 0) + 1
    st.header("Show")
    show = {
        "zones": st.checkbox("Support / resistance zones", value=True),
        "lines": st.checkbox("Trendlines", value=True),
        "chart_patterns": st.checkbox(
            "Pattern recognition (ranges, triangles, wedges, flags, "
            "pennants, staircases)", value=True),
        "patterns": st.checkbox("Candlestick formations", value=True),
        "phases": st.checkbox("Time-cycle phases", value=True),
    }
    st.caption("Formations are labeled only when they form AT a level "
               "(zone, PDH/PDC/PDL, opening-range edge) — the book's rule. "
               "Mid-chart formations appear as faint markers.")


def in_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 570 <= mins < 960


def render():
    try:
        df1 = fetch_bars_1m(symbol, _bucket(live, int(refresh_secs)))
    except Exception as e:
        st.error(f"Could not load bars for {symbol}: {e}")
        return
    if df1.empty or len(df1) < 60:
        st.info(f"Not enough intraday data for {symbol} yet.")
        return
    dates = sorted(set(df1.index.date))
    df = df1[df1.index.normalize() == pd.Timestamp(dates[-1], tz=ET)]
    minutes = {"1 min": 1, "5 min": 5, "15 min": 15}[tf_label]
    # Indicators are computed on the FULL multi-day frame at the chosen bar
    # size, then sliced to today — the analyzer's warm-up rule. Computing
    # them on today's resampled bars alone starved the rolling indicators:
    # a 5-min stochastic needed 14 x 5-min bars and stayed blank until
    # 10:40 ET (1-min bars hid the same gap at 09:43). With prior sessions
    # warming the window, the stochastic (and EMAs) are live from the open;
    # VWAP still resets each session (it is grouped by date).
    full_disp = add_indicators(resample(df1, minutes))
    disp = full_disp[full_disp.index.normalize()
                     == pd.Timestamp(dates[-1], tz=ET)]
    if len(disp) < 10:
        st.info("Only the first few bars of the session so far — the chart "
                "builds as bars arrive.")
        return

    pa = prior_day_levels(df1)
    today_df = df1[df1.index.date == dates[-1]]
    orng = opening_range(today_df)
    zones = sr_zones(disp)
    lines = trendlines(disp)
    chart_pats = detect_chart_patterns(disp)
    pats = candle_patterns(disp)
    level_vals = ([pa["PDH"], pa["PDC"], pa["PDL"]] if pa else []) \
        + [z["center"] for z in zones] \
        + ([orng["high"], orng["low"]] if orng else [])

    last = disp.iloc[-1]
    prev_close = pa["PDC"] if pa else float(disp["Close"].iloc[0])
    chg = (float(last["Close"]) / prev_close - 1) * 100 if prev_close else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(symbol, f"${float(last['Close']):,.2f}", f"{chg:+.2f}% vs prior close")
    c2.metric("Zones on chart",
              f"{sum(1 for z in zones if z['center'] >= float(last['Close']))} selling / "
              f"{sum(1 for z in zones if z['center'] < float(last['Close']))} buying")
    c3.metric("Stochastic %K / %D",
              f"{float(last['StochK']):.0f} / {float(last['StochD']):.0f}"
              if pd.notna(last["StochK"]) else "—")
    checks = confluence(disp, zones, level_vals)
    score = sum(1 for _, ok in checks if ok)
    c4.metric("Confluence (book stack)",
              f"{score} / {len(checks)}" if checks else "—")

    trailing = CHART_TF_TRAILING[chart_tf]
    window_start = (disp.index[-1] - pd.Timedelta(minutes=trailing)
                    if trailing is not None else None)
    st.plotly_chart(build_chart(disp, symbol, zones, lines, pats, level_vals,
                                pa, orng, show, window_start=window_start,
                                chart_patterns=chart_pats),
                    use_container_width=True,
                    key=f"pa_{symbol}_{tf_label}_{chart_tf.replace(' ', '_')}")

    if show.get("chart_patterns"):
        st.markdown("**Pattern recognition** (the book's chart patterns — "
                    "ranges, triangles, wedges, flags/pennants and "
                    "staircases):")
        if chart_pats:
            icons = {"bull": "🟢", "bear": "🔴", "neutral": "🟣"}
            rows = []
            for pat in chart_pats:
                trigger = (f"trigger ${pat['trigger']:,.2f}"
                           if pat["trigger"] is not None else "trigger —")
                invalid = (f"invalidation ${pat['invalidation']:,.2f}"
                           if pat["invalidation"] is not None
                           else "invalidation —")
                rows.append(
                    f"- {icons.get(pat['bias'], '🟣')} **{pat['name']} — "
                    f"{pat['status']}** · {trigger} · {invalid} · "
                    f"{pat['detail']}")
            st.markdown("\n".join(rows))
        else:
            st.caption("No qualifying chart pattern in the current session "
                       "yet. The detector requires a repeated range, "
                       "converging boundaries, a fierce pole followed by a "
                       "contained pause, or a clean stepping sequence — it "
                       "does not force a pattern onto ordinary noise.")

    # Put/Call Ratio strip (user's bands) — slow bucket: PCR does not need
    # the fast refresh cycle, and the chain pull takes a few seconds.
    p = fetch_pcr(symbol, _bucket(live, int(refresh_secs), bucket_secs=300))
    if p is not None:
        zlabel, _zcolor = pcr_zone(p["pcr"])
        pc1, pc2 = st.columns([1, 3])
        with pc1:
            st.metric(f"Put/Call Ratio ({p['kind']})", f"{p['pcr']:.2f}")
            st.markdown(f"**{zlabel}**")
            oi_txt = (f" · PCR by open interest {p['pcr_oi']:.2f}"
                      if p["pcr_oi"] is not None else "")
            st.caption(f"Options chain via {p['source']} · updated "
                       f"{p['asof']} ET · puts {p['pv']:,.0f} / calls "
                       f"{p['cv']:,.0f} traded today{oi_txt}")
        with pc2:
            st.plotly_chart(build_pcr_gauge(p["pcr"]),
                            use_container_width=True, key=f"pcr_{symbol}")
    elif yf is None:
        st.caption("PCR unavailable — Alpaca options returned nothing and "
                   "yfinance is not installed for this app (add it to "
                   "requirements.txt and reboot).")
    else:
        st.caption(f"PCR unavailable for {symbol} — no options chain data.")
    if checks:
        st.markdown("**Confluence checklist** (the book's Premium stack, "
                    "at the last bar):")
        st.markdown("\n".join(
            f"- {'✅' if ok else '⬜'} {label}" for label, ok in checks))
    st.caption(
        f"Alpaca {str(_feed).upper()} · {len(df1):,} 1-min bars loaded "
        f"({dates[0]} → {dates[-1]}) · showing {len(disp):,} {tf_label} bars "
        f"· window: {chart_tf} · "
        f"updated {datetime.now(ET).strftime('%H:%M:%S')} ET")


if live and in_market_hours():
    st.fragment(run_every=int(refresh_secs))(render)()
elif live:
    render()
    st.caption("Live tracking is on, but the market is closed — showing the "
               "last session without auto-refresh.")
else:
    render()
    st.caption("Live tracking is off — this is a one-time snapshot. "
               "Turn on **Live tracking** in the sidebar for auto-refresh, "
               "or use 🔄 Refresh now.")

with st.expander("How to read this chart"):
    st.markdown(
        "- **Green BUYING ZONE / red SELLING ZONE bands** — the book's zones. "
        "A **buying zone** (green, below price) is an area the stock used to "
        "trade at and now trades above because demand stepped in — classically "
        "old resistance that broke and turned into support (labeled \"flipped "
        "resistance\"); the book uses the **bottom of the buying zone as the "
        "stop**. A **selling zone** (red, above price) is a level sellers have "
        "already **rejected 2 or more times** — the book's rule is not to buy "
        "calls into it; only a breakout above it changes its role.\n"
        "- **Dotted trendlines** — through the two most recent swing "
        "highs/lows, kept only when tested at least twice. Price hitting the "
        "upper line is the book's selling area; the lower line, buying area.\n"
        "- **Pattern recognition panel + heavy outlines** — the detector "
        "names the current range, triangle, wedge, flag/pennant or staircase "
        "and states whether it is forming/active or has confirmed a break. "
        "The listed trigger is the boundary that confirms the move; the "
        "invalidation is the opposite boundary. A flag/pennant is only "
        "recognized after a fierce pole followed by a small contained "
        "pause, as the book defines it.\n"
        "- **Triangles / diamonds on the candles** — candlestick formations. "
        "Labeled ones formed **at a level** (that's when the book says they "
        "count); faint unlabeled ones formed mid-chart and are shown for "
        "reference only.\n"
        "- **Amber dashed lines (PDH / PDC / PDL)** — yesterday's high, "
        "close, low: hard levels into today's open (the book's AHPMB idea).\n"
        "- **Shaded bands** — time-cycle phases of the session: the opening "
        "drive (first hour), the midday lull, and the final hour, when moves "
        "tend to start or exhaust.\n"
        "- **Stochastic (bottom)** — momentum shifts: blue %K crossing orange "
        "%D is the event; crossing up from under 20 or down from over 80 is "
        "the one that matters, especially right at a level.\n"
        "- **Put/Call Ratio strip** — options sentiment for the symbol "
        "(by traded volume; the open-interest PCR is shown alongside). "
        "Under 0.7 = oversold, reversal watch · 0.7–0.9 = bearish zone · "
        "0.9–1.1 = sideways · 1.1–1.6 = bullish zone · over 1.6 = overbought, "
        "reversal watch. Volume PCR builds during the session, so the first "
        "hour's reading rests on thin volume.")
