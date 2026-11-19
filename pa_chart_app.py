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
  * MURPHY REVERSAL PATTERNS (John J. Murphy, "Technical Analysis of the
    Financial Markets", Ch. 5-6) — head & shoulders top / inverse bottom,
    double tops & bottoms (triple variant folded in), and broadening
    formations / megaphones; strict detectors in the same style, with the
    book's confirmation rules (close beyond the line, never intraday
    penetration alone), volume confirmation on breakouts (essential at
    bottoms, advisory at tops), and measured minimum targets projected
    from the pattern height. Intraday note: the book's patterns are
    daily-chart patterns that take weeks to form; what is detected here
    are miniature fractal versions on one session of bars, and the
    book's reliability claims do not transfer to them.
  * MINERVINI (Mark Minervini, "Trade Like a Stock Market Wizard", Ch. 5,
    10, 13) — the TREND TEMPLATE checklist (SMA 50/150/200, 200-day
    slope, 52-week high/low distances) with a four-stage label, computed
    on daily bars at full scale; the VCP (volatility contraction pattern:
    successive pullbacks each ~half the previous one, volume drying
    across the sequence, ending in a tight pivot); and the PIVOT ENTRY
    presentation on every pattern (pivot = trigger, stop = invalidation,
    volume pace vs the average day, a "chased" warning when price has
    run >0.50% past the pivot). The book's patterns are multi-week
    daily-chart patterns; the intraday VCP is a fractal of the geometry,
    and the book's reliability claims do not transfer to it. The
    Template is daily context only — it never changes the weighted
    confluence score.
  * BROOKS — SPY & QQQ TAB (Al Brooks, "Trading Price Action Trends",
    Ch. 1-26) — a side-by-side 5-minute view of SPY and QQQ with the
    book's intraday read on each: the day type (spike and channel,
    trend from the open, trending trading range, reversal day, trend
    resumption, stairs, or a trading-range day), the always-in
    direction, the spike/channel phase, the High 1 / High 2 / Low 1 /
    Low 2 pullback count, measured-move targets (spike, channel and
    range height — profit targets, not reversal spots), and the Ch. 19
    strength checklist. The book's numbers are Emini-based, so the
    tab translates them into each ETF's own average daily range and
    bar size. The day types are read off one session; the book's
    reliability claims do not transfer to a single day. Display only.
  * CANDLESTICK FORMATIONS marked where they occur — highlighted ONLY when
    they form AT a level (the book's core rule: a candle mid-chart means
    nothing); mid-chart formations are drawn faint
  * MOVING AVERAGES — EMA 9 / 21 (the book's crossover pair), EMA 50 / 200
    trend-context lines on the chosen bar size (sidebar toggle), + session
    VWAP. DAILY EMA 50 / 200 flat reference lines (sidebar toggle) —
    computed from daily closes through the prior completed session, so
    the daily clock reads on the intraday chart. Display only.
  * Prior-day levels (the book's AHPMB levels): PDH / PDC / PDL, and the
    30-minute opening range
  * SESSION VOLUME PROFILE — the day's volume by price level (24 bins
    at the chosen bar size): the point of control (POC, most volume
    traded) and the 70% value area (VAH/VAL), drawn as a histogram on
    the right of the price panel. Display only — no score, no signal.
  * STOCHASTIC %K / %D — the book's momentum-shift gauge (bottom panel)
  * TIME CYCLES — the book names the technique but gives no method; this app
    draws the intraday session phases (opening drive / first hour, midday,
    final hour) as the cycle guide
  * A WEIGHTED confluence score — the indicator hierarchy scored at the
    last bar: location & structure 30, intraday trend 25, volume 15,
    pattern & candle at a level 10, stochastic trigger 10, session
    context (PCR zone + fresh EMA cross) 10. Bullish and bearish
    evidence are scored separately; the larger total is the lean, and a
    Tier-1-vs-Tier-2 conflict is flagged rather than averaged away.
    Display only — no verdict, no gates.

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

# Version stamp: the footer shows this tag plus a hash computed live from
# the deployed file itself, so the page proves which build is running —
# compare it with the hash named when the file was delivered.
BUILD_TAG = "2026-11-18"


def _build_hash() -> str:
    try:
        import hashlib
        from pathlib import Path
        return hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:6]
    except Exception:
        return "??????"


BUILD_HASH = _build_hash()
LOAD_DAYS = 7            # calendar days of 1-min bars (covers ~5 sessions)
DAILY_EMA_DAYS = 300     # calendar days of daily bars for the daily-EMA
                         # reference lines (~208 sessions — enough to warm
                         # a 200-day EMA); the 1-min frame is far too short
DAILY_TEMPLATE_DAYS = 400  # calendar days of daily OHLCV for the Minervini
                         # Trend Template: 252 sessions for the 52-week
                         # high/low, plus SMA200 warm-up and the 1-month
                         # slope lookback
DAILY_EMA_MAX_DIST = 0.15  # a daily-EMA line more than 15% from the last
                           # price is hidden (it would stretch the price
                           # panel); its value is still named in the caption
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
                  days: int = LOAD_DAYS, end: str = "") -> pd.DataFrame:
    """Last `days` calendar days of 1-minute bars, regular hours only
    (09:30-16:00 ET), paginated (Alpaca's limit is per page). `end`
    (an ET date, "YYYY-MM-DD") anchors the window to a past session for
    the date view: bars through the end of that date are loaded, so the
    selected session is the frame's last session and the usual warm-up
    and prior-day logic apply to it unchanged."""
    _, _, feed = _secrets()
    end_param = None
    if end:
        end_et = datetime.strptime(end, "%Y-%m-%d").replace(
            tzinfo=ET) + timedelta(days=1)
        start = (end_et - timedelta(days=days)).astimezone(
            ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_param = end_et.astimezone(
            ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        start = (datetime.now(ET) - timedelta(days=days)).strftime(
            "%Y-%m-%dT00:00:00Z")
    rows, token = [], None
    while True:
        params = {"timeframe": "1Min", "start": start, "limit": 10000,
                  "feed": feed, "sort": "asc", "adjustment": "split"}
        if end_param:
            params["end"] = end_param
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


@st.cache_data(show_spinner=False)
def fetch_daily_bars(symbol: str, bucket: int = 0,
                     days: int = DAILY_TEMPLATE_DAYS,
                     end: str = "") -> pd.DataFrame:
    """Daily OHLCV bars — one cheap 1Day-bars pull. The 1-min frame only
    covers LOAD_DAYS (~5 sessions), far too short for a 200-day average
    or a 52-week high/low, so this is a separate fetch. `end` anchors the
    window to a past session for the date view, mirroring fetch_bars_1m.
    The index is the trading date (the UTC calendar date of each daily
    bar) — that matches the session date in either of Alpaca's daily-bar
    timestamp conventions."""
    _, _, feed = _secrets()
    if end:
        end_et = datetime.strptime(end, "%Y-%m-%d").replace(
            tzinfo=ET) + timedelta(days=1)
        start = (end_et - timedelta(days=days)).astimezone(
            ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_param = end_et.astimezone(
            ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        start = (datetime.now(ET) - timedelta(days=days)).strftime(
            "%Y-%m-%dT00:00:00Z")
        end_param = None
    rows, token = [], None
    while True:
        params = {"timeframe": "1Day", "start": start, "limit": 10000,
                  "feed": feed, "sort": "asc", "adjustment": "split"}
        if end_param:
            params["end"] = end_param
        if token:
            params["page_token"] = token
        data = alpaca_get(f"/stocks/{symbol}/bars", params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"])
    ts = pd.to_datetime([r["t"] for r in rows], utc=True)
    df = pd.DataFrame({
        "Open": [float(r["o"]) for r in rows],
        "High": [float(r["h"]) for r in rows],
        "Low": [float(r["l"]) for r in rows],
        "Close": [float(r["c"]) for r in rows],
        "Volume": [float(r["v"]) for r in rows]},
        index=pd.DatetimeIndex(ts).date)
    return df.sort_index()


@st.cache_data(show_spinner=False)
def fetch_daily_closes(symbol: str, bucket: int = 0,
                       days: int = DAILY_EMA_DAYS,
                       end: str = "") -> pd.Series:
    """Daily closes for the daily-EMA reference lines — the Close column
    of fetch_daily_bars. Kept as its own function so the EMA path is
    unchanged (same window, same values)."""
    bars = fetch_daily_bars(symbol, bucket, days, end)
    if bars.empty:
        return pd.Series(dtype=float)
    return bars["Close"]


def daily_ema_levels(closes: pd.Series, session_date) -> dict | None:
    """Daily EMA 50 / 200 as of the last COMPLETED session before
    `session_date` — the session's own close is not known while it is
    trading (and in date view it is the day being replayed), so it is
    excluded: no lookahead. An EMA draws only when enough closes warmed
    it (EMA 50 needs >= 50 closes, EMA 200 needs >= 200); None when
    neither qualifies."""
    if closes is None or len(closes) < 2:
        return None
    prior = closes[[d < session_date for d in closes.index]]
    n = len(prior)
    if n < 2:
        return None
    out = {"asof": prior.index[-1], "n": n, "ema50": None, "ema200": None}
    if n >= 50:
        out["ema50"] = float(ema(prior, 50).iloc[-1])
    if n >= 200:
        out["ema200"] = float(ema(prior, 200).iloc[-1])
    if out["ema50"] is None and out["ema200"] is None:
        return None
    return out


def trend_template(daily: pd.DataFrame, session_date) -> dict | None:
    """Minervini's Trend Template (Ch. 5, p. 79) + four-stage label, from
    daily OHLCV through the last COMPLETED session before `session_date`
    — no lookahead. Simple moving averages per the book (NOT the app's
    daily EMA lines — EMA and SMA disagree near crosses, and the book's
    numbers are SMA). Needs >= 252 sessions (52-week high/low + SMA200
    warm-up + the 1-month slope lookback); None when the history is too
    short. Criterion 8 (IBD RS ranking >= 70) is vendor data the app
    cannot compute — it is reported as n/a, never approximated. The
    checklist is daily context only; it never changes the weighted
    confluence score."""
    if daily is None or daily.empty or len(daily) < 252:
        return None
    prior = daily[[d < session_date for d in daily.index]]
    n = len(prior)
    if n < 252:
        return None
    C = prior["Close"].astype(float)
    H = prior["High"].astype(float)
    L = prior["Low"].astype(float)
    V = prior["Volume"].astype(float)
    O = prior["Open"].astype(float)
    sma = lambda k: float(C.rolling(k).mean().iloc[-1])
    sma50, sma150, sma200 = sma(50), sma(150), sma(200)
    sma200_s = C.rolling(200).mean()
    sma200_21 = float(sma200_s.iloc[-22])
    sma200_84 = float(sma200_s.iloc[-85])
    c0 = float(C.iloc[-1])
    high52 = float(H.tail(252).max())
    low52 = float(L.tail(252).min())
    criteria = [
        {"label": f"Price ${c0:,.2f} above SMA150 ${sma150:,.2f} and "
                  f"SMA200 ${sma200:,.2f}",
         "met": bool(c0 > sma150 and c0 > sma200)},
        {"label": f"SMA150 ${sma150:,.2f} above SMA200 ${sma200:,.2f}",
         "met": bool(sma150 > sma200)},
        {"label": f"SMA200 rising >= 1 month "
                  f"(${sma200_21:,.2f} -> ${sma200:,.2f}; 4-5 mo: "
                  f"${sma200_84:,.2f} -> ${sma200:,.2f})",
         "met": bool(sma200 > sma200_21)},
        {"label": f"SMA50 ${sma50:,.2f} above SMA150 and SMA200",
         "met": bool(sma50 > sma150 and sma50 > sma200)},
        {"label": f"Price ${c0:,.2f} above SMA50 ${sma50:,.2f}",
         "met": bool(c0 > sma50)},
        {"label": f"Price >= 30% above 52-week low ${low52:,.2f} "
                  f"({(c0 / low52 - 1) * 100:.0f}% above)",
         "met": bool(c0 >= 1.30 * low52)},
        {"label": f"Price within 25% of 52-week high ${high52:,.2f} "
                  f"({(1 - c0 / high52) * 100:.0f}% below)",
         "met": bool(c0 >= 0.75 * high52)},
        {"label": "RS ranking >= 70 — n/a (IBD data not available)",
         "met": None},
    ]
    met = sum(1 for c in criteria[:7] if c["met"])

    # Stage label (Weinstein's four stages, Minervini Ch. 5, pp. 63-78).
    # Stage 2: Template trend criteria met + daily higher highs/lows
    # stair-stepping + more volume on up days than down days (last 20).
    # Stage 4: below a falling 200-day and near the 52-week low. Stage 1:
    # below a FLAT 200-day (slope < 0.10% over a month — adaptation;
    # "flat" is qualitative in the book). Stage 3: above the 200-day but
    # the 200-day rolling over, or a major one-day break in the last 20
    # sessions (decline > 3% — adaptation of "largest one-day decline",
    # p. 73). Anything else: no label rather than a forced one.
    slope21 = sma200 - sma200_21
    slope_prev21 = sma200_21 - float(sma200_s.iloc[-43])
    hi_idx, lo_idx = swing_points(prior, w=5)
    hh_up = len(hi_idx) >= 2 and float(H.iloc[hi_idx[-1]]) > float(
        H.iloc[hi_idx[-2]])
    ll_up = len(lo_idx) >= 2 and float(L.iloc[lo_idx[-1]]) > float(
        L.iloc[lo_idx[-2]])
    last20 = prior.tail(20)
    up_vol = float(V.tail(20)[C.tail(20) > O.tail(20)].sum())
    dn_vol = float(V.tail(20)[C.tail(20) < O.tail(20)].sum())
    worst_day = float((C.tail(20) / C.tail(20).shift(1) - 1).min())
    if (all(c["met"] for c in criteria[:5]) and hh_up and ll_up
            and up_vol > dn_vol):
        stage = "Stage 2 (advancing)"
    elif c0 < sma200 and slope21 < 0 and c0 <= 1.10 * low52:
        stage = "Stage 4 (declining)"
    elif c0 < sma200 and abs(slope21) / sma200 < 0.001:
        stage = "Stage 1 (basing)"
    elif c0 > sma200 and ((slope21 < 0 < slope_prev21)
                          or worst_day < -0.03):
        stage = "Stage 3 (topping)"
    else:
        stage = "Stage unclear — no label"
    return {"asof": prior.index[-1], "n": n, "criteria": criteria,
            "met": met, "stage": stage, "sma50": sma50, "sma150": sma150,
            "sma200": sma200, "high52": high52, "low52": low52}


def _volume_pace(disp: pd.DataFrame,
                 avg_daily_vol: float | None) -> float | None:
    """Projected day volume vs the average daily volume (Minervini's
    intraday rule, p. 229): cumulative session volume / fraction of the
    session elapsed, divided by the average daily volume. The book's
    example is 300-400% of the average day; the app reports the number
    and never gates on it — the 1.2x break-volume gate stays the
    confirmation rule. None when it cannot be computed."""
    if avg_daily_vol is None or avg_daily_vol <= 0 or len(disp) < 2:
        return None
    cum_vol = float(disp["Volume"].sum())
    idx = disp.index
    bar_min = (idx[1] - idx[0]).total_seconds() / 60.0
    if bar_min <= 0:
        return None
    elapsed = (idx[-1] - idx[0]).total_seconds() / 60.0 + bar_min
    frac = elapsed / 390.0
    if frac <= 0 or frac > 1.5:
        return None
    return (cum_vol / frac) / avg_daily_vol


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
    """EMA 9/21/50/200, session-anchored VWAP, stochastic — computed on
    the frame as given (for 1-min frames the VWAP is the exact session
    VWAP; on resampled frames it is the standard resampled
    approximation). The EMAs run across the whole loaded frame, so the
    50/200 arrive at today's open already warmed by the prior sessions."""
    df = df.copy()
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
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


def session_volume_profile(df_day: pd.DataFrame, n_bins: int = 24,
                           va_pct: float = 0.70) -> dict | None:
    """Session volume profile from the session's displayed bars.

    Each bar's volume is assigned to the price bin holding its typical
    price (H+L+C)/3 — the standard bar-based approximation of the
    trade-by-trade profile. POC is the bin with the most volume; the
    value area is the contiguous range around the POC holding `va_pct`
    of the session's volume, grown outward one adjacent bin at a time
    (always taking the higher-volume side first).

    Computed on the full session at the chosen bar size, like the zones
    and trendlines — the visible window only crops the chart, never the
    profile. Display only — no score, no signal. Returns None when the
    session has no usable price range or no volume.
    """
    if df_day is None or df_day.empty or len(df_day) < 2:
        return None
    lo = float(df_day["Low"].min())
    hi = float(df_day["High"].max())
    vol = df_day["Volume"].to_numpy(dtype=float)
    total = float(vol.sum())
    if total <= 0 or not (hi > lo):
        return None
    n = max(2, min(n_bins, len(df_day)))
    edges = np.linspace(lo, hi, n + 1)
    tp = ((df_day["High"] + df_day["Low"] + df_day["Close"]) / 3.0
          ).to_numpy(dtype=float)
    idx = np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, n - 1)
    bin_vol = np.bincount(idx, weights=vol, minlength=n)
    if bin_vol.max() <= 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_i = int(np.argmax(bin_vol))
    # Value area: grow from the POC, always taking the higher-volume
    # adjacent bin, until va_pct of the session's volume is inside.
    lo_i = hi_i = poc_i
    inside = float(bin_vol[poc_i])
    while inside < va_pct * total and (lo_i > 0 or hi_i < n - 1):
        up = bin_vol[hi_i + 1] if hi_i < n - 1 else -1.0
        dn = bin_vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            inside += float(bin_vol[hi_i])
        else:
            lo_i -= 1
            inside += float(bin_vol[lo_i])
    return {
        "centers": centers, "volumes": bin_vol,
        "bin_width": float(edges[1] - edges[0]),
        "poc_i": poc_i, "poc": float(centers[poc_i]),
        "vah": float(edges[hi_i + 1]), "val": float(edges[lo_i]),
        "total_volume": total, "va_volume": inside,
        "n_bins": n,
    }


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
    V = df["Volume"].to_numpy()
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
            brk_i = None
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
                    brk_i = event_i
                    status = ("BREAKOUT CONFIRMED" if C[-1] >= hi
                              else "BREAKOUT / BACK IN RANGE")
                    bias = "bull"
                elif first_below < first_above:
                    event_i = end + 1 + first_below
                    if first_below > 5 or n - 1 - event_i > 8:
                        continue
                    if C[-1] > hi:
                        continue
                    brk_i = event_i
                    status = ("BREAKDOWN CONFIRMED" if C[-1] <= lo
                              else "BREAKDOWN / BACK IN RANGE")
                    bias = "bear"
                elif end < n - 4:
                    continue
            trigger = hi if bias != "bear" else lo
            invalidation = lo if bias != "bear" else hi
            # Murphy (Ch. 7): a breakout should come on heavier volume —
            # essential at bottoms, advisory at tops. The measured target
            # is the range height projected from the break (p. 151).
            vol_note = ""
            target_note = ""
            vol_bonus = 0
            if brk_i is not None:
                brk_vol = _break_vol_ratio(V, start, end, brk_i)
                vol_note = f" · vol ×{brk_vol:.1f} on break"
                if status == "BREAKOUT CONFIRMED":
                    if brk_vol < 1.2:
                        status = "FORMING"
                        vol_note += " — breakout lacks volume"
                    else:
                        target_note = (f" · target "
                                       f"${hi + (hi - lo):,.2f} (min)")
                elif status == "BREAKDOWN CONFIRMED":
                    if brk_vol >= 1.5:
                        vol_bonus = 8
                    target_note = (f" · target "
                                   f"${lo - (hi - lo):,.2f} (min)")
            score = (upper_tests + lower_tests) * 10 + min(length, 60) \
                + (30 if status != "FORMING" else 12) - width * 3 \
                - abs(n - 1 - end) + vol_bonus
            cand = _make_chart_pattern(
                "Range / Consolidation", bias, status, start, n - 1,
                [{"x0": start, "y0": hi, "x1": n - 1, "y1": hi},
                 {"x0": start, "y0": lo, "x1": n - 1, "y1": lo}],
                start, hi, trigger, invalidation,
                f"{length} bars · {upper_tests} upper tests / "
                f"{lower_tests} lower tests · width {width:.2f}%"
                f"{vol_note}{target_note}",
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
    V = df["Volume"].to_numpy()
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
            # Murphy (Ch. 6-7): breakout between 2/3 and 3/4 of the
            # base-to-apex width, or the triangle loses potency (p. 133);
            # a breakout should come on heavier volume (essential on the
            # upside); the measured target is the base height projected
            # from the breakout point (p. 135).
            apex_penalty = 0
            if upper[0] != lower[0]:
                apex_x = (lower[1] - upper[1]) / (upper[0] - lower[0])
                if apex_x > first and \
                        (n - 1) > first + 0.75 * (apex_x - first):
                    apex_penalty = 15
            vol_note = ""
            target_note = ""
            vol_bonus = 0
            if status in ("BREAKOUT CONFIRMED", "BREAKDOWN CONFIRMED"):
                brk_vol = _break_vol_ratio(V, first, end, cross_i)
                vol_note = f" · vol ×{brk_vol:.1f} on break"
                if status == "BREAKOUT CONFIRMED":
                    if brk_vol < 1.2:
                        status = "FORMING"
                        vol_note += " — breakout lacks volume"
                    else:
                        target_note = (f" · target "
                                       f"${u_last + start_width:,.2f} (min)")
                else:
                    if brk_vol >= 1.5:
                        vol_bonus = 8
                    target_note = (f" · target "
                                   f"${l_last - start_width:,.2f} (min)")
            score = len(highs) + len(lows) + containment * 20 \
                + (1.0 - compression) * 25 + min(end - first, 60) * 0.25 \
                + (25 if status != "FORMING" else 8) \
                + vol_bonus - apex_penalty
            cand = _make_chart_pattern(
                name, bias, status, first, n - 1,
                [{"x0": first, "y0": u0, "x1": n - 1, "y1": u_last},
                 {"x0": first, "y0": l0, "x1": n - 1, "y1": l_last}],
                max(first, n - 12), (u_last + l_last) / 2.0,
                trigger, invalidation,
                f"{len(highs)} swing highs / {len(lows)} swing lows · "
                f"range compressed to {compression:.0%} · "
                f"{containment:.0%} contained"
                f"{vol_note}{target_note}",
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
                    # Murphy (p. 143): volume should dry up noticeably
                    # while the flag/pennant forms.
                    pause_vol = float(V[pole_end:end + 1].mean())
                    if pause_vol > 0.8 * pole_vol:
                        continue
                    # The pause must be RECENT: the fitted pause lines are
                    # extrapolated to the last bar, so a stale pause's
                    # lines drift and the level check below reads a
                    # phantom break (a rising pause drifts above price
                    # and fakes a breakdown — a Bear Flag with an upside
                    # target). The range detector has the same tail-
                    # freshness rule; the FORMING branch below is stricter
                    # still (end >= n - 4).
                    if end < n - 7:
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
                    # Murphy (Ch. 7): a breakout should come on heavier
                    # volume — essential on the upside. The measured target
                    # is the pole length projected from the breakout
                    # ("fly at half-mast", pp. 143-144).
                    vol_note = ""
                    target_note = ""
                    vol_bonus = 0
                    if status in ("BREAKOUT CONFIRMED",
                                  "BREAKDOWN CONFIRMED"):
                        brk_vol = _break_vol_ratio(V, pole_start, end, n - 1)
                        vol_note = f" · vol ×{brk_vol:.1f} on break"
                        if status == "BREAKOUT CONFIRMED":
                            if brk_vol < 1.2:
                                status = "FORMING"
                                vol_note += " — breakout lacks volume"
                            else:
                                target_note = (f" · target "
                                               f"${u_last + move_abs:,.2f} "
                                               f"(min)")
                        else:
                            if brk_vol >= 1.5:
                                vol_bonus = 8
                            target_note = (f" · target "
                                           f"${l_last - move_abs:,.2f} (min)")
                    score = move * 8 + min(vol_ratio, 2.0) * 8 \
                        + float(inside.mean()) * 15 + min(cons_len, 30) \
                        + (25 if status != "FORMING" else 8) + vol_bonus
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
                        f"pause {cons_len + 1} bars"
                        f"{vol_note}{target_note}",
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


# ---------------------------------------------------------------- Murphy reversal patterns
# John J. Murphy, "Technical Analysis of the Financial Markets", Ch. 5-6.
# The book's patterns are daily-chart patterns that take weeks to form; the
# detectors below find miniature fractal versions on one session of bars.
# Strict by design (same philosophy as the other detectors): geometry gates
# are hard, confirmation is a CLOSE beyond the line (never intraday
# penetration alone), and a pattern that does not qualify is not drawn.
# Bar-count and height minimums are intraday adaptations — the book gives
# durations in weeks/months, which cannot exist in one session.

def _vol_str(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def _alternating_swings(df: pd.DataFrame) -> list[tuple[int, str, float]]:
    """Swing points as an alternating H/L sequence [(idx, kind, price)].
    Consecutive same-kind swings collapse to the most extreme one."""
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
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
    return points


def _line_cross(C, vals, n, side, pct=0.0010, window=7):
    """Fresh close beyond a line, Murphy-style: a close beyond the line by
    >= pct of the line value, OR two consecutive closes beyond it (the
    intraday analog of the book's two-day rule), with the crossing bar
    inside the last `window` bars. side=-1: break below; +1: break above.
    Returns the crossing bar index, or None (no break, or a stale one)."""
    def s(j, buffer):
        lo = vals[j] * (1.0 - buffer)
        hi = vals[j] * (1.0 + buffer)
        if side < 0:
            if C[j] < lo:
                return -1
            if C[j] > hi:
                return 1
            return 0
        if C[j] > hi:
            return 1
        if C[j] < lo:
            return -1
        return 0
    if s(n - 1, pct) == side:
        for j in range(max(1, n - window), n):
            if s(j, pct) == side and s(j - 1, pct) != side:
                return j
        return None
    if s(n - 1, 0.0) == side and s(n - 2, 0.0) == side:
        for j in range(max(1, n - window), n):
            if s(j, 0.0) == side and s(j - 1, 0.0) != side:
                return j
    return None


def _break_vol_ratio(V, start, end, brk_i):
    """Breakout-bar volume vs the median bar volume inside the pattern
    window. Local median only — 1-min volume has a strong session U-shape,
    so a session average would mislabel most morning breakouts."""
    window = V[start:end + 1]
    med = float(np.median(window)) if len(window) else 0.0
    if med <= 0:
        return 0.0
    return float(V[brk_i]) / med


def _detect_head_shoulders_pattern(df: pd.DataFrame) -> dict | None:
    """Head & Shoulders top / Inverse Head & Shoulders bottom (Murphy
    Ch. 5, pp. 103-112): three peaks/troughs, the middle one the head,
    neckline through the two intervening reaction points. The book's most
    reliable reversal pattern — and the one with the most specific volume
    rules: right-shoulder volume lightest; at bottoms, heavy volume on the
    neckline break is absolutely essential."""
    n = len(df)
    if n < 30:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    pts = _alternating_swings(df)
    if len(pts) < 5:
        return None
    mid = float(np.median(C))
    best = None
    for direction in (1, -1):
        # direction 1: top (H L H L H); -1: bottom (L H L H L).
        want = "HLHLH" if direction == 1 else "LHLHL"
        for e in range(4, len(pts)):
            seq = pts[e - 4:e + 1]
            if "".join(p[1] for p in seq) != want:
                continue
            (i1, _, p1), (i2, _, p2), (i3, _, p3), (i4, _, p4), (i5, _, p5) = seq
            if i5 - i1 < 25 or n - 1 - i5 > 30:
                continue
            if min(i2 - i1, i3 - i2, i4 - i3, i5 - i4) < 5:
                continue
            if i1 < 5:
                continue
            # Prior-trend gate (Murphy preliminary point #1): the pattern
            # must reverse something visible — the left shoulder sits at
            # least 0.40% beyond the extreme that preceded it.
            if direction == 1:
                if p1 < float(L[:i1].min()) * 1.0040:
                    continue
            else:
                if p1 > float(H[:i1].max()) * 0.9960:
                    continue
            neck = _fit_line([i2, i4], [p2, p4])
            if neck is None:
                continue
            head = p3
            neck_head = _line_at(neck, i3)
            height = abs(head - neck_head)
            # Head must be the extreme, and clearly so.
            if direction == 1:
                if not (head > p1 and head > p5):
                    continue
                if head - max(p1, p5) < 0.0015 * mid:
                    continue
                # Shoulders about the same height.
                if abs(p5 - p1) > max(0.0015 * mid, 0.30 * height):
                    continue
            else:
                if not (head < p1 and head < p5):
                    continue
                if min(p1, p5) - head < 0.0015 * mid:
                    continue
                if abs(p5 - p1) > max(0.0015 * mid, 0.30 * height):
                    continue
            # Volume: per-peak mean over the 5 bars ending at each peak.
            def _peak_vol(idx):
                return float(V[max(0, idx - 4):idx + 1].mean())
            v1, v2, v3 = _peak_vol(i1), _peak_vol(i3), _peak_vol(i5)
            score = 40.0
            if direction == 1:
                if v2 < v1:
                    score += 5
                if v3 < v2 and v3 < v1:
                    score += 10
            else:
                # Rally from the head should run on heavier volume than
                # the rally from the left shoulder (Murphy p. 112).
                rally_head = float(V[i3:i4 + 1].mean()) if i4 > i3 else 0.0
                rally_left = float(V[i1:i2 + 1].mean()) if i2 > i1 else 0.0
                if rally_head > rally_left:
                    score += 10
            # Right-shoulder retracement: often 1/2 to 2/3 of the C->D leg.
            if direction == 1:
                retrace = (p5 - p4) / (p3 - p4) if p3 != p4 else 0.0
            else:
                retrace = (p4 - p5) / (p4 - p3) if p4 != p3 else 0.0
            if 0.50 <= retrace <= 0.66:
                score += 5
            # Neckline slope: tops usually slope slightly up, bottoms down.
            if (direction == 1 and neck[0] > 0) or \
                    (direction == -1 and neck[0] < 0):
                score += 5
            score += min(i5 - i1, 60) * 0.25 - (n - 1 - i5)
            vals = neck[0] * np.arange(n) + neck[1]
            neck_last = _line_at(neck, n - 1)
            side = -1 if direction == 1 else 1
            cross = _line_cross(C, vals, n, side)
            name = "Head & Shoulders Top" if direction == 1 \
                else "Inverse Head & Shoulders"
            target = None
            vol_note = ""
            if cross is not None:
                brk_vol = _break_vol_ratio(V, i1, i5, cross)
                if direction == 1:
                    status, bias = "BREAKDOWN CONFIRMED", "bear"
                    trigger, invalidation = vals[cross], head
                    target = vals[cross] - height
                    vol_note = f" · vol ×{brk_vol:.1f} on break"
                    if brk_vol >= 1.5:
                        score += 8
                    score += 25
                    label_x, label_y = cross, vals[cross]
                elif brk_vol < 1.2:
                    # At bottoms heavy volume is absolutely essential.
                    status, bias = "FORMING", "bull"
                    trigger, invalidation = neck_last, head
                    vol_note = (f" · breakout lacks volume "
                                f"(×{brk_vol:.1f})")
                    label_x, label_y = i3, head
                else:
                    status, bias = "BREAKOUT CONFIRMED", "bull"
                    trigger, invalidation = vals[cross], head
                    target = vals[cross] + height
                    vol_note = f" · vol ×{brk_vol:.1f} on break"
                    score += 25
                    label_x, label_y = cross, vals[cross]
            else:
                # Beyond the line but no fresh cross = a stale break, not
                # a forming pattern.
                if direction == 1 and C[-1] < neck_last * (1 - 0.0010):
                    continue
                if direction == -1 and C[-1] > neck_last * (1 + 0.0010):
                    continue
                status = "FORMING"
                bias = "bear" if direction == 1 else "bull"
                trigger, invalidation = neck_last, head
                label_x, label_y = i3, head
            detail = (f"3 peaks · head ${head:,.2f} · neckline "
                      f"${p2:,.2f}→${neck_last:,.2f} · RS retraced "
                      f"{retrace:.0%} · vol "
                      f"{_vol_str(v1)}/{_vol_str(v2)}/{_vol_str(v3)}")
            if target is not None:
                detail += f" · target ${target:,.2f} (min)"
            detail += vol_note
            cand = _make_chart_pattern(
                name, bias, status, i1, n - 1,
                [{"x0": i1, "y0": p1, "x1": i2, "y1": p2},
                 {"x0": i2, "y0": p2, "x1": i3, "y1": p3},
                 {"x0": i3, "y0": p3, "x1": i4, "y1": p4},
                 {"x0": i4, "y0": p4, "x1": i5, "y1": p5},
                 {"x0": i2, "y0": p2, "x1": n - 1, "y1": neck_last}],
                label_x, label_y, trigger, invalidation, detail,
                score, "reversal")
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def _detect_double_top_pattern(df: pd.DataFrame) -> dict | None:
    """Double top / Double bottom (Murphy Ch. 5, pp. 117-125), triple
    variant folded in: two peaks at about the same level with a trough
    between. The pattern completes ONLY when the middle trough breaks on a
    closing basis — the second peak alone is never a pattern (the book's
    'greatly overused' warning)."""
    n = len(df)
    if n < 25:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    pts = _alternating_swings(df)
    if len(pts) < 3:
        return None
    mid = float(np.median(C))
    best = None
    for direction in (1, -1):
        # direction 1: top (H L H); -1: bottom (L H L).
        want = "HLH" if direction == 1 else "LHL"
        for e in range(2, len(pts)):
            seq = pts[e - 2:e + 1]
            if "".join(p[1] for p in seq) != want:
                continue
            (ia, _, pa), (ib, _, pb), (ic, _, pc) = seq
            if min(ib - ia, ic - ib) < 5 or ic - ia < 15:
                continue
            if ia < 5:
                continue
            height = abs(pa - pb)
            if height < 0.0030 * mid:
                continue
            tol = max(0.0015 * mid, 0.20 * height)
            # Peaks about the same level; a second peak slightly above the
            # first is the book's bull-trap case, allowed only if the last
            # close is back below the first peak.
            if direction == 1:
                if pa < float(L[:ia].min()) * 1.0040:
                    continue
                if pc > pa:
                    if pc - pa > 0.0030 * mid or C[-1] >= pa:
                        continue
                elif pa - pc > tol:
                    continue
            else:
                if pa > float(H[:ia].max()) * 0.9960:
                    continue
                if pc < pa:
                    if pa - pc > 0.0030 * mid or C[-1] <= pa:
                        continue
                elif pc - pa > tol:
                    continue
            # Triple variant: a third peak within the same tolerance, with
            # its own intervening trough; completion then needs BOTH
            # troughs broken (Murphy p. 115).
            triple = False
            ib2, ic2, pb2, pc2 = ib, ic, pb, pc
            if e + 2 < len(pts):
                seq2 = pts[e + 1:e + 3]
                if "".join(p[1] for p in seq2) == \
                        ("LH" if direction == 1 else "HL"):
                    (ib2, _, pb2), (ic2, _, pc2) = seq2
                    if min(ib2 - ic, ic2 - ib2) >= 5 \
                            and n - 1 - ic2 <= 30:
                        ok = True
                        if direction == 1:
                            if pc2 > pa:
                                if pc2 - pa > 0.0030 * mid or C[-1] >= pa:
                                    ok = False
                            elif pa - pc2 > tol:
                                ok = False
                        else:
                            if pc2 < pa:
                                if pa - pc2 > 0.0030 * mid or C[-1] <= pa:
                                    ok = False
                            elif pc2 - pa > tol:
                                ok = False
                        triple = ok
            end_idx = ic2 if triple else ic
            # Recency is judged on the pattern's actual end — the third
            # peak for a triple, the second peak for a double.
            if n - 1 - end_idx > 30:
                continue
            if direction == 1:
                peak = max(pa, pc, pc2) if triple else max(pa, pc)
                trig_level = min(pb, pb2) if triple else pb
                invalid = max(pa, pc, pc2) if triple else pc
            else:
                peak = min(pa, pc, pc2) if triple else min(pa, pc)
                trig_level = max(pb, pb2) if triple else pb
                invalid = min(pa, pc, pc2) if triple else pc
            height = abs(peak - trig_level)
            # Volume: lighter on the second peak; heavier on the decline.
            def _rise_vol(idx):
                return float(V[max(0, idx - 4):idx + 1].mean())
            va, vc = _rise_vol(ia), _rise_vol(ic)
            score = 35.0
            if vc < va:
                score += 8
            if n - 1 - ic >= 3:
                if float(V[ic:].mean()) > vc:
                    score += 5
            if triple:
                score += 5
            score += min(end_idx - ia, 60) * 0.25 - (n - 1 - end_idx)
            vals = np.full(n, trig_level)
            side = -1 if direction == 1 else 1
            cross = _line_cross(C, vals, n, side)
            name = ("Triple Top" if triple and direction == 1 else
                    "Triple Bottom" if triple else
                    "Double Top" if direction == 1 else "Double Bottom")
            target = None
            vol_note = ""
            if cross is not None:
                brk_vol = _break_vol_ratio(V, ia, end_idx, cross)
                if direction == 1:
                    status, bias = "BREAKDOWN CONFIRMED", "bear"
                    trigger, invalidation = trig_level, invalid
                    target = trig_level - height
                    vol_note = f" · vol ×{brk_vol:.1f} on break"
                    if brk_vol >= 1.5:
                        score += 8
                    score += 25
                    label_x, label_y = cross, trig_level
                elif brk_vol < 1.2:
                    status, bias = "FORMING", "bull"
                    trigger, invalidation = trig_level, invalid
                    vol_note = (f" · breakout lacks volume "
                                f"(×{brk_vol:.1f})")
                    label_x, label_y = ic, pc
                else:
                    status, bias = "BREAKOUT CONFIRMED", "bull"
                    trigger, invalidation = trig_level, invalid
                    target = trig_level + height
                    vol_note = f" · vol ×{brk_vol:.1f} on break"
                    score += 25
                    label_x, label_y = cross, trig_level
            else:
                if direction == 1 and C[-1] < trig_level * (1 - 0.0010):
                    continue
                if direction == -1 and C[-1] > trig_level * (1 + 0.0010):
                    continue
                status = "FORMING"
                bias = "bear" if direction == 1 else "bull"
                trigger, invalidation = trig_level, invalid
                label_x, label_y = ic, pc
            npeaks = 3 if triple else 2
            detail = (f"{npeaks} peaks · trough ${trig_level:,.2f} · "
                      f"height {height / mid * 100:.2f}% · vol "
                      f"{_vol_str(va)}/{_vol_str(vc)}")
            if target is not None:
                detail += f" · target ${target:,.2f} (min)"
            detail += vol_note
            segments = [{"x0": ia, "y0": pa, "x1": ib, "y1": pb},
                        {"x0": ib, "y0": pb, "x1": ic, "y1": pc}]
            if triple:
                segments += [{"x0": ic, "y0": pc, "x1": ib2, "y1": pb2},
                             {"x0": ib2, "y0": pb2, "x1": ic2, "y1": pc2}]
            segments.append({"x0": ib, "y0": trig_level,
                             "x1": n - 1, "y1": trig_level})
            cand = _make_chart_pattern(
                name, bias, status, ia, n - 1, segments,
                label_x, label_y, trigger, invalidation, detail,
                score, "reversal")
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def _detect_broadening_pattern(df: pd.DataFrame) -> dict | None:
    """Broadening formation / megaphone top (Murphy Ch. 6, pp. 140-141):
    three successively higher peaks and two declining troughs, diverging
    boundary lines, volume expanding with the swings. The only Murphy
    pattern the triangle detector structurally excludes (it requires
    converging lines). No measured target — the book gives none."""
    n = len(df)
    if n < 30:
        return None
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    pts = _alternating_swings(df)
    if len(pts) < 5:
        return None
    mid = float(np.median(C))
    best = None
    for e in range(4, len(pts)):
        seq = pts[e - 4:e + 1]
        if "".join(p[1] for p in seq) != "HLHLH":
            continue
        (ih1, _, h1), (il1, _, l1), (ih2, _, h2), (il2, _, l2), (ih3, _, h3) = seq
        if ih3 - ih1 < 20 or n - 1 - ih3 > 30 or ih1 < 5:
            continue
        if min(il1 - ih1, ih2 - il1, il2 - ih2, ih3 - il2) < 5:
            continue
        if h1 < float(L[:ih1].min()) * 1.0050:
            continue
        if not (h2 > h1 * 1.001 and h3 > h2 * 1.001):
            continue
        if not (l2 < l1 * 0.999):
            continue
        upper = _fit_line([ih1, ih2, ih3], [h1, h2, h3])
        lower = _fit_line([il1, il2], [l1, l2])
        if upper is None or lower is None:
            continue
        if not (upper[0] > 0 and lower[0] < 0):
            continue
        w0 = _line_at(upper, ih1) - _line_at(lower, ih1)
        w1 = _line_at(upper, ih3) - _line_at(lower, ih3)
        if w0 <= 0 or w1 / w0 < 1.15:
            continue
        # Volume must expand with the widening swings (Murphy p. 140) —
        # the opposite of every other triangle, and definitional here.
        window_v = V[ih1:ih3 + 1]
        third = max(1, len(window_v) // 3)
        first_vol = float(window_v[:third].mean())
        last_vol = float(window_v[-third:].mean())
        if first_vol <= 0 or last_vol <= first_vol:
            continue
        vol_expand = last_vol / first_vol
        low_vals = lower[0] * np.arange(n) + lower[1]
        low_last = _line_at(lower, n - 1)
        cross = _line_cross(C, low_vals, n, -1)
        score = 35.0 + min(ih3 - ih1, 60) * 0.25 - (n - 1 - ih3)
        if cross is not None:
            brk_vol = _break_vol_ratio(V, ih1, ih3, cross)
            status, bias = "BREAKDOWN CONFIRMED", "bear"
            trigger, invalidation = low_vals[cross], h3
            vol_note = f" · vol ×{brk_vol:.1f} on break"
            if brk_vol >= 1.5:
                score += 8
            score += 25
            label_x, label_y = cross, low_vals[cross]
        elif C[-1] < low_last * (1 - 0.0010):
            continue
        else:
            status, bias = "FORMING", "bear"
            trigger, invalidation = l2, h3
            vol_note = ""
            label_x, label_y = ih3, h3
        detail = (f"3 higher peaks / 2 lower troughs · width "
                  f"+{(w1 / w0 - 1) * 100:.0f}% · volume expanding "
                  f"×{vol_expand:.1f} · no measured target (Murphy)"
                  f"{vol_note}")
        cand = _make_chart_pattern(
            "Broadening Formation", bias, status, ih1, n - 1,
            [{"x0": ih1, "y0": _line_at(upper, ih1),
              "x1": n - 1, "y1": _line_at(upper, n - 1)},
             {"x0": ih1, "y0": _line_at(lower, ih1),
              "x1": n - 1, "y1": _line_at(lower, n - 1)}],
            label_x, label_y, trigger, invalidation, detail,
            score, "reversal")
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _detect_vcp_pattern(df: pd.DataFrame) -> dict | None:
    """VCP — Volatility Contraction Pattern (Minervini, Ch. 10, pp.
    198-206): successive pullbacks from a session high, each about half
    the previous one ("about half (plus or minus a reasonable amount)",
    p. 199), with volume drying up across the sequence, ending in a
    tight pivot. The book's VCP is 3-65 weeks on daily bars; the
    intraday version is a fractal — the geometry and confirmation logic
    transfer, the durations and percent depths do not, and every
    threshold below that comes from the book's daily scale is marked as
    an adaptation. What makes it NOT a duplicate of the range/flag
    detectors: the range checks one width, the flag one pole + one
    pause; the VCP checks a SEQUENCE of contractions getting tighter
    left-to-right with volume drying at each step. A V-shape (one
    straight down-leg, no intervening rally) is rejected as time
    compression (pp. 211-212), not a VCP."""
    n = len(df)
    if n < 30:
        return None
    C = df["Close"].to_numpy()
    V = df["Volume"].to_numpy()
    L = df["Low"].to_numpy()
    pts = _alternating_swings(df)
    if len(pts) < 5:
        return None
    med_vol = float(np.median(V)) if n else 0.0
    # Anchor: the highest swing high with a >= 0.50% up-leg from the
    # session low before it (adaptation: the book requires a prior
    # advance of >= 25% over 3-36 months, p. 243 — a single session
    # cannot show that; the 0.50% intraday leg is the fractal stand-in).
    # Try swing highs in descending price order; the first that yields a
    # valid contraction sequence wins.
    highs = sorted((p for p in pts if p[1] == "H"),
                   key=lambda p: -p[2])
    for (h0_i, _, h0_v) in highs:
        lo_before = float(L[:h0_i + 1].min())
        if (h0_v - lo_before) / h0_v < 0.0050:
            continue
        rest = [p for p in pts if p[0] > h0_i]
        if len(rest) < 4 or rest[0][1] != "L":
            continue
        # Contractions: (high bar, high val, low bar, low val) — the
        # first from H0 to L1, then from each rally high to the next low.
        contr = []
        cur_h = (h0_i, h0_v)
        j = 0
        while j < len(rest) and len(contr) < 6:
            li, _, lv = rest[j]
            if rest[j][1] != "L":
                break
            contr.append((cur_h[0], cur_h[1], li, lv))
            if j + 1 >= len(rest) or rest[j + 1][1] != "H":
                break
            cur_h = (rest[j + 1][0], rest[j + 1][2])
            j += 2
        if len(contr) < 2:
            continue
        depths = [(hv - lv) / hv for (_, hv, _, lv) in contr]
        # Each contraction ~half the previous one (loose bound 0.65x).
        if any(depths[i + 1] > 0.65 * depths[i]
               for i in range(len(depths) - 1)):
            continue
        # Last contraction tight: <= 0.30% of price (adaptation of the
        # book's 2-3% final pullbacks on daily bars, scaled down roughly
        # an order of magnitude for a single session).
        if depths[-1] > 0.0030:
            continue
        # No V-shape: every rally between contractions must retrace at
        # least 0.10% off its low.
        rallies = []
        for i in range(len(contr) - 1):
            _, _, li, lv = contr[i]
            hj = contr[i + 1][0]
            rallies.append((C[hj] - lv) / lv if lv > 0 else 0.0)
        if any(r < 0.0010 for r in rallies):
            continue
        # Volume drying across the sequence: last contraction's mean
        # volume <= 0.70x the first contraction's, and at least one bar
        # in the last contraction below the session-median bar volume
        # (adaptation of "final-contraction volume below the 50-day
        # average, with one or two days extremely low", p. 226).
        def _cvol(a, b):
            return float(V[a:b + 1].mean()) if b >= a else 0.0
        cvols = [_cvol(hb, lb) for (hb, _, lb, _) in contr]
        if cvols[0] <= 0 or cvols[-1] > 0.70 * cvols[0]:
            continue
        last_lo_bar = contr[-1][2]
        last_hi_bar = contr[-1][0]
        if not any(v < med_vol for v in V[last_hi_bar:last_lo_bar + 1]):
            continue
        # Pivot = the last contraction's high; it must be recent (same
        # 30-bar recency convention as the H&S / double-top detectors).
        pivot = contr[-1][1]
        pivot_i = contr[-1][0]
        if n - 1 - pivot_i > 30:
            continue
        last_low = contr[-1][3]
        start = h0_i
        end = last_lo_bar
        footprint = (f"{len(contr)}T · "
                     + " → ".join(f"{d * 100:.1f}%" for d in depths)
                     + " · vol drying")
        score = 40.0
        if len(contr) >= 3:
            score += 10
        if cvols[-1] == min(cvols):
            score += 5
        score += min(end - start, 60) * 0.25 - (n - 1 - pivot_i)
        vals = [pivot] * n
        low_vals = [last_low] * n
        cross_up = _line_cross(C, vals, n, 1)
        cross_dn = _line_cross(C, low_vals, n, -1)
        if cross_up is not None:
            brk_vol = _break_vol_ratio(V, start, end, cross_up)
            if brk_vol < 1.2:
                status, bias = "FORMING", "bull"
                trigger, invalidation = pivot, last_low
                vol_note = " · breakout lacks volume"
                label_x, label_y = pivot_i, pivot
            else:
                status, bias = "BREAKOUT CONFIRMED", "bull"
                trigger, invalidation = pivot, last_low
                vol_note = f" · vol ×{brk_vol:.1f} on break"
                score += 25
                label_x, label_y = cross_up, vals[cross_up]
        elif cross_dn is not None:
            # A VCP that breaks down is a failed base, not a short
            # signal — Minervini is long-only in stage 2.
            status, bias = "BREAKDOWN CONFIRMED", "bear"
            trigger, invalidation = last_low, pivot
            vol_note = " · base failed — VCP broken"
            score += 25
            label_x, label_y = cross_dn, low_vals[cross_dn]
        else:
            status, bias = "FORMING", "bull"
            trigger, invalidation = pivot, last_low
            vol_note = ""
            label_x, label_y = pivot_i, pivot
        detail = (f"{footprint} · no measured target (Minervini)"
                  f"{vol_note}")
        seq_pts = [(h0_i, h0_v)]
        for k, (hb, hv, lb, lv) in enumerate(contr):
            seq_pts.append((lb, lv))
            if k + 1 < len(contr):
                seq_pts.append((contr[k + 1][0], contr[k + 1][1]))
        segs = []
        for k in range(len(seq_pts) - 1):
            segs.append({"x0": seq_pts[k][0], "y0": seq_pts[k][1],
                         "x1": seq_pts[k + 1][0],
                         "y1": seq_pts[k + 1][1]})
        return _make_chart_pattern(
            "VCP — Volatility Contraction", bias, status, start, end,
            segs, label_x, label_y, trigger, invalidation, detail,
            score, "continuation")
    return None


def detect_chart_patterns(df: pd.DataFrame) -> list[dict]:
    """Current chart-pattern recognition for the displayed session. Returns
    at most three non-duplicative patterns, confirmed breaks first."""
    found = []
    for detector in (_detect_range_pattern, _detect_triangle_pattern,
                     _detect_flag_pattern, _detect_staircase_pattern,
                     _detect_head_shoulders_pattern,
                     _detect_double_top_pattern,
                     _detect_broadening_pattern,
                     _detect_vcp_pattern):
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

    def _dup_group(fam: str) -> str:
        # A VCP is a continuation base; when it sits on the same bars as
        # a range/flag/triangle it is the same observation (Minervini: a
        # flat base without contraction is still a valid base, p. 199) —
        # so the VCP dedups against the consolidation family, not only
        # against itself.
        return "consolidation" if fam == "continuation" else fam

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
            # An H&S and a double top built from the same swings are the
            # same observation (Murphy: H&S vs triple-top is "academic",
            # p. 115) — keep the higher score, as for consolidations.
            if _dup_group(pattern["family"]) \
                    == _dup_group(other["family"]) \
                    and pattern["family"] in (
                        "consolidation", "reversal", "continuation") \
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
# The indicator hierarchy, weighted. Bullish and bearish evidence are
# scored tier by tier and the larger total is the lean — a flat count of
# checks let a stochastic cross outvote price structure, which is
# backwards: everything below Tier 1 is derived from price. Weights:
# location & structure 30, intraday trend 25, volume 15, pattern &
# candle at a level 10, stochastic trigger 10, session context 10.
TIER_WEIGHTS = [("Location & structure", 30), ("Intraday trend", 25),
                ("Volume", 15), ("Pattern & candle at a level", 10),
                ("Stochastic trigger", 10), ("Session context", 10)]


def weighted_confluence(df: pd.DataFrame, zones: list[dict],
                        levels: list[float], pa: dict | None,
                        orng: dict | None, chart_pats: list[dict],
                        pcr_label: str | None) -> dict | None:
    """Weighted confluence at the last bar. Display only — no verdict.

    Returns None until there are enough bars, else a dict with the lean
    direction, its 0-100 score, both raw totals, the Tier-1-vs-Tier-2
    conflict flag, one display row per tier and any filter notes."""
    if len(df) < 25:
        return None
    last = df.iloc[-1]
    price = float(last["Close"])
    bull = [0] * 6
    bear = [0] * 6
    bd: list[list[str]] = [[] for _ in range(6)]
    rd: list[list[str]] = [[] for _ in range(6)]

    # Tier 1 — location & structure (30): where price sits vs the levels
    if pa:
        if price > pa["PDH"]:
            bull[0] += 8
            bd[0].append("above PDH")
        elif price < pa["PDL"]:
            bear[0] += 8
            rd[0].append("below PDL")
        if price > pa["PDC"]:
            bull[0] += 8
            bd[0].append("above prior close")
        elif price < pa["PDC"]:
            bear[0] += 8
            rd[0].append("below prior close")
    if orng:
        if price > orng["high"]:
            bull[0] += 6
            bd[0].append("above the opening range")
        elif price < orng["low"]:
            bear[0] += 6
            rd[0].append("below the opening range")
    zone_done = False
    for z in zones:
        if z["low"] <= price <= z["high"]:
            if z["hi_idx"] and z["lo_idx"] \
                    and min(z["hi_idx"]) < max(z["lo_idx"]):
                bull[0] += 8
                bd[0].append("inside a flipped-resistance BUYING ZONE")
            elif z["hi_idx"] and z["lo_idx"] \
                    and min(z["lo_idx"]) < max(z["hi_idx"]):
                bear[0] += 8
                rd[0].append("inside a flipped-support SELLING ZONE")
            elif z["center"] <= price:
                bull[0] += 8
                bd[0].append("inside a BUYING ZONE")
            else:
                bear[0] += 8
                rd[0].append("inside a SELLING ZONE")
            zone_done = True
            break
    if not zone_done:
        below = [z for z in zones if z["center"] < price]
        above = [z for z in zones if z["center"] >= price]
        if below and price - below[-1]["high"] <= 0.003 * price:
            bull[0] += 8
            bd[0].append("resting on a BUYING ZONE")
        elif above and above[0]["low"] - price <= 0.003 * price:
            bear[0] += 8
            rd[0].append("pressed under a SELLING ZONE")

    # Tier 2 — intraday trend (25): EMA 9/21 alignment + VWAP side
    e9, e21, vwap = last["EMA9"], last["EMA21"], last["VWAP"]
    if pd.notna(e9) and pd.notna(e21):
        if e9 > e21:
            bull[1] += 15
            bd[1].append("EMA 9 above 21")
        elif e9 < e21:
            bear[1] += 15
            rd[1].append("EMA 9 below 21")
    if pd.notna(vwap):
        if price > vwap:
            bull[1] += 10
            bd[1].append("above VWAP")
        elif price < vwap:
            bear[1] += 10
            rd[1].append("below VWAP")

    # Tier 3 — volume (15): graded by relative volume, credited to the
    # direction of the last-5-bar move it is funding
    vol = df["Volume"]
    recent = float(vol.iloc[-5:].mean())
    base = float(vol.iloc[-25:-5].mean())
    if base > 0:
        rv = recent / base
        pts = 15 if rv >= 1.5 else 12 if rv >= 1.3 else \
            9 if rv >= 1.15 else 6 if rv >= 1.0 else 2
        move = price - float(df["Close"].iloc[-6])
        if move > 0:
            bull[2] += pts
            bd[2].append(f"{rv:.2f}x volume behind the last-5-bar rise")
        elif move < 0:
            bear[2] += pts
            rd[2].append(f"{rv:.2f}x volume behind the last-5-bar fall")

    # Tier 4 — pattern & candle at a level (10)
    for pat in chart_pats:
        if pat["bias"] in ("bull", "bear"):
            pts = 7 if "CONFIRMED" in pat["status"] else \
                5 if pat["status"] == "ACTIVE" else 4
            if pat["bias"] == "bull":
                bull[3] += pts
                bd[3].append(f"{pat['name']} — {pat['status']}")
            else:
                bear[3] += pts
                rd[3].append(f"{pat['name']} — {pat['status']}")
            break
    pats = candle_patterns(df)
    for i in range(len(df) - 3, len(df)):
        if i in pats and pats[i][1] in ("bull", "bear"):
            b = df.iloc[i]
            if at_level(float(b["Low"]), float(b["High"]), levels):
                if pats[i][1] == "bull":
                    bull[3] += 3
                    bd[3].append(f"{pats[i][0]} at a level")
                else:
                    bear[3] += 3
                    rd[3].append(f"{pats[i][0]} at a level")
                break
    bull[3] = min(bull[3], 10)
    bear[3] = min(bear[3], 10)

    # Tier 5 — stochastic trigger (10): a fresh cross, else %K/%D side
    k, d = df["StochK"], df["StochD"]
    scross = None
    for i in range(max(1, len(df) - 5), len(df)):
        if pd.notna(k.iloc[i - 1]) and pd.notna(k.iloc[i]) \
                and pd.notna(d.iloc[i - 1]) and pd.notna(d.iloc[i]):
            if k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i]:
                scross = "up"
            elif k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i]:
                scross = "down"
    if scross == "up":
        bull[4] += 10
        bd[4].append("%K crossed up in the last 5 bars")
    elif scross == "down":
        bear[4] += 10
        rd[4].append("%K crossed down in the last 5 bars")
    elif pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
        if k.iloc[-1] > d.iloc[-1]:
            bull[4] += 5
            bd[4].append("%K above %D")
        elif k.iloc[-1] < d.iloc[-1]:
            bear[4] += 5
            rd[4].append("%K below %D")

    # Tier 6 — session context (10): PCR zone (the user's bands; the
    # oversold/overbought extremes score as reversal evidence) + a fresh
    # 9/21 EMA cross in the last 10 bars
    if pcr_label:
        if pcr_label.startswith("Bullish"):
            bull[5] += 6
            bd[5].append("PCR in the bullish zone")
        elif pcr_label.startswith("Oversold"):
            bull[5] += 6
            bd[5].append("PCR oversold — reversal evidence")
        elif pcr_label.startswith("Bearish"):
            bear[5] += 6
            rd[5].append("PCR in the bearish zone")
        elif pcr_label.startswith("Overbought"):
            bear[5] += 6
            rd[5].append("PCR overbought — reversal evidence")
    ecross = None
    e9s, e21s = df["EMA9"], df["EMA21"]
    for i in range(max(1, len(df) - 10), len(df)):
        if pd.notna(e9s.iloc[i - 1]) and pd.notna(e9s.iloc[i]):
            if e9s.iloc[i - 1] <= e21s.iloc[i - 1] \
                    and e9s.iloc[i] > e21s.iloc[i]:
                ecross = "up"
            elif e9s.iloc[i - 1] >= e21s.iloc[i - 1] \
                    and e9s.iloc[i] < e21s.iloc[i]:
                ecross = "down"
    if ecross == "up":
        bull[5] += 4
        bd[5].append("fresh 9/21 cross up (last 10 bars)")
    elif ecross == "down":
        bear[5] += 4
        rd[5].append("fresh 9/21 cross down (last 10 bars)")

    bt, rt = sum(bull), sum(bear)
    direction = "Bullish" if bt > rt + 5 else \
        "Bearish" if rt > bt + 5 else "Mixed"
    score = round(100 * max(bt, rt) / (100 if pcr_label else 94))
    conflict = (bull[0] - bear[0]) * (bull[1] - bear[1]) < 0 \
        and abs(bull[0] - bear[0]) >= 8 and abs(bull[1] - bear[1]) >= 10
    rows = []
    for t, (name, w) in enumerate(TIER_WEIGHTS):
        if direction == "Bullish":
            pts, det = bull[t], bd[t]
        elif direction == "Bearish":
            pts, det = bear[t], rd[t]
        else:
            pts, det = max(bull[t], bear[t]), (bd[t] or rd[t])
        rows.append((f"Tier {t + 1} · {name} ({w}%)", pts, w,
                     "; ".join(det) if det else "no evidence either way"))
    notes = []
    if conflict:
        notes.append("⚠ Tier 1 (structure) and Tier 2 (trend) disagree — "
                     "by the hierarchy that is a no-directional-call "
                     "setup, whatever the total says.")
    kk = float(last["StochK"]) if pd.notna(last["StochK"]) else None
    if direction == "Bullish" and kk is not None and kk >= 80:
        notes.append("⚠ Filter check: %K ≥ 80 — momentum is stretched. "
                     "Extremes can pin in a strong trend, but chasing "
                     "here carries reversal risk.")
    if direction == "Bearish" and kk is not None and kk <= 20:
        notes.append("⚠ Filter check: %K ≤ 20 — momentum is stretched "
                     "to the downside; pressing shorts here carries "
                     "squeeze risk.")
    if not pcr_label:
        notes.append("PCR unavailable for this symbol — Tier 6 scored "
                     "out of 4 and the total re-based to 100.")
    return {"direction": direction, "score": score, "bull": bt,
            "bear": rt, "conflict": conflict, "rows": rows,
            "notes": notes}


# ---------------------------------------------------------------- Brooks — SPY & QQQ tab
BROOKS_VIEW = "SPY & QQQ — Brooks (5-min)"
BROOKS_SYMBOLS = ("SPY", "QQQ")
BROOKS_BAR_MIN = 5
BROOKS_ADR_DAYS = 20  # sessions for the average daily range


def _brooks_median_range(df: pd.DataFrame) -> float:
    r = (df["High"] - df["Low"]).to_numpy(dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    return float(np.median(r)) if len(r) else 0.0


def _brooks_spike(d5: pd.DataFrame,
                  pdc: float | None) -> dict | None:
    """Opening spike (Brooks Ch. 21): a run of >= 3 same-direction trend
    bars (body >= 40% of the bar, each close extending beyond the prior
    close) starting in the first hour, or a single exceptionally large
    first bar, or a gap open (>= 0.4%) plus a strong first bar in the
    gap direction. A spike is a breakout, a climax and a gap at once —
    it is where a trend begins. Returns None when no such run exists."""
    if d5 is None or len(d5) < 3:
        return None
    med = _brooks_median_range(d5)
    if med <= 0:
        return None
    o = d5["Open"].to_numpy(dtype=float)
    h = d5["High"].to_numpy(dtype=float)
    l = d5["Low"].to_numpy(dtype=float)
    c = d5["Close"].to_numpy(dtype=float)
    n = len(d5)

    def _run(start: int, direction: int) -> int:
        run = 0
        for i in range(start, n):
            body = abs(c[i] - o[i])
            rng = h[i] - l[i]
            if i == start:
                extends = True if start == 0 else (
                    c[i] > c[i - 1] if direction > 0
                    else c[i] < c[i - 1])
            elif direction > 0:
                extends = c[i] > c[i - 1]
            else:
                extends = c[i] < c[i - 1]
            if body >= 0.4 * rng and extends:
                run += 1
            else:
                break
        return run

    gap_pct = (float(o[0]) / pdc - 1) * 100 if pdc else 0.0

    def _height(start: int, direction: int, run: int):
        """Spike height: from the run's open — or from the prior close
        when the spike starts on a gap open (Brooks: a gap opening is
        itself a spike, Ch. 21)."""
        end = start + run - 1
        gap_up = start == 0 and abs(gap_pct) >= 0.4 and gap_pct > 0
        gap_dn = start == 0 and abs(gap_pct) >= 0.4 and gap_pct < 0
        if direction > 0:
            extreme = float(h[start:end + 1].max())
            base = pdc if gap_up else float(o[start])
            return extreme, extreme - base
        extreme = float(l[start:end + 1].min())
        base = pdc if gap_dn else float(o[start])
        return extreme, base - extreme

    best = None
    for start in range(0, min(12, n - 2)):
        for direction in (1, -1):
            run = _run(start, direction)
            if run < 3:
                continue
            _, height = _height(start, direction, run)
            # A spike must MOVE — a slow drift of trend bars is not one
            # (Brooks: a spike is steep). 0.30% of price is the floor.
            if height < 0.003 * float(o[start]):
                continue
            if best is None or start < best[0]:
                best = (start, direction, run)
    if best is None:
        # A single exceptionally large first bar (or a gap + strong bar)
        # also counts as a spike (Brooks: a spike can be one trend bar).
        if (h[0] - l[0]) >= 2.5 * med:
            direction = 1 if c[0] > o[0] else -1 if c[0] < o[0] else 0
            if direction:
                _, height = _height(0, direction, 1)
                if height >= 0.003 * float(o[0]):
                    best = (0, direction, 1)
        if best is None and abs(gap_pct) >= 0.4:
            direction = 1 if gap_pct > 0 else -1
            first_strong = (abs(c[0] - o[0]) >= 0.4 * (h[0] - l[0])
                            and ((c[0] > o[0]) == (direction > 0)))
            if first_strong:
                _, height = _height(0, direction, 1)
                if height >= 0.003 * float(o[0]):
                    best = (0, direction, 1)
    if best is None:
        return None
    start, direction, run = best
    end = start + run - 1
    extreme, height = _height(start, direction, run)
    return {"direction": direction, "start": int(start), "bars": int(run),
            "end": int(end), "extreme": extreme, "height": float(height),
            "gap_pct": float(gap_pct)}


def _brooks_legs(d5: pd.DataFrame, direction: int) -> tuple[int, str]:
    """Pullback legs since the session trend extreme began, counted from
    the alternating swing sequence (w=3) — Brooks' High 1 / High 2 /
    Low 1 / Low 2 (Ch. 10, 18). Returns (legs, setup label). The count
    is the number of completed pullback legs; a break above the prior
    bar's high after leg N is a High N setup."""
    if d5 is None or len(d5) < 5:
        return 0, "no swings yet"
    swings = _alternating_swings(d5)
    if not swings:
        return 0, "no swings yet"
    c = d5["Close"].to_numpy(dtype=float)
    n = len(d5)
    prev_high = float(d5["High"].iloc[-2]) if n > 1 \
        else float(d5["High"].iloc[-1])
    prev_low = float(d5["Low"].iloc[-2]) if n > 1 \
        else float(d5["Low"].iloc[-1])
    if direction > 0:
        hs = [s for s in swings if s[1] == "H"]
        if not hs:
            return 0, "no pullback yet — spike phase"
        # Each swing low after the first swing high ends one pullback
        # leg (the first high is where the trend's first leg ends).
        first_h = hs[0][0]
        legs = sum(1 for s in swings if s[1] == "L" and s[0] > first_h)
        if legs == 0:
            return 0, "no pullback yet — spike phase"
        if c[-1] > prev_high:
            tag = f"High {min(legs, 3)}"
            if legs >= 3:
                tag += " (wedge flag)"
            return legs, f"{tag} — pullback break, buy setup"
        if swings[-1][1] == "L":
            done = max(legs - 1, 0)
            return legs, (f"in pullback — leg {done + 1} forming, "
                          f"watch for High {min(done + 1, 3)}")
        return legs, f"{legs} pullback leg(s) done — no break yet"
    ls = [s for s in swings if s[1] == "L"]
    if not ls:
        return 0, "no pullback yet — spike phase"
    first_l = ls[0][0]
    legs = sum(1 for s in swings if s[1] == "H" and s[0] > first_l)
    if legs == 0:
        return 0, "no pullback yet — spike phase"
    if c[-1] < prev_low:
        tag = f"Low {min(legs, 3)}"
        if legs >= 3:
            tag += " (wedge flag)"
        return legs, f"{tag} — pullback break, sell setup"
    if swings[-1][1] == "H":
        done = max(legs - 1, 0)
        return legs, (f"in pullback — leg {done + 1} forming, "
                      f"watch for Low {min(done + 1, 3)}")
    return legs, f"{legs} pullback leg(s) done — no break yet"


def _brooks_counter_spikes(d5: pd.DataFrame, direction: int) -> tuple[int, int]:
    """Countertrend spikes (a bar whose BODY is >= 1.5x the median body,
    in the opposite direction) and how many of them failed — the next
    bar closed back in the trend direction (Ch. 19: reversal spikes
    with no follow-through become flags). Body, not bar range: a spike
    is a big-bodied bar."""
    if d5 is None or len(d5) < 3 or not direction:
        return 0, 0
    o = d5["Open"].to_numpy(dtype=float)
    c = d5["Close"].to_numpy(dtype=float)
    bodies = np.abs(c - o)
    med_body = float(np.median(bodies)) if len(bodies) else 0.0
    if med_body <= 0:
        return 0, 0
    total = failed = 0
    for i in range(len(d5) - 1):
        if bodies[i] < 1.5 * med_body:
            continue
        counter = (direction > 0 and c[i] < o[i]) or \
            (direction < 0 and c[i] > o[i])
        if not counter:
            continue
        total += 1
        back = (direction > 0 and c[i + 1] > c[i]) or \
            (direction < 0 and c[i + 1] < c[i])
        if back:
            failed += 1
    return total, failed


def brooks_day_read(d5: pd.DataFrame, pa: dict | None,
                    daily: pd.DataFrame | None, orng: dict | None,
                    sel_date, lines: list[dict] | None = None) -> dict:
    """The Brooks read for one symbol's 5-min session (Al Brooks,
    Trading Price Action Trends): day type, always-in direction,
    spike/channel phase, High/Low leg count, measured moves, and the
    Ch. 19 strength checklist. Display only — no score, no signal.
    d5 is the session slice at 5-min bars with EMA9/21, VWAP and an
    EMA20 column (computed on the warmed multi-day frame, so the
    moving average is live from the open)."""
    out = {"day_type": "Too early — not enough bars", "day_evidence": "",
           "always_in": 0, "always_label": "Unclear", "phase": "range",
           "spike": None, "legs": 0, "leg_label": "no swings yet",
           "measured_moves": [], "strength": [],
           "or_ratio": None, "adr": None, "gap_pct": 0.0}
    if d5 is None or len(d5) < 10:
        return out
    n = len(d5)
    med = _brooks_median_range(d5)
    pdc = pa["PDC"] if pa else None
    o = d5["Open"].to_numpy(dtype=float)
    h = d5["High"].to_numpy(dtype=float)
    l = d5["Low"].to_numpy(dtype=float)
    c = d5["Close"].to_numpy(dtype=float)

    # Average daily range from the prior sessions (the book's yardstick
    # for opening-range size and pullback size).
    adr = None
    if daily is not None and not daily.empty:
        try:
            prior = daily[[d < sel_date for d in daily.index]]
            if len(prior) >= 10:
                adr = float((prior["High"] - prior["Low"]).tail(
                    BROOKS_ADR_DAYS).mean())
        except Exception:
            adr = None
    out["adr"] = adr
    or_ratio = None
    if orng and adr:
        or_ratio = (orng["high"] - orng["low"]) / adr
    out["or_ratio"] = or_ratio
    gap_pct = (float(o[0]) / pdc - 1) * 100 if pdc else 0.0
    out["gap_pct"] = gap_pct

    spike = _brooks_spike(d5, pdc)
    out["spike"] = spike

    # Always-in: the direction of the most recent 5-min signal — price
    # vs VWAP, EMA 9 vs 21, trending swings, and the last close above
    # the prior bar's high (bull) or below its low (bear). 3 of 4 with
    # no conflict, else unclear.
    last = d5.iloc[-1]
    bull = bear = 0
    if pd.notna(last.get("VWAP")):
        if float(last["Close"]) > float(last["VWAP"]):
            bull += 1
        else:
            bear += 1
    if pd.notna(last.get("EMA9")) and pd.notna(last.get("EMA21")):
        # A hair's-width gap is noise, not a signal — the pair needs
        # to be 0.01% of price apart before it counts as a vote.
        gap = float(last["EMA9"]) - float(last["EMA21"])
        if gap > 0.0001 * float(last["Close"]):
            bull += 1
        elif gap < -0.0001 * float(last["Close"]):
            bear += 1
    swings = _alternating_swings(d5)
    hs = [s for s in swings if s[1] == "H"]
    ls = [s for s in swings if s[1] == "L"]
    # Swing trend: the last two swing highs (or lows) trending — either
    # pair counts, but only if the other pair does not contradict it
    # (a range day has rising lows and falling highs at once).
    h_up = len(hs) >= 2 and hs[-1][2] > hs[-2][2]
    h_dn = len(hs) >= 2 and hs[-1][2] < hs[-2][2]
    l_up = len(ls) >= 2 and ls[-1][2] > ls[-2][2]
    l_dn = len(ls) >= 2 and ls[-1][2] < ls[-2][2]
    if (h_up or l_up) and not (h_dn or l_dn):
        bull += 1
    elif (h_dn or l_dn) and not (h_up or l_up):
        bear += 1
    if n > 1:
        if float(last["Close"]) > float(d5["High"].iloc[-2]):
            bull += 1
        elif float(last["Close"]) < float(d5["Low"].iloc[-2]):
            bear += 1
    if bull >= 3 and bull > bear:
        always = 1
    elif bear >= 3 and bear > bull:
        always = -1
    else:
        always = 0
    out["always_in"] = always
    out["always_label"] = "Long" if always > 0 else \
        "Short" if always < 0 else "Unclear"

    # Phase: spike, channel, or range.
    if lines is None:
        lines = trendlines(d5)
    if spike and n <= spike["end"] + 2:
        phase = "spike"
    elif spike:
        pulled_back = float(l[spike["end"]:].min()) <= spike["extreme"] - 0.5 * med \
            if spike["direction"] > 0 else \
            float(h[spike["end"]:].max()) >= spike["extreme"] + 0.5 * med
        phase = "channel" if (pulled_back or lines) else "spike"
    elif always != 0 and lines:
        phase = "channel"
    else:
        phase = "range"
    out["phase"] = phase

    # Pullback legs (High 1 / High 2 / Low 1 / Low 2).
    leg_dir = always if always != 0 else (
        spike["direction"] if spike else 0)
    if leg_dir:
        legs, leg_label = _brooks_legs(d5, leg_dir)
    else:
        legs, leg_label = 0, "no trend — no leg count"
    out["legs"] = legs
    out["leg_label"] = leg_label

    # Largest pullback from the running extreme, as a share of ADR.
    max_pb = None
    if adr:
        if always >= 0:
            run_max = np.maximum.accumulate(h)
            max_pb = float(np.max(run_max - l)) / adr
        else:
            run_min = np.minimum.accumulate(l)
            max_pb = float(np.max(h - run_min)) / adr

    # Morning direction (first-hour close vs open) — the reversal test
    # compares the always-in now against the morning.
    morning_dir = 0
    if n > 11:
        if float(c[11]) > float(o[0]):
            morning_dir = 1
        elif float(c[11]) < float(o[0]):
            morning_dir = -1

    # Opening-range breakout bar and the second range after it.
    brk_i = None
    brk_dir = 0
    if orng:
        for i in range(n):
            if float(c[i]) > orng["high"]:
                brk_i, brk_dir = i, 1
                break
            if float(c[i]) < orng["low"]:
                brk_i, brk_dir = i, -1
                break
    second_range = False
    if brk_i is not None and adr:
        for j in range(brk_i + 1, n - 4):
            seg = d5.iloc[j:j + 5]
            if float(seg["High"].max() - seg["Low"].min()) < 0.5 * adr:
                second_range = True
                break

    # Midday tight range (>= 12 bars, range < 30% of ADR) + a late
    # breakout in the morning direction — the trend-resumption shape.
    # The first hour must have been a real trend (>= 50% of the avg
    # day range), not a drift, and the late breakout must clear the
    # band by >= 10% of ADR — otherwise any range day with a tight
    # stretch and a higher close would qualify.
    tight = None
    first_hour_move = None
    if adr and n > 11:
        first_hour_move = abs(float(c[11]) - float(o[0])) / adr
        for j in range(12, n - 11):
            seg = d5.iloc[j:j + 12]
            if float(seg["High"].max() - seg["Low"].min()) < 0.30 * adr:
                tight = (j, j + 11)
                break
    late_brk = False
    if tight and morning_dir and first_hour_move is not None \
            and first_hour_move >= 0.50:
        t0, t1 = tight
        after = d5.iloc[t1 + 1:]
        band_hi = float(d5.iloc[t0:t1 + 1]["High"].max())
        band_lo = float(d5.iloc[t0:t1 + 1]["Low"].min())
        if morning_dir > 0 and len(after) \
                and float(after["Close"].max()) > band_hi + 0.10 * adr:
            late_brk = True
        elif morning_dir < 0 and len(after) \
                and float(after["Close"].min()) < band_lo - 0.10 * adr:
            late_brk = True

    # Stairs: >= 3 swings each way with >= 2 overlaps — a pullback
    # retracing beyond the prior breakout point. The swings must NET
    # trend in the always-in direction and span at least half the avg
    # day range: a range day's oscillations overlap by construction
    # but do not trend, and a narrow channel is not stairs.
    stairs = False
    if len(hs) >= 3 and len(ls) >= 3 and always != 0 and adr:
        if always > 0:
            trending = hs[-1][2] > hs[0][2] and ls[-1][2] > ls[0][2]
        else:
            trending = hs[-1][2] < hs[0][2] and ls[-1][2] < ls[0][2]
        span = (max(s[2] for s in hs) - min(s[2] for s in ls)) / adr
        if trending and span >= 0.50:
            overlaps = 0
            for k in range(1, len(hs)):
                hi_next = hs[k + 1][0] if k + 1 < len(hs) else n
                following = [s for s in ls if hs[k][0] < s[0] < hi_next]
                if following and min(s[2] for s in following) < hs[k][2]:
                    overlaps += 1
            for k in range(1, len(ls)):
                lo_next = ls[k + 1][0] if k + 1 < len(ls) else n
                following = [s for s in hs if ls[k][0] < s[0] < lo_next]
                if following and max(s[2] for s in following) > ls[k][2]:
                    overlaps += 1
            stairs = overlaps >= 2

    # Day type, in the book's order of recognition.
    day_type = None
    evidence = ""
    if always != 0 and morning_dir != 0 and always != morning_dir:
        # A pullback that begins with a strong countertrend spike can
        # grow into a trend channel the other way (Ch. 24). A spike
        # here is a bar with a body >= 2x the median body.
        med_body = float(np.median(np.abs(c - o))) if n else 0.0
        counter = False
        for i in range(12, n):
            if med_body <= 0 or abs(c[i] - o[i]) < 2.0 * med_body:
                continue
            if always > 0 and c[i] > o[i]:
                counter = True
                break
            if always < 0 and c[i] < o[i]:
                counter = True
                break
        if counter:
            day_type = "Reversal day"
            evidence = ("morning trend flipped — always-in is now "
                        f"{out['always_label'].lower()} after a strong "
                        "countertrend spike (Ch. 24)")
    if day_type is None and spike and always != 0:
        # The day's extreme formed in the first few bars (Ch. 23).
        if always > 0:
            extreme_early = float(l[:5].min()) == float(l.min())
        else:
            extreme_early = float(h[:5].max()) == float(h.max())
        if extreme_early and max_pb is not None and max_pb < 0.30:
            day_type = "Trend from the open" + (
                " — small pullback" if max_pb < 0.20 else "")
            evidence = (f"extreme formed in the first 5 bars, largest "
                        f"pullback {max_pb:.0%} of the avg day range "
                        "(Ch. 23)")
    if day_type is None and spike and phase == "channel":
        day_type = "Spike and channel"
        evidence = (f"spike of {spike['bars']} bars "
                    f"(${spike['height']:.2f}) then a channel phase "
                    "(Ch. 21)")
    if day_type is None and or_ratio is not None \
            and 0.30 <= or_ratio <= 0.55 and brk_i is not None \
            and second_range:
        day_type = "Trending trading range"
        evidence = (f"opening range {or_ratio:.0%} of the avg day range, "
                    "broke out, then a second range formed (Ch. 22)")
    if day_type is None and morning_dir != 0 and tight and late_brk:
        day_type = "Trend resumption"
        evidence = ("strong first hour, tight midday range, then the "
                    "trend resumed late in the day (Ch. 25)")
    if day_type is None and stairs:
        day_type = "Stairs (broad channel)"
        evidence = ("3+ trending swings with overlapping pullbacks "
                    "(Ch. 26)")
    if day_type is None and always == 0 and not spike:
        day_type = "Trading range day"
        evidence = ("no trend-line break, no trending swings, no spike "
                    "— trade both directions (Part IV intro)")
    if day_type is None and always != 0:
        day_type = "Trend — type forming"
        evidence = (f"always-in {out['always_label'].lower()}, day type "
                    "not yet decided by the opening-range / spike rules")
    if day_type is None:
        day_type = "Range — forming"
        evidence = "no trend yet — watching the first hour"
    out["day_type"] = day_type
    out["day_evidence"] = evidence

    # Measured moves — profit targets, not reversal spots (Ch. 21).
    mms = []
    if spike:
        if spike["direction"] > 0:
            pb_low = float(l[spike["end"]:].min())
            mms.append(("Spike measured move",
                        pb_low + spike["height"]))
        else:
            pb_high = float(h[spike["end"]:].max())
            mms.append(("Spike measured move",
                        pb_high - spike["height"]))
    res = next((x for x in lines if x["kind"] == "resistance"), None)
    sup = next((x for x in lines if x["kind"] == "support"), None)
    if res and sup:
        height = abs(float(res["y1"]) - float(sup["y1"]))
        if always >= 0:
            mms.append(("Channel measured move",
                        float(sup["y1"]) + height))
        else:
            mms.append(("Channel measured move",
                        float(res["y1"]) - height))
    if orng and brk_dir != 0:
        or_h = orng["high"] - orng["low"]
        if brk_dir > 0:
            mms.append(("Range measured move", orng["high"] + or_h))
        else:
            mms.append(("Range measured move", orng["low"] - or_h))
    out["measured_moves"] = mms

    # Strength checklist (Ch. 19) — only where a trend direction exists.
    checks = []
    trend_dir = always if always != 0 else (
        spike["direction"] if spike else 0)
    if trend_dir:
        checks.append((
            f"Gap open {gap_pct:+.1f}% in the trend direction",
            abs(gap_pct) >= 0.3 and (gap_pct > 0) == (trend_dir > 0)))
        bodies = np.abs(c - o)
        ranges = h - l
        trend_bars = ((bodies >= 0.5 * np.where(ranges > 0, ranges, 1))
                      & (((c > o) & (trend_dir > 0))
                         | ((c < o) & (trend_dir < 0)))).mean()
        checks.append((
            f"Most bars trend bars in the trend direction "
            f"({trend_bars:.0%})", trend_bars >= 0.5))
        overlaps = []
        for i in range(1, n):
            lo1, hi1 = min(o[i - 1], c[i - 1]), max(o[i - 1], c[i - 1])
            lo2, hi2 = min(o[i], c[i]), max(o[i], c[i])
            inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
            smaller = min(hi1 - lo1, hi2 - lo2)
            if smaller > 0:
                overlaps.append(inter / smaller)
        if overlaps:
            checks.append((
                "Little body overlap between consecutive bars "
                f"(median {float(np.median(overlaps)):.0%})",
                float(np.median(overlaps)) < 0.30))
        if "EMA20" in d5.columns:
            e20 = d5["EMA20"].to_numpy(dtype=float)
            touches = [i for i in range(n)
                       if l[i] <= e20[i] <= h[i]
                       and np.isfinite(e20[i])]
            last_touch = touches[-1] if touches else -1
            ma_gap = n - 1 - last_touch
            checks.append((
                f"{ma_gap} bars since the last touch of the 20-bar MA",
                ma_gap >= 20))
            if n >= 2 and np.isfinite(e20[-1]) \
                    and np.isfinite(e20[-2]):
                wrong = sum(
                    1 for j in (-1, -2)
                    if (trend_dir > 0 and c[j] < e20[j])
                    or (trend_dir < 0 and c[j] > e20[j]))
                checks.append((
                    "No 2 consecutive closes on the wrong side of "
                    "the MA", wrong < 2))
        if max_pb is not None:
            checks.append((
                f"Largest pullback {max_pb:.0%} of the avg day range",
                max_pb < 0.30))
        n_cs, n_failed = _brooks_counter_spikes(d5, trend_dir)
        if n_cs:
            checks.append((
                f"Countertrend spikes failed ({n_failed}/{n_cs})",
                n_failed == n_cs))
    out["strength"] = checks
    return out


def _brooks_symbol_data(sym: str):
    """1-min bars -> warmed 5-min frame for one Brooks symbol, plus
    prior-day levels, opening range and daily bars. Returns
    (data dict, error string)."""
    try:
        df1 = fetch_bars_1m(
            sym, _bucket(live, int(refresh_secs)),
            end=view_date.isoformat() if past_mode else "")
    except Exception as e:
        return None, f"Could not load bars for {sym}: {e}"
    if df1.empty or len(df1) < 60:
        return None, f"Not enough intraday data for {sym} yet."
    dates = sorted(set(df1.index.date))
    if past_mode and view_date not in dates:
        return None, (f"No trading session for {sym} on {view_date} — "
                      f"the market was closed that day, or the symbol "
                      f"did not trade.")
    sel = view_date if past_mode else dates[-1]
    full5 = add_indicators(resample(df1, BROOKS_BAR_MIN))
    full5["EMA20"] = ema(full5["Close"], 20)
    disp = full5[full5.index.normalize() == pd.Timestamp(sel, tz=ET)]
    if len(disp) < 10:
        return None, (f"Only the first few bars of the {sym} session "
                      "so far — the read builds as bars arrive.")
    pa = prior_day_levels(df1)
    today_df = df1[df1.index.date == sel]
    orng = opening_range(today_df)
    daily = None
    try:
        daily = fetch_daily_bars(
            sym, _bucket(live, int(refresh_secs), bucket_secs=86400),
            end=view_date.isoformat() if past_mode else "")
    except Exception:
        daily = None
    return {"symbol": sym, "sel_date": sel, "disp": disp, "pa": pa,
            "orng": orng, "daily": daily}, None


def render_brooks():
    """SPY & QQQ side by side on 5-minute bars, each with the full
    chart (zones, trendlines, patterns, volume profile) and the Brooks
    read underneath: day type, always-in, spike/channel phase, High/Low
    leg count, measured moves, strength checklist. No PCR, no weighted
    confluence, no Trend Template here — those live on the
    single-symbol tab."""
    show_b = dict(show)
    show_b["daily_ema"] = False
    cols = st.columns(2)
    for col, sym in zip(cols, BROOKS_SYMBOLS):
        with col:
            data, err = _brooks_symbol_data(sym)
            if err:
                st.info(err)
                continue
            disp = data["disp"]
            pa = data["pa"]
            orng = data["orng"]
            zones = sr_zones(disp)
            lines = trendlines(disp)
            vp = session_volume_profile(disp)
            chart_pats = detect_chart_patterns(disp)
            pats = candle_patterns(disp)
            level_vals = ([pa["PDH"], pa["PDC"], pa["PDL"]] if pa else []) \
                + [z["center"] for z in zones] \
                + ([orng["high"], orng["low"]] if orng else [])
            read = brooks_day_read(disp, pa, data["daily"], orng,
                                   data["sel_date"], lines=lines)
            last = disp.iloc[-1]
            prev_close = pa["PDC"] if pa else float(disp["Close"].iloc[0])
            chg = (float(last["Close"]) / prev_close - 1) * 100 \
                if prev_close else 0.0
            st.metric(sym, f"${float(last['Close']):,.2f}",
                      f"{chg:+.2f}% vs prior close")
            st.plotly_chart(
                build_chart(disp, sym, zones, lines, pats, level_vals,
                            pa, orng, show_b, window_start=None,
                            chart_patterns=chart_pats, vp=vp,
                            daily_ema=None),
                use_container_width=True, key=f"brooks_{sym}_5min")
            st.markdown(f"**{sym} — Brooks read: {read['day_type']}**")
            rows = [
                f"- **Always-in:** {read['always_label']}",
                f"- **Phase:** {read['phase']}"
                + (f" — spike {read['spike']['bars']} bars, "
                   f"${read['spike']['height']:.2f} from the open"
                   if read["spike"] else ""),
                f"- **Pullback count:** {read['leg_label']}",
            ]
            if read["measured_moves"]:
                rows.append("- **Measured moves:** " + " · ".join(
                    f"{label} ${px:,.2f}"
                    for label, px in read["measured_moves"]))
            if read["or_ratio"] is not None:
                rows.append(
                    f"- **Opening range:** {read['or_ratio']:.0%} of the "
                    f"{BROOKS_ADR_DAYS}-day avg day range "
                    f"(${read['adr']:,.2f})")
            st.markdown("\n".join(rows))
            if read["strength"]:
                met = sum(1 for _, m in read["strength"] if m)
                slines = [f"**Strength checklist (Ch. 19) — "
                          f"{met}/{len(read['strength'])}**"]
                for label, m in read["strength"]:
                    slines.append(f"- {'✓' if m else '✗'} {label}")
                st.markdown("\n".join(slines))
            st.caption(
                f"{read['day_evidence']} — Brooks, Trading Price "
                "Action Trends, on 5-min bars. The book's day types "
                "are read off one session; its reliability claims do "
                "not transfer to a single day. Display only — no "
                "score, no signal.")
    st.caption(
        f"Alpaca {str(_feed).upper()} · 5-min bars, full session · "
        f"SPY + QQQ · updated {datetime.now(ET).strftime('%H:%M:%S')} "
        f"ET · build {BUILD_TAG} · {BUILD_HASH}")


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
                chart_patterns: list[dict] | None = None,
                vp: dict | None = None,
                daily_ema: dict | None = None) -> go.Figure:
    """3 panels: price (candles, EMA 9/21/50/200, VWAP, daily EMA 50/200
    reference lines, zones, trendlines, PDH/PDC/PDL, opening range,
    pattern markers, time-cycle phases, session volume profile), volume,
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
    if show.get("ema_long"):
        fig.add_trace(go.Scatter(x=x, y=df["EMA50"], name="EMA 50",
                                 line=dict(width=1.2, color="#db2777")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["EMA200"], name="EMA 200",
                                 line=dict(width=1.8, color="#1f2937")),
                      row=1, col=1)
    # Daily EMA 50/200 reference lines: flat lines from daily closes
    # (completed sessions only) — the daily clock, not the bar-size
    # clock. Same colours as the intraday EMA 50/200 so the colour
    # language matches; flat + "(daily)" tells them apart. Two-point
    # traces, so they appear in the legend with their values.
    if show.get("daily_ema") and daily_ema:
        for key, color, width in (("ema50", "#db2777", 2.2),
                                  ("ema200", "#1f2937", 2.2)):
            val = daily_ema.get(key)
            if val is None:
                continue
            fig.add_trace(go.Scatter(
                x=[x[0], x[-1]], y=[val, val],
                name=f"EMA {key[3:]} (daily) ${val:,.2f}",
                mode="lines", line=dict(width=width, color=color)),
                row=1, col=1)
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
            # Minervini pivot line (Ch. 10, p. 223): the trigger is the
            # "call to action" level, drawn dashed from the pattern
            # start to the right edge for FORMING patterns as well as
            # confirmed ones — "let the stock break above the pivot and
            # prove itself" (p. 229). A shape, not a trace, so it never
            # stretches the price axis; a far pivot stays drawn but
            # off-view, like every other level.
            if pat.get("trigger") is not None:
                fig.add_shape(
                    type="line", xref="x", yref="y",
                    x0=x[pat["start"]], x1=x[-1],
                    y0=pat["trigger"], y1=pat["trigger"],
                    line=dict(color=color, width=1.2, dash="dash"),
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
        # Daily EMA lines are levels too — they join `cands` and follow
        # the same one-span rule. (The first version extended the axis
        # to any drawn daily line, which on a narrow-range day stretched
        # the panel to 10x the session range and squashed the candles
        # into a sliver — the Nov 12 AAPL screenshot. Never again.)
        if show.get("daily_ema") and daily_ema:
            for key in ("ema50", "ema200"):
                v = daily_ema.get(key)
                if v is not None:
                    cands.append(v)
        for v in cands:
            if base_lo - span <= v <= base_hi + span:
                ylo = min(ylo, v - pad * 0.5)
                yhi = max(yhi, v + pad * 0.5)
        fig.update_yaxes(range=[ylo, yhi], row=1, col=1)

    # Session volume profile: horizontal histogram on the price panel,
    # drawn on a secondary x-axis (x4) that overlays the right-hand
    # fifth of the panel, so it never collides with the candles. The
    # POC bin is amber; POC is the amber line, VAH/VAL bracket the 70%
    # value area. x4 is used on purpose, NOT x2: in this 3-row figure
    # with shared x-axes, x2 is already the Volume panel's axis and x3
    # the Stochastic panel's — redefining x2 (as the first version did)
    # hijacked the shared-x range with a volume scale and pushed the
    # candles, EMAs, VWAP and volume bars off-screen. x4 is a fresh
    # overlay axis with no other role.
    # This block runs AFTER the window-range update above on purpose:
    # update_xaxes(range=...) applies to every x-axis present, and x4's
    # range is a volume scale, not a time scale.
    if show.get("volume_profile") and vp is not None:
        vols = vp["volumes"]
        m = float(vols.max())
        if m > 0:
            colors = ["rgba(245, 158, 11, 0.55)"
                      if i == vp["poc_i"] else "rgba(37, 99, 235, 0.30)"
                      for i in range(len(vols))]
            fig.add_trace(go.Bar(
                y=vp["centers"], x=vols, orientation="h",
                width=vp["bin_width"] * 0.9, marker_color=colors,
                name="Volume profile", showlegend=True,
                hovertemplate="Price %{y:,.2f}<br>Volume %{x:,.0f}"
                              "<extra></extra>"),
                row=1, col=1)
            fig.data[-1].update(xaxis="x4")
            fig.update_layout(
                xaxis4=dict(overlaying="x", side="top",
                            range=[-4 * m, m], showgrid=False,
                            showticklabels=False, zeroline=False,
                            fixedrange=True))
            fig.add_hline(y=vp["poc"], line_dash="solid",
                          line_color="#f59e0b", line_width=1.6,
                          row=1, col=1,
                          annotation_text=f"POC {vp['poc']:,.2f}",
                          annotation_position="top right")
            fig.add_hline(y=vp["vah"], line_dash="dot",
                          line_color="#9ca3af", line_width=1.2,
                          row=1, col=1,
                          annotation_text=f"VAH {vp['vah']:,.2f}",
                          annotation_position="top left")
            fig.add_hline(y=vp["val"], line_dash="dot",
                          line_color="#9ca3af", line_width=1.2,
                          row=1, col=1,
                          annotation_text=f"VAL {vp['val']:,.2f}",
                          annotation_position="bottom left")
    return fig


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Price-Action Charting", layout="wide")
st.title("📐 Price-Action Charting")
st.caption("The charting techniques from the *Price Action Trading* book — "
           "zones, trendlines, chart-pattern recognition, formations at "
           "levels, EMA 9/21/50/200, session volume profile, stochastic — "
           "drawn on any US stock. "
           "Display only: no verdicts, no signals.")

_key, _secret, _feed = _secrets()
if not _key or not _secret:
    st.error("Alpaca keys are not set for this app. Add APCA_API_KEY_ID and "
             "APCA_API_SECRET_KEY (same values as the stock-analyzer) under "
             "Settings → Secrets, then reboot the app.")
    st.stop()

with st.sidebar:
    view = st.radio(
        "View",
        ["Single symbol", BROOKS_VIEW],
        index=0,
        help="Single symbol is the full charting app for one ticker. "
             "SPY & QQQ — Brooks shows both indexes side by side on "
             "5-minute bars with the Brooks day read on each.")
    st.header("Chart")
    symbol = st.text_input("Stock symbol", value="AAPL").strip().upper() or "AAPL"
    _today_et = datetime.now(ET).date()
    view_date = st.date_input(
        "Session date", value=_today_et, max_value=_today_et,
        min_value=_today_et - timedelta(days=365),
        help="Pick a past trading day to view that session's chart — "
             "zones, patterns, formations and the weighted confluence "
             "are computed for that session, with its own prior-day "
             "levels. PCR is live-only and shows as unavailable for "
             "past dates. Pick today to return to the live chart.")
    past_mode = bool(view_date) and view_date < _today_et
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
            "pennants, staircases, head & shoulders, double tops/bottoms, "
            "broadening formations, volatility contraction (VCP))",
            value=True),
        "patterns": st.checkbox("Candlestick formations", value=True),
        "phases": st.checkbox("Time-cycle phases", value=True),
        "ema_long": st.checkbox("EMA 50 / 200 (trend context)", value=True),
        "daily_ema": st.checkbox("Daily EMA 50 / 200 reference lines",
                                 value=True),
        "trend_template": st.checkbox(
            "Trend Template (Minervini) — daily SMA 50/150/200, "
            "52-week high/low, stage label", value=True),
        "volume_profile": st.checkbox(
            "Volume profile (POC / value area)", value=True),
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
        df1 = fetch_bars_1m(
            symbol, _bucket(live, int(refresh_secs)),
            end=view_date.isoformat() if past_mode else "")
    except Exception as e:
        st.error(f"Could not load bars for {symbol}: {e}")
        return
    if df1.empty or len(df1) < 60:
        st.info(f"Not enough intraday data for {symbol} yet.")
        return
    dates = sorted(set(df1.index.date))
    if past_mode and view_date not in dates:
        st.info(f"No trading session for {symbol} on {view_date} — the "
                f"market was closed that day, or the symbol did not "
                f"trade. Sessions in the loaded window: "
                f"{dates[0]} → {dates[-1]}. Pick one of those dates.")
        return
    sel_date = view_date if past_mode else dates[-1]
    df = df1[df1.index.normalize() == pd.Timestamp(sel_date, tz=ET)]
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
                     == pd.Timestamp(sel_date, tz=ET)]
    if len(disp) < 10:
        st.info("Only the first few bars of the session so far — the chart "
                "builds as bars arrive.")
        return

    pa = prior_day_levels(df1)
    today_df = df1[df1.index.date == sel_date]
    orng = opening_range(today_df)
    zones = sr_zones(disp)
    lines = trendlines(disp)
    vp = session_volume_profile(disp)
    chart_pats = detect_chart_patterns(disp)

    # Minervini daily context (Trend Template + volume pace): one daily
    # OHLCV pull on a slow daily bucket, shared by both. A failed or
    # too-short pull just hides the Template and the pace, never the
    # chart — the intraday detectors do not depend on it.
    daily_bars = None
    template = None
    avg_daily_vol = None
    if show.get("trend_template") or show.get("chart_patterns"):
        try:
            daily_bars = fetch_daily_bars(
                symbol,
                _bucket(live, int(refresh_secs), bucket_secs=86400),
                end=view_date.isoformat() if past_mode else "")
        except Exception:
            daily_bars = None
    if daily_bars is not None and not daily_bars.empty:
        if show.get("trend_template"):
            try:
                template = trend_template(daily_bars, sel_date)
            except Exception:
                template = None
        try:
            prior_vol = daily_bars["Volume"][
                [d < sel_date for d in daily_bars.index]]
            if len(prior_vol) >= 20:
                avg_daily_vol = float(prior_vol.mean())
        except Exception:
            avg_daily_vol = None
    vol_pace = (_volume_pace(disp, avg_daily_vol)
                if chart_pats else None)
    pats = candle_patterns(disp)
    level_vals = ([pa["PDH"], pa["PDC"], pa["PDL"]] if pa else []) \
        + [z["center"] for z in zones] \
        + ([orng["high"], orng["low"]] if orng else [])

    # The PCR chain is fetched before the metrics: its zone feeds Tier 6
    # of the weighted confluence, so the header score and the checklist
    # under the chart are one computation. (Slow 300s bucket — the chain
    # pull takes a few seconds and PCR does not need the fast cycle.)
    p = None if past_mode else fetch_pcr(
        symbol, _bucket(live, int(refresh_secs), bucket_secs=300))

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
    wconf = weighted_confluence(
        disp, zones, level_vals, pa, orng, chart_pats,
        pcr_zone(p["pcr"])[0] if p is not None else None)
    c4.metric("Weighted confluence",
              f"{wconf['score']} / 100" if wconf else "—",
              wconf["direction"] if wconf else None, delta_color="off")

    # Daily EMA 50/200 reference lines — daily closes through the last
    # completed session (no lookahead). Fetched only when the toggle is
    # on, on a slow daily bucket (the values change once a day); a failed
    # or too-short pull just skips the lines, never the chart. A line
    # more than DAILY_EMA_MAX_DIST from the last price is not drawn at
    # all (too far to be a useful reference); a drawn line extends the
    # price axis only when it sits within one session-range of the
    # action, like every other level — farther lines stay off-view until
    # price approaches them, and the caption names their values.
    daily_levels = None
    daily_ema = None
    daily_ema_skipped: list[tuple[str, float]] = []
    if show.get("daily_ema"):
        try:
            closes = fetch_daily_closes(
                symbol,
                _bucket(live, int(refresh_secs), bucket_secs=86400),
                end=view_date.isoformat() if past_mode else "")
            daily_levels = daily_ema_levels(closes, sel_date)
        except Exception:
            daily_levels = None
        if daily_levels:
            daily_ema = dict(daily_levels)
            last_px = float(last["Close"])
            for key in ("ema50", "ema200"):
                val = daily_ema.get(key)
                if (val is not None
                        and abs(val - last_px) / last_px
                        > DAILY_EMA_MAX_DIST):
                    daily_ema_skipped.append((key, val))
                    daily_ema[key] = None
            if (daily_ema.get("ema50") is None
                    and daily_ema.get("ema200") is None):
                daily_ema = None

    trailing = CHART_TF_TRAILING[chart_tf]
    window_start = (disp.index[-1] - pd.Timedelta(minutes=trailing)
                    if trailing is not None else None)
    st.plotly_chart(build_chart(disp, symbol, zones, lines, pats, level_vals,
                                pa, orng, show, window_start=window_start,
                                chart_patterns=chart_pats, vp=vp,
                                daily_ema=daily_ema),
                    use_container_width=True,
                    key=f"pa_{symbol}_{tf_label}_{chart_tf.replace(' ', '_')}")

    if show.get("volume_profile"):
        if vp is not None:
            st.caption(
                f"Volume profile — session {sel_date}, {tf_label} bars, "
                f"{vp['n_bins']} price bins: POC "
                f"${vp['poc']:,.2f} (most volume traded here) · value area "
                f"${vp['val']:,.2f} – ${vp['vah']:,.2f} "
                f"({vp['va_volume'] / vp['total_volume']:.0%} of "
                f"{vp['total_volume']:,.0f} shares). Price holding above "
                f"the POC with the value area below it is acceptance "
                f"higher; a close back below the POC after holding above "
                f"it is the profile's rejection read. Display only — not "
                f"scored in the weighted confluence.")
        else:
            st.caption("Volume profile unavailable for this session — no "
                       "usable price range or volume.")

    if show.get("daily_ema"):
        if daily_ema:
            last_px = float(last["Close"])
            parts = []
            for key, label in (("ema50", "Daily EMA 50"),
                               ("ema200", "Daily EMA 200")):
                val = daily_ema.get(key)
                if val is not None:
                    pct = (val - last_px) / last_px * 100
                    rel = (f"{abs(pct):.1f}% below price" if pct < -0.05
                           else f"{abs(pct):.1f}% above price" if pct > 0.05
                           else "at price")
                    parts.append(f"{label} ${val:,.2f} ({rel})")
            skipped = ""
            if daily_ema_skipped:
                skipped = " " + "; ".join(
                    f"EMA {k[3:]} (daily) ${v:,.2f} is more than "
                    f"{DAILY_EMA_MAX_DIST:.0%} from price — line hidden"
                    for k, v in daily_ema_skipped) + "."
            st.caption(
                " · ".join(parts) + " — flat reference lines from daily "
                f"closes through {daily_ema['asof']} "
                f"({daily_ema['n']} sessions). The EMA 50 / 200 lines on "
                f"the chart are the last 50 / 200 minutes on {tf_label} "
                f"bars; these are the daily clock.{skipped} Display only "
                f"— not scored in the weighted confluence.")
        elif daily_levels is not None:
            # Levels loaded, but every line was hidden by the distance guard.
            st.caption(
                "Daily EMA 50 / 200 reference lines hidden — "
                + "; ".join(
                    f"EMA {k[3:]} (daily) ${v:,.2f}"
                    for k, v in daily_ema_skipped)
                + f" are more than {DAILY_EMA_MAX_DIST:.0%} from price.")
        else:
            st.caption("Daily EMA 50 / 200 reference lines unavailable — "
                       "daily bars could not be loaded for this session.")

    if show.get("trend_template"):
        if template:
            lines = [f"**Trend Template (Minervini)** — "
                     f"{template['met']}/7 met · {template['stage']}"]
            for c in template["criteria"]:
                mark = ("✓" if c["met"] is True
                        else "✗" if c["met"] is False else "—")
                lines.append(f"- {mark} {c['label']}")
            st.markdown("\n".join(lines))
            st.caption("Minervini's Trend Template (Ch. 5, p. 79) on "
                       "daily bars through "
                       f"{template['asof']} ({template['n']} sessions). "
                       "Simple moving averages per the book — not the "
                       "app's daily EMA lines. RS ranking is IBD data "
                       "and is not available here. Context only — it "
                       "does not change the weighted confluence score.")
        else:
            st.caption("Trend Template unavailable — need 252 daily "
                       "sessions; daily bars could not be loaded for "
                       "this session.")

    if show.get("chart_patterns"):
        st.markdown("**Pattern recognition** (the book's chart patterns — "
                    "ranges, triangles, wedges, flags/pennants, "
                    "staircases, head & shoulders, double tops/bottoms, "
                    "broadening formations, volatility contraction (VCP)):")
        if chart_pats:
            icons = {"bull": "🟢", "bear": "🔴", "neutral": "🟣"}
            rows = []
            for pat in chart_pats:
                # Minervini pivot entry (Ch. 10, p. 223): the trigger IS
                # the pivot — the "call to action" level — and the
                # invalidation IS the stop. The book's 7-8% daily stop
                # does not transfer intraday (7% of QQQ is ~$46);
                # the pattern's own invalidation level is the
                # intraday equivalent of the stop.
                trigger = (f"Pivot ${pat['trigger']:,.2f}"
                           if pat["trigger"] is not None else "Pivot —")
                invalid = (f"Stop ${pat['invalidation']:,.2f} "
                           f"(invalidation)"
                           if pat["invalidation"] is not None
                           else "Stop —")
                extra = ""
                if vol_pace is not None:
                    extra += f" · vol pace ×{vol_pace:.1f} of avg day"
                if (pat["status"] == "BREAKOUT CONFIRMED"
                        and pat["trigger"] is not None):
                    run = (float(last["Close"]) / pat["trigger"] - 1) * 100
                    if run > 0.50:
                        extra += (f" · chased: price is {run:.1f}% above "
                                  f"the pivot — wait for a pullback")
                rows.append(
                    f"- {icons.get(pat['bias'], '🟣')} **{pat['name']} — "
                    f"{pat['status']}** · {trigger} · {invalid} · "
                    f"{pat['detail']}{extra}")
            st.markdown("\n".join(rows))
            if template and template["stage"].startswith("Stage 4"):
                st.caption("⚠️ These patterns fired in a Stage 4 "
                           "(declining) daily context — Minervini would "
                           "not buy a base here (p. 190). The patterns "
                           "are shown, not suppressed.")
        else:
            st.caption("No qualifying chart pattern in the current session "
                       "yet. The detector requires a repeated range, "
                       "converging boundaries, a fierce pole followed by a "
                       "contained pause, or a clean stepping sequence — it "
                       "does not force a pattern onto ordinary noise.")

    # Put/Call Ratio strip (user's bands) — chain fetched above.
    if past_mode:
        st.caption(f"Put/Call Ratio is live-only — not available for "
                   f"the {sel_date} session. The weighted confluence "
                   f"above is re-based without it.")
    elif p is not None:
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
    if wconf:
        st.markdown("**Weighted confluence** (at the last bar — tiers "
                    "in priority order, higher tiers overrule lower "
                    "ones; both directions scored, the larger total "
                    "is the lean):")
        lines = []
        for label, pts, w, det in wconf["rows"]:
            icon = "✅" if pts >= 0.6 * w else "🟡" if pts > 0 else "⬜"
            lines.append(f"- {icon} **{label}** — {pts}/{w} pts · {det}")
        lines.append(f"- **Lean: {wconf['direction']} "
                     f"{wconf['score']}/100** (bull {wconf['bull']} · "
                     f"bear {wconf['bear']})")
        st.markdown("\n".join(lines))
        for note in wconf["notes"]:
            st.caption(note)
    st.caption(
        f"Alpaca {str(_feed).upper()} · {len(df1):,} 1-min bars loaded "
        f"({dates[0]} → {dates[-1]}) · session {sel_date} · "
        f"showing {len(disp):,} {tf_label} bars "
        f"· window: {chart_tf} · "
        f"updated {datetime.now(ET).strftime('%H:%M:%S')} ET · "
        f"build {BUILD_TAG} · {BUILD_HASH}")


# The dispatch runs one view: the Brooks tab or the single-symbol chart.
_show = render_brooks if view == BROOKS_VIEW else render
if past_mode:
    _show()
    st.caption(f"Viewing the {view_date} session — live tracking is "
               f"paused in date view. Set the Session date back to "
               f"today for the live chart.")
elif live and in_market_hours():
    st.fragment(run_every=int(refresh_secs))(_show)()
elif live:
    _show()
    st.caption("Live tracking is on, but the market is closed — showing the "
               "last session without auto-refresh.")
else:
    _show()
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
        "- **Minervini: Trend Template, VCP, pivot** — the Template panel "
        "under the chart checks the stock against the book's eight "
        "criteria (SMA 50/150/200, 200-day slope, 52-week high/low) on "
        "daily bars and names the life-cycle stage (2 = advancing, 4 = "
        "declining); a range in a Stage 4 context is shown, not "
        "suppressed, with a warning. A **VCP** is a sequence of "
        "successive pullbacks each about half the previous one with "
        "volume drying up — the pivot is the last contraction's high. "
        "Every pattern's trigger is labeled **Pivot** and its "
        "invalidation **Stop**: buy at the pivot, don't chase more than "
        "a few points past it (the list flags a chased break), and the "
        "volume pace tells you how the day's volume is running versus "
        "the average day. The book's patterns take weeks on daily "
        "charts; the intraday VCP is a miniature of the geometry, not "
        "of the book's reliability.\n"
        "- **Brooks tab (SPY & QQQ)** — the 5-minute read: the day type "
        "from the opening range (under 25% of the average day range is "
        "breakout mode; a third to a half is a trending trading range "
        "day), the always-in direction, the spike/channel phase, the "
        "High 1 / High 2 pullback count, and measured moves as profit "
        "targets. In a trend about 80% of reversal attempts fail and "
        "become flags — take the with-trend entries; in a range about "
        "80% of breakouts fail — fade the extremes.\n"
        "- **Triangles / diamonds on the candles** — candlestick formations. "
        "Labeled ones formed **at a level** (that's when the book says they "
        "count); faint unlabeled ones formed mid-chart and are shown for "
        "reference only.\n"
        "- **Amber dashed lines (PDH / PDC / PDL)** — yesterday's high, "
        "close, low: hard levels into today's open (the book's AHPMB idea).\n"
        "- **Shaded bands** — time-cycle phases of the session: the opening "
        "drive (first hour), the midday lull, and the final hour, when moves "
        "tend to start or exhaust.\n"
        "- **Volume profile (right side of the price panel)** — the "
        "session's volume by price level, in 24 bins at the chosen bar "
        "size, computed on the full session like the zones. The amber "
        "line is the POC (point of control — where the most volume "
        "traded, the session's fair-value anchor); the dotted lines are "
        "VAH/VAL, the 70% value area holding most of the day's volume. "
        "Price holding above the POC with the value area below it is "
        "acceptance higher; a close back below the POC after holding "
        "above it is the profile's rejection read. Display only — it is "
        "not scored in the weighted confluence. Toggle it off in the "
        "sidebar (Show → Volume profile) if the chart feels crowded.\n"
        "- **EMA lines** — 9 (amber) and 21 (blue) are the book's "
        "crossover pair; 50 (pink) and 200 (dark) are trend context at "
        "the chosen bar size, warmed up on the loaded prior sessions "
        "like the stochastic — on 1-minute bars the 200 spans roughly "
        "the last 3⅓ hours, on 5-minute bars roughly the last 2½ "
        "sessions. Price holding above both is the wind at a long's "
        "back; below both, at a short's. Toggle them off in the sidebar "
        "(Show → EMA 50 / 200) if the chart feels crowded.\n"
        "- **Stochastic (bottom)** — momentum shifts: blue %K crossing orange "
        "%D is the event; crossing up from under 20 or down from over 80 is "
        "the one that matters, especially right at a level.\n"
        "- **Put/Call Ratio strip** — options sentiment for the symbol "
        "(by traded volume; the open-interest PCR is shown alongside). "
        "Under 0.7 = oversold, reversal watch · 0.7–0.9 = bearish zone · "
        "0.9–1.1 = sideways · 1.1–1.6 = bullish zone · over 1.6 = overbought, "
        "reversal watch. Volume PCR builds during the session, so the first "
        "hour's reading rests on thin volume.\n"
        "- **Weighted confluence** — the score and checklist under the "
        "chart weight the indicator hierarchy instead of counting checks "
        "flat: location & structure 30%, intraday trend (EMA 9/21 + VWAP) "
        "25%, volume 15%, pattern & candle at a level 10%, stochastic "
        "trigger 10%, session context (PCR zone + fresh EMA cross) 10%. "
        "Bullish and bearish evidence are scored separately and the "
        "bigger total is the lean; when Tier 1 and Tier 2 disagree the "
        "checklist flags it — by the hierarchy, that's a "
        "no-directional-call setup.")
