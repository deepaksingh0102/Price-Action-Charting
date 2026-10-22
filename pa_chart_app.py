"""Price-Action Charting app — the techniques from the "Price Action Trading"
book (@FORRNEERO), drawn on the chart for any US stock:

  * Support / resistance ZONES — price clusters touched >= 2-3 times
    (the book: a level only counts after repeated tests; broken resistance
    flips to support and vice versa)
  * TRENDLINES from swing highs / lows — drawn only when tested >= 2 times;
    the upper line is the book's "selling area", the lower the buying area
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

ET = ZoneInfo("America/New_York")
ALPACA_BASE = "https://data.alpaca.markets/v2"
LOAD_DAYS = 7            # calendar days of 1-min bars (covers ~5 sessions)
ZONE_TOL = 0.0015        # 0.15% — prices within this band are "the same level"
LEVEL_TOL = 0.0020       # 0.20% — a bar is "at" a level within this band


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


@st.cache_data(ttl=60, show_spinner=False)
def fetch_bars_1m(symbol: str, days: int = LOAD_DAYS) -> pd.DataFrame:
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
    touches (swing count), has_highs / has_lows (both => a flipped zone).
    Only clusters with >= 2 touches qualify — the book's rule."""
    highs, lows = swing_points(df)
    pts = [(float(df["High"].iloc[i]), "H") for i in highs]
    pts += [(float(df["Low"].iloc[i]), "L") for i in lows]
    if not pts:
        return []
    pts.sort()
    zones, cur = [], [pts[0]]
    for p, kind in pts[1:]:
        if abs(p - cur[-1][0]) / max(cur[-1][0], 1e-9) <= tol:
            cur.append((p, kind))
        else:
            zones.append(cur)
            cur = [(p, kind)]
    zones.append(cur)
    out = []
    for cl in zones:
        if len(cl) < 2:
            continue
        prices = [p for p, _ in cl]
        out.append({
            "center": float(np.median(prices)),
            "low": float(min(prices)), "high": float(max(prices)),
            "touches": len(cl),
            "has_highs": any(k == "H" for _, k in cl),
            "has_lows": any(k == "L" for _, k in cl),
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


# ---------------------------------------------------------------- chart
PHASES = [(570, 630, "Opening drive", "rgba(37, 99, 235, 0.05)"),
          (750, 780, "Midday", "rgba(107, 114, 128, 0.05)"),
          (900, 960, "Final hour", "rgba(249, 115, 22, 0.06)")]


def build_chart(df: pd.DataFrame, symbol: str, zones: list[dict],
                lines: list[dict], pats: dict[int, tuple[str, str]],
                levels: list[float], pa: dict | None, orng: dict | None,
                show: dict) -> go.Figure:
    """3 panels: price (candles, EMA 9/21, VWAP, zones, trendlines, PDH/PDC/
    PDL, opening range, pattern markers, time-cycle phases), volume,
    stochastic."""
    price = float(df["Close"].iloc[-1])
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.12, 0.26], vertical_spacing=0.05,
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
    if show.get("zones"):
        above = sorted([z for z in zones if z["center"] >= price],
                       key=lambda z: z["center"])[:3]
        below = sorted([z for z in zones if z["center"] < price],
                       key=lambda z: -z["center"])[:3]
        for z in above + below:
            role = "Resistance" if z["center"] >= price else "Support"
            flipped = z["has_highs"] and z["has_lows"]
            fig.add_hrect(
                y0=z["low"], y1=z["high"], row=1, col=1,
                fillcolor=("rgba(220, 38, 38, 0.10)" if role == "Resistance"
                           else "rgba(22, 163, 74, 0.10)"),
                line=dict(color=("rgba(220, 38, 38, 0.45)" if role == "Resistance"
                                 else "rgba(22, 163, 74, 0.45)"), width=1),
                annotation_text=(f"{role} x{z['touches']}"
                                 + (" (flipped)" if flipped else "")),
                annotation_position="top right")

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
                txts.append(name if labels else "")
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
        height=800, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0))
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Price-Action Charting", layout="wide")
st.title("📐 Price-Action Charting")
st.caption("The charting techniques from the *Price Action Trading* book — "
           "zones, trendlines, formations at levels, EMA 9/21, stochastic — "
           "drawn on any US stock. Display only: no verdicts, no signals.")

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
    span_label = st.selectbox("Sessions shown",
                              ["Today", "Last 3 sessions", "All loaded"], index=0)
    st.header("Show")
    show = {
        "zones": st.checkbox("Support / resistance zones", value=True),
        "lines": st.checkbox("Trendlines", value=True),
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
        df1 = fetch_bars_1m(symbol)
    except Exception as e:
        st.error(f"Could not load bars for {symbol}: {e}")
        return
    if df1.empty or len(df1) < 60:
        st.info(f"Not enough intraday data for {symbol} yet.")
        return
    dates = sorted(set(df1.index.date))
    if span_label == "Today":
        keep = dates[-1:]
    elif span_label == "Last 3 sessions":
        keep = dates[-3:]
    else:
        keep = dates
    df = df1[df1.index.date.isin(keep)]
    minutes = {"1 min": 1, "5 min": 5, "15 min": 15}[tf_label]
    disp = add_indicators(resample(df, minutes))
    if len(disp) < 30:
        st.info("Not enough bars in this view yet — try a smaller bar size "
                "or more sessions.")
        return

    pa = prior_day_levels(df1)
    today_df = df1[df1.index.date == dates[-1]]
    orng = opening_range(today_df) if span_label == "Today" else None
    zones = sr_zones(disp)
    lines = trendlines(disp)
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
              f"{sum(1 for z in zones if z['center'] >= float(last['Close']))} R / "
              f"{sum(1 for z in zones if z['center'] < float(last['Close']))} S")
    c3.metric("Stochastic %K / %D",
              f"{float(last['StochK']):.0f} / {float(last['StochD']):.0f}"
              if pd.notna(last["StochK"]) else "—")
    checks = confluence(disp, zones, level_vals)
    score = sum(1 for _, ok in checks if ok)
    c4.metric("Confluence (book stack)", f"{score} / {len(checks)}")

    st.plotly_chart(build_chart(disp, symbol, zones, lines, pats, level_vals,
                                pa, orng, show),
                    use_container_width=True,
                    key=f"pa_{symbol}_{tf_label}_{span_label}")
    if checks:
        st.markdown("**Confluence checklist** (the book's Premium stack, "
                    "at the last bar):")
        st.markdown("\n".join(
            f"- {'✅' if ok else '⬜'} {label}" for label, ok in checks))
    st.caption(
        f"Alpaca {str(_feed).upper()} · {len(df1):,} 1-min bars loaded "
        f"({dates[0]} → {dates[-1]}) · showing {len(disp):,} {tf_label} bars · "
        f"updated {datetime.now(ET).strftime('%H:%M:%S')} ET")


if in_market_hours():
    st.fragment(run_every=30)(render)()
else:
    render()
    st.caption("Market is closed — showing the last loaded session without "
               "auto-refresh.")

with st.expander("How to read this chart"):
    st.markdown(
        "- **Colored bands** — support (green) and resistance (red) zones: "
        "prices where the stock has turned **2 or more times**. The book's "
        "rule: a level only counts after repeated tests. A band marked "
        "\"(flipped)\" has acted as both support and resistance.\n"
        "- **Dotted trendlines** — through the two most recent swing "
        "highs/lows, kept only when tested at least twice. Price hitting the "
        "upper line is the book's selling area; the lower line, buying area.\n"
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
        "the one that matters, especially right at a level.")
