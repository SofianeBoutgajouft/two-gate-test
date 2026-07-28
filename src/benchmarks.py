"""
Public benchmark strategies and the fixed-payoff trade mapper.

Every benchmark is defined by a transparent rule on daily prices, so the
full table of the paper can be regenerated from free data sources.

Data sources used in the paper (all free):
  - daily FX and ETF prices : OANDA via QuantConnect
  - short rates (carry)     : FRED
  - volatility regime (P2)  : CBOE VIX

This module is data-source agnostic: it takes a price DataFrame and returns
trades. See run_benchmarks.py for the QuantConnect loader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed reward-to-risk payoff, held constant across strategies (Gate 2 input).
C_RISK = 0.5          # risk unit = C_RISK * trailing 20-day volatility
RR_TARGET = 3.0       # a path reaching +3R books +3
TIME_STOP = 20        # a 20-day time-stop books 0
RAW_WINDOW = 20       # horizon of the raw directional return (Gate 1 input)


# --------------------------------------------------------------------------
# Signal definitions
# --------------------------------------------------------------------------
def _sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df).fillna(0)


def build_signals(px: pd.DataFrame, fx_pairs: list[str],
                  instruments: list[str]) -> dict[str, pd.DataFrame]:
    """Return {name: daily signal DataFrame of {-1, 0, +1}}.

    Each rule is standard and published; none is tuned on the sample.
    """
    Z = pd.DataFrame(0.0, index=px.index, columns=instruments)
    signals: dict[str, pd.DataFrame] = {}

    # Mean reversion: fade the sign of the trailing 5-day return.
    signals["MeanRev5"] = -_sign(px.pct_change(5))

    # Trend: sign of the moving-average gap.
    signals["Trend2060"] = _sign(px.rolling(20).mean() - px.rolling(60).mean())
    signals["MA1050"] = _sign(px.rolling(10).mean() - px.rolling(50).mean())

    # MACD: sign of the 12/26 EMA gap.
    signals["MACD"] = _sign(px.ewm(span=12).mean() - px.ewm(span=26).mean())

    # Time-series momentum: sign of the trailing return.
    signals["Mom60"] = _sign(px.pct_change(60))
    signals["TSMOM252"] = (_sign(px[fx_pairs].pct_change(252))
                           .reindex(columns=instruments).fillna(0))

    # Donchian breakout: enter on a range break.
    for window in (20, 55):
        hi = px.rolling(window).max().shift(1)
        lo = px.rolling(window).min().shift(1)
        signals[f"Donchian{window}"] = ((px > hi).astype(float)
                                        - (px < lo).astype(float))

    # Bollinger mean reversion: fade a two-sigma deviation.
    z = (px - px.rolling(20).mean()) / px.rolling(20).std()
    signals["Bollinger"] = ((-(z > 2).astype(float) + (z < -2).astype(float))
                            [fx_pairs].reindex(columns=instruments).fillna(0))

    # RSI(2) reversal.
    d = px.diff()
    up = d.clip(lower=0).rolling(2).mean()
    dn = (-d.clip(upper=0)).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    signals["RSI2"] = (((rsi2 < 10).astype(float) - (rsi2 > 90).astype(float))
                       [fx_pairs].reindex(columns=instruments).fillna(0))

    # Cross-sectional momentum and reversal: long the top two, short the bottom two.
    for name, lookback, reverse in [("XSMom60", 60, False), ("XSRev5", 5, True)]:
        scores = px[fx_pairs].pct_change(lookback)
        sig = Z.copy()
        for dt in px.index:
            if dt not in scores.index:
                continue
            row = scores.loc[dt].dropna()
            if len(row) < 5:
                continue
            ranked = row.sort_values()
            long_side, short_side = ranked.index[-2:], ranked.index[:2]
            if reverse:
                long_side, short_side = short_side, long_side
            for p in long_side:
                sig.loc[dt, p] = 1.0
            for p in short_side:
                sig.loc[dt, p] = -1.0
        signals[name] = sig

    return signals


def build_carry_signal(px: pd.DataFrame, rates: dict[str, pd.Series],
                       fx_pairs: list[str], instruments: list[str]) -> pd.DataFrame:
    """Cross-sectional carry: long the two highest-yielding currencies,
    short the two lowest, rebalanced monthly. Rates are published policy rates."""
    sig = pd.DataFrame(0.0, index=px.index, columns=instruments)
    for month_end in px.resample("ME").last().index:
        snapshot = {c: s.asof(month_end) for c, s in rates.items()
                    if not pd.isna(s.asof(month_end))}
        if len(snapshot) < 5:
            continue
        ranked = sorted(snapshot, key=snapshot.get)
        lows, highs = set(ranked[:2]), set(ranked[-2:])
        future = px.index > month_end
        if not future.any():
            continue
        span = px.index[future & (px.index <= month_end + pd.offsets.MonthEnd(1))]
        for pair in fx_pairs:
            base, quote = pair[:3], pair[3:]
            ccy = base if base != "USD" else quote
            direction = 1 if ccy in highs else (-1 if ccy in lows else 0)
            if base == "USD":
                direction = -direction
            if direction:
                sig.loc[span, pair] = direction
    return sig


# --------------------------------------------------------------------------
# Trade mapper
# --------------------------------------------------------------------------
def run_signal(signal: pd.DataFrame, px: pd.DataFrame, vol20: pd.DataFrame,
               instruments: list[str] | None = None,
               start_year: int = 2008) -> pd.DataFrame:
    """Map a daily signal to non-overlapping trades.

    Entries sit on a fixed grid of RAW_WINDOW trading days, so trades never
    overlap and the trade count does not depend on exit speed.

    Returns a DataFrame with columns [instrument, date, R_raw, R_payoff]:
      R_raw     : raw directional return over RAW_WINDOW days (Gate 1)
      R_payoff  : fixed reward-to-risk R-multiple (Gate 2)
    """
    instruments = instruments or [c for c in signal.columns if c in px.columns]
    rows = []
    horizon = max(TIME_STOP, RAW_WINDOW)

    for inst in instruments:
        prices = px[inst].dropna()
        sig = signal[inst].reindex(prices.index).fillna(0)
        vol = vol20[inst].reindex(prices.index)
        idx = prices.index
        i, n = 0, len(idx)

        while i < n - horizon - 1:
            date, s = idx[i], sig.iloc[i]
            if (date.year < start_year or s == 0
                    or np.isnan(vol.iloc[i]) or vol.iloc[i] <= 0):
                i += 1
                continue

            risk = C_RISK * vol.iloc[i]
            entry = prices.iloc[i]
            r_payoff = 0.0                       # time-stop books 0
            for k in range(1, TIME_STOP + 1):
                cumulative = s * (prices.iloc[i + k] / entry - 1.0)
                if cumulative >= RR_TARGET * risk:
                    r_payoff = RR_TARGET
                    break
                if cumulative <= -risk:
                    r_payoff = -1.0
                    break

            r_raw = s * (prices.iloc[i + RAW_WINDOW] / entry - 1.0)
            rows.append((inst, date, r_raw, r_payoff))
            i += RAW_WINDOW

    return pd.DataFrame(rows, columns=["instrument", "date", "R_raw", "R_payoff"])
