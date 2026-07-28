"""
Reproduce the benchmark table of the paper.

Intended to run in a QuantConnect Research Notebook, where the free data
sources used in the paper are available (OANDA FX, CBOE VIX, FRED rates).
Paste the contents of this file into a notebook cell, or import it if the
repository is on the notebook path.

Outputs the two-gate classification for every public benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks import build_signals, build_carry_signal, run_signal
from two_gate import two_gate

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FX_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "EURGBP"]
ETFS = ["SPY", "GLD"]
INSTRUMENTS = FX_PAIRS + ETFS

START, END = "2007-01-01", "2025-12-31"
SAMPLE_START_YEAR = 2008          # 2007 is warm-up for the 252-day lookback
N_TIME_BLOCKS = 3
T_STAR = 3.0
B_NULL = 10_000

FRED_RATES = {
    "USD": "DFF", "EUR": "ECBDFR", "GBP": "IUDSOIA",
    "JPY": "IRSTCI01JPM156N", "CHF": "IRSTCI01CHM156N",
    "AUD": "IRSTCI01AUM156N", "NZD": "IRSTCI01NZM156N",
    "CAD": "IRSTCI01CAM156N",
}


# --------------------------------------------------------------------------
# Data loading (QuantConnect)
# --------------------------------------------------------------------------
def load_data(qb):
    """Return (prices, vix, rates) from free QuantConnect data sources."""
    from AlgorithmImports import Resolution, Market, CBOE, Fred  # noqa

    symbols = {}
    for pair in FX_PAIRS:
        symbols[pair] = qb.add_forex(pair, Resolution.DAILY, Market.OANDA).symbol
    for etf in ETFS:
        symbols[etf] = qb.add_equity(etf, Resolution.DAILY).symbol

    hist = qb.history(list(symbols.values()), pd.Timestamp(START), pd.Timestamp(END),
                      Resolution.DAILY)
    px = hist["close"].unstack(level=0)
    px.columns = [next(k for k, v in symbols.items() if str(v) in str(c))
                  for c in px.columns]
    px = px.T.groupby(level=0).first().T[INSTRUMENTS].ffill()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[~px.index.duplicated(keep="last")]

    vix_symbol = qb.add_data(CBOE, "VIX", Resolution.DAILY).symbol
    vh = qb.history(CBOE, vix_symbol, pd.Timestamp(START), pd.Timestamp(END))
    vix = vh["close"].droplevel(0) if isinstance(vh.index, pd.MultiIndex) else vh["close"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    vix = vix[~vix.index.duplicated(keep="last")].sort_index()

    rates = {}
    for ccy, series in FRED_RATES.items():
        try:
            sym = qb.add_data(Fred, series, Resolution.DAILY).symbol
            h = qb.history(Fred, sym, pd.Timestamp(START), pd.Timestamp(END))
            v = h["value"].droplevel(0) if isinstance(h.index, pd.MultiIndex) else h["value"]
            v.index = pd.to_datetime(v.index).tz_localize(None).normalize()
            rates[ccy] = v[~v.index.duplicated(keep="last")].sort_index()
        except Exception as exc:
            print(f"  {ccy} ({series}) unavailable, excluded from carry: {exc}")

    return px, vix, rates


# --------------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------------
def make_partition_labellers(px, vix):
    """P2 splits the VIX at its in-sample median; P3 splits time into blocks."""
    vix_median = vix.loc["2008":"2025"].median()
    days = px.loc["2008":].index
    edges = [days[int(len(days) * k / N_TIME_BLOCKS)] for k in range(1, N_TIME_BLOCKS)]

    def label_regime(date):
        v = vix.asof(date)
        return None if pd.isna(v) else ("HIGH" if v > vix_median else "LOW")

    def label_time(date):
        for i, edge in enumerate(edges):
            if date < edge:
                return f"B{i + 1}"
        return f"B{N_TIME_BLOCKS}"

    return label_regime, label_time, vix_median


def classify(trades, label_regime, label_time):
    tr = trades.copy()
    tr["P1"] = tr["instrument"]
    tr["P2"] = [label_regime(d) for d in tr["date"]]
    tr["P3"] = [label_time(d) for d in tr["date"]]
    tr = tr.dropna(subset=["P2", "R_raw", "R_payoff"])
    partitions = {k: tr[k].values for k in ("P1", "P2", "P3")}
    return two_gate(tr["R_raw"].values, tr["R_payoff"].values,
                    partitions, t_star=T_STAR, B=B_NULL)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(qb):
    px, vix, rates = load_data(qb)
    vol20 = px.pct_change().rolling(20).std()
    label_regime, label_time, vix_median = make_partition_labellers(px, vix)
    print(f"Prices {px.shape} | in-sample VIX median {vix_median:.2f}")

    signals = build_signals(px, FX_PAIRS, INSTRUMENTS)
    signals["CarryFRED"] = build_carry_signal(px, rates, FX_PAIRS, INSTRUMENTS)

    results = {}
    for name, sig in signals.items():
        trades = run_signal(sig, px, vol20, start_year=SAMPLE_START_YEAR)
        if len(trades) < 50:
            print(f"{name}: only {len(trades)} trades, skipped")
            continue
        results[name] = classify(trades, label_regime, label_time)

    table = pd.DataFrame([
        {
            "Strategy": name,
            "N": r["N"],
            "t_edge": r["t_edge"],
            "SIS_P1": r["components"]["P1"]["SIS"],
            "SIS_P2": r["components"]["P2"]["SIS"],
            "SIS_P3": r["components"]["P3"]["SIS"],
            "Gate1": "pass" if r["gate1"] else "fail",
            "Verdict": r["verdict"],
        }
        for name, r in results.items()
    ]).sort_values("t_edge", ascending=False)

    print("\n" + table.to_string(index=False))
    return table


if __name__ == "__main__":
    print(__doc__)
    print("Run inside a QuantConnect Research Notebook:")
    print("    from run_benchmarks import main")
    print("    table = main(QuantBook())")
