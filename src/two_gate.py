"""
Two-gate test for structural edge in systematic trading strategies.

The test operates on a strategy's trade-level output alone: a sequence of
R-multiples and partition labels. No access to the signal is required.

Gate 1 (edge)       : t-statistic of the mean, evaluated on the raw
                      directional return (free of stop/target parameters).
Gate 2 (invariance) : per-dimension Wasserstein-based invariance score
                      compared against a Monte-Carlo null.

Reference: Boutgajouft (2026), "A Two-Gate Test for Structural Alpha in
Systematic Trading Strategies", Quantitative Finance (under review).

Dependencies: numpy only.
"""

from __future__ import annotations

import zlib
import numpy as np

__all__ = ["t_edge", "sis", "null_threshold", "two_gate"]

# Quantile grid used for the 1-D Wasserstein-1 distance.
# W1(a,b) = int_0^1 |F_a^{-1}(u) - F_b^{-1}(u)| du, evaluated at mid-points.
_Q_GRID = 500
_U = (np.arange(_Q_GRID) + 0.5) / _Q_GRID


# --------------------------------------------------------------------------
# Gate 1
# --------------------------------------------------------------------------
def t_edge(R) -> float:
    """One-sample t-statistic of the mean of R.

    Parameters
    ----------
    R : array-like
        Per-trade returns. For benchmarks this is the raw directional return;
        for a strategy reaching its own exits, its realised R-multiples.
    """
    R = np.asarray(R, dtype=float)
    n = len(R)
    if n < 2:
        return 0.0
    s = R.std(ddof=1)
    return float(R.mean() * np.sqrt(n) / s) if s > 0 else 0.0


# --------------------------------------------------------------------------
# Gate 2
# --------------------------------------------------------------------------
def _sis_core(x: np.ndarray, masks: list[np.ndarray]) -> float:
    """Normalised mean pairwise W1 distance between group distributions."""
    quantiles = [np.quantile(x[m], _U) for m in masks]
    K = len(quantiles)
    if K < 2:
        return 0.0
    total, count = 0.0, 0
    for j in range(K):
        for k in range(j + 1, K):
            total += np.mean(np.abs(quantiles[j] - quantiles[k]))
            count += 1
    sd = x.std(ddof=1)
    return float((total / count) / sd) if sd > 0 else 0.0


def _group_masks(labels: np.ndarray, min_size: int = 2) -> list[np.ndarray]:
    return [np.where(labels == g)[0]
            for g in np.unique(labels)
            if (labels == g).sum() >= min_size]


def sis(R, labels) -> float:
    """Structural invariance score for one partition dimension.

    SIS_d = (1 / sigma_R) * mean_{j<k} W1(F_j, F_k)

    Lower values indicate greater distributional invariance across groups.
    """
    R = np.asarray(R, dtype=float)
    labels = np.asarray(labels)
    return _sis_core(R, _group_masks(labels))


def null_threshold(labels, N: int, B: int = 10_000, pct: float = 95.0,
                   rng: np.random.Generator | None = None,
                   chunk: int = 2500) -> float:
    """Monte-Carlo null threshold tau_d = P_pct(SIS_d | H0).

    Under H0 the returns are i.i.d. across groups. Because SIS is scale-free
    (it is normalised by sigma_R), the null is simulated with standard normal
    draws without loss of generality.

    Per-dimension thresholds are required: tau_d grows with the number of
    groups K_d, so a single global threshold would be miscalibrated.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.asarray(labels)
    masks = _group_masks(labels)
    K = len(masks)
    if K < 2:
        return 0.0

    out = np.empty(B)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        Z = rng.standard_normal((b, N))
        group_quantiles = []
        for m in masks:
            block = np.sort(Z[:, m], axis=1)
            pos = _U * (block.shape[1] - 1)
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, block.shape[1] - 1)
            frac = pos - lo
            group_quantiles.append(block[:, lo] * (1 - frac) + block[:, hi] * frac)
        sd = Z.std(axis=1, ddof=1)
        total = np.zeros(b)
        count = 0
        for j in range(K):
            for k in range(j + 1, K):
                total += np.mean(np.abs(group_quantiles[j] - group_quantiles[k]), axis=1)
                count += 1
        out[done:done + b] = (total / count) / sd
        done += b
    return float(np.percentile(out, pct))


# --------------------------------------------------------------------------
# Combined test
# --------------------------------------------------------------------------
def two_gate(R_edge, R_invariance, partitions: dict, t_star: float = 3.0,
             B: int = 10_000, seed: int = 42) -> dict:
    """Run the two-gate test.

    Parameters
    ----------
    R_edge : array-like
        Returns used for Gate 1. Free of stop/target parameters.
    R_invariance : array-like
        Returns used for Gate 2. Fixed reward-to-risk R-multiples, holding
        payoff discretisation constant across strategies.
    partitions : dict
        Mapping {'P1': labels, 'P2': labels, 'P3': labels} where P1 is the
        cross-section (instrument), P2 the volatility regime, P3 time.
    t_star : float
        Gate 1 bar. Default 3.0 (Harvey, Liu and Zhu, 2016).
    B : int
        Monte-Carlo replications for the null calibration.

    Returns
    -------
    dict with keys: t_edge, N, gate1, gate2, verdict, components.

    Notes
    -----
    Scope: the test applies to strategies defined by a signal producing
    discrete trades. A static buy-and-hold position has no trade structure
    to test and lies outside its scope.

    Power: the null threshold widens as N falls and as the number of groups
    K_d rises. A passing component at moderate N should be read as an absence
    of detected deviation rather than as established invariance. See
    power_analysis.py.
    """
    te = t_edge(R_edge)
    Rp = np.asarray(R_invariance, dtype=float)
    N = len(Rp)

    components = {}
    for name, labels in partitions.items():
        labels = np.asarray(labels)
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()))
        observed = sis(Rp, labels)
        tau = null_threshold(labels, N, B=B, rng=rng)
        components[name] = {
            "SIS": round(observed, 4),
            "tau": round(tau, 4),
            "pass": bool(observed < tau),
        }

    gate1 = bool(te >= t_star)
    gate2 = all(c["pass"] for c in components.values())

    if not gate1:
        verdict = "NO EDGE"
    elif gate2:
        verdict = "STRUCTURAL"
    elif not components.get("P2", {}).get("pass", True):
        verdict = "REGIME-CONTINGENT"
    elif not components.get("P3", {}).get("pass", True):
        verdict = "TEMPORAL INSTABILITY"
    else:
        verdict = "CROSS-SECTIONAL INSTABILITY"

    return {
        "t_edge": round(te, 2),
        "N": N,
        "gate1": gate1,
        "gate2": gate2,
        "verdict": verdict,
        "components": components,
    }
