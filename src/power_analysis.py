"""
Power analysis for the invariance gate (Gate 2).

Derives the sample-size recommendation reported in the paper. The experiment
is entirely synthetic: no market data is used, so the calibration does not
depend on any particular sample.

Design (following the graded experiment of the appendix):
  - fixed reward-to-risk payoff: R in {+3, 0, -1}
  - deviation from invariance injected as a win-rate gap between groups
  - null: i.i.d. standard normal, tau = 95th percentile

Run:  python power_analysis.py
"""

from __future__ import annotations

import numpy as np

from two_gate import _sis_core, _group_masks, null_threshold

BASE_WIN_RATE = 0.45
SCRATCH_SHARE = 0.30      # share of non-winners exiting at breakeven


def generate_returns(n: int, win_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Fixed-payoff R-multiples: +3 on a win, 0 on a scratch, -1 on a loss."""
    u = rng.random(n)
    scratch_cut = win_rate + SCRATCH_SHARE * (1.0 - win_rate)
    return np.where(u < win_rate, 3.0, np.where(u < scratch_cut, 0.0, -1.0))


def rejection_rate(N: int, K: int, gap_pp: float, n_rep: int = 400,
                   B: int = 1500, seed: int = 0) -> tuple[float, float]:
    """Fraction of replications in which the invariance gate rejects.

    At gap_pp = 0 this is the type-I error rate; above it, the power.
    """
    rng = np.random.default_rng(seed)
    labels = np.arange(N) % K
    masks = _group_masks(labels)
    tau = null_threshold(labels, N, B=B, rng=rng)

    win_rates = np.linspace(BASE_WIN_RATE + gap_pp / 200,
                            BASE_WIN_RATE - gap_pp / 200, K)
    rejections = 0
    for _ in range(n_rep):
        R = np.empty(N)
        for g, m in enumerate(masks):
            R[m] = generate_returns(len(m), win_rates[g], rng)
        if R.std(ddof=1) > 0 and _sis_core(R, masks) >= tau:
            rejections += 1
    return rejections / n_rep, tau


def main() -> None:
    sample_sizes = [200, 300, 400, 500, 600, 800, 1000, 1200]
    gaps = [0, 5, 10, 15, 20]
    dimensions = [(2, "P2  (2 groups: volatility regime)"),
                  (3, "P3  (3 groups: time blocks)"),
                  (7, "P1  (7 groups: instrument)")]

    for K, label in dimensions:
        print(f"\n=== Rejection rate, {label} ===")
        print("gap(pp) " + "".join(f"{n:>8}" for n in sample_sizes))
        for gap in gaps:
            row = [rejection_rate(N, K, gap, seed=abs(hash((N, K, gap))) % 10_000)[0]
                   for N in sample_sizes]
            note = "   <- type-I error rate" if gap == 0 else ""
            print(f"{gap:>7} " + "".join(f"{p:>8.2f}" for p in row) + note)

    print("\nRead: the type-I rate should sit at or below the nominal 0.05.")
    print("Power rises with N and with the injected gap, and falls as the")
    print("number of groups K_d rises, since each group holds N/K_d observations")
    print("and the threshold must cover K_d(K_d-1)/2 pairwise comparisons.")


if __name__ == "__main__":
    main()
