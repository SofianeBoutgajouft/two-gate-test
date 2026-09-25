# Two-Gate Test for Structural Edge

Reference implementation and replication code for:

The test asks whether a systematic trading strategy's edge is **structural**,
that is invariant across instruments, volatility regimes and time, or
**regime-contingent**. It operates on the strategy's trade-level output alone:
a sequence of R-multiples and partition labels. **No access to the signal is
required**, which is what allows a third party to validate a strategy whose
mechanism is not disclosed.

---

## The test

**Gate 1 (edge).** The one-sample *t*-statistic of the mean per-trade return,
evaluated on the **raw directional return**, which carries no stop or target
parameter. Passed if `t >= 3` (Harvey, Liu and Zhu, 2016). Judging the edge on
a fixed reward-to-risk payoff instead would make the verdict depend on the
arbitrary stop distance.

**Gate 2 (invariance).** For each partition dimension *d*, the normalised mean
pairwise Wasserstein-1 distance between group return distributions:

```
SIS_d = (1 / sigma_R) * mean_{j<k} W1(F_dj, F_dk)
```

compared against its own Monte-Carlo null threshold `tau_d = P95(SIS_d | H0)`.
Per-dimension thresholds are required because `tau_d` grows with the number of
groups. Passed if `SIS_d < tau_d` for all dimensions.

A strategy is classified as structural only if it clears **both** gates. A
single invariance score is not sufficient: a zero-expectancy strategy that is
i.i.d. across partitions is distributionally invariant by construction, and
would be misclassified by an invariance-only metric.

When a strategy fails, the failure locates the pathology: `t < 3` indicates no
edge; an elevated `SIS_P2` indicates regime contingency; an elevated `SIS_P3`
indicates temporal instability.

---

## Repository contents

```
src/
  two_gate.py         core test: Gate 1, Gate 2, null calibration
  benchmarks.py       public benchmark rules + fixed-payoff trade mapper
  run_benchmarks.py   QuantConnect runner reproducing the paper's table
  power_analysis.py   Monte-Carlo power study behind the sample-size guidance
data/
  tested_strategy_trades.csv    trade-level R-multiples of the tested strategy
results/
  benchmark_table.csv           two-gate classification of the benchmarks
```

**Not included:** the signal-generating code of the tested strategy. This is by
design rather than omission. The test requires only the realised output, so the
strategy's trade sequence is sufficient to reproduce every figure reported for
it; the mechanism remains undisclosed, which is precisely the setting the test
addresses.

---

## Usage

```python
from two_gate import two_gate

result = two_gate(
    R_edge=raw_returns,           # Gate 1 input: raw directional returns
    R_invariance=r_multiples,     # Gate 2 input: fixed-payoff R-multiples
    partitions={
        "P1": instrument_labels,  # cross-section
        "P2": regime_labels,      # volatility regime
        "P3": time_labels,        # time blocks
    },
)

print(result["t_edge"], result["verdict"])
```

`two_gate.py` depends on NumPy only and can be applied to any trade sequence.

### Reproducing the paper's table

`run_benchmarks.py` is written for a QuantConnect Research Notebook, where the
free data sources used in the paper are available:

| Input | Source | Cost |
|---|---|---|
| Daily FX and ETF prices | OANDA via QuantConnect | free |
| Short rates (carry benchmark) | FRED | free |
| Volatility regime (P2) | CBOE VIX | free |

```python
from run_benchmarks import main
table = main(QuantBook())
```

### Reproducing the power analysis

Entirely synthetic, so it runs anywhere:

```bash
python src/power_analysis.py
```

---

## Benchmarks

Fourteen publicly specified strategies, each defined by a transparent rule and
replicable from published descriptions: 5-day mean reversion; 20/60 and 10/50
moving-average trend; MACD; Bollinger mean reversion; Donchian breakout (20-
and 55-day); 60-day time-series momentum; 12-month time-series momentum
(Moskowitz, Ooi and Pedersen); cross-sectional momentum and reversal; RSI(2)
reversal; and cross-sectional FX carry built from published policy rates.

No benchmark parameter is tuned on the sample: every rule uses its conventional
published setting.

---

## Scope and limitations

**Scope.** The test applies to strategies defined by a signal producing discrete
trades. A static buy-and-hold position has no entry or exit signal and therefore
no trade structure to test; such positions lie outside the test's scope, and
their return reflects a directional risk premium rather than a systematic edge.

**What the test does not do.** It characterises the robustness of an edge, not
its decomposition into alpha and beta. Separating the two requires a factor
attribution, which is complementary to a signal-blind test. It also does not
detect overfitting in the strict sense, which is a property of in-sample
selection not recoverable from a single realised output sequence; we recommend
reporting the probability of backtest overfitting (Bailey et al., 2017)
alongside.

**Power.** The null threshold widens as N falls and as the number of groups
rises, so a passing component at moderate N should be read as an absence of
detected deviation rather than as established invariance. The cross-sectional
component, having the most groups, is the least powerful at a given N. Run
`power_analysis.py` for the rejection rates behind this guidance.

---

## References

Bailey, D.H., Borwein, J.M., López de Prado, M. and Zhu, Q.J. (2017). The
probability of backtest overfitting. *Journal of Computational Finance*, 20,
39–69.

Hallin, M., Mordant, G. and Segers, J. (2021). Multivariate goodness-of-fit
tests based on Wasserstein distance. *Electronic Journal of Statistics*, 15,
1328–1371.

Harvey, C.R., Liu, Y. and Zhu, H. (2016). ...and the cross-section of expected
returns. *Review of Financial Studies*, 29, 5–68.

Peyré, G. and Cuturi, M. (2019). Computational optimal transport. *Foundations
and Trends in Machine Learning*, 11, 355–607.

Villani, C. (2009). *Optimal Transport: Old and New.* Springer.

---

## License

CC BY-NC 4.0. See `LICENSE`.

## Contact

Sofiane Boutgajouft, OrgaX LLC — founder@orgaxtech.com
