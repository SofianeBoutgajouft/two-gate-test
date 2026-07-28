# Data

## `tested_strategy_trades.csv`

Trade-level output of the tested strategy, regenerated on public data (OANDA
daily FX via QuantConnect). This is the only information the test requires: the
signal that produced these trades is not disclosed, which is the setting the
paper addresses.

### Schema

| Column | Type | Description |
|---|---|---|
| `instrument` | string | Currency pair, e.g. `EURUSD` |
| `date` | ISO date | Entry date, used to assign the P2 and P3 partition labels |
| `R` | float | Realised R-multiple of the trade |

Three columns, nothing further. Entry and exit prices, position sizes, internal
scores and exit reasons are deliberately excluded: they are not needed by the
test and could bear on the undisclosed mechanism.

### Reproducing the reported figures

```python
import pandas as pd, numpy as np
from two_gate import two_gate

trades = pd.read_csv("data/tested_strategy_trades.csv", parse_dates=["date"])

# Gate 1 uses the realised R-multiples, since the strategy reaches its payoff
# through its own exit logic rather than a mechanical fixed payoff.
R = trades["R"].values

partitions = {
    "P1": trades["instrument"].values,
    "P2": [label_regime(d) for d in trades["date"]],   # VIX at entry, median split
    "P3": [label_time(d) for d in trades["date"]],     # equal time blocks
}

print(two_gate(R, R, partitions))
```

The partition labellers are defined in `src/run_benchmarks.py` and use the same
CBOE VIX series and time blocks as the benchmarks, so the classification is
like-for-like.
