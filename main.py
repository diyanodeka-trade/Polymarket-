#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecast: two-window adaptive volatility, horizon-growing
asymmetric spread, inner/outer quantile shaping, and jump-reversion median.

Model (practice pinball loss 0.261594 vs house baseline 0.363589, v3 0.267157):
- sigma: two-window blend, 0.3 * mean|1-min change| over the last 30 min
  + 0.7 * mean|1-min change| over the last 120 min, times 1.2533 and 1.5,
  clipped to [0.0005, 0.005]. The slow window dominates: the regime matters
  more than the last half hour.
- Asymmetric spread: future moves are skewed down (23 of 24 candidates drift
  toward 0 while one jumps up), so the upside is squeezed and the squeeze
  relaxes with horizon: outer upside quantile scaled by
  kup(h) = 0.25 + 0.75 * (1 - exp(-h/45)); downside scaled by 1.0.
  Inner quantiles (|z| = 0.674) are tighter still: upside x0.225, downside x0.9.
- Jump-reversion median: if the last 1-minute change is large (|d| > 0.03),
  the median reverts half of that jump, fading in over ~90 minutes
  (median(h) = x - 0.5*d*(1-exp(-h/90))). Prices that spike in one minute
  usually pull partway back; otherwise the median is the checkpoint price.
- No drift: every plain drift / mean-reversion anchor tested raised the loss.
  A regression median was tested and REJECTED: it helps in-sample but fails
  forward out-of-sample validation (train early -> test late hurts).
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

MEANABS_TO_SIGMA = 1.2533
SIGMA_MULT = 1.5
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005
W_FAST = 30
W_SLOW = 120
W_FAST_WEIGHT = 0.3
KUP_BASE = 0.25
KUP_HALF_LIFE = 45.0
KDOWN = 1.0
INNER_UP = 0.225
INNER_DOWN = 0.9
JUMP_THRESH = 0.03
JUMP_REVERT = 0.5
JUMP_HALF_LIFE = 90.0


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def mean_abs_change(prices, window):
    sub = prices[-(window + 1):] if len(prices) > window + 1 else prices
    if len(sub) < 2:
        return 0.0
    diffs = [abs(sub[i] - sub[i - 1]) for i in range(1, len(sub))]
    return sum(diffs) / len(diffs)


def market_sigma(prices):
    fast = mean_abs_change(prices, W_FAST)
    slow = mean_abs_change(prices, W_SLOW)
    blended = W_FAST_WEIGHT * fast + (1.0 - W_FAST_WEIGHT) * slow
    sig = blended * MEANABS_TO_SIGMA * SIGMA_MULT
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))


def median_path(prices, horizon):
    x = prices[-1]
    if len(prices) >= 2:
        d = prices[-1] - prices[-2]
    else:
        d = 0.0
    if abs(d) > JUMP_THRESH:
        return [
            x - JUMP_REVERT * d * (1.0 - math.exp(-h / JUMP_HALF_LIFE))
            for h in range(1, horizon + 1)
        ]
    return [x] * horizon


def forecast_market(prices, horizon):
    sig = market_sigma(prices)
    med = median_path(prices, horizon)
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        m = med[h - 1]
        kup = KUP_BASE + (1.0 - KUP_BASE) * (1.0 - math.exp(-h / KUP_HALF_LIFE))
        row = []
        for z in Z:
            if z == 0:
                row.append(price6(m))
            elif z > 0:
                scale = kup if abs(z) > 1 else INNER_UP
                row.append(price6(m + z * spread * scale))
            else:
                scale = KDOWN if abs(z) > 1 else INNER_DOWN
                row.append(price6(m + z * spread * scale))
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    for path in in_files:
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)

        if list(case.get("quantiles", [])) != QUANTILES:
            raise ValueError(f"Quantiles mismatch in {path}")

        forecasts = []
        for m in case.get("markets", []):
            q_rows = forecast_market(m["prices"], case["horizon_minutes"])
            forecasts.append({"id": m["id"], "q": q_rows})

        out_data = {"forecasts": forecasts}
        out_path = output_dir / f"{path.stem}.out"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out_data) + "\n")


if __name__ == "__main__":
    main()
