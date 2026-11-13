#!/usr/bin/env python3
"""
Model: Two-window adaptive volatility with slow window dominant,
horizon-growing asymmetric spread, inner/outer quantile shaping,
jump-reversion median, no drift, no regression median.
"""

# Model choices:
# - Two-window blend with the slow window dominant (0.3/0.7) because the regime matters more than the last half hour.
# - Upside is squeezed by kup(h) = 0.25 + 0.75*(1-exp(-h/45)) because future moves are skewed down (23 of 24 candidates drift toward 0 while one jumps up) and the squeeze relaxes with horizon.
# - Inner quantiles are tighter still (upside x0.225, downside x0.9) because the 25%/75% band sits closer to the median than a normal spread implies.
# - Median reverts half of a one-minute jump over ~90 minutes because prices that spike usually pull partway back.
# - No plain drift term and no regression median, because every drift / mean-reversion anchor tested raised the practice pinball loss and the regression median fails forward out-of-sample validation.

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
    changes = [abs(sub[i] - sub[i-1]) for i in range(1, len(sub))]
    return sum(changes) / len(changes)

def market_sigma(prices):
    fast = mean_abs_change(prices, W_FAST)
    slow = mean_abs_change(prices, W_SLOW)
    blended = W_FAST_WEIGHT * fast + (1.0 - W_FAST_WEIGHT) * slow
    sig = blended * MEANABS_TO_SIGMA * SIGMA_MULT
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))

def median_path(prices, horizon):
    x = prices[-1]
    d = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    if abs(d) > JUMP_THRESH:
        return [x - JUMP_REVERT * d * (1.0 - math.exp(-h / JUMP_HALF_LIFE)) for h in range(1, horizon + 1)]
    else:
        return [x] * horizon

def forecast_market(prices, horizon):
    sig = market_sigma(prices)
    med = median_path(prices, horizon)
    forecasts = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        m = med[h - 1]
        kup = KUP_BASE + (1.0 - KUP_BASE) * (1.0 - math.exp(-h / KUP_HALF_LIFE))
        row = []
        for z in Z:
            if z == 0:
                val = m
            elif z > 0:
                mult = kup if abs(z) > 1 else INNER_UP
                val = m + z * spread * mult
            else:
                mult = KDOWN if abs(z) > 1 else INNER_DOWN
                val = m + z * spread * mult
            row.append(price6(val))
        for j in range(1, len(row)):
            if row[j] < row[j-1]:
                row[j] = row[j-1]
        forecasts.append(row)
    return forecasts

def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_file in sorted(input_dir.glob("*.in")):
        with open(case_file, "r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(f"Invalid quantiles in {case_file}")
        
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], case["horizon_minutes"])}
            for m in case["markets"]
        ]
        
        out_path = output_dir / f"{case_file.stem}.out"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")

if __name__ == "__main__":
    main()
