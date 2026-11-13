#!/usr/bin/env python3
"""
Polymarket NVIDIA CEO Forecast Model

This model forecasts minute-by-minute Yes prices 4 hours ahead (240 horizons)
as five quantiles per minute using per-market adaptive volatility.
"""

import json
import math
import os
from pathlib import Path

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

WINDOW = 60
MEANABS_TO_SIGMA = 1.2533
SIGMA_MULT = 1.5
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005

def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0

def market_sigma(prices):
    sub = prices[-(WINDOW + 1):] if len(prices) > WINDOW + 1 else prices
    if len(sub) < 2:
        return SIGMA_FLOOR
    diffs = [abs(sub[i] - sub[i-1]) for i in range(1, len(sub))]
    mean_abs = sum(diffs) / len(diffs)
    sig = mean_abs * MEANABS_TO_SIGMA * SIGMA_MULT
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig))

def forecast_market(prices, horizon):
    x = prices[-1]
    sig = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sig * math.sqrt(h)
        row = [price6(x + z * spread) for z in Z]
        for j in range(1, len(row)):
            if row[j] < row[j-1]:
                row[j] = row[j-1]
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
