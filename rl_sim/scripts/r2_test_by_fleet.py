"""
r2_test_by_fleet.py — the round-2 varied test, split by fleet size.

Reads out/2c_r2_test.csv (no new runs) and writes
out/2c_r2_test_by_fleet.csv: for each group of M-drone counts, the mean total
loss of every policy and the paired difference of each policy against
midpoint, on the varied test worlds 651-750.

Usage:
    python scripts/r2_test_by_fleet.py
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_c_ppo import write_csv
from scripts.train_c_ppo_round1 import read_csv

GROUPS = ((1, 6), (7, 12), (13, 18))
POLICIES = ("control", "A", "B", "C", "direct", "midpoint", "hover")


def main(out_dir: str = "out") -> None:
    rows = [r for r in read_csv(os.path.join(out_dir, "2c_r2_test.csv"))
            if r["set"] == "varied"]
    loss = {(r["policy"], r["placement_seed"]): float(r["total_loss"])
            for r in rows}
    n_m = {r["placement_seed"]: int(r["n_m"]) for r in rows}
    out = []
    for lo, hi in GROUPS:
        worlds = sorted(k for k, n in n_m.items() if lo <= n <= hi)
        for name in POLICIES:
            diffs = [loss[(name, w)] - loss[("midpoint", w)] for w in worlds]
            out.append({
                "m_drones": f"{lo}-{hi}", "worlds": len(worlds),
                "policy": name,
                "total_loss": statistics.fmean(loss[(name, w)]
                                               for w in worlds),
                "minus_midpoint": statistics.fmean(diffs),
                "lower_than_midpoint": sum(d < 0 for d in diffs),
            })
            print(f"  M {lo:>2}-{hi:<2} {len(worlds):>3} worlds  {name:>8} "
                  f"{out[-1]['total_loss']:6.2f}  vs midpoint "
                  f"{out[-1]['minus_midpoint']:+6.2f}  lower on "
                  f"{out[-1]['lower_than_midpoint']}/{len(worlds)}")
    write_csv(os.path.join(out_dir, "2c_r2_test_by_fleet.csv"), out,
              list(out[0]), "phase 2c round 2 varied test 651-750, split by "
              "number of M-drones, from out/2c_r2_test.csv")


if __name__ == "__main__":
    main()
