# rules for choosing test districts and groups
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TARGET_SHARE = 0.20
MIN_PER_GROUP = 3


def choose_test_districts(counts: pd.DataFrame, area_km2: pd.Series, sizes=(3, 4),
                          target: float = TARGET_SHARE, min_per_group: int = MIN_PER_GROUP):
    counts = counts.fillna(0)
    tot = counts.sum()
    groups = [g for g in counts.columns if tot[g] > 0]
    big = [g for g in groups if tot[g] >= 20]
    names = sorted(counts.index)
    rows = []
    for k in sizes:
        for combo in itertools.combinations(names, k):
            s = counts.loc[list(combo), groups].sum()
            share = s / tot[groups]
            if (s[big] < min_per_group).any():
                continue
            worst = float(np.max(np.abs(share - target)))
            total_share = float(s.sum() / tot[groups].sum())
            rows.append({"districts": combo, "k": k, "worst_abs_dev": round(worst, 4),
                         "total_share": round(total_share, 4),
                         "area_km2": round(float(area_km2.loc[list(combo)].sum()), 1),
                         **{f"share_{g}": round(float(share[g]), 3) for g in groups},
                         **{f"n_{g}": int(s[g]) for g in groups}})
    tab = pd.DataFrame(rows).sort_values(["worst_abs_dev", "area_km2", "districts"]).reset_index(drop=True)
    return list(tab.iloc[0]["districts"]), tab.head(20)


def greedy_folds(counts: pd.Series, n_folds: int = 5) -> dict[str, int]:
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    totals = [0] * n_folds
    out = {}
    for name, n in order:
        k = int(np.argmin(totals))
        out[name] = k
        totals[k] += int(n)
    return out
