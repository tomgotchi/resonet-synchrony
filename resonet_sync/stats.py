"""Paired statistics over units (subject / session / seed). Never pool windows or trials."""
from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- paired statistics
def paired_test(a, b, n_boot=10000, seed=0):
    """Paired comparison over units (seeds/sessions/subjects). d = a - b.
    Wilcoxon signed-rank (two-sided), bootstrap 95% CI on mean(d), Cohen's dz, rank-biserial r."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    res = dict(n=int(n), mean_diff=float(np.mean(d)) if n else np.nan,
               median_diff=float(np.median(d)) if n else np.nan)
    if n < 2:
        return res | dict(p=np.nan, ci95=[np.nan, np.nan], dz=np.nan, rank_biserial=np.nan)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, (n_boot, n))].mean(1)
    sd = d.std(ddof=1)
    if np.all(d == 0):
        p, rb = 1.0, 0.0
    else:
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        p = float(w.pvalue)
        nz = d[d != 0]
        rk = stats.rankdata(np.abs(nz))
        rb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum())
    return res | dict(p=p, ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                      dz=float(np.mean(d) / sd) if sd > 0 else np.nan, rank_biserial=rb)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def contrast_table(per_unit, contrasts, n_boot=10000, seed=0):
    """per_unit: list of dicts {arm: score}. contrasts: list of (name, arm_a, arm_b).
    Returns rows with paired stats; Holm applied within the given family."""
    rows = []
    for name, a, b in contrasts:
        A = [u.get(a, np.nan) for u in per_unit]
        B = [u.get(b, np.nan) for u in per_unit]
        r = paired_test(A, B, n_boot=n_boot, seed=seed)
        r.update(contrast=name, a=a, b=b, wins=int(np.sum(np.asarray(A) > np.asarray(B))))
        rows.append(r)
    adj = holm([r["p"] if np.isfinite(r["p"]) else 1.0 for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)
    return rows
