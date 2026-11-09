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


def sign_flip_p(d, n_mc=200_000, seed=0):
    """Two-sided sign-flip permutation p of the mean paired difference: exact (all 2^n patterns) for n <= 16,
    Monte Carlo otherwise. Ties with the observed |mean| count (conservative)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return np.nan
    obs = abs(d.mean()) - 1e-12
    if n <= 16:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = np.random.default_rng(seed).choice((-1, 1), (n_mc, n))
    return float(np.mean(np.abs(signs @ d) / n >= obs))


def t_ci(d, level=0.95):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return [np.nan, np.nan]
    h = stats.t.ppf(0.5 + level / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(d.mean() - h), float(d.mean() + h)]


def tost_p(d, margin):
    """Two one-sided t-tests for equivalence within +/- margin; returns the larger one-sided p."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    p_lo = stats.t.sf((d.mean() + margin) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p_lo, p_hi))


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


INFERENTIAL_KEYS = ("p", "p_holm", "ci95", "dz", "rank_biserial", "p_wilcoxon", "p_wilcoxon_exact", "ci95_t",
                    "ci90_t", "p_tost", "p_tost_holm", "outcome")


def phase1_rows(rows, margin=0.05, alpha=0.05):
    """DEVIATION 23: replace the primary p by the exact sign-flip p (Wilcoxon kept alongside), add t CIs and TOST,
    and classify each tested row as superior / inferior / equivalent / inconclusive. Holm over tested rows."""
    for r in rows:
        d = np.asarray(r.pop("_d"), float)
        if r["n"] < 2 or r.get("incomplete"):
            continue
        r["p_wilcoxon"] = r["p"]
        try:
            r["p_wilcoxon_exact"] = float(stats.wilcoxon(d, method="exact").pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            r["p_wilcoxon_exact"] = np.nan
        r["p"] = sign_flip_p(d)
        r["ci95_t"], r["ci90_t"], r["p_tost"] = t_ci(d, 0.95), t_ci(d, 0.90), tost_p(d, margin)
    tested = [r for r in rows if r["n"] >= 2 and not r.get("incomplete") and np.isfinite(r["p"])]
    for key, out in (("p", "p_holm"), ("p_tost", "p_tost_holm")):
        adj = holm([r[key] for r in tested]) if tested else []
        for r, p in zip(tested, adj):
            r[out] = float(p)
    for r in tested:
        r["holm_m"] = len(tested)
        r["outcome"] = ("superior" if r["p_holm"] < alpha and r["mean_diff"] > 0 else
                        "inferior" if r["p_holm"] < alpha and r["mean_diff"] < 0 else
                        "equivalent" if r["p_tost_holm"] < alpha else "inconclusive")
    return rows


def contrast_table(per_unit, contrasts, n_boot=10000, seed=0, min_n=None, phase1=False, margin=0.05):
    """per_unit: list of dicts {arm: score}. contrasts: list of (name, arm_a, arm_b).
    Returns rows with paired stats; Holm applied within the given family.

    Holm runs over the rows that were actually tested (finite p and, if given, n >= min_n); its size is stored
    as `holm_m`. Rows that could not be tested (an arm that never ran) no longer enter Holm as p = 1 (board
    2026-11-09: that silently doubled the family for banks that were not run). With `min_n`, rows below it are
    descriptive only: their inferential fields are NaN and `incomplete` is True (no interim p-values).
    phase1=True applies the DEVIATION 23 rules (exact sign-flip primary test, t CIs, TOST, outcome); see phase1_rows."""
    rows = []
    for name, a, b in contrasts:
        A = [u.get(a, np.nan) for u in per_unit]
        B = [u.get(b, np.nan) for u in per_unit]
        r = paired_test(A, B, n_boot=n_boot, seed=seed)
        r.update(contrast=name, a=a, b=b, wins=int(np.sum(np.asarray(A) > np.asarray(B))),
                 _d=(np.asarray(A, float) - np.asarray(B, float)).tolist())
        rows.append(r)
    tested = [i for i, r in enumerate(rows) if np.isfinite(r["p"]) and (min_n is None or r["n"] >= min_n)]
    adj = holm([rows[i]["p"] for i in tested]) if tested else []
    for r in rows:
        r["p_holm"], r["holm_m"] = np.nan, len(tested)
    for i, p in zip(tested, adj):
        rows[i]["p_holm"] = float(p)
    if min_n is not None:
        for r in rows:
            r["incomplete"] = bool(r["n"] < min_n)
            if r["incomplete"]:              # NaN (not None): older notebook copies format these fields
                r.update({k: ([np.nan, np.nan] if k.startswith("ci") else None if k == "outcome" else np.nan)
                          for k in INFERENTIAL_KEYS})
    if phase1:
        return phase1_rows(rows, margin=margin)
    for r in rows:
        r.pop("_d")
    return rows


def boot_ci(d, n=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean (moved from the screen-1 runner; used by every screen).
    Review board: anti-conservative at small n (coverage ~0.90 at n=10, ~0.87 at n=7)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
