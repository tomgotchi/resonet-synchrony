"""Selection-aware parametric bootstrap, one per selected pair (board round 3, Zhao D1/D3; Volkov round 2).

Why: a pair is chosen *because* its formation residual looked stationary. That makes, on the very pairs we trade,
  - the AR(1) half-life too short even after Kendall's correction (OLS shrinkage + selection),
  - the frozen formation mean/sd too tight, so |z| > 2 fires ~2-3x more often than the nominal 4.6%,
  - a CUSUM threshold set for known parameters and Gaussian noise far too trigger-happy (14-31% false alarms).
How: simulate the pair's own fitted model many times (resampled AR residuals -> fat tails kept; B's path is the
observed one in formation and a resampled random walk afterwards), re-run the same selection test, keep the
copies that would have been selected, and calibrate on those:
  1. phi: iterate phi <- phi + (phi_obs - median phi_hat of selected copies)  (bias correction by simulation);
  2. entry scale: the 95.4% quantile of |z| over the following trading window, so z/scale > 2 has the nominal
     4.6% rate on a pair that is genuinely cointegrated;
  3. CUSUM h: the 98% quantile of the window's max CUSUM statistic, so a healthy pair alarms <= 2% of windows.
"""
from __future__ import annotations

import numpy as np

from .coint import ar1_fit, cost_filter_ok, ols_alpha_beta, symmetric_eg

NOMINAL_TAIL = 0.0455      # P(|N(0,1)| > 2)


def _sim_spread(phi, eps, n, rng):
    e = rng.choice(eps, size=n, replace=True)
    s = np.empty(n)
    s[0] = e[0] / np.sqrt(max(1e-6, 1 - phi ** 2))
    for t in range(1, n):
        s[t] = phi * s[t - 1] + e[t]
    return s


def calibrate_pair(la: np.ndarray, lb: np.ndarray, p_threshold: float, cfg, seed: int,
                   n_sims: int = 150, iters: int = 2) -> dict:
    n = len(la)
    m = cfg.trading_days
    alpha, beta = ols_alpha_beta(la, lb)
    s_obs = la - alpha - beta * lb
    c, phi_obs, sig = ar1_fit(s_obs)
    eps = s_obs[1:] - c - phi_obs * s_obs[:-1]
    eps = eps - eps.mean()
    rb = np.diff(lb)
    rb = rb - rb.mean()
    rng = np.random.default_rng(seed)
    phi = min(0.999, (n * phi_obs + 1.0) / (n - 3.0))          # start from Kendall
    keep_info = None
    it = 0
    while it <= iters:
        phis, zabs, cmax = [], [], []
        for _ in range(n_sims):
            s = _sim_spread(phi, eps, n + m, rng)
            lb_full = np.concatenate([lb, lb[-1] + np.cumsum(rng.choice(rb, size=m))])
            la_full = alpha + beta * lb_full + s
            fa, fb = la_full[:n], lb_full[:n]
            r = symmetric_eg(fa, fb, cfg.adf_lags)
            if r["p"] > p_threshold:
                continue
            res = fa - r["alpha"] - r["beta"] * fb
            if not cost_filter_ok(fa, fb, cfg):      # Study 2: copies face the same filter (Zhao M3)
                continue
            cc, ph, sg = ar1_fit(res)
            phis.append(ph)
            if it == iters:
                mu, sd = res.mean(), res.std(ddof=1)
                cont = la_full[n:] - r["alpha"] - r["beta"] * lb_full[n:]
                zabs.append(np.abs((cont - mu) / sd))
                prev = np.concatenate([[res[-1]], cont[:-1]])
                nu = np.clip((cont - cc - ph * prev) / sg, -cfg.cusum_clip, cfg.cusum_clip)
                cp = cm = mx = 0.0
                for v in nu:
                    cp = max(0.0, cp + v - cfg.cusum_k)
                    cm = max(0.0, cm - v - cfg.cusum_k)
                    mx = max(mx, cp, cm)
                cmax.append(mx)
        accepted = len(phis)
        if accepted < 20 and p_threshold < 1.0:  # too few selected copies: redo without the selection step
            p_threshold = 1.0
            continue
        if accepted < 20:
            break
        if it < iters:
            phi = float(np.clip(phi + (phi_obs - np.median(phis)), 0.0, 0.999))
        else:
            zz = np.concatenate(zabs)
            keep_info = {"phi": phi, "accepted": accepted,
                         "z_scale": float(max(1.0, np.quantile(zz, 1 - NOMINAL_TAIL * cfg.z_tail_factor) / 2.0)),
                         "tail_before": float((zz > 2).mean()),
                         "cusum_h": float(max(4.0, np.quantile(cmax, cfg.cusum_quantile)))}
        it += 1
    if keep_info is None:
        keep_info = {"phi": phi, "accepted": 0, "z_scale": 1.0, "tail_before": float("nan"),
                     "cusum_h": cfg.cusum_h}
    keep_info["half_life"] = float(-np.log(2) / np.log(keep_info["phi"])) if 0 < keep_info["phi"] < 1 else float("inf")
    return keep_info
