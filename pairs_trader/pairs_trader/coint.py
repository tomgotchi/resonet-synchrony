"""Engle-Granger cointegration with the correct (MacKinnon) critical values, vectorised over many pairs.

Board finding (verified): a plain ADF p-value on OLS residuals rejects ~14% of independent random-walk pairs.
We use statsmodels' MacKinnon response surface for N=2 variables with a constant, which gives ~5%.
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.adfvalues import mackinnonp


def ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    beta = ((x - xm) * (y - ym)).sum() / vx
    return float(ym - beta * xm), float(beta)


def adf_tstat(r: np.ndarray, lags: int = 1) -> float:
    """ADF t-statistic on an OLS residual: no constant (the residual is already demeaned) and `lags` lagged
    differences. This is the regression statsmodels.coint runs; critical values come from MacKinnon N=2, 'c'."""
    dr = np.diff(r)
    n = len(dr) - lags
    cols = [r[lags:-1]]
    for k in range(1, lags + 1):
        cols.append(dr[lags - k:len(dr) - k])
    X = np.column_stack(cols)
    yv = dr[lags:]
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    s2 = resid @ resid / (n - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[0] / np.sqrt(cov[0, 0]))


def eg_pvalue(y: np.ndarray, x: np.ndarray, lags: int = 1) -> tuple[float, float, float, float]:
    """Engle-Granger test of y on x. Returns (pvalue, tstat, alpha, beta)."""
    a, b = ols_alpha_beta(y, x)
    t = adf_tstat(y - a - b * x, lags)
    return float(mackinnonp(t, regression="c", N=2)), t, a, b


def symmetric_eg(la: np.ndarray, lb: np.ndarray, lags: int = 1) -> dict:
    """Test both regression directions; the pair's p-value is the larger one (both must pass)."""
    p_ab, t_ab, a, b = eg_pvalue(la, lb, lags)
    p_ba, t_ba, _, _ = eg_pvalue(lb, la, lags)
    return {"p": max(p_ab, p_ba), "p_ab": p_ab, "p_ba": p_ba, "tstat": t_ab, "alpha": a, "beta": b}


def ar1_fit(s: np.ndarray) -> tuple[float, float, float]:
    """s_t = c + phi s_{t-1} + e. Returns (c, phi, sigma_e) by OLS."""
    c_beta = ols_alpha_beta(s[1:], s[:-1])
    c, phi = c_beta
    e = s[1:] - c - phi * s[:-1]
    return c, phi, float(e.std(ddof=2))


def half_life_corrected(phi: float, n: int) -> float:
    """Half-life (bars) after Kendall's small-sample bias correction E[phi_hat] ~ phi - (1 + 3 phi)/n."""
    phi_c = (n * phi + 1.0) / (n - 3.0)
    if not 0.0 < phi_c < 1.0:
        return float("inf") if phi_c >= 1.0 else 0.0
    return float(-np.log(2.0) / np.log(phi_c))


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of discoveries at FDR level q."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


def cost_filter_ok(spread: np.ndarray, beta: float, cfg) -> bool:
    """Study 2 pre-BH independent filter (board 2026-11-19, Zhao C1 / Volkov C2): a pair can only pay for its
    own trading if one typical day's spread move is at least the round-trip cost per dollar of leg A,
    sd(diff spread) >= m * 2 * c * (1 + |beta|). A scale statistic, roughly independent of the scale-free EG t
    under the null, so filtering before BH keeps FDR control. m = 0 disables it (Study 1)."""
    m = getattr(cfg, "cost_filter_m", 0.0)
    if m <= 0:
        return True
    sd_diff = float(np.std(np.diff(spread), ddof=1))
    return sd_diff >= m * 2.0 * cfg.screen_cost_bps / 1e4 * (1.0 + abs(beta))
