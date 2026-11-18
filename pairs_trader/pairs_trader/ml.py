"""Optional ML entry filter (OFF by default).

Board findings applied:
- Label = sign of the trade's realised P&L *net of costs* (not "reverted before stops", which is ~96% true even
  on pure random walks because z is measured against a mean).
- Training data = only trades whose exit fill is strictly before the current window starts (all trades close by
  their window's end, so this is a purged walk-forward split with no label overlap).
- Judged by the net P&L of the trades it keeps against the rules-only baseline (see `compare_ml`), not by AUC.
The features are functions of the same two price series, so expect little; the code exists to *test* that.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["abs_z", "half_life", "tstat", "beta", "sd", "cusum"]


def make_filter_factory(cfg):
    def factory(past_trades):
        rows = [t for t in past_trades if t.features is not None and t.exit_fill is not None]
        if len(rows) < cfg.ml_min_train_trades:
            return None                                  # not enough history: take every rule signal
        X = np.array([[t.features[f] for f in FEATURES] for t in rows], float)
        y = np.array([t.pnl > 0 for t in rows], int)
        if y.min() == y.max():
            return None
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(X, y)

        def keep(pair, t, feats):
            x = np.array([[feats[f] for f in FEATURES]], float)
            return bool(model.predict_proba(x)[0, 1] > cfg.ml_threshold)
        return keep
    return factory


def compare_ml(prices, cfg, sectors=None) -> dict:
    """Rules-only vs rules+ML. Also a fair baseline: rules with a stricter entry z giving a similar trade count."""
    from .backtest import sharpe, walk_forward
    base = walk_forward(prices, cfg, sectors)
    ml = walk_forward(prices, cfg.with_(use_ml_filter=True), sectors,
                      entry_filter_factory=make_filter_factory(cfg))
    target = len(ml.trades)
    strict = None
    for z in np.arange(cfg.entry_z, cfg.stop_z - 0.25, 0.25):
        r = walk_forward(prices, cfg.with_(entry_z=float(z)), sectors)
        if len(r.trades) <= target:
            strict = (float(z), r)
            break
    out = {"rules_sharpe": sharpe(base.daily_ret), "rules_trades": len(base.trades),
           "ml_sharpe": sharpe(ml.daily_ret), "ml_trades": target}
    if strict:
        out.update({"matched_entry_z": strict[0], "matched_sharpe": sharpe(strict[1].daily_ret),
                    "matched_trades": len(strict[1].trades)})
    return out
