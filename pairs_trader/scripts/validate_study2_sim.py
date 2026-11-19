"""Study 2 gate 0 (pre-registered in docs/STUDY2.md): the cost filter on synthetic panels with known truth.
Usage: python scripts/validate_study2_sim.py [n_panels] [out.json]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairs_trader.coint import cost_filter_stat                       # noqa: E402
from pairs_trader.config import StrategyConfig                        # noqa: E402
from pairs_trader.simulate2 import NULL_KINDS, TRUE_KINDS, simulate_study2_panel   # noqa: E402
from pairs_trader.strategy import select_pairs                        # noqa: E402

CFG = StrategyConfig().with_(study=2, cost_filter_m=1.0, integer_shares=True, calibrate=False)
THR = CFG.cost_filter_m * 2 * CFG.screen_cost_bps / 1e4


def one(seed):
    p = simulate_study2_panel(seed)
    f = p.prices.iloc[:504]
    planted = {(r.a, r.b): r.kind for r in p.truth.itertuples()}
    out = {}
    for name, cfg in (("filter", CFG), ("nofilter", CFG.with_(cost_filter_m=0.0))):
        _, tab = select_pairs(f, cfg, p.sectors, return_table=True)
        tab["kind"] = [planted.get((a, b), planted.get((b, a), "cross")) for a, b in zip(tab.a, tab.b)]
        out[name] = tab[["a", "b", "kind", "p", "cost_ok", "fdr_pass"]].assign(seed=seed)
    lf = np.log(f)
    out["filter"]["stat"] = [cost_filter_stat(lf[a].to_numpy(), lf[b].to_numpy())
                             for a, b in zip(out["filter"].a, out["filter"].b)]
    return out


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return float("nan")
    ph = k / n
    return (ph + z * z / (2 * n) - z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def fdr(tab):
    per = []
    for _, g in tab.groupby("seed"):
        d = g[g.fdr_pass & g.kind.isin(NULL_KINDS | TRUE_KINDS)]
        per.append((d.kind.isin(NULL_KINDS)).sum() / max(len(d), 1))
    per = np.array(per)
    return float(per.mean()), float(per.std(ddof=1) / np.sqrt(len(per)))


def main(n_panels=200, out_path="results/study2_gate0.json"):
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, range(n_panels)))
    F = pd.concat([r["filter"] for r in res])
    N = pd.concat([r["nofilter"] for r in res])
    clones, coint = F[F.kind == "clone"], F[F.kind == "coint"]
    nulls = F[F.kind.isin(NULL_KINDS)]
    fp, ff = nulls[nulls.cost_ok], nulls[~nulls.cost_ok]
    r_pass, r_fail = (fp.p < 0.05).mean(), (ff.p < 0.05).mean()
    se = np.sqrt(r_pass * (1 - r_pass) / max(len(fp), 1) + r_fail * (1 - r_fail) / max(len(ff), 1))
    fdr_f, fdr_se = fdr(F)
    sel = F[F.fdr_pass]
    g = {
        "P1_clones_removed": {"rate": float((~clones.cost_ok).mean()), "n": len(clones),
                              "wilson_lo": wilson_lo(int((~clones.cost_ok).sum()), len(clones))},
        "P2_coint_kept": {"rate": float(coint.cost_ok.mean()), "n": len(coint),
                          "wilson_lo": wilson_lo(int(coint.cost_ok.sum()), len(coint))},
        "P3_fdr": {"fdr": fdr_f, "mc_se": fdr_se, "bound": CFG.fdr_q + 2 * fdr_se},
        "P4_selected_below_threshold": int((sel.stat < THR).sum()),
        "P5_null_rejection": {"pass": float(r_pass), "fail": float(r_fail), "n_pass": len(fp), "n_fail": len(ff),
                              "diff_se": float(se)},
    }
    g["pass"] = {
        "P1": g["P1_clones_removed"]["wilson_lo"] >= 0.90,
        "P2": g["P2_coint_kept"]["wilson_lo"] >= 0.95,
        "P3": fdr_f <= CFG.fdr_q + 2 * fdr_se,
        "P4": g["P4_selected_below_threshold"] == 0,
        "P5": len(ff) >= 1000 and abs(r_pass - r_fail) <= 2 * se,
    }
    g["all_pass"] = bool(all(g["pass"].values()))
    nsel = N[N.fdr_pass]
    g["P6_reported"] = {
        "fdr_nofilter": fdr(N)[0],
        "selected_by_kind_filter": sel.kind.value_counts().to_dict(),
        "selected_by_kind_nofilter": nsel.kind.value_counts().to_dict(),
        "null_share_selected_filter": float(sel.kind.isin(NULL_KINDS).mean()),
        "null_share_selected_nofilter": float(nsel.kind.isin(NULL_KINDS).mean()),
        "untradable_share_selected_nofilter": float(nsel.kind.isin({"clone", "clone_null"}).mean()),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(g, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o)))
    print(json.dumps(g, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else int(o)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, sys.argv[2] if len(sys.argv) > 2 else "results/study2_gate0.json")
