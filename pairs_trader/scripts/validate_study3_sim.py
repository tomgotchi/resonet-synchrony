"""Study 3 gate 0' (pre-registered in docs/STUDY3.md). Fresh seeds 20261120+i, straddle_null kind, P5a/P5b."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairs_trader.coint import cost_filter_stat                                # noqa: E402
from pairs_trader.simulate2 import NULL_KINDS, simulate_study2_panel           # noqa: E402
from pairs_trader.strategy import select_pairs                                 # noqa: E402
from validate_study2_sim import CFG, THR, fdr, wilson_lo                       # noqa: E402

SEED0 = 20261120


def one(i):
    p = simulate_study2_panel(SEED0 + i, straddle_per_sector=4)
    f = p.prices.iloc[:504]
    planted = {(r.a, r.b): (r.kind, r.bounce_bp) for r in p.truth.itertuples()}
    lf = np.log(f)
    out = {}
    for name, cfg in (("filter", CFG), ("nofilter", CFG.with_(cost_filter_m=0.0))):
        _, tab = select_pairs(f, cfg, p.sectors, return_table=True)
        info = [planted.get((a, b), planted.get((b, a), ("cross", 2.0))) for a, b in zip(tab.a, tab.b)]
        tab["kind"] = [k for k, _ in info]
        tab["bounce"] = [bb for _, bb in info]
        out[name] = tab[["a", "b", "kind", "bounce", "p", "cost_ok", "fdr_pass"]].assign(seed=SEED0 + i)
    out["filter"]["stat"] = [cost_filter_stat(lf[a].to_numpy(), lf[b].to_numpy())
                             for a, b in zip(out["filter"].a, out["filter"].b)]
    return out


def main(n_panels=200, out_path="results/study3_gate0.json"):
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, range(n_panels)))
    F = pd.concat([r["filter"] for r in res])
    N = pd.concat([r["nofilter"] for r in res])
    clones, coint = F[F.kind == "clone"], F[F.kind == "coint"]
    sel = F[F.fdr_pass]
    fdr_f, fdr_se = fdr(F)
    g = {"P1_clones_removed": {"rate": float((~clones.cost_ok).mean()), "n": len(clones),
                               "wilson_lo": wilson_lo(int((~clones.cost_ok).sum()), len(clones))},
         "P2_coint_kept": {"rate": float(coint.cost_ok.mean()), "n": len(coint),
                           "wilson_lo": wilson_lo(int(coint.cost_ok.sum()), len(coint))},
         "P3_fdr": {"fdr": fdr_f, "mc_se": fdr_se, "bound": CFG.fdr_q + 2 * fdr_se},
         "P4_selected_below_threshold": int((sel.stat < THR).sum())}
    p5a = {}
    for k in ("null", "cross", "straddle_null"):
        kept = F[(F.kind == k) & F.cost_ok]
        n = len(kept)
        r = float((kept.p < 0.05).mean()) if n else float("nan")
        bound = 0.05 + 2 * np.sqrt(0.05 * 0.95 / n) if n else float("nan")
        p5a[k] = {"n_kept": n, "rate": r, "bound": float(bound), "gated": n >= 1000, "ok": bool(n < 1000 or r <= bound)}
    g["P5a_kept_null_size"] = p5a
    st = F[F.kind == "straddle_null"]
    tables, strata = [], {}
    for bb, s in st.groupby("bounce"):
        t = np.array([[((s.cost_ok) & (s.p < 0.05)).sum(), ((s.cost_ok) & (s.p >= 0.05)).sum()],
                      [((~s.cost_ok) & (s.p < 0.05)).sum(), ((~s.cost_ok) & (s.p >= 0.05)).sum()]], float)
        strata[str(bb)] = {"pass_rate": float(t[0, 0] / max(t[0].sum(), 1)), "fail_rate": float(t[1, 0] / max(t[1].sum(), 1)),
                           "n_pass": int(t[0].sum()), "n_fail": int(t[1].sum())}
        tables.append(t + 0.5)                        # Haldane correction against empty cells
    cmh = StratifiedTable(tables).test_null_odds(correction=True)
    n_pass, n_fail = int(st.cost_ok.sum()), int((~st.cost_ok).sum())
    g["P5b_straddle_independence"] = {"cmh_p": float(cmh.pvalue), "n_pass": n_pass, "n_fail": n_fail, "strata": strata,
                                      "pooled_diff": float((st[st.cost_ok].p < 0.05).mean() - (st[~st.cost_ok].p < 0.05).mean())}
    g["pass"] = {"P1": g["P1_clones_removed"]["wilson_lo"] >= 0.90, "P2": g["P2_coint_kept"]["wilson_lo"] >= 0.95,
                 "P3": fdr_f <= CFG.fdr_q + 2 * fdr_se, "P4": g["P4_selected_below_threshold"] == 0,
                 "P5a": all(v["ok"] for v in p5a.values()) and any(v["gated"] for v in p5a.values()),
                 "P5b": n_pass >= 1000 and n_fail >= 1000 and cmh.pvalue >= 0.05}
    g["all_pass"] = bool(all(g["pass"].values()))
    nsel = N[N.fdr_pass]
    cn = F[F.kind == "clone_null"]
    g["reported"] = {"clone_null_rejection_all": float((cn.p < 0.05).mean()),
                     "clone_null_selected_filter": int(sel.kind.eq("clone_null").sum()),
                     "clone_null_selected_nofilter": int(nsel.kind.eq("clone_null").sum()),
                     "fdr_nofilter": fdr(N)[0],
                     "null_share_selected_filter": float(sel.kind.isin(NULL_KINDS).mean()),
                     "null_share_selected_nofilter": float(nsel.kind.isin(NULL_KINDS).mean()),
                     "selected_by_kind_filter": sel.kind.value_counts().to_dict(),
                     "selected_by_kind_nofilter": nsel.kind.value_counts().to_dict(),
                     "straddle_rejection_by_stat_decile": st.assign(dec=pd.qcut(st.stat, 10, labels=False))
                     .groupby("dec").p.apply(lambda x: float((x < 0.05).mean())).to_dict()}
    conv = lambda o: bool(o) if isinstance(o, np.bool_) else (int(o) if isinstance(o, np.integer) else float(o))
    Path(out_path).write_text(json.dumps(g, indent=1, default=conv))
    print(json.dumps(g, indent=1, default=conv))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, sys.argv[2] if len(sys.argv) > 2 else "results/study3_gate0.json")
