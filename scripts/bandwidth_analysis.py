"""Read the band x sampling-rate control by the rule of DEVIATION 29, addendum 1 (frozen before any result was read).

Cells (test accuracy per subject and arm):
  R  published restricted  (about 4-48 Hz, decim 4)   <published>/RESULTS_n14_COMPLETE/ablation_results.json
  F  published full        (4-125 Hz, decim 2)        <published>/results_hg/ablation_results.json
  A  new: lp48_decim2      (4-48 Hz, decim 2)         results/bandwidth_control/lp48_decim2/subj_XX.json
  B  new: full_decim4      DESCRIPTIVE ONLY: the decimator removes everything above about 60 Hz at decim 4
  F' repro_full_decim2, R' repro_lp48_decim4          pipeline checks (the published cells rerun by the new script)

Per arm, paired by subject:  T = F - R (total),  Bn = F - A (band, at 250 Hz),  Rt = A - R (rate, low-passed band);
T = Bn + Rt exactly. Reading: see `reading()`.

  python scripts/bandwidth_analysis.py [--published ../] [--new results/bandwidth_control] [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync.stats import sign_flip_p, t_ci  # noqa: E402

ARMS = ("nl_sync", "nl_power", "lin_sync", "lin_power", "eegnet", "shallow")
RN_ARMS = ARMS[:4]


def load_published(path):
    """list of {subject, arm: acc, ...} -> {subject: {arm: acc}}"""
    p = Path(path)
    if not p.exists():
        return {}
    return {int(r["subject"]): {a: float(r[a]) for a in ARMS if a in r} for r in json.loads(p.read_text())}


def load_new(folder):
    out = {}
    for f in sorted(Path(folder).glob("subj_[0-9]*.json")):
        r = json.loads(f.read_text())
        out[int(r["subject"])] = {a: float(v) for a, v in r["acc"].items()}
    return out


def diff(X, Y, arm):
    """Paired X - Y over the subjects that have `arm` in both cells."""
    subs = sorted(s for s in X if s in Y and arm in X[s] and arm in Y[s])
    return subs, np.array([X[s][arm] - Y[s][arm] for s in subs])


def stat(d):
    if len(d) < 3:
        return dict(n=int(len(d)), mean=float(np.mean(d)) if len(d) else None, ci95=None, p_signflip=None)
    return dict(n=int(len(d)), mean=float(d.mean()), ci95=[float(v) for v in t_ci(d, 0.95)], p_signflip=float(sign_flip_p(d)))


def excludes_zero(s):
    return s["ci95"] is not None and (s["ci95"][0] > 0 or s["ci95"][1] < 0)


def reading(T, Bn, Rt):
    """DEVIATION 29, addendum 1."""
    if T["ci95"] is None:
        return "too few subjects"
    if not T["ci95"][0] > 0:
        return "no gain to attribute"
    if excludes_zero(Bn) and Bn["mean"] >= 0.5 * T["mean"]:
        return "mostly band"
    if excludes_zero(Rt) and Rt["mean"] >= 0.5 * T["mean"] and not excludes_zero(Bn):
        return "mostly sampling rate"
    return "mixed or unresolved"


def decompose(R, F, A, arm):
    subs = sorted(s for s in R if s in F and s in A and all(arm in c[s] for c in (R, F, A)))
    if not subs:
        return None
    r, f, a = [np.array([c[s][arm] for s in subs]) for c in (R, F, A)]
    T, Bn, Rt = stat(f - r), stat(f - a), stat(a - r)
    return dict(subjects=subs, T=T, Bn=Bn, Rt=Rt, reading=reading(T, Bn, Rt))


def pipeline_check(pub, new):
    d = [new[s][a] - pub[s][a] for s in sorted(new) if s in pub for a in ARMS if a in new[s] and a in pub[s]]
    if not d:
        return None
    d = np.array(d)
    return dict(n_cells=int(len(d)), mean_signed=float(d.mean()), mean_abs=float(np.abs(d).mean()),
                comparable=bool(abs(d.mean()) <= 0.03))


def analyse(R, F, A, B=None, Fp=None, Rp=None, clean=None):
    out = dict(rule="DEVIATION 29, addendum 1", arms={}, pipeline_check={})
    for arm in ARMS:
        rec = dict(role="RN arm (read)" if arm in RN_ARMS else "reference", published=decompose(R, F, A, arm))
        if Rp:                                                   # the same reading inside the new pipeline (subset)
            rec["with_rerun_restricted"] = decompose(Rp, F, A, arm)
            pub = rec["published"]
            if pub and rec["with_rerun_restricted"]:
                rec["readings_agree"] = pub["reading"] == rec["with_rerun_restricted"]["reading"]
        if B:                                                    # descriptive: at most about 4-60 Hz at 125 Hz
            rec["B_descriptive"] = {k: stat(diff(X, Y, arm)[1]) for k, (X, Y) in
                                    {"B - R": (B, R), "F - B": (F, B)}.items() if len(diff(X, Y, arm)[1])}
        if clean:
            subs, d = diff(clean, R, arm)
            rec["ica_cleaned_minus_restricted"] = stat(d) if len(d) else None
        out["arms"][arm] = rec
    if Fp:
        out["pipeline_check"]["repro_full_decim2 - published full"] = pipeline_check(F, Fp)
    if Rp:
        out["pipeline_check"]["repro_lp48_decim4 - published restricted"] = pipeline_check(R, Rp)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default=str(ROOT.parent))
    ap.add_argument("--new", default=str(ROOT / "results" / "bandwidth_control"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pub, new = Path(a.published), Path(a.new)
    R = load_published(pub / "RESULTS_n14_COMPLETE" / "ablation_results.json")
    F = load_published(pub / "results_hg" / "ablation_results.json")
    clean = load_published(pub / "results_nonemg" / "ablation_results.json")
    A, B, Fp, Rp = [load_new(new / t) for t in ("lp48_decim2", "full_decim4", "repro_full_decim2", "repro_lp48_decim4")]
    if not A:
        print("No lp48_decim2 results in", new, "- nothing to read yet."); return None
    res = analyse(R, F, A, B, Fp, Rp, clean)
    out = Path(a.out) if a.out else new / "analysis.json"
    out.write_text(json.dumps(res, indent=1))
    fmt = lambda s: "n/a" if not s or s["ci95"] is None else f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
    for arm, rec in res["arms"].items():
        for key in ("published", "with_rerun_restricted"):
            d = rec.get(key)
            if d:
                print(f"{arm:10s} {key:22s} n={d['T']['n']:2d}  T {fmt(d['T'])}  band {fmt(d['Bn'])}  rate {fmt(d['Rt'])}  -> {d['reading']}")
    for k, v in res["pipeline_check"].items():
        if v:
            print(f"pipeline check, {k}: {v['n_cells']} cells, mean signed {v['mean_signed']:+.3f}, mean abs {v['mean_abs']:.3f}"
                  f"{'' if v['comparable'] else '  -> NOT COMPARABLE with the published cells (outside +/-0.03)'}")
    return res


if __name__ == "__main__":
    main()
