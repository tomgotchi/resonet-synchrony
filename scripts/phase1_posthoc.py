"""Phase 1 post-hoc analyses on the SAVED trained banks (DEVIATION 23): F5 (phase beyond band power), the D2
probes and the standard baselines B1-B3. Needs the HGD cache and results/phase1/weights/ (written by run.py from
2026-11-09 on). CPU is enough (Colab).

  python scripts/phase1_posthoc.py --subjects 1-14 [--cache DIR] [--no-tangent]
Writes results/phase1/posthoc/subj_XX.json and results/phase1/posthoc/summary.json. Inference (F5, Holm m = 2)
only when all 14 subjects are present; the baseline B for F5 is the one DEVIATION 24 selected
(results/calib_p1/results.json -> decisions.F5_baseline); both candidates are always reported."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resonet_sync import baselines as BL, data_hgd  # noqa: E402
from resonet_sync.run import (CONFIRMATORY_BANKS, N_SUBJECTS, P1, build, load_config, provenance,  # noqa: E402
                              write_json_atomic)
from resonet_sync.stats import contrast_table  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-14")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-tangent", action="store_true")
    a = ap.parse_args(argv)
    cfg = dict(P1)
    cache = a.cache or load_config().get("paths", {}).get("hgd_cache_hg")
    p1 = ROOT / "results" / "phase1"
    out = p1 / "posthoc"; out.mkdir(parents=True, exist_ok=True)
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2]
    fs = data_hgd.FS_NATIVE / cfg["decim"]
    for s in range(int(lo), int(hi) + 1):
        f = out / f"subj_{s:02d}.json"
        wdir = p1 / "weights" / f"subj_{s:02d}"
        rec = json.loads(f.read_text()) if f.exists() else dict(subject=s, banks={})
        todo = [b for b in CONFIRMATORY_BANKS if b not in rec["banks"] and (wdir / f"{b}__sync_plus_power.pt").exists()]
        if not todo and "baselines" in rec:
            continue
        Xtr, ytr, Xte, yte = data_hgd.load(cache, s, cfg["decim"], cfg["cap"], cfg["seed"])
        vmask = data_hgd.train_val_split(ytr, cfg["val_frac"], cfg["seed"])
        if "baselines" not in rec:
            rec["baselines"] = BL.baselines(Xtr, ytr, Xte, yte, vmask, fs, tangent=not a.no_tangent)
        rec.setdefault("provenance", []).append(provenance(torch.device("cpu")))
        for b in todo:
            wf = wdir / f"{b}__sync_plus_power.pt"
            m = build(f"{b}:sync+power", Xtr.shape[1], Xtr.shape[2], fs, cfg)
            m.load_state_dict(torch.load(wf))
            rec["banks"][b] = BL.frozen_bank_analyses(m, Xtr, ytr, Xte, yte, vmask, fs, cfg["n_osc"], cfg["seed"])
            rec["banks"][b]["weights_sha256"] = data_hgd.sha256(wf)        # traceable to the exact trained bank
        write_json_atomic(f, rec)
        print(s, json.dumps(rec)[:400], flush=True)
    per = [json.loads(p.read_text()) for p in sorted(out.glob("subj_[0-9]*.json"))]
    rows = [{f"{b}:{k}": v for b, r in p["banks"].items() for k, v in r.items() if k != "weights_sha256"} for p in per]
    fam = {B: contrast_table(rows, [(f"{b}: ({B}+sync)-({B}+surrogate_random)", f"{b}:{B}+sync",
                                     f"{b}:{B}+surrogate_random") for b in CONFIRMATORY_BANKS],
                             min_n=N_SUBJECTS, phase1=True) for B in ("FBCSP", "FBLV")}
    calib = ROOT / "results" / "calib_p1" / "results.json"
    chosen = json.loads(calib.read_text()).get("decisions", {}).get("F5_baseline") if calib.exists() else None
    summ = dict(n_subjects=len(per), F5_baseline_from_DEVIATION_24=chosen,
                F5_confirmatory=fam.get(chosen) if chosen else None, F5_both_candidates=fam,
                baselines_mean={k: float(np.mean([p["baselines"][k] for p in per if p["baselines"].get(k) is not None]))
                                for k in per[0]["baselines"]} if per else {},
                note_D2="D2 probes are fitted on train minus the carve and scored on TEST (DEVIATION 23 says train-CV; "
                        "corrected wording in DEVIATIONS, board round 3)",
                D2_probe_means={f"{b}:{k}": float(np.mean([p["banks"][b][k] for p in per if b in p["banks"]]))
                                for b in CONFIRMATORY_BANKS for k in ("probe:sync", "probe:surrogate", "probe:surrogate_random")
                                if any(b in p["banks"] for p in per)})
    write_json_atomic(out / "summary.json", json.loads(json.dumps(summ, default=float)))
    print(json.dumps(summ, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
