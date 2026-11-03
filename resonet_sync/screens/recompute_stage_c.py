"""DEVIATIONS 20 pre-commitment (item 2): the stage C process loaded decide() before the NaN fix (c43eaf7).
Recompute its verdict with the fixed decide() on the final per-dataset values and report both.
python -m resonet_sync.screens.recompute_stage_c"""
from __future__ import annotations

import json

from .screen10 import CELLS_C, ROOT, STAGES, _strict_json, decide


def main():
    d = ROOT / "results" / "screen10" / "stage_C"
    old = json.loads((d / "results.json").read_text())           # written by the running (pre-fix) process
    vals, sessions = old["per_dataset"], STAGES["C"][0]
    res, verdict = decide(vals, CELLS_C, sessions, "S_TE_al")
    res_v2, verdict_v2 = decide(vals, CELLS_C, sessions, "S_TE_v2", "S_inst_dn_v2")
    out = dict(protocol="DEVIATIONS.md #20 (pre-commitment item 2: recomputed with fixed decide)",
               te_statistic="S_TE_al", sessions=sessions, per_cell=res, verdict=verdict,
               verdict_as_written_by_run=old["verdict"], unmatched_v2=dict(per_cell=res_v2, verdict=verdict_v2),
               same_verdict=verdict["false_positives"] == old["verdict"]["false_positives"]
               and verdict["misses"] == old["verdict"]["misses"])
    (d / "results_recomputed.json").write_text(_strict_json(out, indent=1))
    print(f"{'cell':30s} dTE_al [CI] (n>0)                rel dInst [CI]        dfc    dR     fires should | v2")
    for c, r in res.items():
        print(f"{c:30s} {r['dTE']:+.4f} [{r['dTE_ci'][0]:+.4f},{r['dTE_ci'][1]:+.4f}] ({r['dTE_pos']:2d})  "
              f"{r['rel']:+.3f} [{r['rel_ci'][0]:+.3f},{r['rel_ci'][1]:+.3f}]  {r['d_fc']:+.2f} {r['d_R']:+.3f}  "
              f"{str(r['fires']):5s} {str(r['should_fire']):5s} | {res_v2[c]['fires']}")
    print("\nVERDICT (fixed decide):", json.dumps(verdict))
    print("VERDICT (as written by the run):", json.dumps(old["verdict"]))
    print("VERDICT v2 unmatched (fixed decide):", json.dumps(verdict_v2))


if __name__ == "__main__":
    main()
