"""Figures for the synthetic mechanism-statistic programme (screens 7-10, diagnostics, review board).
python scripts/make_figures.py [--diag DIR]  -> results/figures/*.png  (diagnostics default: results/screen10/*.json)

Palette: reference categorical slots 1-3 (validated light-mode), status colours only with icon + label."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resonet_sync.stats import boot_ci  # noqa: E402

OUT = ROOT / "results" / "figures"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"     # status (always with icon + label)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "font.family": "DejaVu Sans", "ytick.major.size": 0})


def _load(p):
    return json.loads((ROOT / p).read_text())


def _title(fig, title, sub):
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, sub, ha="left", va="top", fontsize=9.5, color=INK2)


# ------------------------------------------------------------------ 1. nuisance traps v2 -> v3 -> v3r
CELLS = [("F1_freq_up", "Theta +1 Hz", False), ("F2_freq_down", "Theta -1 Hz", False),
         ("N1_snr_up", "SNR 10 vs 0 dB", False), ("N2_snr_down", "SNR 0 vs 10 dB", False),
         ("C1_coupling_up", "Coupling K 40 vs 25", True)]


def fig_traps(diag_b, stage_c=None):
    s9 = _load("results/screen9/screen9_results.json")["per_cell"]
    sd = _load("results/screen10/stage_D/results.json")["per_cell"]
    s9["C1_coupling_up"] = s9.pop("C1_coupling_up_control")
    dev = {}
    if diag_b is not None:
        for cell, _, o in diag_b:
            dev.setdefault(cell, []).append(o["A"]["relband_fix_rot"] - o["B"]["relband_fix_rot"])
    rows = [("Rule v2 (screen 9)", S1, s9), ("Rule v3 (stage D)", S2, sd)]
    if stage_c:
        rows.append(("Rule v3r (stage C, fresh sessions)", S3, stage_c))
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.4), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.74, bottom=0.2, wspace=0.28)
    _title(fig, "Nuisance traps: does the rule fire when only a nuisance differs?",
           "Mean change in transfer entropy (A - B) with 95% bootstrap CI over 10 sessions.  "
           "Filled = rule fired, hollow = did not fire.\nGrey diamond = development check of the v3r frequency fix (TE leg only).")
    labels = [r[0] for r in rows] + (["Rule v3r (development, TE leg)"] if dev else [])
    for ax, (key, name, should) in zip(axes, CELLS):
        ax.axvline(0, color=MUTED, lw=1)
        for i, (lab, col, src) in enumerate(rows):
            r = src.get(key)
            if r is None:
                continue
            y = len(labels) - 1 - i
            ax.plot(r["dTE_ci"], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["dTE"], y, "o", ms=8, mec=col, mew=2, mfc=col if r["fires"] else SURF, zorder=3)
            ok = r["correct"]
            ax.text(1.04, y, ("✓" if ok else "✗"), transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=11, fontweight="bold", color=GOOD if ok else CRIT, clip_on=False)
        if dev and key in dev:
            d = np.array(dev[key]); ci = boot_ci(d); y = len(labels) - 1 - len(rows)
            ax.plot(ci, [y, y], color=MUTED, lw=2)
            ax.plot(d.mean(), y, "D", ms=6, color=MUTED, mfc=SURF, mew=1.8)
        ax.set_title(f"{name}\n{'must fire' if should else 'must NOT fire'}", fontsize=10,
                     color=INK if should else INK2)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1000:+.1f}" if abs(v) > 1e-9 else "0"))
    axes[0].set_yticks(range(len(labels)), labels[::-1])
    axes[2].set_xlabel("Change in S_TE (millibits)")
    fig.text(0.2, 0.04, "✓ / ✗ = cell correct / wrong under that rule.  Sessions 80-89 for v2, v3 and development; "
             "fresh sessions 90-99 for stage C.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig1_nuisance_traps.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 2. frequency-fix diagnostic
VARS = [("fixband_fixdelay", "a", "Fixed 6-10 Hz band, 32 ms delay (v2)"),
        ("fixband_cycdelay", "a", "Fixed band, delay in cycles (v3)"),
        ("relband_fixdelay", "a", "Centred band, 32 ms delay"),
        ("relband_cycdelay", "a", "Centred band, delay in cycles"),
        ("fixband_fix_rot", "b", "Fixed band, 32 ms, phase rotation"),
        ("fixband_fix_dith4", "b", "Fixed band, 32 ms, bin dither (invalid test)"),
        ("relband_fix_rot", "b", "Centred band, 32 ms, phase rotation (v3r)")]


def fig_freq(diag_a, diag_b):
    src = {"a": diag_a, "b": diag_b}
    cols = [("F1_freq_up", "Theta +1 Hz  (want 0)"), ("F2_freq_down", "Theta -1 Hz  (want 0)"),
            ("C1_coupling_up", "Coupling up  (want > 0)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.99, top=0.76, bottom=0.14, wspace=0.1)
    _title(fig, "Which estimator change removes the theta-frequency confound?",
           "Development diagnostic on used sessions 80-89, TE leg only.  Mean change in S_TE (A - B) with 95% "
           "bootstrap CI.  Only centred band + phase rotation is near zero for both shifts.")
    n = len(VARS)
    for ax, (cell, name) in zip(axes, cols):
        ax.axvline(0, color=MUTED, lw=1)
        for i, (v, s, lab) in enumerate(VARS):
            d = np.array([o["A"][v] - o["B"][v] for c, _, o in src[s] if c == cell])
            ci = boot_ci(d); y = n - 1 - i
            chosen = v == "relband_fix_rot"
            col = S2 if chosen else S1
            ax.plot(ci, [y, y], color=col, lw=2.2 if chosen else 1.8)
            ax.plot(d.mean(), y, "o", ms=8 if chosen else 6.5, color=col, mec=SURF, mew=1)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1000:+.0f}" if v else "0"))
    axes[0].set_yticks(range(n), [lab for _, _, lab in VARS][::-1])
    for t in axes[0].get_yticklabels():
        if "v3r" in t.get_text():
            t.set_fontweight("bold"); t.set_color(INK)
    axes[1].set_xlabel("Change in S_TE (millibits)")
    fig.savefig(OUT / "fig2_frequency_diagnostic.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 3. screen history
def fig_history(stage_c_verdict=None):
    screens = [("Screen 7\nrule v1, 20 dB", "results/screen7/screen7_results.json"),
               ("Screen 8\nrule v2, 0 dB", "results/screen8/screen8_results.json"),
               ("Screen 9\nrule v2, nuisance traps", "results/screen9/screen9_results.json"),
               ("Screen 10 D\nrule v3, nuisance traps", "results/screen10/stage_D/results.json")]
    rows = []
    for lab, p in screens:
        pc = _load(p)["per_cell"].values()
        rows.append((lab, sum(r["correct"] for r in pc), sum(r["fires"] and not r["should_fire"] for r in pc),
                     sum((not r["fires"]) and r["should_fire"] for r in pc)))
    if stage_c_verdict:
        rows.append(stage_c_verdict)
    H = 0.75 * len(rows) + 2.0
    fig, ax = plt.subplots(figsize=(10, H))
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 1.25 / H, bottom=0.55 / H)
    _title(fig, "Within-session rule: cells correct per screen",
           "Each cell is a 10-session contrast that must or must not fire.  Sessions are fresh in every screen.")
    for i, (lab, ok, fp, miss) in enumerate(rows):
        y = len(rows) - 1 - i; x = 0
        for n, col, name in ((ok, GOOD, "✓ correct"), (fp, CRIT, "✗ false positive"), (miss, WARN, "! miss")):
            if n:
                ax.barh(y, n - 0.06, left=x + 0.03, height=0.56, color=col)
                ax.text(x + n / 2, y, f"{n}", ha="center", va="center", color="white" if col != WARN else INK,
                        fontsize=10, fontweight="bold")
                x += n
        ax.text(x + 0.15, y, f"{ok}/{ok + fp + miss}", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("Number of cells")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r[1] + r[2] + r[3] for r in rows) + 1.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, CRIT, WARN)]
    ax.legend(handles, ["✓ correct", "✗ false positive", "! miss"], loc="lower right", ncol=3,
              bbox_to_anchor=(1, 1.0), fontsize=9)
    fig.savefig(OUT / "fig3_screen_history.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 4. integrator artefact
def fig_integrator():
    lam, dt = -2 * np.pi, 1e-3
    f = np.linspace(4, 12, 81)
    euler = np.log(np.abs(1 + (lam + 1j * 2 * np.pi * f) * dt)) / dt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(left=0.12, right=0.8, top=0.78, bottom=0.14)
    _title(fig, "Why the frequency traps also changed the physics",
           "Effective damping per unit in the simulator (dt = 1 ms).  Euler weakens damping by ω²dt/2.")
    ax.plot(f, euler, color=S2, lw=2)
    ax.plot(f, np.full_like(f, lam), color=S1, lw=2)
    ax.text(12.1, euler[-1], "Euler-Maruyama\n(screens 1-10)", color=INK, va="center", fontsize=9)
    ax.text(12.1, lam, "Exact integrator\n(= nominal λ)", color=INK, va="center", fontsize=9)
    for fx in (7, 8, 9):
        v = np.log(abs(1 + (lam + 1j * 2 * np.pi * fx) * dt)) / dt
        ax.plot(fx, v, "o", color=S2, ms=7, mec=SURF)
        ax.annotate(f"{v:.2f}", (fx, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK2)
    ax.set_xlabel("Unit natural frequency (Hz)")
    ax.set_ylabel("Effective damping (1/s)")
    fig.savefig(OUT / "fig4_integrator_damping.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 5. example traces
def fig_traces():
    from resonet_sync import synth
    regimes = [("Coupling only  (K 40, no common input)", dict(K=40.0, rho=0.0)),
               ("White common input only  (K 0, ρ 0.85)", dict(K=0.0, rho=0.85)),
               ("Rhythmic common input only  (K 0, κ_nb 12)", dict(K=0.0, rho=0.0, kappa_nb=12.0))]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.86, bottom=0.09, hspace=0.45)
    _title(fig, "What the synthetic regimes look like",
           "Three of eight units, one trial, 1 s at 250 Hz, z-scored, no observation noise.  Same session template (seed 5).")
    t = np.arange(250) / 250.0
    for ax, (name, p) in zip(axes, regimes):
        X = synth.make_dataset(5, "mixed", 1, 1, dict(p, obs_snr_db=None))[0][0]
        for j, col in enumerate((S1, S2, S3)):
            ax.plot(t, X[j] - 3.2 * j, color=col, lw=1.4)
            ax.text(1.01, -3.2 * j, f"unit {j + 1}", color=INK2, va="center", fontsize=8.5,
                    transform=ax.get_yaxis_transform())
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_yticks([]); ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(OUT / "fig5_example_traces.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 6. review board
BOARD = [("Halvorsen · statistics", 2, 4, 1), ("Echeverría · hippocampus", 3, 5, 6), ("Zhao · information theory", 2, 3, 2),
         ("Volkov · dynamics", 2, 2, 3), ("Asante · closed-loop DSP", 3, 7, 4), ("Novak · code audit", 0, 3, 5)]


def fig_board():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.2)
    _title(fig, "Review board, round 1: findings by severity",
           "Six independent reviewer personas.  All code defects verified by the judge were fixed in commits c43eaf7 / 2d1924b.")
    for i, (name, c, M, m) in enumerate(BOARD):
        y = len(BOARD) - 1 - i; x = 0
        for n, col, lab in ((c, CRIT, "critical"), (M, "#ec835a", "major"), (m, "#c9c7c0", "minor")):
            if n:
                ax.barh(y, n - 0.08, left=x + 0.04, height=0.56, color=col)
                ax.text(x + n / 2, y, str(n), ha="center", va="center", fontsize=9.5, fontweight="bold",
                        color="white" if col == CRIT else INK)
                x += n
    ax.set_yticks(range(len(BOARD)), [b[0] for b in BOARD][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of findings")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CRIT, "#ec835a", "#c9c7c0")]
    ax.legend(handles, ["▲ critical", "● major", "○ minor"], loc="lower right", ncol=3, bbox_to_anchor=(1, 1.0))
    fig.savefig(OUT / "fig6_review_board.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 7. board round 2: the drift statistic
# Numbers reported by reviewer A. Volkov in round 2 (scratchpad dd2.py/dd3.py; mixed, exact integrator, seed 80,
# 10 dB, 50 trials, unfiltered Hilbert, lag 3). Reviewer-run check, NOT a frozen repo result.
DRIFT = [("Independent units", 0.88, False), ("White common input, ρ 0.7", 1.47, False),
         ("Rhythmic common input, κ 20", 1.57, False), ("Rhythmic common input, κ 40", 2.80, False),
         ("Coupling K 25", 3.89, True), ("Coupling K 40", 5.49, True)]


def fig_drift():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.74, bottom=0.14)
    _title(fig, "Board round 2: rhythmic common input leaks into the 'coupling' statistic",
           "Off-diagonal drift S_drift (Volkov's proposal) on synthetic networks.  White common input stays out "
           "of drift;\nrhythmic common input with NO coupling climbs toward the coupled range.  Reviewer-run check, "
           "not a frozen result.")
    for i, (lab, v, cpl) in enumerate(DRIFT):
        y = len(DRIFT) - 1 - i
        col = S1 if cpl else S2
        ax.barh(y, v, height=0.56, color=col)
        ax.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(DRIFT)), [d[0] for d in DRIFT][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("S_drift (1/s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2)]
    ax.legend(handles, ["coupling present", "no coupling"], loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.savefig(OUT / "fig7_board_round2_drift.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 8. screen 11 terminal calibration
def fig_screen11():
    rp = ROOT / "results" / "screen11" / "report.json"
    if not rp.exists():
        return
    R = json.loads(rp.read_text())
    cells = R["cells"]
    names = list(cells)
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(names) + 2.2))
    fig.subplots_adjust(left=0.3, right=0.95, top=1 - 1.25 / (0.36 * len(names) + 2.2), bottom=0.6 / (0.36 * len(names) + 2.2))
    v = R["verdict"]
    _title(fig, "Screen 11: how often does rule v3r fire? (terminal calibration)",
           f"P(fire) per cell; full rule where run (tier F), otherwise the TE-leg upper bound (tier T).  "
           f"Effective alpha {v['effective_alpha']:.2f}; retire rule: {'YES' if v['retire_TE_LDS_rule'] else 'no'}.")
    for i, c in enumerate(names):
        r = cells[c]; y = len(names) - 1 - i
        col = S1 if r["should_fire"] else S2
        tier_f = r.get("p_rule") is not None
        ax.barh(y, r["p_fire"], height=0.6, color=col if tier_f else SURF, edgecolor=col, linewidth=1.5)
        ok = (r["p_fire"] >= 0.8) if r["should_fire"] else (r["p_fire"] <= 0.05)
        ax.text(min(r["p_fire"], 1.0) + 0.015, y, f"{r['p_fire']:.2f} {'✓' if ok else '✗'}", va="center", fontsize=9,
                color=GOOD if ok else CRIT)
    ax.axvline(0.05, color=CRIT, lw=1, ls="--"); ax.axvline(0.8, color=GOOD, lw=1, ls="--")
    ax.set_yticks(range(len(names)), names[::-1]); ax.set_xlim(0, 1.15); ax.grid(axis="y", visible=False)
    ax.set_xlabel("P(fire)   (dashed: 0.05 false-positive limit, 0.8 power target)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=S1), plt.Rectangle((0, 0), 1, 1, color=S2),
               plt.Rectangle((0, 0), 1, 1, facecolor=SURF, edgecolor=INK2)]
    ax.legend(handles, ["must fire (filled: full rule)", "must NOT fire (filled: full rule)", "hollow: TE-leg bound only"],
              loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, fontsize=8.5)
    fig.savefig(OUT / "fig8_screen11_calibration.png", dpi=150)
    plt.close(fig)


def fig_calib_p1():
    rp = ROOT / "results" / "calib_p1" / "results.json"
    if not rp.exists():
        return
    R = json.loads(rp.read_text())
    summ, dec = R["summary"], R.get("decisions", {})
    gens = [("G0", "G0: no information"), ("G1", "G1: source power only"), ("G1r", "G1r: one source's power"),
            ("G2", "G2: coupling only")]
    names = list(summ["G0"]["contrasts"])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.975, top=0.76, bottom=0.2, wspace=0.12)
    _title(fig, "Calibration of the Phase 1 contrasts (DEVIATION 24): which contrast measures phase?",
           "Mean over 10 synthetic datasets with 90% t-CI; grey band +/-0.05. "
           f"Pre-committed outcome: no contrast passes all four panels -> no Phase 1 result is called 'phase'.")
    for ax, (g, label) in zip(axes, gens):
        ax.axvspan(-0.05, 0.05, color=GRID, alpha=0.7, lw=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        for i, c in enumerate(names):
            v = summ[g]["contrasts"][c]; y = len(names) - 1 - i
            if g == "G2":
                ok = v["sensitive"]; col = GOOD if ok else CRIT
            else:
                ok = v["valid"]; col = GOOD if ok else CRIT
            ax.plot(v["ci90"], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.plot(v["mean"], y, "o", color=col, ms=5)
            ax.text(1.02, y, "✓" if ok else "✗", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=col, fontsize=10, clip_on=False)
        ax.set_title(label, fontsize=9.5, loc="left")
        ax.set_xlim(-0.25, 0.45); ax.grid(axis="y", visible=False)
        ax.set_xlabel("difference in balanced accuracy")
    axes[0].set_yticks(range(len(names)), names[::-1], fontsize=8.5)
    fig.text(0.25, 0.025, "✓ = valid (G0, G1, G1r: CI inside +/-0.05) or sensitive (G2: p < 0.05 and mean >= 0.05).  "
             "A contrast may be called 'phase' only with ✓ in all four panels.", fontsize=8.5, color=INK2)
    fig.savefig(OUT / "fig9_calibration_phase1_contrasts.png", dpi=150)
    plt.close(fig)


def fig_v2():
    """Fig 10: the critique in one picture, and ResoNet v2 (docs/CRITIQUE_AND_V2.md)."""
    fx, pn = ROOT / "results" / "published_forensics.json", ROOT / "results" / "published_null" / "results.json"
    if not fx.exists():
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.10, wspace=0.24, hspace=0.42)
    _title(fig, "ResoNet: where the oscillators are, what the published contrast measures, and v2",
           "(a) published checkpoints; (b) published code on simulated data with NO phase information; "
           "(c) v2's lag block on four simulated worlds; (d) v2 on real EEG (development folds).")
    # (a) frequencies
    ax = axes[0, 0]
    F = json.loads(fx.read_text())
    ax.axvspan(4, 125, color=GRID, alpha=0.8, lw=0)
    ax.text(22, 3.62, "EEG fed to the model: 4-125 Hz", ha="center", fontsize=8.5, color=INK2)
    rows = [("published model\n(trained, 14 subjects)", np.array(F.get("trained_f0_hz_all") or F["trained_f0_hz"]), None, CRIT),
            ("Hz bank (Phase 1)", np.geomspace(1, 40, 16), 0.28, S2),
            ("v2 (synthetic study)", np.geomspace(4, 40, 8), "q", S1), ("v2 (HGD config)", np.geomspace(4, 100, 12), "q", S3)]
    for i, (lab, f0, bw, col) in enumerate(rows):
        y = len(rows) - 1 - i
        if bw == "q":
            r = (f0[-1] / f0[0]) ** (1 / (len(f0) - 1)); h = f0 * (r - 1) / 2
            for f, w in zip(f0, h):
                ax.plot([f - w, f + w], [y, y], color=col, lw=5, alpha=0.45, solid_capstyle="butt")
        ax.plot(f0, np.full(len(f0), y), "|", color=col, ms=13, mew=1.6 if len(f0) < 40 else 0.6, alpha=1 if len(f0) < 40 else 0.45)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8.5); ax.set_ylim(-0.6, 3.9)
    ax.set_xscale("log"); ax.set_xlim(0.15, 140); ax.set_xticks([0.2, 0.5, 1, 2, 4, 10, 20, 40, 100], ["0.2", "0.5", "1", "2", "4", "10", "20", "40", "100"])
    ax.set_xlabel("oscillator frequency (Hz); shaded bars = -3 dB width"); ax.grid(axis="y", visible=False)
    ax.set_title("(a) The published oscillators sit below the signal they are fed", loc="left", fontsize=10)
    # (b) published null test
    ax = axes[0, 1]
    if pn.exists():
        P = json.loads(pn.read_text())
        rs = [r for r in P["rows"] if r["gen"] == "G1"]
        arms = [("power", "power", S3), ("sync", "sync\n(\"phase\")", S1), ("surrogate", "surrogate\n(phase destroyed)", S2)]
        for j, (k, lab, col) in enumerate(arms):
            v = np.array([r["arms"][k]["bal_acc"] for r in rs])
            ax.bar(j, v.mean(), 0.6, color=col); ax.plot(np.full(len(v), j) + np.linspace(-0.18, 0.18, len(v)), v, "o", ms=3.5, color=INK, alpha=0.55)
            ax.text(j, v.max() + 0.025, f"{v.mean():.2f}", ha="center", fontsize=9.5, fontweight="bold")
        d = np.array([r["arms"]["sync"]["bal_acc"] - r["arms"]["surrogate"]["bal_acc"] for r in rs])
        ax.axhline(0.25, color=MUTED, ls="--", lw=1); ax.text(0.5, 0.262, "chance", fontsize=8, color=MUTED, ha="center")
        ax.set_xticks(range(3), [a[1] for a in arms], fontsize=9); ax.set_ylim(0, 0.95); ax.grid(axis="x", visible=False)
        ax.set_ylabel("balanced accuracy")
        ax.text(0.5, 0.93, f"sync - surrogate = {d.mean():+.2f} ({int((d > 0).sum())}/{len(d)} datasets)\nalthough the classes differ ONLY in source power",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=CRIT)
    ax.set_title("(b) Published code, power-only data: the \"phase\" contrast fires", loc="left", fontsize=10)
    # (c) what the lag block can and cannot tell apart (closed-form path; board 2026-11-13)
    ax = axes[1, 0]
    vp, tp = ROOT / "results" / "v2_probe" / "synthetic.json", ROOT / "results" / "v2_probe" / "traps.json"
    if vp.exists() and tp.exists():
        V = dict(json.loads(vp.read_text())["summary"], **json.loads(tp.read_text())["summary"])
        worlds = [("G1", "power only,\nno lags\nanywhere"), ("T_bg", "power only +\na fixed lagged\nbackground"),
                  ("T_amp", "SAME coupling\nin every class,\namplitude differs"), ("G2i", "coupling\ndiffers between\nclasses")]
        worlds = [w_ for w_ in worlds if w_[0] in V]
        arms = [("im", "lag block (coherency)", S1), ("im_raw", "lag block, unnormalised", S3), ("im_null", "shift-null", "#9dbfe8")]
        w = 0.26
        for gi, (g, _) in enumerate(worlds):
            for k_, (k, lab, col) in enumerate(arms):
                if k not in V[g]["means"]:
                    continue
                m = V[g]["means"][k]
                ax.bar(gi + (k_ - 1) * w, m, w * 0.92, color=col, label=lab if gi == 0 else None)
                ax.text(gi + (k_ - 1) * w, m + 0.015, f"{m:.2f}", ha="center", fontsize=8, color=INK)
        ax.axhline(0.25, color=MUTED, ls="--", lw=1)
        ax.axvspan(1.5, 3.5, color=CRIT, alpha=0.07, lw=0)
        ax.text(2.5, 0.93, "indistinguishable: both decode", ha="center", fontsize=9, color=CRIT)
        ax.set_xticks(range(len(worlds)), [w_[1] for w_ in worlds], fontsize=8.5); ax.set_ylim(0, 1.0); ax.grid(axis="x", visible=False)
        ax.set_ylabel("balanced accuracy, lag block alone"); ax.legend(ncol=1, fontsize=8, loc="upper left")
    ax.set_title("(c) Simulated: a decoding lag block does not show a coupling change", loc="left", fontsize=10)
    # (d) HGD development subjects (closed-form path)
    ax = axes[1, 1]
    hd = ROOT / "results" / "v2_probe" / "hgd_b12s8_4-100Hz_raw" / "summary.json"       # same folds, adds im_raw
    hp = [hd] if hd.exists() else sorted((ROOT / "results" / "v2_probe").glob("hgd_*/summary.json"))
    if hp:
        H = json.loads(hp[-1].read_text())
        arms = [("FBCSP_LDA", "FBCSP\n+ LDA", MUTED), ("power", "v2\npower", S3), ("power+re+im", "v2 all\nblocks", S1),
                ("im", "lag\nblock", S1), ("im_raw", "same, un-\nnormalised", S3),
                ("im_null", "shift-\nnull", "#9dbfe8"), ("im_evoked", "evoked\ncontrol", "#b9b8b2")]
        arms = [a_ for a_ in arms if a_[0] in H["mean_over_subjects"]]
        subs = sorted(H["per_subject"], key=int)
        for k_, (k, lab, col) in enumerate(arms):
            x = k_ + (0.5 if k_ >= 3 else 0)
            v = np.array([H["per_subject"][s_][k] for s_ in subs])
            ax.bar(x, v.mean(), 0.66, color=col)
            ax.plot(np.full(len(v), x) + np.linspace(-0.17, 0.17, len(v)), v, "o", ms=3.5, color=INK, alpha=0.55)
            ax.text(x, max(v.max(), v.mean()) + 0.025, f"{v.mean():.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(0.25, color=MUTED, ls="--", lw=1)
        ax.set_xticks([k_ + (0.5 if k_ >= 3 else 0) for k_ in range(len(arms))], [a_[1] for a_ in arms], fontsize=8)
        ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False); ax.set_ylabel("balanced accuracy, held-out folds")
        ax.set_title(f"(d) Real EEG: HGD subjects {subs[0]}-{subs[-1]}, CV inside the training data", loc="left", fontsize=10)
    else:
        ax.set_title("(d) Real EEG: not run", loc="left", fontsize=10)
    fig.savefig(OUT / "fig10_critique_and_v2.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--diag", default=None, help="dir with diag10.json, diag10b.json")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    da = db = None
    if a.diag:                                   # explicit directory with diag10.json / diag10b.json
        da = json.loads((Path(a.diag) / "diag10.json").read_text())
        db = json.loads((Path(a.diag) / "diag10b.json").read_text())
    elif (ROOT / "results/screen10/diag10_variants.json").exists():      # committed copies (default)
        da = json.loads((ROOT / "results/screen10/diag10_variants.json").read_text())
        db = json.loads((ROOT / "results/screen10/diag10b_alignment.json").read_text())
    sc = ROOT / "results" / "screen10" / "stage_C" / "results_recomputed.json"   # fixed decide (DEVIATIONS 20)
    if not sc.exists():
        sc = sc.with_name("results.json")
    stage_c = json.loads(sc.read_text())["per_cell"] if sc.exists() else None
    hist_c = None
    if stage_c:
        hist_c = ("Screen 10 C\nrule v3r, fresh sessions", sum(r["correct"] for r in stage_c.values()),
                  sum(r["fires"] is True and not r["should_fire"] for r in stage_c.values()),
                  sum(r["fires"] is False and r["should_fire"] for r in stage_c.values()))
    fig_traps(db, stage_c)
    if da and db:
        fig_freq(da, db)
    fig_history(hist_c)
    fig_integrator()
    fig_traces()
    fig_board()
    fig_drift()
    fig_screen11()
    fig_calib_p1()
    fig_v2()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
