"""Figures for the synthetic mechanism-statistic programme (screens 7-10, diagnostics, review board).
python scripts/make_figures.py [--diag DIR]  -> results/figures/*.png

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
from resonet_sync.screen import boot_ci  # noqa: E402

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


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--diag", default=None, help="dir with diag10.json, diag10b.json")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    da = db = None
    if a.diag:
        da = json.loads((Path(a.diag) / "diag10.json").read_text())
        db = json.loads((Path(a.diag) / "diag10b.json").read_text())
    sc = ROOT / "results" / "screen10" / "stage_C" / "results.json"
    stage_c = json.loads(sc.read_text())["per_cell"] if sc.exists() else None
    hist_c = None
    if stage_c:
        hist_c = ("Screen 10 C\nrule v3r, fresh sessions", sum(r["correct"] for r in stage_c.values()),
                  sum(bool(r["fires"]) and not r["should_fire"] for r in stage_c.values()),
                  sum(r["fires"] is False and r["should_fire"] for r in stage_c.values()))
    fig_traps(db, stage_c)
    if da and db:
        fig_freq(da, db)
    fig_history(hist_c)
    fig_integrator()
    fig_traces()
    fig_board()
    fig_drift()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
