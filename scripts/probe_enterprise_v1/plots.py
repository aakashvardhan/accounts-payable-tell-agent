"""Render report figures from JSON outputs (run with .venv-inspect/bin/python; matplotlib only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/probe_training/enterprise_v1"
EV = OUT / "evaluation"
FIG = OUT / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PARTS = [("test", "Held-out test"), ("lexical_challenge", "Lexical challenge"), ("delayed_memory_ood", "Delayed-memory OOD")]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "legend.frameon": False, "lines.linewidth": 2})


def curves():
    c = json.loads((EV / "curves_and_scores.json").read_text())
    om = json.loads((EV / "overall_metrics.json").read_text())
    for kind, xl, yl, fn in (("roc", "False positive rate", "True positive rate", "roc_curve.png"),
                             ("pr", "Recall", "Precision", "pr_curve.png")):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for (p, label), col in zip(PARTS, SERIES):
            x, y = c[p][kind]
            stat = om[p]["auroc"] if kind == "roc" else om[p]["auprc"]
            ax.plot(x, y, color=col, label=f"{label} ({'AUROC' if kind == 'roc' else 'AUPRC'} {stat:.3f})",
                    drawstyle="steps-post" if kind == "pr" else "default")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"Frozen Tell probe: {'ROC' if kind == 'roc' else 'precision-recall'}", loc="left", fontsize=11)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG / fn, dpi=160); plt.close(fig)


def hist():
    c = json.loads((EV / "curves_and_scores.json").read_text())["hist"]
    t = json.loads((EV / "overall_metrics.json").read_text())["_frozen"]["threshold"]
    parts = PARTS + [("calibration_test", "Initial-state clean (calibration)")]
    fig, axes = plt.subplots(len(parts), 1, figsize=(6.4, 2.1 * len(parts)), sharex=True)
    bins = [i / 40 for i in range(41)]
    for ax, (p, label) in zip(axes, parts):
        for cls, col in (("clean", SERIES[0]), ("attacked", SERIES[1])):
            v = c[p].get(cls, [])
            if v:
                ax.hist(v, bins=bins, color=col, alpha=0.75, label=f"{cls} (n={len(v)})", edgecolor=SURFACE, linewidth=1)
        ax.axvline(t, color=INK, lw=1, ls="--")
        ax.set_title(label, loc="left", fontsize=10); ax.set_ylabel("count"); ax.legend(fontsize=8, loc="upper center")
    axes[-1].set_xlabel(f"alarm score (dashed line = frozen threshold {t:.4f})")
    fig.tight_layout(); fig.savefig(FIG / "alarm_score_distribution.png", dpi=160); plt.close(fig)


def layers():
    r = json.loads((OUT / "candidate_validation_metrics.json").read_text())
    cfg = json.loads((OUT / "frozen_probe/probe_config.json").read_text())
    Cs = sorted({x["C"] for x in r}); Ls = sorted({x["layer"] for x in r})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, key, name in ((axes[0], "val_auroc", "Validation AUROC"), (axes[1], "val_auprc", "Validation AUPRC")):
        for C, col in zip(Cs, SERIES):
            ys = [next(x[key] for x in r if x["layer"] == L and x["C"] == C) for L in Ls]
            ax.plot(Ls, ys, color=col, marker="o", ms=6, label=f"C={C:g}")
        sel = next(x for x in r if x["layer"] == cfg["selected_layer"] and x["C"] == cfg["selected_C"])
        ax.scatter([sel["layer"]], [sel[key]], s=140, facecolors="none", edgecolors=INK, lw=1.5, zorder=5)
        ax.set_xticks(Ls); ax.set_xticklabels([f"hidden state {L}" for L in Ls], fontsize=8)
        ax.set_title(name, loc="left", fontsize=11)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Layer comparison (validation only; circle = selected probe)", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(); fig.savefig(FIG / "layer_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    layers(); curves(); hist()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
