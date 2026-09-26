"""Figures for the operational-threshold report (matplotlib; run with .venv-inspect/bin/python)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def roc_curve(y, s):
    thr = np.unique(s)[::-1]
    thr = np.concatenate(([thr[0] + 1], thr))
    p, n = (y == 1).sum(), (y == 0).sum()
    tpr = np.array([(s[y == 1] >= t).sum() / p for t in thr])
    fpr = np.array([(s[y == 0] >= t).sum() / n for t in thr])
    return fpr, tpr, thr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/routing_design/operational_threshold_v1"
FIG = OUT / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "legend.frameon": False, "lines.linewidth": 2})


def load_validation():
    rows = [json.loads(l) for l in open(OUT / "validation_scores_v1.jsonl")]
    s = np.array([r["score"] for r in rows]); y = np.array([r["exposure_label"] for r in rows])
    return s, y


def roc_with_thresholds():
    s, y = load_validation()
    fpr, tpr, thr = roc_curve(y, s)
    op = json.load(open(OUT / "operational_threshold_v1.json"))
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(fpr, tpr, color=SERIES[0], label="Validation ROC")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    for label, t, col in (("Primary 0.1708", op["primary_threshold_unchanged"], INK),
                         (f"Operational {op['selected_operational_threshold']:.3f}", op["selected_operational_threshold"], SERIES[1])):
        i = int(np.argmin(np.abs(thr - t)))
        ax.scatter([fpr[i]], [tpr[i]], s=90, color=col, zorder=5, label=f"{label} (FPR {fpr[i]:.2f}, TPR {tpr[i]:.2f})")
    ax.axvline(0.20, color=SERIES[2], lw=1, ls="--", label="FPR = 0.20 constraint")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Validation ROC: primary vs operational threshold", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "validation_roc_thresholds.png", dpi=160); plt.close(fig)


def recall_vs_fpr():
    s, y = load_validation()
    thr = np.unique(s)
    rec, fpr = [], []
    for t in thr:
        pred = s >= t
        tp = (pred & (y == 1)).sum(); fn = (~pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum(); tn = (~pred & (y == 0)).sum()
        rec.append(tp / (tp + fn)); fpr.append(fp / (fp + tn))
    op = json.load(open(OUT / "operational_threshold_v1.json"))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(fpr, rec, color=SERIES[0])
    ax.axvline(0.20, color=SERIES[2], lw=1, ls="--", label="FPR <= 0.20 constraint")
    ax.scatter([op["validation_fpr"]], [op["validation_recall"]], s=100, color=SERIES[1], zorder=5,
              label=f"Selected operational ({op['validation_fpr']:.2f}, {op['validation_recall']:.2f})")
    ax.set_xlabel("Validation FPR"); ax.set_ylabel("Validation recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Validation recall vs. FPR trade-off", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "recall_vs_fpr_validation.png", dpi=160); plt.close(fig)


def routing_comparison():
    vol = json.load(open(OUT / "routing_volume_by_prevalence.json"))
    prevs = ["0.01", "0.05", "0.1"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    w = 0.35
    x = np.arange(len(prevs))
    for i, (name, col, label) in enumerate([("primary_0.1708", SERIES[0], "Primary (0.1708)"), ("operational", SERIES[1], "Operational")]):
        vals = [vol[name]["by_prevalence"][p]["routed_per_1000_invoices"] for p in prevs]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{float(p)*100:.0f}% prevalence" for p in prevs])
    ax.set_ylabel("Routed to Agent S per 1,000 invoices")
    ax.set_title("Expected routing volume: primary vs operational threshold", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "routing_volume_comparison.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    roc_with_thresholds(); recall_vs_fpr(); routing_comparison()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
