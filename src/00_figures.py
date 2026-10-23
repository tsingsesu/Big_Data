"""论文概念图：技术路线、预处理流程、Q1 建模框架。

运行：
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/00_figures.py
输出：
    img/fig_roadmap.pdf/.png
    img/fig_preprocess.pdf/.png
    img/fig_q1_framework.pdf/.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
IMG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "figure.dpi": 150})

COLORS = {
    "data": "#4C72B0",
    "q1": "#DD8452",
    "q2": "#55A868",
    "q3": "#C44E52",
    "val": "#8172B3",
    "bg": "#F7F7F7",
}


def box(ax, xy, w, h, text, color, fontsize=8, textcolor="white"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0.8, edgecolor="white", facecolor=color, alpha=0.95,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, wrap=True)


def arrow(ax, start, end, color="#555555"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=color, shrinkA=2, shrinkB=2,
        )
    )


def save(fig, name):
    fig.savefig(IMG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def roadmap():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    box(ax, (0.01, 0.36), 0.22, 0.28,
        "Data foundation\n1.csv–8.xlsx\nbeer-key alignment\nper-column transform\nmissing mask",
        COLORS["data"], 7.5)
    box(ax, (0.34, 0.70), 0.30, 0.20,
        "Q1  Sensory scoring\ncontinuous regression\n+ robust 0–6 calibration\nmissingness stress test",
        COLORS["q1"], 7.5)
    box(ax, (0.34, 0.40), 0.30, 0.20,
        "Q2  Compound–flavour\nco-expression network\nTop-3 + bootstrap/Jaccard\nchemical plausibility",
        COLORS["q2"], 7.5)
    box(ax, (0.34, 0.10), 0.30, 0.20,
        "Q3  Style generation\nconditional copula / OT\nPils→Tripel transfer\ndefect constraints + portrait",
        COLORS["q3"], 7.5)
    box(ax, (0.74, 0.36), 0.24, 0.28,
        "Validation & paper\nCV / bootstrap\nMMD / classifier\nLaTeX + reproducible code",
        COLORS["val"], 7.5)

    arrow(ax, (0.23, 0.50), (0.34, 0.80))
    arrow(ax, (0.23, 0.50), (0.34, 0.50))
    arrow(ax, (0.23, 0.50), (0.34, 0.20))
    arrow(ax, (0.64, 0.80), (0.74, 0.55))
    arrow(ax, (0.64, 0.50), (0.74, 0.50))
    arrow(ax, (0.64, 0.20), (0.74, 0.45))
    ax.text(0.5, 0.96, "Technical roadmap", ha="center", va="center", fontsize=11, weight="bold")
    save(fig, "fig_roadmap")


def preprocess():
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    y, h = 0.38, 0.26
    xs = [0.01, 0.21, 0.41, 0.61, 0.81]
    texts = [
        "8 raw tables\n1.csv … 8.xlsx",
        "Align by beer\n(not beer_id)\nverify 250 samples",
        "Per-column transform\n37 log10 / 33 linear\n161 inferred log10",
        "Missing mask\nnatural 73 cells\ntrain-fold imputation",
        "Model inputs\nX 250×231\ny 250×50\n22 styles",
    ]
    colors = [COLORS["data"], COLORS["data"], COLORS["q1"], COLORS["q2"], COLORS["val"]]
    for x, t, c in zip(xs, texts, colors):
        box(ax, (x, y), 0.17, h, t, c, 6.8)
    for i in range(4):
        arrow(ax, (xs[i] + 0.17, y + h / 2), (xs[i + 1], y + h / 2))
    ax.text(0.5, 0.88, "Data preprocessing and leakage control", ha="center", va="center",
            fontsize=10.5, weight="bold")
    ax.text(0.5, 0.12, "All imputers / scalers / calibration thresholds are fitted inside training folds only.",
            ha="center", va="center", fontsize=7, color="#555555")
    save(fig, "fig_preprocess")


def q1_framework():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    box(ax, (0.08, 0.70), 0.50, 0.18,
        "Baseline\nper-target Ridge / PLS\n(no nonlinearity)",
        COLORS["data"], 7.8)
    box(ax, (0.08, 0.42), 0.50, 0.18,
        "Improved\nmulti-output PLS/Ridge\n+ missing-mask imputation",
        COLORS["q1"], 7.8)
    box(ax, (0.08, 0.14), 0.50, 0.18,
        "Final\nequal-weight ensemble (5 models)\n+ robust min-max 0–6 calibration",
        COLORS["q2"], 7.8)
    box(ax, (0.66, 0.30), 0.30, 0.34,
        "Stress tests\nMCAR 5/10/20/30%\nMAR proxy\nnatural missing",
        COLORS["val"], 7.5)
    arrow(ax, (0.33, 0.70), (0.33, 0.60))
    arrow(ax, (0.33, 0.42), (0.33, 0.32))
    arrow(ax, (0.58, 0.40), (0.66, 0.45))
    ax.text(0.5, 0.94, "Q1 modelling framework", ha="center", va="center",
            fontsize=10.5, weight="bold")
    save(fig, "fig_q1_framework")


if __name__ == "__main__":
    roadmap()
    preprocess()
    q1_framework()
    print("Saved img/fig_roadmap, img/fig_preprocess, img/fig_q1_framework")
