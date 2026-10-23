"""数据概况与 EDA 图：缺失、目标相关、PCA、变换统计、4.csv vs 7.csv 一致性。

运行：
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/01_profile.py
输出：
    img/eda_*.png / .pdf
    data/processed/profile.json
    data/processed/transform_table.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

import data_utils as du

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
OUT = ROOT / "data" / "processed"
IMG.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})


def save(fig, name: str) -> None:
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight")
    fig.savefig(IMG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    D = du.load_all()
    X, y, styles = D["X"], D["y"], pd.Series(D["styles"])
    fm, tm = D["feature_meta"], D["target_meta"]

    # 1) 缺失概览
    miss = X.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    if len(miss):
        miss.head(20).iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_xlabel("Missing cells")
        ax.set_title("Natural missingness in 1.csv (top 20 features)")
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    save(fig, "eda_missingness")

    # 2) 目标相关聚类热图
    fig = sns.clustermap(
        y.corr(),
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(8.2, 8.2),
        linewidths=0.2,
        cbar_kws={"label": "Pearson r"},
    )
    fig.fig.suptitle("Correlation structure of 50 sensory targets", y=1.02)
    fig.savefig(IMG / "eda_target_corr.png", bbox_inches="tight")
    fig.savefig(IMG / "eda_target_corr.pdf", bbox_inches="tight")
    plt.close("all")

    # 3) PCA：分位数变换后按主要风格着色
    pipe = du.make_preprocessor()
    Z = pipe.fit_transform(X)
    pcs = PCA(n_components=2, random_state=42).fit_transform(Z)
    top = styles.value_counts().head(8).index.tolist()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    other = ~styles.isin(top)
    ax.scatter(pcs[other, 0], pcs[other, 1], s=12, c="#B0B0B0", alpha=0.55, label="Other")
    pal = sns.color_palette("colorblind", n_colors=len(top))
    for color, style in zip(pal, top):
        m = (styles == style).values
        ax.scatter(pcs[m, 0], pcs[m, 1], s=22, color=color, alpha=0.85, label=style)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of chemical features (quantile-normalized)")
    ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    save(fig, "eda_pca_styles")

    # 3b) 风格样本量（Q3 的不平衡背景）
    counts = styles.value_counts()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    colors = [
        "#C44E52" if s in ("Tripel", "Kriek", "Lambic", "Pils/Lager") else "#4C72B0"
        for s in counts.index
    ]
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_ylabel("Number of beers")
    ax.set_title("Style distribution (red: Q3 target/source styles)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    save(fig, "eda_style_counts")

    # 4) 变换统计
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fm["transform"].value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Column transform")
    axes[0].set_ylabel("Count")
    fm["transform_source"].value_counts().plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("How the transform was decided")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    save(fig, "eda_transform_counts")

    # 5) 4.csv vs 7.csv 同概念一致性（Ratebeer 独立感官集）
    pairs = [
        ("acidity", "taste_sour"),
        ("bitternes", "taste_bitter"),
        ("alcohol", "taste_alcohol"),
        ("sweetness", "taste_sweet"),
        ("t2n", "taste_stale"),
        ("dms", "taste_corn"),
        ("diacetyl", "taste_diacetyl"),
        ("body", "palate_blanket"),
        ("aftertaste", "palate_aftertaste"),
    ]
    df4, df7 = D["df4"], D["df7"]
    rows = []
    for a, b in pairs:
        if a in df4.columns and b in df7.columns:
            x = pd.to_numeric(df4[a], errors="coerce")
            z = pd.to_numeric(df7[b], errors="coerce")
            m = x.notna() & z.notna()
            rows.append({"target_4csv": a, "ratebeer_7csv": b, "r": np.corrcoef(x[m], z[m])[0, 1], "n": int(m.sum())})
    cons = pd.DataFrame(rows).sort_values("r")
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.barh(cons["target_4csv"], cons["r"], color="#C44E52")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("Pearson r between 4.csv and 7.csv")
    ax.set_title("Cross-panel agreement for matched concepts")
    save(fig, "eda_4vs7")

    # 6) 落盘
    fm.to_csv(OUT / "transform_table.csv", index=False)
    profile = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_targets": int(y.shape[1]),
        "n_styles": int(styles.nunique()),
        "style_counts": styles.value_counts().to_dict(),
        "natural_missing_cells": int(X.isna().sum().sum()),
        "transform_counts": fm["transform"].value_counts().to_dict(),
        "transform_source_counts": fm["transform_source"].value_counts().to_dict(),
        "target_std_range": [float(y.std().min()), float(y.std().max())],
        "target_corr_median": float(np.median(y.corr().values[np.triu_indices_from(y.corr(), 1)])),
        "target_corr_mean": float(np.mean(y.corr().values[np.triu_indices_from(y.corr(), 1)])),
        "cross_panel": cons.to_dict(orient="records"),
    }
    (OUT / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("\nSaved figures to img/eda_* and profile to data/processed/profile.json")


if __name__ == "__main__":
    main()
