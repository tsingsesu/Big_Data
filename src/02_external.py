"""外部数据校验：Kaggle 风格级感官原型 vs 附件 4.csv / 7.csv 的风格均值。

运行：
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/02_external.py
输出：
    data/external/style_validation.csv
    img/external_style_validation.png/.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import data_utils as du

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
EXT = ROOT / "data" / "external"
IMG.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "figure.dpi": 150})


def safe_r(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan
    if method == "pearson":
        r, p = pearsonr(x[m], y[m])
    else:
        r, p = spearmanr(x[m], y[m])
    return float(r), float(p)


def main() -> None:
    D = du.load_all()
    df4, df7 = D["df4"], D["df7"]
    proto = pd.read_csv(EXT / "style_prototypes.csv").rename(columns={"风格": "style"})
    styles = proto["style"].tolist()

    to4 = [
        "acidity", "bitternes", "sweetness", "alcohol",
        "A_malt_all", "A_hops_all", "A_esters_fruity", "clove", "coriander", "body",
    ]
    to7 = [
        "taste_sour", "taste_bitter", "taste_sweet", "taste_alcohol",
        "aroma_malt", "aroma_hoppy", "aroma_fruity", "aroma_clove", "aroma_coriander", "palate_blanket",
    ]
    m4 = df4.groupby("tasting_category_fine")[to4].mean().reindex(styles)
    m7 = df7.groupby("tasting_category_fine")[to7].mean().reindex(styles)

    # Spices 用 clove + coriander 的均值
    m4["spices"] = m4[["clove", "coriander"]].mean(axis=1)
    m7["spices"] = m7[["aroma_clove", "aroma_coriander"]].mean(axis=1)

    pairs = [
        ("Sour", "Sour", "acidity", "taste_sour"),
        ("Bitter", "Bitter", "bitternes", "taste_bitter"),
        ("Sweet", "Sweet", "sweetness", "taste_sweet"),
        ("Alcohol", "Alc", "alcohol", "taste_alcohol"),
        ("Malty", "Malty", "A_malt_all", "aroma_malt"),
        ("Hoppy", "Hoppy", "A_hops_all", "aroma_hoppy"),
        ("Fruits", "Fruits", "A_esters_fruity", "aroma_fruity"),
        ("Spices", "Spices", "spices", "spices"),
        ("Body", "Body", "body", "palate_blanket"),
    ]

    rows = []
    for name, kcol, c4, c7 in pairs:
        x = proto[kcol].values
        r4, p4 = safe_r(x, m4[c4].values)
        r7, p7 = safe_r(x, m7[c7].values)
        s4, _ = safe_r(x, m4[c4].values, "spearman")
        s7, _ = safe_r(x, m7[c7].values, "spearman")
        # leave-one-style-out 范围（Pearson）
        loo4, loo7 = [], []
        for i in range(len(styles)):
            mask = np.arange(len(styles)) != i
            rr4, _ = safe_r(x[mask], m4[c4].values[mask])
            rr7, _ = safe_r(x[mask], m7[c7].values[mask])
            if np.isfinite(rr4):
                loo4.append(rr4)
            if np.isfinite(rr7):
                loo7.append(rr7)
        rows.append({
            "dimension": name,
            "kaggle_col": kcol,
            "target_4csv": c4,
            "ratebeer_7csv": c7,
            "pearson_4csv": r4,
            "spearman_4csv": s4,
            "pearson_7csv": r7,
            "spearman_7csv": s7,
            "loo4_min": np.min(loo4) if loo4 else np.nan,
            "loo4_max": np.max(loo4) if loo4 else np.nan,
            "loo7_min": np.min(loo7) if loo7 else np.nan,
            "loo7_max": np.max(loo7) if loo7 else np.nan,
            "n_styles": len(styles),
        })
    val = pd.DataFrame(rows)
    val.to_csv(EXT / "style_validation.csv", index=False)
    print(val.round(3).to_string(index=False))

    def plot_scatter(m, filename, ylabel, title):
        fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.0))
        for ax, (name, kcol, c4, c7) in zip(axes.ravel(), pairs):
            x = proto[kcol].values
            y = m[c4 if m is m4 else c7].values
            ax.scatter(x, y, s=22, color="#4C72B0", alpha=0.85)
            for s, xi, yi in zip(styles, x, y):
                if s in ("Tripel", "Kriek", "Lambic", "Pils/Lager"):
                    ax.annotate(s, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
            r, _ = safe_r(x, y)
            ax.set_title(f"{name}: r={r:.2f}", fontsize=8)
            ax.set_xlabel("Kaggle prototype (word share)", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(title, y=1.01, fontsize=10)
        fig.tight_layout()
        fig.savefig(IMG / f"{filename}.png", bbox_inches="tight")
        fig.savefig(IMG / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    plot_scatter(
        m4, "external_style_validation",
        "4.csv style mean",
        "External validity: Kaggle prototypes vs trained panel (4.csv), 22 styles",
    )
    plot_scatter(
        m7, "external_style_validation_7csv",
        "7.csv style mean",
        "External validity: Kaggle prototypes vs Ratebeer descriptors (7.csv), 22 styles",
    )
    print("\nSaved data/external/style_validation.csv and img/external_style_validation*.*")


if __name__ == "__main__":
    main()
