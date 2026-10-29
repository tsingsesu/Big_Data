# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本次仅整理运行入口、路径和可视化范围，核心计算保留原程序；更早的AI参与见使用详情说明。
"""数据概况、变换表与4/7.csv同概念一致性；仅输出数据，不生成EDA图。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import 公共数据处理 as du
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    D = du.load_all()
    X, y, styles = D["X"], D["y"], pd.Series(D["styles"])
    fm, tm = D["feature_meta"], D["target_meta"]

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
    fm.to_csv(OUT / "特征变换表.csv", index=False)
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
    (OUT / "数据概况.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print("Saved 数据概况.json and 特征变换表.csv")


if __name__ == "__main__":
    main()
