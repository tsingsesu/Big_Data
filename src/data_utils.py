"""数据加载与预处理：统一读取附件 1-8，构建建模用 X / y / 风格 / 元数据。

用法：
    from data_utils import load_all
    D = load_all()
    X, y = D["X"], D["y"]

约定：
- 所有表以 `beer`（1-250）对齐；`beer_id` 在 4/7.csv 中被存成科学计数法，不能用于连接。
- `1.csv` 不是整表 log10：70 个可校准化合物中 37 列 log10、33 列线性；其余列按负值判断。
- 原始附件只读，派生数据写到 data/（已 gitignore）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTACH = ROOT / "附件"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["beer", "beer_id", "tasting_category_fine"]
STYLE_COL = "tasting_category_fine"

# 图表用中文标签：感官维度与模型名（逻辑不变，仅显示用）
TARGET_CN = {
    "A_malt_all": "麦芽香-总体", "A_malt_grain": "麦芽香-谷物",
    "A_malt_bread": "麦芽香-面包", "A_malt_cara": "麦芽香-焦糖",
    "A_malt_burn": "麦芽香-烟熏",
    "A_hops_all": "酒花香-总体", "A_hops_citrus": "酒花香-柑橘",
    "A_hops_tropical": "酒花香-热带水果", "A_hops_noble": "酒花香-贵族酒花",
    "A_hops_woody": "酒花香-木质",
    "A_esters_all": "酯香-总体", "A_esters_ethac": "酯香-乙酸乙酯",
    "A_esters_isoaa": "酯香-乙酸异戊酯", "A_esters_flower": "酯香-花香",
    "A_esters_fruity": "酯香-果香",
    "F_malt_all": "麦芽味-总体", "F_malt_grain": "麦芽味-谷物",
    "F_malt_bread": "麦芽味-面包", "F_malt_cara": "麦芽味-焦糖",
    "F_malt_burn": "麦芽味-烟熏",
    "F_hops_all": "酒花味-总体", "F_hops_citrus": "酒花味-柑橘",
    "F_hops_tropical": "酒花味-热带水果", "F_hops_noble": "酒花味-贵族酒花",
    "F_hops_woody": "酒花味-木质",
    "F_esters_all": "酯味-总体", "F_esters_ethac": "酯味-乙酸乙酯",
    "F_esters_isoaa": "酯味-乙酸异戊酯", "F_esters_flower": "酯味-花香",
    "F_esters_fruity": "酯味-果香",
    "acidity": "酸度", "bitternes": "苦味", "sweetness": "甜味",
    "X4vg": "4-乙烯基愈创木酚", "diacetyl": "双乙酰", "dms": "二甲基硫",
    "metallic": "金属味", "stale_hops": "陈旧酒花", "t2n": "反-2-壬烯醛",
    "orange": "橙", "coriander": "芫荽", "clove": "丁香",
    "lactic": "乳酸", "acetic": "醋酸", "barnyard": "谷仓",
    "alcohol": "酒精", "aftertaste": "余味", "body": "酒体",
    "co2": "二氧化碳", "overall": "总体",
}

MODEL_CN = {
    "mean": "均值", "ridge": "岭回归", "pls": "PLS",
    "extratrees": "极端随机树", "randomforest": "随机森林",
    "lightgbm": "LightGBM", "blend_avg": "等权融合",
    "stack_ridge": "Ridge 堆叠",
}


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ATTACH / name, dtype={"beer_id": str}, encoding="utf-8-sig")


def _read_xlsx(sheet: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(ATTACH / "8.xlsx", sheet_name=sheet, dtype={"beer_id": str}, **kwargs)


def load_raw() -> dict[str, pd.DataFrame]:
    """读取全部原始表，不做任何变换。"""
    return {
        "df1": _read_csv("1.csv"),
        "df4": _read_csv("4.csv"),
        "df7": _read_csv("7.csv"),
        "corr2": pd.read_csv(ATTACH / "2.csv", index_col=0, encoding="utf-8-sig"),
        "corr5": pd.read_csv(ATTACH / "5.csv", index_col=0, encoding="utf-8-sig"),
        "corr6": pd.read_csv(ATTACH / "6.csv", index_col=0, encoding="utf-8-sig"),
        "thresholds": _read_xlsx("Sensory thresholds"),
        "snames": _read_xlsx("Sensory names"),
        "cal": _read_xlsx("Calibrated chemical dataset"),
        "prices": _read_xlsx("Beer Prices"),
        "gbm_imp": pd.read_excel(
            ATTACH / "8.xlsx", sheet_name="GBM importance MDI", header=None
        ),
    }


def _classify_transform(df1: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """逐列判断 1.csv 的数值是 log10 还是线性浓度。

    对 70 个可校准列：比较 10^x 与 x 哪个更接近 8.xlsx 的校准浓度；
    其余列：有负值判为 log10，否则记为 unknown（建模时用分位数变换兜底）。
    """
    features = [c for c in df1.columns if c not in ID_COLS]
    cal_cols = [c for c in cal.columns if c not in ID_COLS]
    common = [c for c in features if c in cal_cols]

    rows = []
    x = df1.set_index("beer")
    c = cal.set_index("beer")
    for feat in features:
        rec = {
            "feature": feat,
            "in_cal_dataset": feat in cal_cols,
            "transform": "unknown",
            "transform_source": "unresolved",
        }
        if feat in common:
            a = pd.to_numeric(x[feat], errors="coerce")
            b = pd.to_numeric(c[feat], errors="coerce")
            m = a.notna() & b.notna() & (b > 0)
            if m.sum() >= 10:
                log_ratio = b[m] / np.power(10.0, a[m].clip(-12, 12))
                lin_ratio = b[m] / a[m].replace(0, np.nan)
                med_log = float(np.nanmedian(log_ratio))
                med_lin = float(np.nanmedian(lin_ratio))
                d_log = abs(np.log10(med_log)) if med_log > 0 else 99
                d_lin = abs(np.log10(med_lin)) if med_lin > 0 else 99
                if d_log < 0.15 and d_log <= d_lin:
                    rec["transform"] = "log10"
                    rec["transform_source"] = "calibrated_table"
                elif d_lin < 0.15 and d_lin < d_log:
                    rec["transform"] = "linear"
                    rec["transform_source"] = "calibrated_table"
                rec["median_ratio_log10"] = med_log
                rec["median_ratio_linear"] = med_lin
        else:
            vals = pd.to_numeric(df1[feat], errors="coerce")
            # 未校准列：GC-MS 丰度基本是 log10；有负值直接判定，全正但值域 <12 也按 log10
            if (vals < 0).any() or float(vals.max()) < 12:
                rec["transform"] = "log10"
                rec["transform_source"] = "inferred_from_range"
        rows.append(rec)
    return pd.DataFrame(rows)


def build_meta(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """构建特征元数据、目标元数据、化合物映射。"""
    df1, df4 = raw["df1"], raw["df4"]
    thresholds, snames = raw["thresholds"], raw["snames"]
    features = [c for c in df1.columns if c not in ID_COLS]
    compounds = list(raw["corr2"].columns)
    assert len(features) == len(compounds) == 231, (len(features), len(compounds))

    # 1.csv 的特征顺序与 2.csv 的化合物顺序逐位一致（已核对 212 个可匹配名称）
    feats = pd.DataFrame({"feature": features, "compound_2csv": compounds})

    meta_cols = [
        "field_name_R",
        "field_name_python",
        "chemical_name",
        "unit",
        "calibrated",
        "origin_paper",
        "aging",
        "off_flavors",
        "threshold_mean",
        "threshold_median",
    ]
    th = thresholds[meta_cols].drop_duplicates("field_name_R").set_index("field_name_R")
    feats = feats.merge(
        th, left_on="feature", right_index=True, how="left", validate="one_to_one"
    )

    transforms = _classify_transform(df1, raw["cal"])
    feats = feats.merge(transforms, on="feature", how="left", validate="one_to_one")

    # 目标元数据：只取 trained panel（4.csv 的 50 个目标）
    tmeta = snames[snames["dataset"] == "Trained panel"][
        ["R_name", "Category", "CleanName_Paper", "CleanName_PanelFeedback", "DataType"]
    ].drop_duplicates("R_name")
    targets = [c for c in df4.columns if c not in ID_COLS]
    tmeta = tmeta.rename(columns={"R_name": "target"}).set_index("target")
    tmeta = tmeta.reindex(targets)
    tmeta["target"] = tmeta.index

    return {"feature_meta": feats.reset_index(drop=True), "target_meta": tmeta.reset_index(drop=True)}


def make_grade_labels(
    y: pd.DataFrame,
    proportions: tuple[float, ...] = (0.025, 0.075, 0.15, 0.25, 0.25, 0.15, 0.10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把标准化连续目标映射为 0-6 整数分级。

    返回 (grades, cutpoints)：
    - grades: 与 y 同形状的整数 DataFrame（0-6）；
    - cutpoints: 每个目标 6 个分位阈值，供预测值复用。

    这是“秩校准”：每列按固定的目标分布切分，保持样本间排序不变。
    论文中会同时报告 min-max / 高斯潜变量两种校准的敏感性分析。
    """
    assert abs(sum(proportions) - 1.0) < 1e-9 and len(proportions) == 7
    cum = np.cumsum(proportions)[:-1]
    cut = pd.DataFrame(
        {c: np.quantile(y[c].dropna(), cum) for c in y.columns}, index=range(1, 7)
    ).T
    grades = pd.DataFrame(index=y.index, columns=y.columns, dtype=int)
    for c in y.columns:
        grades[c] = np.searchsorted(cut.loc[c].values, y[c].values, side="right")
    return grades, cut


def apply_grade_cutpoints(z: pd.DataFrame | np.ndarray, cutpoints: pd.DataFrame) -> np.ndarray:
    """把连续预测值按训练集 cutpoints 映射为 0-6。"""
    arr = np.asarray(z)
    out = np.zeros_like(arr, dtype=int)
    cols = list(cutpoints.index)
    for j, c in enumerate(cols):
        out[:, j] = np.searchsorted(cutpoints.loc[c].values, arr[:, j], side="right")
    return out


def make_preprocessor():
    """缺失值中位数填补 + 分位数正态化；应在交叉验证训练折内 fit。"""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import QuantileTransformer

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "quantile",
                QuantileTransformer(
                    n_quantiles=200, output_distribution="normal", random_state=42
                ),
            ),
        ]
    )


def load_all() -> dict:
    """一键加载并构建所有建模对象。"""
    raw = load_raw()
    meta = build_meta(raw)
    df1, df4, cal = raw["df1"], raw["df4"], raw["cal"]

    # 样本对齐检查
    assert df1["beer"].is_unique and df4["beer"].is_unique
    assert (df1["beer"].values == df4["beer"].values).all()
    assert (df1["beer"].values == raw["df7"]["beer"].values).all()
    assert (df1["beer"].values == cal["beer"].values).all()

    features = [c for c in df1.columns if c not in ID_COLS]
    targets = [c for c in df4.columns if c not in ID_COLS]
    cal_features = [c for c in cal.columns if c not in ID_COLS]

    X = df1[features].apply(pd.to_numeric, errors="coerce")
    y = df4[targets].apply(pd.to_numeric, errors="coerce")
    X_cal = cal[cal_features].apply(pd.to_numeric, errors="coerce")

    return {
        **raw,
        **meta,
        "X": X,
        "y": y,
        "X_cal": X_cal,
        "meta": df1[ID_COLS].copy(),
        "styles": df1[STYLE_COL].astype(str).values,
        "features": features,
        "targets": targets,
        "cal_features": cal_features,
    }


if __name__ == "__main__":
    D = load_all()
    print("X:", D["X"].shape, "y:", D["y"].shape, "X_cal:", D["X_cal"].shape)
    print("styles:", pd.Series(D["styles"]).value_counts().to_dict())
    fm = D["feature_meta"]
    print("transform counts:", fm["transform"].value_counts().to_dict())
    print("feature meta columns:", list(fm.columns))
    print("target meta missing:", D["target_meta"].isna().sum().to_dict())
    print("calibrated columns:", fm["in_cal_dataset"].sum())
