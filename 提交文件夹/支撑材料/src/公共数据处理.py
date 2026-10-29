# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本次仅整理运行入口、路径和可视化范围，核心计算保留原程序；更早的AI参与见使用详情说明。
"""数据加载与预处理：统一读取附件 1-8，构建建模用 X / y / 风格 / 元数据。

用法：
    from 公共数据处理 import load_all
    D = load_all()
    X, y = D["X"], D["y"]

约定：
- 所有表以 `beer`（1-250）对齐；`beer_id` 在 4/7.csv 中被存成科学计数法，不能用于连接。
- `1.csv` 不是整表 log10：70 个可校准化合物中 37 列 log10、33 列线性；其余 161 列底数未确认，不做浓度反变换。
- 原始附件只读，派生数据写到 data/（已 gitignore）。
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTACH = Path(os.environ.get("BEER_ATTACHMENT_DIR", str(ROOT / "附件"))).expanduser().resolve()
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
    """Compare linear/log10/ln on positive calibrated values; never invert unknown columns."""
    x, c = df1.set_index("beer"), cal.set_index("beer")
    rows = []
    for feat in [f for f in df1 if f not in ID_COLS]:
        rec = {"feature": feat, "in_cal_dataset": feat in c,
               "transform": "unknown", "transform_source": "unresolved",
               "range_hypothesis": "none"}
        a = pd.to_numeric(x[feat], errors="coerce")
        if feat in c:
            v = pd.to_numeric(c[feat], errors="coerce")
            m = a.notna() & v.notna() & (v > 0)
            candidates = {"linear": a, "log10": 10.0**a.clip(-100,100),
                          "ln": np.exp(a.clip(-100,100))}
            errors = {}
            for name, pred in candidates.items():
                errors[name] = float(np.median(np.abs(pred[m]-v[m])/v[m]))
                rec["median_relative_error_"+name] = errors[name]
            best = min(errors, key=errors.get)
            if errors[best] < 0.01:
                rec.update(transform=best, transform_source="calibrated_positive_values")
            zeros = v.eq(0) & a.notna()
            rec["calibrated_zero_count"] = int(zeros.sum())
            rec["zero_inversion_mismatch_count"] = int((candidates[best][zeros].abs()>1e-8).sum())
            rec["positive_match_count"] = int(m.sum())
        elif (a<0).any() or float(a.max())<12:
            rec["range_hypothesis"] = "log_like_base_unverified"
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


def make_preprocessor(add_indicator=False):
    """Fit all learned operations inside each CV split; optional all-column indicators."""
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import QuantileTransformer
    values = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("quantile", QuantileTransformer(n_quantiles=100,
            output_distribution="normal", random_state=42)),
    ])
    if add_indicator:
        return FeatureUnion([("values", values),
            ("missing", MissingIndicator(features="all", error_on_new=False))])
    return values


def scoring_card_map():
    """52 named entries; unlabeled acetaldehyde is explicit, never silently omitted."""
    rows=[]
    for prefix, side, score_col in [("A", "Aroma", "Q"),("F", "Flavour", "AG")]:
        for group, positions, names in [
            ("malt", [7,10,12,14,16], ["all","grain","bread","cara","burn"]),
            ("hops", [20,23,25,27,29], ["all","citrus","tropical","noble","woody"]),
            ("esters", [33,36,38,40,42], ["all","ethac","isoaa","flower","fruity"])]:
            for pos,name in zip(positions,names):
                rows.append(dict(side=side, cell=f"{score_col}{pos}",
                    item=f"{prefix}_{group}_{name}", target=f"{prefix}_{group}_{name}"))
        rows.append(dict(side=side,cell=f"{score_col}45",
                         item=prefix+"_acetaldehyde",target=None))
    for side,col,items in [
        ("Aroma","Q",[(50,"body"),(54,"bitternes"),(58,"co2"),(62,"aftertaste"),
          (68,"dms"),(72,"metallic"),(76,"X4vg"),(82,"orange"),(86,"coriander"),
          (90,"clove"),(94,"overall")]),
        ("Flavour","AG",[(50,"alcohol"),(54,"sweetness"),(58,"acidity"),
          (68,"t2n"),(72,"diacetyl"),(76,"stale_hops"),(82,"lactic"),
          (86,"acetic"),(90,"barnyard")])]:
        for pos,t in items:
            rows.append(dict(side=side,cell=f"{col}{pos}",item=t,target=t))
    out=pd.DataFrame(rows)
    out["status"]=np.where(out.target.notna(),"continuous_label_available_relative_grade_only",
                           "unidentifiable_no_sensory_label")
    assert len(out)==52 and out.target.dropna().nunique()==50
    return out


def provenance(paths=()):
    """Input/source fingerprints and installed package versions for each result set."""
    import hashlib, importlib, importlib.metadata, platform
    inputs=list(ATTACH.glob("*.csv"))+list(ATTACH.glob("*.xlsx"))
    sources=list((ROOT/"src").glob("*.py"))
    return {"seed":42,"python":platform.python_version(),
        "packages":{n:importlib.import_module(module).__version__ for n,module in
            {"numpy":"numpy","pandas":"pandas","scipy":"scipy","scikit-learn":"sklearn",
             "lightgbm":"lightgbm","networkx":"networkx"}.items()},
        "package_metadata":{n:importlib.metadata.version(n) for n in
            ["numpy","pandas","scipy","scikit-learn","lightgbm","networkx"]},
        "sha256":{(f"附件/{p.name}" if p.parent == ATTACH else str(p.relative_to(ROOT))):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs+sources+list(paths)}}


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

    observed=X.corr(method="spearman").to_numpy()
    assert np.allclose(observed, raw["corr2"].to_numpy(),atol=1e-8), "Feature/correlation positional alignment failed"

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
