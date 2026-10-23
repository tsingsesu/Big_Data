#!/usr/bin/env python3
"""Q1: 数据缺失条件下的感官评分。

把 1.csv 的 231 个化学特征映射到 4.csv 的 50 个感官维度：
- 连续预测：RepeatedKFold(5 folds x 3 repeats)，折内 fit 预处理器；
- 基线：均值、Ridge、PLS；主模型：ExtraTrees、RandomForest、LightGBM、OOF 堆叠；
- 0-6 分级：稳健 min-max（主方案）、quantile、Gaussian latent（敏感性）；
- 缺失鲁棒性：自然缺失 + MCAR 5/10/20/30% + 分位数驱动 MAR。

运行：
    cd /home/tsingsesu/桌面/Big_Data
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/q1.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

from data_utils import apply_grade_cutpoints, load_all, make_grade_labels, make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "q1"
IMG_DIR = ROOT / "img"

SEED = 42
N_SPLITS = 5
N_REPEATS = 3
N_REPEATS_MISSING = 2
GRADE_PROPORTIONS = (0.025, 0.075, 0.15, 0.25, 0.25, 0.15, 0.10)

BASE_MODELS = ["ridge", "pls", "extratrees", "randomforest", "lightgbm"]
STACK_MODELS = list(BASE_MODELS)
ALL_MODELS = ["mean"] + BASE_MODELS + ["blend_avg", "stack_ridge"]
CALIBRATIONS = ["isotonic", "minmax", "quantile_rank", "quantile_map", "linear", "gaussian"]
METRIC_COLS = ["r2", "rmse", "mae", "spearman", "explained_variance"]


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _nanmean(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.asarray(x, dtype=float)))


def _nanstd(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = np.asarray(x, dtype=float)
        if np.sum(np.isfinite(arr)) <= 1:
            return 0.0
        return float(np.nanstd(arr, ddof=1))


def macro_r2_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """仅 R² 的宏平均，供内层 CV 快速选参。"""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2[~np.isfinite(r2)] = np.nan
    return _nanmean(r2)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    try:
        return float(spearmanr(a, b).statistic)
    except Exception:
        return np.nan


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """逐目标 R² / RMSE / MAE / Spearman / explained_variance。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
        ev = 1.0 - np.var(err, axis=0) / np.var(y_true, axis=0)
    r2[~np.isfinite(r2)] = np.nan
    ev[~np.isfinite(ev)] = np.nan
    n_targets = y_true.shape[1]
    sp = np.array([_safe_spearman(y_true[:, j], y_pred[:, j]) for j in range(n_targets)])
    return {
        "r2": r2,
        "rmse": np.sqrt(np.mean(err**2, axis=0)),
        "mae": np.mean(np.abs(err), axis=0),
        "spearman": sp,
        "explained_variance": ev,
    }


def macro_from_per_target(m: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: _nanmean(v) for k, v in m.items()}


def record_fold_metrics(
    rows: list[dict],
    model: str,
    repeat: int,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: list[str],
    n_train: int,
    n_test: int,
) -> None:
    if not np.isfinite(y_pred).all():
        raise RuntimeError(f"{model} repeat={repeat} fold={fold}: non-finite prediction")
    m = per_target_metrics(y_true, y_pred)
    for j, target in enumerate(targets):
        rows.append(
            {
                "model": model,
                "repeat": repeat,
                "fold": fold,
                "target": target,
                "n_train": n_train,
                "n_test": n_test,
                **{k: float(m[k][j]) for k in METRIC_COLS},
            }
        )
    rows.append(
        {
            "model": model,
            "repeat": repeat,
            "fold": fold,
            "target": "MACRO",
            "n_train": n_train,
            "n_test": n_test,
            **{k: _nanmean(m[k]) for k in METRIC_COLS},
        }
    )


# ---------------------------------------------------------------------------
# 0-6 校准
# ---------------------------------------------------------------------------
def fit_minmax_calibration(y_train: pd.DataFrame) -> dict:
    """稳健 min-max 校准参数：训练折内 y 的 1%/99% 分位。"""
    return {
        "q01": y_train.quantile(0.01).to_numpy(dtype=float),
        "q99": y_train.quantile(0.99).to_numpy(dtype=float),
    }


def apply_minmax(z: np.ndarray, cal: dict) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    span = cal["q99"] - cal["q01"]
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    s = 6.0 * (z - cal["q01"]) / span
    return np.clip(np.rint(s), 0, 6).astype(int)


def fit_quantile_calibration(y_train: pd.DataFrame) -> dict:
    _, cut = make_grade_labels(y_train, proportions=GRADE_PROPORTIONS)
    return {"cut": cut}


def apply_quantile(z: np.ndarray, cal: dict, targets: list[str]) -> np.ndarray:
    z_df = pd.DataFrame(np.asarray(z, dtype=float), columns=targets)
    return apply_grade_cutpoints(z_df, cal["cut"])


def fit_gaussian_calibration(y_train: pd.DataFrame) -> dict:
    return {
        "mean": y_train.mean().to_numpy(dtype=float),
        "std": y_train.std(ddof=0).to_numpy(dtype=float),
    }


def apply_gaussian(z: np.ndarray, cal: dict) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    std = np.where(cal["std"] < 1e-12, 1.0, cal["std"])
    latent = (z - cal["mean"]) / std
    return np.clip(np.rint(3.0 + 1.5 * latent), 0, 6).astype(int)


def fit_calibration(
    cal_name: str,
    y_train: pd.DataFrame,
    pred_train: np.ndarray,
    targets: list[str],
) -> dict:
    """在训练折的 OOF 预测上拟合 0-6 校准层。

    主方案 isotonic：pred -> y 的单调映射，再用训练折 y 的 q01/q99 做稳健 min-max。
    敏感性方案：raw min-max、quantile rank、quantile map、linear recalibration、Gaussian latent。
    """
    y_arr = y_train.to_numpy(dtype=float)
    pred_train = np.asarray(pred_train, dtype=float)

    if cal_name in ("isotonic", "minmax", "quantile_map", "linear"):
        base = {"minmax": fit_minmax_calibration(y_train)}
    else:
        base = {}

    if cal_name == "isotonic":
        base["models"] = []
        for j in range(len(targets)):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pred_train[:, j], y_arr[:, j])
            base["models"].append(iso)
        return base

    if cal_name == "minmax":
        return base

    if cal_name == "quantile_rank":
        return fit_quantile_calibration(y_train)

    if cal_name == "quantile_map":
        maps = []
        for j in range(len(targets)):
            order = np.argsort(pred_train[:, j])
            xs = pred_train[order, j]
            ps = (np.arange(len(xs)) + 0.5) / len(xs)
            ys = np.sort(y_arr[:, j])
            qs = (np.arange(len(ys)) + 0.5) / len(ys)
            maps.append({"xs": xs, "ps": ps, "qs": qs, "ys": ys})
        base["maps"] = maps
        return base

    if cal_name == "linear":
        base["models"] = [
            LinearRegression().fit(pred_train[:, [j]], y_arr[:, j])
            for j in range(len(targets))
        ]
        return base

    if cal_name == "gaussian":
        return fit_gaussian_calibration(y_train)

    raise ValueError(cal_name)


def apply_calibration(cal_name: str, z: np.ndarray, cal: dict, targets: list[str]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if cal_name == "isotonic":
        y_cal = np.column_stack(
            [iso.predict(z[:, j]) for j, iso in enumerate(cal["models"])]
        )
        return apply_minmax(y_cal, cal["minmax"])
    if cal_name == "minmax":
        return apply_minmax(z, cal["minmax"])
    if cal_name == "quantile_rank":
        return apply_quantile(z, cal, targets)
    if cal_name == "quantile_map":
        y_cal = np.zeros_like(z)
        for j, m in enumerate(cal["maps"]):
            u = np.interp(z[:, j], m["xs"], m["ps"])
            y_cal[:, j] = np.interp(u, m["qs"], m["ys"])
        return apply_minmax(y_cal, cal["minmax"])
    if cal_name == "linear":
        y_cal = np.column_stack(
            [m.predict(z[:, [j]]) for j, m in enumerate(cal["models"])]
        )
        return apply_minmax(y_cal, cal["minmax"])
    if cal_name == "gaussian":
        return apply_gaussian(z, cal)
    raise ValueError(cal_name)


def reference_grades(y: pd.DataFrame, cal_name: str) -> np.ndarray:
    """全数据参考分级，仅用于评估；不参与任何模型拟合。"""
    if cal_name in ("isotonic", "minmax", "quantile_map", "linear"):
        return apply_minmax(y.to_numpy(dtype=float), fit_minmax_calibration(y))
    if cal_name == "quantile_rank":
        grades, _ = make_grade_labels(y, proportions=GRADE_PROPORTIONS)
        return grades.to_numpy(dtype=int)
    if cal_name == "gaussian":
        return apply_gaussian(y.to_numpy(dtype=float), fit_gaussian_calibration(y))
    raise ValueError(cal_name)


def grade_metrics_per_target(true_grades: np.ndarray, pred_grades: np.ndarray) -> dict[str, np.ndarray]:
    true_grades = np.asarray(true_grades, dtype=int)
    pred_grades = np.asarray(pred_grades, dtype=int)
    n_targets = true_grades.shape[1]
    out = {
        "qwk": np.full(n_targets, np.nan),
        "grade_mae": np.full(n_targets, np.nan),
        "exact_hit": np.full(n_targets, np.nan),
        "within1_hit": np.full(n_targets, np.nan),
        "true_mean": np.full(n_targets, np.nan),
        "pred_mean": np.full(n_targets, np.nan),
    }
    for j in range(n_targets):
        t = true_grades[:, j]
        p = pred_grades[:, j]
        out["grade_mae"][j] = np.mean(np.abs(t - p))
        out["exact_hit"][j] = np.mean(t == p)
        out["within1_hit"][j] = np.mean(np.abs(t - p) <= 1)
        out["true_mean"][j] = np.mean(t)
        out["pred_mean"][j] = np.mean(p)
        if np.array_equal(t, p):
            out["qwk"][j] = 1.0
        elif len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
            out["qwk"][j] = np.nan
        else:
            out["qwk"][j] = cohen_kappa_score(
                t, p, weights="quadratic", labels=list(range(7))
            )
    return out


def grade_distribution(grades: np.ndarray) -> np.ndarray:
    """返回 7 个等级的比例，形状 (7,)。"""
    vals = np.asarray(grades, dtype=int).ravel()
    counts = np.bincount(vals, minlength=7).astype(float)
    total = counts.sum()
    return counts / total if total else counts


# ---------------------------------------------------------------------------
# 模型：内层 CV 选参 + 折内拟合
# ---------------------------------------------------------------------------
def tune_ridge(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> tuple[float, float]:
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_alpha, best_score = 1.0, -np.inf
    for alpha in np.logspace(0, 6, 13):
        scores = []
        for it, iv in inner.split(Xtr):
            model = Ridge(alpha=float(alpha))
            model.fit(Xtr[it], ytr[it])
            scores.append(macro_r2_fast(ytr[iv], model.predict(Xtr[iv])))
        score = _nanmean(scores)
        if score > best_score:
            best_alpha, best_score = float(alpha), score
    return best_alpha, best_score


def tune_pls(Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> tuple[int, float]:
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_nc, best_score = 2, -np.inf
    for nc in range(2, 21):
        scores = []
        for it, iv in inner.split(Xtr):
            model = PLSRegression(n_components=nc, scale=False)
            model.fit(Xtr[it], ytr[it])
            scores.append(macro_r2_fast(ytr[iv], model.predict(Xtr[iv])))
        score = _nanmean(scores)
        if score > best_score:
            best_nc, best_score = nc, score
    return best_nc, best_score


def tune_tree_leaf(cls, Xtr: np.ndarray, ytr: np.ndarray, seed: int) -> tuple[int, float]:
    inner = KFold(n_splits=3, shuffle=True, random_state=seed)
    best_leaf, best_score = 1, -np.inf
    for leaf in (1, 2, 4):
        scores = []
        for it, iv in inner.split(Xtr):
            model = cls(
                n_estimators=200,
                min_samples_leaf=leaf,
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(Xtr[it], ytr[it])
            scores.append(macro_r2_fast(ytr[iv], model.predict(Xtr[iv])))
        score = _nanmean(scores)
        if score > best_score:
            best_leaf, best_score = leaf, score
    return best_leaf, best_score


def fit_predict_model(
    name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    seed: int,
    tune: bool = True,
    frozen: dict | None = None,
) -> tuple[np.ndarray, dict]:
    frozen = {} if frozen is None else frozen

    if name == "mean":
        pred = np.tile(np.nanmean(ytr, axis=0), (len(Xte), 1))
        return pred, {"type": "training_mean"}

    if name == "ridge":
        if tune:
            alpha, score = tune_ridge(Xtr, ytr, seed)
        else:
            alpha, score = float(frozen["alpha"]), np.nan
        model = Ridge(alpha=alpha)
        model.fit(Xtr, ytr)
        return model.predict(Xte), {"alpha": float(alpha), "inner_r2": float(score)}

    if name == "pls":
        if tune:
            nc, score = tune_pls(Xtr, ytr, seed)
        else:
            nc, score = int(frozen["n_components"]), np.nan
        model = PLSRegression(n_components=nc, scale=False)
        model.fit(Xtr, ytr)
        return model.predict(Xte), {"n_components": int(nc), "inner_r2": float(score)}

    if name in ("extratrees", "randomforest"):
        cls = ExtraTreesRegressor if name == "extratrees" else RandomForestRegressor
        if tune:
            leaf, score = tune_tree_leaf(cls, Xtr, ytr, seed)
        else:
            leaf, score = int(frozen["min_samples_leaf"]), np.nan
        model = cls(
            n_estimators=500,
            min_samples_leaf=leaf,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(Xtr, ytr)
        return model.predict(Xte), {
            "n_estimators": 500,
            "min_samples_leaf": int(leaf),
            "inner_r2": float(score),
        }

    if name == "lightgbm":
        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            ),
            n_jobs=-1,
        )
        model.fit(Xtr, ytr)
        return model.predict(Xte), {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }

    raise ValueError(name)


def predict_frozen(
    name: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    seed: int,
    fold_key: tuple[int, int],
    hp: dict,
    meta_models: dict,
    targets: list[str],
) -> np.ndarray:
    """缺失实验中用主 CV 选定的超参数重新拟合（不再调参）。"""
    if name == "stack_ridge":
        base_pred = {}
        for m in STACK_MODELS:
            base_pred[m], _ = fit_predict_model(
                m, Xtr, ytr, Xte, seed, tune=False, frozen=hp[(fold_key[0], fold_key[1], m)]
            )
        pred = np.zeros((len(Xte), len(targets)))
        for j, target in enumerate(targets):
            feats = np.column_stack([base_pred[m][:, j] for m in STACK_MODELS])
            pred[:, j] = meta_models[fold_key][target].predict(feats)
        return pred

    if name == "blend_avg":
        pred = np.mean(
            [
                fit_predict_model(
                    m, Xtr, ytr, Xte, seed, tune=False, frozen=hp[(fold_key[0], fold_key[1], m)]
                )[0]
                for m in BASE_MODELS
            ],
            axis=0,
        )
        return pred

    frozen = hp.get((fold_key[0], fold_key[1], name))
    return fit_predict_model(name, Xtr, ytr, Xte, seed, tune=False, frozen=frozen)[0]


# ---------------------------------------------------------------------------
# 主 CV
# ---------------------------------------------------------------------------
def run_main_cv(X: pd.DataFrame, y: pd.DataFrame, targets: list[str]):
    n, p = X.shape
    n_targets = len(targets)
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rkf.split(X))
    oof = {m: np.full((N_REPEATS, n, n_targets), np.nan) for m in ALL_MODELS}
    fold_rows: list[dict] = []
    hp: dict = {}
    per_fold_hp: list[dict] = []
    meta_models: dict = {}

    print(f"[main CV] RepeatedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}), {len(splits)} folds")
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        repeat, fold = divmod(k, N_SPLITS)
        seed = SEED + k
        pre = make_preprocessor()
        Xtr = pre.fit_transform(X.iloc[tr])
        Xte = pre.transform(X.iloc[te])
        ytr = y.iloc[tr].to_numpy(dtype=float)
        yte = y.iloc[te].to_numpy(dtype=float)

        r2s = {}
        for name in ["mean"] + BASE_MODELS:
            pred, info = fit_predict_model(name, Xtr, ytr, Xte, seed, tune=True)
            oof[name][repeat, te] = pred
            hp[(repeat, fold, name)] = info
            record_fold_metrics(
                fold_rows, name, repeat, fold, yte, pred, targets, len(tr), len(te)
            )
            r2s[name] = macro_r2_fast(yte, pred)

        blend = np.mean([oof[m][repeat, te] for m in BASE_MODELS], axis=0)
        oof["blend_avg"][repeat, te] = blend
        record_fold_metrics(
            fold_rows, "blend_avg", repeat, fold, yte, blend, targets, len(tr), len(te)
        )
        r2s["blend_avg"] = macro_r2_fast(yte, blend)

        per_fold_hp.append(
            {
                "repeat": repeat,
                "fold": fold,
                "ridge_alpha": hp[(repeat, fold, "ridge")]["alpha"],
                "pls_n_components": hp[(repeat, fold, "pls")]["n_components"],
                "extratrees_min_samples_leaf": hp[(repeat, fold, "extratrees")]["min_samples_leaf"],
                "randomforest_min_samples_leaf": hp[(repeat, fold, "randomforest")][
                    "min_samples_leaf"
                ],
            }
        )
        summary = " ".join(f"{m}:{r2s[m]:.3f}" for m in ["ridge", "pls", "extratrees", "randomforest", "lightgbm", "blend_avg"])
        print(
            f"  fold {k + 1:02d}/{len(splits)} (rep={repeat},fold={fold}) "
            f"{time.time() - t0:6.1f}s  {summary}"
        )

    # ---- OOF 堆叠：用 OOF 预测训练 Ridge 元学习器 ----
    print("[stack] fitting Ridge meta-learner on OOF predictions (per target, per fold)")
    for k, (tr, te) in enumerate(splits):
        repeat, fold = divmod(k, N_SPLITS)
        pred = np.zeros((len(te), n_targets))
        fold_meta = {}
        for j, target in enumerate(targets):
            ftr = np.column_stack([oof[m][repeat, tr, j] for m in STACK_MODELS])
            fte = np.column_stack([oof[m][repeat, te, j] for m in STACK_MODELS])
            if not (np.isfinite(ftr).all() and np.isfinite(fte).all()):
                raise RuntimeError(f"stack features contain NaN at repeat={repeat}, fold={fold}, target={target}")
            # 元学习器：RidgeCV 在小网格内选 alpha，避免固定 alpha 把融合权重整体收缩到 0。
            meta = RidgeCV(alphas=np.logspace(-3, 2, 11), cv=3)
            meta.fit(ftr, y.iloc[tr, j].to_numpy(dtype=float))
            pred[:, j] = meta.predict(fte)
            fold_meta[target] = meta
        oof["stack_ridge"][repeat, te] = pred
        meta_models[(repeat, fold)] = fold_meta
        record_fold_metrics(
            fold_rows,
            "stack_ridge",
            repeat,
            fold,
            y.iloc[te].to_numpy(dtype=float),
            pred,
            targets,
            len(tr),
            len(te),
        )
    print(f"[main CV] done in {time.time() - t0:.1f}s")

    # ---- 汇总 ----
    cv_df = pd.DataFrame(fold_rows)
    summary = {}
    for model in ALL_MODELS:
        sub = cv_df[(cv_df["model"] == model) & (cv_df["target"] == "MACRO")]
        summary[model] = {
            metric: {
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0,
                "median": float(sub[metric].median()),
            }
            for metric in METRIC_COLS
        }

    best_model = max(
        ALL_MODELS,
        key=lambda m: (summary[m]["r2"]["mean"], -summary[m]["rmse"]["mean"]),
    )

    final_oof = {m: np.nanmean(oof[m], axis=0) for m in ALL_MODELS}
    for m, arr in final_oof.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"final OOF for {m} contains NaN")

    final_oof_summary = {}
    for m in ALL_MODELS:
        mt = per_target_metrics(y.to_numpy(dtype=float), final_oof[m])
        final_oof_summary[m] = macro_from_per_target(mt)

    # 堆叠权重
    stack_weight_summary = None
    if meta_models:
        w = np.array(
            [
                [meta_models[(r, f)][t].coef_ for t in targets]
                for r in range(N_REPEATS)
                for f in range(N_SPLITS)
            ]
        )  # (n_folds, n_targets, n_base)
        stack_weight_summary = {
            "base_models": STACK_MODELS,
            "meta": "RidgeCV(alphas=1e-3..1e2, cv=3), one model per target",
            "mean_alpha": float(
                np.mean(
                    [
                        meta_models[(r, f)][t].alpha_
                        for r in range(N_REPEATS)
                        for f in range(N_SPLITS)
                        for t in targets
                    ]
                )
            ),
            "mean_weights": {
                m: float(w[:, :, i].mean()) for i, m in enumerate(STACK_MODELS)
            },
            "std_weights": {
                m: float(w[:, :, i].std(ddof=0)) for i, m in enumerate(STACK_MODELS)
            },
            "mean_intercept": float(
                np.mean(
                    [
                        meta_models[(r, f)][t].intercept_
                        for r in range(N_REPEATS)
                        for f in range(N_SPLITS)
                        for t in targets
                    ]
                )
            ),
            "per_target_mean_weights": {
                t: {m: float(w[:, j, i].mean()) for i, m in enumerate(STACK_MODELS)}
                for j, t in enumerate(targets)
            },
        }

    return {
        "splits": splits,
        "oof": oof,
        "final_oof": final_oof,
        "cv_df": cv_df,
        "summary": summary,
        "final_oof_summary": final_oof_summary,
        "best_model": best_model,
        "hp": hp,
        "per_fold_hp": per_fold_hp,
        "meta_models": meta_models,
        "stack_weight_summary": stack_weight_summary,
        "runtime_main": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 分级评估
# ---------------------------------------------------------------------------
def evaluate_grades(cv: dict, y: pd.DataFrame, targets: list[str]) -> dict:
    best = cv["best_model"]
    oof = cv["oof"]
    splits = cv["splits"]
    n, n_targets = y.shape

    true_grades = {cal: reference_grades(y, cal) for cal in CALIBRATIONS}
    pooled_pred = {cal: np.full((N_REPEATS, n, n_targets), -1, dtype=int) for cal in CALIBRATIONS}
    per_fold = []

    for k, (tr, te) in enumerate(splits):
        repeat, fold = divmod(k, N_SPLITS)
        pred_train = oof[best][repeat, tr]   # 训练折样本的主 CV OOF 预测
        pred_test = oof[best][repeat, te]
        y_train = y.iloc[tr]
        for cal in CALIBRATIONS:
            fitted = fit_calibration(cal, y_train, pred_train, targets)
            pg = apply_calibration(cal, pred_test, fitted, targets)
            pooled_pred[cal][repeat, te] = pg
            gm = grade_metrics_per_target(true_grades[cal][te], pg)
            per_fold.append(
                {
                    "calibration": cal,
                    "repeat": repeat,
                    "fold": fold,
                    "qwk": _nanmean(gm["qwk"]),
                    "grade_mae": _nanmean(gm["grade_mae"]),
                    "exact_hit": _nanmean(gm["exact_hit"]),
                    "within1_hit": _nanmean(gm["within1_hit"]),
                    "n_valid_qwk": int(np.isfinite(gm["qwk"]).sum()),
                }
            )

    # pooled 指标（每条样本在每个 repeat 出现一次）
    rows = []
    pooled_metrics = {}
    for cal in CALIBRATIONS:
        pg = pooled_pred[cal].reshape(-1, n_targets)
        tg = np.tile(true_grades[cal], (N_REPEATS, 1))
        gm = grade_metrics_per_target(tg, pg)
        pooled_metrics[cal] = gm
        for j, target in enumerate(targets + ["MACRO"]):
            if target == "MACRO":
                metrics = {k: _nanmean(v) for k, v in gm.items()}
                t_dist = np.mean([grade_distribution(tg[:, jj]) for jj in range(n_targets)], axis=0)
                p_dist = np.mean([grade_distribution(pg[:, jj]) for jj in range(n_targets)], axis=0)
            else:
                metrics = {k: float(v[j]) for k, v in gm.items()}
                t_dist = grade_distribution(tg[:, j])
                p_dist = grade_distribution(pg[:, j])
            rows.append(
                {
                    "scope": "pooled_folds",
                    "calibration": cal,
                    "target": target,
                    **metrics,
                    **{f"true_{g}": float(t_dist[g]) for g in range(7)},
                    **{f"pred_{g}": float(p_dist[g]) for g in range(7)},
                }
            )

    # 最终 OOF 分级：主方案 isotonic；每个样本取 3 个 repeat 的折内校准分级中位数。
    # 不使用全数据 y 的分位数，避免把目标分布信息带进预测分级。
    final_grades = np.rint(np.median(pooled_pred["isotonic"], axis=0)).astype(int)
    gm_final = grade_metrics_per_target(true_grades["isotonic"], final_grades)
    for j, target in enumerate(targets + ["MACRO"]):
        if target == "MACRO":
            metrics = {k: _nanmean(v) for k, v in gm_final.items()}
            t_dist = np.mean([grade_distribution(true_grades["isotonic"][:, jj]) for jj in range(n_targets)], axis=0)
            p_dist = np.mean([grade_distribution(final_grades[:, jj]) for jj in range(n_targets)], axis=0)
        else:
            metrics = {k: float(v[j]) for k, v in gm_final.items()}
            t_dist = grade_distribution(true_grades["isotonic"][:, j])
            p_dist = grade_distribution(final_grades[:, j])
        rows.append(
            {
                "scope": "oof_final",
                "calibration": "isotonic",
                "target": target,
                **metrics,
                **{f"true_{g}": float(t_dist[g]) for g in range(7)},
                **{f"pred_{g}": float(p_dist[g]) for g in range(7)},
            }
        )

    grade_df = pd.DataFrame(rows)

    # 折间 macro 统计
    per_fold_df = pd.DataFrame(per_fold)
    fold_summary = {}
    for cal in CALIBRATIONS:
        sub = per_fold_df[per_fold_df["calibration"] == cal]
        fold_summary[cal] = {
            metric: {
                "mean": float(sub[metric].mean()),
                "std": float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0,
            }
            for metric in ["qwk", "grade_mae", "exact_hit", "within1_hit", "n_valid_qwk"]
        }

    return {
        "grade_df": grade_df,
        "per_fold_df": per_fold_df,
        "fold_summary": fold_summary,
        "final_grades": final_grades,
        "true_grades_primary": true_grades["isotonic"],
        "pooled_metrics": pooled_metrics,
    }


# ---------------------------------------------------------------------------
# 缺失模拟
# ---------------------------------------------------------------------------
def make_mcar_mask(shape: tuple[int, int], rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) < rate


def make_mar_mask(X: pd.DataFrame, driver: str, seed: int) -> np.ndarray:
    """按 driver 的分位箱决定每个 cell 的缺失概率（简单 MAR）。"""
    rng = np.random.default_rng(seed)
    values = X[driver].to_numpy(dtype=float)
    bins = pd.qcut(values, q=4, labels=False, duplicates="drop")
    probs_by_bin = np.array([0.05, 0.15, 0.30, 0.50])
    # qcut 若因并列少于一箱，用实际箱数截取概率
    n_bins = int(np.nanmax(bins)) + 1
    probs = probs_by_bin[:n_bins]
    mask = np.zeros(X.shape, dtype=bool)
    for j, col in enumerate(X.columns):
        if col == driver:
            continue
        p = probs[bins.astype(int)]
        mask[:, j] = rng.random(len(X)) < p
    return mask


def run_missingness(cv: dict, X: pd.DataFrame, y: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    best = cv["best_model"]
    hp = cv["hp"]
    meta_models = cv["meta_models"]
    splits = cv["splits"]
    n_rep = N_REPEATS_MISSING

    # 自然缺失基线：与缺失实验相同的 repeat 子集
    base_rows = [
        r
        for r in cv["cv_df"].to_dict("records")
        if r["model"] == best and r["target"] == "MACRO" and r["repeat"] < n_rep
    ]
    base_df = pd.DataFrame(base_rows)
    base_r2 = float(base_df["r2"].mean())
    base_r2_std = float(base_df["r2"].std(ddof=1))
    base_rmse = float(base_df["rmse"].mean())
    base_rmse_std = float(base_df["rmse"].std(ddof=1))

    natural_na = X.isna().to_numpy()
    n_cells = X.size
    natural_rate = float(natural_na.sum() / n_cells)

    rows = [
        {
            "scenario": "natural",
            "missing_type": "natural",
            "nominal_rate": 0.0,
            "achieved_additional_rate": 0.0,
            "total_missing_rate": natural_rate,
            "model": best,
            "n_folds": len(base_df),
            "r2_mean": base_r2,
            "r2_std": base_r2_std,
            "rmse_mean": base_rmse,
            "rmse_std": base_rmse_std,
            "delta_r2": 0.0,
            "delta_r2_pct": 0.0,
            "delta_rmse": 0.0,
            "delta_rmse_pct": 0.0,
        }
    ]

    scenarios = [("MCAR", rate) for rate in (0.05, 0.10, 0.20, 0.30)] + [("MAR", None)]
    driver = "ethanol..v.v."
    if driver not in X.columns:
        driver = X.columns[0]

    print(f"[missingness] best model = {best}; repeats = {n_rep}; natural NA = {natural_na.sum()}/{n_cells}")
    t0 = time.time()
    for scenario, rate in scenarios:
        fold_r2, fold_rmse = [], []
        achieved = []
        for repeat in range(n_rep):
            if scenario == "MCAR":
                mask = make_mcar_mask(X.shape, float(rate), SEED + 1000 * repeat + int(round(float(rate) * 100)))
            else:
                mask = make_mar_mask(X, driver, SEED + 5000 + repeat)
            additional = mask & ~natural_na
            achieved.append(float(additional.sum() / n_cells))
            Xm = X.copy()
            Xm.values[mask] = np.nan

            for fold in range(N_SPLITS):
                tr, te = splits[repeat * N_SPLITS + fold]
                pre = make_preprocessor()
                Xtr = pre.fit_transform(Xm.iloc[tr])
                Xte = pre.transform(Xm.iloc[te])
                ytr = y.iloc[tr].to_numpy(dtype=float)
                yte = y.iloc[te].to_numpy(dtype=float)
                pred = predict_frozen(
                    best,
                    Xtr,
                    ytr,
                    Xte,
                    SEED + repeat * N_SPLITS + fold,
                    (repeat, fold),
                    hp,
                    meta_models,
                    targets,
                )
                if not np.isfinite(pred).all():
                    raise RuntimeError(f"missingness prediction non-finite: {scenario}, repeat={repeat}, fold={fold}")
                mt = per_target_metrics(yte, pred)
                fold_r2.append(_nanmean(mt["r2"]))
                fold_rmse.append(_nanmean(mt["rmse"]))

        r2_mean = float(np.mean(fold_r2))
        r2_std = float(np.std(fold_r2, ddof=1))
        rmse_mean = float(np.mean(fold_rmse))
        rmse_std = float(np.std(fold_rmse, ddof=1))
        rows.append(
            {
                "scenario": scenario if scenario == "MAR" else f"MCAR {int(round(rate * 100))}%",
                "missing_type": scenario,
                "nominal_rate": float(rate) if rate is not None else np.nan,
                "achieved_additional_rate": float(np.mean(achieved)),
                "total_missing_rate": float(natural_rate + np.mean(achieved)),
                "model": best,
                "n_folds": len(fold_r2),
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "delta_r2": r2_mean - base_r2,
                "delta_r2_pct": 100.0 * (r2_mean - base_r2) / abs(base_r2) if abs(base_r2) > 1e-12 else np.nan,
                "delta_rmse": rmse_mean - base_rmse,
                "delta_rmse_pct": 100.0 * (rmse_mean - base_rmse) / base_rmse if base_rmse > 1e-12 else np.nan,
            }
        )
        nominal_str = f"{rate:.2f}" if rate is not None else "MAR"
        print(
            f"  {scenario:4s} nominal={nominal_str:>4s} "
            f"achieved={np.mean(achieved) * 100:5.2f}%  "
            f"R2={r2_mean:.4f}±{r2_std:.4f}  RMSE={rmse_mean:.4f}±{rmse_std:.4f}  "
            f"ΔR2={r2_mean - base_r2:+.4f}  ΔRMSE={100 * (rmse_mean - base_rmse) / base_rmse:+.2f}%  "
            f"({time.time() - t0:.1f}s)"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 风格先验消融：chemical-only vs +style one-hot vs +style target encoding
# ---------------------------------------------------------------------------
def _onehot_style_features(styles_train, styles_test, all_styles):
    """折内 fit one-hot 编码；未见过的 style 在测试折中编码为全 0。"""
    enc = OneHotEncoder(
        categories=[all_styles],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=float,
    )
    enc.fit(np.asarray(styles_train).reshape(-1, 1))
    ftr = enc.transform(np.asarray(styles_train).reshape(-1, 1))
    fte = enc.transform(np.asarray(styles_test).reshape(-1, 1))
    return ftr, fte, [f"style_{s}" for s in all_styles]


def _target_encode_style(styles_train, styles_test, y_train, targets, smoothing=10.0):
    """折内 target encoding：训练样本用 leave-one-out + 平滑，测试样本用全训练折均值。

    返回 50 维（每个感官目标一维）风格均值特征。
    """
    y_train = np.asarray(y_train, dtype=float)
    styles_train = np.asarray(styles_train)
    styles_test = np.asarray(styles_test)
    df = pd.DataFrame(y_train, columns=targets)
    df["_style"] = styles_train
    sums = df.groupby("_style")[targets].sum()
    counts = df.groupby("_style").size()
    global_mean = y_train.mean(axis=0)

    te_train = np.zeros((len(styles_train), len(targets)))
    te_test = np.zeros((len(styles_test), len(targets)))
    for j, target in enumerate(targets):
        s_tr = sums[target].reindex(styles_train).to_numpy(dtype=float)
        c_tr = counts.reindex(styles_train).to_numpy(dtype=float)
        te_train[:, j] = (
            s_tr - y_train[:, j] + smoothing * global_mean[j]
        ) / (c_tr - 1.0 + smoothing)

        s_te = np.nan_to_num(
            sums[target].reindex(styles_test).to_numpy(dtype=float), nan=0.0
        )
        c_te = np.nan_to_num(
            counts.reindex(styles_test).to_numpy(dtype=float), nan=0.0
        )
        te_test[:, j] = (s_te + smoothing * global_mean[j]) / (c_te + smoothing)
    return te_train, te_test, [f"te_{t}" for t in targets]


def run_style_ablation(
    cv: dict,
    X: pd.DataFrame,
    y: pd.DataFrame,
    styles,
    targets: list[str],
    models: tuple[str, ...] = ("extratrees", "ridge"),
    feature_sets: tuple[str, ...] = ("chemical", "onehot", "targetenc"),
    n_repeats: int = 1,
) -> pd.DataFrame:
    """同折同超参数消融（不再调参）：默认第一轮 repeat 的 5 折。"""
    styles = np.asarray(styles)
    all_styles = sorted(pd.unique(styles).tolist())
    splits = cv["splits"][: N_SPLITS * n_repeats]
    hp = cv["hp"]
    meta_models = cv["meta_models"]
    rows = []
    fold_store = {}
    print(
        f"[style ablation] models={models}, feature_sets={feature_sets}, "
        f"folds={len(splits)}, styles={len(all_styles)}"
    )
    t0 = time.time()
    for feat_mode in feature_sets:
        for model_name in models:
            fold_metrics = []
            for k, (tr, te) in enumerate(splits):
                repeat, fold = divmod(k, N_SPLITS)
                x_tr = X.iloc[tr].reset_index(drop=True)
                x_te = X.iloc[te].reset_index(drop=True)
                if feat_mode == "chemical":
                    x_tr_aug, x_te_aug = x_tr, x_te
                elif feat_mode == "onehot":
                    ftr, fte, names = _onehot_style_features(
                        styles[tr], styles[te], all_styles
                    )
                    x_tr_aug = pd.concat(
                        [x_tr, pd.DataFrame(ftr, columns=names)], axis=1
                    )
                    x_te_aug = pd.concat(
                        [x_te, pd.DataFrame(fte, columns=names)], axis=1
                    )
                else:
                    ftr, fte, names = _target_encode_style(
                        styles[tr], styles[te], y.iloc[tr].to_numpy(dtype=float), targets
                    )
                    x_tr_aug = pd.concat(
                        [x_tr, pd.DataFrame(ftr, columns=names)], axis=1
                    )
                    x_te_aug = pd.concat(
                        [x_te, pd.DataFrame(fte, columns=names)], axis=1
                    )
                pre = make_preprocessor()
                x_tr_p = pre.fit_transform(x_tr_aug)
                x_te_p = pre.transform(x_te_aug)
                pred = predict_frozen(
                    model_name,
                    x_tr_p,
                    y.iloc[tr].to_numpy(dtype=float),
                    x_te_p,
                    SEED + k,
                    (repeat, fold),
                    hp,
                    meta_models,
                    targets,
                )
                m = per_target_metrics(y.iloc[te].to_numpy(dtype=float), pred)
                fold_metrics.append({k: _nanmean(m[k]) for k in METRIC_COLS})
            fold_store[(model_name, feat_mode)] = fold_metrics
            means = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in METRIC_COLS}
            stds = {k: float(np.std([fm[k] for fm in fold_metrics], ddof=1)) for k in METRIC_COLS}
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feat_mode,
                    "n_folds": len(splits),
                    **{f"{k}_mean": means[k] for k in METRIC_COLS},
                    **{f"{k}_std": stds[k] for k in METRIC_COLS},
                }
            )
            print(
                f"  {feat_mode:9s} {model_name:12s} "
                f"R2={means['r2']:.4f}±{stds['r2']:.4f}  "
                f"Spearman={means['spearman']:.4f}  ({time.time() - t0:.1f}s)"
            )

    df = pd.DataFrame(rows)
    for col in [
        "delta_r2_vs_chemical",
        "delta_r2_std",
        "delta_spearman_vs_chemical",
        "delta_spearman_std",
    ]:
        df[col] = np.nan
    for model_name in models:
        if (model_name, "chemical") not in fold_store:
            continue
        base = fold_store[(model_name, "chemical")]
        base_r2 = np.array([fm["r2"] for fm in base])
        base_sp = np.array([fm["spearman"] for fm in base])
        for feat_mode in feature_sets:
            key = (model_name, feat_mode)
            if key not in fold_store:
                continue
            cur = fold_store[key]
            d_r2 = np.array([fm["r2"] for fm in cur]) - base_r2
            d_sp = np.array([fm["spearman"] for fm in cur]) - base_sp
            mask = (df["model"] == model_name) & (df["feature_set"] == feat_mode)
            df.loc[mask, "delta_r2_vs_chemical"] = float(np.mean(d_r2))
            df.loc[mask, "delta_r2_std"] = (
                float(np.std(d_r2, ddof=1)) if len(d_r2) > 1 else 0.0
            )
            df.loc[mask, "delta_spearman_vs_chemical"] = float(np.mean(d_sp))
            df.loc[mask, "delta_spearman_std"] = (
                float(np.std(d_sp, ddof=1)) if len(d_sp) > 1 else 0.0
            )
    return df


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _save(fig, name: str):
    fig.savefig(IMG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(IMG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(summary: dict, best_model: str):
    order = sorted(ALL_MODELS, key=lambda m: summary[m]["r2"]["mean"])
    metrics = [("r2", "Macro R²"), ("rmse", "Macro RMSE"), ("spearman", "Macro Spearman ρ")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (metric, label) in zip(axes, metrics):
        means = np.array([summary[m][metric]["mean"] for m in order])
        stds = np.array([summary[m][metric]["std"] for m in order])
        stds_plot = np.nan_to_num(stds, nan=0.0)
        colors = ["#d62728" if m == best_model else "#4c72b0" for m in order]
        ax.barh(order, means, xerr=stds_plot, color=colors, alpha=0.85, capsize=3)
        ax.axvline(0, color="0.45", lw=0.8, ls="--")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_ylim(-0.6, len(order) - 0.4)
        span = max(np.nanmax(np.abs(means)) + np.nanmax(stds_plot), 1e-6)
        for i, v in enumerate(means):
            if not np.isfinite(v):
                ax.text(0.01 * span, i, "n/a", va="center", ha="left", fontsize=7, color="0.4")
                continue
            offset = 0.02 * span
            if v >= 0:
                x_text = v + stds_plot[i] + offset
                ha, color = "left", "black"
            else:
                x_text = 0.01 * span
                ha, color = "left", "black"
            ax.text(x_text, i, f"{v:.3f}", va="center", ha=ha, fontsize=8, color=color)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"Q1 model comparison: {N_SPLITS}-fold × {N_REPEATS} repeats ({N_SPLITS * N_REPEATS} folds; mean ± 1 fold SD); best = {best_model}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "q1_model_comparison")


def plot_target_heatmap(cv_df: pd.DataFrame, best_model: str):
    per_target = (
        cv_df[cv_df["target"] != "MACRO"]
        .groupby(["target", "model"])["r2"]
        .mean()
        .unstack()
        .reindex(columns=ALL_MODELS)
    )
    order = per_target[best_model].sort_values().index
    mat = per_target.loc[order, ALL_MODELS].to_numpy(dtype=float)
    vmin = min(-0.2, float(np.nanmin(mat)))
    vmax = max(0.8, float(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(10, 16))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(ALL_MODELS, rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for lbl, target in zip(ax.get_yticklabels(), order):
        if target.startswith("A_"):
            lbl.set_color("#1f77b4")
        elif target.startswith("F_"):
            lbl.set_color("#d95f02")
        else:
            lbl.set_color("#2ca02c")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color="black" if -0.05 < v < 0.55 else "white",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("R² (mean over folds)")
    ax.set_title(
        f"Per-target cross-validated R² by model (mean over {N_SPLITS}-fold × {N_REPEATS} repeats)\n"
        f"rows sorted by {best_model}; blue=Aroma, orange=Flavour, green=other",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "q1_target_r2_heatmap")


def plot_missingness(missing_df: pd.DataFrame, best_model: str):
    natural = missing_df[missing_df["missing_type"] == "natural"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, label in zip(axes, ["r2", "rmse"], ["Macro R²", "Macro RMSE"]):
        ax.axhline(
            natural[f"{metric}_mean"],
            color="0.45",
            ls="--",
            lw=1.0,
            label=f"Natural missing ({natural['total_missing_rate'] * 100:.2f}%)",
        )
        ax.plot(
            [0],
            [natural[f"{metric}_mean"]],
            marker="D",
            color="0.45",
            ms=5,
            ls="none",
            zorder=5,
        )
        mcar = missing_df[missing_df["missing_type"] == "MCAR"].sort_values("nominal_rate")
        ax.errorbar(
            mcar["nominal_rate"] * 100,
            mcar[f"{metric}_mean"],
            yerr=mcar[f"{metric}_std"],
            marker="o",
            ms=5,
            capsize=3,
            lw=1.8,
            label="MCAR",
        )
        mar = missing_df[missing_df["missing_type"] == "MAR"]
        if len(mar):
            ax.errorbar(
                mar["achieved_additional_rate"] * 100,
                mar[f"{metric}_mean"],
                yerr=mar[f"{metric}_std"],
                marker="s",
                ms=8,
                capsize=3,
                lw=1.5,
                label="MAR (driver-quantile)",
            )
        ax.set_xlabel("Additional missing rate (%)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs missingness")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Missing-data robustness of best model: {best_model}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "q1_missingness")


def plot_grade(grade_eval: dict, grade_fold_summary: dict):
    true_g = grade_eval["true_grades_primary"].ravel()
    pred_g = grade_eval["final_grades"].ravel()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    # A. 分级分布
    x = np.arange(7)
    width = 0.38
    axes[0].bar(x - width / 2, grade_distribution(true_g), width, label="True (reference)", color="#4c72b0")
    axes[0].bar(x + width / 2, grade_distribution(pred_g), width, label="Predicted (OOF)", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Grade (0–6)")
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Grade distribution (isotonic, primary)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    # B. 混淆矩阵（行归一化）
    n_targets = grade_eval["true_grades_primary"].shape[1]
    cm = np.zeros((7, 7), dtype=float)
    for j in range(n_targets):
        for a, b in zip(grade_eval["true_grades_primary"][:, j], grade_eval["final_grades"][:, j]):
            cm[a, b] += 1
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    im = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(x)
    axes[1].set_yticks(x)
    axes[1].set_xlabel("Predicted grade")
    axes[1].set_ylabel("True grade")
    axes[1].set_title("Confusion matrix (row-normalized)")
    for i in range(7):
        for j in range(7):
            axes[1].text(
                j,
                i,
                f"{cm_norm[i, j] * 100:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # C. 六种校准的 QWK / MAE
    labels = {
        "isotonic": "Isotonic\n(primary)",
        "minmax": "Raw\nmin-max",
        "quantile_rank": "Quantile\nrank",
        "quantile_map": "Quantile\nmap",
        "linear": "Linear\nrecal.",
        "gaussian": "Gaussian\n(3+1.5z)",
    }
    colors = ["#d62728", "#4c72b0", "#55a868", "#8172b3", "#ccb974", "#64b5cd"]
    x3 = np.arange(len(CALIBRATIONS))
    means = np.array([grade_fold_summary[c]["qwk"]["mean"] for c in CALIBRATIONS])
    stds = np.array([grade_fold_summary[c]["qwk"]["std"] for c in CALIBRATIONS])
    bars = axes[2].bar(x3, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels([labels[c] for c in CALIBRATIONS], fontsize=8)
    axes[2].axhline(0, color="0.45", lw=0.8, ls="--")
    axes[2].set_ylabel("Macro QWK")
    axes[2].set_title("Grade agreement by calibration")
    axes[2].grid(axis="y", alpha=0.25)
    y_top = max(0.05, float(np.nanmax(means + stds))) * 1.45
    axes[2].set_ylim(0.0, y_top)
    for bar, c, v, e in zip(bars, CALIBRATIONS, means, stds):
        mae = grade_fold_summary[c]["grade_mae"]["mean"]
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            v + e + 0.015 * y_top,
            f"{v:.3f}\nMAE {mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.suptitle("0–6 grading: isotonic primary calibration and sensitivity analyses", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Reference grades: robust min-max for isotonic/minmax/quantile_map/linear; quantile-rank for quantile_rank; Gaussian for gaussian.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, "q1_grade_confusion")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sanitize_json(obj):
    """把 NaN/Inf 转成 None，保证输出是标准 JSON。"""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(v) for v in obj.tolist()]
    return obj


def main():
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("Q1: chemical features -> sensory dimensions, missingness-robust scoring")
    print("=" * 78)
    D = load_all()
    X = D["X"]
    y = D["y"]
    targets = list(y.columns)
    print(f"X: {X.shape}, y: {y.shape}, natural NA cells: {int(X.isna().sum().sum())}")

    # ---- 主 CV ----
    cv = run_main_cv(X, y, targets)
    best = cv["best_model"]
    print(f"[best] {best}  macro R2 = {cv['summary'][best]['r2']['mean']:.4f} ± {cv['summary'][best]['r2']['std']:.4f}")

    # ---- 分级 ----
    t_grade = time.time()
    grade_eval = evaluate_grades(cv, y, targets)
    runtime_grade = time.time() - t_grade
    print(f"[grade] evaluated in {runtime_grade:.1f}s; primary = isotonic")
    print(
        "  fold macro QWK: "
        + "  ".join(
            f"{c}={grade_eval['fold_summary'][c]['qwk']['mean']:.3f}±{grade_eval['fold_summary'][c]['qwk']['std']:.3f}"
            for c in CALIBRATIONS
        )
    )

    # ---- 缺失鲁棒性 ----
    t_miss = time.time()
    missing_df = run_missingness(cv, X, y, targets)
    runtime_missing = time.time() - t_miss

    # ---- 风格先验消融 ----
    t_style = time.time()
    style_main = run_style_ablation(
        cv,
        X,
        y,
        D["styles"],
        targets,
        models=("extratrees", "ridge"),
        n_repeats=N_REPEATS,
    )
    style_blend = run_style_ablation(
        cv,
        X,
        y,
        D["styles"],
        targets,
        models=("blend_avg",),
        feature_sets=("chemical", "targetenc"),
        n_repeats=N_REPEATS,
    )
    style_df = pd.concat([style_main, style_blend], ignore_index=True)
    runtime_style = time.time() - t_style

    # ---- 输出 CSV ----
    cv_df = cv["cv_df"].copy()
    cv_df = cv_df.sort_values(["model", "repeat", "fold", "target"]).reset_index(drop=True)
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False, float_format="%.6f")

    # OOF 预测：最佳模型连续预测 + isotonic 0-6 分级
    final_pred = cv["final_oof"][best]
    final_grades = grade_eval["final_grades"]
    pred_df = pd.DataFrame(final_pred, columns=[f"pred_{t}" for t in targets])
    grade_df = pd.DataFrame(final_grades, columns=[f"grade_{t}" for t in targets])
    oof_out = pd.concat(
        [
            pd.DataFrame(
                {
                    "beer": D["meta"]["beer"].to_numpy(),
                    "tasting_category_fine": D["styles"],
                }
            ),
            pred_df,
            grade_df,
        ],
        axis=1,
    )
    oof_out.to_csv(OUT_DIR / "oof_predictions.csv", index=False, float_format="%.6f")

    missing_df.to_csv(OUT_DIR / "missingness.csv", index=False, float_format="%.6f")
    grade_eval["grade_df"].to_csv(OUT_DIR / "grade_metrics.csv", index=False, float_format="%.6f")
    style_df.to_csv(OUT_DIR / "style_ablation.csv", index=False, float_format="%.6f")

    # ---- 图 ----
    t_plot = time.time()
    plot_model_comparison(cv["summary"], best)
    plot_target_heatmap(cv["cv_df"], best)
    plot_missingness(missing_df, best)
    plot_grade(grade_eval, grade_eval["fold_summary"])
    runtime_plot = time.time() - t_plot
    print(f"[plots] 4 figures saved to img/q1_*.png/.pdf in {runtime_plot:.1f}s")

    # ---- model_summary.json ----
    natural_na_series = X.isna().sum()
    natural_missing = {
        "n_cells": int(X.size),
        "n_missing": int(X.isna().sum().sum()),
        "pct_cells": float(100.0 * X.isna().sum().sum() / X.size),
        "n_columns_affected": int((natural_na_series > 0).sum()),
        "by_column": {
            str(c): int(v) for c, v in natural_na_series[natural_na_series > 0].sort_values(ascending=False).items()
        },
        "note": "All main-CV metrics use this natural missingness; median imputation is fit inside each training fold.",
    }

    summary_payload = {
        "meta": {
            "task": "Q1: sensory scores from chemical features under missing data",
            "seed": SEED,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_targets": len(targets),
            "targets": targets,
            "cv": f"RepeatedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={SEED})",
            "final_cv": f"{N_SPLITS}-fold x {N_REPEATS} repeats (15 folds total); metrics are mean ± std across folds",
            "preprocessing": "data_utils.make_preprocessor(): SimpleImputer(median) + QuantileTransformer(n_quantiles=200, output_distribution=normal); fit on each training fold only.",
            "y_note": "4.csv targets are continuous column-standardized values, not raw 0-6 scores; grades are calibrated post hoc.",
            "grade_primary": "isotonic: per target, IsotonicRegression(out_of_bounds='clip') fit on training-fold OOF predictions -> training-fold y; then robust min-max with training-fold y q01/q99 -> clip [0,6] -> rint",
            "grade_sensitivity": {
                "minmax": "raw robust min-max: q01/q99 of training-fold y -> s=6*(z-q01)/(q99-q01), clip [0,6], rint",
                "quantile_rank": "data_utils.make_grade_labels with proportions (0.025,0.075,0.15,0.25,0.25,0.15,0.10), cutpoints fit on training fold",
                "quantile_map": "empirical CDF of training-fold OOF predictions mapped to training-fold y quantiles, then robust min-max",
                "linear": "per target y ~ a + b*pred fit on training-fold OOF predictions, then robust min-max",
                "gaussian": "z=(pred-mean_train)/std_train; grade=rint(3+1.5z), clip [0,6]",
            },
            "models": {
                "baselines": ["mean", "ridge", "pls"],
                "main": ["extratrees", "randomforest", "lightgbm"],
                "fusion": ["blend_avg (equal weights over base models)", "stack_ridge (RidgeCV meta-learner on OOF predictions)"],
            },
        },
        "natural_missingness": natural_missing,
        "models": {
            m: {
                "macro_mean_std_over_folds": cv["summary"][m],
                "final_oof_macro": cv["final_oof_summary"][m],
            }
            for m in ALL_MODELS
        },
        "best_model": {
            "name": best,
            "criterion": "highest mean macro R2 over folds; RMSE used only as tie-breaker",
            "macro_mean_std_over_folds": cv["summary"][best],
            "final_oof_macro": cv["final_oof_summary"][best],
        },
        "hyperparameters": cv["per_fold_hp"],
        "stacking": cv["stack_weight_summary"],
        "grade": {
            "primary_calibration": "isotonic",
            "macro_mean_std_over_folds": grade_eval["fold_summary"],
            "pooled_fold_macro": {
                cal: {k: _nanmean(v) for k, v in grade_eval["pooled_metrics"][cal].items()}
                for cal in CALIBRATIONS
            },
            "final_oof_isotonic_macro": {
                row["target"]: {k: row[k] for k in ["qwk", "grade_mae", "exact_hit", "within1_hit", "true_mean", "pred_mean"]}
                for row in grade_eval["grade_df"][
                    (grade_eval["grade_df"]["scope"] == "oof_final")
                ].to_dict("records")
                if row["target"] == "MACRO"
            },
            "final_oof_note": "Final OOF grades = median of the 3 repeat-level, training-fold-calibrated isotonic grades; no full-data target quantiles are used.",
            "distribution_columns_note": "true_0..6 / pred_0..6 in grade_metrics.csv are within-target proportions.",
        },
        "style_ablation": {
            "note": "Same 5-fold × 3-repeat splits (15 folds) and frozen chemical-only hyperparameters; feature sets: chemical (231), +style one-hot (22), +style target encoding (50; LOO + smoothing=10 within each training fold). Models: ExtraTrees and Ridge (all three feature sets), blend_avg (chemical and target encoding only, for runtime). Delta columns are paired per-fold differences vs the chemical-only baseline (mean ± std across folds).",
            "rows": style_df.astype(object).where(pd.notna(style_df), None).to_dict("records"),
        },
        "missingness": {
            "best_model": best,
            "n_repeats_used": N_REPEATS_MISSING,
            "mar_driver": "ethanol..v.v.",
            "mar_definition": "For each cell (except driver), p(missing) is 0.05/0.15/0.30/0.50 by the driver's quartile; independent Bernoulli draws.",
            "rows": missing_df.astype(object).where(pd.notna(missing_df), None).to_dict("records"),
        },
        "runtime_sec": {
            "main_cv": float(cv["runtime_main"]),
            "grade_eval": float(runtime_grade),
            "missingness": float(runtime_missing),
            "style_ablation": float(runtime_style),
            "plots": float(runtime_plot),
            "total": float(time.time() - t_start),
        },
        "notes": [
            f"Final CV: RepeatedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={SEED}); 15 folds total.",
            "No preprocessing is fit on the full dataset; every imputer/quantile transformer and style encoder is fit on the training fold only.",
            "The isotonic calibration layer is fit on the training-fold OOF predictions (main-CV OOF rows), never on the outer test fold.",
            "Stacking uses OOF predictions: for each outer fold, the RidgeCV meta-learner is fit on the OOF rows of the training samples of the same repeat and evaluated on the OOF rows of the test samples.",
            "Missingness experiments use the final 5-fold split with 2 repeats and refit base models with hyperparameters selected on complete data (nested tuning is not repeated) to keep runtime bounded.",
            "Natural missingness (73 NA cells) is included in every main-CV fold; the MCAR/MAR masks are applied on top of it.",
        ],
    }
    with open(OUT_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            _sanitize_json(summary_payload),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )

    # ---- 终端摘要 ----
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    for m in sorted(ALL_MODELS, key=lambda z: cv["summary"][z]["r2"]["mean"], reverse=True):
        s = cv["summary"][m]
        print(
            f"  {m:14s} R2={s['r2']['mean']:.4f}±{s['r2']['std']:.4f}  "
            f"RMSE={s['rmse']['mean']:.4f}  Spearman={s['spearman']['mean']:.4f}  EV={s['explained_variance']['mean']:.4f}"
        )
    print(f"\n  best model: {best}")
    gs = grade_eval["fold_summary"]
    for cal in CALIBRATIONS:
        print(
            f"  grade[{cal:13s}] QWK={gs[cal]['qwk']['mean']:.4f}±{gs[cal]['qwk']['std']:.4f}  "
            f"MAE={gs[cal]['grade_mae']['mean']:.4f}  exact={gs[cal]['exact_hit']['mean']:.4f}  "
            f"±1={gs[cal]['within1_hit']['mean']:.4f}"
        )
    print("\n  style ablation (same 15 folds, frozen hyperparameters; paired deltas):")
    for _, r in style_df.iterrows():
        print(
            f"    {r['feature_set']:9s} {r['model']:12s} "
            f"R2={r['r2_mean']:.4f}±{r['r2_std']:.4f}  "
            f"ΔR2={r['delta_r2_vs_chemical']:+.4f}±{r['delta_r2_std']:.4f}  "
            f"Spearman={r['spearman_mean']:.4f} "
            f"(Δρ={r['delta_spearman_vs_chemical']:+.4f}±{r['delta_spearman_std']:.4f})"
        )
    print("\n  missingness (best model):")
    for _, r in missing_df.iterrows():
        print(
            f"    {r['scenario']:10s} achieved={r['achieved_additional_rate'] * 100:5.2f}%  "
            f"R2={r['r2_mean']:.4f} (Δ{r['delta_r2']:+.4f})  "
            f"RMSE={r['rmse_mean']:.4f} (Δ{r['delta_rmse_pct']:+.2f}%)"
        )
    print(f"\n  total runtime: {time.time() - t_start:.1f}s")
    print(f"  outputs: {OUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
