# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本次仅整理运行入口、路径和可视化范围，核心计算保留原程序；更早的AI参与见使用详情说明。
"""Q1: 数据缺失条件下的感官评分。

把 1.csv 的 231 个化学特征映射到 4.csv 的 50 个感官维度：
- 连续预测：RepeatedKFold(5 folds x 3 repeats)，折内 fit 预处理器；
- 基线：均值、Ridge、PLS；主模型：ExtraTrees、RandomForest、LightGBM、OOF 堆叠；
- 0-6 分级：训练折稳健 min-max（预设主方案）、quantile、Gaussian latent（敏感性）；
- 缺失鲁棒性：自然缺失 + MCAR 5/10/20/30% + 分位数驱动 MAR。

运行：
    python src/问题一_感官评分.py
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
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
from 公共数据处理 import MODEL_CN, TARGET_CN, apply_grade_cutpoints, load_all, make_grade_labels, make_preprocessor, scoring_card_map, provenance
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'q1'
IMG_DIR = ROOT / 'img'
SEED = 42
N_SPLITS = 5
N_REPEATS = 3
N_REPEATS_MISSING = 3
GRADE_PROPORTIONS = (0.025, 0.075, 0.15, 0.25, 0.25, 0.15, 0.1)
BASE_MODELS = ['ridge', 'pls', 'extratrees', 'randomforest', 'lightgbm']
STACK_MODELS = list(BASE_MODELS)
ALL_MODELS = ['mean', 'style_mean'] + BASE_MODELS + ['blend_two', 'blend_avg', 'stack_ridge', 'selected_pipeline', 'extratrees_indicator', 'ridge_style']
CALIBRATIONS = ['isotonic', 'minmax', 'quantile_rank', 'quantile_map', 'linear', 'gaussian']
METRIC_COLS = ['r2', 'rmse', 'mae', 'spearman', 'explained_variance']

def _nanmean(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(np.asarray(x, dtype=float)))

def _nanstd(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        arr = np.asarray(x, dtype=float)
        if np.sum(np.isfinite(arr)) <= 1:
            return 0.0
        return float(np.nanstd(arr, ddof=1))

def macro_r2_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """仅 R² 的宏平均，供内层 CV 快速选参。"""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
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
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - ss_res / ss_tot
        ev = 1.0 - np.var(err, axis=0) / np.var(y_true, axis=0)
    r2[~np.isfinite(r2)] = np.nan
    ev[~np.isfinite(ev)] = np.nan
    n_targets = y_true.shape[1]
    sp = np.array([_safe_spearman(y_true[:, j], y_pred[:, j]) for j in range(n_targets)])
    return {'r2': r2, 'rmse': np.sqrt(np.mean(err ** 2, axis=0)), 'mae': np.mean(np.abs(err), axis=0), 'spearman': sp, 'explained_variance': ev}

def macro_from_per_target(m: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: _nanmean(v) for (k, v) in m.items()}

def record_fold_metrics(rows: list[dict], model: str, repeat: int, fold: int, y_true: np.ndarray, y_pred: np.ndarray, targets: list[str], n_train: int, n_test: int) -> None:
    if not np.isfinite(y_pred).all():
        raise RuntimeError(f'{model} repeat={repeat} fold={fold}: non-finite prediction')
    m = per_target_metrics(y_true, y_pred)
    for (j, target) in enumerate(targets):
        rows.append({'model': model, 'repeat': repeat, 'fold': fold, 'target': target, 'n_train': n_train, 'n_test': n_test, **{k: float(m[k][j]) for k in METRIC_COLS}})
    rows.append({'model': model, 'repeat': repeat, 'fold': fold, 'target': 'MACRO', 'n_train': n_train, 'n_test': n_test, **{k: _nanmean(m[k]) for k in METRIC_COLS}})

def fit_minmax_calibration(y_train: pd.DataFrame) -> dict:
    """稳健 min-max 校准参数：训练折内 y 的 1%/99% 分位。"""
    return {'q01': y_train.quantile(0.01).to_numpy(dtype=float), 'q99': y_train.quantile(0.99).to_numpy(dtype=float)}

def apply_minmax(z: np.ndarray, cal: dict) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    span = cal['q99'] - cal['q01']
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    s = 6.0 * (z - cal['q01']) / span
    return np.clip(np.rint(s), 0, 6).astype(int)

def fit_quantile_calibration(y_train: pd.DataFrame) -> dict:
    (_, cut) = make_grade_labels(y_train, proportions=GRADE_PROPORTIONS)
    return {'cut': cut}

def apply_quantile(z: np.ndarray, cal: dict, targets: list[str]) -> np.ndarray:
    z_df = pd.DataFrame(np.asarray(z, dtype=float), columns=targets)
    return apply_grade_cutpoints(z_df, cal['cut'])

def fit_gaussian_calibration(y_train: pd.DataFrame) -> dict:
    return {'mean': y_train.mean().to_numpy(dtype=float), 'std': y_train.std(ddof=0).to_numpy(dtype=float)}

def apply_gaussian(z: np.ndarray, cal: dict) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    std = np.where(cal['std'] < 1e-12, 1.0, cal['std'])
    latent = (z - cal['mean']) / std
    return np.clip(np.rint(3.0 + 1.5 * latent), 0, 6).astype(int)

def fit_calibration(cal_name: str, y_train: pd.DataFrame, pred_train: np.ndarray, targets: list[str]) -> dict:
    """在训练折的 OOF 预测上拟合 0-6 校准层。

    主方案 isotonic：pred -> y 的单调映射，再用训练折 y 的 q01/q99 做稳健 min-max。
    敏感性方案：raw min-max、quantile rank、quantile map、linear recalibration、Gaussian latent。
    """
    y_arr = y_train.to_numpy(dtype=float)
    pred_train = np.asarray(pred_train, dtype=float)
    if cal_name in ('isotonic', 'minmax', 'quantile_map', 'linear'):
        base = {'minmax': fit_minmax_calibration(y_train)}
    else:
        base = {}
    if cal_name == 'isotonic':
        base['models'] = []
        for j in range(len(targets)):
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(pred_train[:, j], y_arr[:, j])
            base['models'].append(iso)
        return base
    if cal_name == 'minmax':
        return base
    if cal_name == 'quantile_rank':
        return fit_quantile_calibration(y_train)
    if cal_name == 'quantile_map':
        maps = []
        for j in range(len(targets)):
            order = np.argsort(pred_train[:, j])
            xs = pred_train[order, j]
            ps = (np.arange(len(xs)) + 0.5) / len(xs)
            ys = np.sort(y_arr[:, j])
            qs = (np.arange(len(ys)) + 0.5) / len(ys)
            maps.append({'xs': xs, 'ps': ps, 'qs': qs, 'ys': ys})
        base['maps'] = maps
        return base
    if cal_name == 'linear':
        base['models'] = [LinearRegression().fit(pred_train[:, [j]], y_arr[:, j]) for j in range(len(targets))]
        return base
    if cal_name == 'gaussian':
        return fit_gaussian_calibration(y_train)
    raise ValueError(cal_name)

def apply_calibration(cal_name: str, z: np.ndarray, cal: dict, targets: list[str]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if cal_name == 'isotonic':
        y_cal = np.column_stack([iso.predict(z[:, j]) for (j, iso) in enumerate(cal['models'])])
        return apply_minmax(y_cal, cal['minmax'])
    if cal_name == 'minmax':
        return apply_minmax(z, cal['minmax'])
    if cal_name == 'quantile_rank':
        return apply_quantile(z, cal, targets)
    if cal_name == 'quantile_map':
        y_cal = np.zeros_like(z)
        for (j, m) in enumerate(cal['maps']):
            u = np.interp(z[:, j], m['xs'], m['ps'])
            y_cal[:, j] = np.interp(u, m['qs'], m['ys'])
        return apply_minmax(y_cal, cal['minmax'])
    if cal_name == 'linear':
        y_cal = np.column_stack([m.predict(z[:, [j]]) for (j, m) in enumerate(cal['models'])])
        return apply_minmax(y_cal, cal['minmax'])
    if cal_name == 'gaussian':
        return apply_gaussian(z, cal)
    raise ValueError(cal_name)

def grade_metrics_per_target(true_grades: np.ndarray, pred_grades: np.ndarray) -> dict[str, np.ndarray]:
    true_grades = np.asarray(true_grades, dtype=int)
    pred_grades = np.asarray(pred_grades, dtype=int)
    n_targets = true_grades.shape[1]
    out = {'qwk': np.full(n_targets, np.nan), 'grade_mae': np.full(n_targets, np.nan), 'exact_hit': np.full(n_targets, np.nan), 'within1_hit': np.full(n_targets, np.nan), 'true_mean': np.full(n_targets, np.nan), 'pred_mean': np.full(n_targets, np.nan)}
    for j in range(n_targets):
        t = true_grades[:, j]
        p = pred_grades[:, j]
        out['grade_mae'][j] = np.mean(np.abs(t - p))
        out['exact_hit'][j] = np.mean(t == p)
        out['within1_hit'][j] = np.mean(np.abs(t - p) <= 1)
        out['true_mean'][j] = np.mean(t)
        out['pred_mean'][j] = np.mean(p)
        if np.array_equal(t, p):
            out['qwk'][j] = 1.0
        elif len(np.unique(t)) == 1 and len(np.unique(p)) == 1:
            out['qwk'][j] = 0.0
        else:
            out['qwk'][j] = cohen_kappa_score(t, p, weights='quadratic', labels=list(range(7)))
    return out

def grade_distribution(grades: np.ndarray) -> np.ndarray:
    """返回 7 个等级的比例，形状 (7,)。"""
    vals = np.asarray(grades, dtype=int).ravel()
    counts = np.bincount(vals, minlength=7).astype(float)
    total = counts.sum()
    return counts / total if total else counts
import argparse
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def model_factory(name, value=None, seed=42, trees=200):
    if name == 'ridge':
        return Ridge(alpha=float(value))
    if name == 'pls':
        return PLSRegression(n_components=int(value), scale=False)
    if name in ('extratrees', 'randomforest'):
        cls = ExtraTreesRegressor if name == 'extratrees' else RandomForestRegressor
        return cls(n_estimators=trees, min_samples_leaf=int(value), n_jobs=6, random_state=seed)
    if name == 'lightgbm':
        return MultiOutputRegressor(LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=10, min_child_samples=15, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=1.0, n_jobs=1, random_state=seed, verbosity=-1), n_jobs=6)
    raise ValueError(name)

def inner_search(X, y, seed, trees=200):
    splits = list(KFold(3, shuffle=True, random_state=seed).split(X))
    transformed = []
    for (tr, va) in splits:
        pre = make_preprocessor()
        transformed.append((tr, va, pre.fit_transform(X.iloc[tr]), pre.transform(X.iloc[va])))
    grid = {'ridge': np.logspace(0, 6, 7), 'pls': [2, 4, 8, 12], 'extratrees': [1, 3, 6], 'randomforest': [2, 4], 'lightgbm': [None]}
    (hp, inner) = ({}, {})
    for (name, values) in grid.items():
        best_score = -np.inf
        for value in values:
            pred = np.zeros_like(y)
            for (tr, va, xt, xv) in transformed:
                model = model_factory(name, value, seed, trees)
                model.fit(xt, y[tr])
                pred[va] = model.predict(xv)
            score = macro_r2_fast(y, pred)
            if score > best_score:
                best_score = score
                hp[name] = float(value) if value is not None else None
                inner[name] = pred.copy()
    inner['blend_two'] = (inner['ridge'] + inner['extratrees']) / 2
    inner['blend_avg'] = np.mean([inner[n] for n in BASE_MODELS], axis=0)
    return (hp, inner)

def fit_outer(X, y, seed, hp, trees=500, indicator=False):
    pre = make_preprocessor(add_indicator=indicator)
    z = pre.fit_transform(X)
    models = {}
    for name in BASE_MODELS if not indicator else ['extratrees']:
        m = model_factory(name, hp[name], seed, trees)
        m.fit(z, y)
        models[name] = m
    return (pre, models)

def predict_models(fitted, X):
    (pre, models) = fitted
    z = pre.transform(X)
    pred = {name: m.predict(z) for (name, m) in models.items()}
    if len(models) == 5:
        pred['blend_two'] = (pred['ridge'] + pred['extratrees']) / 2
        pred['blend_avg'] = np.mean([pred[n] for n in BASE_MODELS], axis=0)
    return pred

def style_mean(y, train_styles, test_styles, kappa=5):
    out = np.zeros((len(test_styles), y.shape[1]))
    global_mean = y.mean(axis=0)
    for (i, st) in enumerate(test_styles):
        vals = y[train_styles == st]
        out[i] = (vals.sum(axis=0) + kappa * global_mean) / (len(vals) + kappa)
    return out

def paired_comparison(y, oof, n_boot=2000):
    rng = np.random.default_rng(SEED + 8000)
    rows = []
    primary = oof['selected_pipeline'].mean(axis=0)
    for name in ['mean', 'style_mean', 'extratrees', 'blend_two', 'blend_avg']:
        other = oof[name].mean(axis=0)
        point = macro_r2_fast(y, primary) - macro_r2_fast(y, other)
        diffs = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(y), len(y))
            diffs.append(macro_r2_fast(y[ix], primary[ix]) - macro_r2_fast(y[ix], other[ix]))
        rows.append(dict(reference=name, delta_r2=point, ci_low=np.quantile(diffs, 0.025), ci_high=np.quantile(diffs, 0.975), n_beers=len(y), n_boot=n_boot, scope='paired_beer_bootstrap_of_averaged_OOF_predictions_conditional_on_fitted_CV_models'))
    return pd.DataFrame(rows)

def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def main():
    parser = argparse.ArgumentParser(description='Nested Q1 evaluation; defaults are the paper run.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    D = load_all()
    (X, y) = (D['X'], D['y'])
    targets = list(y)
    Y = y.values
    styles = np.asarray(D['styles'])
    n = len(X)
    nrep = args.repeats
    splits = list(RepeatedKFold(n_splits=5, n_repeats=nrep, random_state=SEED).split(X))
    oof = {name: np.full((nrep, n, len(targets)), np.nan) for name in ALL_MODELS}
    grades = {cal: np.full((nrep, n, len(targets)), -1, dtype=int) for cal in CALIBRATIONS}
    records = []
    grade_rows = []
    missing_rows = []
    selection = []
    hp_rows = []
    split_rows = []
    t0 = time.time()
    import hashlib
    pr = provenance()
    relevant = {k: v for (k, v) in pr['sha256'].items() if k in ['src/问题一_感官评分.py', 'src/公共数据处理.py'] or k.startswith('附件/')}
    source_hash = hashlib.sha256(json.dumps({'sha256': relevant, 'packages': pr['packages']}, sort_keys=True).encode()).hexdigest()
    for (k, (tr, te)) in enumerate(splits):
        (rep, fold) = divmod(k, 5)
        seed = SEED + k
        for (scope, indices) in [('train', tr), ('test', te)]:
            split_rows.extend((dict(repeat=rep, fold=fold, beer=int(D['meta'].iloc[i].beer), scope=scope) for i in indices))
        checkpoint = OUT_DIR / f'折次_{k:02d}.pkl'
        if args.resume and checkpoint.exists():
            import pickle
            with checkpoint.open('rb') as f:
                cached = pickle.load(f)
            if cached['source_hash'] == source_hash and cached['trees'] == args.trees:
                for (name, pred) in cached['predictions'].items():
                    oof[name][rep, te] = pred
                for (cal, pred) in cached['grades'].items():
                    grades[cal][rep, te] = pred
                records += cached['records']
                grade_rows += cached['grade_rows']
                missing_rows += cached['missing_rows']
                selection.append(cached['selection'])
                hp_rows.append(cached['hp'])
                print(f'[Q1] fold {k + 1}/{len(splits)} resumed', flush=True)
                continue
        print(f'[Q1] fold {k + 1}/{len(splits)}: train={len(tr)}, test={len(te)}; inner search', flush=True)
        (Xtr, Xte) = (X.iloc[tr], X.iloc[te])
        Ytr = Y[tr]
        (hp, inner) = inner_search(Xtr, Ytr, seed, min(args.trees, 200))
        candidates = BASE_MODELS + ['blend_two', 'blend_avg']
        chosen = max(candidates, key=lambda name: macro_r2_fast(Ytr, inner[name]))
        fitted = fit_outer(Xtr, Ytr, seed, hp, args.trees)
        pred = predict_models(fitted, Xte)
        pred['selected_pipeline'] = pred[chosen]
        pred['mean'] = np.tile(Ytr.mean(axis=0), (len(te), 1))
        pred['style_mean'] = style_mean(Ytr, styles[tr], styles[te])
        ind = fit_outer(Xtr, Ytr, seed, hp, args.trees, True)
        pred['extratrees_indicator'] = predict_models(ind, Xte)['extratrees']
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        str_ = enc.fit_transform(styles[tr, None])
        ste = enc.transform(styles[te, None])
        pre = fitted[0]
        m = Ridge(alpha=hp['ridge']).fit(np.c_[pre.transform(Xtr), str_], Ytr)
        pred['ridge_style'] = m.predict(np.c_[pre.transform(Xte), ste])
        train_meta = np.stack([inner[name] for name in BASE_MODELS], axis=-1)
        test_meta = np.stack([pred[name] for name in BASE_MODELS], axis=-1)
        pred['stack_ridge'] = np.column_stack([Ridge(alpha=1.0).fit(train_meta[:, j], Ytr[:, j]).predict(test_meta[:, j]) for j in range(Y.shape[1])])
        fold_records = []
        for (name, p) in pred.items():
            oof[name][rep, te] = p
            record_fold_metrics(fold_records, name, rep, fold, Y[te], p, targets, len(tr), len(te))
        fold_grades = []
        fold_grade_values = {}
        anchor = fit_minmax_calibration(y.iloc[tr])
        reference = apply_minmax(Y[te], anchor)
        for cal in CALIBRATIONS:
            fitted_cal = fit_calibration(cal, y.iloc[tr], inner[chosen], targets)
            pg = apply_calibration(cal, pred['selected_pipeline'], fitted_cal, targets)
            grades[cal][rep, te] = pg
            fold_grade_values[cal] = pg
            metrics = grade_metrics_per_target(reference, pg)
            for (j, target) in enumerate(targets + ['MACRO']):
                fold_grades.append(dict(scope='outer_fold', repeat=rep, fold=fold, calibration=cal, target=target, **{key: _nanmean(v) if target == 'MACRO' else float(v[j]) for (key, v) in metrics.items()}))
        fold_missing = []
        for (kind, rate) in [('natural', 0.0), ('MCAR', 0.05), ('MCAR', 0.1), ('MCAR', 0.2), ('MCAR', 0.3), ('MAR', 0.25)]:
            xm = Xte.copy()
            rng = np.random.default_rng(seed + int(rate * 10000) + 3000)
            if kind == 'MCAR':
                mask = rng.random(xm.shape) < rate
            elif kind == 'MAR':
                driver = 'ethanol..v.v.'
                cuts = np.quantile(Xtr[driver], [0.25, 0.5, 0.75])
                bins = np.searchsorted(cuts, xm[driver].values)
                mask = rng.random(xm.shape) < np.array([0.05, 0.15, 0.3, 0.5])[bins, None]
                mask[:, X.columns.get_loc(driver)] = False
            else:
                mask = np.zeros(xm.shape, bool)
            xm = xm.mask(mask)
            p = predict_models(fitted, xm)[chosen]
            mt = per_target_metrics(Y[te], p)
            fold_missing.append(dict(repeat=rep, fold=fold, scenario=f'MCAR {rate:.0%}' if kind == 'MCAR' else kind, nominal_rate=rate, total_missing_rate=float(xm.isna().values.mean()), r2=_nanmean(mt['r2']), rmse=_nanmean(mt['rmse']), evaluation='fixed_model_test_only_mask'))
        sel = dict(repeat=rep, fold=fold, model=chosen, inner_r2=macro_r2_fast(Ytr, inner[chosen]), selection_scope='outer_training_only', n_train=len(tr), n_test=len(te))
        hpr = dict(repeat=rep, fold=fold, parameters=hp)
        records += fold_records
        grade_rows += fold_grades
        missing_rows += fold_missing
        selection.append(sel)
        hp_rows.append(hpr)
        import pickle
        with checkpoint.open('wb') as f:
            pickle.dump(dict(source_hash=source_hash, trees=args.trees, predictions=pred, grades=fold_grade_values, records=fold_records, grade_rows=fold_grades, missing_rows=fold_missing, selection=sel, hp=hpr), f)
        print(f"[Q1] fold {k + 1} done; chose {chosen}; outer R²={macro_r2_fast(Y[te], pred['selected_pipeline']):.4f}; elapsed={time.time() - t0:.0f}s", flush=True)
    cv = pd.DataFrame(records)
    grade = pd.DataFrame(grade_rows)
    mf = pd.DataFrame(missing_rows)
    summary = {name: {metric: {'mean': float(sub[metric].mean()), 'std': float(sub[metric].std())} for metric in METRIC_COLS} for name in ALL_MODELS for sub in [cv[(cv.model == name) & (cv.target == 'MACRO')]]}
    missing = mf.groupby('scenario', sort=False).agg(total_missing_rate=('total_missing_rate', 'mean'), r2_mean=('r2', 'mean'), r2_std=('r2', 'std'), rmse_mean=('rmse', 'mean'), n_folds=('fold', 'size')).reset_index()
    natural = float(missing.loc[missing.scenario == 'natural', 'r2_mean'].iloc[0])
    missing['delta_r2_pct'] = 100 * (missing.r2_mean - natural) / abs(natural)
    cv.to_csv(OUT_DIR / '交叉验证结果.csv', index=False)
    grade.to_csv(OUT_DIR / '等级评价指标.csv', index=False)
    missing.to_csv(OUT_DIR / '缺失情景评价.csv', index=False)
    mf.to_csv(OUT_DIR / '缺失情景各折结果.csv', index=False)
    pd.DataFrame(selection).to_csv(OUT_DIR / '模型选择记录.csv', index=False)
    pd.DataFrame(split_rows).to_csv(OUT_DIR / '样本划分.csv', index=False)
    compare = paired_comparison(Y, oof)
    compare.to_csv(OUT_DIR / '模型配对比较.csv', index=False)
    output_columns = {'beer': D['meta'].beer, 'tasting_category_fine': styles}
    for (j, t) in enumerate(targets):
        output_columns['pred_' + t] = oof['selected_pipeline'].mean(axis=0)[:, j]
        output_columns['grade_' + t] = np.median(grades['minmax'], axis=0)[:, j].astype(int)
    final = pd.DataFrame(output_columns)
    final.to_csv(OUT_DIR / '折外预测.csv', index=False)
    mapping = scoring_card_map()
    mapping.to_csv(OUT_DIR / '评分卡栏位映射.csv', index=False)
    card = []
    for r in final.itertuples(index=False):
        for entry in mapping.itertuples(index=False):
            card.append(dict(beer=r.beer, side=entry.side, cell=entry.cell, item=entry.item, target=entry.target, relative_grade=getattr(r, 'grade_' + entry.target) if pd.notna(entry.target) else np.nan, status=entry.status))
    pd.DataFrame(card).to_csv(OUT_DIR / '完整评分卡.csv', index=False)
    ablation_rows = []
    macros = cv.query('target=="MACRO"')
    for (name, base, feature) in [('ridge', 'ridge', 'chemical'), ('ridge_style', 'ridge', 'style_onehot'), ('extratrees', 'extratrees', 'chemical'), ('extratrees_indicator', 'extratrees', 'missing_indicator')]:
        a = macros[macros.model == name].sort_values(['repeat', 'fold'])
        b = macros[macros.model == base].sort_values(['repeat', 'fold'])
        ablation_rows.append(dict(model=name, feature_set=feature, n_folds=len(a), **{m + '_mean': float(a[m].mean()) for m in ['r2', 'rmse', 'mae', 'spearman']}, delta_r2_vs_chemical=float(np.mean(a.r2.values - b.r2.values))))
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / '风格特征消融.csv', index=False)
    payload = {'meta': {'schema_version': 2, 'n_splits': 5, 'n_repeats': nrep, 'trees': args.trees, 'n_samples': n, 'n_targets': len(targets), 'selection': 'inner_training_only; no outer scores select a learner', 'grade_primary': 'predeclared_raw_training_y_q01_q99_minmax', 'grade_reference': 'same outer-training y q01/q99 for all methods; never full-data quantiles', 'calibration_note': 'inner HP-selected OOF predictions within outer training; selection and calibration evaluated on untouched outer test', 'missingness': 'test_only_frozen_model; MAR quartiles fit on outer training', 'warning': 'Relative grades are not recovered original 0-6 labels; outer candidate comparisons exploratory', 'runtime_note': 'runtime_sec measures this process; resumed fold fitting time is not included'}, 'models': {name: {'macro_mean_std_over_folds': summary[name], 'final_oof_macro': macro_from_per_target(per_target_metrics(Y, oof[name].mean(axis=0)))} for name in ALL_MODELS}, 'best_model': {'name': 'selected_pipeline', 'criterion': 'learner chosen inside each outer training fold'}, 'grade': {'primary_calibration': 'minmax', 'macro_mean_std_over_folds': {cal: {m: {'mean': float(a[m].mean()), 'std': float(a[m].std())} for m in ['qwk', 'grade_mae', 'exact_hit', 'within1_hit']} for (cal, a) in grade.query('target=="MACRO"').groupby('calibration')}}, 'hyperparameters': hp_rows, 'selection': selection, 'paired_comparison': compare.to_dict('records'), 'missingness': missing.to_dict('records'), 'runtime_sec': time.time() - t0, 'provenance': provenance(), 'limitations': ['Unlabeled aroma/flavour acetaldehyde retained explicitly without invented score', 'No absolute grade accuracy is identifiable', 'Paired bootstrap conditions on fitted CV models; not full retraining uncertainty']}
    (OUT_DIR / '模型结果汇总.json').write_text(json.dumps(_sanitize_json(payload), ensure_ascii=False, indent=2, allow_nan=False))
    print('[Q1] complete', json.dumps(payload['models']['selected_pipeline'], ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
