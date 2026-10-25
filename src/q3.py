#!/usr/bin/env python3
"""Q3: style-conditional generation and Pils/Lager -> Tripel transfer.

Data contract
-------------
* Input is ``data_utils.load_all()["X_cal"]`` (250 x 70 linear calibrated
  concentrations), not the mixed-scale columns of ``1.csv``.
* All generative modelling is done after ``log1p`` and per-column
  standardisation; the inverse transform is stored and used to return to
  concentration space.
* Defect thresholds follow ``项目进度/化学合理性文献.md`` and the attachment:
  DMS 17.25 ug/L (strict), diacetyl 0.09 mg/L (strict), 4VG 0.10 mg/L
  (non-wheat safety value; attachment mean 0.206), trans-2-nonenal 0.10 ug/L
  (external, attachment NA).  The T2N column in X_cal is numerically in ug/L
  (Pils median ~0.002, Tripel ~0.011), consistent with the external threshold.
* Stale hops is not a compound; the transfer reports a proxy composite built
  from T2N, DMS, ethyl 2-methylbutyrate, ethyl isovalerate and 4VG.

Outputs: data/q3/{generated_profiles,transfer_profiles,metrics,portraits}.csv,
data/q3/summary.json, img/q3_*.png/pdf.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import QuantileTransformer

from data_utils import load_all

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "q3"
IMG_DIR = ROOT / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_GEN = 300
N0 = 20.0                      # shrinkage strength towards the global estimate
ZERO_THRESHOLD = 0.10          # columns with >=10% zeros get hurdle restoration
TARGET_STYLES = ["Tripel", "Kriek", "Lambic"]
SOURCE_STYLE = "Pils/Lager"
TARGET_STYLE = "Tripel"
MODEL_ORDER = ["mean_noise", "global_cov", "copula", "ot", "ot_constrained"]
MODEL_ORDER_BASE = ["mean_noise", "global_cov", "copula", "ot"]
MODEL_LABELS = {
    "mean_noise": "风格均值+噪声",
    "global_cov": "全局协方差+风格均值",
    "copula": "条件 copula",
    "ot": "条件 OT",
    "ot_constrained": "OT+风格上限",
}

# Colour-blind-safe first slots (validated with the dataviz palette validator).
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_VIOLET = "#4a3aa7"
C_RED = "#e34948"
C_GRAY = "#898781"
C_INK = "#0b0b0b"
C_SURFACE = "#fcfcfb"
C_MAGENTA = "#e87ba4"
MODEL_COLORS = {
    "mean_noise": C_YELLOW,
    "global_cov": C_AQUA,
    "copula": C_BLUE,
    "ot": C_ORANGE,
    "ot_constrained": C_MAGENTA,
}
STYLE_COLORS = {
    "Tripel": C_BLUE,
    "Kriek": C_ORANGE,
    "Lambic": C_AQUA,
    "Pils/Lager": C_YELLOW,
}
VARIANT_COLORS = {
    "source": C_GRAY,
    "raw": C_YELLOW,
    "strict": C_BLUE,
    "style": C_ORANGE,
}

DEFECTS = {
    "DMS": {"col": "DMS", "unit": "ug/L", "strict": 17.25},
    "Diacetyl": {"col": "X2.3.butanedione", "unit": "mg/L", "strict": 0.09},
    "4VG": {"col": "X4VG", "unit": "mg/L", "strict": 0.10},
    "T2N": {"col": "X.E..2.nonenal", "unit": "ug/L", "strict": 0.10},
}
def style_caps_for(P, style):
    """Defect caps max(strict threshold, that style's p75); 4VG uses 0.10."""
    p75 = P["style_p75"][style]
    return {
        "DMS": max(DEFECTS["DMS"]["strict"], p75["DMS"]),
        "X2.3.butanedione": max(DEFECTS["Diacetyl"]["strict"], p75["X2.3.butanedione"]),
        "X4VG": 0.10,
        "X.E..2.nonenal": max(DEFECTS["T2N"]["strict"], p75["X.E..2.nonenal"]),
    }


PROXY_COLS = ["X.E..2.nonenal", "DMS", "ethyl_2.methyl_butyrate",
              "ethyl_isovalerate", "X4VG"]
KEY_COMPOUNDS = [
    "ethanol..v.v.", "glycerol", "ethyl_acetate", "isopentyl_acetate",
    "ethyl_hexanoate", "ethyl_octanoate", "isoamyl_alcohol",
    "linalool", "geraniol", "citronellol",
]
KEY_LABELS = {
    "ethanol..v.v.": "Ethanol",
    "glycerol": "Glycerol",
    "ethyl_acetate": "Ethyl acetate",
    "isopentyl_acetate": "Isoamyl acetate",
    "ethyl_hexanoate": "Ethyl hexanoate",
    "ethyl_octanoate": "Ethyl octanoate",
    "isoamyl_alcohol": "Isoamyl alcohol",
    "linalool": "Linalool",
    "geraniol": "Geraniol",
    "citronellol": "Citronellol",
}
RADAR_DIMS = [
    "A_malt_all", "A_hops_all", "A_esters_all",
    "F_malt_all", "F_hops_all", "F_esters_all",
    "acidity", "bitternes", "sweetness", "alcohol", "body", "overall",
]
RADAR_LABELS = {
    "A_malt_all": "麦芽香", "A_hops_all": "酒花香",
    "A_esters_all": "酯香", "F_malt_all": "麦芽味",
    "F_hops_all": "酒花味", "F_esters_all": "酯味",
    "acidity": "酸度", "bitternes": "苦味",
    "sweetness": "甜味", "alcohol": "酒精",
    "body": "酒体", "overall": "总体",
}

VARIANT_LABELS = {
    "source": "源样本", "raw": "原始 OT",
    "strict": "严格投影", "style": "风格投影",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ensure_pd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    vmin = float(vals.min()) if len(vals) else 0.0
    if vmin < eps:
        cov = cov + np.eye(cov.shape[0]) * (eps - vmin)
    return cov


def sqrt_psd(cov: np.ndarray, inverse: bool = False) -> np.ndarray:
    cov = (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    floor = max(1e-10, 1e-8 * float(vals.max()) if vals.size else 1e-10)
    vals = np.clip(vals, floor, None)
    vals = 1.0 / np.sqrt(vals) if inverse else np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return ensure_pd(corr)


def inverse_from_z(z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    # Guard against overflow from extreme Gaussian draws in the baselines.
    return np.expm1(np.clip(z, -50.0, 50.0) * sd + mu)


def median_heuristic(x: np.ndarray) -> float:
    d = pdist(x, metric="sqeuclidean")
    med = float(np.median(d)) if d.size else 1.0
    return 1.0 / (2.0 * max(med, 1e-8))


def mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean")).mean()
    return float(kxx + kyy - 2.0 * kxy)


def to_py(obj):
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(IMG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# preprocessing and style statistics
# ---------------------------------------------------------------------------
def load_and_prepare():
    D = load_all()
    X_raw = D["X_cal"].copy()
    feats = list(D["cal_features"])
    styles = np.asarray(D["styles"], dtype=object)
    beer_ids = D["meta"]["beer"].astype(int).values

    n_missing = int(X_raw.isna().sum().sum())
    med = X_raw.median(axis=0)
    X_imp = X_raw.fillna(med)
    neg_mask = X_imp < 0
    n_negative = int(neg_mask.sum().sum())
    neg_by_col = {c: int(neg_mask[c].sum()) for c in feats if neg_mask[c].any()}
    X_nn = np.clip(X_imp.values, 0.0, None)

    L = np.log1p(X_nn)
    mu = L.mean(axis=0)
    sd = L.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    Z = (L - mu) / sd

    qt = QuantileTransformer(
        n_quantiles=min(250, len(L)), output_distribution="normal",
        random_state=SEED,
    )
    U = qt.fit_transform(L)
    u_mean = U.mean(axis=0)
    u_sd = U.std(axis=0, ddof=0)
    u_sd = np.where(u_sd < 1e-10, 1.0, u_sd)
    U_std = (U - u_mean) / u_sd

    cov_global_z = ensure_pd(LedoitWolf().fit(Z).covariance_)
    cov_global_u = ensure_pd(LedoitWolf().fit(U_std).covariance_)

    zero_rates = {
        st: (X_nn[styles == st] == 0).mean(axis=0)
        for st in sorted(set(styles))
    }
    style_p75 = {}
    for st in sorted(set(styles)):
        xs = X_nn[styles == st]
        style_p75[st] = {
            info["col"]: float(np.quantile(xs[:, feats.index(info["col"])], 0.75))
            for info in DEFECTS.values()
        }

    return {
        "D": D, "X_raw": X_raw, "X_imp": X_imp, "X_nn": X_nn,
        "L": L, "Z": Z, "U_std": U_std, "qt": qt,
        "u_mean": u_mean, "u_sd": u_sd, "mu": mu, "sd": sd,
        "feats": feats, "styles": styles, "beer_ids": beer_ids,
        "cov_global_z": cov_global_z, "cov_global_u": cov_global_u,
        "zero_rates": zero_rates, "style_p75": style_p75,
        "n_missing": n_missing, "n_negative": n_negative,
        "neg_by_col": neg_by_col,
        "global_min": X_nn.min(axis=0), "global_max": X_nn.max(axis=0),
        "targets": list(D["targets"]),
    }


def z_style_params(Z, styles, style, cov_global, n0=N0):
    zs = Z[styles == style]
    n = len(zs)
    mu = zs.mean(axis=0)
    cov_lw = LedoitWolf().fit(zs).covariance_
    alpha = n0 / (n0 + n)
    cov_blend = ensure_pd((1.0 - alpha) * cov_lw + alpha * cov_global)
    var_s = zs.var(axis=0, ddof=0)
    w = n / (n + n0)
    sigma = np.sqrt(np.maximum(w * var_s + (1.0 - w) * 1.0, 1e-8))
    return {
        "n": n, "mu": mu, "cov_lw": cov_lw, "cov_blend": cov_blend,
        "alpha": alpha, "w": w, "sigma": sigma,
    }


def copula_style_params(L, styles, style, global_corr, n0=N0):
    """Per-style Gaussian copula.

    Marginals: a per-style QuantileTransformer to normal scores (n_quantiles
    is capped by the style sample size).  Dependence: Ledoit-Wolf covariance
    of the normal scores, with its correlation matrix shrunk towards the
    global correlation matrix (weight n0/(n0+n)); the per-feature scale is
    shrunk towards the global unit variance.  This is the small-sample
    strategy requested for Kriek/Lambic/Tripel.
    """
    ls = L[styles == style]
    n = len(ls)
    qt = QuantileTransformer(
        n_quantiles=min(max(n, 5), 100), output_distribution="normal",
        random_state=SEED,
    )
    us = qt.fit_transform(ls)
    u_mean = us.mean(axis=0)
    u_sd = us.std(axis=0, ddof=0)
    u_sd = np.where(u_sd < 1e-10, 1.0, u_sd)
    ustd = (us - u_mean) / u_sd
    mu = ustd.mean(axis=0)
    cov_lw = LedoitWolf().fit(ustd).covariance_
    corr_s = cov_to_corr(cov_lw)
    alpha = n0 / (n0 + n)
    corr = ensure_pd((1.0 - alpha) * corr_s + alpha * global_corr)
    var_s = ustd.var(axis=0, ddof=0)
    w = n / (n + n0)
    sigma = np.sqrt(np.maximum(w * var_s + (1.0 - w) * 1.0, 1e-8))
    sigma_mat = np.outer(sigma, sigma) * corr
    return {
        "n": n, "mu": mu, "cov_lw": cov_lw, "corr": corr,
        "sigma": sigma, "Sigma": ensure_pd(sigma_mat),
        "alpha": alpha, "w": w, "qt": qt,
        "u_mean": u_mean, "u_sd": u_sd,
    }


def rank_calibrate_to_style(X_gen, X_target):
    """Map each generated column to the target style's empirical quantiles.

    The affine OT map matches means/covariances but not the marginal shapes
    (especially zero-inflated trace compounds).  This per-feature rank map
    restores the target marginals while preserving the within-sample ranking
    produced by the affine transport.
    """
    out = np.empty_like(X_gen, dtype=float)
    n = X_gen.shape[0]
    for j in range(X_gen.shape[1]):
        order = np.argsort(X_gen[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n)
        q = (ranks + 0.5) / n
        out[:, j] = np.quantile(X_target[:, j], q, method="linear")
    return out


def postprocess_concentration(X_raw, style, zero_rates, global_max, rng):
    n, d = X_raw.shape
    neg_frac = float((X_raw < 0).mean())
    X = np.clip(X_raw, 0.0, None)
    p0 = zero_rates[style]
    sel = np.where(p0 >= ZERO_THRESHOLD)[0]
    n_zero_forced = 0
    if len(sel):
        idx = np.arange(n)[:, None]
        mask = rng.random((n, len(sel))) < p0[sel]
        X[idx, sel] = np.where(mask, 0.0, X[idx, sel])
        n_zero_forced = int(mask.sum())
    n_upper = int((X > global_max).sum())
    X = np.minimum(X, global_max)
    return X, {
        "neg_frac_preclip": neg_frac,
        "n_zero_forced": n_zero_forced,
        "n_upper_clipped": n_upper,
    }


# ---------------------------------------------------------------------------
# generative models
# ---------------------------------------------------------------------------
def generate_profiles(P, n_gen=N_GEN):
    Z = P["Z"]
    U_std = P["U_std"]
    styles = P["styles"]
    feats = P["feats"]
    mu, sd = P["mu"], P["sd"]
    qt = P["qt"]
    u_mean, u_sd = P["u_mean"], P["u_sd"]
    cov_global_z = P["cov_global_z"]
    zero_rates = P["zero_rates"]
    global_max = P["global_max"]

    params_z = {
        st: z_style_params(Z, styles, st, cov_global_z)
        for st in sorted(set(styles))
    }
    global_corr = cov_to_corr(P["cov_global_u"])
    params_u = {
        st: copula_style_params(P["L"], styles, st, global_corr)
        for st in sorted(set(styles))
    }
    Lg = sqrt_psd(cov_global_z)

    generated = {}
    generated_z = {}
    gen_stats = {}
    for si, style in enumerate(TARGET_STYLES):
        pool_mask = styles != style
        pool = Z[pool_mask]
        mu_pool = pool.mean(axis=0)
        cov_pool = ensure_pd(LedoitWolf().fit(pool).covariance_)
        A_ot = sqrt_psd(params_z[style]["cov_blend"]) @ sqrt_psd(cov_pool, inverse=True)

        for mi, model in enumerate(MODEL_ORDER_BASE):
            rng = np.random.default_rng(SEED + 1000 * si + 100 * mi)
            if model == "mean_noise":
                z_gen = params_z[style]["mu"] + rng.standard_normal((n_gen, Z.shape[1])) * params_z[style]["sigma"]
                X_raw = inverse_from_z(z_gen, mu, sd)
            elif model == "global_cov":
                z_gen = params_z[style]["mu"] + rng.standard_normal((n_gen, Z.shape[1])) @ Lg.T
                X_raw = inverse_from_z(z_gen, mu, sd)
            elif model == "copula":
                cop = params_u[style]
                Lc = sqrt_psd(cop["Sigma"])
                u_gen = cop["mu"] + rng.standard_normal((n_gen, U_std.shape[1])) @ Lc.T
                u = u_gen * cop["u_sd"] + cop["u_mean"]
                X_raw = np.expm1(cop["qt"].inverse_transform(u))
            elif model == "ot":
                idx = rng.integers(0, len(pool), n_gen)
                z_gen = params_z[style]["mu"] + (pool[idx] - mu_pool) @ A_ot.T
                X_raw = inverse_from_z(z_gen, mu, sd)
                X_pre, _ = postprocess_concentration(
                    X_raw, style, zero_rates, global_max, rng
                )
                X_raw = rank_calibrate_to_style(
                    X_pre, P["X_nn"][styles == style]
                )
            else:
                raise ValueError(model)

            X_gen, stats = postprocess_concentration(
                X_raw, style, zero_rates, global_max, rng
            )
            z_metric = (np.log1p(X_gen) - mu) / sd
            stats["n_style_capped"] = 0
            stats["rank_calibrated"] = (model == "ot")
            generated[(model, style)] = X_gen
            generated_z[(model, style)] = z_metric
            gen_stats[(model, style)] = stats

        # Final generator: OT followed by the style cap
        # max(strict threshold, style p75) for the four named defects.
        style_caps = style_caps_for(P, style)
        xc = generated[("ot", style)].copy()
        n_cap = 0
        for col, cap in style_caps.items():
            j = feats.index(col)
            over = xc[:, j] > cap
            n_cap += int(over.sum())
            xc[over, j] = cap
        generated[("ot_constrained", style)] = xc
        generated_z[("ot_constrained", style)] = (np.log1p(xc) - mu) / sd
        gen_stats[("ot_constrained", style)] = {
            "neg_frac_preclip": 0.0, "n_zero_forced": 0,
            "n_upper_clipped": 0, "n_style_capped": n_cap,
            "rank_calibrated": True, "style_caps": style_caps,
        }

    return generated, generated_z, gen_stats, params_z, params_u


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def diversity_metrics(real_z, gen_z):
    real_pair = pdist(real_z, metric="euclidean")
    gen_pair = pdist(gen_z, metric="euclidean")
    d_gr = cdist(gen_z, real_z, metric="euclidean")
    gen_nn = d_gr.min(axis=1)
    d_rr = cdist(real_z, real_z, metric="euclidean")
    np.fill_diagonal(d_rr, np.inf)
    real_nn = d_rr.min(axis=1)
    thr_cov = float(np.median(real_nn))
    thr_prec = float(np.percentile(real_nn, 95))
    return {
        "gen_pair_median": float(np.median(gen_pair)),
        "real_pair_median": float(np.median(real_pair)),
        "gen_nn_real_median": float(np.median(gen_nn)),
        "real_loo_nn_median": float(np.median(real_nn)),
        "coverage": float((d_gr.min(axis=0) <= thr_cov).mean()),
        "precision": float((gen_nn <= thr_prec).mean()),
        "coverage_threshold": thr_cov,
        "precision_threshold": thr_prec,
    }


def classifier_metrics(model, z_gen, target):
    proba = model.predict_proba(z_gen)
    classes = model.classes_
    pred = classes[proba.argmax(axis=1)]
    top3 = classes[np.argsort(proba, axis=1)[:, -3:]]
    hit = float((pred == target).mean())
    top3_hit = float(np.any(top3 == target, axis=1).mean())
    maxp = float(proba.max(axis=1).mean())
    ti = np.where(classes == target)[0]
    target_p = float(proba[:, ti[0]].mean()) if len(ti) else 0.0
    return hit, top3_hit, maxp, target_p


def defect_exceedance(X, style, feats, style_p75):
    out = {}
    for name, info in DEFECTS.items():
        j = feats.index(info["col"])
        vals = X[:, j]
        strict = info["strict"]
        p75 = style_p75[style][info["col"]]
        out[f"{name}_mean"] = float(np.mean(vals))
        out[f"{name}_median"] = float(np.median(vals))
        out[f"{name}_p75"] = float(np.quantile(vals, 0.75))
        out[f"{name}_max"] = float(np.max(vals))
        out[f"{name}_above_strict"] = float((vals > strict).mean())
        out[f"{name}_above_style_p75"] = float((vals > p75).mean())
        out[f"{name}_max_oav"] = float(np.max(vals / strict)) if strict > 0 else np.nan
    oav = np.column_stack([
        X[:, feats.index(info["col"])] / info["strict"]
        for info in DEFECTS.values()
    ])
    out["max_oav_median"] = float(np.median(oav.max(axis=1)))
    return out


def stale_proxy(X, feats, mu, sd):
    idx = [feats.index(c) for c in PROXY_COLS]
    zp = (np.log1p(np.clip(X[:, idx], 0.0, None)) - mu[idx]) / sd[idx]
    return zp.mean(axis=1)


def sensory_fidelity(pred_med, real_med):
    r = float(pearsonr(pred_med, real_med)[0]) if np.std(pred_med) > 1e-12 and np.std(real_med) > 1e-12 else 0.0
    rho = float(spearmanr(pred_med, real_med)[0]) if np.std(pred_med) > 1e-12 and np.std(real_med) > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean((pred_med - real_med) ** 2)))
    return {"pearson": r, "spearman": rho, "rmse": rmse}


# ---------------------------------------------------------------------------
# sensory surrogate: PLS with inner CV, Ridge comparison
# ---------------------------------------------------------------------------
def select_pls_k(z_tr, y_tr, ks, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_k, best_score = ks[0], -np.inf
    for k in ks:
        preds = np.zeros_like(y_tr)
        for tr, te in kf.split(z_tr):
            m = PLSRegression(n_components=k, scale=False).fit(z_tr[tr], y_tr[tr])
            preds[te] = m.predict(z_tr[te])
        score = float(np.mean(r2_score(y_tr, preds, multioutput="raw_values")))
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score


def select_ridge_alpha(z_tr, y_tr, alphas, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_a, best_score = alphas[0], -np.inf
    for a in alphas:
        preds = np.zeros_like(y_tr)
        for tr, te in kf.split(z_tr):
            m = Ridge(alpha=a).fit(z_tr[tr], y_tr[tr])
            preds[te] = m.predict(z_tr[te])
        score = float(np.mean(r2_score(y_tr, preds, multioutput="raw_values")))
        if score > best_score:
            best_score, best_a = score, a
    return best_a, best_score


def fit_sensory_surrogate(Z, Y, seed=SEED):
    ks = list(range(1, 16))
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(Y)
    chosen_ks = []
    for tr, te in outer.split(Z):
        k, _ = select_pls_k(Z[tr], Y[tr], ks, seed)
        chosen_ks.append(int(k))
        m = PLSRegression(n_components=k, scale=False).fit(Z[tr], Y[tr])
        oof[te] = m.predict(Z[te])
    r2 = r2_score(Y, oof, multioutput="raw_values")
    k_final, inner_score = select_pls_k(Z, Y, ks, seed)
    final = PLSRegression(n_components=k_final, scale=False).fit(Z, Y)
    ridge_alpha, ridge_score = select_ridge_alpha(
        Z, Y, [0.1, 1.0, 10.0, 100.0, 1000.0], seed
    )
    return {
        "oof": oof,
        "final": final,
        "k_final": int(k_final),
        "chosen_ks": chosen_ks,
        "inner_score": float(inner_score),
        "cv_r2": r2,
        "cv_r2_mean": float(np.mean(r2)),
        "cv_r2_median": float(np.median(r2)),
        "cv_r2_min": float(np.min(r2)),
        "ridge_alpha": float(ridge_alpha),
        "ridge_cv_r2_mean": float(ridge_score),
    }


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_pca(P, generated, gen_sensory_med):
    Z = P["Z"]
    styles = P["styles"]
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    pz = pca.transform(Z)
    var = pca.explained_variance_ratio_ * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for st in sorted(set(styles)):
        m = styles == st
        axes[0].scatter(pz[m, 0], pz[m, 1], s=7, color="#d9d8d2", alpha=0.65)
    for st in TARGET_STYLES + [SOURCE_STYLE]:
        m = styles == st
        axes[0].scatter(pz[m, 0], pz[m, 1], s=22, color=STYLE_COLORS[st],
                        edgecolor="white", linewidth=0.4, label=st, zorder=3)
    axes[0].set_title("Real profiles (all 22 styles in grey)")
    axes[0].legend(frameon=False, loc="best")

    for ax, model, title in [
        (axes[1], "copula", "Conditional copula"),
        (axes[2], "ot", "Conditional OT + rank calibration"),
    ]:
        for st in sorted(set(styles)):
            m = styles == st
            ax.scatter(pz[m, 0], pz[m, 1], s=5, color="#e1e0d9", alpha=0.7)
        for st in TARGET_STYLES:
            zg = generated[(model, st)]
            zg = (np.log1p(zg) - P["mu"]) / P["sd"]
            pg = pca.transform(zg)
            ax.scatter(pg[:, 0], pg[:, 1], s=7, color=STYLE_COLORS[st],
                       alpha=0.35, label=st, zorder=2)
        ax.set_title(f"{title} (300/style)")
        ax.legend(frameon=False, markerscale=2.0, loc="best")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    savefig(fig, "q3_pca")


def plot_distributions(P, generated):
    feats = P["feats"]
    X_nn = P["X_nn"]
    styles = P["styles"]
    comps = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal",
             "ethyl_acetate", "isopentyl_acetate", "linalool", "geraniol"]
    units = {
        "DMS": "DMS (ug/L)", "X2.3.butanedione": "Diacetyl (mg/L)",
        "X4VG": "4VG (mg/L)", "X.E..2.nonenal": "T2N (ug/L)",
        "ethyl_acetate": "Ethyl acetate (mg/L)",
        "isopentyl_acetate": "Isoamyl acetate (mg/L)",
        "linalool": "Linalool (mg/L)", "geraniol": "Geraniol (mg/L)",
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
    axes = axes.ravel()
    for ax, comp in zip(axes, comps):
        j = feats.index(comp)
        data, colors, labels, hatches = [], [], [], []
        for st in TARGET_STYLES:
            for model in ["real", "copula", "ot", "ot_constrained"]:
                if model == "real":
                    vals = X_nn[styles == st, j]
                else:
                    vals = generated[(model, st)][:, j]
                data.append(np.log10(1.0 + vals))
                colors.append({
                    "real": C_INK, "copula": C_BLUE,
                    "ot": C_ORANGE, "ot_constrained": C_MAGENTA,
                }[model])
                labels.append(f"{st}\n{model.replace('_', '+')}")
                hatches.append("//" if model == "ot_constrained" else "")
        bp = ax.boxplot(data, patch_artist=True, widths=0.72, showfliers=False)
        for patch, c, h in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
            patch.set_edgecolor(C_INK)
            patch.set_linewidth(0.5)
            patch.set_hatch(h)
        for med in bp["medians"]:
            med.set_color(C_INK)
            med.set_linewidth(0.8)
        ax.set_xticks(np.arange(1, len(data) + 1))
        ax.set_xticklabels(labels, fontsize=5.6, rotation=0)
        ax.set_title(units[comp])
        ax.set_ylabel("log10(1 + concentration)")
        ax.set_ylim(bottom=0.0)
        defect_name = {
            "DMS": "DMS", "X2.3.butanedione": "Diacetyl",
            "X4VG": "4VG", "X.E..2.nonenal": "T2N",
        }.get(comp)
        if defect_name is not None:
            ax.axhline(np.log10(1.0 + DEFECTS[defect_name]["strict"]),
                       color=C_RED, linestyle="--", linewidth=0.8)
    handles = [
        Line2D([0], [0], color=C_INK, lw=6, label="Real"),
        Line2D([0], [0], color=C_BLUE, lw=6, label="Copula"),
        Line2D([0], [0], color=C_ORANGE, lw=6, label="OT"),
        Line2D([0], [0], color=C_MAGENTA, lw=6, label="OT + style cap"),
        Line2D([0], [0], color=C_RED, lw=1, linestyle="--", label="Strict threshold"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.055))
    savefig(fig, "q3_distributions")


def _radar_values(values, angles):
    vals = list(values)
    vals += vals[:1]
    return vals


def plot_radar(P, gen_sensory_med, real_sensory_med):
    targets = P["targets"]
    dim_idx = [targets.index(d) for d in RADAR_DIMS]
    all_vals = []
    for st in TARGET_STYLES:
        all_vals.append(real_sensory_med[st][dim_idx])
        for model in ["copula", "ot"]:
            all_vals.append(gen_sensory_med[(model, st)][dim_idx])
    all_vals = np.vstack(all_vals)
    rmin = float(np.floor(all_vals.min() * 1.1))
    rmax = float(np.ceil(all_vals.max() * 1.1))
    if rmin > 0:
        rmin = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), subplot_kw=dict(polar=True),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.35, h_pad=0.10)
    angles = np.linspace(0, 2 * np.pi, len(RADAR_DIMS), endpoint=False).tolist()
    angles += angles[:1]
    panel_labels = [RADAR_LABELS[d] for d in RADAR_DIMS]
    for ax, st in zip(axes, TARGET_STYLES):
        for model, color, lw, ls in [
            ("real", C_INK, 1.6, "-"),
            ("copula", C_BLUE, 1.8, "-"),
            ("ot", C_ORANGE, 1.8, "--"),
            ("ot_constrained", C_MAGENTA, 1.4, ":"),
        ]:
            vals = real_sensory_med[st][dim_idx] if model == "real" else gen_sensory_med[(model, st)][dim_idx]
            ax.plot(angles, _radar_values(vals, angles), color=color, lw=lw,
                    ls=ls, label=model)
            ax.fill(angles, _radar_values(vals, angles), color=color, alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(panel_labels, fontsize=7.5)
        ax.set_rlim(rmin, rmax)
        ax.set_yticklabels([])
        ax.set_title(st, pad=34)
        ax.grid(color="#e1e0d9", linewidth=0.6)
    handles = [
        Line2D([0], [0], color=C_INK, lw=1.6, label="真实（OOF）"),
        Line2D([0], [0], color=C_BLUE, lw=1.8, label="条件 copula"),
        Line2D([0], [0], color=C_ORANGE, lw=1.8, ls="--", label="条件 OT"),
        Line2D([0], [0], color=C_MAGENTA, lw=1.4, ls=":", label="OT + 风格上限"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    savefig(fig, "q3_radar")


def plot_transfer_radar(P, transfer_sensory_med, real_sensory_med):
    targets = P["targets"]
    dim_idx = [targets.index(d) for d in RADAR_DIMS]
    series = [
        ("source", "Pils/Lager (source)", C_YELLOW, "-"),
        ("raw", "Transfer raw", C_GRAY, ":"),
        ("strict", "Transfer strict", C_BLUE, "-"),
        ("style", "Transfer style-p75", C_ORANGE, "--"),
        ("target", "Tripel (real)", C_VIOLET, "-"),
    ]
    values = []
    for key, _, _, _ in series:
        if key == "target":
            values.append(real_sensory_med[TARGET_STYLE][dim_idx])
        elif key == "source":
            values.append(real_sensory_med[SOURCE_STYLE][dim_idx])
        else:
            values.append(transfer_sensory_med[key][dim_idx])
    values = np.vstack(values)
    rmin = min(0.0, float(np.floor(values.min() * 1.1)))
    rmax = float(np.ceil(values.max() * 1.1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), subplot_kw=dict(polar=True),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.35, h_pad=0.10)
    groups = [RADAR_DIMS[:6], RADAR_DIMS[6:]]
    for ax, dims in zip(axes, groups):
        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]
        pos = [RADAR_DIMS.index(d) for d in dims]
        for (key, label, color, ls), vals in zip(series, values):
            v = vals[pos]
            ax.plot(angles, _radar_values(v, angles), color=color, lw=1.6,
                    ls=ls, label=label)
            ax.fill(angles, _radar_values(v, angles), color=color, alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([RADAR_LABELS[d] for d in dims], fontsize=8)
        ax.set_rlim(rmin, rmax)
        ax.set_yticklabels([])
        ax.grid(color="#e1e0d9", linewidth=0.6)
    axes[0].set_title("Aroma dimensions", pad=18)
    axes[1].set_title("Taste / palate / overall dimensions", pad=18)
    handles = [Line2D([0], [0], color=c, lw=1.6, ls=ls, label=lab)
               for _, lab, c, ls in series]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    savefig(fig, "q3_radar_transfer")


def plot_mmd_classifier(gen_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), constrained_layout=True)
    x = np.arange(len(TARGET_STYLES))
    n_models = len(MODEL_ORDER)
    width = 0.80 / n_models
    for ax, metric, ylabel, title in [
        (axes[0], "mmd", "MMD$^2$（RBF，Z 空间）", "分布距离"),
        (axes[1], "rf_hit", "RF 风格 top-1 命中率",
         "风格分类命中率（颜色同左）"),
    ]:
        for i, model in enumerate(MODEL_ORDER):
            offset = (i - (n_models - 1) / 2.0) * width
            vals = [gen_metrics[(model, st)][metric] for st in TARGET_STYLES]
            ax.bar(x + offset, vals, width, color=MODEL_COLORS[model],
                   label=MODEL_LABELS[model], edgecolor=C_SURFACE, linewidth=0.8,
                   hatch="//" if model == "ot_constrained" else None)
            if model in ("copula", "ot", "ot_constrained"):
                for xi, v in zip(x + offset, vals):
                    ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.0, color=C_INK)
        if metric == "mmd":
            for xi, st in zip(x, TARGET_STYLES):
                floor = gen_metrics[("mean_noise", st)]["mmd_floor"]
                ceil = gen_metrics[("mean_noise", st)]["mmd_ceiling"]
                ax.hlines(floor, xi - 0.42, xi + 0.42, color=C_INK, lw=1.2,
                          linestyle="--")
                ax.hlines(ceil, xi - 0.42, xi + 0.42, color=C_GRAY, lw=1.2,
                          linestyle=":")
            handles, labels = ax.get_legend_handles_labels()
            handles += [Line2D([0], [0], color=C_INK, lw=1.2, ls="--"),
                        Line2D([0], [0], color=C_GRAY, lw=1.2, ls=":")]
            labels += ["真实 bootstrap 下限", "其他风格上限"]
            ax.legend(handles, labels, frameon=False, ncol=2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_STYLES)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
    savefig(fig, "q3_mmd_classifier")


def plot_transfer_defects(transfer_metrics, pils_proxy_p75, tripel_proxy_p75):
    variants = ["source", "raw", "strict", "style"]
    defect_names = list(DEFECTS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(defect_names))
    width = 0.19
    for ax, key, title, ylabel in [
        (axes[0], "above_strict", "超严格绝对阈值", "比例"),
        (axes[1], "above_style_p75", "超 Tripel 风格 p75", "比例"),
        (axes[2], "above_style_cap", "超风格上限 max(阈值, p75)", "比例"),
    ]:
        for i, v in enumerate(variants):
            vals = [transfer_metrics[v][f"{d}_{key}"] for d in defect_names]
            ax.bar(x + (i - 1.5) * width, vals, width, color=VARIANT_COLORS[v],
                   label=VARIANT_LABELS[v], edgecolor=C_SURFACE, linewidth=0.8)
            for xi, val in zip(x + (i - 1.5) * width, vals):
                ax.text(xi, val + 0.01, f"{val:.2f}", ha="center",
                        va="bottom", fontsize=5.8, color=C_INK)
        ax.set_xticks(x)
        ax.set_xticklabels(defect_names)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    # stale hops proxy
    ax = axes[3]
    vals = [transfer_metrics[v]["stale_proxy_above_pils_p75"] for v in variants]
    ax.bar(np.arange(len(variants)), vals,
           color=[VARIANT_COLORS[v] for v in variants],
           edgecolor=C_SURFACE, linewidth=0.8)
    for xi, val in enumerate(vals):
        ax.text(xi, val + 0.01, f"{val:.2f}", ha="center", va="bottom",
                fontsize=7, color=C_INK)
    ax.axhline(0.0, color=C_GRAY, lw=0.8)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("比例")
    ax.set_title("陈旧酒花代理超过 Pils p75")
    ax.text(0.02, 0.96, f"Pils 代理 p75 = {pils_proxy_p75:.2f}\nTripel 代理 p75 = {tripel_proxy_p75:.2f}",
            transform=ax.transAxes, fontsize=7, va="top", color=C_INK)
    handles = [Line2D([0], [0], color=VARIANT_COLORS[v], lw=7, label=VARIANT_LABELS[v])
               for v in variants]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    savefig(fig, "q3_transfer_defects")


def plot_generation_defects(gen_metrics):
    models = ["copula", "ot", "ot_constrained"]
    rows = [f"{st} / {m}" for st in TARGET_STYLES for m in models]
    defect_names = list(DEFECTS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "above_strict", "Above strict threshold"),
        (axes[1], "above_style_p75", "Above style p75"),
        (axes[2], "above_style_cap", "Above style cap max(threshold, p75)"),
    ]:
        mat = np.array([
            [gen_metrics[(m, st)][f"{d}_{key}"] for d in defect_names]
            for st in TARGET_STYLES for m in models
        ])
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val > 0.55 else C_INK
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.8, color=color)
        ax.set_xticks(np.arange(len(defect_names)))
        ax.set_xticklabels(defect_names)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows, fontsize=7)
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    savefig(fig, "q3_generation_defects")


def plot_transfer_compounds(transfer_metrics, P):
    feats = P["feats"]
    src_mean = transfer_metrics["source"]["key_mean"]
    trp_mean = transfer_metrics["target"]["key_mean"]
    fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    x = np.arange(len(KEY_COMPOUNDS))
    width = 0.26
    for i, variant in enumerate(["raw", "strict", "style"]):
        ratios = []
        for c in KEY_COMPOUNDS:
            a = transfer_metrics[variant]["key_mean"][c]
            b = src_mean[c]
            ratios.append(np.log2(max(a, 1e-12) / max(b, 1e-12)))
        ax.bar(x + (i - 1) * width, ratios, width, color=VARIANT_COLORS[variant],
               label=variant, edgecolor=C_SURFACE, linewidth=0.8)
    target_ratio = [
        np.log2(max(trp_mean[c], 1e-12) / max(src_mean[c], 1e-12))
        for c in KEY_COMPOUNDS
    ]
    ax.scatter(x, target_ratio, marker="D", s=28, color=C_VIOLET,
               zorder=5, label="Tripel median / Pils mean")
    ax.axhline(0.0, color=C_INK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([KEY_LABELS[c] for c in KEY_COMPOUNDS], rotation=25,
                       ha="right")
    ax.set_ylabel("log2(post-transfer mean / source mean)")
    ax.set_title("Pils/Lager -> Tripel: key compound changes")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(axis="y")
    savefig(fig, "q3_transfer_compounds")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    P = load_and_prepare()
    feats = P["feats"]
    styles = P["styles"]
    mu, sd = P["mu"], P["sd"]
    Z = P["Z"]
    D = P["D"]

    metric_rows = []

    def add_metric(section, model, style, metric, value, n=None, note=""):
        metric_rows.append({
            "section": section, "model": model, "style": style,
            "metric": metric, "value": value, "n": n, "note": note,
        })

    # -- preprocessing / reference metrics ---------------------------------
    add_metric("preprocessing", "all", "all", "n_samples", len(styles), len(styles),
               "X_cal samples")
    add_metric("preprocessing", "all", "all", "n_features", len(feats), len(feats),
               "calibrated compound columns")
    add_metric("preprocessing", "all", "all", "n_missing", P["n_missing"], len(styles),
               "median-imputed values")
    add_metric("preprocessing", "all", "all", "n_negative_clipped", P["n_negative"], len(styles),
               f"columns: {P['neg_by_col']}")
    add_metric("preprocessing", "all", "all", "zero_fraction", float((P["X_nn"] == 0).mean()),
               len(styles), "after non-negative clipping")
    add_metric("preprocessing", "all", "all", "n_zero_restored_columns",
               int((P["zero_rates"][TARGET_STYLES[0]] >= ZERO_THRESHOLD).sum()),
               len(feats), f"p0 >= {ZERO_THRESHOLD} in at least one target style")

    for st in sorted(set(styles)):
        n = int((styles == st).sum())
        add_metric("style_reference", "real", st, "n", n, n, "")
        for name, info in DEFECTS.items():
            j = feats.index(info["col"])
            vals = P["X_nn"][styles == st, j]
            add_metric("style_reference", "real", st, f"{name}_p50",
                       float(np.median(vals)), n, info["unit"])
            add_metric("style_reference", "real", st, f"{name}_p75",
                       float(np.quantile(vals, 0.75)), n, info["unit"])

    # -- generation ---------------------------------------------------------
    print("[Q3] generating profiles ...", flush=True)
    generated, generated_z, gen_stats, params_z, params_u = generate_profiles(P)

    # -- classifiers on real data ------------------------------------------
    print("[Q3] fitting style classifiers ...", flush=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    )
    lr = LogisticRegression(max_iter=3000, class_weight="balanced",
                            random_state=SEED)
    rf_cv = cross_val_predict(
        RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1),
        Z, styles, cv=skf,
    )
    lr_cv = cross_val_predict(
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           random_state=SEED),
        Z, styles, cv=skf,
    )
    rf.fit(Z, styles)
    lr.fit(Z, styles)
    add_metric("classifier_reference", "RF", "all", "cv_accuracy",
               float((rf_cv == styles).mean()), len(styles), "3-fold stratified")
    add_metric("classifier_reference", "LR", "all", "cv_accuracy",
               float((lr_cv == styles).mean()), len(styles), "3-fold stratified")
    for st in TARGET_STYLES + [SOURCE_STYLE]:
        m = styles == st
        add_metric("classifier_reference", "RF", st, "cv_recall",
                   float((rf_cv[m] == st).mean()), int(m.sum()), "")
        add_metric("classifier_reference", "LR", st, "cv_recall",
                   float((lr_cv[m] == st).mean()), int(m.sum()), "")

    # -- sensory surrogate --------------------------------------------------
    print("[Q3] fitting sensory surrogate (PLS, nested CV) ...", flush=True)
    Y = D["y"].values
    sensory = fit_sensory_surrogate(Z, Y)
    targets = P["targets"]
    real_sensory_oof = sensory["oof"]
    real_sensory_med = {
        st: np.median(real_sensory_oof[styles == st], axis=0)
        for st in sorted(set(styles))
    }
    add_metric("sensory_model", "PLS", "all", "cv_r2_mean",
               sensory["cv_r2_mean"], len(targets), "nested 5-fold")
    add_metric("sensory_model", "PLS", "all", "cv_r2_median",
               sensory["cv_r2_median"], len(targets), "nested 5-fold")
    add_metric("sensory_model", "PLS", "all", "cv_r2_min",
               sensory["cv_r2_min"], len(targets), "nested 5-fold")
    add_metric("sensory_model", "PLS", "all", "n_components",
               sensory["k_final"], len(feats), f"inner CV on all data; per-fold {sensory['chosen_ks']}")
    add_metric("sensory_model", "Ridge", "all", "cv_r2_mean",
               sensory["ridge_cv_r2_mean"], len(targets),
               f"alpha={sensory['ridge_alpha']}")
    for j, t in enumerate(targets):
        add_metric("sensory_model", "PLS", "all", f"cv_r2_{t}",
                   float(sensory["cv_r2"][j]), len(styles), "")

    def predict_sensory(X_conc):
        zz = (np.log1p(np.clip(X_conc, 0.0, None)) - mu) / sd
        return sensory["final"].predict(zz)

    # -- generation metrics -------------------------------------------------
    print("[Q3] evaluating generated profiles ...", flush=True)
    gen_metrics = {}
    gen_sensory_med = {}
    rng_m = np.random.default_rng(SEED + 777)
    for st in TARGET_STYLES:
        real_z = Z[styles == st]
        real_x = P["X_nn"][styles == st]
        n_real = len(real_z)
        gamma = median_heuristic(real_z)
        # real bootstrap floor and other-style ceiling
        floors = []
        for _ in range(30):
            a = real_z[rng_m.integers(0, n_real, n_real)]
            b = real_z[rng_m.integers(0, n_real, n_real)]
            floors.append(mmd2_rbf(a, b, gamma))
        mmd_floor = float(np.mean(floors))
        other_z = Z[styles != st]
        ceils = []
        for _ in range(20):
            idx = rng_m.integers(0, len(other_z), n_real)
            ceils.append(mmd2_rbf(real_z, other_z[idx], gamma))
        mmd_ceiling = float(np.mean(ceils))
        real_mean = real_z.mean(axis=0)
        real_std = real_z.std(axis=0, ddof=0)

        for model in MODEL_ORDER:
            xg = generated[(model, st)]
            zg = generated_z[(model, st)]
            mmds = []
            for _ in range(20):
                idx = rng_m.integers(0, len(zg), n_real)
                mmds.append(mmd2_rbf(real_z, zg[idx], gamma))
            mmd = float(np.mean(mmds))

            div = diversity_metrics(real_z, zg)
            rf_hit, rf_top3, rf_maxp, rf_tp = classifier_metrics(rf, zg, st)
            lr_hit, lr_top3, lr_maxp, lr_tp = classifier_metrics(lr, zg, st)
            # chemical validity
            in_global = float(((xg >= P["global_min"]) & (xg <= P["global_max"])).mean())
            smin = P["X_nn"][styles == st].min(axis=0)
            smax = P["X_nn"][styles == st].max(axis=0)
            in_style = float(((xg >= smin) & (xg <= smax)).mean())
            zero_frac = float((xg == 0).mean())
            real_zero = float((real_x == 0).mean())
            mean_corr = float(pearsonr(xg.mean(axis=0), real_x.mean(axis=0))[0])
            std_ratio = float(np.median(
                (xg.std(axis=0) + 1e-12) / (real_x.std(axis=0) + 1e-12)
            ))
            defect = defect_exceedance(xg, st, feats, P["style_p75"])
            caps_st = style_caps_for(P, st)
            for name, info in DEFECTS.items():
                j = feats.index(info["col"])
                defect[f"{name}_above_style_cap"] = float(
                    (xg[:, j] > caps_st[info["col"]]).mean()
                )
            pred_y = predict_sensory(xg)
            pred_med = np.median(pred_y, axis=0)
            gen_sensory_med[(model, st)] = pred_med
            sens_fid = sensory_fidelity(pred_med, real_sensory_med[st])
            centroid_dist = float(np.linalg.norm(zg.mean(axis=0) - real_mean))
            other_centroids = [
                np.linalg.norm(zg.mean(axis=0) - Z[styles == s2].mean(axis=0))
                for s2 in set(styles) if s2 != st
            ]
            gen_metrics[(model, st)] = {
                "mmd": mmd, "mmd_floor": mmd_floor, "mmd_ceiling": mmd_ceiling,
                "gamma": gamma,
                "rf_hit": rf_hit, "rf_top3": rf_top3, "rf_maxprob": rf_maxp,
                "rf_target_prob": rf_tp,
                "lr_hit": lr_hit, "lr_top3": lr_top3, "lr_maxprob": lr_maxp,
                "lr_target_prob": lr_tp,
                **div,
                "in_global_range": in_global, "in_style_range": in_style,
                "zero_frac": zero_frac, "zero_frac_real": real_zero,
                "mean_corr": mean_corr, "std_ratio": std_ratio,
                **defect,
                "sensory_pearson": sens_fid["pearson"],
                "sensory_spearman": sens_fid["spearman"],
                "sensory_rmse": sens_fid["rmse"],
                "centroid_dist": centroid_dist,
                "centroid_dist_nearest_other": float(min(other_centroids)),
                "neg_frac_preclip": gen_stats[(model, st)]["neg_frac_preclip"],
                "n_upper_clipped": gen_stats[(model, st)]["n_upper_clipped"],
                "n_zero_forced": gen_stats[(model, st)]["n_zero_forced"],
                "n_style_capped": gen_stats[(model, st)]["n_style_capped"],
                "rank_calibrated": int(gen_stats[(model, st)]["rank_calibrated"]),
            }

            for key in ["mmd", "mmd_floor", "mmd_ceiling",
                        "rf_hit", "rf_top3", "rf_maxprob", "lr_hit", "lr_top3",
                        "gen_pair_median", "real_pair_median",
                        "gen_nn_real_median", "real_loo_nn_median",
                        "coverage", "precision", "in_global_range",
                        "in_style_range", "zero_frac", "mean_corr",
                        "std_ratio", "sensory_pearson", "sensory_spearman",
                        "sensory_rmse", "centroid_dist",
                        "centroid_dist_nearest_other", "neg_frac_preclip",
                        "n_upper_clipped", "n_style_capped", "rank_calibrated",
                        "max_oav_median"]:
                add_metric("generation", model, st, key, float(gen_metrics[(model, st)][key]),
                           N_GEN, "")
            for d in DEFECTS:
                for suffix in ["above_strict", "above_style_p75", "above_style_cap"]:
                    add_metric("generation", model, st, f"{d}_{suffix}",
                               float(gen_metrics[(model, st)][f"{d}_{suffix}"]),
                               N_GEN, DEFECTS[d]["unit"])

    # -- transfer -----------------------------------------------------------
    print("[Q3] Pils/Lager -> Tripel transfer ...", flush=True)
    src_mask = styles == SOURCE_STYLE
    tgt_mask = styles == TARGET_STYLE
    Z_src = Z[src_mask]
    X_src = P["X_nn"][src_mask]
    mu_s = params_z[SOURCE_STYLE]["mu"]
    mu_t = params_z[TARGET_STYLE]["mu"]
    A_tr = sqrt_psd(params_z[TARGET_STYLE]["cov_blend"]) @ sqrt_psd(
        params_z[SOURCE_STYLE]["cov_blend"], inverse=True
    )
    z_post = mu_t + (Z_src - mu_s) @ A_tr.T
    X_post_raw = np.clip(inverse_from_z(z_post, mu, sd), 0.0, None)
    X_post_raw = np.minimum(X_post_raw, P["global_max"])
    n_neg_post = int((inverse_from_z(z_post, mu, sd) < 0).sum())

    tripel_p75 = P["style_p75"][TARGET_STYLE]
    strict_caps = {info["col"]: info["strict"] for info in DEFECTS.values()}
    style_caps = {
        "DMS": max(DEFECTS["DMS"]["strict"], tripel_p75["DMS"]),
        "X2.3.butanedione": max(DEFECTS["Diacetyl"]["strict"], tripel_p75["X2.3.butanedione"]),
        "X4VG": 0.10,
        "X.E..2.nonenal": max(DEFECTS["T2N"]["strict"], tripel_p75["X.E..2.nonenal"]),
    }

    def apply_caps(X, caps, source=None):
        """Project values above a cap back to the cap.

        When ``source`` is given, the per-sample cap is
        ``min(cap, source_value)``, which guarantees the defect does not
        increase relative to the source profile (no-worsening projection).
        """
        Xc = X.copy()
        counts = {}
        for col, cap in caps.items():
            j = feats.index(col)
            target_cap = cap if source is None else np.minimum(cap, source[:, j])
            over = Xc[:, j] > target_cap
            counts[col] = int(over.sum())
            if source is None:
                Xc[over, j] = cap
            else:
                Xc[over, j] = target_cap[over]
        if source is not None:
            # Stale-hop proxy: no-worsening on every proxy feature, so the
            # composite proxy also cannot increase relative to the source.
            n_proxy = 0
            for col in PROXY_COLS:
                j = feats.index(col)
                over = Xc[:, j] > source[:, j]
                n_proxy += int(over.sum())
                Xc[over, j] = source[over, j]
            counts["proxy_columns_no_worsen"] = n_proxy
        return Xc, counts

    X_strict, cap_counts_strict = apply_caps(X_post_raw, strict_caps, source=X_src)
    X_style, cap_counts_style = apply_caps(X_post_raw, style_caps)
    variants = {
        "source": X_src,
        "raw": X_post_raw,
        "strict": X_strict,
        "style": X_style,
    }

    # stale proxy reference
    proxy_src = stale_proxy(X_src, feats, mu, sd)
    proxy_tgt = stale_proxy(P["X_nn"][tgt_mask], feats, mu, sd)
    pils_proxy_p75 = float(np.quantile(proxy_src, 0.75))
    tripel_proxy_p75 = float(np.quantile(proxy_tgt, 0.75))

    transfer_metrics = {}
    transfer_sensory_med = {}
    transfer_df_rows = []
    for variant, Xv in variants.items():
        style_label = SOURCE_STYLE if variant == "source" else TARGET_STYLE
        zv = (np.log1p(np.clip(Xv, 0.0, None)) - mu) / sd
        rf_hit, rf_top3, rf_maxp, rf_tp = classifier_metrics(rf, zv, TARGET_STYLE)
        lr_hit, lr_top3, lr_maxp, lr_tp = classifier_metrics(lr, zv, TARGET_STYLE)
        rf_hit_src = float((rf.predict(zv) == SOURCE_STYLE).mean())
        lr_hit_src = float((lr.predict(zv) == SOURCE_STYLE).mean())
        pred_y = predict_sensory(Xv)
        pred_med = np.median(pred_y, axis=0)
        transfer_sensory_med[variant] = pred_med
        fid_tgt = sensory_fidelity(pred_med, real_sensory_med[TARGET_STYLE])
        fid_src = sensory_fidelity(pred_med, real_sensory_med[SOURCE_STYLE])
        delta_post = pred_med - real_sensory_med[SOURCE_STYLE]
        delta_target = real_sensory_med[TARGET_STYLE] - real_sensory_med[SOURCE_STYLE]
        cos = float(np.dot(delta_post, delta_target) / (
            np.linalg.norm(delta_post) * np.linalg.norm(delta_target) + 1e-12
        ))
        defect = defect_exceedance(Xv, TARGET_STYLE, feats, P["style_p75"])
        for name, info in DEFECTS.items():
            j = feats.index(info["col"])
            defect[f"{name}_above_style_cap"] = float(
                (Xv[:, j] > style_caps[info["col"]]).mean()
            )
        # paired worsening vs source
        worsening = {}
        for name, info in DEFECTS.items():
            j = feats.index(info["col"])
            worsening[f"{name}_worsened_frac"] = float((Xv[:, j] > X_src[:, j]).mean())
            worsening[f"{name}_mean_delta"] = float(np.mean(Xv[:, j] - X_src[:, j]))
        proxy = stale_proxy(Xv, feats, mu, sd)
        proxy_stats = {
            "stale_proxy_mean": float(np.mean(proxy)),
            "stale_proxy_p75": float(np.quantile(proxy, 0.75)),
            "stale_proxy_above_pils_p75": float((proxy > pils_proxy_p75).mean()),
            "stale_proxy_above_tripel_p75": float((proxy > tripel_proxy_p75).mean()),
        }
        key_mean = {c: float(np.mean(Xv[:, feats.index(c)])) for c in KEY_COMPOUNDS}
        key_median = {c: float(np.median(Xv[:, feats.index(c)])) for c in KEY_COMPOUNDS}
        transfer_metrics[variant] = {
            "rf_hit_tripel": rf_hit, "rf_top3_tripel": rf_top3,
            "rf_maxprob": rf_maxp, "rf_target_prob": rf_tp,
            "rf_hit_pils": rf_hit_src,
            "lr_hit_tripel": lr_hit, "lr_top3_tripel": lr_top3,
            "lr_maxprob": lr_maxp, "lr_target_prob": lr_tp,
            "lr_hit_pils": lr_hit_src,
            "sensory_pearson_tripel": fid_tgt["pearson"],
            "sensory_spearman_tripel": fid_tgt["spearman"],
            "sensory_rmse_tripel": fid_tgt["rmse"],
            "sensory_pearson_pils": fid_src["pearson"],
            "sensory_rmse_pils": fid_src["rmse"],
            "sensory_direction_cosine": cos,
            **defect, **worsening, **proxy_stats,
            "key_mean": key_mean, "key_median": key_median,
        }
        for key in ["rf_hit_tripel", "rf_top3_tripel", "rf_maxprob",
                    "rf_hit_pils", "lr_hit_tripel", "lr_top3_tripel",
                    "lr_maxprob", "lr_hit_pils",
                    "sensory_pearson_tripel", "sensory_spearman_tripel",
                    "sensory_rmse_tripel", "sensory_pearson_pils",
                    "sensory_rmse_pils", "sensory_direction_cosine",
                    "stale_proxy_mean", "stale_proxy_p75",
                    "stale_proxy_above_pils_p75", "stale_proxy_above_tripel_p75",
                    "max_oav_median"]:
            add_metric("transfer", variant, style_label, key,
                       float(transfer_metrics[variant][key]), len(X_src), "")
        for name in DEFECTS:
            for suffix in ["above_strict", "above_style_p75", "above_style_cap",
                           "worsened_frac"]:
                add_metric("transfer", variant, style_label, f"{name}_{suffix}",
                           float(transfer_metrics[variant][f"{name}_{suffix}"]),
                           len(X_src), "")
        for c in KEY_COMPOUNDS:
            add_metric("transfer", variant, style_label, f"key_mean_{c}",
                       float(key_mean[c]), len(X_src), DEFECTS.get(c, {}).get("unit", ""))
        for i in range(len(X_src)):
            row = {"style": style_label, "sample_id": f"pils_{P['beer_ids'][src_mask][i]:03d}",
                   "variant": variant}
            row.update(dict(zip(feats, Xv[i])))
            transfer_df_rows.append(row)

    # target reference key means
    transfer_metrics["target"] = {
        "key_mean": {c: float(np.mean(P["X_nn"][tgt_mask, feats.index(c)])) for c in KEY_COMPOUNDS},
        "key_median": {c: float(np.median(P["X_nn"][tgt_mask, feats.index(c)])) for c in KEY_COMPOUNDS},
    }
    # add strict/style cap counts to summary later
    # -- portraits ----------------------------------------------------------
    print("[Q3] writing outputs ...", flush=True)
    portrait_rows = []
    for i in range(len(styles)):
        row = {"kind": "real", "model": "real_oof", "style": styles[i],
               "sample_id": str(P["beer_ids"][i])}
        row.update(dict(zip(targets, real_sensory_oof[i])))
        portrait_rows.append(row)
    for (model, st), Xg in generated.items():
        pred_y = predict_sensory(Xg)
        for i in range(len(Xg)):
            row = {"kind": "generated", "model": model, "style": st,
                   "sample_id": f"{st}_{model}_{i:04d}"}
            row.update(dict(zip(targets, pred_y[i])))
            portrait_rows.append(row)
    for variant, Xv in variants.items():
        style_label = SOURCE_STYLE if variant == "source" else TARGET_STYLE
        pred_y = predict_sensory(Xv)
        for i in range(len(Xv)):
            row = {"kind": "transfer", "model": variant, "style": style_label,
                   "sample_id": f"pils_{P['beer_ids'][src_mask][i]:03d}"}
            row.update(dict(zip(targets, pred_y[i])))
            portrait_rows.append(row)

    # -- DataFrames ---------------------------------------------------------
    gen_frames = []
    for (model, st), Xg in generated.items():
        df = pd.DataFrame(Xg, columns=feats)
        df.insert(0, "generator", model)
        df.insert(0, "sample_id", [f"{st}_{model}_{i:04d}" for i in range(len(Xg))])
        df.insert(0, "style", st)
        gen_frames.append(df)
    gen_df = pd.concat(gen_frames, ignore_index=True)
    gen_df.to_csv(OUT_DIR / "generated_profiles.csv", index=False, float_format="%.8g")

    transfer_df = pd.DataFrame(transfer_df_rows)
    transfer_df.to_csv(OUT_DIR / "transfer_profiles.csv", index=False, float_format="%.8g")

    portraits_df = pd.DataFrame(portrait_rows)
    portraits_df.to_csv(OUT_DIR / "portraits.csv", index=False, float_format="%.8g")

    # -- figures ------------------------------------------------------------
    print("[Q3] plotting ...", flush=True)
    plot_pca(P, generated, gen_sensory_med)
    plot_distributions(P, generated)
    plot_radar(P, gen_sensory_med, real_sensory_med)
    plot_transfer_radar(P, transfer_sensory_med, real_sensory_med)
    plot_mmd_classifier(gen_metrics)
    plot_generation_defects(gen_metrics)
    plot_transfer_defects(transfer_metrics, pils_proxy_p75, tripel_proxy_p75)
    plot_transfer_compounds(transfer_metrics, P)

    # -- metrics CSV --------------------------------------------------------
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False, float_format="%.8g")

    # -- summary ------------------------------------------------------------
    runtime = time.time() - t0
    # best model per style by MMD and by RF hit
    best_mmd = {st: min(MODEL_ORDER, key=lambda m: gen_metrics[(m, st)]["mmd"])
                for st in TARGET_STYLES}
    best_hit = {st: max(MODEL_ORDER, key=lambda m: gen_metrics[(m, st)]["rf_hit"])
                for st in TARGET_STYLES}

    oc = {st: gen_metrics[("ot_constrained", st)] for st in TARGET_STYLES}
    oc_mean_mmd = float(np.mean([oc[st]["mmd"] for st in TARGET_STYLES]))
    oc_defect_style = {
        d: {st: float(oc[st][f"{d}_above_style_p75"]) for st in TARGET_STYLES}
        for d in DEFECTS
    }
    oc_defect_strict = {
        d: {st: float(oc[st][f"{d}_above_strict"]) for st in TARGET_STYLES}
        for d in DEFECTS
    }
    oc_defect_cap = {
        d: {st: float(oc[st][f"{d}_above_style_cap"]) for st in TARGET_STYLES}
        for d in DEFECTS
    }
    oc_mean_style = {d: float(np.mean(list(oc_defect_style[d].values()))) for d in DEFECTS}
    oc_mean_strict = {d: float(np.mean(list(oc_defect_strict[d].values()))) for d in DEFECTS}
    oc_mean_cap = {d: float(np.mean(list(oc_defect_cap[d].values()))) for d in DEFECTS}
    oc_overall_style = float(np.mean([oc_mean_style[d] for d in DEFECTS]))
    oc_overall_strict = float(np.mean([oc_mean_strict[d] for d in DEFECTS]))
    oc_overall_cap = float(np.mean([oc_mean_cap[d] for d in DEFECTS]))
    oc_style_caps = {
        st: {k: float(v) for k, v in gen_stats[("ot_constrained", st)]["style_caps"].items()}
        for st in TARGET_STYLES
    }
    # key findings strings, all from computed metrics
    findings = []
    for st in TARGET_STYLES:
        findings.append(
            f"{st}: best MMD = {best_mmd[st]} ({gen_metrics[(best_mmd[st], st)]['mmd']:.4f}); "
            f"best RF hit = {best_hit[st]} ({gen_metrics[(best_hit[st], st)]['rf_hit']:.3f}); "
            f"copula RF hit = {gen_metrics[('copula', st)]['rf_hit']:.3f}, "
            f"OT RF hit = {gen_metrics[('ot', st)]['rf_hit']:.3f}."
        )
    findings.append(
        "Transfer: strict caps -> above-strict rates "
        + ", ".join(f"{d}={transfer_metrics['strict'][f'{d}_above_strict']:.2f}" for d in DEFECTS)
        + "; style-p75 caps -> "
        + ", ".join(f"{d}={transfer_metrics['style'][f'{d}_above_strict']:.2f}" for d in DEFECTS)
        + " (above strict), and above-style-p75 "
        + ", ".join(f"{d}={transfer_metrics['style'][f'{d}_above_style_p75']:.2f}" for d in DEFECTS)
        + "."
    )
    findings.append(
        f"Transfer RF: Tripel hit source={transfer_metrics['source']['rf_hit_tripel']:.2f}, "
        f"raw={transfer_metrics['raw']['rf_hit_tripel']:.2f}, "
        f"strict={transfer_metrics['strict']['rf_hit_tripel']:.2f}, "
        f"style={transfer_metrics['style']['rf_hit_tripel']:.2f}."
    )
    findings.append(
        f"Final generator ot_constrained: mean MMD across Tripel/Kriek/Lambic = "
        f"{oc_mean_mmd:.4f}; mean above-style-p75 rate = {oc_overall_style:.4f} "
        f"(per defect: " + ", ".join(f"{d}={oc_mean_style[d]:.3f}" for d in DEFECTS)
        + f"); mean above-strict rate = {oc_overall_strict:.4f} "
        f"(per defect: " + ", ".join(f"{d}={oc_mean_strict[d]:.3f}" for d in DEFECTS)
        + f"); mean above-style-cap rate = {oc_overall_cap:.4f} "
        f"(per defect: " + ", ".join(f"{d}={oc_mean_cap[d]:.3f}" for d in DEFECTS) + ")."
    )
    worst_sens = sorted(
        [(t, float(sensory["cv_r2"][j])) for j, t in enumerate(targets)],
        key=lambda x: x[1]
    )[:8]
    findings.append(
        "Sensory surrogate: PLS nested-CV mean R2 = "
        f"{sensory['cv_r2_mean']:.3f}; weakest dimensions "
        + ", ".join(f"{t} ({v:.2f})" for t, v in worst_sens) + "."
    )

    summary = {
        "meta": {
            "task": "Q3 style-conditional generation and Pils/Lager -> Tripel transfer",
            "seed": SEED,
            "n_generated_per_style": N_GEN,
            "target_styles": TARGET_STYLES,
            "source_style": SOURCE_STYLE,
            "target_style": TARGET_STYLE,
            "runtime_s": round(runtime, 2),
            "n_real": int(len(styles)),
            "n_features": len(feats),
        },
        "preprocessing": {
            "input": "X_cal (linear calibrated concentrations)",
            "transform": "log1p -> per-column standardisation; inverse = expm1(z*sd+mu)",
            "n_missing": P["n_missing"],
            "n_negative_clipped": P["n_negative"],
            "negative_columns": P["neg_by_col"],
            "zero_fraction_after_clip": float((P["X_nn"] == 0).mean()),
            "zero_restoration_threshold": ZERO_THRESHOLD,
            "note": "Negatives in X_cal (betaglucan, unfermentables) are below-detection/calibration artefacts; clipped to 0 and counted.",
        },
        "generation": {
            "models": MODEL_ORDER,
            "model_labels": MODEL_LABELS,
            "ot_rank_calibration": "the affine OT map is followed by a per-feature rank/quantile map to the target style's empirical marginals",
            "n_per_style": N_GEN,
            "style_params": {
                st: {
                    "n": int(params_z[st]["n"]),
                    "ot_cov_shrinkage_to_global": float(params_z[st]["alpha"]),
                    "ot_scale_shrinkage_to_global": float(params_z[st]["w"]),
                    "copula_cov_shrinkage_to_global": float(params_u[st]["alpha"]),
                    "copula_scale_shrinkage_to_global": float(params_u[st]["w"]),
                    "copula_n_quantiles": int(min(max(params_u[st]["n"], 5), 100)),
                } for st in TARGET_STYLES
            },
            "metrics": {
                f"{model}|{st}": {
                    k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in gen_metrics[(model, st)].items()
                } for model in MODEL_ORDER for st in TARGET_STYLES
            },
            "best_model_by_mmd": best_mmd,
            "best_model_by_rf_hit": best_hit,
        },
        "final_generator": {
            "model": "ot_constrained",
            "description": "Conditional affine OT (with target-marginal rank calibration) projected onto max(strict threshold, style p75) for DMS/diacetyl/4VG/T2N; 4VG uses the 0.10 mg/L non-wheat safety value.",
            "mean_mmd_across_styles": oc_mean_mmd,
            "per_style_mmd": {st: float(oc[st]["mmd"]) for st in TARGET_STYLES},
            "style_caps": oc_style_caps,
            "above_style_p75_by_style": oc_defect_style,
            "mean_above_style_p75_by_defect": oc_mean_style,
            "mean_above_style_p75_all_defects": oc_overall_style,
            "above_strict_by_style": oc_defect_strict,
            "mean_above_strict_by_defect": oc_mean_strict,
            "mean_above_strict_all_defects": oc_overall_strict,
            "above_style_cap_by_style": oc_defect_cap,
            "mean_above_style_cap_by_defect": oc_mean_cap,
            "mean_above_style_cap_all_defects": oc_overall_cap,
            "n_style_capped": {
                st: int(gen_stats[("ot_constrained", st)]["n_style_capped"])
                for st in TARGET_STYLES
            },
        },
        "classifier": {
            "rf_cv_accuracy": float((rf_cv == styles).mean()),
            "lr_cv_accuracy": float((lr_cv == styles).mean()),
            "cv": "3-fold stratified on 250 real samples, 22 classes",
        },
        "transfer": {
            "source": SOURCE_STYLE,
            "target": TARGET_STYLE,
            "n_source": int(src_mask.sum()),
            "n_negative_before_clip": n_neg_post,
            "strict_caps": strict_caps,
            "style_caps": style_caps,
            "strict_projection": "per-sample min(strict_cap, source_value) for the four named defects, plus all stale-hop proxy features capped at their source values: guarantees post <= source and post <= strict cap (proxy included)",
            "style_projection": "cap only at max(strict threshold, Tripel p75); may raise a clean source profile up to the style cap",
            "tripel_p75": tripel_p75,
            "cap_counts_strict": cap_counts_strict,
            "cap_counts_style": cap_counts_style,
            "pils_stale_proxy_p75": pils_proxy_p75,
            "tripel_stale_proxy_p75": tripel_proxy_p75,
            "metrics": {
                v: {k: (float(x) if isinstance(x, (int, float, np.floating, np.integer)) else x)
                    for k, x in transfer_metrics[v].items()}
                for v in ["source", "raw", "strict", "style"]
            },
            "target_key_means": transfer_metrics["target"]["key_mean"],
        },
        "sensory_model": {
            "family": "PLSRegression (Ridge comparison)",
            "n_components": sensory["k_final"],
            "inner_cv_components_per_outer_fold": sensory["chosen_ks"],
            "cv_r2_mean": sensory["cv_r2_mean"],
            "cv_r2_median": sensory["cv_r2_median"],
            "cv_r2_min": sensory["cv_r2_min"],
            "ridge_alpha": sensory["ridge_alpha"],
            "ridge_cv_r2_mean": sensory["ridge_cv_r2_mean"],
            "cv_r2_by_target": {t: float(sensory["cv_r2"][j]) for j, t in enumerate(targets)},
            "note": "Nested 5-fold CV; DMS/diacetyl/stale_hops/t2n are weakly predicted and their portraits are indicative only.",
        },
        "key_findings": findings,
        "outputs": [
            "data/q3/generated_profiles.csv",
            "data/q3/transfer_profiles.csv",
            "data/q3/metrics.csv",
            "data/q3/portraits.csv",
            "data/q3/summary.json",
            "img/q3_pca.png/pdf",
            "img/q3_distributions.png/pdf",
            "img/q3_radar.png/pdf",
            "img/q3_radar_transfer.png/pdf",
            "img/q3_mmd_classifier.png/pdf",
            "img/q3_generation_defects.png/pdf",
            "img/q3_transfer_defects.png/pdf",
            "img/q3_transfer_compounds.png/pdf",
        ],
        "limitations": [
            "X_cal has 69 missing values (median-imputed) and 3 negative values in two columns (clipped).",
            "Kriek (n=10) and Lambic (n=11) are small; per-style covariance and p75 estimates are unstable.",
            "Stale hops is not a compound; the proxy uses T2N, DMS, ethyl 2-methylbutyrate, ethyl isovalerate and 4VG because 4-ethylphenol/4-ethylguaiacol are not in the 70 calibrated columns.",
            "T2N is treated as ug/L based on its numerical range; the attachment unit label says mg/L and is inconsistent.",
            "Sensory predictions are a surrogate (PLS on 50 sensory targets), not measured panel scores; DMS/diacetyl/stale_hops/t2n have low CV R2.",
            "The style classifier is trained on 250 real samples with 3-31 samples/class; generated hit rates are descriptive, not a formal two-sample test.",
        ],
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(to_py(summary), f, ensure_ascii=False, indent=2)

    # print a concise report for the terminal
    print(f"[Q3] done in {runtime:.1f}s", flush=True)
    print(f"[Q3] generated rows: {len(gen_df)}, transfer rows: {len(transfer_df)}, "
          f"portrait rows: {len(portraits_df)}, metric rows: {len(metrics_df)}", flush=True)
    for st in TARGET_STYLES:
        print(f"[Q3] {st}: " + "; ".join(
            f"{m}: MMD={gen_metrics[(m, st)]['mmd']:.3f}, RFhit={gen_metrics[(m, st)]['rf_hit']:.2f}, "
            f"coh={gen_metrics[(m, st)]['coverage']:.2f}"
            for m in MODEL_ORDER), flush=True)
    print("[Q3] transfer:", flush=True)
    for v in ["source", "raw", "strict", "style"]:
        print(f"  {v}: RF Tripel hit={transfer_metrics[v]['rf_hit_tripel']:.2f}, "
              f"DMS>{DEFECTS['DMS']['strict']}: {transfer_metrics[v]['DMS_above_strict']:.2f}, "
              f"Dia>{DEFECTS['Diacetyl']['strict']}: {transfer_metrics[v]['Diacetyl_above_strict']:.2f}, "
              f"T2N>{DEFECTS['T2N']['strict']}: {transfer_metrics[v]['T2N_above_strict']:.2f}, "
              f"stale proxy>Pils p75: {transfer_metrics[v]['stale_proxy_above_pils_p75']:.2f}", flush=True)


if __name__ == "__main__":
    main()
