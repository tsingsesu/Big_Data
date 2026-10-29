# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本次仅整理运行入口、路径和可视化范围，核心计算保留原程序；更早的AI参与见使用详情说明。
"""Q2: compound co-expression network -> sensory Top-3 compounds + stability + chemistry.

Run:
    python src/问题二_化合物关联.py

Outputs:
    data/q2/*.csv
    绘图数据由01_数据预处理.py --plots准备，图形由论文配图.R生成。

Design notes
------------
* 2.csv is verified as Spearman; the given graph is fixed during bootstrap.
* 1.csv supplies the 250 x 231 abundance matrix; 4.csv supplies 50 sensory targets.
* Five ranking methods are kept side by side for audit:
  A marginal Spearman (BH-FDR within each target)
  B per-target ElasticNet with train-only preprocessing and grouped 3-fold pipeline CV
  C per-target ExtraTrees impurity importance
  D per-target LightGBM descriptive mean |SHAP|, identical point/bootstrap fits
  E network-smoothed marginal correlation (own + 1-hop neighbours)
* Primary Top-3 restricts A-E Borda fusion to 214 chemical candidates; all-indicator results are diagnostic.
* Stability: 500 bootstrap resamples of the 250 beers; A-E are rerun with
  identical estimator budgets, refit preprocessing and retuned ElasticNet; the graph is fixed.
* 6.csv is used as an external consistency check only for the 44 compounds
  that can be matched by name.  It is NOT an independent data source: it is
  computed from the same 250 beers, so method A reproduces it almost exactly.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
import re
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import hypergeom, rankdata, spearmanr, t as student_t
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from 公共数据处理 import TARGET_CN, load_all, provenance
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'q2'
IMG = ROOT / 'img'
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BOOT = int(os.environ.get('Q2_NBOOT', '500'))
N_JOBS = int(os.environ.get('Q2_NJOBS', str(min(12, os.cpu_count() or 4))))
N_TREES_FULL = 100
N_TREES_BOOT = N_TREES_FULL
LGB_FULL = 100
LGB_BOOT = LGB_FULL
NET_LAMBDA = 0.3
METHODS = ['A', 'B', 'C', 'D', 'E']
METHOD_LABEL = {'A': '边际 Spearman', 'B': '弹性网', 'C': '极端随机树', 'D': 'LightGBM SHAP', 'E': '网络平滑', 'FUSED': '融合（Borda）'}
METHOD_SHORT = {'A': '边际', 'B': '弹性网', 'C': '极端树', 'D': 'SHAP', 'E': '网络平滑'}
C6_COMPOUND_MAP = {'Ethyl acetate': 'ethyl_acetate', 'Ethyl decanoate': 'ethyl_decanoate', 'Ethyl hexanoate': 'ethyl_hexanoate', 'Ethyl octanoate': 'ethyl_octanoate', 'Ethyl propionate': 'ethyl_propionate', 'Isobutyl acetate': 'isobutyl_acetate', 'Isoamyl acetate': 'isopentyl_acetate', 'Octyl acetate': 'octyl_acetate', 'Phenethyl acetate': 'phenethyl_acetate', 'Ethyl 2-methylbutyrate': 'ethyl_2.methyl_butyrate', 'Ethyl 3-methylbutyrate': 'ethyl_isovalerate', 'Ethanol': 'ethanol..v.v.', 'Isoamyl alcohol': 'isoamyl_alcohol', 'Isobutanol': 'isobutanol', 'Phenethyl alcohol': 'phenethyl_alcohol', 'Glycerol': 'glycerol', 'Iso-alpha acids': 'bitterness', 'Alpha-humulene': 'alpha.humulene', 'Caryophyllene': 'E.caryophyllene', 'Alpha-terpineol': 'X1.alpha.terpineol', 'Citronellol': 'citronellol', 'Geraniol': 'geraniol', 'Linalool': 'linalool', 'Z-nerolidol': 'Z.nerolidol', 'Acetic acid': 'acetic_acid.g.L.', 'Lactic acid': 'lactic_acid.mg.L.', 'Diacetyl': 'X2.3.butanedione', '4-vinylguaiacol': 'X4VG', 'Acetaldehyde': 'acetaldehyde', 'Carbon disulfide': 'CS2', 'Dimethyl sulfide': 'DMS', 'Diethyl sulfide': 'DES', 'Hydrogen sulfide': 'H2S', 'Methanethiol': 'MeSH', 'Sulfur dioxide': 'SO2.mg.L.', 'Carbon dioxide': 'CO2.PSI.', 'pH': 'pH', 'Ammonia': 'ammonia.mg.L.', 'Betaglucan': 'betaglucan.mg.L.', 'Color': 'color.EBC.', 'Iron': 'iron', 'Protein': 'protein.g.L.', 'Sugars': 'GFS', 'Caloric value': 'kcalperc'}
DERIVED = {'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum'}

def normalise_name(s) -> str:
    s = str(s).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ')
    return re.sub('\\s+', ' ', s).strip()

def chemical_class(feature: str, chemical_name: str) -> str:
    """Coarse, documented chemical class from the measured compound name.

    The attachment has no compound-class column, so classes are derived from
    IUPAC/trivial-name functional-group suffixes.  Rules are ordered so that
    terpenoids / phenols / lactones win over generic '-ol', '-one', '-al'.
    """
    s = feature.lower()
    if feature in DERIVED:
        return 'Derived index'
    if feature == 'bitterness':
        return 'Hop acids'
    if s in ('dms', 'h2s', 'mesh', 'cs2', 'des') or s.startswith('so2'):
        return 'Sulfur compounds'
    if 'lactone' in s:
        return 'Lactones'
    if 'acid' in s:
        return 'Organic acids'
    if s in ('x4vg', 'x4.ethyl.phenol', 'x4.ethyl.guaiacol', 'eugenol', 'vanillin', 'ortho.cresol', 'guaiacol') or 'isoeugenol' in s:
        return 'Volatile phenols'
    if any((k in s for k in ['aldehyde', 'furfural', 'nonenal', 'decadienal', 'nonanal', 'methylbutanal', 'butenal', 'apofarnesal', 'farnesal', 'cyclohomocitral', 'cinnamaldehyde', 'dodecanal', 'citral'])):
        return 'Aldehydes'
    if any((k in s for k in ['butanedione', 'undecanone', 'methyltetrahydrofuran.3.one', 'hepten.2.one', 'car.3.en.2.one', 'damascenone', 'ionone', 'nootkatone'])):
        return 'Ketones'
    if any((k in s for k in ['pyrazine', 'pyrrole', 'acetylfuran', 'furanmethanol', 'pentyl.furan', 'methyl.benzofuran', 'isomaltol', 'furanone'])):
        return 'Furans/pyrazines'
    if any((k in s for k in ['linalool', 'geraniol', 'citronellol', 'nerol', 'terpineol', 'nerolidol', 'terpinen.4.ol', 'eucalyptol', 'menthol', 'borneol', 'caryophyllene.oxide', 'humulene.epoxide', 'epoxy', 'caryophyllane', 'muurolol', 'cadinol', 'eudesmol', 'spathulenol', 'manool', 'ambroxide', 'farnesol', 'ocimenone', 'hydroxy', 'cubenol', 'aromadendrene', 'geosmin'])):
        return 'Terpenoids'
    if any((k in s for k in ['humulene', 'caryophyllene', 'farnesene', 'ocimene', 'phellandrene', 'limonene', 'terpinolene', 'germacrene', 'cadalene', 'calacorene', 'valencene', 'cymenene', 'pinene', 'myrcene', 'sabinene', 'carene', 'muurolene', 'cadinene'])):
        return 'Terpene hydrocarbons'
    if any((k in s for k in ['acetate', 'butyrate', 'hexanoate', 'octanoate', 'decanoate', 'propionate', 'isobutyrate', 'isovalerate', 'formate', 'lactate', 'succinate', 'nicotinate', 'benzoate', 'cinnamate', 'salicylate', 'laurate', 'tetradecanoate', 'pentadecanoate', 'hexadecanoate', 'octadecanoate', 'tiglate', 'furoate', 'phenylacetate', 'hydrocinnamate', 'anisate', 'geranate', 'farnesoate', 'pentanoate', 'hexenoate', 'decenoate', 'butanoate', 'propanoate', 'isobutanoate', 'enoate', 'caprylate'])):
        return 'Esters'
    if s in ('glycerol', 'propylene.glycol'):
        return 'Polyols'
    if any((k in s for k in ['isoamyl_alcohol', 'isobutanol', 'propanol', 'pentanol', 'butanol', 'hexanol', 'heptanol', 'octanol', 'nonanol', 'decanol', 'dodecanol', 'tetradecanol', 'phenethyl_alcohol', 'benzyl.alcohol', 'methyl.1.butanol', 'methyl.2.hexanol', 'octen.3.ol', 'methyl.1.pentanol', 'methyl.1.octanol', 'furanmethanol', 'hexenol', 'nonan.2.ol', 'ethanol..v.v.'])):
        return 'Alcohols'
    if any((k in s for k in ['gfs', 'unfermentables', 'betaglucan', 'protein', 'color', 'ph', 'co2', 'kcal', 'iron', 'ammonia'])):
        return 'Physicochemical'
    return 'Other'

def load_data() -> dict:
    D = load_all()
    fm = D['feature_meta'].copy()
    feats = fm['feature'].tolist()
    corr2 = D['corr2'].copy()
    assert list(fm['compound_2csv']) == list(corr2.columns), 'feature/corr2 order mismatch'
    assert list(corr2.index) == list(corr2.columns)
    X = D['X'][feats].astype(float)
    y = D['y'].copy()
    assert X.shape == (250, 231) and y.shape == (250, 50)
    assert not X.isna().all().any(), 'all-NaN feature'
    fm['chemical_class'] = [chemical_class(f, n) for (f, n) in zip(fm['feature'], fm['chemical_name'])]
    med = X.median()
    X_filled = X.fillna(med)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_filled.values)
    return {'D': D, 'fm': fm, 'feats': feats, 'corr2': corr2, 'X': X, 'X_filled': X_filled, 'Xs': Xs, 'y': y, 'y_arr': y.values.astype(float), 'target_meta': D['target_meta'].copy(), 'c6': D['corr6'].copy(), 'thresholds': D['thresholds'].copy(), 'styles': D['styles']}

def build_corr6_maps(data: dict) -> dict:
    """Map 44 corr6 rows to 1.csv features and 50 targets to corr6 columns."""
    feats = data['feats']
    c6 = data['c6']
    tm = data['target_meta'].set_index('target')
    assert all((r in c6.index for r in C6_COMPOUND_MAP))
    assert all((f in feats for f in C6_COMPOUND_MAP.values()))
    match_feats = [C6_COMPOUND_MAP[r] for r in c6.index]
    match_idx = np.array([feats.index(f) for f in match_feats])
    feat_to_c6row = {f: r for (r, f) in C6_COMPOUND_MAP.items()}
    cols = {normalise_name(c): c for c in c6.columns}
    target_to_c6 = {}
    for t in data['y'].columns:
        cn = normalise_name(tm.loc[t, 'CleanName_Paper'])
        if t.startswith('A_'):
            (a, b) = cn.split(' - ', 1)
            cand = f'{a} aroma - {b}'
        elif t.startswith('F_'):
            (a, b) = cn.split(' - ', 1)
            cand = f'{a} taste - {b}'
        else:
            cand = cn
        col = cols.get(cand)
        if col is None:
            low = cand.lower()
            for c in cols:
                if low == c.lower() or low == c.lower().replace(' (trained panel)', ''):
                    col = cols[c]
                    break
        target_to_c6[t] = col
    assert all((v is not None for v in target_to_c6.values())), [t for (t, v) in target_to_c6.items() if v is None]
    return {'match_idx': match_idx, 'match_feats': match_feats, 'feat_to_c6row': feat_to_c6row, 'target_to_c6': target_to_c6}

def build_graph(corr2: pd.DataFrame, feats: list[str], scheme: str, k: int | None=None, threshold: float | None=None):
    vals = corr2.values.astype(float)
    abs_vals = np.abs(vals).copy()
    np.fill_diagonal(abs_vals, 0.0)
    n = len(feats)
    if scheme == 'abs_threshold':
        mask = abs_vals >= float(threshold)
    elif scheme == 'topk':
        mask = np.zeros_like(abs_vals, dtype=bool)
        order = np.argsort(-abs_vals, axis=1)[:, :int(k)]
        for i in range(n):
            mask[i, order[i]] = True
        mask = mask | mask.T
        np.fill_diagonal(mask, False)
    else:
        raise ValueError(scheme)
    adj = pd.DataFrame(np.where(mask, abs_vals, 0.0), index=feats, columns=feats)
    G = nx.from_pandas_adjacency(adj)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    for (u, v) in G.edges():
        G[u][v]['r'] = float(corr2.loc[u, v])
        G[u][v]['weight'] = float(abs(corr2.loc[u, v]))
    return (G, mask)

def community_stats(G: nx.Graph, comms: list[set], corr2: pd.DataFrame) -> dict:
    sizes = sorted([len(c) for c in comms], reverse=True)
    mod = nx.community.modularity(G, comms, weight='weight') if G.number_of_edges() else np.nan
    comm_of = {}
    for (ci, c) in enumerate(comms):
        for node in c:
            comm_of[node] = ci
    intra_edges = [abs(corr2.loc[u, v]) for (u, v) in G.edges() if comm_of.get(u) == comm_of.get(v)]
    inter_edges = [abs(corr2.loc[u, v]) for (u, v) in G.edges() if comm_of.get(u) != comm_of.get(v)]
    intra_pairs = []
    for c in comms:
        nodes = list(c)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                intra_pairs.append(abs(corr2.loc[nodes[i], nodes[j]]))
    return {'n_communities': len(comms), 'n_communities_ge2': int(sum((1 for c in comms if len(c) >= 2))), 'community_sizes': sizes, 'community_sizes_ge2': [s for s in sizes if s >= 2], 'modularity': float(mod) if np.isfinite(mod) else np.nan, 'mean_intra_abs_r_edges': float(np.mean(intra_edges)) if intra_edges else np.nan, 'mean_inter_abs_r_edges': float(np.mean(inter_edges)) if inter_edges else np.nan, 'mean_intra_abs_r_allpairs': float(np.mean(intra_pairs)) if intra_pairs else np.nan, 'n_intra_edges': len(intra_edges), 'n_inter_edges': len(inter_edges)}

def build_network(data: dict) -> dict:
    feats = data['feats']
    corr2 = pd.DataFrame(data['corr2'].values, index=feats, columns=feats)
    specs = [('abs_r_ge_0.5', 'abs_threshold', None, 0.5), ('top5', 'topk', 5, None), ('top10', 'topk', 10, None)]
    schemes = {}
    rows = []
    for (name, scheme, k, thr) in specs:
        (G, mask) = build_graph(corr2, feats, scheme, k=k, threshold=thr)
        comps = list(nx.connected_components(G))
        comms = list(nx.community.louvain_communities(G, weight='weight', seed=SEED))
        comms = sorted(comms, key=len, reverse=True)
        st = community_stats(G, comms, corr2)
        rows.append({'scheme': name, 'n_nodes': G.number_of_nodes(), 'n_edges': G.number_of_edges(), 'density': float(nx.density(G)), 'n_components': len(comps), 'giant_component': int(max((len(c) for c in comps))) if comps else 0, 'n_isolates': int(sum((1 for c in comps if len(c) == 1))), **st})
        schemes[name] = {'G': G, 'comms': comms, 'stats': st, 'mask': mask}
        print(f"[network] {name}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} communities={st['n_communities']} Q={st['modularity']:.3f} isolates={rows[-1]['n_isolates']}", flush=True)
    main_name = 'top10'
    G = schemes[main_name]['G']
    comms = schemes[main_name]['comms']
    stats = schemes[main_name]['stats']
    comm_of = {}
    for (ci, c) in enumerate(comms, start=1):
        for node in c:
            comm_of[node] = f'C{ci}'
    degree = dict(G.degree())
    strength = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G, normalized=True, weight=None)
    try:
        eigen = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=2000)
    except Exception:
        eigen = nx.eigenvector_centrality(G, weight='weight', max_iter=2000, tol=1e-06)

    def rank_norm(x: np.ndarray) -> np.ndarray:
        return rankdata(x, method='average') / len(x)
    deg_v = np.array([degree[f] for f in feats], float)
    btw_v = np.array([betweenness[f] for f in feats], float)
    eig_v = np.array([eigen[f] for f in feats], float)
    hub_score = (rank_norm(deg_v) + rank_norm(btw_v) + rank_norm(eig_v)) / 3.0
    hub_rank = rankdata(-hub_score, method='average').astype(int)
    nodes = data['fm'][['feature', 'compound_2csv', 'chemical_name', 'origin_paper', 'chemical_class', 'threshold_mean', 'threshold_median', 'unit']].copy()
    nodes['community'] = nodes['feature'].map(comm_of)
    nodes['community_size'] = nodes['community'].map(nodes['community'].value_counts()).astype(int)
    nodes['degree'] = nodes['feature'].map(degree).astype(int)
    nodes['strength'] = nodes['feature'].map(strength).astype(float)
    nodes['betweenness'] = nodes['feature'].map(betweenness).astype(float)
    nodes['eigenvector'] = nodes['feature'].map(eigen).astype(float)
    nodes['hub_score'] = hub_score
    nodes['hub_rank'] = hub_rank
    nodes['is_hub_top10'] = hub_rank <= 10
    nodes['top_degree10'] = rankdata(-deg_v, method='average') <= 10
    nodes['top_betweenness10'] = rankdata(-btw_v, method='average') <= 10
    nodes['top_eigenvector10'] = rankdata(-eig_v, method='average') <= 10
    edges = []
    for (u, v, d) in G.edges(data=True):
        edges.append({'source': u, 'target': v, 'r': d['r'], 'abs_r': abs(d['r']), 'community_source': comm_of[u], 'community_target': comm_of[v], 'same_community': comm_of[u] == comm_of[v]})
    edges_df = pd.DataFrame(edges).sort_values('abs_r', ascending=False)
    comm_rows = []
    for (ci, c) in enumerate(comms, start=1):
        sub = nodes[nodes['feature'].isin(c)]
        orig = sub['origin_paper'].fillna('NA').value_counts()
        cls = sub['chemical_class'].value_counts()
        top_members = sub.sort_values(['hub_score', 'degree'], ascending=False)['feature'].head(10).tolist()
        intra = [abs(corr2.loc[u, v]) for (u, v) in G.edges() if comm_of.get(u) == comm_of.get(v) == f'C{ci}']
        pairs = []
        cc = list(c)
        for i in range(len(cc)):
            for j in range(i + 1, len(cc)):
                pairs.append(abs(corr2.loc[cc[i], cc[j]]))
        comm_rows.append({'community': f'C{ci}', 'size': len(c), 'top_origins': '; '.join((f'{k}:{v}' for (k, v) in orig.head(4).items())), 'top_classes': '; '.join((f'{k}:{v}' for (k, v) in cls.head(4).items())), 'mean_intra_abs_r_edges': float(np.mean(intra)) if intra else np.nan, 'mean_intra_abs_r_allpairs': float(np.mean(pairs)) if pairs else np.nan, 'top_members': '; '.join(top_members)})
    comm_df = pd.DataFrame(comm_rows)
    comm5 = schemes['top5']['comms']
    comm5_of = {}
    for (ci, c) in enumerate(comm5):
        for node in c:
            comm5_of[node] = ci
    labels5 = np.array([comm5_of[f] for f in feats])
    labels10 = np.array([int(comm_of[f][1:]) - 1 for f in feats])
    ari = float(adjusted_rand_score(labels5, labels10))
    return {'main_scheme': main_name, 'G': G, 'comms': comms, 'stats': stats, 'scheme_table': pd.DataFrame(rows), 'nodes': nodes, 'edges': edges_df, 'communities': comm_df, 'comm_of': comm_of, 'ari_top5_top10': ari, 'W': _row_normalised_adjacency(G, feats)}

def _row_normalised_adjacency(G: nx.Graph, feats: list[str]) -> np.ndarray:
    n = len(feats)
    idx = {f: i for (i, f) in enumerate(feats)}
    W = np.zeros((n, n), dtype=float)
    for u in G.nodes():
        nbrs = list(G.neighbors(u))
        if nbrs:
            w = 1.0 / len(nbrs)
            for v in nbrs:
                W[idx[u], idx[v]] = w
    return W

def spearman_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed Spearman r, shape (n_features, n_targets), vectorised by ranks."""
    Xr = rankdata(X, axis=0, method='average')
    Yr = rankdata(y, axis=0, method='average')
    Xc = Xr - Xr.mean(axis=0)
    Yc = Yr - Yr.mean(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0)[:, None] * (Yc ** 2).sum(axis=0)[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        S = Xc.T @ Yc / den
    return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

def spearman_pvalues(r: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        t = r * np.sqrt((n - 2) / np.maximum(1e-15, 1 - r ** 2))
    p = 2 * student_t.sf(np.abs(t), df=n - 2)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)

def fdr_by_target(p: np.ndarray) -> np.ndarray:
    q = np.ones_like(p)
    for j in range(p.shape[1]):
        q[:, j] = multipletests(p[:, j], method='fdr_bh')[1]
    return q

def rank_features(score: np.ndarray, by_abs: bool=True) -> np.ndarray:
    s = np.abs(score) if by_abs else score
    out = np.empty_like(s, dtype=float)
    for j in range(s.shape[1]):
        out[:, j] = rankdata(-s[:, j], method='average')
    return out

def top3_mask(ranks: np.ndarray) -> np.ndarray:
    idx = np.argsort(ranks, axis=0, kind='stable')[:3, :]
    mask = np.zeros(ranks.shape, dtype=np.uint8)
    np.put_along_axis(mask, idx, 1, axis=0)
    return mask

def topk_indices(ranks: np.ndarray, k: int=3) -> np.ndarray:
    return np.argsort(ranks, axis=0, kind='stable')[:k, :]

def smooth_scores(S: np.ndarray, W: np.ndarray, lam: float=NET_LAMBDA) -> np.ndarray:
    return (1.0 - lam) * S + lam * (W @ S)

def _elasticnet_target(j, X, yv, alphas, l1s, groups=None):
    pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()), ('model', ElasticNet(max_iter=10000, tol=0.001, random_state=SEED))])
    cv = list(GroupKFold(3).split(X, groups=groups)) if groups is not None else list(GroupKFold(3).split(X, groups=np.arange(len(X))))
    search = GridSearchCV(pipe, {'model__alpha': alphas, 'model__l1_ratio': l1s}, cv=cv, scoring='neg_mean_squared_error', n_jobs=1)
    search.fit(X, yv[:, j])
    m = search.best_estimator_.named_steps['model']
    return (j, m.coef_.copy(), float(m.alpha), float(m.l1_ratio), float(r2_score(yv[:, j], search.predict(X))))

def elastic_net_scores(X, y, n_jobs):
    n_t = y.shape[1]
    alphas = [0.03, 0.1, 0.3]
    res = Parallel(n_jobs=n_jobs, batch_size=1)((delayed(_elasticnet_target)(j, X, y, alphas, [0.3, 0.7]) for j in range(n_t)))
    coef = np.zeros((X.shape[1], n_t))
    alpha = np.zeros(n_t)
    l1 = np.zeros(n_t)
    r2 = np.zeros(n_t)
    for (j, c, a, l, r) in res:
        coef[:, j] = c
        alpha[j] = a
        l1[j] = l
        r2[j] = r
    return (coef, alpha, l1, r2)

def _et_target(j, X, y, n_estimators, seed):
    model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=8, max_features='sqrt', min_samples_leaf=3, n_jobs=1, random_state=seed)
    model.fit(X, y[:, j])
    return (j, model.feature_importances_.copy())

def extra_trees_scores(X, y, n_estimators, n_jobs, seed=SEED):
    res = Parallel(n_jobs=n_jobs, batch_size=1)((delayed(_et_target)(j, X, y, n_estimators, seed + j) for j in range(y.shape[1])))
    imp = np.zeros((X.shape[1], y.shape[1]))
    for (j, v) in res:
        imp[:, j] = v
    return imp

def _lgb_oof_target(j, X, y, n_splits=5, n_estimators=LGB_FULL, seed=SEED):
    """Descriptive in-sample attribution, identically used in point and bootstrap fits."""
    model = lgb.LGBMRegressor(n_estimators=n_estimators, num_leaves=15, learning_rate=0.05, feature_fraction=0.7, subsample=0.8, subsample_freq=1, min_child_samples=10, verbose=-1, n_jobs=1, random_state=seed + j, force_col_wise=True)
    model.fit(X, y[:, j])
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    return (j, np.abs(contrib).mean(axis=0))

def lightgbm_shap_scores(X, y, n_jobs, n_estimators=LGB_FULL):
    res = Parallel(n_jobs=n_jobs, batch_size=1)((delayed(_lgb_oof_target)(j, X, y, 5, n_estimators, SEED) for j in range(y.shape[1])))
    shap = np.zeros((X.shape[1], y.shape[1]))
    for (j, v) in res:
        shap[:, j] = v
    return shap

def run_full_methods(data: dict, net: dict) -> dict:
    (Xs, y) = (data['Xs'], data['y_arr'])
    t0 = time.time()
    S = spearman_scores(Xs, y)
    p = spearman_pvalues(S, Xs.shape[0])
    q = fdr_by_target(p)
    print(f'[method A] Spearman matrix {S.shape}, {(q < 0.05).sum()} target-feature pairs q<0.05, {time.time() - t0:.1f}s', flush=True)
    t0 = time.time()
    (B, alpha, l1, r2) = elastic_net_scores(data['X'].values, y, N_JOBS)
    print(f'[method B] ElasticNet pipeline CV done, descriptive in-sample R2={np.mean(r2):.3f}, alpha median={np.median(alpha):.3g}, {time.time() - t0:.1f}s', flush=True)
    t0 = time.time()
    C = extra_trees_scores(Xs, y, N_TREES_FULL, N_JOBS, SEED)
    print(f'[method C] ExtraTrees {N_TREES_FULL} trees done, {time.time() - t0:.1f}s', flush=True)
    t0 = time.time()
    D = lightgbm_shap_scores(Xs, y, N_JOBS, LGB_FULL)
    print(f'[method D] LightGBM descriptive SHAP {LGB_FULL} trees done, {time.time() - t0:.1f}s', flush=True)
    E = smooth_scores(S, net['W'], NET_LAMBDA)
    print(f'[method E] network smoothing lambda={NET_LAMBDA} done', flush=True)
    scores = {'A': S, 'B': B, 'C': C, 'D': D, 'E': E}
    ranks = {'A': rank_features(S, True), 'B': rank_features(B, True), 'C': rank_features(C, False), 'D': rank_features(D, False), 'E': rank_features(E, True)}
    fused_rank = np.mean(np.stack([ranks[m] for m in METHODS]), axis=0)
    (n_features, n_targets) = fused_rank.shape
    fused_score = 1.0 - (fused_rank - 1.0) / (n_features - 1.0)
    sign_A = np.sign(S)
    z = {}
    for m in METHODS:
        arr = scores[m].copy()
        if m in ('C', 'D'):
            arr = arr * sign_A
        scale = np.std(arr, axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        z[m] = arr / scale
    consensus_signed = np.mean(np.stack([z[m] for m in METHODS]), axis=0)
    return {'scores': scores, 'ranks': ranks, 'q': q, 'p': p, 'alpha': alpha, 'l1': l1, 'r2': r2, 'fused_rank': fused_rank, 'fused_score': fused_score, 'consensus_signed': consensus_signed}
_BOOT = {}

def _bootstrap_one(b):
    rng = np.random.default_rng(SEED + 100000 + b)
    idx = rng.integers(0, len(_BOOT['X_raw']), len(_BOOT['X_raw']))
    raw = _BOOT['X_raw'][idx]
    yb = _BOOT['y'][idx]
    Xb = StandardScaler().fit_transform(SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(raw))
    (nf, nt) = (_BOOT['n_feat'], _BOOT['n_targets'])
    Sb = spearman_scores(Xb, yb)
    Bb = np.zeros((nf, nt))
    Cb = np.zeros((nf, nt))
    Db = np.zeros((nf, nt))
    for j in range(nt):
        (_, coef, _, _, _) = _elasticnet_target(j, raw, yb, [0.03, 0.1, 0.3], [0.3, 0.7], groups=idx)
        Bb[:, j] = coef
        Cb[:, j] = _et_target(j, Xb, yb, N_TREES_FULL, SEED + j)[1]
        Db[:, j] = _lgb_oof_target(j, Xb, yb, n_estimators=LGB_FULL)[1]
    Eb = smooth_scores(Sb, _BOOT['W'], NET_LAMBDA)
    ranks = [rank_features(Sb, True), rank_features(Bb, True), rank_features(Cb, False), rank_features(Db, False), rank_features(Eb, True)]
    fused = np.mean(ranks, axis=0)
    score = 1 - (fused - 1) / (nf - 1)
    masks = [top3_mask(r) for r in ranks] + [top3_mask(fused)]
    for lam in [0.0, 0.1, 0.2, 0.5]:
        masks.append(top3_mask(rank_features(score + lam * _BOOT['prior'], False)))
    masks.append(top3_mask(np.mean(ranks[:4], axis=0)))
    only = fused.copy()
    only[~_BOOT['chemical_mask']] = np.inf
    masks.append(top3_mask(only))
    return (b, np.stack(masks).transpose(0, 2, 1).astype(np.uint8))

def run_bootstrap(data, net, methods, n_boot):
    import hashlib
    (nf, nt) = methods['fused_rank'].shape
    (prior, _) = build_domain_prior(data)
    pm = np.zeros((nf, nt))
    for (j, t) in enumerate(data['y'].columns):
        for (f, w) in prior[t].items():
            pm[data['feats'].index(f), j] = w
    chemical = ~data['fm'].chemical_class.isin(['Physicochemical', 'Derived index']).values
    _BOOT.clear()
    _BOOT.update(X_raw=data['X'].values, y=data['y_arr'], W=net['W'], n_feat=nf, n_targets=nt, prior=pm, chemical_mask=chemical)
    labels = ['A', 'B', 'C', 'D', 'E', 'FUSED'] + [f'DOMAIN_{x}' for x in [0.0, 0.1, 0.2, 0.5]] + ['FUSED_NO_E', 'COMPOUND_ONLY']
    counts = np.zeros((len(labels), nt, nf), dtype=np.int32)
    done = set()
    path = OUT / '自助重采样检查点.npz'
    fingerprint = hashlib.sha256(json.dumps({'provenance': provenance(), 'n_boot': n_boot, 'trees': N_TREES_FULL, 'shap_trees': LGB_FULL}, sort_keys=True).encode()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if str(old['fingerprint']) == fingerprint and np.all(old['done'] < n_boot):
                counts = old['counts']
                done = set(old['done'].tolist())
    print(f'[bootstrap] same estimators and tuning in every replicate; resume {len(done)}/{n_boot}', flush=True)
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=N_JOBS) as pool:
        for (b, m) in pool.imap_unordered(_bootstrap_one, [i for i in range(n_boot) if i not in done], chunksize=1):
            counts += m
            done.add(b)
            if len(done) % 10 == 0 or len(done) == n_boot:
                np.savez_compressed(path, counts=counts, done=np.array(sorted(done)), fingerprint=fingerprint)
                print(f'[bootstrap] {len(done)}/{n_boot}; elapsed {time.time() - t0:.1f}s', flush=True)
    return ({lab: counts[i] / n_boot for (i, lab) in enumerate(labels)}, time.time() - t0)

def jaccard_tables(methods: dict, n_targets: int) -> tuple[pd.DataFrame, np.ndarray]:
    tops = {m: topk_indices(methods['ranks'][m], 3) for m in METHODS}
    rows = []
    for j in range(n_targets):
        sets = {m: set(tops[m][:, j].tolist()) for m in METHODS}
        for a in range(len(METHODS)):
            for b in range(a + 1, len(METHODS)):
                (ma, mb) = (METHODS[a], METHODS[b])
                jac = len(sets[ma] & sets[mb]) / len(sets[ma] | sets[mb])
                rows.append({'target': j, 'method_i': ma, 'method_j': mb, 'jaccard': jac})
    long = pd.DataFrame(rows)
    mat = np.eye(len(METHODS))
    for a in range(len(METHODS)):
        for b in range(a + 1, len(METHODS)):
            v = long[(long.method_i == METHODS[a]) & (long.method_j == METHODS[b])]['jaccard'].mean()
            mat[a, b] = mat[b, a] = v
    return (long, mat)

def enrichment_table(data: dict, methods: dict) -> pd.DataFrame:
    fm = data['fm'].set_index('feature')
    classes = sorted(fm['chemical_class'].unique())
    universe = len(fm)
    rows = []
    for (j, target) in enumerate(data['y'].columns):
        order = np.argsort(methods['fused_rank'][:, j], kind='stable')
        for k in (3, 10):
            sel = order[:k]
            sel_classes = fm.iloc[sel]['chemical_class'].value_counts()
            for cls in classes:
                m = int((fm['chemical_class'] == cls).sum())
                obs = int(sel_classes.get(cls, 0))
                exp = k * m / universe
                p = float(hypergeom.sf(obs - 1, universe, m, k)) if obs > 0 else 1.0
                rows.append({'target': target, 'k': k, 'chemical_class': cls, 'class_size': m, 'observed': obs, 'expected': exp, 'fold_enrichment': obs / exp if exp > 0 else np.nan, 'p': p})
    df = pd.DataFrame(rows)
    df['q'] = np.nan
    for ((target, k), idx) in df.groupby(['target', 'k']).groups.items():
        sub = df.loc[idx]
        df.loc[idx, 'q'] = multipletests(sub['p'].values, method='fdr_bh')[1]
    return df

def _method_signed_scores(methods: dict) -> dict:
    sign_A = np.sign(methods['scores']['A'])
    return {'A': methods['scores']['A'], 'B': methods['scores']['B'], 'C': methods['scores']['C'] * sign_A, 'D': methods['scores']['D'] * sign_A, 'E': methods['scores']['E'], 'FUSED': methods['consensus_signed']}

def benchmark_6csv(data: dict, maps: dict, methods: dict) -> pd.DataFrame:
    c6 = data['c6']
    feats = data['feats']
    match_idx = maps['match_idx']
    signed = _method_signed_scores(methods)
    final_tops = {m: topk_indices(methods['ranks'][m] if m != 'FUSED' else methods['fused_rank'], 3) for m in list(METHODS) + ['FUSED']}
    rows = []
    for (j, target) in enumerate(data['y'].columns):
        col = maps['target_to_c6'][target]
        r6 = c6[col].values.astype(float)
        top3_r6 = set(np.argsort(-np.abs(r6))[:3].tolist())
        for m in list(METHODS) + ['FUSED']:
            sc = signed[m][match_idx, j]
            corr = spearmanr(sc, r6).statistic
            if not np.isfinite(corr):
                corr = 0.0
            top3_sc = set(np.argsort(-np.abs(sc))[:3].tolist())
            overlap_restricted = len(top3_sc & top3_r6) / 3.0
            sign_restricted = np.mean([np.sign(sc[i]) == np.sign(r6[i]) for i in top3_sc if r6[i] != 0]) if any((r6[i] != 0 for i in top3_sc)) else np.nan
            final = final_tops[m][:, j]
            match_position = {int(idx): pos for (pos, idx) in enumerate(match_idx)}
            final_match = [int(f) for f in final if int(f) in match_position]
            final_c6_pos = [match_position[f] for f in final_match]
            overlap_final = len(set(final_c6_pos) & top3_r6) / 3.0
            sign_final = np.mean([np.sign(signed[m][f, j]) == np.sign(r6[pos]) for (f, pos) in zip(final_match, final_c6_pos) if r6[pos] != 0]) if final_match and any((r6[pos] != 0 for pos in final_c6_pos)) else np.nan
            top10_sc = set(np.argsort(-np.abs(sc))[:10].tolist())
            top10_r6 = set(np.argsort(-np.abs(r6))[:10].tolist())
            overlap10 = len(top10_sc & top10_r6) / 10.0
            rows.append({'target': target, 'method': m, 'n_match': len(match_idx), 'rank_corr_spearman': float(corr), 'top3_overlap_restricted': overlap_restricted, 'top10_overlap_restricted': overlap10, 'top3_overlap_final': overlap_final, 'sign_agreement_restricted': sign_restricted, 'sign_agreement_final': sign_final, 'n_matchable_in_final_top3': len(final_match)})
    return pd.DataFrame(rows)

def make_tables(data: dict, net: dict, methods: dict, jac_long: pd.DataFrame, jac_mat: np.ndarray, bench: pd.DataFrame, enrich: pd.DataFrame, boot_freq: dict, n_boot: int) -> dict:
    feats = data['feats']
    fm = data['fm'].set_index('feature')
    targets = list(data['y'].columns)
    tm = data['target_meta'].set_index('target')
    nodes = net['nodes'].set_index('feature')
    tops = topk_indices(methods['fused_rank'], 3)
    ranks = methods['ranks']
    top_rows = []
    stab_rows = []
    for (j, target) in enumerate(targets):
        order = tops[:, j]
        for (rank, fi) in enumerate(order, start=1):
            f = feats[int(fi)]
            row = {'target': target, 'target_label': normalise_name(tm.loc[target, 'CleanName_Paper']), 'category': tm.loc[target, 'Category'], 'rank': rank, 'feature': f, 'compound_2csv': fm.loc[f, 'compound_2csv'], 'chemical_name': fm.loc[f, 'chemical_name'], 'chemical_class': fm.loc[f, 'chemical_class'], 'origin': fm.loc[f, 'origin_paper'], 'score': float(methods['fused_score'][fi, j]), 'mean_rank': float(methods['fused_rank'][fi, j]), 'signed_score': float(methods['consensus_signed'][fi, j]), 'effect_sign': int(np.sign(methods['consensus_signed'][fi, j])), 'stability_freq': float(boot_freq['FUSED'][j, fi]), 'freq_A': float(boot_freq['A'][j, fi]), 'freq_B': float(boot_freq['B'][j, fi]), 'freq_C': float(boot_freq['C'][j, fi]), 'freq_D': float(boot_freq['D'][j, fi]), 'freq_E': float(boot_freq['E'][j, fi]), 'rank_A': float(ranks['A'][fi, j]), 'rank_B': float(ranks['B'][fi, j]), 'rank_C': float(ranks['C'][fi, j]), 'rank_D': float(ranks['D'][fi, j]), 'rank_E': float(ranks['E'][fi, j]), 'spearman_r': float(methods['scores']['A'][fi, j]), 'fdr_q': float(methods['q'][fi, j]), 'en_alpha': float(methods['alpha'][j]), 'en_l1_ratio': float(methods['l1'][j]), 'community': nodes.loc[f, 'community']}
            row['methods'] = ';'.join((f"{m}{row['rank_' + m]:g}" for m in METHODS))
            top_rows.append(row)
            stab_rows.append({'target': target, 'rank': rank, 'feature': f, 'compound_2csv': fm.loc[f, 'compound_2csv'], 'stability_freq': row['stability_freq'], 'freq_A': row['freq_A'], 'freq_B': row['freq_B'], 'freq_C': row['freq_C'], 'freq_D': row['freq_D'], 'freq_E': row['freq_E'], 'n_boot': n_boot})
    top3 = pd.DataFrame(top_rows)
    stability = pd.DataFrame(stab_rows)
    audit = []
    for (j, target) in enumerate(targets):
        for (fi, f) in enumerate(feats):
            audit.append({'target': target, 'feature': f, 'compound_2csv': fm.loc[f, 'compound_2csv'], 'A_spearman': float(methods['scores']['A'][fi, j]), 'A_q': float(methods['q'][fi, j]), 'A_rank': float(ranks['A'][fi, j]), 'B_coef': float(methods['scores']['B'][fi, j]), 'B_rank': float(ranks['B'][fi, j]), 'C_importance': float(methods['scores']['C'][fi, j]), 'C_rank': float(ranks['C'][fi, j]), 'D_mean_abs_shap': float(methods['scores']['D'][fi, j]), 'D_rank': float(ranks['D'][fi, j]), 'E_smoothed': float(methods['scores']['E'][fi, j]), 'E_rank': float(ranks['E'][fi, j]), 'fused_rank': float(methods['fused_rank'][fi, j]), 'fused_score': float(methods['fused_score'][fi, j]), 'consensus_signed': float(methods['consensus_signed'][fi, j])})
    audit = pd.DataFrame(audit)
    bf_rows = []
    for (j, target) in enumerate(targets):
        for (fi, f) in enumerate(feats):
            bf_rows.append({'target': target, 'feature': f, 'compound_2csv': fm.loc[f, 'compound_2csv'], 'freq_fused': float(boot_freq['FUSED'][j, fi]), 'freq_A': float(boot_freq['A'][j, fi]), 'freq_B': float(boot_freq['B'][j, fi]), 'freq_C': float(boot_freq['C'][j, fi]), 'freq_D': float(boot_freq['D'][j, fi]), 'freq_E': float(boot_freq['E'][j, fi])})
    boot_df = pd.DataFrame(bf_rows)
    jac_rows = []
    for (j, target) in enumerate(targets):
        sub = jac_long[jac_long.target == j]
        for (_, r) in sub.iterrows():
            jac_rows.append({'target': target, 'method_i': r.method_i, 'method_j': r.method_j, 'jaccard': r.jaccard})
    for a in range(len(METHODS)):
        for b in range(a + 1, len(METHODS)):
            jac_rows.append({'target': 'MEAN', 'method_i': METHODS[a], 'method_j': METHODS[b], 'jaccard': jac_mat[a, b]})
    jac_df = pd.DataFrame(jac_rows)
    return {'top3': top3, 'stability': stability, 'audit': audit, 'boot_freq': boot_df, 'jaccard': jac_df}

def case_table(data: dict, maps: dict, methods: dict, boot_freq: dict, cases: list[str]) -> pd.DataFrame:
    feats = data['feats']
    fm = data['fm'].set_index('feature')
    rows = []
    for target in cases:
        j = data['y'].columns.get_loc(target)
        order = np.argsort(methods['fused_rank'][:, j], kind='stable')[:5]
        c6col = maps['target_to_c6'][target]
        for (rank, fi) in enumerate(order, start=1):
            f = feats[int(fi)]
            c6row = maps['feat_to_c6row'].get(f)
            c6r = float(data['c6'].loc[c6row, c6col]) if c6row else np.nan
            rows.append({'target': target, 'rank': rank, 'feature': f, 'compound_2csv': fm.loc[f, 'compound_2csv'], 'chemical_name': fm.loc[f, 'chemical_name'], 'chemical_class': fm.loc[f, 'chemical_class'], 'origin': fm.loc[f, 'origin_paper'], 'fused_score': float(methods['fused_score'][fi, j]), 'signed_score': float(methods['consensus_signed'][fi, j]), 'freq_fused': float(boot_freq['FUSED'][j, fi]), 'rank_A': float(methods['ranks']['A'][fi, j]), 'rank_B': float(methods['ranks']['B'][fi, j]), 'rank_C': float(methods['ranks']['C'][fi, j]), 'rank_D': float(methods['ranks']['D'][fi, j]), 'rank_E': float(methods['ranks']['E'][fi, j]), 'corr6_r': c6r, 'in_final_top3': rank <= 3})
    return pd.DataFrame(rows)

def build_domain_prior(data: dict) -> tuple[dict, list[str]]:
    """Target-specific candidate sets from 项目进度/化学合理性文献.md §4.

    Weights encode prior strength, not causality:
      1.0 eponymous/direct compound; 0.7 specific §4 marker;
      0.4 broad chemical family; 0.3 physicochemical / style proxy.
    """
    fm = data['fm']
    targets = list(data['y'].columns)
    by_name = set(fm['feature'])
    prior = {t: {} for t in targets}
    missing = []

    def put(t, names, w):
        for n in names:
            if n in by_name:
                prior[t][n] = max(prior[t].get(n, 0.0), w)
            else:
                missing.append(n)

    def put_class(t, classes, w):
        for cls in classes:
            for f in fm.loc[fm['chemical_class'] == cls, 'feature']:
                prior[t][f] = max(prior[t].get(f, 0.0), w)

    def put_origin(t, origins, w):
        for origin in origins:
            for f in fm.loc[fm['origin_paper'] == origin, 'feature']:
                prior[t][f] = max(prior[t].get(f, 0.0), w)
    malt_markers = ['furfural', 'X3.methylbutanal', 'methylpyrazine', 'X.E..beta.damascenone', 'X2.acetylfuran', 'X5.methylfurfural', 'isomaltol', 'X2.acetylpyrrole', 'X3.methylthio.propanol', 'gamma.nonalactone']
    phys_proxies = ['color.EBC.', 'kcalperc', 'pH', 'protein.g.L.', 'GFS', 'unfermentables']
    hops_markers = ['bitterness', 'linalool', 'geraniol', 'citronellol', 'alpha.humulene', 'E.caryophyllene', 'aroma_hops_sum']
    terpene_markers = ['linalool', 'geraniol', 'citronellol', 'limonene', 'X1.alpha.terpineol', 'nerol', 'terpinen.4.ol', 'X.E..beta.ocimene', 'humulene.epoxide.II', 'caryophyllene.oxide', 'E.caryophyllene', 'alpha.humulene', 'Z.nerolidol', 'E.nerolidol']
    ester_markers = ['esters_sum', 'ethyl_acetate', 'isopentyl_acetate', 'ethyl_hexanoate', 'ethyl_octanoate', 'phenethyl_acetate', 'ethyl.butyrate', 'ethyl_decanoate', 'ethyl_2.methyl_butyrate', 'ethyl_isovalerate', 'isobutyl_acetate', 'octyl_acetate', 'ethyl_propionate', 'ethyl.pentanoate']
    alcohol_markers = ['isoamyl_alcohol', 'isobutanol', 'X1.propanol', 'phenethyl_alcohol', 'n.hexanol', 'glycerol']
    base = {'malt_all': [('names', malt_markers, 0.7), ('names', phys_proxies, 0.3), ('class', ['Furans/pyrazines'], 0.4), ('origin', ['Malt'], 0.4)], 'malt_grain': [('names', ['X3.methylbutanal', 'X2.methyltetrahydrofuran.3.one', 'methylpyrazine', 'furfural', 'GFS', 'X2.6.dimethyl.pyrazine'], 0.7), ('class', ['Furans/pyrazines'], 0.4), ('origin', ['Malt'], 0.4)], 'malt_bread': [('names', ['furfural', 'X2.acetylfuran', 'X2.acetylpyrrole', 'X3.methylthio.propanol', 'isomaltol', 'X5.methylfurfural'], 0.7), ('class', ['Furans/pyrazines'], 0.4), ('origin', ['Malt'], 0.4)], 'malt_cara': [('names', ['furfural', 'X5.methylfurfural', 'isomaltol', 'gamma.nonalactone', 'X.E..beta.damascenone', 'color.EBC.', 'kcalperc'], 0.7), ('class', ['Furans/pyrazines'], 0.4), ('origin', ['Malt'], 0.4)], 'malt_burn': [('names', ['X4.ethyl.guaiacol', 'eugenol', 'X4.ethyl.phenol', 'vanillin', 'methylpyrazine', 'X2.acetylpyrrole'], 0.7), ('class', ['Volatile phenols'], 0.4), ('origin', ['Malt'], 0.4)], 'hops_all': [('names', hops_markers, 0.7), ('class', ['Terpenoids', 'Terpene hydrocarbons'], 0.4), ('origin', ['Hops'], 0.4)], 'hops_citrus': [('names', terpene_markers, 0.7), ('class', ['Terpenoids', 'Terpene hydrocarbons'], 0.4), ('origin', ['Hops'], 0.4)], 'hops_tropical': [('names', ['ethyl_hexanoate', 'ethyl_octanoate', 'isopentyl_acetate', 'ethyl.butyrate', 'linalool', 'geraniol', 'citronellol', 'aroma_hops_sum'], 0.7), ('class', ['Esters', 'Terpenoids'], 0.4), ('origin', ['Hops'], 0.4)], 'hops_noble': [('names', ['X3Z.hexenol', 'alpha.humulene', 'E.caryophyllene', 'humulene.epoxide.II', 'caryophyllene.oxide', 'caryophyllene.alcohol'], 0.7), ('class', ['Terpenoids', 'Terpene hydrocarbons'], 0.4), ('origin', ['Hops'], 0.4)], 'hops_woody': [('names', ['E.caryophyllene', 'alpha.humulene', 'caryophyllene.oxide', 'humulene.epoxide.II', 'caryophyllene.alcohol', 'alpha.muurolol', 'X1.10.di.epi.cubenol', 'tau.muurolol'], 0.7), ('class', ['Terpenoids', 'Terpene hydrocarbons'], 0.4), ('origin', ['Hops'], 0.4)], 'esters_all': [('names', ester_markers + alcohol_markers, 0.7), ('class', ['Esters', 'Alcohols'], 0.4)], 'esters_ethac': [('names', ['ethyl_acetate', 'isopentyl_acetate', 'ethyl.lactate', 'ethyl_propionate', 'butyl.acetate', 'amyl.acetate'], 0.7), ('class', ['Esters'], 0.4)], 'esters_isoaa': [('names', ['isopentyl_acetate', 'isobutyl_acetate', 'isoamyl_alcohol', 'ethyl.butyrate', 'amyl.acetate', 'isopentyl.butanoate'], 0.7), ('class', ['Esters', 'Alcohols'], 0.4)], 'esters_flower': [('names', ['phenethyl_acetate', 'phenethyl_alcohol', 'linalool', 'geraniol', 'citronellol', 'X1.alpha.terpineol', 'benzyl.alcohol', 'nerol'], 0.7), ('class', ['Esters', 'Terpenoids'], 0.4)], 'esters_fruity': [('names', ester_markers, 0.7), ('class', ['Esters'], 0.4)]}
    for t in targets:
        if t.startswith(('A_', 'F_')):
            b = t[2:]
            for (kind, value, w) in base.get(b, []):
                if kind == 'names':
                    put(t, value, w)
                elif kind == 'class':
                    put_class(t, value, w)
                elif kind == 'origin':
                    put_origin(t, value, w)
    sulfur_markers = ['DMS', 'sulfur_sum', 'SO2.mg.L.', 'H2S', 'MeSH', 'DES', 'CS2']
    phenol_markers = ['X4VG', 'X4.ethyl.guaiacol', 'X4.ethyl.phenol', 'eugenol', 'vanillin']
    aldehyde_markers = ['X.E..2.nonenal', 'n.nonanal', 'X.E.E..2.4.decadienal', 'furfural']
    specific = {'acidity': [('names', ['lactic_acid.mg.L.', 'acetic_acid.g.L.', 'acids_sum', 'pH'], 0.7), ('class', ['Organic acids'], 0.4)], 'bitternes': [('names', ['bitterness'], 1.0), ('class', ['Hop acids'], 0.4)], 'sweetness': [('names', ['GFS', 'glycerol', 'ethanol..v.v.', 'unfermentables', 'protein.g.L.'], 0.7), ('class', ['Polyols'], 0.4), ('class', ['Physicochemical'], 0.3)], 'X4vg': [('names', ['X4VG'], 1.0), ('names', ['X4.ethyl.guaiacol', 'X4.ethyl.phenol', 'eugenol', 'vanillin'], 0.7), ('class', ['Volatile phenols'], 0.4)], 'diacetyl': [('names', ['X2.3.butanedione'], 1.0), ('class', ['Ketones'], 0.4)], 'dms': [('names', ['DMS'], 1.0), ('names', sulfur_markers, 0.7), ('class', ['Sulfur compounds'], 0.4)], 'metallic': [('names', ['iron', 'X.E..2.nonenal', 'n.nonanal', 'X.E.E..2.4.decadienal'], 0.7), ('class', ['Aldehydes'], 0.4)], 'stale_hops': [('names', ['X.E..2.nonenal', 'DMS', 'X4.ethyl.phenol', 'X4.ethyl.guaiacol', 'ethyl_2.methyl_butyrate', 'ethyl_isovalerate'], 0.7), ('class', ['Sulfur compounds', 'Volatile phenols', 'Aldehydes'], 0.4)], 't2n': [('names', ['X.E..2.nonenal'], 1.0), ('names', ['n.nonanal', 'X.E.E..2.4.decadienal', 'DMS', 'furfural'], 0.7), ('class', ['Aldehydes'], 0.4)], 'orange': [('names', ['limonene', 'linalool', 'X1.alpha.terpineol', 'X.E..beta.ocimene', 'citronellol'], 0.7), ('class', ['Terpenoids', 'Terpene hydrocarbons'], 0.4)], 'coriander': [('names', ['linalool', 'geraniol', 'citronellol', 'X1.alpha.terpineol', 'terpinen.4.ol'], 0.7), ('class', ['Terpenoids'], 0.4)], 'clove': [('names', ['X4VG', 'eugenol', 'X4.ethyl.guaiacol', 'X4.ethyl.phenol', 'vanillin'], 0.7), ('class', ['Volatile phenols'], 0.4)], 'lactic': [('names', ['lactic_acid.mg.L.'], 1.0), ('names', ['acids_sum', 'pH', 'acetic_acid.g.L.'], 0.7), ('class', ['Organic acids'], 0.4)], 'acetic': [('names', ['acetic_acid.g.L.'], 1.0), ('names', ['acids_sum', 'pH', 'lactic_acid.mg.L.'], 0.7), ('class', ['Organic acids'], 0.4)], 'barnyard': [('names', ['X4.ethyl.phenol', 'X4.ethyl.guaiacol', 'X4VG', 'geosmin', 'ethyl_2.methyl_butyrate', 'ethyl_isovalerate'], 0.7), ('class', ['Volatile phenols', 'Sulfur compounds'], 0.4)], 'alcohol': [('names', ['ethanol..v.v.'], 1.0), ('names', ['isoamyl_alcohol', 'isobutanol', 'X1.propanol', 'glycerol'], 0.7), ('class', ['Alcohols'], 0.4)], 'aftertaste': [('names', ['bitterness', 'ethanol..v.v.', 'X.E..2.nonenal', 'esters_sum', 'acids_sum'], 0.7)], 'body': [('names', ['glycerol', 'ethanol..v.v.', 'protein.g.L.', 'unfermentables', 'betaglucan.mg.L.', 'kcalperc'], 0.7), ('class', ['Physicochemical'], 0.3)], 'co2': [('names', ['CO2.PSI.'], 1.0)], 'overall': [('names', ['esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum', 'bitterness', 'ethanol..v.v.', 'glycerol'], 0.7), ('class', ['Esters', 'Alcohols'], 0.3)]}
    for (t, rules) in specific.items():
        for (kind, value, w) in rules:
            if kind == 'names':
                put(t, value, w)
            elif kind == 'class':
                put_class(t, value, w)
            elif kind == 'origin':
                put_origin(t, value, w)
    missing = sorted(set(missing))
    if missing:
        print(f'[domain] WARNING: {len(missing)} literature names not in 1.csv: {missing}', flush=True)
    return (prior, missing)

def domain_tables(data: dict, maps: dict, methods: dict, boot_freq: dict, prior: dict, lambdas=(0.0, 0.1, 0.2, 0.5)) -> dict:
    feats = data['feats']
    targets = list(data['y'].columns)
    fm = data['fm'].set_index('feature')
    target_to_c6 = maps['target_to_c6']
    feat_to_c6row = maps['feat_to_c6row']
    n_feat = len(feats)
    prior_mat = np.zeros((n_feat, len(targets)))
    for (j, t) in enumerate(targets):
        for (f, w) in prior[t].items():
            prior_mat[feats.index(f), j] = w
    data_rank = methods['fused_rank']
    data_score = methods['fused_score']
    data_top = topk_indices(data_rank, 3)
    sensitivity = []
    lam_tables = {}
    for lam in lambdas:
        score = data_score + lam * prior_mat
        rank = rank_features(score, by_abs=False)
        tops = topk_indices(rank, 3)
        lam_tables[lam] = (score, rank, tops)
        for (j, t) in enumerate(targets):
            sel = tops[:, j]
            data_sel = set(data_top[:, j].tolist())
            overlap_data = len(set(sel.tolist()) & data_sel) / 3.0
            stab = float(np.mean([boot_freq[f'DOMAIN_{lam}'][j, f] for f in sel]))
            mean_data_rank = float(np.mean([data_rank[f, j] for f in sel]))
            n_direct = int(sum((prior_mat[f, j] >= 1.0 for f in sel)))
            c6col = target_to_c6[t]
            r6 = data['c6'][c6col].values.astype(float)
            match_position = {feats.index(f): pos for (pos, f) in enumerate(maps['match_feats'])}
            final_match = [(f, match_position[f]) for f in sel if f in match_position]
            c6_top = set(np.argsort(-np.abs(r6))[:3].tolist())
            overlap_final = len({pos for (_, pos) in final_match} & c6_top) / 3.0
            signs = [np.sign(methods['consensus_signed'][f, j]) == np.sign(r6[pos]) for (f, pos) in final_match if r6[pos] != 0]
            sign_final = float(np.mean(signs)) if signs else np.nan
            sc44 = score[maps['match_idx'], j]
            top44 = set(np.argsort(-np.abs(sc44))[:3].tolist())
            overlap_res = len(top44 & c6_top) / 3.0
            signed44 = methods['consensus_signed'][maps['match_idx'], j]
            signed44 = signed44 + lam * prior_mat[maps['match_idx'], j] * np.sign(signed44)
            top44s = set(np.argsort(-np.abs(signed44))[:3].tolist())
            signs_res = [np.sign(signed44[i]) == np.sign(r6[i]) for i in top44s if r6[i] != 0]
            sensitivity.append({'target': t, 'lambda': lam, 'rank1_feature': feats[int(sel[0])], 'rank2_feature': feats[int(sel[1])], 'rank3_feature': feats[int(sel[2])], 'overlap_with_data_top3': overlap_data, 'mean_stability_fused': stab, 'frequency_scope': 'own_prior_augmented_selector', 'mean_data_fused_rank': mean_data_rank, 'n_direct_prior_in_top3': n_direct, 'top3_overlap_6csv_final': overlap_final, 'sign_agreement_6csv_final': sign_final, 'top3_overlap_6csv_restricted': overlap_res, 'sign_agreement_6csv_restricted': float(np.mean(signs_res)) if signs_res else np.nan})
    lam_main = 0.5
    (score, rank, tops) = lam_tables[lam_main]
    rows = []
    for (j, t) in enumerate(targets):
        sel = tops[:, j]
        data_sel = set(data_top[:, j].tolist())
        for (r, f) in enumerate(sel, start=1):
            f = int(f)
            name = feats[f]
            c6row = feat_to_c6row.get(name)
            c6r = float(data['c6'].loc[c6row, target_to_c6[t]]) if c6row else np.nan
            w = float(prior_mat[f, j])
            row = {'target': t, 'rank': r, 'feature': name, 'compound_2csv': fm.loc[name, 'compound_2csv'], 'chemical_name': fm.loc[name, 'chemical_name'], 'chemical_class': fm.loc[name, 'chemical_class'], 'origin': fm.loc[name, 'origin_paper'], 'domain_score': float(score[f, j]), 'data_fused_rank': float(data_rank[f, j]), 'data_fused_score': float(data_score[f, j]), 'prior_weight': w, 'prior_bonus': float(lam_main * w), 'prior_tier': 'direct' if w >= 1.0 else 'specific' if w >= 0.7 else 'broad' if w >= 0.4 else 'proxy' if w >= 0.3 else 'none', 'stability_freq': float(boot_freq[f'DOMAIN_{lam_main}'][j, f]), 'pure_selector_frequency': float(boot_freq['FUSED'][j, f]), 'data_in_top3': int(f in data_sel), 'rank_A': float(methods['ranks']['A'][f, j]), 'rank_B': float(methods['ranks']['B'][f, j]), 'rank_C': float(methods['ranks']['C'][f, j]), 'rank_D': float(methods['ranks']['D'][f, j]), 'rank_E': float(methods['ranks']['E'][f, j]), 'spearman_r': float(methods['scores']['A'][f, j]), 'corr6_r': c6r}
            rows.append(row)
    domain_df = pd.DataFrame(rows)
    sens_df = pd.DataFrame(sensitivity)
    agg = []
    for lam in lambdas:
        sub = sens_df[sens_df['lambda'] == lam]
        agg.append({'lambda': lam, 'mean_overlap_with_data_top3': round(float(sub.overlap_with_data_top3.mean()), 4), 'mean_stability_fused': round(float(sub.mean_stability_fused.mean()), 4), 'mean_data_fused_rank': round(float(sub.mean_data_fused_rank.mean()), 2), 'n_direct_prior_in_top3_total': int(sub.n_direct_prior_in_top3.sum()), 'mean_top3_overlap_6csv_final': round(float(sub.top3_overlap_6csv_final.mean()), 4), 'mean_sign_agreement_6csv_final': round(float(sub.sign_agreement_6csv_final.mean()), 4), 'mean_top3_overlap_6csv_restricted': round(float(sub.top3_overlap_6csv_restricted.mean()), 4), 'mean_sign_agreement_6csv_restricted': round(float(sub.sign_agreement_6csv_restricted.mean()), 4)})
    return {'domain': domain_df, 'sensitivity': sens_df, 'aggregate': pd.DataFrame(agg), 'lambda_main': lam_main}


def subset_validation(data, net):
    """Evaluate 3-feature sets chosen strictly inside each training fold; graph is supplied prior."""
    rows = []
    X = data['X']
    Y = data['y_arr']
    for (fold, (tr, te)) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        print(f'[subset CV] fold {fold + 1}/5', flush=True)
        raw = X.iloc[tr].reset_index(drop=True)
        z = StandardScaler().fit_transform(SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(raw))
        inner = {**data, 'X': raw, 'Xs': z, 'y_arr': Y[tr]}
        result = run_full_methods(inner, net)
        schemes = {'marginal': result['ranks']['A'], 'fused5': result['fused_rank'], 'fused4_no_network': np.mean([result['ranks'][m] for m in METHODS[:4]], axis=0)}
        compound_rank = result['fused_rank'].copy()
        compound_rank[data['fm'].chemical_class.isin(['Physicochemical', 'Derived index']).values] = np.inf
        schemes['compound_only'] = compound_rank
        for (name, rank) in schemes.items():
            for (j, t) in enumerate(data['y'].columns):
                cols = np.argsort(rank[:, j], kind='stable')[:3]
                pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0))])
                pipe.fit(X.iloc[tr, cols], Y[tr, j])
                pred = pipe.predict(X.iloc[te, cols])
                rows.append(dict(fold=fold, method=name, target=t, r2=r2_score(Y[te, j], pred), rmse=float(np.sqrt(np.mean((Y[te, j] - pred) ** 2))), features=';'.join((data['feats'][c] for c in cols)), scope='training_only_top3_selection_fixed_given_graph; ridge_alpha10'))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / '前三候选预测交叉验证.csv', index=False)
    return out.groupby('method')[['r2', 'rmse']].mean().to_dict('index')

def additional_tables(data, net, methods, boot_freq):
    rows = []
    ranks = {'fused4_no_network': np.mean([methods['ranks'][m] for m in METHODS[:4]], axis=0)}
    chemical = ~data['fm'].chemical_class.isin(['Physicochemical', 'Derived index']).values
    cr = methods['fused_rank'].copy()
    cr[~chemical] = np.inf
    ranks['compound_only'] = cr
    for (scheme, r) in ranks.items():
        key = 'FUSED_NO_E' if scheme == 'fused4_no_network' else 'COMPOUND_ONLY'
        for (j, t) in enumerate(data['y'].columns):
            for (order, i) in enumerate(np.argsort(r[:, j], kind='stable')[:3], 1):
                rows.append(dict(scheme=scheme, target=t, rank=order, feature=data['feats'][i], chemical_class=data['fm'].iloc[i].chemical_class, stability_freq=boot_freq[key][j, i]))
    pd.DataFrame(rows).to_csv(OUT / '前三候选替代方案.csv', index=False)
    records = []
    for lam in [0.0, 0.1, 0.2, 0.5]:
        for (j, t) in enumerate(data['y'].columns):
            for (i, f) in enumerate(data['feats']):
                records.append(dict(lambda_prior=lam, target=t, feature=f, frequency=boot_freq[f'DOMAIN_{lam}'][j, i]))
    pd.DataFrame(records).to_csv(OUT / '机理先验重采样频率.csv', index=False)
    sensitivities = []
    for (scheme, k, thr) in [('top5', 5, None), ('top10', 10, None), ('abs_r_ge_0.5', None, 0.5)]:
        corr = pd.DataFrame(data['corr2'].values, index=data['feats'], columns=data['feats'])
        (graph, _) = build_graph(corr, data['feats'], 'topk' if k else 'abs_threshold', k=k, threshold=thr)
        W = _row_normalised_adjacency(graph, data['feats'])
        for lam in [0.0, 0.1, 0.3, 0.6]:
            erank = rank_features(smooth_scores(methods['scores']['A'], W, lam), True)
            fused = np.mean([methods['ranks'][m] for m in METHODS[:4]] + [erank], axis=0)
            for (j, t) in enumerate(data['y'].columns):
                ref = set(np.argsort(methods['fused_rank'][:, j])[:3])
                sel = set(np.argsort(fused[:, j])[:3])
                sensitivities.append(dict(scheme=scheme, lambda_smoothing=lam, target=t, top3_jaccard=len(ref & sel) / len(ref | sel)))
    pd.DataFrame(sensitivities).to_csv(OUT / '网络方案候选敏感性.csv', index=False)
    return pd.DataFrame(rows)

def main():
    t_start = time.time()
    print('=' * 78, flush=True)
    print('Q2: compound co-expression network, sensory Top-3, stability, chemistry', flush=True)
    print(f'n_boot={N_BOOT}, n_jobs={N_JOBS}, lambda_network={NET_LAMBDA}', flush=True)
    print('=' * 78, flush=True)
    data = load_data()
    maps = build_corr6_maps(data)
    (prior, prior_missing) = build_domain_prior(data)
    print(f'[domain] prior sets built for {sum((bool(v) for v in prior.values()))}/50 targets; missing literature names={len(prior_missing)}', flush=True)
    print(f"[data] X={data['X'].shape}, y={data['y'].shape}, corr2={data['corr2'].shape}, corr6={data['c6'].shape}", flush=True)
    print('[data] chemical classes:', data['fm']['chemical_class'].value_counts().to_dict(), flush=True)
    net = build_network(data)
    methods = run_full_methods(data, net)
    (jac_long, jac_mat) = jaccard_tables(methods, data['y'].shape[1])
    print(f'[jaccard] mean pairwise={jac_mat[np.triu_indices(5, 1)].mean():.3f}', flush=True)
    (boot_freq, boot_sec) = run_bootstrap(data, net, methods, N_BOOT)
    alternatives = additional_tables(data, net, methods, boot_freq)
    primary = alternatives[alternatives.scheme.eq('compound_only')].copy()
    primary = primary.merge(data['fm'][['feature', 'compound_2csv', 'chemical_name']], on='feature', validate='many_to_one')
    primary['scope'] = '214 chemical candidates; 13 physicochemical and 4 derived indices excluded'
    top3_probe = topk_indices(methods['fused_rank'], 3)
    probe_freq = np.mean([boot_freq['FUSED'][j, top3_probe[:, j]] for j in range(50)])
    print(f'[bootstrap] mean fused stability of final Top-3 = {probe_freq:.3f} ({boot_sec:.1f}s)', flush=True)
    enrich = enrichment_table(data, methods)
    bench = benchmark_6csv(data, maps, methods)
    print('[benchmark] mean rank corr / top3 overlap / sign agreement:', flush=True)
    print(bench.groupby('method')[['rank_corr_spearman', 'top3_overlap_restricted', 'sign_agreement_restricted']].mean().round(3).to_string(), flush=True)
    tables = make_tables(data, net, methods, jac_long, jac_mat, bench, enrich, boot_freq, N_BOOT)
    cases = ['diacetyl', 'dms', 'X4vg', 't2n', 'stale_hops', 'A_esters_fruity', 'A_hops_citrus']
    case_df = case_table(data, maps, methods, boot_freq, cases)
    domain = domain_tables(data, maps, methods, boot_freq, prior)
    tables['top3'].to_csv(OUT / '前三关键指标_全指标诊断.csv', index=False, encoding='utf-8-sig')
    primary.to_csv(OUT / '前三关键化合物.csv', index=False, encoding='utf-8-sig')
    primary[['target', 'rank', 'feature', 'stability_freq', 'scope']].to_csv(OUT / '关键化合物稳定性.csv', index=False, encoding='utf-8-sig')
    tables['boot_freq'].to_csv(OUT / '自助重采样入选频率.csv', index=False, encoding='utf-8-sig')
    tables['jaccard'].to_csv(OUT / '方法间入选集合相似度.csv', index=False, encoding='utf-8-sig')
    tables['audit'].to_csv(OUT / '各方法候选排名.csv', index=False, encoding='utf-8-sig')
    net['nodes'].to_csv(OUT / '网络节点.csv', index=False, encoding='utf-8-sig')
    net['edges'].to_csv(OUT / '网络连边.csv', index=False, encoding='utf-8-sig')
    net['scheme_table'].to_csv(OUT / '网络方案对比.csv', index=False, encoding='utf-8-sig')
    net['communities'].to_csv(OUT / '网络社区.csv', index=False, encoding='utf-8-sig')
    bench.to_csv(OUT / '附件六相关性一致性.csv', index=False, encoding='utf-8-sig')
    enrich.to_csv(OUT / '化学类别富集.csv', index=False, encoding='utf-8-sig')
    case_df.to_csv(OUT / '典型维度前五候选.csv', index=False, encoding='utf-8-sig')
    domain['domain'].to_csv(OUT / '机理先验前三候选.csv', index=False, encoding='utf-8-sig')
    domain['sensitivity'].to_csv(OUT / '机理先验敏感性明细.csv', index=False, encoding='utf-8-sig')
    domain['aggregate'].to_csv(OUT / '机理先验敏感性汇总.csv', index=False, encoding='utf-8-sig')
    print('[tables] written to data/q2/', flush=True)
    t0 = time.time()
    elapsed = time.time() - t_start
    subset_validation(data, net)
    f = primary.stability_freq
    g = primary.groupby('target').stability_freq
    print('\n' + '=' * 78, flush=True)
    print('NETWORK MODULES', flush=True)
    print(net['communities'][['community', 'size', 'top_origins', 'top_classes']].to_string(index=False), flush=True)
    print('\nTOP HUBS (composite degree/betweenness/eigenvector rank)', flush=True)
    print(net['nodes'].sort_values('hub_rank')[['feature', 'compound_2csv', 'origin_paper', 'chemical_class', 'community', 'degree', 'betweenness', 'eigenvector']].head(12).to_string(index=False), flush=True)
    print('\nSTABILITY', flush=True)
    print(f"mean fused Top-3 frequency = {f.mean():.3f}; entries>=0.8 = {int((f >= 0.8).sum())}/150; targets with all 3 >=0.8 = {int((g.min() >= 0.8).sum())}/50", flush=True)
    print('\nJACCARD mean pairwise =', round(float(jac_mat[np.triu_indices(5, 1)].mean()), 3), flush=True)
    print('BENCHMARK 6.csv (44 matched):', flush=True)
    print(bench.groupby('method')[['rank_corr_spearman', 'top3_overlap_restricted', 'top3_overlap_final', 'sign_agreement_restricted', 'sign_agreement_final']].mean().round(3).to_string(), flush=True)
    print('\nCASE TOP-3', flush=True)
    print(case_df[case_df.in_final_top3][['target', 'rank', 'compound_2csv', 'chemical_class', 'freq_fused', 'rank_A', 'rank_B', 'rank_C', 'rank_D', 'rank_E', 'corr6_r']].to_string(index=False), flush=True)
    print('\nDOMAIN-INFORMED SENSITIVITY (lambda grid)', flush=True)
    print(domain['aggregate'].to_string(index=False), flush=True)
    print('\nDOMAIN-INFORMED CASE TOP-3 (lambda=0.5)', flush=True)
    dd = domain['domain']
    print(dd[dd.target.isin(cases)][['target', 'rank', 'compound_2csv', 'chemical_class', 'prior_tier', 'data_fused_rank', 'stability_freq', 'corr6_r']].to_string(index=False), flush=True)
    print('\nDONE in %.1fs' % elapsed, flush=True)
if __name__ == '__main__':
    main()
