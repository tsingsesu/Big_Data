"""论文概念图：技术路线、预处理流程、问题一建模框架（中文、黑白朴素样式）。

运行：
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/00_figures.py
输出：
    img/fig_roadmap.pdf/.png
    img/fig_preprocess.pdf/.png
    img/fig_q1_framework.pdf/.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
IMG.mkdir(exist_ok=True)

# 中文字体：优先衬线（与论文正文风格一致），找不到就退回无衬线。
for fp in [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        break

plt.rcParams.update(
    {
        "font.family": "Noto Serif CJK JP",
        "font.size": 8,
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

INK = "#111111"
LIGHT = "#F2F2F2"


def box(ax, xy, w, h, text, fontsize=7.5, fill="white"):
    """方角、细黑边、白/浅灰填充的朴素方框。"""
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="square,pad=0.012",
        linewidth=0.9,
        edgecolor=INK,
        facecolor=fill,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.25,
    )


def arrow(ax, start, end, style="-|>", linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=0.8,
            color=INK,
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def save(fig, name):
    fig.savefig(IMG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def title(ax, text, y=0.965):
    ax.text(0.5, y, text, ha="center", va="center", fontsize=12, weight="bold", color=INK)


def roadmap():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    title(ax, "技术路线")

    box(
        ax,
        (0.01, 0.35),
        0.22,
        0.30,
        "数据底座\n1.csv–8.xlsx\n以 beer 对齐\n逐列变换\n缺失掩码",
        7.8,
    )
    box(
        ax,
        (0.34, 0.69),
        0.30,
        0.21,
        "问题一  感官评分\n连续回归\n+ 稳健 0–6 校准\n+ 缺失压力测试",
        8.0,
    )
    box(
        ax,
        (0.34, 0.39),
        0.30,
        0.21,
        "问题二  化合物–风味\n共表达网络\nTop-3 + bootstrap/Jaccard\n化学合理性",
        8.0,
    )
    box(
        ax,
        (0.34, 0.09),
        0.30,
        0.21,
        "问题三  风格生成\n条件 copula / 正确高斯 OT\nPils→Tripel 迁移\n缺陷约束 + 感官画像",
        8.0,
    )
    box(
        ax,
        (0.74, 0.35),
        0.24,
        0.30,
        "验证与论文\n交叉验证 / bootstrap\n独立留出 MMD² / 实测画像\nLaTeX + 可复现代码",
        7.8,
        fill=LIGHT,
    )

    arrow(ax, (0.23, 0.50), (0.34, 0.795))
    arrow(ax, (0.23, 0.50), (0.34, 0.495))
    arrow(ax, (0.23, 0.50), (0.34, 0.195))
    arrow(ax, (0.64, 0.795), (0.74, 0.56))
    arrow(ax, (0.64, 0.495), (0.74, 0.50))
    arrow(ax, (0.64, 0.195), (0.74, 0.44))
    save(fig, "fig_roadmap")


def preprocess():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    title(ax, "数据预处理与防泄漏", y=0.93)

    y, h = 0.40, 0.30
    xs = [0.01, 0.21, 0.41, 0.61, 0.81]
    texts = [
        "8 张原始表\n1.csv … 8.xlsx",
        "以 beer 对齐\n（不用 beer_id）\n核对 250 个样本",
        "逐列变换\n37 列 log10\n33 列线性\n161 列底数未确认",
        "缺失掩码\n自然缺失 73 格\n训练折内插补",
        "模型输入\nX 250×231\ny 250×50\n22 个风格",
    ]
    for x, t in zip(xs, texts):
        box(ax, (x, y), 0.17, h, t, 7.3)
    for i in range(4):
        arrow(ax, (xs[i] + 0.17, y + h / 2), (xs[i + 1], y + h / 2))
    ax.text(
        0.5,
        0.14,
        "所有插补、标准化与校准阈值只在训练折内拟合。",
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
    )
    save(fig, "fig_preprocess")


def q1_framework():
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    title(ax, "问题一建模框架", y=0.94)

    box(ax, (0.08, 0.70), 0.50, 0.18, "基线\n逐维均值 / 岭回归\n＋风格均值对照", 8.8)
    box(ax, (0.08, 0.42), 0.50, 0.18, "改进\n五类学习器与小融合\n缺失指示/风格独立消融", 8.8)
    box(
        ax,
        (0.08, 0.14),
        0.50,
        0.18,
        "最终\n外层训练内选模型\n训练折锚定相对 0–6",
        8.8,
        fill=LIGHT,
    )
    box(
        ax,
        (0.66, 0.30),
        0.30,
        0.34,
        "压力测试\nMCAR 5/10/20/30%\nMAR 代理\n自然缺失",
        8.0,
    )
    arrow(ax, (0.33, 0.70), (0.33, 0.60))
    arrow(ax, (0.33, 0.42), (0.33, 0.32))
    arrow(ax, (0.58, 0.40), (0.66, 0.45), linestyle="--")
    save(fig, "fig_q1_framework")



def validate_results():
    """Check mathematical constraints and artifact consistency, not merely file existence."""
    import hashlib
    from data_utils import load_all,scoring_card_map,provenance
    from q1 import grade_metrics_per_target
    from q3 import gaussian_mapping,ConcentrationTransform,DEFECTS,UNIT_SCENARIOS
    d=load_all();checks=[]
    def passed(name):checks.append(name);print('[validate]',name,flush=True)
    assert len(scoring_card_map())==52
    cards=pd.read_csv(ROOT/'data/q1/card_scores.csv')
    assert len(cards)==250*52 and cards.relative_grade.isna().sum()==500
    assert cards.relative_grade.dropna().between(0,6).all()
    passed('52 card items retained; 50 supervised, 2 acetaldehyde explicitly unavailable')
    assert grade_metrics_per_target(np.array([[0],[1],[2],[3]]),np.ones((4,1),int))['qwk'][0]==0
    passed('constant prediction with varying reference has defined zero QWK')
    for q,repeats in [('q1',3),('q3',5)]:
        splits=pd.read_csv(ROOT/f'data/{q}/splits.csv')
        keys=['repeat','fold'] if q=='q1' else ['repeat']
        for _,group in splits.groupby(keys):
            tr=set(group.loc[group.scope=='train','beer']);te=set(group.loc[group.scope=='test','beer'])
            assert tr.isdisjoint(te) and len(tr|te)==250
        assert splits['repeat'].nunique()==repeats
    passed('all Q1 outer folds and Q3 style holdouts have disjoint beer IDs')
    q1=json.loads((ROOT/'data/q1/model_summary.json').read_text())
    cv=pd.read_csv(ROOT/'data/q1/cv_results.csv')
    actual=cv[(cv.model=='selected_pipeline')&(cv.target=='MACRO')].r2.mean()
    assert np.isclose(actual,q1['models']['selected_pipeline']['macro_mean_std_over_folds']['r2']['mean'])
    passed('Q1 summary equals fold table; all 15 selected-pipeline folds present')
    freq=pd.read_csv(ROOT/'data/q2/bootstrap_freq.csv')
    for col in [c for c in freq if c.startswith('freq_')]:
        assert freq[col].between(0,1).all() and np.allclose(freq.groupby('target')[col].sum(),3)
    mf=pd.read_csv(ROOT/'data/q2/mechanism_bootstrap_freq.csv')
    assert np.allclose(mf.groupby(['lambda_prior','target']).frequency.sum(),3)
    merged=mf[mf.lambda_prior==0].merge(freq,on=['target','feature'])
    assert np.allclose(merged.frequency,merged.freq_fused)
    passed('each bootstrap selector has exactly three expected selections; zero prior equals pure selector')
    q2=json.loads((ROOT/'data/q2/summary.json').read_text());assert q2['bootstrap']['n_boot']==500
    primary=pd.read_csv(ROOT/'data/q2/top3.csv')
    assert len(primary)==150 and primary.groupby('target').size().eq(3).all()
    assert not primary.chemical_class.isin(['Physicochemical','Derived index']).any()
    assert primary.stability_freq.between(0,1).all()
    assert np.isclose(primary.stability_freq.mean(),q2['top3']['mean_stability'])
    passed('Q2 150 primary chemical entries exclude physicochemical/derived indices and use all 500 resamples')
    a=np.array([[2.,.6],[.6,1.]]);b=np.array([[1.,-.3],[-.3,3.]])
    ot=gaussian_mapping(a,b);wc=gaussian_mapping(a,b,False)
    assert np.allclose(ot,ot.T) and np.allclose(ot@a@ot.T,b)
    assert np.trace(a+b-2*ot@a)<=np.trace(a+b-2*wc@a)+1e-8
    passed('noncommuting Gaussian OT is symmetric, matches covariance, and costs no more than WC')
    x=np.array([[.2,2.],[.4,3.],[.5,4.],[.8,6.]])
    z1=ConcentrationTransform().fit(x).transform(x)
    x[:,0]*=1000;z2=ConcentrationTransform().fit(x).transform(x)
    assert np.allclose(z1,z2)
    passed('dimensionless concentration geometry is invariant to consistent unit conversion')
    profiles=pd.read_csv(ROOT/'data/q3/transfer_profiles.csv')
    for scenario,t2n in UNIT_SCENARIOS.items():
        sub=profiles[profiles.unit_scenario==scenario]
        source=sub[sub.variant=='source'].set_index('sample_id')
        strict=sub[sub.variant=='strict'].set_index('sample_id').loc[source.index]
        excess=sub[sub.variant=='excess'].set_index('sample_id').loc[source.index]
        for name,(col,tau) in DEFECTS.items():
            tau=t2n if name=='T2N' else tau
            assert np.all(strict[col]<=source[col]+1e-9) and np.all(strict[col]<=tau+1e-9)
            assert np.all(np.maximum(excess[col]-tau,0)<=np.maximum(source[col]-tau,0)+1e-9)
    passed('exported spectra satisfy strict and excess constraints in both unit scenarios')
    metrics=pd.read_csv(ROOT/'data/q3/metrics.csv')
    assert len(metrics)==5*3*10 and np.isfinite(metrics.select_dtypes('number')).all().all()
    assert metrics['scope'].eq('heldout_real_reference_all_parameters_train_only').all()
    gen=pd.read_csv(ROOT/'data/q3/generated_profiles.csv')
    assert len(gen)==3*8*300 and np.isfinite(gen[d['cal_features']]).all().all()
    assert gen[d['cal_features']].ge(0).all().all()
    passed('Q3 heldout metrics and final descriptive spectra have distinct scopes and finite values')
    for q,file in [('q1','model_summary.json'),('q2','summary.json'),('q3','summary.json')]:
        report=json.loads((ROOT/'data'/q/file).read_text())
        for path,sha in report['provenance']['sha256'].items():
            if path.startswith('附件/'):
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
    passed('all result summaries match the unchanged raw attachment fingerprints')
    out={'passed':True,'checks':checks,'provenance':provenance(),
         'limits':['No original absolute sensory grades','T2N unit not resolved','No stale_hops guarantee','No brewing feasibility guarantee']}
    (ROOT/'data/processed/revision_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    return out


def main():
    """Single sequential entry point; no parallel user terminals or hidden logs."""
    import argparse,subprocess,sys
    parser=argparse.ArgumentParser()
    parser.add_argument('--pipeline',action='store_true',help='run profile, external, Q1, Q2, Q3 sequentially, then concept figures')
    parser.add_argument('--validate',action='store_true',help='verify final tables, identities and physical proxy constraints')
    args=parser.parse_args()
    if args.validate:
        validate_results();return
    if args.pipeline:
        for script in ['01_profile.py','02_external.py','q1.py','q2.py','q3.py']:
            cmd=[sys.executable,str(ROOT/'src'/script)]
            if script=='q1.py':cmd.append('--resume')
            print('[pipeline]',script,flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
        validate_results()
    roadmap();preprocess();q1_framework()
    print('概念图已更新；结果图由各问题脚本与结果一起生成。',flush=True)


if __name__=='__main__':main()
