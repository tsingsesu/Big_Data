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
        "问题三  风格生成\n条件 copula / 最优传输\nPils→Tripel 迁移\n缺陷约束 + 感官画像",
        8.0,
    )
    box(
        ax,
        (0.74, 0.35),
        0.24,
        0.30,
        "验证与论文\n交叉验证 / bootstrap\nMMD / 风格分类器\nLaTeX + 可复现代码",
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
        "逐列变换\n37 列 log10\n33 列线性\n161 列推断 log10",
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

    box(ax, (0.08, 0.70), 0.50, 0.18, "基线\n逐维均值 / 岭回归\n（无非线性）", 8.8)
    box(ax, (0.08, 0.42), 0.50, 0.18, "改进\n多输出 PLS / 岭回归\n+ 缺失指示 + 风格先验", 8.8)
    box(
        ax,
        (0.08, 0.14),
        0.50,
        0.18,
        "最终\n五模型等权融合\n+ 保序校准 + 稳健 0–6",
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



def paper_result_figures():
    """从 data/q1–q3 的已保存结果重绘论文中的 8 张数据图（标签中文、颜色保持原样）。"""
    data = ROOT / "data"
    if not (data / "q1" / "cv_results.csv").exists():
        print("未找到 data/q1–q3 结果，跳过数据图重绘。")
        return

    import q1
    import q2
    import q3

    # q1/q2/q3 模块在导入时会设置各自的 rcParams；这里统一改成中文字体。
    plt.rcParams["font.family"] = "Noto Serif CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    # ---- Q1 ----
    cv = pd.read_csv(data / "q1" / "cv_results.csv")
    q1.plot_target_heatmap(cv, "blend_avg")
    miss = pd.read_csv(data / "q1" / "missingness.csv")
    q1.plot_missingness(miss, "blend_avg")

    # ---- Q2 ----
    nodes = pd.read_csv(data / "q2" / "network_nodes.csv")
    edges = pd.read_csv(data / "q2" / "network_edges.csv")
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["feature"])
    for _, r in edges.iterrows():
        G.add_edge(r["source"], r["target"], weight=float(r["r"]))
    s2 = json.loads((data / "q2" / "summary.json").read_text(encoding="utf-8"))
    net = {"G": G, "nodes": nodes, "stats": {"modularity": s2["network"]["modularity"]}}
    q2.plot_network(net)
    top3 = pd.read_csv(data / "q2" / "top3.csv")
    q2.plot_top3_stability(top3, n_boot=int(s2["bootstrap"]["n_boot"]))
    jac = pd.read_csv(data / "q2" / "jaccard.csv")
    methods = q2.METHODS
    mat = np.zeros((len(methods), len(methods)))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            sub = jac[(jac.method_i == a) & (jac.method_j == b)]
            mat[i, j] = float(sub["jaccard"].mean()) if len(sub) else (1.0 if a == b else np.nan)
    for i in range(len(methods)):
        mat[i, i] = 1.0
        for j in range(i + 1, len(methods)):
            mat[j, i] = mat[i, j]
    q2.plot_jaccard(mat, jac)

    # ---- Q3 ----
    s3 = json.loads((data / "q3" / "summary.json").read_text(encoding="utf-8"))
    gen_metrics = {}
    for key, val in s3["generation"]["metrics"].items():
        model, style = key.split("|", 1)
        gen_metrics[(model, style)] = val
    q3.plot_mmd_classifier(gen_metrics)
    tr = s3["transfer"]
    q3.plot_transfer_defects(
        tr["metrics"], tr["pils_stale_proxy_p75"], tr["tripel_stale_proxy_p75"]
    )
    ports = pd.read_csv(data / "q3" / "portraits.csv")
    target_cols = [c for c in ports.columns if c not in ("kind", "model", "style", "sample_id")]
    real_med = {
        st: ports[(ports.kind == "real") & (ports["style"] == st)][target_cols].median().values
        for st in q3.TARGET_STYLES
    }
    gen_med = {}
    for st in q3.TARGET_STYLES:
        for model in ["copula", "ot", "ot_constrained"]:
            sub = ports[(ports.kind == "generated") & (ports.model == model) & (ports["style"] == st)]
            gen_med[(model, st)] = sub[target_cols].median().values
    q3.plot_radar({"targets": target_cols}, gen_med, real_med)


if __name__ == "__main__":
    roadmap()
    preprocess()
    q1_framework()
    paper_result_figures()
    print("已输出概念图（中文黑白）与数据图（中文标签、保留原配色）")
