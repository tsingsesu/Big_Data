# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本次仅整理运行入口、路径和可视化范围，核心计算保留原程序；更早的AI参与见使用详情说明。
"""数据预处理；加--plots可在三问计算完成后准备七张论文图的数据。"""
import sys
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
    print('样本数、特征数、感官维度数：', len(X), X.shape[1], y.shape[1])
    print('自然缺失单元格数：', int(X.isna().sum().sum()))
    print('风格样本量：', styles.value_counts().to_dict())
    print('特征变换：', fm['transform'].value_counts().to_dict())
    print(cons.to_string(index=False))



def network_layout(work):
    import os, shutil, subprocess, tempfile
    nodes = pd.read_csv(ROOT / 'data/q2/网络节点.csv')[['feature']]
    edges = pd.read_csv(ROOT / 'data/q2/网络连边.csv')[['source', 'target', 'abs_r']]
    node_file = work / '关联网络布局节点.tsv'
    edge_file = work / '关联网络布局连边.tsv'
    layout_file = work / '关联网络布局坐标.csv'
    same = False
    if node_file.exists() and edge_file.exists() and layout_file.exists():
        old_nodes = pd.read_csv(node_file, sep='\t')
        old_edges = pd.read_csv(edge_file, sep='\t')
        same = (nodes.equals(old_nodes) and len(edges) == len(old_edges)
                and edges[['source', 'target']].equals(old_edges[['source', 'target']])
                and np.allclose(edges.abs_r, old_edges.abs_r, rtol=0, atol=1e-12))
    nodes.to_csv(node_file, sep='\t', index=False)
    edges.to_csv(edge_file, sep='\t', index=False)
    if same:
        return
    home = os.environ.get('GEPHI_HOME')
    if not home:
        raise RuntimeError('网络已变化或缺少坐标；重算布局需要设置GEPHI_HOME。')
    gephi = Path(home)
    jars = [str(f) for f in gephi.rglob('*.jar') if 'locale' not in f.parts]
    bundled = list(gephi.glob('jre-x64/*/bin/java'))
    java = str(bundled[0]) if bundled else shutil.which('java')
    javac = shutil.which('javac')
    if not java or not javac or not jars:
        raise RuntimeError('需要Gephi 0.11.3、Java 17和javac。')
    with tempfile.TemporaryDirectory(dir=work) as classes:
        subprocess.run([javac, '-encoding', 'UTF-8', '-d', classes,
                        str(ROOT / 'src/化合物网络布局.java')], check=True)
        subprocess.run([java, '-Djava.awt.headless=true', '-cp', os.pathsep.join([classes] + jars),
                        '化合物网络布局', str(node_file), str(edge_file), str(layout_file)], check=True)

def prepare_plot_data():
    from 公共数据处理 import load_all, TARGET_CN
    from 问题三_风味生成与迁移 import ConcentrationTransform
    work = ROOT / 'data/processed/paper_plot_data'
    work.mkdir(parents=True, exist_ok=True)
    data = load_all()
    targets = data['targets']
    label = lambda t: TARGET_CN.get(t, t).replace('-总体', '').replace('4-乙烯基愈创木酚', '4VG').replace('反-2-壬烯醛', 'T2N')
    pd.DataFrame(dict(target=targets, label=[label(t) for t in targets], order=range(50))).to_csv(work / '感官维度标签.csv', index=False)
    observed = data['X_cal'].copy()
    observed.insert(0, 'style', np.asarray(data['styles']))
    observed.to_csv(work / '校准浓度与风格.csv', index=False)
    generated = pd.read_csv(ROOT / 'data/q3/生成化合物谱.csv')
    portraits = pd.read_csv(ROOT / 'data/q3/感官画像.csv')
    transfer = pd.read_csv(ROOT / 'data/q3/迁移化合物谱.csv')
    dims = ['A_malt_all', 'A_hops_all', 'A_esters_all', 'F_malt_all', 'F_hops_all', 'F_esters_all', 'acidity', 'bitternes', 'sweetness', 'alcohol', 'body', 'overall']
    reference = portraits[portraits.kind.eq('real') & portraits.model.eq('observed')]
    rows = []

    def add_radar(frame, kind, style, model):
        medians = frame[dims].median()
        for target in dims:
            values = reference[target].dropna().to_numpy()
            value = float(medians[target])
            percentile = 100 * (np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values)
            rows.append(dict(kind=kind, style=style, model=model, target=target, response=value, percentile=float(percentile), n=len(frame), reference_n=len(values)))
    for style in ['Tripel', 'Kriek', 'Lambic']:
        add_radar(reference[reference['style'].eq(style)], 'real', style, 'observed')
        add_radar(portraits[portraits.kind.eq('generated') & portraits.model.eq('copula') & portraits['style'].eq(style)], 'generated', style, 'copula')
    for (model, frame) in portraits[portraits.kind.eq('transfer')].groupby('model'):
        add_radar(frame, 'transfer', 'Tripel', model)
    pd.DataFrame(rows).to_csv(work / '雷达画像数据.csv', index=False)
    from sklearn.decomposition import PCA
    import sklearn
    columns = data['cal_features']
    transform = ConcentrationTransform().fit(data['X_cal'].values)
    z = transform.transform(data['X_cal'].values)
    pca = PCA(n_components=2, svd_solver='full', random_state=42).fit(z)
    rows = []

    def add_projection(values, styles, samples, dataset):
        xy = pca.transform(transform.transform(values))
        rows.append(pd.DataFrame(dict(dataset=dataset, style=np.asarray(styles), sample_id=np.asarray(samples), pc1=xy[:, 0], pc2=xy[:, 1])))
    add_projection(data['X_cal'].values, data['styles'], data['meta']['beer'], 'real')
    for method in ['copula', 'ot_rank']:
        group = generated[generated.generator.eq(method)]
        add_projection(group[columns].values, group['style'], group.sample_id, method)
    for variant in ['source', 'excess']:
        group = transfer[transfer.unit_scenario.eq('attachment_mg_L') & transfer.variant.eq(variant)]
        add_projection(group[columns].values, group['style'], group.sample_id, 'transfer_' + variant)
    pd.concat(rows, ignore_index=True).to_csv(work / '主成分投影坐标.csv', index=False)
    pd.DataFrame({'component': [1, 2], 'explained_variance_ratio': pca.explained_variance_ratio_}).to_csv(work / '主成分解释率.csv', index=False)
    network_layout(work)

if __name__ == '__main__':
    if '--plots' in sys.argv:
        prepare_plot_data()
    else:
        main()
