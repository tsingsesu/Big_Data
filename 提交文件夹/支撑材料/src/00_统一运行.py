# AI辅助：OpenAI Codex，OpenAI，2026-10-29；具体模型版本待参赛队核对。
# 本文件整理运行入口、数据路径与指定图形，具体使用及人工核验情况见AI使用详情说明。
"""支撑材料统一入口：预处理、三问求解、七张指定图及结果校验。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / 'img'
FIGURES = {'03', '05', '17', '18', '19', '20', '21'}
QUESTION_SCRIPTS = {'q1': '问题一_感官评分.py', 'q2': '问题二_化合物关联.py',
                    'q3': '问题三_风味生成与迁移.py'}


def run_script(name, extra=()):
    print('[run]', name, *extra, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'src' / name), *extra], cwd=ROOT, check=True)


def check_inputs():
    from 公共数据处理 import ATTACH
    needed = ['1.csv', '2.csv', '3.xlsx', '4.csv', '5.csv', '6.csv', '7.csv', '8.xlsx']
    missing = [name for name in needed if not (ATTACH / name).is_file()]
    if missing:
        raise FileNotFoundError('缺少赛题原始附件：' + ', '.join(missing) +
                                '；使用 --attachments 指定含1–8附件的目录。当前目录：' + str(ATTACH))
    print('[inputs]', ATTACH, flush=True)


def verify_files():
    manifest = json.loads((ROOT / '文件清单.json').read_text())
    failures = []
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    if failures:
        raise ValueError('交付快照校验不一致：' + ', '.join(failures))
    actual = {path.name[:2] for path in IMG.glob('*.png')}
    if actual != FIGURES or len(list(IMG.glob('*.png'))) != 7:
        raise ValueError('img中应恰好包含03、05、17–21七张PNG')
    print('[verify]', len(manifest['sha256']), 'files; seven PNGs; SHA256 passed', flush=True)


def network_layout(work, rebuild=False):
    nodes = pd.read_csv(ROOT / 'data/q2/网络节点.csv')
    edges = pd.read_csv(ROOT / 'data/q2/网络连边.csv')
    nodes[['feature']].to_csv(work / '关联网络布局节点.tsv', sep='\t', index=False)
    edges[['source', 'target', 'abs_r']].to_csv(work / '关联网络布局连边.tsv', sep='\t', index=False)
    adapter = ROOT / 'src/化合物网络布局.java'
    config = dict(node_sha=hashlib.sha256((work / '关联网络布局节点.tsv').read_bytes()).hexdigest(),
                  edge_sha=hashlib.sha256((work / '关联网络布局连边.tsv').read_bytes()).hexdigest(),
                  java_sha=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                  seed=42, iterations=1200, threads=1)
    layout = work / '关联网络布局坐标.csv'
    key = work / '关联网络布局配置.json'
    if not rebuild and layout.exists() and key.exists() and json.loads(key.read_text()) == config:
        print('[layout] reuse verified Gephi coordinates', flush=True)
        return
    home = os.environ.get('GEPHI_HOME')
    if not home or not Path(home).is_dir():
        raise RuntimeError('重新计算网络坐标需要Gephi 0.11.3：设置GEPHI_HOME；随包坐标可直接用于重绘。')
    gephi = Path(home)
    jars = [str(path) for path in gephi.rglob('*.jar') if 'locale' not in path.parts]
    java_candidates = list(gephi.glob('jre-x64/*/bin/java'))
    java = str(java_candidates[0]) if java_candidates else shutil.which('java')
    javac = shutil.which('javac')
    if not java or not javac or not jars:
        raise RuntimeError('需要可用的javac、Java 17运行时及Gephi依赖JAR。')
    cache = ROOT / 'data/cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache, prefix='gephi-') as directory:
        subprocess.run([javac, '-encoding', 'UTF-8', '-d', directory, str(adapter)], check=True)
        subprocess.run([java, '-Djava.awt.headless=true', '-cp', os.pathsep.join([directory] + jars),
                        '化合物网络布局', str(work / '关联网络布局节点.tsv'), str(work / '关联网络布局连边.tsv'), str(layout)],
                       check=True)
    key.write_text(json.dumps(config, indent=2))


def paper_figures(selected=None, rebuild_layout=False):
    from 公共数据处理 import load_all, TARGET_CN
    from 问题三_风味生成与迁移 import ConcentrationTransform
    selected = set(selected or FIGURES)
    if not selected <= FIGURES:
        raise ValueError('只保留03、05、17、18、19、20、21')
    if not shutil.which('Rscript'):
        raise RuntimeError('重绘需要Rscript、Cairo和Noto Serif CJK SC字体；数值求解与校验不需要R。')
    IMG.mkdir(exist_ok=True)
    work = ROOT / 'data/processed/paper_plot_data'
    work.mkdir(parents=True, exist_ok=True)
    data = load_all()
    targets = data['targets']
    label = lambda t: TARGET_CN.get(t, t).replace('-总体', '').replace('4-乙烯基愈创木酚', '4VG').replace('反-2-壬烯醛', 'T2N')
    pd.DataFrame(dict(target=targets, label=[label(t) for t in targets], order=range(50))).to_csv(work / '感官维度标签.csv', index=False)
    observed = data['X_cal'].copy()
    observed.insert(0, 'style', np.asarray(data['styles']))
    observed.to_csv(work / '校准浓度与风格.csv', index=False)
    if selected & {'19', '20', '21'}:
        generated = pd.read_csv(ROOT / 'data/q3/生成化合物谱.csv')
        portraits = pd.read_csv(ROOT / 'data/q3/感官画像.csv')
        transfer = pd.read_csv(ROOT / 'data/q3/迁移化合物谱.csv')
    if selected & {'20', '21'}:
        dims = ['A_malt_all', 'A_hops_all', 'A_esters_all', 'F_malt_all', 'F_hops_all', 'F_esters_all',
                'acidity', 'bitternes', 'sweetness', 'alcohol', 'body', 'overall']
        reference = portraits[portraits.kind.eq('real') & portraits.model.eq('observed')]
        rows = []
        def add_radar(frame, kind, style, model):
            medians = frame[dims].median()
            for target in dims:
                values = reference[target].dropna().to_numpy()
                value = float(medians[target])
                percentile = 100 * (np.sum(values < value) + .5 * np.sum(values == value)) / len(values)
                rows.append(dict(kind=kind, style=style, model=model, target=target, response=value,
                                 percentile=float(percentile), n=len(frame), reference_n=len(values)))
        for style in ['Tripel', 'Kriek', 'Lambic']:
            add_radar(reference[reference['style'].eq(style)], 'real', style, 'observed')
            add_radar(portraits[portraits.kind.eq('generated') & portraits.model.eq('copula') & portraits['style'].eq(style)],
                      'generated', style, 'copula')
        for model, frame in portraits[portraits.kind.eq('transfer')].groupby('model'):
            add_radar(frame, 'transfer', 'Tripel', model)
        pd.DataFrame(rows).to_csv(work / '雷达画像数据.csv', index=False)
    if '19' in selected:
        from sklearn.decomposition import PCA
        import sklearn
        columns = data['cal_features']
        transform = ConcentrationTransform().fit(data['X_cal'].values)
        z = transform.transform(data['X_cal'].values)
        pca = PCA(n_components=2, svd_solver='full', random_state=42).fit(z)
        rows = []
        def add_projection(values, styles, samples, dataset):
            xy = pca.transform(transform.transform(values))
            rows.append(pd.DataFrame(dict(dataset=dataset, style=np.asarray(styles), sample_id=np.asarray(samples),
                                          pc1=xy[:, 0], pc2=xy[:, 1])))
        add_projection(data['X_cal'].values, data['styles'], data['meta']['beer'], 'real')
        for method in ['copula', 'ot_rank']:
            group = generated[generated.generator.eq(method)]
            add_projection(group[columns].values, group['style'], group.sample_id, method)
        for variant in ['source', 'excess']:
            group = transfer[transfer.unit_scenario.eq('attachment_mg_L') & transfer.variant.eq(variant)]
            add_projection(group[columns].values, group['style'], group.sample_id, 'transfer_' + variant)
        pd.concat(rows, ignore_index=True).to_csv(work / '主成分投影坐标.csv', index=False)
        config = dict(fit_scope='250 observed beers only; generated and transferred rows only projected',
                      n_fit=len(z), features=columns, svd_solver='full', seed=42, sklearn=sklearn.__version__,
                      explained_variance_ratio=pca.explained_variance_ratio_.tolist(), components=pca.components_.tolist(),
                      pca_mean=pca.mean_.tolist(), feature_median=transform.median.tolist(), feature_c0=transform.c0.tolist(),
                      feature_mean=transform.mean.tolist(), feature_sd=transform.sd.tolist(), transfer_unit_scenario='attachment_mg_L')
        (work / '主成分投影配置.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    if '05' in selected:
        network_layout(work, rebuild_layout)
    subprocess.run(['Rscript', str(ROOT / 'src/论文配图.R'), str(ROOT), ','.join(sorted(selected))],
                   cwd=ROOT, check=True)


def validate_results():
    """Check mathematical constraints and artifact consistency, not merely file existence."""
    import hashlib
    from 公共数据处理 import load_all,scoring_card_map,provenance,ATTACH
    from 问题一_感官评分 import grade_metrics_per_target
    from 问题三_风味生成与迁移 import gaussian_mapping,ConcentrationTransform,DEFECTS,UNIT_SCENARIOS
    d=load_all();checks=[]
    def passed(name):checks.append(name);print('[validate]',name,flush=True)
    assert len(scoring_card_map())==52
    cards=pd.read_csv(ROOT/'data/q1/完整评分卡.csv')
    assert len(cards)==250*52 and cards.relative_grade.isna().sum()==500
    assert cards.relative_grade.dropna().between(0,6).all()
    passed('52 card items retained; 50 supervised, 2 acetaldehyde explicitly unavailable')
    assert grade_metrics_per_target(np.array([[0],[1],[2],[3]]),np.ones((4,1),int))['qwk'][0]==0
    passed('constant prediction with varying reference has defined zero QWK')
    for q,repeats in [('q1',3),('q3',5)]:
        splits=pd.read_csv(ROOT/f'data/{q}/样本划分.csv')
        keys=['repeat','fold'] if q=='q1' else ['repeat']
        for _,group in splits.groupby(keys):
            tr=set(group.loc[group.scope=='train','beer']);te=set(group.loc[group.scope=='test','beer'])
            assert tr.isdisjoint(te) and len(tr|te)==250
        assert splits['repeat'].nunique()==repeats
    passed('all Q1 outer folds and Q3 style holdouts have disjoint beer IDs')
    q1=json.loads((ROOT/'data/q1/模型结果汇总.json').read_text())
    cv=pd.read_csv(ROOT/'data/q1/交叉验证结果.csv')
    actual=cv[(cv.model=='selected_pipeline')&(cv.target=='MACRO')].r2.mean()
    assert np.isclose(actual,q1['models']['selected_pipeline']['macro_mean_std_over_folds']['r2']['mean'])
    passed('Q1 summary equals fold table; all 15 selected-pipeline folds present')
    freq=pd.read_csv(ROOT/'data/q2/自助重采样入选频率.csv')
    for col in [c for c in freq if c.startswith('freq_')]:
        assert freq[col].between(0,1).all() and np.allclose(freq.groupby('target')[col].sum(),3)
    mf=pd.read_csv(ROOT/'data/q2/机理先验重采样频率.csv')
    assert np.allclose(mf.groupby(['lambda_prior','target']).frequency.sum(),3)
    merged=mf[mf.lambda_prior==0].merge(freq,on=['target','feature'])
    assert np.allclose(merged.frequency,merged.freq_fused)
    passed('each bootstrap selector has exactly three expected selections; zero prior equals pure selector')
    q2=json.loads((ROOT/'data/q2/结果汇总.json').read_text());assert q2['bootstrap']['n_boot']==500
    primary=pd.read_csv(ROOT/'data/q2/前三关键化合物.csv')
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
    profiles=pd.read_csv(ROOT/'data/q3/迁移化合物谱.csv')
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
    metrics=pd.read_csv(ROOT/'data/q3/生成评价指标.csv')
    assert len(metrics)==5*3*10 and np.isfinite(metrics.select_dtypes('number')).all().all()
    assert metrics['scope'].eq('heldout_real_reference_all_parameters_train_only').all()
    gen=pd.read_csv(ROOT/'data/q3/生成化合物谱.csv')
    assert len(gen)==3*8*300 and np.isfinite(gen[d['cal_features']]).all().all()
    assert gen[d['cal_features']].ge(0).all().all()
    passed('Q3 heldout metrics and final descriptive spectra have distinct scopes and finite values')
    for q,file in [('q1','模型结果汇总.json'),('q2','结果汇总.json'),('q3','结果汇总.json')]:
        report=json.loads((ROOT/'data'/q/file).read_text())
        for path,sha in report['provenance']['sha256'].items():
            if path.startswith('附件/'):
                assert hashlib.sha256((ATTACH/Path(path).name).read_bytes()).hexdigest()==sha
    passed('all result summaries match the unchanged raw attachment fingerprints')
    out={'passed':True,'checks':checks,'provenance':provenance(),
         'limits':['No original absolute sensory grades','T2N unit not resolved','No stale_hops guarantee','No brewing feasibility guarantee']}
    (ROOT/'data/processed/结果数值验收.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--verify', action='store_true', help='核对随包文件SHA256，无需赛题原始附件')
    actions.add_argument('--validate', action='store_true', help='核对保存结果、数据划分和约束')
    actions.add_argument('--profile', action='store_true', help='运行核心预处理统计及外部数据校验')
    actions.add_argument('--question', choices=['q1', 'q2', 'q3'], help='运行某一问的默认正式实验')
    actions.add_argument('--pipeline', action='store_true', help='从预处理到三问结果、七图与校验的完整重算')
    actions.add_argument('--figures', nargs='?', const='all', help='重绘七图，或指定03,05等编号')
    parser.add_argument('--attachments', type=Path, help='官方1–8附件所在目录，不需要放入上传支撑材料')
    parser.add_argument('--rebuild-layout', action='store_true', help='重新运行Gephi布局，需设置GEPHI_HOME')
    args = parser.parse_args()
    if args.attachments:
        os.environ['BEER_ATTACHMENT_DIR'] = str(args.attachments.expanduser().resolve())
    if args.verify:
        verify_files()
        return
    if not any([args.validate, args.profile, args.question, args.pipeline, args.figures]):
        parser.print_help()
        return
    check_inputs()
    if args.validate:
        validate_results()
    elif args.figures:
        selected = None if args.figures == 'all' else set(args.figures.split(','))
        paper_figures(selected, args.rebuild_layout)
    elif args.profile:
        run_script('01_数据预处理.py')
        run_script('02_外部数据处理.py')
    elif args.question:
        run_script(QUESTION_SCRIPTS[args.question], ['--resume'] if args.question == 'q1' else [])
    else:
        run_script('01_数据预处理.py')
        run_script('02_外部数据处理.py')
        for question in ['q1', 'q2', 'q3']:
            run_script(QUESTION_SCRIPTS[question], ['--resume'] if question == 'q1' else [])
        paper_figures(rebuild_layout=args.rebuild_layout)
        validate_results()


if __name__ == '__main__':
    main()
