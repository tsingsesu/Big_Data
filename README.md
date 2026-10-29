# MathorCup 大数据竞赛 A 题 · 提交材料

更新日期：2026-10-29。材料位于根目录的 `提交文件夹/`；本次不生成或复制 TeX、PDF，不提交 GitHub，也不执行比赛平台上传。

## 需要提交什么

依照组委会发布的[2026 年提交规范](https://m.saikr.com/contest/notice_detail/49567)，准备论文 PDF、签名承诺书、支撑材料及题目要求的计算结果。题目要求可复现源代码，因此本题的支撑材料需要保留代码与结果。官方规范不要求将赛题原始数据重复装入支撑材料，也未规定必须上传 TeX。

使用 AI 的队伍还须按[2026 年 AI 使用规定](https://m.saikr.com/contest/notice_detail/49194)，在支撑材料中放入固定名称的 `AI工具使用详情说明.pdf`，并在论文和代码中说明实际使用情况。当前只有可编辑说明稿，PDF及最终人工核对交给队友。

论文附录需列出实际使用的软件、命令及全部自编源程序；这部分由论文负责人同步整理。

[本届竞赛主页](https://m.saikr.com/vse/bigdata2026)公布的初赛截止时间为 **2026-10-30 20:00**。最终文件名按报名系统中的参赛队号填写；官方示例为 `A2600001.pdf`、`A2600001承诺书`、`A2600001附件.zip`，不要把示例编号直接用于本队。

## 目录结构

```text
提交文件夹/
├── 支撑材料/                     准备压缩上传的目录
│   ├── README.md                 本说明的随包副本
│   ├── requirements.txt          Python 依赖版本
│   ├── 文件清单.json             随包文件的 SHA256 校验值
│   ├── 环境与验收.json           环境、裁剪范围与实际验证记录
│   ├── src/
│   │   ├── 00_figures.py          统一运行、重绘与校验入口
│   │   ├── data_utils.py          附件读取、对齐、元数据、变换与分级工具
│   │   ├── 01_profile.py          数据概况、变换表和跨表一致性统计
│   │   ├── 02_external.py         外部数据清洗、风格映射及外部校验
│   │   ├── q1.py                 问题一求解与验证
│   │   ├── q2.py                 问题二网络、Top-3和稳定性分析
│   │   ├── q3.py                 问题三条件生成与约束迁移
│   │   ├── paper_figures.R       仅保留03、05、17–21的绘图代码
│   │   └── GephiLayout.java      Gephi ForceAtlas2布局适配器
│   ├── data/
│   │   ├── q1/                  完整评分、OOF预测、划分及评价结果
│   │   ├── q2/                  完整候选、网络、频率与合理性诊断
│   │   ├── q3/                  完整生成谱、画像、迁移谱和验证结果
│   │   ├── external/            外部原始数据、字段/风格映射及来源许可
│   │   └── processed/           变换表、数据概况及绘图所需派生数据
│   └── img/                     仅7张450 dpi PNG：03、05、17–21
└── 待补充/
    ├── 2026年MathorCup数学应用挑战赛—大数据竞赛承诺书.docx
    └── AI工具使用详情说明.md       队友核实后转换为规定的PDF
```

`支撑材料/` 内不含原始赛题附件、论文、模型检查点、虚拟环境、运行日志、旧图或 Git 文件。承诺书含实名信息，放在独立的待补充目录，不与匿名论文混用。DOCX保留现有填写和嵌入图片，未改动或代签；请队长核对所有签名、队号和日期。

## 三问结果在哪里

以下路径均相对于 `提交文件夹/支撑材料/`。

|内容|主要文件|口径|
|---|---|---|
|Q1完整评分|`data/q1/card_scores.csv`|250款酒×52个卡片位置，共13000行；相对等级，不是恢复出的原始人工分数|
|Q1连续预测与等级|`data/q1/oof_predictions.csv`|250款酒的OOF连续预测及等级；卡片映射见`card_mapping.csv`|
|Q1验证|`cv_results.csv`、`grade_metrics.csv`、`missingness.csv`、`splits.csv`、`model_summary.json`|保留每折、缺失情景、模型对照与汇总|
|Q2正式Top-3|`data/q2/top3.csv`|50个感官维度×3，共150项；仅化学候选|
|Q2共表达网络|`network_nodes.csv`、`network_edges.csv`、`network_communities.csv`|231节点、1683边、8社区|
|Q2稳定性与诊断|`stability.csv`、`bootstrap_freq.csv`、`jaccard.csv`、`method_rankings.csv`、`summary.json`等|正式化学候选与全指标诊断有不同作用域，勿混用|
|Q3生成化合物谱|`data/q3/generated_profiles.csv`|3风格×8方法×300条，共7200条70维生成谱|
|Q3感官画像|`data/q3/portraits.csv`|实测、生成和迁移画像，通过`kind`、`model`、`sample_id`区分|
|Q3迁移化合物谱|`data/q3/transfer_profiles.csv`|11款源酒×6方案×2种T2N单位情景，共132条|
|Q3验证与约束|`metrics.csv`、`sensory_cv.csv`、`transfer_metrics.csv`、`splits.csv`、`summary.json`等|独立留出指标与全量拟合后的描述性产物分开保存|

A题没有提供独立计算结果上传表的固定列模板，本次保留完整CSV而不另造“官方格式”。若报名系统另有结果文件模板，队长按其要求选择或整理这些结果后上传。

## 如何运行

所有命令都在 **`提交文件夹/支撑材料/`** 中执行；若阅读解压后的随包README，即在该README所在目录执行。已验证环境为Linux、Python 3.10.12；Python依赖固定在`requirements.txt`。

建议在独立虚拟环境准备依赖，不修改机器的系统Python：

```bash
python3 -m venv ../运行环境
source ../运行环境/bin/activate
python -m pip install -r requirements.txt
```

赛题原始数据不在上传包中。将官方1–8附件放在一个目录，通过`--attachments`指定；下例的`/path/to/官方附件`需要替换成真实路径。也可设置`BEER_ATTACHMENT_DIR`，程序只读该目录。

```bash
# 快速核对本次交付文件，无需官方附件
python src/00_figures.py --verify

# 核对已有数值结果、数据划分与缺陷约束
python src/00_figures.py --validate --attachments /path/to/官方附件

# 仅运行核心数据处理与外部校验
python src/00_figures.py --profile --attachments /path/to/官方附件

# 每问独立运行；均使用正式实验默认参数
python src/00_figures.py --question q1 --attachments /path/to/官方附件
python src/00_figures.py --question q2 --attachments /path/to/官方附件
python src/00_figures.py --question q3 --attachments /path/to/官方附件

# 仅重绘保留的七图，也可写 --figures 03,05
python src/00_figures.py --figures --attachments /path/to/官方附件

# 一条命令从预处理到三问求解、七图及结果校验
python src/00_figures.py --pipeline --attachments /path/to/官方附件
```

完整重算会覆盖相应`data/`结果，建议在解压副本中运行。Q1默认3次重复×5折，Q2默认500次bootstrap，Q3默认5次风格内留出、每风格每方法生成300条；均保留随机种子42。Q1保留恢复计算逻辑，Q2并行数可用`Q2_NJOBS`控制。不要将缩减次数的调试结果当作正式结果。

重绘还需要R/Cairo和`Noto Serif CJK SC`字体，本次使用R 4.6.1。05随包保存了经图数据指纹核验的Gephi坐标，普通重绘不必启动Gephi。如需重新计算布局，准备Gephi 0.11.3、Java 17运行时与`javac`，并设置`GEPHI_HOME`。

若只用随包的已保存绘图数据重绘七图、不重新计算派生数据，可直接运行，无需赛题原始附件：

```bash
Rscript src/paper_figures.R . 03,05,17,18,19,20,21
```

重新计算Gephi布局的命令为：

```bash
python src/00_figures.py --figures 05 --rebuild-layout --attachments /path/to/官方附件
```

## 外部数据与结果解释

外部数据为Stephen Polozoff的[Beer Information – Tasting Profiles](https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information)，公开许可为[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。2026-10-29复核Kaggle元数据为版本3。保留原外部CSV及字段说明，新增了风格映射、质量筛选、词频比例化与风格聚合：5558行中4760行能映射，质量筛选后4492行用于22种风格的外部校验；不进入Q1或Q3训练。原实验记录见`data/external/provenance.json`，本次许可复核见`source_metadata.json`。

必须保留以下解释：Q1有50项可监督响应，两项乙醛没有感官标签，不能补造分数；Q2的候选关联不等于因果；Q3的T2N单位仍保留双情景，四类直接化合物的约束不代表已保证stale hops，感官代理预测不等于真实品评。05的C1–C8为按规模降序编号的Louvain社区，节点大小随度增加；20/21雷达图采用统一实测参考下的百分位，19的前两主成分仅解释约28.3%。

## 本次整理验证

已在现有项目依赖中运行预处理、外部校验、数值验收及七图重绘；七图与开发目录最新图片逐字节一致。88个核心函数或类与原程序保持相同AST，旧绘图函数已移除，主流程仅裁剪旧图调用及调整可移植路径。原始数据、原模型结果和队友论文均未改动。

本次未重新训练完整Q1–Q3实验，未做从零安装依赖的验证，也未在打包阶段重新运行Gephi引擎。`文件清单.json`用于核对交付快照；完整重算后运行时间、程序指纹等可能变化，应再用`--validate`检查数值，不把重新生成校验值当作结果验证。

## 最终提交清单

- [x] 整理三问核心源代码、公共数据处理与统一运行入口。
- [x] 保存完整评分、Top-3、生成谱、感官画像、迁移谱与验证表。
- [x] 可视化代码和PNG仅保留03、05、17–21，均为450 dpi。
- [x] 保留外部数据、来源、许可、映射与质量筛选记录；不装入赛题原始附件。
- [x] 提供依赖版本、运行说明、文件校验值与实际验收记录；排除环境、缓存和旧稿。
- [ ] 队友完成最终论文PDF；按官方要求检查匿名性、正文页数、代码/软件命令附录及AI标注。TeX由队友管理，本包不含。
- [ ] 队员核对承诺书全部签名、队号与日期，按平台要求导出PDF或图片，单独提交。
- [ ] 队员补齐AI具体版本、关键交互和实际人工修改记录；队友将说明稿转为`AI工具使用详情说明.pdf`并放入支撑材料目录。
- [ ] 队长核对报名系统中的A赛道、实际队号与各上传项格式；如需单独计算结果文件，按平台模板整理已有CSV。
- [ ] 队友将正文结论与本包结果口径逐项对齐，披露相对评分、T2N双情景和stale hops限制。
- [ ] 如在副本中完整重算，打包前移除新生成的检查点、缓存、日志和字节码；运行环境保持在支撑材料目录之外。
- [ ] 补齐上述文档后，仅压缩`支撑材料/`中的最终内容，按实际队号命名附件ZIP/RAR；不要把`待补充/`草稿和实名承诺书混入其中。
- [ ] 队长在截止前完成平台提交，并确认电子档状态显示已提交；本次没有执行平台上传或GitHub提交。
