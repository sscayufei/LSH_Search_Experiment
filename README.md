# Hash + LSH 语义检索实验

目前已完成 Phase 1–4：数据准备、基准、LSH，以及验证集参数选择与配置锁定。
指导文件位于 `docs/EXPERIMENT_GUIDE.md.md`。

## Windows PowerShell 运行

从项目根目录执行，直接使用现有虚拟环境，无需激活：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.prepare_data --config configs/experiment.yaml
.\.venv\Scripts\python.exe -m src.prepare_data --config configs/experiment.yaml --verify-only
```

`requirements.txt` 声明兼容范围；首次成功安装后，`requirements-lock.txt` 记录本次完整依赖版本。
重现实验时在相同 Python/操作系统中安装该锁定清单。首次运行需要访问 Hugging Face，下载缓存位于 `data/raw/`。
下载或编码失败会以非零退出码退出，完整异常写入 `data/processed/prepare.log`，不会生成替代向量。
Windows 无 symlink 权限时 Hugging Face 可能使用额外磁盘空间，不影响向量计算。

## 数据规则与输出

固定 seed=42；读取 Quora `pair/train`，去除无效文本对。完整原字符串相等才算重复，
不做 strip、小写化等隐式文本转换。唯一 anchor 首次出现顺序确定后用局部随机生成器打乱，
取 10,000 条建立历史库；记录顺序即稳定 doc_id。

只从已入库 anchor 对应的 positive 构造语义查询，排除历史库完全相同文本。
相同 positive 合并全部来源，保留原数据行号 pair_id 和 corpus_id。打乱后划分
50/100 条验证/测试语义查询；历史库另取 10/20 条不重叠 Exact 查询。
若合格数据不足，明确失败。相关配对不等于余弦阈值 Ground Truth；本阶段不计算 Recall。

模型为 `sentence-transformers/all-MiniLM-L6-v2`，CPU、4 个 Torch 线程、batch_size=64，
统一归一化并保存 float32。首次真实运行把数据集和模型的 `main` 解析为不可变 commit，写入 manifest；
当前默认配置已固定为本次实际使用的 commit，方便重建同一数据。
同一依赖、数据版本和种子下切分确定；跨硬件编码可能有浮点微小差异。

`data/processed/` 包含：

- `corpus.jsonl`、`validation_queries.jsonl`、`test_queries.jsonl`：文本、ID、查询类型和来源。
- 三份对应的 `*_embeddings.npy`：矩阵第 i 行严格对应 JSONL 第 i 条记录。
- `manifest.json`：运行配置、数据/模型 commit、环境版本、文件大小与 SHA-256。
- `statistics.json`：源数据规模、anchor 重复比例（无效行剔除后）、查询数量、编码时间与向量大小。
- `verification.json`：保存后重新加载的质量检查结果。
- `prepare.log`：真实运行与失败日志。

完整产物已存在时默认拒绝覆盖，使用 `--verify-only` 复用；重新实验可修改配置输出目录。
独立重载只读取本地文件，无需加载模型或联网。它检查文件校验和、数量、唯一性、查询隔离、
来源 ID 范围、矩阵行数/维度/dtype、有限值与单位范数（容差 1e-4）。

## 设计与复杂度

列表保持原始顺序，字典/集合完成原字符串去重和成员检查，保留来源避免配对错位。
设读取 M 个有效文本对，平均文本长度为 l，平均字典操作下数据扫描约 O(Ml)，
打乱和构建 O(M+N+Q)，内存 O(Ml+(N+Q)d)。模型编码成本由实际文本 token 长度与模型决定，
不将其简化成一次 O(d) 操作；归一化和矩阵存储为 O((N+Q)d)。

测试中的人工小向量仅用于检验校验器与文件重载，不作为实验 Embedding。
真实全量编码及新进程 `--verify-only` 均通过后，才满足 Phase 1 验收。

已生成真实产物时，测试还会从本地 Arrow 数据重建切分、核对全部语义查询来源，
并用缓存模型重新编码六条抽样文本以核对保存向量。无产物时这两项集成测试明确跳过；
正式验收要求它们通过。测试收集仅限 `tests/`，避免扫描下载缓存。
本次实测和遗留事项见 `docs/PHASE1_SUMMARY.md`。

## Phase 2：Exact Hash 与 Ground Truth

```powershell
.\.venv\Scripts\python.exe -m src.ground_truth --config configs/experiment.yaml --split validation --tau 0.75
.\.venv\Scripts\python.exe -m pytest -q
```

`tau` 必须显式指定，`0.75` 仅用于本阶段正确性验证，尚未通过验证集校准。
默认仅处理验证集。CLI 支持 `--split test`，本阶段不在测试集上执行阈值筛选或调参。
若目标结果已存在，程序拒绝覆盖；可用 `--output data/ground_truth_another_run` 保存新一次运行。
无需重新下载模型或生成 Embedding。

- `ExactHashIndex(texts).search(text)` 返回完整原字符串相等的全部历史行 ID，重复文本可对应多个 ID。
  字典键保留原文，Python 字典在哈希冲突时比较完整字符串，返回前再次核对原文。
  不做大小写、空白或 Unicode 规范化。不同字符串即使意思接近也不会命中，因此它无法处理语义近似匹配。
- `BruteForceSearch(X).scores(q)` 对每条历史向量计算真实余弦；`search(q, tau)` 返回全部 `score >= tau` 的 ID。
  按历史行 ID 升序返回，不做 top-k 截断。即使 Exact 命中，也独立计算全库 Ground Truth。
  为减少数值误差，以 float64 工作副本归一化并计算，输入文件仍保持 Phase 1 的 float32。
  极端幅度向量先按行最大绝对值缩放，再求范数；零向量、非有限数值和维度不匹配均拒绝。
  点积仅裁剪到数学范围 [-1,1]，阈值比较不加 epsilon。后续 LSH 精筛应复用相同数值定义。

在历史文本总长度 T、查询长度 l、命中数 m 下，Exact 建表平均 O(T+N)，索引引用空间 O(N)，
查询平均 O(l+ml)，字典查桶操作平均 O(1)；极端哈希冲突可退化到 O(Nl+ml)。
Brute-force 初始化 O(Nd)，工作矩阵 O(Nd)，单次查询 O(Nd)，得分和 ID 额外空间 O(N)。
批量保存 Q 条完整得分占 O(QN) 空间；这是本阶段便于核验的产物，不是检索索引。

`data/ground_truth/` 保存：

- `validation_tau_0.75.jsonl`：逐查询 ID、类型、Exact 命中、Ground Truth ID 与对应得分。
- `validation_tau_0.75_scores.npy`：60×10,000 的 float64 全库余弦得分，行序对应验证查询，列序对应历史库。
- `validation_tau_0.75_manifest.json`：临时阈值、数量、输入/输出校验和、环境、重复运行及落盘检查状态。
- `phase2.log`：实际运行与失败日志。

测试包含人为制造的哈希碰撞、相同/正交/反向向量、阈值等号、空结果、异常输入和重复查询。
产物生成后还会检查全部真实得分与独立余弦公式一致、全部 Ground Truth ID 与原始精确匹配结果正确。
真实 Phase 2 产物缺失时，该集成测试明确跳过；正式验收要求其通过。
阶段总结见 `docs/PHASE2_SUMMARY.md`。

## Phase 3：Random Hyperplane LSH

```powershell
.\.venv\Scripts\python.exe -m src.validate_lsh --config configs/experiment.yaml --k 12 --L 16 --tau 0.75
.\.venv\Scripts\python.exe -m pytest -q
```

`k=12,L=16,tau=0.75` 仅是指导文件建议的初始正确性配置，尚未调优。
校验命令仅运行 validation，不读取测试查询来选择参数；需要 Phase 2 对应阈值的真实 Ground Truth 文件。
目标产物已存在时拒绝覆盖，可通过 `--output` 指定新目录。

```python
from src.random_hyperplane_lsh import RandomHyperplaneLSH, candidate_probability
from src.retrieval import HybridRetriever

lsh = RandomHyperplaneLSH(k=12, L=16, random_state=42).fit(X)
C = lsh.candidates(q)
ids = lsh.search(q, tau=0.75)
details = lsh.search_with_stats(q, tau=0.75)
hybrid = HybridRetriever(corpus_texts, lsh)
result = hybrid.search(query_text, q, tau=0.75)
# 精确命中时允许 q=None，跳过查询编码及 LSH；此路径 scores=None。
theoretical_probability = candidate_probability(0.75, k=12, L=16)
```

每张表使用 k 个独立高斯超平面，`dot >= 0` 记为 1；bit j 权重为 `2**j`。
签名打包为 Python 整数，支持超过 64 bit。L 张表独立取样；查询始终复用已保存的超平面。
`signature -> tuple(doc_id)` 桶保持全部历史记录；候选取 L 个桶的并集，去重后按 ID 排序。
只收集候选向量进行真实余弦比较，严格筛选 `>= tau`；空候选直接返回空集，不回退到全库。
混合检索先查原文，命中时跳过向量访问和 LSH；未命中才使用上述候选流程。

单 bit 碰撞概率 `p=1-acos(s)/pi`，候选概率理论模型为 `1-(1-p**k)**L`。
该公式不是实际 Recall；参数选择与网格实验留待 Phase 4。

为让任意候选子集和全库扫描在浮点阈值边界严格一致，Phase 2/3 共享
`cosine_scores_normalized` 逐行 einsum 点积。历史 float32 产物不变，工作副本仍为 float64。
旧 Phase 2 完整得分的末位可能与当前逐行计算有微小差异；其全部验证查询 Ground Truth ID
已核验保持一致，原有得分文件和校验和保持原样。

建索引时间 O(NLkd)，额外索引空间 O(Lkd+NL)，归一化工作矩阵 O(Nd)。
设读取的桶引用数为 B、去重候选数为 c，查询签名 O(Lkd)，并集平均 O(B)，
排序 O(c log c)，精筛 O(cd)；候选向量收集临时空间 O(cd)。
`search_with_stats` 独立记录签名、查桶/去重、精筛时间与候选数、比较次数；
`fit_stats` 记录建索引与签名阶段时间，`bucket_stats()` 提供每表桶诊断。

`data/lsh_validation/` 包含：

- `hyperplanes.npy`：本次实际使用的 L×k×d 超平面。
- `tables.json`：每张表的签名桶与全部历史 ID（不使用 pickle）。
- `bucket_stats.json`：已占用桶、相对全部 `2**k` 签名的空桶比例、最大桶和 ID 引用数。
- `queries.jsonl`：全部 60 条验证查询的候选、LSH/混合结果、独立 Ground Truth 与阶段耗时/计数。
- `summary.json`、`phase3.log`：配置、真实诊断、输入/输出及代码 SHA-256、检查状态和日志。

这些时间来自单次正确性运行，不是经过热身/重复测量的性能 benchmark。
校验器中的全库扫描与重复拟合只用于验收，不属于 LSH 查询路径。
重启后集成测试会重建索引并与保存的超平面和桶逐项对照，验证每条查询满足 `R=C∩G`。
真实 Phase 3 产物缺失时这两项集成测试明确跳过；正式验收要求通过。
本次结果及边界测试修正记录见 `docs/PHASE3_SUMMARY.md`。

## Phase 4：验证集参数选择

```powershell
.\.venv\Scripts\python.exe -m src.tune_parameters --config configs/experiment.yaml --plan configs/phase4.yaml
.\.venv\Scripts\python.exe -m src.plot_validation_grid
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --ignore=tests/test_phase1_artifacts.py --basetemp=data/phase4-test-tmp
```

上述前两条命令已真实执行，默认拒绝覆盖已有选择结果。重现实验可指定新的
`--results-dir` 和 `--lock-file`；绘图补充命令只针对默认 results 生成一次额外的对数横轴视图。
先安装更新的 requirements.txt（新增 matplotlib、threadpoolctl），或使用本次 requirements-lock.txt。

参数选择脚本只打开 Phase 1 manifest、历史库和验证查询文件；只使用其中 50 条语义查询。
不调用会读取测试数据的 `verify_saved`。本阶段回归显式排除原有两项读取测试查询的 Phase 1 集成测试。
它们在此前阶段已通过，未删除或修改；本轮不使用真实测试查询进行检索、调参或其他数据检查。

预先固定的 configs/phase4.yaml 指定阈值候选、来源覆盖目标、随机参考样本数、网格和预算。
阈值选择先于 LSH 拟合：在 {0.70,0.75,0.80} 中，选择来源配对查询覆盖率至少 90%、
随机非来源参考对超阈值比例不超过 1% 的最高阈值。每查询随机采样 100 个非来源历史 ID，
合计 5,000 个参考对。它们未被人工标注为无关，不能把超阈值比例称为真实误报率。
有多个来源时按“至少一个来源达阈值”定义查询覆盖，来源对分布另报全部唯一 query/corpus 对。

固定主阈值后测试 k={8,12,16}、L={4,8,16,32}。优先在平均候选率≤5%、P95≤10%、
额外 LSH 索引估算≤32 MiB 内最大化条件 Recall，同分时选候选率更低、L 更小、k 更小者。
规则和不可行时的显式备选规则都保存在 plan，不能看结果后改变预算。
Recall 是非空 Ground Truth 查询的宏平均；空 Ground Truth 的 Recall 保存为空，查询本身保留。
候选率、耗时和比较数统计全部 50 条查询。

每配置热身 1 轮、测量 3 轮，限制底层线程数为 1。耗时包含预编码查询的完整 LSH API 调用，
不包含 Embedding、独立 Ground Truth、索引构建或内存遍历。索引内存按唯一 Python 对象递归估算，
包括超平面数组、桶字典、键、ID 元组与整数；排除历史向量工作副本，不是进程 RSS。

本次锁定于 `configs/selected_experiment.yaml`：**tau=0.75、k=12、L=32、seed=42**。
验证 Recall=0.91824495，平均候选率=1.4702%，P95=1.9385%，额外 LSH 索引估算=22.705 MiB。
理论边界候选概率=0.75827693，未达 0.90 的理论目标；该概率和实测查询 Recall 不是同一统计量。
以上均是验证集结果，测试集最终结果留待 Phase 5。

主要产物：

- `results/raw/phase4_plan.json`：实验开始时保存的策略快照。
- `threshold_analysis.json`、`threshold_candidates.csv`、`threshold_reference_pairs.csv`：阈值依据与全部抽样对。
- `theory_grid.csv`：理论候选概率、达到目标所需 L 与条件化无关候选数推导。
- `validation_grid.csv`：12 组真实 Recall、候选率、耗时、比较次数、内存估算和建表时间。
- `validation_per_query.csv`（600 行）、`validation_timings.csv`（1,800 行）：逐查询结果和全部计时样本。
- `validation_semantic_scores.npy`、`validation_ground_truth.jsonl`：同一批验证查询的全库得分和主阈值相关集合。
- `parameter_selection.json`、`phase4_manifest.json`：选择依据、环境/线程信息、输入/输出及代码校验和。
- `results/figures/recall_candidate.png`、`recall_candidate_log.png`：标注所选配置的权衡图，后者放大小候选率区域。
- `results/figures/threshold_analysis.png`：来源对与随机参考对得分分布。
- `results/raw/phase4_plot_manifest.json`：补充对数图的独立输入、输出与代码追溯。

只有配置与绘图缓存路径为输出之外的例外：锁定 YAML 位于 configs，matplotlib 缓存位于忽略的 results/.matplotlib。
运行细节、算法复杂度、全网格结果及理论与实测差异见 `docs/PHASE4_SUMMARY.md`。
