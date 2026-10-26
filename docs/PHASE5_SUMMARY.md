# Phase 5 工作总结

日期：2026-10-26（Asia/Hong_Kong）。状态：Phase 5 验收通过，未进入 Phase 6。

## 完成内容

- 新增 src/benchmark.py：核验参数冻结、生成测试 Ground Truth、交错热身/计时、保存逐查询结果和环境追溯。
- 新增 src/benchmark_metrics.py：四项指标、查询分组汇总和空 Ground Truth 处理。
- 新增 src/plot_benchmark.py：根据真实汇总生成方法对比和查询分组图。
- 新增 configs/benchmark.yaml 和单元/真实产物测试；更新 README。
- 保持 Phase 1–4 数据、核心算法和选定参数原样，未重新选参或训练模型。
- 在 10,000 条历史向量上，对 120 条测试查询真实计算全库 Ground Truth，再比较三种方法。

## 参数冻结与输入

沿用 configs/selected_experiment.yaml：tau=0.75、k=12、L=32、seed=42。
启动时比较 Phase 4 manifest 保存的锁定配置哈希，并核对实际检索参数与 Phase 4 选择项。
读取测试数据前保存 benchmark_plan.json；结束后再次验证锁定配置哈希不变。
模型/数据 commit 与 Phase 1 Embedding manifest 一致，Phase 1 数据完整性复核通过。
没有测试集参数搜索、样本筛选或根据测试结果调整 k/L/tau。

测试查询包含 100 条纯语义查询和 20 条 Exact 查询。
语义查询 95 条 Ground Truth 非空、5 条为空；Exact 查询 20 条全部非空。
全部 120 条保留于原始结果，空 G 的 Recall 留空，不记为 0 或 1；成本统计仍包含这些查询。

## 三种方法与指标设计

1. Brute-force：查询归一化后，与全部 N 条历史向量计算真实余弦，返回所有 >=tau 的项。
2. Exact Hash Only：按原字符串匹配，只返回相同文本 ID，无向量比较。
3. Hash + LSH：原文命中时立即返回，否则读取 L 张表对应桶、去重候选并仅在候选内计算余弦。

三法使用同样的测试文本/预编码向量和相同阈值。全库 Ground Truth 在计时前独立生成，
LSH 查询不会读取 Ground Truth，也没有空候选时的全库回退。
余弦计算复用前序阶段的 float64 归一化与逐行 einsum 定义，比较包含等号、不加 epsilon。

Recall 按集合 |R∩G|/|G| 定义，重复 ID 不重复计数；只在 G 非空时定义。
主要 Semantic Recall 是 95 条非空语义查询的宏平均，不把 20 条原文命中查询混入。
Candidate Ratio 记录精筛前候选数/N：Brute-force 为全库，Exact-only 为精确命中 ID，混合方法按实际路径计算。
余弦向量比较次数分别为 N、0、|C|（混合 Exact 路径为 0）。
次数来自实际候选数及算法确定的扫描范围，不是 CPU 指令计数；CSV 的 count_basis 明确标识来源。

## 公平计时与内存口径

配置：单线程、热身 2 轮、测量 5 轮；固定 seed 打乱查询顺序与每条查询的方法调用顺序。
记录所有实际调用序号，测试可以重建这一调度，不根据时延重新排序或选择样本。
API 外部 perf_counter 计时，包含各自 API 的归一化、查询、结果构造和固有统计开销；
原生结果转储/核验在停止计时后进行。三者使用相同的两参数回调形式，回调开销包含在内。
数据加载、编码、Ground Truth、建索引、内存遍历、绘图均不混入查询耗时。
本轮复用 Phase 1 Embedding，没有重新编码；当时的准备/编码时间可在 data/processed/statistics.json 查看，未冒充为本轮测量。

每查询 CSV 保存五次测量的均值、P50、P95；方法/分组汇总的 P50/P95 使用全部原始重复计时。
这一口径与 Phase 4 的“查询均值分位数”不同，已在 environment.json 明确记录。
主要对比是检索策略，不包含文本编码的端到端延迟。

额外索引统一按唯一可达 Python 对象递归 sys.getsizeof 估算：

- Exact：原文引用元组、字典、ID 列表/整数；原始字符串对象作为共享输入排除。
- LSH：随机超平面及缓冲、桶列表/字典、签名键、ID 元组/整数。
- 混合：一次遍历 Exact+LSH 对象图，共享对象不重复计数。
- Brute-force：没有额外检索索引，计 0；归一化工作矩阵单列，不能理解为总内存为 0。

原始历史向量有效载荷 15,360,000 字节，测试查询向量 184,320 字节。
Brute-force 和混合方法各有 30,720,000 字节 float64 归一化历史工作矩阵；
本次同进程比较的两份工作矩阵实际合计 61,440,000 字节，未隐藏在索引比较中。
Ground Truth 得分矩阵及验证脚手架不计入检索索引。该空间口径不是 RSS、峰值内存或分配器完整占用。

## 真实测试结果

所有数字来自 results/raw/method_summary.csv、test_per_query.csv 和 environment.json。
下表语义查询成本统计全部 100 条，Recall 统计其中 95 条非空 G 查询。

| 方法 | Semantic Recall | 平均候选率 | 平均耗时 ms | P50 ms | P95 ms | 平均向量比较数 | 额外索引 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brute-force | 1.000000 | 100.0000% | 1.989598 | 1.883800 | 2.699180 | 10,000 | 0 |
| Exact Hash Only | 0.000000 | 0.0000% | 0.002492 | 0.002350 | 0.003400 | 0 | 1.380592 |
| Hash + LSH | 0.967665 | 1.5307% | 0.545120 | 0.487350 | 0.798845 | 153.07 | 24.079079 |

Hash + LSH 的语义查询候选率 P95 为 2.2005%。
本机这次测量，语义查询均值相对 Brute-force 加速约 3.649836 倍，平均向量比较数减少 98.4693%。
候选规模压缩比与时延加速比不同：还需计算随机投影、执行 Python 查桶/集合/排序和收集候选矩阵，
因此比较次数的大幅下降不会等比例转化为时延下降。没有据此宣称跨规模或跨硬件保证加速。
Exact-only 很快，但对不同原文的语义查询没有召回，不能仅凭时间称其更优。

### Exact 查询与混合查询的区别

| 方法 | 20 条 Exact 查询 Recall | Exact 均值 ms | 120 条混合查询 Recall | 混合均值 ms |
|---|---:|---:|---:|---:|
| Brute-force | 1.000000 | 2.036013 | 1.000000 | 1.997334 |
| Exact Hash Only | 0.820833 | 0.003158 | 0.142754 | 0.002603 |
| Hash + LSH | 0.820833 | 0.034205 | 0.942129 | 0.459967 |

20 条 Exact 查询的原文命中率是 100%，但原文相同不表示 G 只含这一条历史文本。
例如 test-0107 的 G 有 4 个 ID，Exact 提前返回只保留原文一项，Recall 为 1/4。
因此混合方法在该组的 Recall 为 82.0833%，不能以原文命中率替代或掩盖漏召回。
混合系统全体查询的 Recall 用 115 条非空 G 查询宏平均，为 94.2129%，并非语义 Recall 的另一个名称。

混合语义路径平均比较 153.07 项；Exact 路径候选为 1、比较数为 0。
因此 120 条混合查询平均候选数为 127.725，平均比较数为 127.558333，两者不强行写成相等。

### 空间与构建成本

| 方法 | 额外索引字节 | 构建/准备秒 |
|---|---:|---:|
| Brute-force | 0 | 0.041626 |
| Exact Hash Only | 1,447,656 | 0.002240 |
| Hash + LSH | 25,248,744 | 0.594091 |

Brute-force 的准备时间含归一化工作矩阵；混合构建含 LSH 工作矩阵、随机超平面/桶和 Exact 索引。
LSH 组件本身为 23,808,284 字节，与 Phase 4 相同；本轮 24.079 MiB 是包含 Exact 的完整混合索引，
不能直接拿它与 Phase 4 的 LSH-only 22.705 MiB 当作同口径性能变化。
构建时长为本次一次真实测量，未伪装成多次平均。

## 算法、数据结构与复杂度

没有改变 Phase 2/3 核心算法。新增评测用查询 ID/方法键追溯结果，用集合交集计算 Recall，
用列表保留全部重复计时样本，CSV/JSONL 存储可逐项重算的数据。

- 单查询 Brute-force 仍为 O(Nd)，Exact 仍是字符串字典平均查桶加原文复核。
- 混合语义路径仍为 O(Lkd+B+c log c+cd)，Exact 路径跳过向量和 LSH。
- Ground Truth 总时间 O(QNd)，保存完整得分 O(QN) 空间。
- 指标交集平均 O(|R|+|G|)，分位数由 NumPy 对真实计时样本计算。
- 为便于审计，当前 Python 汇总代码逐方法/查询筛选计时记录，最坏可达 O(M²Q²R)；本次 M=3、Q=120、R=5，规模固定且位于计时区间外。
- 内存遍历按对象图平均线性，查询结果转储与汇总均不属于检索计时。

## 原始材料与图表

results/raw 新增：

- benchmark_plan.json：读测试数据前的参数/配置哈希和测量设置快照。
- test_queries.csv：120 行，类型、|G|、原文命中与 ID。
- test_ground_truth.jsonl、test_scores.npy：120 条独立相关集合及 120×10,000 float64 完整得分。
- test_results.jsonl：360 条方法/查询结果，保留返回 ID 和候选；Brute-force 以 all_corpus 标明全部历史库，避免重复写 10,000 个 ID。
- test_per_query.csv：360 行，包含 Recall、候选率、比较次数、时间均值/P50/P95 和次数来源。
- test_timings.csv：1,800 行，五轮每方法每查询的实际调用顺序与原始时间。
- method_summary.csv：三方法×语义/Exact/混合，共 9 行汇总。
- environment.json：参数快照、前后配置哈希、真实依赖/平台/线程、共享存储和空间/耗时/次数定义。
- phase5_manifest.json、phase5.log：输入/输出及实现代码校验和与真实执行日志。

图表 results/figures/method_comparison.png 展示语义查询四项指标，
query_group_comparison.png 展示分组 Recall 与耗时；均经视觉检查，单位和对数时间轴已标明。
Phase 4 的验证网格和图表没有覆盖或改写。

## 实际执行命令和测试结果

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.benchmark --config configs/selected_experiment.yaml --split test
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark.py tests/test_phase5_artifacts.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=data/phase5-test-tmp
.\.venv\Scripts\python.exe -m pip check
```

核心单测首次输出：5 passed in 0.29s。
专项单测和真实产物审计：8 passed in 2.53s，完整输出见 docs/phase5-unit-tests.log。
全部回归：88 passed in 38.29s，命令与真实输出摘要见 docs/phase5-tests.log。
本阶段已授权测试集评测，因此恢复执行之前 Phase 4 为隔离测试数据而未收集的两项 Phase 1 集成测试；
本次全部 88 项通过，无跳过项。pip check 返回 No broken requirements found。

新增验证覆盖：

- 人工集合 Recall 的分母、重复 ID、空 G、成本统计保留空 G 查询。
- 语义/Exact/混合分组与原始计时分位数；原文命中但 G 有多个相关项的 Recall。
- 原文字串内存作为共享输入排除；混合对象图共享引用去重。
- 冻结配置校验和及参数改变时拒绝运行。
- 全部 1,200,000 个真实得分与独立矩阵余弦公式一致（浮点容差）；全部 120 条 G 的 ID 集精确一致。
- 全部 360 条方法结果、候选和比较计数与独立复核一致；LSH 结果严格等于 C∩G。
- 逐查询指标、五轮样本及全部 9 行汇总可重算；完整随机调用调度可由 seed 重建。
- 全部输入/输出及实现代码哈希、锁定配置前后一致、共享空间与额外索引计量通过检查。

真实 benchmark 首次执行成功，没有失败后替换数据或反复测量挑选更好的一次。
完整回归使用已确认所需的 pytest 临时目录权限；没有新增下载或环境阻塞。

## 遗留事项与验收

至少 100 条测试查询的真实 Ground Truth、三法对照、四项指标、原始数据追溯、集合指标测试和计量口径均已满足。
本次结果的时间和内存是本机当前实现/解释器下的测量/估算，不是跨机器常数或操作系统峰值。
没有端到端重新编码计时，不把预编码检索结果称为文本输入到输出总时延。
Phase 5 无阻塞。数据规模分析与可选改进属于 Phase 6，本轮未提前实现。
