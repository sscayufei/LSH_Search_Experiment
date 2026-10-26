# Phase 3 工作总结

日期：2026-10-26（Asia/Hong_Kong）。状态：Phase 3 验收通过，未进入 Phase 4。

## 完成内容

- `src/random_hyperplane_lsh.py`：实现 fit、signatures、candidates、search、search_with_stats、桶诊断与理论候选概率。
- `src/retrieval.py`：实现 Exact 优先、未命中再走 LSH 的混合检索。
- `src/validate_lsh.py`：单配置真实验证 CLI，保存超平面、全部桶、逐查询结果、阶段耗时与计数、版本和校验和。
- `src/ground_truth.py`：提取共享的逐行余弦函数，使候选子集与全库扫描的点积归约保持一致。
- 新增 LSH、混合检索与真实产物测试，更新 README 与包说明。
- 没有实现参数网格、最终 benchmark、规模实验或最终报告。

## 原理、正确性与数据结构

对每张表独立抽取 k 个 `N(0,I)` 高斯向量，保存为超平面。归一化输入后，
`r dot x >= 0` 记为 bit 1，否则为 0。bit j 的权重为 2^j，打包为 Python 整数，避免 64 bit 上限。
每张表建立 `signature -> tuple(doc_id)` 字典，每条历史记录在每张表只出现一次。
固定 seed=42，查询期间不重新取样；fit 再次使用相同 seed 时可重建相同超平面与桶。

查询计算 L 个签名，读取对应桶，使用 set 求并集，去重并按 ID 排序得到 C(q)。
精筛时先索引收集 `X[C]`，仅计算这些行的真实余弦，再保留 `>= tau` 的 ID。
空候选直接返回空结果，不调用余弦函数；没有 Ground Truth 注入、隐式全库回退或外部 ANN 库。
因此对当前数值定义，最终结果严格为 `R(q)=C(q)∩G(q)`；候选可以包含不相关项，但精筛结果不能包含误报。

为消除全库和不同大小候选矩阵可能选择不同 BLAS 内核产生的阈值末位差异，
基准与 LSH 共享 float64 逐行 einsum 点积，裁剪至 [-1,1]，阈值比较不加 epsilon。
原 Phase 1/2 文件未改写。旧 Phase 2 完整得分与当前计算允许极小末位差异；
本次全部 60 条验证查询的阈值 ID 集与已保存 Ground Truth 完全一致，独立公式测试也通过。

混合检索使用 Phase 2 原字符串索引。Exact 命中时立即返回原文 ID，不读取查询向量、
不执行 LSH 和余弦比较，scores 设为 None；该路径不保证返回所有余弦相近文本。
语义路径则完整执行候选生成与精筛，不额外使用全库结果。

## 理论碰撞概率

夹角 theta=acos(s)，单 bit 碰撞概率为 `p=1-theta/pi`。
同表 k bit 全相同的概率为 p^k，L 张独立表至少一表命中的概率为：

```text
P_candidate(s;k,L) = 1 - (1 - p(s)^k)^L
```

辅助函数使用 log1p/expm1 降低小概率情况下的消减误差。
相同、反向和正交向量的端点值及 k/L 单调关系已测试。
当前 `s=0.75,k=12,L=16` 对应理论概率约 0.5083465951，属于理论模型值，不能称为实测 Recall。
本轮仅采用指导文件初始配置验证正确性，没有据此调整或选择参数。

## 时间与空间复杂度

设 N 条历史向量，维度 d，L 张表、每张 k bit；一次查询读取的桶引用数 B，去重候选数 c。

| 操作 | 开销 |
|---|---|
| 归一化工作矩阵 | 时间/空间 O(Nd) |
| 建索引 | 时间 O(NLkd)，含签名与分桶 |
| 额外索引 | 超平面 O(Lkd)，桶及历史 ID 引用 O(NL) |
| 建表临时投影 | 每次处理一张表，O(Nk) |
| 查询签名与查表 | O(Lkd+L) |
| 候选并集 | 平均 O(B)，重复 ID 仍需被访问 |
| 稳定 ID 排序 | O(c log c) |
| 候选余弦精筛 | O(cd)，实际比较 c 个向量 |
| 候选向量收集 | 临时空间 O(cd)，另有候选集合/得分 O(c) |

字典/集合操作采用平均复杂度，桶列表可以拥挤，不能无条件宣称次线性或比 NumPy 暴力扫描更快。
归一化历史向量是工作数据，与随机超平面/桶索引分开理解；本阶段没有测量进程或额外索引内存。

## 真实运行结果

使用已落盘的 10,000×384 历史向量与全部 60 条验证查询，配置 `k=12,L=16,tau=0.75,seed=42`。
这些参数未校准，测试集未用于本阶段 LSH 验证或参数选择。

以下数字来自 `data/lsh_validation/summary.json` 和逐查询记录：

| 检查/测量 | 实际结果 |
|---|---:|
| 独立 LSH 验证查询 | 60 |
| 每张表历史 ID 数 | 10,000 |
| 全部表 ID 引用数 | 160,000 = N×L |
| 结果满足 C∩G | 60 / 60 |
| 去重候选总数（跨查询求和） | 4,387 |
| 独立 LSH 余弦比较总数 | 4,387 |
| 每条查询候选数范围 | 35–118 |
| LSH 最终返回 ID 总数（跨查询求和） | 97 |
| 混合 Exact / LSH 路径查询 | 10 / 50 |
| 混合 Exact 路径余弦比较次数 | 0 |
| 混合语义路径余弦比较总数 | 3,734 |
| 首次索引构建 | 0.276254 秒 |
| 其中历史签名计算与打包 | 0.189293 秒 |
| 其中建桶 | 0.033447 秒 |
| 其中归一化 | 0.035550 秒 |

时间由 perf_counter 真实记录，属于单次诊断运行，不是热身/重复后的性能比较。
总建表时间还含随机超平面生成与其他开销。逐查询签名、桶读取/去重排序、候选精筛耗时分别保存，
不包含验证器事后执行的全库核验、重复查询和重复拟合。
4,387 次为对 60 条查询独立运行 LSH 的总比较数；3,734 次为混合系统仅 50 条语义查询的比较数，两者口径不同。

第 0 张表示例：占用 3,100 / 4,096 个可能签名桶，空桶比例 0.2431640625，最大桶 25 条。
全部 16 张表的对应诊断保存在 bucket_stats.json；空桶比例分母为 2^k，存储时只创建非空桶。

## 产物与复现

`data/lsh_validation/` 保存：

- hyperplanes.npy：实际超平面，589,952 字节。
- tables.json：全部签名桶与历史 ID，3,141,334 字节。
- bucket_stats.json：每表诊断，3,866 字节。
- queries.jsonl：候选、LSH/混合结果、独立 Ground Truth、阶段计时和计数，73,686 字节。
- summary.json：配置、版本、检查结果、输入/输出及实现代码 SHA-256。
- phase3.log：真实运行日志。

文件大小只是落盘大小，不作为 Python 对象内存实测。
保存后重载超平面、桶与结果通过；独立进程集成测试再次拟合并逐项核对。
验证器全库扫描仅在 LSH 返回后验收结果，LSH 核心路径不访问 Ground Truth。

实际主要命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_lsh.py tests/test_retrieval.py tests/test_ground_truth.py tests/test_phase2_artifacts.py tests/test_phase3_artifacts.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.validate_lsh --config configs/experiment.yaml --k 12 --L 16 --tau 0.75
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=data/phase3-test-tmp
```

## 测试结果与问题处理

专项测试真实输出，见 docs/phase3-unit-tests.log：

```text
48 passed in 1.72s
```

完整回归真实输出摘要，见 docs/phase3-tests.log：

```text
70 passed in 12.56s
```

覆盖签名位序（含 70 bit）、同向量同签名、精确零投影记为 1、每表 ID 完整性、
候选去重、查询不取样、seed 复现、输入修改隔离、候选内比较计数、空候选不比较/不回退、
阈值等号与任意真实得分边界、异常输入、Exact 路径跳过 LSH、语义路由、
全部真实查询 C∩G、持久化及 Phase 1/2 回归。
候选精筛测试还监视实际点积输入矩阵行数，确保只处理 c 行而非 N 行。

首轮测试 43 通过、1 失败：测试把浮点随机超平面的理论正交向量误当作归一化后精确零投影。
归一化可能使其点积成为极小负值，因此该测试不能判断 `>=0` 的精确零边界。
修正为坐标轴超平面和坐标轴输入，使零投影可精确表示；未改变算法的符号判定，也未引入 epsilon。
原始失败日志保留在 docs/phase3-unit-tests-initial.log。
完整回归使用此前已确认所需的文件权限，避免 pytest 临时目录访问限制；全部测试通过，无跳过项。

## 遗留事项

Phase 3 无阻塞。k、L、tau 尚未选择或冻结；理论概率不是实际 Recall。
下一阶段再开展阈值分析、公式推导和验证集参数搜索。
当前单次时间不能用于宣称 LSH 比 Brute-force 快，正式公平性能评测属于 Phase 5。
