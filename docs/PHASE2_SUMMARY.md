# Phase 2 工作总结

日期：2026-10-26（Asia/Hong_Kong）。状态：Phase 2 验收通过，未进入 Phase 3。

## 完成内容

- 新增 `src/exact_hash.py`：原字符串精确匹配，保留所有重复文本对应的历史 ID。
- 新增 `src/ground_truth.py`：全库余弦计算、包含等号的阈值筛选、独立 Ground Truth CLI 与持久化检查。
- 新增三个测试文件：`test_exact_hash.py`、`test_ground_truth.py`、`test_phase2_artifacts.py`。
- 更新 README 和 src 包说明；复用 Phase 1 产物，没有重新编码或更改原始文本/向量。
- 对全部 60 条验证查询真实扫描 10,000 条历史向量并保存结果，重复计算及保存后重载检查通过。

## 算法原理、数据结构与设计理由

### 原字符串 Exact Hash

以原字符串为 Python 字典键，值为按历史顺序保存的 ID 列表。查询返回原文完全相等的全部 ID，未命中时返回空列表。
Python 字典不会把“哈希值相等”直接视为“键相等”，发生碰撞时仍比较完整字符串；返回前还显式核对保存的原文。
不做大小写转换、空白裁剪或 Unicode 规范化。索引对输入序列建立独立元组，返回新列表，避免外部修改结果破坏内部索引。

设历史文本总长度为 T、历史数量为 N，平均建表时间 O(T+N)、索引引用空间 O(N)。
查询长度为 l、匹配数为 m 时，字典查桶平均 O(1)，包含字符串哈希与原文复核的查询成本平均 O(l+ml)，输出空间 O(m)。
极端哈希碰撞下查找可退化至 O(Nl+ml)，因此不能声称无条件常数时间。

Exact Hash 无法处理语义近似但字符串不同的查询：相似含义不会改变字符串相等规则。
本次 50 条纯语义验证查询全部没有 Exact 命中，其中 48 条在临时阈值下有真实余弦相关项，体现了该局限。

### Brute-force Ground Truth

初始化时将历史向量转为 float64 工作副本并归一化；每次查询归一化后，与全部 N 个历史向量做点积：

```text
u_i = x_i / ||x_i||
v = q / ||q||
scores[i] = clip(u_i dot v, -1, 1)
G(q) = [i for each corpus row i if scores[i] >= tau]
```

每条历史向量均参与计算，所有达标 ID 都被保留，因此输出恰为阈值定义的相关集合。
结果按历史 ID 升序返回，不按分数排序、不取 top-k。
Ground Truth 与 Exact 命中完全独立：即使原文匹配，也完整扫描全部历史向量。
当前模块不调用 LSH、不使用原始配对标签作为 Ground Truth。

使用 float64 降低累加和归一化误差；Phase 1 的 float32 文件保持原样。
对一般输入先按行最大绝对值缩放，避免范数计算在极端幅度下溢出或上溢。
零向量的余弦未定义，明确报错；非有限值、维度错误和不合法阈值同样拒绝。
只把浮点点积裁剪至数学范围 [-1,1]，阈值比较不加 epsilon，严格包含等号。
后续候选精筛需复用相同数值定义，以免阈值边界产生口径差异。

维度 d 下，初始化时间与工作矩阵空间为 O(Nd)，单次查询时间 O(Nd)，得分/结果额外空间 O(N)。
本次 float64 工作矩阵有效载荷为 N×d×8 = 30,720,000 字节（公式计算值，不是进程内存实测）。
保存 Q 条完整得分需要 O(QN) 空间，用于本阶段审计，不能计作未来的 LSH 索引内存。

## 真实运行及结果

临时阈值 tau=0.75 仅用于正确性验证，尚未校准、没有宣称最优。
默认 CLI 仅处理 validation；本阶段没有在 test split 上执行 Phase 2 余弦筛选或参数选择。
原有 Phase 1 质量检查仍检查全部落盘向量的完整性。

| 查询类型 | 查询数 | Exact 命中查询数 | Ground Truth 非空查询数 | Ground Truth 为空查询数 |
|---|---:|---:|---:|---:|
| Semantic validation | 50 | 0 | 48 | 2 |
| Exact validation | 10 | 10 | 10 | 0 |

两条空 Ground Truth 查询保留在原始 JSONL 中，没有删除或给其编造相关项。
不在本阶段生成 Recall、参数搜索曲线或检索性能 benchmark。

产物位于 `data/ground_truth/`：

| 文件 | 实际大小 | 内容 |
|---|---:|---|
| validation_tau_0.75.jsonl | 10,854 字节 | 逐查询类型、Exact ID、Ground Truth ID 及得分 |
| validation_tau_0.75_scores.npy | 4,800,128 字节 | 60×10,000 float64 全库得分 |
| validation_tau_0.75_manifest.json | 见实际文件 | 输入/输出 SHA-256、运行环境、临时阈值及核验状态 |
| phase2.log | 见实际文件 | 真实运行日志 |

JSONL 行序与验证查询一致，得分矩阵行序为 query 顺序、列序为 corpus doc_id。
Ground Truth manifest 保存 Phase 1 输入文件校验和，可追溯到当时的数据/模型版本。
目标文件已存在时拒绝覆盖；再次运行可指定新的输出目录。

## 测试与验收证据

实际主要命令：

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data --config configs/experiment.yaml --verify-only
.\.venv\Scripts\python.exe -m pytest tests/test_exact_hash.py tests/test_ground_truth.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m src.ground_truth --config configs/experiment.yaml --split validation --tau 0.75
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=data/phase2-test-tmp
```

新增核心单测输出，保存于 `docs/phase2-unit-tests.log`：

```text
29 passed in 0.38s
```

完整回归与真实集成测试输出，保存于 `docs/phase2-tests.log`：

```text
................................................                         [100%]
48 passed in 38.28s
```

覆盖以下要求：

- 相同原文、重复文本、空字符串、大小写/空白/Unicode 差异；人为制造相同哈希值但不同字符串，确保没有误匹配。
- 相同、正交、反向与斜向向量；tau=-1、0、1 和精确 0.75 阈值的相邻浮点数，检查 >= 包括等号。
- 非单位向量、极端有限幅度、空历史库、空结果、输入数组被修改及异常输入。
- 与独立逐元素余弦公式对照；相同查询重复运行返回相同 ID。
- 真实产物的全部 600,000 个得分与独立矩阵余弦公式一致（浮点容差校验）。
- 每条真实 Ground Truth ID 与独立公式阈值结果一致；每条 Exact 结果与直接原字符串扫描一致。
- 输入/输出校验和、保存后重载、全部查询重复计算及 Phase 1 回归检查通过。

首次完整回归测试因受限环境无法访问 pytest 临时目录而中断，日志保留于
`docs/phase2-tests-restricted.log`。使用所需文件权限重跑后 48 项全部通过。
核心算法单测及真实 Ground Truth 生成均在首次执行时通过，没有使用虚构输出替代错误。

## 遗留事项与下一阶段前置条件

Phase 2 无阻塞，全部验收项通过。下一阶段可复用 ExactHashIndex、BruteForceSearch 和相同的数值定义，
以独立 Ground Truth 核验候选精筛正确性。
tau 的论证与校准仍属于 Phase 4；测试集最终评价和性能/内存统计属于后续阶段。
LSH、参数搜索、benchmark 及最终报告均未实现。
