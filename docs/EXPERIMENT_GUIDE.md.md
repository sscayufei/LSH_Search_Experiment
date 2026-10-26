# 算法设计大作业 4：Hash + LSH 语义检索实验指导（Codex 协作版）

> **使用方式**：将本文件放在实验项目根目录，作为 Codex 的任务说明与人工验收清单。按阶段逐步实施；每阶段完成后查看真实输出、运行测试并决定是否进入下一阶段。
>
> **依据**：课程附件《算法设计大作业4.pdf》（共 7 页）。文中“作业要求”来自附件；具体数据集、程序结构、参数候选与实施顺序是本指导推荐的方案，不是老师指定的唯一做法。
>
> **最终交付**：一份 **3–4 页 PDF 报告**；实验代码、原始测量数据和图表作为本地可复现材料。课程要求提交一份 PDF；代码是否额外提交，以课程平台要求为准。

---

## 0. 实验目标与边界

### 0.1 研究问题

历史库包含 $N$ 条 Prompt：$q_1,\ldots,q_N$。对新查询 $q$，系统按顺序执行：

1. **Exact Match**：寻找文本完全相同的历史 Prompt；可用普通 Hash Table。
2. **Semantic Match**：未找到完全相同的 Prompt 时，检索满足 $\operatorname{sim}(q,q_i)\ge\tau$ 的历史 Prompt；不得每次都直接比较全部 $N$ 个向量，需设计基于 LSH 的候选生成。
3. **精确核验**：只对 LSH 返回的候选集合 $C(q)$ 计算真实余弦相似度，并用阈值 $\tau$ 筛选。LSH 本身不充当最终相似性判定。

整体流程：

```text
用户查询 q
  ├─ Exact Hash 命中 → 返回完全相同的历史 Prompt
  └─ 未命中 → Embedding(q)
                 → 查询 L 张 LSH 表
                 → 候选并集并去重 C(q)
                 → 对候选计算精确 Cosine Similarity
                 → 保留 sim(q, x) >= τ 的项
```

> **重要评测说明**：按作业流程，Exact Match 命中时可提前返回，但这不保证检索到“所有”余弦相似的历史文本。报告中要清楚区分“完全匹配命中率”和“语义检索 Recall”，主语义 Recall 建议统计**没有完全命中**的查询。可以另外报告混合查询的端到端结果，避免将提前返回导致的漏召回隐藏起来。

### 0.2 作业中不可遗漏的内容

- 数据集：公开文本数据集或自建 **至少 10,000 条文本**；用公开 Embedding 模型得到向量。
- 两层检索：Exact Hash + Semantic LSH；明确选择 $k,L,\tau$ 的依据。
- **至少 100 条查询**，使用暴力搜索获得 Ground Truth。
- 指标：**Recall、Candidate Ratio、Query Cost（耗时或向量比较次数）、Memory Cost**。
- 必须对比 **Brute-force Search / Exact Hash Only / Hash + LSH**。
- 分析数据量从 $10^4\to10^5\to10^6$ 时，在 $k,L$ 不变条件下的候选规模变化，并讨论改进办法。若未实际跑到大规模，要明确标为理论推导或外推，不能伪装成实测。
- 可选扩展：动态桶探测、不同表采用不同 $k$、二级 Hash、拥挤桶拆分等，但需说明改善了什么，以及对 Recall、候选量、时间、内存的影响。
- 最后思考：**不超过 300 字**；回答“为什么大规模语义检索仍是算法设计问题”。
- 无需训练/微调大模型，无需 GPU。

### 0.3 建议的工作原则

- **先做正确性，再测性能**：用小数据单测通过后再测试 10,000 条。
- **固定随机种子**（如 42），保存数据切分、配置、依赖版本和模型标识。
- **验证集调参数，测试集作最终评价**：不能看测试集结果反复改 $k,L,\tau$，然后将该测试集称作无偏最终结果。
- 实验图表、数字、结论必须来自真实程序运行；推导与预测需注明“理论值”。
- 尽量避免一次性交给 Codex 完成所有功能；每阶段要求 Codex 说明“做了什么 / 为什么 / 如何验证”。

---

## 1. 工程方案与目录

### 1.1 技术选型（推荐，不是强制）

| 模块 | 推荐 | 原因 |
|---|---|---|
| 实现语言 | Python 3.11+ | 数据处理、矩阵运算和实验绘图方便 |
| 数据集 | `sentence-transformers/quora-duplicates` 的 `pair` 配置 | 包含 `anchor` 与 `positive` 相关问题对，便于构造语义检索查询 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 输出 384 维向量，可在 CPU 上运行 |
| 向量计算 | NumPy | 实现归一化、向量点积与随机超平面 |
| Exact Hash | Python `dict` | 平均情况下近似 $O(1)$ 查找，并需再用原始文本确认相等 |
| LSH | 自己实现 Random Hyperplane LSH | 直接体现课程要求的 $k,L$ 与哈希碰撞概率 |
| 测试与绘图 | pytest、pandas、matplotlib | 检查正确性并导出测量结果 |

参考资料：
- Quora 数据集说明：https://huggingface.co/datasets/sentence-transformers/quora-duplicates
- Embedding 模型说明：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### 1.2 推荐目录结构

```text
hash-lsh-experiment/
├── EXPERIMENT_GUIDE.md              # 本文档
├── README.md                        # 项目运行说明
├── requirements.txt
├── configs/
│   └── experiment.yaml              # N、seed、模型、τ、k、L 等
├── data/
│   ├── raw/                         # 下载缓存或原始数据（可不提交）
│   └── processed/                   # 切分、文本、向量与元数据
├── src/
│   ├── prepare_data.py
│   ├── exact_hash.py
│   ├── random_hyperplane_lsh.py
│   ├── retrieval.py
│   ├── ground_truth.py
│   ├── metrics.py
│   └── benchmark.py
├── tests/
│   ├── test_exact_hash.py
│   ├── test_lsh.py
│   └── test_retrieval.py
├── results/
│   ├── raw/                         # CSV / JSON 原始结果
│   └── figures/                     # PNG/PDF 图表
└── report/
    └── report.tex                   # 或 Markdown 转 PDF，最终报告 3–4 页
```

### 1.3 依赖与运行建议

```bash
python -m venv .venv
# Linux/WSL/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy pandas matplotlib pytest pyyaml psutil datasets sentence-transformers
```

建议由 Codex 输出 `requirements.txt` 并记录可复现的依赖版本。**以下命令是拟定的统一 CLI 协议，需由 Codex 在对应阶段实现，不代表这些脚本现在已经存在**：

```bash
python -m src.prepare_data --config configs/experiment.yaml
python -m pytest -q
python -m src.benchmark --config configs/experiment.yaml --split validation
python -m src.benchmark --config configs/experiment.yaml --split test
```

---

## 2. Phase 1：数据集、查询集与 Embedding

### 目标

获得统一的历史库向量矩阵 $X\in\mathbb{R}^{N\times d}$ 和验证、测试查询向量；不同方法必须使用同一批数据。

### 建议实施

1. 从 Quora `pair` 配置读取 `anchor`、`positive`。注意该配置**本身没有去重**；按完整字符串去重后建立恰好 $N=10,000$ 条历史文本。
2. 从对应 `positive` 构建语义查询，排除与历史库文本完全相同的查询，以便确实走到 Semantic Match。
3. 固定随机种子，分割两个互不重复的集合：**验证集约 50 条语义查询**，**测试集至少 100 条语义查询**；再添加例如 10 条验证与 20 条测试的 Exact Match 查询（取自历史文本）。
4. 编码历史文本和查询文本；建议 `normalize_embeddings=True`、`float32`，并保存模型名称和向量维度。
5. 为每个语义查询记录来源配对 ID，确保不会因文本去重或配对筛选造成错位；不要把“人工相近问题对”等同于余弦阈值下的全部 Ground Truth。
6. 输出数据统计（历史库大小、验证/测试查询数量、重复文本比例、向量维度、文件大小）。

**示例配置**（只是初始建议，参数需在 Phase 4 论证和调整）：

```yaml
seed: 42
dataset:
  name: sentence-transformers/quora-duplicates
  config: pair
  corpus_size: 10000
  validation_semantic_queries: 50
  test_semantic_queries: 100
  validation_exact_queries: 10
  test_exact_queries: 20
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
  dtype: float32
retrieval:
  tau: 0.75    # 待验证集校准；非已确定的最优阈值
  k: 12        # 待理论推导与验证集确认
  L: 16        # 同上
```

### 验收

- [ ] 历史文本至少 10,000 条，索引 ID 唯一且顺序稳定。
- [ ] 验证/测试查询无重叠；语义测试查询至少 100 条。
- [ ] 历史和查询 Embedding 维度相同，数值有限且近似单位长度。
- [ ] 数据切分、文本、向量和配置信息保存到磁盘，重启后可复用。
- [ ] 记录下载/加载失败时的报错，不使用悄悄生成的随机向量顶替文本 Embedding。

### 交给 Codex 的任务提示词

```text
请阅读项目根目录的 EXPERIMENT_GUIDE.md，严格只实施 Phase 1。
建立 Python 项目与可复现的数据处理流水线，下载 Quora pair 数据，完成去重、
历史库/验证集/测试集构建、SentenceTransformer Embedding 生成与持久化。
不要实现 LSH 或 benchmark。
要求：固定 seed、输出数据质量检查、README 中记录运行命令、添加必要测试。
完成后列出：修改文件、主要设计选择、实际执行命令、真实测试输出、遗留问题。
不要虚构下载成功或模型运行结果；若当前环境无法访问 Hugging Face，请报告阻塞。
```

---

## 3. Phase 2：Exact Hash 和暴力搜索 Ground Truth

### 目标

实现两个简单可靠的基准：Exact Hash Only 与 Brute-force Search；让之后的 LSH 结果有可比较的参照。

### 核心定义

对归一化向量 $\hat q$ 和 $\hat x_i$：

$$
\operatorname{cos}(q,x_i)=\hat q^\top\hat x_i.
$$

给定阈值 $\tau$，定义 Ground Truth：

$$
G(q)=\{i\mid\hat q^\top\hat x_i\ge\tau\}.
$$

- **Brute-force**：逐个计算或矩阵化计算全部 $N$ 个内积，筛出 $G(q)$。
- **Exact Hash Only**：建 `text -> ID(s)` 映射；只返回完全相同的原始字符串；若无则空集。

**为什么 Hash 和 LSH 的“碰撞”不同？** Exact Hash 将碰撞视为需要排除的误匹配，并用原字符串核验；LSH 则利用“相近向量更可能共享哈希签名”，刻意保留有意义的碰撞，再用真实相似度过滤。

### 正确性测试

- [ ] 同一字符串可精确命中，不同字符串不得误判为相等，即使底层哈希值碰撞。
- [ ] 相同的向量在归一化后余弦接近 1；比较结果与 NumPy 参考实现一致。
- [ ] Ground Truth 只由全部历史向量、查询向量和 $\tau$ 定义，不依赖 LSH。
- [ ] 相同查询重复运行得到完全相同的 Ground Truth ID 集合。
- [ ] 使用小型人工向量（相同、正交、反向）检查边界，明确 `>= τ` 包括等号。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，只实施 Phase 2：Exact Hash 与 Brute-force Ground Truth。
请实现原始字符串精确匹配、全部向量的余弦相似度计算和按 τ 取相关集合。
使用 Phase 1 保存的数据，并提供基于小型构造样例的单元测试。
暂不实现 LSH、参数搜索或最终报告。结束时解释为什么 Exact Hash 无法处理语义相近的不同文本。
```

---

## 4. Phase 3：实现 Random Hyperplane LSH

### 目标

用 $L$ 张独立哈希表、每张表 $k$ 个随机超平面，对历史向量构建索引；查询时得到较小的候选集合 $C(q)$。

### 4.1 单个随机超平面

对向量 $x\in\mathbb R^d$，随机抽取 $r\sim\mathcal N(0,I)$，定义：

$$
h_r(x)=\begin{cases}1,&r^\top x\ge 0,\\0,&r^\top x<0.\end{cases}
$$

令 $\theta=\arccos(\operatorname{cos}(x,y))$，单个 bit 的碰撞概率为：

$$
p(x,y)=\Pr[h_r(x)=h_r(y)]=1-\frac{\theta}{\pi}.
$$

即越相似，单 bit 发生相同签名的概率越高。

### 4.2 $k$ 位签名与 $L$ 张表

- 一张表组合 $k$ 个独立哈希 bit，只有整段签名相同时才成为候选；理论碰撞概率为 $p^k$。
- 生成 $L$ 套独立超平面和 $L$ 张表；只要**任意**一张表碰撞，即加入候选集合。
- 理论被召回为候选的概率：

$$
P_{\mathrm{candidate}}(x,y)=1-(1-p(x,y)^k)^L.
$$

这是假定各表随机性独立时的理论近似/模型结论；有限随机实现与数据分布下的实际 Recall 需实测。

### 4.3 实现要求

- `fit(X)`：生成并保存超平面，计算各库向量在每张表的签名，建立 `signature -> [doc_id]` 倒排桶。
- `candidates(q)`：用**相同超平面**计算查询签名，读取 $L$ 张表对应桶，对所有 ID 求并集并去重。
- `search(q, tau)`：**只比较候选集合**中的向量余弦相似度，并筛选 `>= tau`。
- `random_state` 固定；超平面在索引与查询间复用；不能每次查询都重新生成。
- 位签名可打包为整数（注意二进制位顺序一致）；每表的桶大小、空桶比例、最大桶大小可输出用于诊断。
- **不要把暴力搜索结果混入 LSH 候选来“提高 Recall”**，也不要在未命中时暗中全库回退；这会破坏成本对照。

### 算法伪代码

```text
BuildLSH(X, k, L, seed):
    for table in 1..L:
        R[table] = k independent Gaussian vectors in R^d
        for each document i:
            bits = sign(R[table] · X[i])
            buckets[table][pack(bits)].append(i)
    return R, buckets

LSHCandidates(q, R, buckets):
    C = empty set
    for table in 1..L:
        bits = sign(R[table] · q)
        C = C union buckets[table].get(pack(bits), [])
    return C

HybridSearch(text, q_vector, exact_index, R, buckets, X, τ):
    if text exactly exists in exact_index:
        return exact_match_results
    C = LSHCandidates(q_vector, R, buckets)
    return [i in C such that cosine(q_vector, X[i]) >= τ]
```

### 验收

- [ ] 相同向量使用同一组超平面得到相同签名。
- [ ] 对每个表、每条历史记录，只归入一个签名桶；索引 ID 总数（计重复表）为 $N\times L$。
- [ ] 候选集合 ID 不重复，且是历史库 ID 的子集；在同一配置下结果可复现。
- [ ] 最终结果与 `C(q) ∩ G(q)` 一致；LSH 输出不能包含不达阈值的误报。
- [ ] 索引构建时间、签名计算时间、候选数量和精确相似度比较次数可独立记录。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，只实施 Phase 3：Random Hyperplane LSH。
实现 fit、candidates、search，严格遵守 k-bit/L-table 设计，使用固定 seed，
支持配置 k、L 和 τ。给出理论碰撞概率的辅助计算函数。
添加小样本正确性测试和复现测试；将精确核验限制在 LSH 候选集合。
不得调用 FAISS/Annoy 等库替代本实验的 LSH，不得隐式回退到全库搜索。
最后总结实现原理、时间/空间开销和单元测试结果。
```

---

## 5. Phase 4：选择 $k,L,\tau$，解释 Recall 与 Candidate Size 的权衡

### 目标

不凭“多试几次哪个好”决定参数，而是先用公式推导，再用**验证集**选择参数，并锁定参数后测试。

### 5.1 先选阈值 $\tau$

- 作业没有指定唯一 $\tau$。建议先考察验证查询与原配对历史问题的余弦得分分布，例如分位数，以及大量随机无关问题对的得分分布。
- 初始试验可从 $\tau\in\{0.70,0.75,0.80\}$ 讨论，但不是这些阈值都能保证样本有非空 Ground Truth。
- 结合“对哪些文本算足够相似”的解释，**先固定一个主阈值**；其他阈值可作为敏感性分析。注意：Quora 的“重复问题”标签和模型余弦阈值并非完全等价。
- 如果部分查询满足 $|G(q)|=0$，主 Recall 应只对 $|G(q)|>0$ 的查询取均值，并单列**非空 Ground Truth 查询数 / 比例**；不要悄悄删除样本。

### 5.2 再推导 $k,L$

先用与阈值对应的余弦相似度 $s_1=\tau$ 和一个“明显不相似”参考值 $s_0$ 来计算：

$$
p(s)=1-\frac{\arccos(s)}{\pi},\quad P(s;k,L)=1-(1-p(s)^k)^L.
$$

选择参数的解释应包括：

1. **目标召回**：希望阈值附近的真实相关向量有较高的候选命中概率，如理论目标 $P(s_1;k,L)\ge 0.9$（目标而非必达要求）。
2. **无关候选控制**：希望在 $s_0$ 处 $P(s_0;k,L)$ 较低，从而降低扫描的候选总数。
3. **索引空间与查找成本**：$L$ 增大通常让 Recall 与候选数升高，并近似线性增加索引引用数和查询查表次数。
4. **数据量约束**：即使误碰撞概率很小，乘上大 $N$ 后仍可能出现大量无关候选。

分析 $k$ 增大时 $p^k$ 下降（Recall 通常下降、候选缩小），$L$ 增大时 $1-(1-p^k)^L$ 上升（Recall 与候选通常同步上升）。

### 5.3 可执行的验证集网格

推荐先验证 $k\in\{8,12,16\}$、$L\in\{4,8,16,32\}$（这只是搜索空间，不是最优参数结论）。对每组计算：

- 真实 Recall（以验证集 Ground Truth 为基准）
- 平均 / P95 Candidate Ratio
- 平均查询耗时与向量比较次数
- 索引额外内存与构建时间

挑选满足自定内存或候选预算且 Recall 较高的配置。如果无法同时达成目标，绘制 Pareto 权衡图并解释取舍；不允许因为结果不理想而删去低分样例。最终冻结 $\tau,k,L,\mathrm{seed}$，再跑测试集。

### 验收

- [ ] 参数选择说明包含理论公式和至少一组数值推导。
- [ ] `results/raw/validation_grid.csv` 保存每组真实指标。
- [ ] 给出 Recall–Candidate Ratio 曲线/散点图，标注选定配置。
- [ ] 最终测试与参数选择分离，测试前配置锁定并保存。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，只实施 Phase 4。
用验证集先分析 τ 的选择，再结合随机超平面碰撞概率推导 k、L 的候选范围。
执行可复现的参数网格实验，输出 validation_grid.csv 和 Recall-Candidate 图。
清楚记录推导值与实测值，并给出参数选择理由；不要访问测试集来调参。
输出候选配置及其时间、候选率和额外索引内存。不要虚构任何实验值。
```

---

## 6. Phase 5：三种方法的公平对照与四项指标

### 目标

在固定测试查询与 Embedding 上比较三种方法：

1. **Brute-force Search**：全库计算余弦相似度，作为 Ground Truth 和查询成本基线。
2. **Exact Hash Only**：只对完全相同的文本返回结果；对纯语义测试通常为空。
3. **Hash + LSH**：Exact 优先，未命中时 LSH 候选 + 精确余弦复核。

### 6.1 指标定义

| 指标 | 定义 | 注意 |
|---|---|---|
| Recall | $\displaystyle\frac{|R(q)\cap G(q)|}{|G(q)|}$ | $G(q)=\varnothing$ 不纳入条件 Recall；单独统计该类样本比例 |
| Candidate Ratio | $\displaystyle\frac{|C(q)|}{N}$ | LSH 候选要在余弦精筛**之前**统计；Exact-only 可把精确命中 ID 视作候选 |
| Query Cost | 查询耗时（毫秒）+ 真实余弦比较次数 | 全库 Brute-force 为 $N$ 次；LSH 为 $|C(q)|$ 次；Exact-only 通常为 0 次向量比较 |
| Memory Cost | 额外索引大小（字节/MB） | 与共享的原始文本、Embedding 存储分开报告；LSH 包含随机超平面、桶、ID 引用 |

这里 $R(q)$ 指某方法最终返回的结果集合；对 LSH 方法 $R(q)=C(q)\cap G(q)$。对 Exact-first 混合系统，命中时 $R(q)$ 由完全一致文本构成。

**建议另外记录**：平均值、P50/P95 查询耗时；平均候选数量；索引构建耗时；纯语义查询集的 Recall 与 Exact 命中查询的结果。表中“耗时”须明确包含或不包含什么。

### 6.2 性能测量规范

- 准备和编码时长**单独测量**。主算法比较采用预先生成的查询 Embedding：这样比较的是检索策略，而不是不同模型调用时延。
- 可额外给出端到端耗时（文本输入→编码→查找），但需要统一测量范围；Exact-first 系统可跳过编码是其合理优势，需要明确说明。
- 暴力搜索建议使用 NumPy 矩阵运算；如果它在 10k 规模的实际时延优于 Python 层实现的 LSH，**如实报告**并分析 Python 循环/内存访问等因素。
- 每个查询多次运行，热身后测量，报告统计量；固定运行机器与线程数，避免下载模型或准备数据混入耗时。
- 建议用 `time.perf_counter()` 记录时延；用 `tracemalloc` / 对象大小估算工具或一致的序列化统计口径估计索引空间，并清楚写明方法与局限。
- 将 shared corpus/embedding 和 additional index memory 分开，避免一方法仅因重复计量原始向量而显得更费内存。

### 6.3 结果结构与图表

期望自动生成以下文件（字段可由 Codex 根据实现扩展）：

```text
results/raw/
├── test_queries.csv          # query_id, kind, |G|, exact_hit
├── test_per_query.csv        # method, query_id, |C|, |R|, recall, time_ms, comparisons
├── method_summary.csv        # 各方法 Recall、Candidate Ratio、耗时、内存等
├── validation_grid.csv       # Phase 4 结果
└── environment.json         # 环境、模型、seed、参数和采样设置
results/figures/
├── recall_candidate.png
├── method_comparison.png
└── scale_analysis.png
```

**结果表模板（仅表头；禁止填入臆造数值）**：

| 方法 | Semantic Recall | Candidate Ratio | 平均耗时(ms) | 向量比较次数 | 额外索引(MB) |
|---|---:|---:|---:|---:|---:|
| Brute-force | 待测 | 待测 | 待测 | 待测 | 待测 |
| Exact Hash Only | 待测 | 待测 | 待测 | 待测 | 待测 |
| Hash + LSH | 待测 | 待测 | 待测 | 待测 | 待测 |

其中 Brute-force 的理论候选检查比例可按 $100\%$ 计；语义非精确查询的 Exact Hash Only 预期 Recall 为 0，但仍应据真实查询集计算。

### 验收

- [ ] 对至少 100 条测试查询真实计算 Ground Truth，并完成三法对照。
- [ ] 每个原始查询结果都能追溯到最终汇总；保存原始数据，不只保存图片。
- [ ] 指标计算函数有人工构造的集合运算测试（尤其是 Recall 分母与空集处理）。
- [ ] 表格区分理论可确定的比较次数与实测时间。
- [ ] 若 LSH 查询时间没有胜过 Brute-force，报告要说明原因，而不是修改测量结果。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，只实施 Phase 5 的真实 benchmark。
冻结已选 τ、k、L 和随机 seed，在未用于调参的测试集运行三种方法。
严格计算 Recall、Candidate Ratio、Query Cost、Memory Cost，
同时保存 per-query、summary、环境信息与图表。
注意纯语义查询/完全匹配查询要分开统计，所有时间测量口径一致。
请执行测试和 benchmark，粘贴实际日志及文件路径；失败必须如实报告。
```

---

## 7. Phase 6：数据规模分析与可选算法改进

### 目标

回答作业的重要追问：$N$ 增大而 $k,L$ 不变时，候选规模会怎样？小数据表现良好的参数是否仍然合理？

### 7.1 理论分析

把一条不相关历史向量误加入候选的概率记为 $P_{\mathrm{false}}$，则无关候选的期望数量近似为：

$$
\mathbb{E}[F]\approx N\cdot P_{\mathrm{false}}.
$$

在向量相似性分布和碰撞概率近似不变时，若 $k,L$ 不变，候选**数量**可能随 $N$ 近似线性增加；候选**比例**可能近似稳定。举例：若 $P_{\mathrm{false}}=0.001$，则 $N=10^4,10^5,10^6$ 分别约有 $10,100,1000$ 个误候选。这是**条件化的理论示例，不是实测值**。

现实数据集中相似向量密度、桶拥挤程度也可能随规模改变，因此需要检查候选数分布及大桶，而不能只照搬线性结论。

### 7.2 建议实验

- **必做**：理论推导 $10^4\to10^5\to10^6$，说明 $k,L$ 固定时的趋势。
- **推荐**：若资源充足，构建 $N=10^4, 5\times10^4, 10^5$ 的真实子集，保持相同向量模型、$k,L,\tau$ 和可比的查询集，绘制候选数/查询耗时/索引内存随 $N$ 的变化。
- **可选**：对 $N=10^6$ 实测。仅原始 `float32` 的 384 维向量就需要约 **1.536 GB（十进制 GB）**，再加上 Python 数据结构与多个哈希表，机器资源可能不足；不能强行要求，若未做应标注外推。
- 子集采样应嵌套或规则一致，记录抽样方式，防止不同数据分布造成错误的规模结论。

### 7.3 可选改进（不是必选）

选一个小改进即可，避免喧宾夺主：

**方案 A：大桶截断/二次分裂**
- 目标：减少高密度桶带来的候选爆炸和长尾查询时间。
- 风险：如果简单截断，可能损失 Recall；二次分裂还会增加索引与查找成本。

**方案 B：多探测 LSH（Multi-probe）**
- 除精确相同签名的桶外，再探测少量接近的签名桶。
- 目标：在 $L$ 较小的情况下改善 Recall。
- 代价：探测更多桶，Candidate Ratio 与 Query Cost 可能上升。

**方案 C：按桶密度调整参数**
- 目标：降低大量无关候选；需解释如何保证可复现以及查询端能找到对应索引。
- 代价：增加实现复杂度与维护开销。

任何改进必须与**未改进的同一基线**比较，并说明收益及代价；没有测量依据时只写理论分析。

### 验收

- [ ] 清楚区分理论分析、真实测量和外推预测。
- [ ] 解释固定 $k,L$ 时 false candidates 为什么可能与 $N$ 近似成正比。
- [ ] 若加入改进方法，展示同一查询集上的 Recall / Candidate Ratio / Query Cost / Memory 对照。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，只实施 Phase 6：规模分析。
在资源许可时实际运行不同 N 的嵌套子集并绘图；
资源不足时保留 10k 的真实结果，并单独给出大规模条件化理论预测，
不得把推算标为实验数据。
解释在 k、L 固定情况下无关候选数增长的原因。
如提出可选改进，先说明其作用和风险，经人工选择后再实现。
```

---

## 8. Phase 7：整理提交报告（3–4 页 PDF）

### 推荐页码分配

| 页码 | 内容 | 报告重点 |
|---|---|---|
| 第 1 页 | 问题、数据集、Embedding、Exact Hash 与 LSH 设计 | 为什么需要两层检索；为什么两种 Hash 对“碰撞”的态度不同 |
| 第 2 页 | 伪代码、参数 $k,L,\tau$ 依据、评测定义 | 碰撞概率公式、Recall 与 Candidate Ratio 的理论权衡 |
| 第 3 页 | 四项指标、三种方法对照、主要图表 | 用真实测试数据评价方法，说明计算/存储代价 |
| 第 4 页 | 参数敏感性、规模分析、结论与最后思考 | 解释 $N$ 放大时的问题与可能的改进，最后思考 ≤300 字 |

**必须涵盖课程提交清单**：问题与数据集；Exact Hash/LSH 设计；参数依据；核心算法伪代码；四个指标实验结果；与 Brute-force 和 Exact Hash 比较；参数和规模关系分析；最后思考。

### 报告撰写要求

- 推荐黑白、干净简洁的实验报告排版；图表轴标签和表头注明单位。
- 删去与课程重点无关的冗长环境安装过程；保留模型、数据规模、参数、测试环境等可复现信息。
- 至少有一张 Recall–Candidate Ratio 图、一张三方法对比表、一段规模分析。
- 关键伪代码体现 **两层检索、候选去重、余弦精筛**。
- 准确区分候选召回与最终 Recall；如展示语义查询的平均 Recall，应写明 $|G(q)|=0$ 的处理方式。
- 没有测出来的数字不写成“实验发现”；观察与推测分别表述。
- 最后思考答题方向：Embedding 只定义向量表示及相似度，不解决在海量数据中如何高效检索；索引结构、候选空间、参数与空间-时间-召回权衡属于算法设计。

### 交给 Codex 的任务提示词

```text
请阅读 EXPERIMENT_GUIDE.md，执行 Phase 7：撰写最终 3–4 页 PDF 实验报告。
只使用 results/raw/ 的真实实验结果、实际生成的图表和明确标记的理论分析。
报告必须覆盖课程 8 项提交要求，最后思考不超过 300 中文字。
采用简洁黑白学术排版，不使用花哨装饰、页眉页脚（可保留必要页码）。
优先生成可复现的 LaTeX/Markdown 源文件，再编译 PDF。
检查页数、公式可读性、图表清晰度、文本溢出和所有数值来源。
未运行的实验不得编造测量数据；如果必要材料不足，请列出缺项。
```

---

## 9. 每阶段的 Codex 协作约定

为了让 Codex 辅助开发，又不丢失对算法的理解，推荐在每阶段开始时附上以下通用约束：

```text
本项目是算法设计课程实验。请按 EXPERIMENT_GUIDE.md 逐阶段开发。
1. 只做当前阶段，不提前批量实现后续阶段。
2. 任何核心算法都要解释数据结构、复杂度、正确性及设计理由。
3. 新增代码必须包含边界测试与复现手段，固定随机 seed。
4. 运行可行的单测/程序，保存真实输出；不允许伪造实验结果。
5. 若依赖下载、计算资源或环境配置阻塞，说明原因及最小解决路径。
6. 完工时以“本阶段做了什么 / 为什么如此设计 / 如何验证 / 下一阶段前置条件”收尾。
7. 不擅自改变课程要求的指标定义、数据集规模或提交格式。
```

可在每阶段结束后进行简短人工 Review：

1. **算法解释检查**：能否不用看代码，清楚说出本阶段解决了什么问题？
2. **正确性检查**：哪些输入、边界、随机情况已被测试？
3. **实证检查**：日志、原始 CSV、图表之间是否对应？
4. **范围检查**：是否提前引入与本实验无关的复杂架构？
5. **进入下一阶段**：仅当前阶段验收通过才继续。

---

## 10. 最终完成标准

- [ ] `N >= 10000`；使用真实文本 Embedding，且数据处理可复现。
- [ ] Exact Hash、Brute-force、LSH 均由可运行代码实现，具备必要测试。
- [ ] LSH 由 $k$ 个随机超平面组成签名，使用 $L$ 张独立哈希表。
- [ ] 至少 100 条测试查询通过暴力搜索计算 Ground Truth。
- [ ] Recall、Candidate Ratio、Query Cost、Memory Cost 定义统一并有真实结果。
- [ ] $k,L,\tau$ 经过理论解释和验证集选择，不靠测试集临时调优。
- [ ] 完成三方法比较；区分 Exact-only 在语义查询中的局限。
- [ ] 讨论 $N=10^4\to10^5\to10^6$；理论与实测表述严格分开。
- [ ] 输出 3–4 页 PDF，包含所有课程指定内容，最后思考不超过 300 字。
- [ ] 保留实验代码、配置、原始结果和报告源文件，便于复查。

**建议开始顺序**：先把本文件保存为 `EXPERIMENT_GUIDE.md` → 把第 9 节通用约束与 Phase 1 提示词交给 Codex → 看它产生的数据与日志 → 检查 Phase 1 验收项 → 再进入 Phase 2。
