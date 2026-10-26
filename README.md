# Hash + LSH 语义检索实验

目前仅实现 Phase 1：真实数据准备、查询切分、Embedding 与落盘校验。
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
