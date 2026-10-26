# Phase 1 工作总结

日期：2026-10-26（Asia/Hong_Kong）。状态：Phase 1 验收通过；未进入后续阶段。

## 完成内容

- 复用 Python 3.12.6 虚拟环境，安装必要依赖，生成完整版本清单 `requirements-lock.txt`。
- 新增 `configs/experiment.yaml`、`src/prepare_data.py`、单元/集成测试、`pytest.ini`、README 与忽略规则。
- 下载真实 Quora pair/train 和公开 MiniLM 模型，构建并编码历史库及验证/测试查询，保存文本、来源、向量、环境信息、统计、校验和与日志。
- 独立新进程重载通过；真实原始数据复现与缓存模型抽样重编码通过。

## 设计说明

完整原字符串决定文本相等，不对文本做小写化或空白裁剪；只用 `strip()` 判断空白无效输入。
有序列表维持稳定文本/向量行序，字典与集合处理去重和成员判断。
固定局部随机生成器 seed=42，先抽取 10,000 个唯一 anchor 建立历史库，再从其 positive 构建语义查询。
相同 positive 合并所有来源配对，排除历史库精确重合文本，验证/测试查询文本不重叠。
Exact 查询从历史库中无放回另抽取。语义配对不作为完整阈值 Ground Truth。

读取 M 对文本、平均长度 l 时，字典平均操作下扫描与去重约 O(Ml)，抽样和组织记录约 O(M+N+Q)。
保留原始配对与来源需 O(Ml) 内存，输出向量需 O((N+Q)d)；归一化为 O((N+Q)d)。
Transformer 编码成本受 token 长度、模型结构与批次影响，单独实测，不能等同于简单向量运算。

编码采用 CPU、4 个 Torch 线程、batch_size=64、归一化 float32。
文件使用 JSONL 与 NumPy NPY，避免 pickle；manifest 记录逐文件 SHA-256、软件版本、数据和模型 commit。
默认拒绝覆盖完整产物。失败会非零退出并记录 traceback，不生成替代向量。

## 真实数据与测量

下列数字来自 `data/processed/statistics.json` 和真实命令输出。

| 项目 | 实测结果 |
|---|---:|
| 原始文本对 | 149,263 |
| 无效文本对 | 0 |
| 唯一 anchor | 86,169 |
| 原始 anchor 重复比例 | 42.270355% |
| 历史库 | 10,000 |
| 合格唯一语义查询池 | 14,247 |
| 验证语义 / Exact 查询 | 50 / 10 |
| 测试语义 / Exact 查询 | 100 / 20 |
| 三份矩阵维度 | 384 |
| 数据类型 | float32 |
| 最大单位范数误差 | 1.1920928955078125e-7 |
| 历史库向量有效载荷 | 15,360,000 字节 |
| 全部向量有效载荷 | 15,636,480 字节 |
| 数据准备时间 | 17.207 秒 |
| 模型加载时间 | 26.583 秒 |
| 历史库编码时间 | 75.554 秒 |
| 验证查询编码时间 | 0.177 秒 |
| 测试查询编码时间 | 0.328 秒 |
| 流水线计时至落盘、重载前 | 119.956 秒 |

准备/模型加载时间包含本次资源访问与下载；流水线计时不包含依赖安装和前置模块导入。
这些时间是 Phase 1 数据准备测量，不是检索 Query Cost。

数据集 commit：`41f699770310302022a4dd75d4cf903bfef9ea46`。
模型 commit：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
首次运行配置使用 main，解析出的 commit 记录在 manifest；验收后默认 YAML 已改为这两个固定 commit。

## 执行命令与测试结果

实际执行的主要命令（PowerShell，项目根目录）：

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pytest pyyaml datasets sentence-transformers
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding UTF8 requirements-lock.txt
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m src.prepare_data --config configs/experiment.yaml
.\.venv\Scripts\python.exe -m src.prepare_data --config configs/experiment.yaml --verify-only
.\.venv\Scripts\python.exe -m pytest -q -o cache_dir=data/test-cache --basetemp=data/test-tmp
```

最终 pytest 输出保存在 `docs/phase1-tests.log`：

```text
..................                                                       [100%]
18 passed in 12.88s
```

测试覆盖：完整字符串去重与空白保留、重复来源追溯、无效输入、查询与历史库重合剔除、
固定种子复现、数量不足/负数配置、非有限值/零向量/错误维度与 dtype、查询交集、ID/来源/数量错误、
文件重载与校验和损坏检测。两项真实集成检查额外完成：

1. 从固定版本本地 Arrow 重新构建全部文本切分，逐条等于已保存记录；核对全部语义查询来源 pair_id 的 anchor 与 positive。
2. 本地模型离线重载，对三份矩阵各取首尾两条文本重新编码，与保存向量在指定浮点容差内一致。

真实模型冒烟测试（同一真实文本重复编码）通过。
独立进程 `--verify-only` 返回 `status: passed`，行数分别为 10,000 / 60 / 120。
`pip check` 输出 `No broken requirements found.`。

## 遇到的问题及处理

- 受限网络首次安装依赖失败；获准联网后成功安装。
- 受限网络首次访问 Hugging Face 出现 WinError 10061；获准联网后成功下载并完成真实编码，失败 traceback 保留在 prepare.log。
- 首次单测临时目录创建失败，之后使用项目内目录及所需执行权限，16 项单测通过。
- 最终测试最初扫描到了先前的缓存临时目录并发生权限错误，新增 `pytest.ini` 将收集限制到 tests 后，全部 18 项通过。
- Windows 无 symlink 权限时缓存使用普通文件，不影响本次正确性。首轮部分 Hub 元数据使用了库默认用户缓存；当前代码已在导入前将后续 Hub/Xet 缓存指向项目 data/raw。

## 验收与遗留事项

- 历史库唯一、ID 顺序稳定、至少 10,000 条：通过。
- 验证/测试查询无交集、语义测试至少 100 条：通过。
- 历史/查询向量同维、有限、近似单位长度：通过。
- 文本、来源、向量、配置与环境落盘并可重载：通过。
- 失败日志真实保留，没有使用随机向量替代真实模型：通过。人工向量仅出现在校验器单测中。

Phase 1 无阻塞。以后重建需要缓存或 Hugging Face 连通性；不同硬件可能产生微小浮点差异。
Exact Hash、Ground Truth、LSH、参数搜索及 benchmark 均未实现，留待各自阶段。
