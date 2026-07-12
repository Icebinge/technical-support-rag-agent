# Learning Journal

This document records the learning route, implementation decisions, problems,
debugging process, and evidence produced while building this project.

It is intentionally written as an engineering learning log rather than a polished
final report. The goal is to make each project step explainable in interviews:
what was learned, what failed, how it was fixed, and why a design choice was
made.

## How To Use This Journal

Add one entry after each meaningful study or implementation session.

Each entry should separate:

- facts: commands, outputs, files, metrics, links, commit ids
- reasoning: why a method or architecture was chosen
- problems: errors, wrong assumptions, missing knowledge
- resolution: what changed and how it was verified
- next step: the smallest concrete action for the next session

Do not record an experiment as completed unless it was actually run. If a result
is estimated, inferred, or planned, mark it clearly.

## Current Learning Route

| Stage | Topic | Goal | Output |
| --- | --- | --- | --- |
| 0 | Project and data setup | Build a public-safe project with real public data | Repo, docs, download scripts, verification snapshot |
| 1 | Dataset parsing | Understand TechQA fields and corpus structure | PrimeQA loader, NVIDIA eval loader, schema notes |
| 2 | Sparse retrieval | Build BM25 before using any Agent framework | Top-k retriever, hit@k evaluation |
| 3 | Dense retrieval | Compare embeddings with BM25 | Vector index, latency and quality comparison |
| 4 | RAG answering | Generate grounded answers from retrieved documents | Citation-aware answer generator |
| 5 | Agent workflow | Add query rewrite, retrieval grading, and verification | LangGraph workflow and trace logs |
| 6 | Service and report | Package the project for resume/interview use | FastAPI demo, evaluation report, project summary |

## Entry Template

Copy this block when adding a new record.

````markdown
## YYYY-MM-DD - Session Title

### Goal

- TODO

### What I Studied

- TODO

### What I Built Or Changed

- TODO

### Commands And Evidence

```powershell

```

### Problems Encountered

- TODO

### Root Cause

- TODO

### Solution

- TODO

### Why This Choice

- TODO

### Verification

- TODO

### What I Still Do Not Understand

- TODO

### Next Step

- TODO
````

## 2026-07-12 - Stage 0: Project Setup And Dataset Verification

### Goal

- Start a complete Agent/RAG project that can be learned step by step.
- Use public official datasets instead of private enterprise data or generated
  toy data.
- Keep the GitHub repository public-safe while still downloading real data
  locally.

### What I Studied

- Why VLM training and testing is not the right first project direction on the
  current local environment.
- Why a technical-support RAG Agent is more suitable for local learning:
  retrieval, evaluation, data processing, and workflow orchestration can be
  built without training a large model.
- The difference between using a dataset for development/training and reserving
  another dataset for evaluation.

### What I Built Or Changed

- Created the project folder:
  `C:\d_desktop\profile\technical_support_rag_agent`
- Created a public GitHub repository:
  `https://github.com/Icebinge/technical-support-rag-agent`
- Added project structure:
  `src/`, `scripts/`, `tests/`, `docs/`, `data/`, `artifacts/`, `outputs/`
- Added dataset download and verification scripts.
- Added typed dataset models and a loader for NVIDIA TechQA-RAG-Eval.
- Added architecture, data strategy, roadmap, and dataset snapshot documents.

### Commands And Evidence

```powershell
python -m pip install -e ".[dev,data]"
python -m ruff check .
python -m pytest -q
python scripts\download_datasets.py --include-primeqa
python scripts\verify_datasets.py
```

Observed verification facts:

```text
NVIDIA TechQA-RAG-Eval train.json: 3.83 MB
NVIDIA TechQA-RAG-Eval corpus.zip: 43.69 MB
NVIDIA rows: 910
answerable rows: 610
impossible rows: 300
missing referenced files: 0

PrimeQA/TechQA archive exists: true
PrimeQA/TechQA archive size: 2822.85 MB
```

Git commit:

```text
1c574d8 Initialize technical support RAG agent
```

### Problems Encountered

- Original VLM route was not suitable because local training and realistic
  testing were not feasible in the current environment.
- Public enterprise-style support data is not easy to obtain.
- Downloaded datasets and generated artifacts must not be committed to a public
  GitHub repository.
- Editable Python installation generated `egg-info` metadata that should not be
  part of the repository.

### Root Cause

- VLM projects often require GPU memory, model weights, and large-scale
  multimodal data that do not match the current local setup.
- Real company documents are private, so the project needs public technical
  support data with clear provenance.
- GitHub repositories should contain reproducible code and documentation, not
  large raw datasets or local runtime artifacts.
- `pip install -e .` can create local packaging metadata under `src/`.

### Solution

- Changed the project direction to a technical-support RAG Agent.
- Used `PrimeQA/TechQA` as the development/training source and
  `nvidia/TechQA-RAG-Eval` as the evaluation source.
- Added `.gitignore` rules for raw data, processed data, indexes, artifacts,
  outputs, caches, model files, and packaging metadata.
- Verified ignored data paths with `git check-ignore`.

### Why This Choice

- Agent/RAG work maps better to current mainstream engineering and algorithm
  roles: data processing, retrieval, evaluation, model calling, workflow
  orchestration, and service packaging.
- BM25 and retrieval baselines can be learned locally before adding LangChain or
  LangGraph.
- Separating development data from evaluation data makes the project easier to
  defend in interviews because it reduces evaluation leakage.
- Keeping raw datasets out of git protects the repository from large-file and
  licensing problems.

### Verification

- `ruff` passed.
- `pytest` passed.
- Dataset verification script completed.
- GitHub repository was created and pushed.
- Raw datasets were present locally but ignored by git.

### What I Still Do Not Understand

- The exact schema and field meanings inside the original PrimeQA archive.
- How much overlap exists between PrimeQA/TechQA and NVIDIA TechQA-RAG-Eval.
- What BM25 baseline score should be considered acceptable for this dataset.

### Next Step

- Parse the original PrimeQA archive.
- Inspect training/dev question formats and technote corpus formats.
- Build the first BM25 retrieval baseline.
- Evaluate hit@1, hit@5, and hit@10 before adding dense retrieval or Agent
  orchestration.

## 2026-07-12 - Stage 2: BM25 Baseline And Error Analysis

### Goal

- 先完成一个不用大模型的检索 baseline。
- 理解 `hit@1`、`hit@5`、`hit@10`、`MRR` 这些指标的含义。
- 把 BM25 评估和错误分析封装成可以重复运行的脚本。

### What I Studied

- BM25 baseline 不是神经网络模型，也不是大语言模型，而是传统关键词检索算法。
- BM25 的核心依据是词频、逆文档频率和文档长度归一化。
- `hit@k` 衡量正确答案文档是否进入前 k 个检索结果。
- `MRR` 衡量正确答案文档整体排位是否靠前。
- 指标只能说明系统“错了多少”，错误分析才能解释“为什么错”。

### What I Built Or Changed

- 新增 PrimeQA 原始数据读取能力。
- 新增 BM25 检索器。
- 新增 BM25 评估脚本：
  `scripts/evaluate_bm25.py`
- 新增 BM25 错误分析脚本：
  `scripts/analyze_bm25_errors.py`
- 错误分析脚本会保存 answer_doc_id 没有进入 top-k 的失败案例。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\evaluate_bm25.py
python scripts\analyze_bm25_errors.py
```

BM25 dev baseline 结果：

```text
documents: 28482
questions: 310
evaluated_questions: 160

hit@1:  0.45
hit@5:  0.6687
hit@10: 0.7438
MRR:    0.5459
```

错误分析脚本输出：

```text
output: artifacts/bm25_dev_error_cases.json
top_k: 10
limit: 20
saved_error_cases: 20
```

第一个失败案例摘要：

```text
question_id: DEV_Q000
gold_answer_doc_id: swg24042191
top1_doc_id: swg21681385
top1_score: 91.2916
top10_count: 10
```

### Problems Encountered

- 一开始只看到 `hit@k` 和 `MRR`，还不能判断 BM25 为什么失败。
- BM25 不是“模型”，容易和 embedding model、reranker、LLM 混在一起理解。
- PowerShell 直接显示中文源码时出现乱码样式，需要区分“终端显示问题”和“文件真实编码问题”。
- Typer 参数如果直接在默认参数里调用 `typer.Option()`，会被 ruff 的 `B008` 规则拦下。

### Root Cause

- 聚合指标只能给出整体表现，不能展示具体失败样本。
- BM25 属于传统稀疏检索，主要依赖关键词匹配，因此会在同义表达、长问题、关键词偏移、标题不匹配时失败。
- Windows 终端编码显示和文件 UTF-8 编码不是一回事。
- 规范的 Typer 写法需要用 `typing.Annotated` 包装命令行参数。

### Solution

- 保留 BM25 baseline 作为后续 Dense Retrieval、Hybrid Retrieval、Reranker 的对照线。
- 新增 `scripts/analyze_bm25_errors.py`，把 top10 未命中的问题保存成可人工阅读的 JSON。
- 失败案例中记录 question、gold answer doc、gold answer、top-k doc id、title 和 score。
- 用 Python 按 UTF-8 读取源码，确认中文注释和 docstring 实际编码正确。
- 用 `Annotated[..., typer.Option(...)]` 改写脚本参数，保证 `ruff` 通过。

### Why This Choice

- 直接上 Dense Retrieval 或 Agent 容易变成堆工具，不知道每一步是否真正解决了问题。
- 先做错误分析，可以知道 BM25 的短板主要来自关键词不匹配、语义不一致，还是数据标注和候选集合问题。
- 错误案例文件能成为后续模型改进的依据，而不是只看一个分数涨跌。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 7 个测试。
- `python scripts\evaluate_bm25.py` 已真实运行并生成 BM25 指标。
- `python scripts\analyze_bm25_errors.py` 已真实运行并生成 20 条失败案例。
- `artifacts/bm25_dev_metrics.json` 和 `artifacts/bm25_dev_error_cases.json` 都被 `.gitignore` 排除，不会误提交。

### What I Still Do Not Understand

- 这 20 条失败案例中，主要失败类型分别占多少。
- BM25 的 top10 里有多少是“语义接近但不是 gold doc”的情况。
- Dense Retrieval 是否能真正提升这些 top10 未命中样本，而不是只提升已经容易命中的样本。

### Next Step

- 人工阅读 `artifacts/bm25_dev_error_cases.json` 中的失败案例。
- 给失败案例打标签，例如关键词不匹配、问题太短、正确文档标题不相关、相似文档干扰、文档过长。
- 基于错误类型再决定 Dense Retrieval baseline 的模型和评估方式。

## 2026-07-12 - Stage 3: Dense Retrieval Baseline And Hybrid Comparison

### Goal

- 把 BM25 错误分析的结论记录下来，并带上具体失败例子。
- 实现 Dense Retrieval baseline，验证 embedding 是否能补上 BM25 的语义短板。
- 比较 BM25、Dense、BM25 + Dense Hybrid 三种检索方式的 `hit@k` 和 `MRR`。

### What I Studied

- BM25 的错误不只是“找不到文档”，更多是“关键词很像，但不是 gold answer doc”。
- Dense Retrieval 使用 embedding 向量相似度，不依赖完全相同的关键词。
- 小型通用 embedding 模型不一定比 BM25 强，尤其是在技术支持文档这种关键词、版本号、错误码很重要的场景。
- Hybrid Retrieval 可以用 RRF 把 BM25 和 Dense 的排名融合起来，不需要直接比较两种分数的绝对值。

### What I Built Or Changed

- 抽出通用检索评估层：
  `src/ts_rag_agent/application/retrieval_evaluation.py`
- 抽出通用检索结果模型：
  `src/ts_rag_agent/domain/retrieval.py`
- 新增 Dense Retriever：
  `src/ts_rag_agent/infrastructure/dense_retriever.py`
- 新增 Dense 文档向量缓存：
  `src/ts_rag_agent/infrastructure/dense_embedding_cache.py`
- 新增 Hybrid Retriever：
  `src/ts_rag_agent/infrastructure/hybrid_retriever.py`
- 新增 Dense 评估脚本：
  `scripts/evaluate_dense.py`
- 新增 Hybrid 评估脚本：
  `scripts/evaluate_hybrid.py`

### Commands And Evidence

```powershell
python -m pip install -e ".[rag]"
python -m ruff check .
python -m pytest -q
python scripts\evaluate_dense.py
python scripts\evaluate_hybrid.py
python scripts\evaluate_bm25.py
```

使用的 Dense 模型：

```text
sentence-transformers/all-MiniLM-L6-v2
embedding dimension: 384
document_text_max_chars: 1600
documents: 28482
questions: 310
evaluated_questions: 160
```

指标对比：

| Method | hit@1 | hit@5 | hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.45 | 0.6687 | 0.7438 | 0.5459 |
| Dense all-MiniLM-L6-v2 | 0.375 | 0.575 | 0.675 | 0.469 |
| BM25 + Dense RRF | 0.4125 | 0.6937 | 0.75 | 0.5358 |

Dense 首次运行耗时：

```text
load_model: 16.336s
document_embeddings: 441.622s
evaluate: 2.918s
total: 461.87s
cache_status: created
```

Dense 使用缓存复跑：

```text
document_embeddings: 0.189s
total: 21.529s
cache_status: loaded
```

Hybrid 使用缓存运行：

```text
method: reciprocal_rank_fusion
candidate_top_k: 100
rrf_k: 60
dense_cache_status: loaded
hit@10: 0.75
```

### BM25 Error Examples

#### 1. 关键词完全相似，但答案语义更深

```text
question_id: DEV_Q055
question: Table ""."" could not be found
gold: swg21412846
gold title: Wrong codepoints for non-ASCII characters inserted in UTF-8 database
top1: swg21647171
top1 title: Table "<Schema>"."<Table Name>" could not be found in the database.
```

BM25 抓住了 `table could not be found`，所以 top1 看起来非常相关。但 gold 文档真正讲的是 UTF-8 / non-ASCII codepoint 导致的问题。

#### 2. top1 文档看起来也合理，但不是 gold doc

```text
question_id: DEV_Q077
question: I need to transfer my SPSS 24 licence to a new computer
gold: swg21592093
gold title: SPSS Student Version and Graduate Pack Resources
top1: swg21985888
top1 title: How do I transfer my IBM SPSS product/software license from one machine to another?
```

这个例子说明严格 `answer_doc_id` 评估可能低估用户体验，因为 top1 文档标题本身也非常像可用答案。

#### 3. gold 文档标题很间接，BM25 不容易命中

```text
question_id: DEV_Q034
question: Profiler for WebSphere 8
gold: swg21413628
gold title: Java Health Center Client - a low overhead monitoring tool
top1: swg21566549
top1 title: Potential native memory use in reflection delegating classloaders
```

用户说 `Profiler`，gold 文档说 `Health Center / monitoring tool`。这是 Dense Retrieval 理论上应该帮助的同义或近义表达场景。

#### 4. 版本号、产品名、fix pack 强干扰

```text
question_id: DEV_Q039
question: upgrade virtual DataPower Appliance from 5.0 firmware to 6.0+
gold: swg21638268
gold title: Supported Upgrade and Downgrade paths for DataPower Virtual Edition
top1: swg21674513
top1 title: Firmware upgrade via AMP ... virtual appliance ... out of memory
```

BM25 抓住了 `upgrade / firmware / virtual appliance`，但没有理解用户真正问的是支持的升级路径。

#### 5. 重复或近重复文档导致严格 doc id 评估偏低

```text
question_id: DEV_Q066
gold: swg22000947
top1: swg22011696
shared title topic: Action required for IBM Integration Bus Hypervisor Edition ...
```

gold 和 top1 标题几乎一样，但 doc id 不同，因此被算作 top10 未命中。这类样本需要后续人工检查内容是否等价。

### Problems Encountered

- Dense 依赖没有安装，`sentence-transformers`、`torch`、`scikit-learn` 都需要通过 `.[rag]` 安装。
- 第一次导入和加载 `sentence-transformers` 较慢，不能误判为脚本卡死。
- 第一次编码 28482 篇文档耗时较长，CPU 上大约 7 分钟。
- Dense baseline 的指标低于 BM25，和“语义模型一定更强”的直觉相反。
- 如果每次都重新编码文档，实验迭代成本很高。

### Root Cause

- `all-MiniLM-L6-v2` 是小型通用 embedding 模型，不是专门为 IBM 技术支持文档训练的检索模型。
- 技术支持检索高度依赖产品名、版本号、错误码、fix pack 名称，BM25 在这类表面关键词上有天然优势。
- 当前 Dense baseline 只取每篇文档前 1600 个字符，可能截断了 gold answer 所在内容。
- Dense 向量相似度能补充语义，但也可能把关键词精确匹配能力变弱。

### Solution

- 保留 BM25 作为强 baseline，不因为 Dense 是“模型”就默认它更好。
- 增加文档向量缓存，把首次编码结果保存到 `data/indexes/dense/`。
- 用同一个 `evaluate_retrieval` 评估 BM25、Dense 和 Hybrid，保证指标可比。
- 用 RRF 做 BM25 + Dense 融合，避免直接混合 BM25 分数和 cosine 分数。

### Why This Choice

- Dense Retrieval 的价值需要通过实验验证，而不是通过概念判断。
- RRF 是简单稳健的融合方法，适合第一版 hybrid baseline。
- 缓存文档向量能让后续调参和模型对比从几分钟降到几十秒。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 12 个测试。
- `python scripts\evaluate_dense.py` 已真实运行并生成 Dense 指标。
- `python scripts\evaluate_hybrid.py` 已真实运行并生成 Hybrid 指标。
- Dense 文档向量缓存已生成：
  `data/indexes/dense/sentence-transformers__all-MiniLM-L6-v2_1600.npz`
- 指标报告已生成：
  `artifacts/dense_dev_metrics.json`
  `artifacts/hybrid_dev_metrics.json`

### What I Still Do Not Understand

- Dense 低于 BM25 的主要原因是模型太弱、文档截断、还是技术文档本身更适合关键词检索。
- 使用更强的检索模型，例如 BGE / E5，是否能超过 BM25。
- 如果把文档按 section chunk 切分，而不是整篇文档前 1600 字符，Dense 是否会明显改善。
- Hybrid 的最佳 `candidate_top_k`、`rrf_k` 和权重还没有系统搜索。

### Next Step

- 先不要进入 Agent。
- 下一步做 Dense 改进实验：
  1. 换一个检索向 embedding 模型。
  2. 把文档按 section/chunk 切分后再做 Dense。
  3. 对比 BM25、Dense、Hybrid 在同一批 BM25 失败案例上的召回变化。
  4. 如果 Dense 或 Hybrid 能稳定提升，再进入 reranker。

## 2026-07-12 - Stage 3 Follow-up: Dense And Hybrid Result Analysis

### Goal

- 进一步分析 Dense 和 Hybrid 的新结果，而不是只看总分。
- 回答三个问题：
  1. Dense 到底救回了多少 BM25 没命中的问题？
  2. Dense 又丢掉了多少 BM25 已经命中的问题？
  3. Hybrid 为什么 `hit@10` 只小幅提升，但 `hit@1` 和 `MRR` 下降？

### What I Studied

- 检索系统不能只看单个总指标，要看样本级的得失。
- Dense Retrieval 和 BM25 的错误模式不同：Dense 能补一些语义相关问题，但会损失一部分关键词精确匹配问题。
- Hybrid 不是必然优于 BM25。融合方法如果没有调参，可能只提升深层召回，却损害首位排序。

### Commands And Evidence

这次没有新增正式脚本，而是用一次性分析代码读取 BM25、Dense、Hybrid 三个检索器，对 160 个可回答 dev 问题逐题计算 gold 文档是否进入 top10。

核心统计：

```text
evaluated: 160

top10 hits:
BM25:   119
Dense:  108
Hybrid: 120

Dense 救回 BM25 miss: 14
Dense 丢掉 BM25 hit: 25

Hybrid 救回 BM25 miss: 9
Hybrid 丢掉 BM25 hit: 8

BM25 和 Dense 都命中: 94
三种方法都没命中: 27
```

按 top1 / top5 / top10 计数：

```text
BM25:
top1: 72
top5: 107
top10: 119

Dense:
top1: 60
top5: 92
top10: 108

Hybrid:
top1: 66
top5: 111
top10: 120
```

### Key Analysis

#### 1. Dense 确实救回了一些 BM25 的语义失败样本

例子：

```text
DEV_Q039
question: How do I upgrade my virtual DataPower Appliance from 5.0 firmware to 6.0+ firmware?
gold: swg21638268
gold title: Supported Upgrade and Downgrade paths for DataPower Virtual Edition

BM25 rank: miss@10
Dense rank: 3
Hybrid rank: 2
```

BM25 被 `firmware / upgrade / virtual appliance` 这些关键词吸引到其他文档；Dense 更接近“升级路径”这个语义目标。

另一个例子：

```text
DEV_Q052
question: Why do I still get "certificate expired" error after adding new certificate?
gold: swg21500046
gold title: Replacement of an expiring certificate on the IBM WebSphere DataPower SOA Appliance

BM25 rank: miss@10
Dense rank: 4
Hybrid rank: 6
```

这里 Dense 把 `certificate expired` 和 `expiring certificate replacement` 联系起来，说明 embedding 对同义或近义表达有帮助。

再一个例子：

```text
DEV_Q195
question: TLS protocol with ITCAM for Datapower
gold: swg21959224
gold title: TLS support and DataPower appliance

BM25 rank: miss@10
Dense rank: 1
Hybrid rank: 4
```

这是 Dense 最有说服力的成功样本：gold 文档被 Dense 排到第 1。

#### 2. Dense 丢掉了更多 BM25 已经命中的关键词型样本

例子：

```text
DEV_Q028
question: Help with Security Bulletin: Multiple vulnerabilities have been identified in WebSphere Application Server shipped with ...
gold: swg21975747

BM25 rank: 1
Dense rank: miss@10
Hybrid rank: miss@10
```

这类安全公告、CVE、产品名、版本名非常依赖精确关键词。BM25 在这种场景里天然强，而小型通用 Dense 模型反而会把相似公告混在一起。

另一个例子：

```text
DEV_Q080
question: Why is the OUTPUT_TYPE specified in the properties file for the custom scripting feature ignored?
gold: swg21960062

BM25 rank: 1
Dense rank: miss@10
Hybrid rank: 2
```

BM25 能抓住 `OUTPUT_TYPE`、`properties file`、`custom scripting feature` 这类精确 token；Dense 会弱化这些 token 的精确性。

再一个例子：

```text
DEV_Q043
question: Where I can get ITNM 4.2.0.1 GA version download details with Part number?
gold: swg24042656

BM25 rank: 1
Dense rank: miss@10
Hybrid rank: 3
```

版本号和 part number 是典型关键词检索强项。Dense 没有把这些精确标识当作足够强的信号。

#### 3. Hybrid 的 top10 有小幅提升，但首位排序被 Dense 拉低

Hybrid 的结果：

```text
BM25 top10 hits: 119
Hybrid top10 hits: 120
净增: +1

BM25 top1 hits: 72
Hybrid top1 hits: 66
净减: -6

BM25 MRR: 0.5459
Hybrid MRR: 0.5358
下降: -0.0101
```

原因是 Hybrid 用 RRF 融合排名后，Dense 会把一些 BM25 已经排第 1 的 gold 文档往后推，甚至推出 top10。它也会救回一些 BM25 miss，但救回数量和损失数量接近：

```text
Hybrid 救回 BM25 miss: 9
Hybrid 丢掉 BM25 hit: 8
```

所以 top10 只净增 1 个命中。

### Conclusion

这次结果不能简单理解成“Dense 没用”。更准确的结论是：

```text
BM25 更擅长：
- 产品名
- 版本号
- fix pack
- CVE / Security Bulletin
- 错误码
- 配置项名称
- 精确 token

Dense 更擅长：
- 同义表达
- 问题意图相似
- 标题不完全匹配但语义接近
- 用户问法和文档标题不一致

当前 Hybrid：
- top10 召回略升
- top1 和 MRR 下降
- 说明融合参数和 Dense 模型还不够好
```

因此，当前最重要的判断是：

```text
BM25 仍然是这个数据集上的强 baseline。
all-MiniLM-L6-v2 不能直接替代 BM25。
Dense 有补充价值，但需要更强模型、chunk 策略或 reranker 才可能稳定提升。
```

### Why This Matters

- 这让项目从“我用了 embedding”变成“我知道 embedding 在什么样本上有帮助，什么样本上会伤害结果”。
- 这也说明后续做 Agent 之前，必须先把 retrieval 层做扎实。
- 如果 retrieval 本身没有稳定提升，Agent workflow 只会包装错误，而不是解决错误。

### Next Step

- 不直接进入 Agent。
- 下一步优先做两个实验：
  1. 换更适合检索的 embedding 模型，例如 BGE / E5 系列。
  2. 把 technote 从整篇前 1600 字符改为 section/chunk 检索。
- 同时要保留 BM25 精确关键词能力，后续 Hybrid 应该调权重或进入 reranker，而不是简单平均两路结果。

## 2026-07-12 - Stage 3 Follow-up: E5 And Section BM25 Experiments

### Goal

- 验证换一个检索向 embedding 模型是否能超过 BM25。
- 验证 section 粒度检索是否能优于整篇文档 BM25。
- 继续判断是否已经适合进入 Agent。

### What I Built Or Changed

- 给 Dense Retriever 增加 `query_prefix` 和 `document_prefix` 支持。
- 给 dense 文档向量缓存增加 `document_prefix` 校验，避免不同输入格式共用错误缓存。
- 新增 section 级 BM25 检索器：
  `src/ts_rag_agent/infrastructure/section_bm25_retriever.py`
- 新增 section BM25 评估脚本：
  `scripts/evaluate_section_bm25.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\evaluate_dense.py --model-name intfloat/e5-small-v2 --query-prefix "query: " --document-prefix "passage: " --document-text-max-chars 512 --output artifacts\dense_e5_small_v2_512_dev_metrics.json
python scripts\evaluate_hybrid.py --model-name intfloat/e5-small-v2 --query-prefix "query: " --document-prefix "passage: " --document-text-max-chars 512 --output artifacts\hybrid_e5_small_v2_512_dev_metrics.json
python scripts\evaluate_section_bm25.py
```

E5 small 烟雾测试：

```text
model: intfloat/e5-small-v2
embedding_shape: (2, 384)
```

完整对比结果：

| Method | hit@1 | hit@5 | hit@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25 doc | 0.45 | 0.6687 | 0.7438 | 0.5459 |
| Dense all-MiniLM-L6-v2 1600 | 0.375 | 0.575 | 0.675 | 0.469 |
| Hybrid all-MiniLM-L6-v2 1600 | 0.4125 | 0.6937 | 0.75 | 0.5358 |
| Dense E5 small 512 | 0.4 | 0.6062 | 0.6562 | 0.4842 |
| Hybrid E5 small 512 | 0.4562 | 0.6625 | 0.7063 | 0.5498 |
| Section BM25 | 0.4375 | 0.6375 | 0.6875 | 0.5243 |

Section 统计：

```text
documents: 28482
sections: 216648
avg_sections_per_doc: 7.606
```

### Problems Encountered

- `intfloat/e5-small-v2` 在 1600 字符截断设置下完整评估超过 30 分钟没有返回，被终止。
- 改用 512 字符截断后可以完成，但 Dense E5 small 仍然没有超过 BM25。
- Section BM25 的评估耗时明显高于文档级 BM25，而且指标更低。

### Root Cause

- E5 small 虽然是检索向 embedding 模型，但仍然是小模型；在技术支持文档这种强关键词场景下，不一定比 BM25 更强。
- 使用 512 字符截断会进一步损失长文档信息，因此这个结果不能证明 E5 全量 1600 字符一定无效，只能说明当前本机成本下的 E5 512 配置没有优势。
- Section BM25 把 28482 篇文档扩展成 216648 个 section，候选空间变大后，短 section 容易因为局部关键词匹配被推高。
- 当前 Section BM25 使用“父文档取最高 section 分数”的聚合方式，这可能让某些只有局部关键词匹配的 section 过度影响父文档排名。

### Solution

- 不继续盲目堆更大的 dense 模型。
- 先把 E5 small 512 和 Section BM25 作为负结果记录下来。
- 保留 BM25 doc 作为当前最强、最稳定的 baseline。
- 如果后续继续做 chunk，需要重新设计 chunk 聚合方式，而不是简单 max section score。

### Why This Choice

- 负结果同样有价值：它说明项目不是为了套模型而套模型，而是在用实验判断路线。
- 当前证据不支持“直接进入 Agent”，因为 retrieval 层还没有稳定提升。
- 如果 Agent 建在弱 retrieval 上，后续生成、重写、验证都会被错误上下文拖累。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 15 个测试。
- E5 small 512 Dense 指标已生成：
  `artifacts/dense_e5_small_v2_512_dev_metrics.json`
- E5 small 512 Hybrid 指标已生成：
  `artifacts/hybrid_e5_small_v2_512_dev_metrics.json`
- Section BM25 指标已生成：
  `artifacts/section_bm25_dev_metrics.json`
- 以上报告和 dense 缓存都被 `.gitignore` 排除。

### What I Still Do Not Understand

- E5 small 在 1600 字符设置下如果完整跑完，是否会比 512 字符明显更好。
- Section BM25 如果改用 top-n section 聚合、平均聚合或学习式融合，是否能超过文档级 BM25。
- 更强的 BGE / E5 base 模型是否能在本机可接受时间内完成。

### Next Step

- 暂时停止继续换 embedding 模型。
- 下一步更合理的是做 reranker baseline：
  1. 用 BM25 召回 top50 或 top100。
  2. 用 cross-encoder/reranker 重排候选文档。
  3. 只在候选集上计算重排指标，避免全库 dense 编码成本。
- 如果 reranker 能提升 hit@1 和 MRR，再考虑进入 RAG answer generation。

## 2026-07-12 - Stage 4: BM25 Candidate Reranker Baseline

### Goal

- 从全库 Dense/Hybrid 实验转向候选重排。
- 验证 reranker 是否能提升 BM25 的 `hit@1` 和 `MRR`。
- 避免继续做成本很高但收益不明确的全库 dense 编码。

### What I Studied

- Reranker 不负责全库召回，而是在 BM25 已经召回的候选文档里重新排序。
- CrossEncoder 会同时读取 query 和 document，因此理论上比单独 embedding 更适合判断细粒度相关性。
- Reranker 的上限受候选召回限制：如果 gold 文档不在 BM25 top50/top100 中，reranker 无法把它排回来。

### What I Built Or Changed

- 新增 reranker 模块：
  `src/ts_rag_agent/infrastructure/reranker.py`
- 新增 reranker 评估脚本：
  `scripts/evaluate_reranker.py`
- 新增 reranker 单元测试：
  `tests/test_reranker.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\evaluate_reranker.py --candidate-top-k 50 --batch-size 32 --output artifacts\reranker_bm25_top50_dev_metrics.json
```

模型烟雾测试：

```text
model: cross-encoder/ms-marco-MiniLM-L-6-v2
positive pair score: 4.3955
negative pair score: -11.3817
```

BM25 文档级 baseline：

```text
hit@1:  0.45
hit@5:  0.6687
hit@10: 0.7438
MRR:    0.5459
```

BM25 top50 + CrossEncoder reranker：

```text
candidate_top_k: 50
model: cross-encoder/ms-marco-MiniLM-L-6-v2
hit@1:  0.4188
hit@5:  0.6562
hit@10: 0.7063
MRR:    0.5151
total_time: 340.887s
```

BM25 候选召回上限：

```text
BM25 recall@10:  119 / 160 = 0.7438
BM25 recall@20:  127 / 160 = 0.7937
BM25 recall@50:  135 / 160 = 0.8438
BM25 recall@100: 143 / 160 = 0.8938
```

### Problems Encountered

- Reranker 没有超过 BM25，反而降低了 `hit@1`、`hit@5`、`hit@10` 和 `MRR`。
- BM25 top50 里已有 135 个 gold 文档，但 reranker top10 只保住约 113 个。
- CrossEncoder 评估耗时明显更高，160 个可回答问题、top50 候选约 8000 对文本，CPU 上大约 5 分多钟。

### Root Cause

- `cross-encoder/ms-marco-MiniLM-L-6-v2` 是通用 MS MARCO 检索重排模型，不是 IBM 技术支持领域模型。
- 技术支持文档中产品名、版本号、错误码、fix pack、Security Bulletin 等精确 token 非常关键，通用 reranker 可能会弱化这些信号。
- 当前输入给 reranker 的文档仍是整篇文档前 1600 字符，gold answer 可能不在截断内容里。
- BM25 已经把很多 gold 文档排得很靠前，reranker 如果不适配领域，反而会把正确文档往后推。

### Solution

- 不把 reranker 失败包装成成功。
- 把它作为负结果记录下来：通用 MS MARCO reranker 不适合直接替代 BM25 排序。
- 后续如果继续做 reranker，应优先考虑：
  1. 用 section/chunk 作为 reranker 输入，而不是整篇文档前 1600 字符。
  2. 使用更适合技术文档或问答检索的 reranker。
  3. 保留 BM25 排名特征，而不是完全相信 reranker 分数。

### Why This Choice

- 这一步比直接进入 Agent 更有价值，因为它证明了一个关键事实：不是所有“更深的模型”都会提升检索。
- 现在已经有 BM25、Dense、Hybrid、Section BM25、Reranker 多条 baseline，项目的实验链条更完整。
- 这些负结果能在面试中说明：项目不是简单套框架，而是在用实验选择技术路线。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 17 个测试。
- reranker 真实评估报告已生成：
  `artifacts/reranker_bm25_top50_dev_metrics.json`

### What I Still Do Not Understand

- 如果把 BM25 top50 的候选文档切成 gold 相关 section，再给 reranker，会不会提升。
- 如果使用更强 reranker，例如 BGE reranker，是否会超过 BM25。
- 是否应该设计 BM25 score + reranker score 的加权融合，而不是完全用 reranker 重排。

### Next Step

- 当前不建议继续盲目换模型。
- 下一步建议进入 RAG answer generation 的最小闭环，但保持检索器使用当前最强的 BM25 doc baseline。
- RAG 阶段先不做复杂 Agent，只做：
  1. BM25 top5 检索。
  2. 把 top5 文档作为上下文。
  3. 生成带引用的答案。
  4. 对 answerable / unanswerable 做基本评估。

## 2026-07-12 - Stage 5: Minimal Extractive RAG Answer Baseline

### Goal

- 进入 RAG answer generation 的最小闭环。
- 暂时不使用付费 API，也不调用本地大语言模型。
- 先用 BM25 top5 作为上下文，做一个抽取式、可解释、可评估的带引用答案 baseline。

### What I Studied

- RAG 不等于 Agent。RAG 的最小闭环是：检索上下文、生成答案、给出引用、评估答案和引用。
- 生成答案之前必须先确认检索上下文是否可靠。
- 如果没有 answerability 判断，系统很容易对不可回答问题也硬生成答案。
- 抽取式 baseline 虽然不是 LLM，但能帮助验证 citation、refusal 和上下文质量。

### What I Built Or Changed

- 新增答案领域模型：
  `src/ts_rag_agent/domain/answer.py`
- 新增抽取式 RAG 回答器：
  `src/ts_rag_agent/application/rag_answering.py`
- 新增 RAG 评估脚本：
  `scripts/evaluate_extractive_rag.py`
- 新增测试：
  `tests/test_rag_answering.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\evaluate_extractive_rag.py
```

实验配置：

```text
retriever: BM25
retrieval_top_k: 5
answer_generator: extractive_sentence_baseline
max_sentences: 3
min_sentence_score: 2.0
```

真实 dev 结果：

```text
documents: 28482
questions: 310
answerable_questions: 160
unanswerable_questions: 150

answerable_gold_doc_in_context: 107
answerable_gold_doc_in_context_rate: 0.6687

generated_answerable_questions: 160
refused_answerable_questions: 0
refused_unanswerable_questions: 1

gold_doc_citation_rate: 0.4875
answerable_refusal_rate: 0.0
unanswerable_refusal_rate: 0.0067
average_token_f1: 0.2155
```

报告文件：

```text
artifacts/extractive_rag_dev_report.json
```

### Example Observations

#### 1. 检索错了，答案也会跟着错

```text
question_id: DEV_Q000
question: Web GUI 8.1 FP7 requires DASH 3.1.2.1 or later
gold_doc: swg24042191
retrieved_top1: swg21681385
generated_citations: swg21681385
```

生成答案引用了 top1 错误文档，所以即使答案看起来有引用，也不是 gold evidence。

#### 2. gold 文档在 top5 中，但抽取句子仍然可能引用错文档

```text
question_id: DEV_Q002
gold_doc: swg21978390
retrieved_docs: swg21598554, swg21673044, swg1IO10742, swg21978390, swg1IC60317
generated_citations: swg21598554, swg21673044, swg1IC60317
```

这说明“gold 文档进入上下文”不等于“答案会引用 gold 文档”。还需要 evidence selection 或 reranking。

#### 3. 不可回答问题几乎没有被拒答

```text
question_id: DEV_Q001
answerable: False
question: Too many open files messages in the DASH systemOut
refused: False
```

抽取式回答器从检索文档里抽出了看似相关的句子，但数据标注认为这个问题不可回答。这说明 answerability 判断不能只靠关键词重合。

### Problems Encountered

- 抽取式 baseline 能生成引用，但引用不一定是正确证据。
- 对不可回答问题，当前规则几乎都会生成答案，拒答率只有 `0.0067`。
- `answerable_gold_doc_in_context_rate` 是 `0.6687`，但 `gold_doc_citation_rate` 只有 `0.4875`，说明 evidence selection 还有明显损失。
- 平均 token F1 只有 `0.2155`，抽取句和标准答案之间仍有较大差距。

### Root Cause

- BM25 top5 没有覆盖所有 answerable 问题的 gold 文档，这是检索上限问题。
- 即使 gold 文档进入 top5，简单句子打分也可能选择错误文档中的高关键词重合句。
- unanswerable 问题仍然会检索到表面相关文档，抽取式生成器缺少证据充分性判断。
- 当前 baseline 没有 LLM 的归纳和压缩能力，也没有专门的 answerability classifier。

### Solution

- 如实记录这个 baseline 是“抽取式 RAG”，不是 LLM RAG。
- 把 citation 命中率、拒答率和 token F1 都记录下来，不只记录是否生成答案。
- 暂时保留 BM25 作为上下文检索器，因为它仍然是当前最强 retrieval baseline。

### Why This Choice

- 这一步让项目从 retrieval 进入 answer generation，但仍然保持可控和可评估。
- 在没有 LLM 前先跑抽取式 baseline，可以明确后续 LLM 需要解决什么：
  1. 更好地选择证据。
  2. 更好地组织答案。
  3. 正确拒答不可回答问题。
  4. 保证引用来自真实上下文。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 20 个测试。
- `python scripts\evaluate_extractive_rag.py` 已真实运行。
- 报告文件 `artifacts/extractive_rag_dev_report.json` 已生成。

### What I Still Do Not Understand

- 不可回答问题应该如何判断，是用阈值、分类器，还是让 LLM 判断证据充分性。
- LLM 生成答案后，如何自动判断它是否真正 grounded。
- 引用 gold doc 和用户实际满意度之间是否完全一致。

### Next Step

- 下一步不要直接做复杂 Agent。
- 应该先做 RAG 的 answerability / citation verification：
  1. 判断 top5 文档是否足够回答。
  2. 如果证据不足，拒答。
  3. 检查生成答案中的引用是否来自检索上下文。
  4. 评估 refusal accuracy 和 citation correctness。

## 2026-07-12 - Stage 6: Answer Verification And Quality Analysis

### Goal

- 在抽取式 RAG baseline 之后，补上一层 answerability / citation verification。
- 不追求把答案包装得更像 Agent，而是先让系统具备“证据不够就拒答”的能力。
- 分析验证层到底拦住了什么：是合理拒答、检索失败、证据选择失败，还是阈值误杀。

### What I Studied

- RAG 的质量不能只看有没有生成答案，还要看答案是否 grounded。
- 引用来自检索上下文本身只是最低要求，不代表引用的是正确 gold doc。
- answerability 判断会带来取舍：拒答率提高可以减少乱答，但也可能误杀可回答问题。
- 阈值型验证器容易解释、容易复现，但它不能真正理解答案语义，只能作为第一版质量门控。

### What I Built Or Changed

- 新增答案验证器：
  `src/ts_rag_agent/application/answer_verification.py`
- 新增验证版 RAG 评估服务：
  `src/ts_rag_agent/application/verified_rag_evaluation.py`
- 新增验证版 RAG 评估脚本：
  `scripts/evaluate_verified_rag.py`
- 新增质量分析模块：
  `src/ts_rag_agent/application/verified_rag_quality_analysis.py`
- 新增质量分析脚本：
  `scripts/analyze_verified_rag_quality.py`
- 新增测试：
  `tests/test_answer_verification.py`
  `tests/test_verified_rag_quality_analysis.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\evaluate_verified_rag.py
python scripts\analyze_verified_rag_quality.py
```

验证层配置：

```text
retriever: BM25
retrieval_top_k: 5
answer_generator: extractive_sentence_baseline
max_sentences: 3
min_sentence_score: 2.0
answer_verifier: citation_and_evidence_gate
min_evidence_score: 8.0
max_citation_rank: 3
min_citations: 1
```

真实 dev 结果：

```text
documents: 28482
questions: 310
answerable_questions: 160
unanswerable_questions: 150
answerable_gold_doc_in_context: 107

original gold_doc_citation_rate: 0.4875
verified gold_doc_citation_rate: 0.5263

original answerable_refusal_rate: 0.0
verified answerable_refusal_rate: 0.2875

original unanswerable_refusal_rate: 0.0067
verified unanswerable_refusal_rate: 0.2533

original average_token_f1: 0.2155
verified average_token_f1: 0.2237
```

质量分析结果：

```text
newly_refused: 83
newly_refused_answerable: 46
newly_refused_unanswerable: 37

reasonable_refusal_unanswerable: 37
safe_refusal_retrieval_miss: 16
possible_threshold_over_refusal_gold_cited: 18
evidence_selection_miss_gold_available: 12
unknown_new_refusal: 0

near_threshold_count: 53
max_evidence_score_min: 3.0
max_evidence_score_median: 6.0
max_evidence_score_mean: 5.8247
max_evidence_score_max: 7.0

unanswerable_still_answered: 112
answerable_still_answered: 114
answerable_answered_without_gold_citation: 54
answerable_answered_without_gold_citation_rate: 0.4737
unanswerable_still_answered_rate: 0.7467
```

报告文件：

```text
artifacts/verified_rag_dev_report.json
artifacts/verified_rag_quality_analysis_dev.json
```

### Example Observations

#### 1. 合理拒答：数据集标注为不可回答

```text
question_id: DEV_Q015
question: DFHTS0001 0C4 AKEA at offset 3A1E in DFHTSPT
answerable: False
cited_doc_ids: swg1PM05454, swg1PM05454, swg21597996
max_evidence_score: 6.0
verification_reasons: weak_evidence_score
```

原始抽取式回答器会从表面相关文档里抽句子，但数据集标注认为这个问题不可回答。验证层拒答是正确方向。

#### 2. 安全拒答：gold 文档没有进入 top5

```text
question_id: DEV_Q000
question: Web GUI 8.1 FP7 requires DASH 3.1.2.1 or later
gold_doc: swg24042191
retrieved_docs: swg21681385, swg21962250, swg21987786, swg21960632, swg21984598
cited_doc_ids: swg21681385, swg21681385, swg21681385
max_evidence_score: 6.0
```

这里真正答案文档没有进入检索上下文。即使生成器能抽出看似相关的句子，也不应该强答。

#### 3. 可能误杀：gold 文档已经被引用，但证据分没过阈值

```text
question_id: DEV_Q029
question: Recurrent RES StaleConnectionException
gold_doc: swg21496354
retrieved_top1: swg21496354
cited_doc_ids: swg21496354, swg21496354, swg21496354
max_evidence_score: 7.0
threshold: 8.0
```

这个样本说明当前 `min_evidence_score=8.0` 可能太硬。gold 文档已经被检索并引用，但验证层仍然因为分数不足拒答。

#### 4. 证据选择失败：gold 在 top5，但生成器引用了相邻错误文档

```text
question_id: DEV_Q008
question: How can I export a private key from DataPower Gateway Appliance?
gold_doc: swg21412061
retrieved_docs: swg21412060, swg1IT01034, swg21412061, swg21446015, swg1IT07604
cited_doc_ids: swg21412060, swg21412060, swg21412060
max_evidence_score: 5.0
```

gold 文档其实已经进了 top5，但抽取器选择了相邻主题的错误文档。这个问题不该靠简单调阈值解决，而应该改 evidence selection 或 reranking。

### Problems Encountered

- 验证层提升了 unanswerable 拒答率，但仍然有 `112 / 150` 个不可回答问题被继续回答。
- 验证层让 gold doc citation rate 从 `0.4875` 提升到 `0.5263`，提升有限。
- `46 / 160` 个可回答问题被拒答，其中 `18` 个已经引用了 gold 文档，存在阈值误杀风险。
- 新增拒答样本中 `53 / 83` 个最大证据分在阈值附近，说明简单固定阈值比较粗糙。

### Root Cause

- 当前验证器只看 evidence score、citation rank 和引用是否来自上下文，没有真正判断“这段证据是否回答了这个问题”。
- 抽取式 evidence score 基于 query-term overlap，无法理解同义表达、答案蕴含和技术因果关系。
- BM25 top5 仍然有检索上限问题：部分 answerable 问题的 gold 文档不在上下文中。
- 即使 gold 文档进入上下文，简单句子打分也可能更偏好关键词更多的错误相邻文档。

### Solution

- 保留当前验证层作为第一版可解释质量门控。
- 将新增拒答拆成四类，而不是只看一个总拒答率。
- 把“可能阈值误杀”和“证据选择失败”分开，因为它们对应不同优化方向：
  1. 阈值误杀：需要调参、校准分数或用更细的判断器。
  2. 证据选择失败：需要 reranking、section-level selection 或 answer-aware evidence selection。
- 继续记录剩余风险，尤其是 `unanswerable_still_answered`，避免只汇报提升项。

### Why This Choice

- 企业项目里 RAG 的核心不是“接一个框架”，而是能解释错误来自检索、证据选择、生成还是验证。
- 质量分析脚本能复跑，后面换 Dense、Hybrid、Reranker 或 LLM answerer 时，可以用同一套分析方法比较。
- 现在的结果说明项目还没到复杂 Agent 阶段，应该先把 RAG 的证据质量闭环做扎实。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 24 个测试。
- `python scripts\evaluate_verified_rag.py` 已真实运行。
- `python scripts\analyze_verified_rag_quality.py` 已真实运行。
- `artifacts/verified_rag_quality_analysis_dev.json` 已生成，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- `min_evidence_score=8.0` 是否应该下调，还是应该改成按问题类型、检索分布动态校准。
- 对不可回答问题，是否需要单独训练或构造 answerability classifier。
- gold doc citation 是否足够代表真实可用性；有些非 gold 文档可能也包含有用答案。
- 后续如果接 LLM，应该让 LLM 负责生成答案，还是先让 LLM 做 evidence sufficiency judge。

### Next Step

- 下一步先不要进复杂 Agent。
- 更合理的方向是做 threshold sweep / calibration：
  1. 扫描不同 `min_evidence_score`。
  2. 比较 answerable_refusal_rate、unanswerable_refusal_rate、gold_doc_citation_rate、average_token_f1。
  3. 找一个更合理的拒答阈值。
  4. 再决定是否引入 answerability classifier 或 LLM judge。

## 2026-07-12 - Stage 7: Threshold Sweep And Calibration

### Goal

- 不再凭感觉设置 `min_evidence_score=8.0`。
- 扫描 `retrieval_top_k`、`min_evidence_score`、`max_citation_rank`，观察拒答率、引用命中和答案 F1 的取舍。
- 判断当前问题主要是阈值问题、检索召回问题，还是证据选择问题。

### What I Studied

- RAG 的阈值不是单一参数，至少包括：
  1. `retrieval_top_k`：检索阶段取多少篇文档。
  2. `min_sentence_score`：抽取式回答器是否认为一个句子像证据。
  3. `min_evidence_score`：验证器是否认为答案证据足够强。
  4. `max_citation_rank`：允许引用排名多靠后的文档。
- 阈值搜索不能只看一个指标，因为提高拒答率通常会同时提高安全性和误拒风险。
- `Pareto candidate` 只能说明该配置没有被其他配置在多指标上明显压过，不代表它就是业务最优。

### What I Built Or Changed

- 给抽取式回答器增加文档切句缓存，避免阈值扫描时重复切同一篇文档：
  `src/ts_rag_agent/application/rag_answering.py`
- 新增阈值扫描应用层：
  `src/ts_rag_agent/application/verified_rag_threshold_sweep.py`
- 新增阈值扫描脚本：
  `scripts/sweep_verified_rag_thresholds.py`
- 新增阈值扫描测试：
  `tests/test_verified_rag_threshold_sweep.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\sweep_verified_rag_thresholds.py
```

扫描网格：

```text
retrieval_top_k: 5, 10, 20
min_evidence_score: 4, 5, 6, 7, 8
max_citation_rank: 3, 5
total_configs: 30
```

运行耗时：

```text
load_data: 1.087s
bm25_index: 4.391s
sweep: 266.613s
total: 272.092s
```

报告文件：

```text
artifacts/verified_rag_threshold_sweep_dev.json
```

### Key Results

固定 `retrieval_top_k=5`、`max_citation_rank=3` 时：

| min_evidence_score | answerable_refusal_rate | unanswerable_refusal_rate | gold_doc_citation_rate | average_token_f1 | newly_refused |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4.0 | 0.0063 | 0.0133 | 0.4906 | 0.2163 | 2 |
| 5.0 | 0.0500 | 0.0333 | 0.4934 | 0.2174 | 12 |
| 6.0 | 0.1250 | 0.0733 | 0.5143 | 0.2184 | 30 |
| 7.0 | 0.2062 | 0.1533 | 0.5276 | 0.2268 | 55 |
| 8.0 | 0.2875 | 0.2533 | 0.5263 | 0.2237 | 83 |

`retrieval_top_k` 对 gold 文档进入上下文有帮助：

```text
top5  answerable_gold_doc_in_context: 107 / 160
top10 answerable_gold_doc_in_context: 119 / 160
top20 answerable_gold_doc_in_context: 127 / 160
```

但是 `retrieval_top_k=10/20` 没有显著提升最终答案指标：

```text
top5  min_evidence_score=7.0 average_token_f1: 0.2268
top10 min_evidence_score=7.0 average_token_f1: 0.2268
top20 min_evidence_score=7.0 average_token_f1: 0.2268
```

`max_citation_rank=3` 和 `max_citation_rank=5` 在这次实验里结果完全相同或几乎相同，说明当前抽取器实际主要引用靠前文档，放宽引用排名没有带来收益。

### Example Interpretation

#### 1. `min_evidence_score=8.0` 太激进

```text
answerable_refusal_rate: 0.2875
unanswerable_refusal_rate: 0.2533
newly_refused: 83
possible_threshold_over_refusal_gold_cited: 18
```

它确实多拦了一些不可回答问题，但也误拒了较多可回答问题，且 18 条是 gold 文档已经被引用却被阈值挡掉。

#### 2. `min_evidence_score=4.0` 太宽松

```text
answerable_refusal_rate: 0.0063
unanswerable_refusal_rate: 0.0133
newly_refused: 2
```

它几乎不误拒可回答问题，但也几乎拦不住不可回答问题，和原始 RAG 差异很小。

#### 3. `min_evidence_score=7.0` 是当前较合理的折中点

```text
answerable_refusal_rate: 0.2062
unanswerable_refusal_rate: 0.1533
gold_doc_citation_rate: 0.5276
average_token_f1: 0.2268
newly_refused: 55
```

它不是绝对最优，只是当前指标下比较平衡：F1 和 gold citation rate 最高，同时比 8.0 少误拒很多可回答问题。

### Problems Encountered

- 阈值扫描如果每组参数都重新检索和生成，会非常慢。
- top10/top20 让更多 gold 文档进入上下文，但最终答案没有明显变好。
- `max_citation_rank` 放宽到 5 没有效果，说明当前失败不是“引用排名限制太严格”导致的。
- 不同目标下没有唯一最优阈值：
  1. 少误拒可回答问题：5.0 或 6.0 更保守。
  2. 提高不可回答拒答率：8.0 更强硬。
  3. 综合当前 F1 和 citation：7.0 更像折中点。

### Root Cause

- 当前抽取器的 evidence selection 仍然偏向关键词重合最高的句子，所以即使 gold 文档进入 top10/top20，也未必会被选中。
- `min_evidence_score` 只能过滤低重合证据，不能判断证据是否真正回答问题。
- `max_citation_rank` 没有效果，是因为生成器大多已经在 top3 文档内选句子。
- top_k 扩大后，检索召回上限提升了，但 answer selection 没有同步提升。

### Solution

- 增加阈值扫描脚本，并在报告中保留完整 30 组参数结果。
- 使用同一 `retrieval_top_k` 下的检索和原始答案缓存，避免每组阈值重复跑完整 RAG。
- 使用 Pareto candidate 标记多指标下没有被明显支配的配置，但不把它伪装成业务最优。
- 暂时把 `min_evidence_score=7.0` 作为下一阶段的候选阈值，而不是继续使用 8.0。

### Why This Choice

- 阈值校准让项目从“拍脑袋设规则”变成“用实验选择质量门控参数”。
- 这一步证明问题不只是阈值：top_k 增加带来了更多 gold 上下文，但答案仍然没有更好，说明 evidence selection 是下一瓶颈。
- 继续堆 Agent workflow 之前，先解决证据选择问题更有价值。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 25 个测试。
- `python scripts\sweep_verified_rag_thresholds.py` 已真实运行。
- `artifacts/verified_rag_threshold_sweep_dev.json` 已生成，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- 为什么 top10/top20 中新增的 gold 文档没有被抽取器利用，需要进一步看句子级候选分布。
- 是否应该把 evidence selection 从 document-level 改成 section/chunk-level。
- 是否需要为 answerability 单独训练一个 classifier，还是先用更好的 evidence selector。
- `min_evidence_score=7.0` 是否能在 NVIDIA TechQA-RAG-Eval 上保持类似趋势。

### Next Step

- 下一步不应该继续盲目扫阈值。
- 应该做 evidence selection analysis：
  1. 对比 top5/top10/top20 中 gold 文档新增但未被引用的样本。
  2. 检查 gold 文档里的句子为什么分数不如错误文档。
  3. 决定是做 section-level BM25 answer selection，还是做 query-aware sentence reranking。
  4. 再用 `min_evidence_score=7.0` 作为候选阈值复测。

## 2026-07-12 - Stage 8: Evidence Selection Analysis

### Goal

- 分析 `retrieval_top_k` 从 5 扩到 10 / 20 后，为什么新增进入上下文的 gold 文档没有被答案引用。
- 找出当前抽取式回答器的 evidence selection 失败机制。
- 判断下一步应该继续调阈值，还是改证据选择器。

### What I Studied

- 检索召回和证据选择是两层不同问题：
  1. 检索召回解决 gold 文档是否进入上下文。
  2. 证据选择解决 gold 文档进入上下文后是否被真正使用。
- `top_k` 变大只能提供更多候选文档，不能自动保证生成器选中正确证据。
- 简单 query-term overlap 会放大日志、错误码、模板化字段和长句子的影响。

### What I Built Or Changed

- 将抽取式回答器内部的候选句排序能力公开为可复用方法：
  `ExtractiveAnswerGenerator.rank_sentence_candidates`
- 新增候选证据句结构：
  `SentenceEvidenceCandidate`
- 新增 evidence selection 分析模块：
  `src/ts_rag_agent/application/evidence_selection_analysis.py`
- 新增 evidence selection 分析脚本：
  `scripts/analyze_evidence_selection.py`
- 新增测试：
  `tests/test_evidence_selection_analysis.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q
python scripts\analyze_evidence_selection.py
```

真实 dev 结果：

```text
top5:
gold_in_context: 107
gold_cited: 78
gold_in_context_not_cited: 29
gold_not_in_context: 53
gold_candidate_below_min_sentence_score: 3
gold_candidate_loses_to_wrong_sentences: 26

top10:
gold_in_context: 119
gold_cited: 78
gold_in_context_not_cited: 41
gold_newly_available_after_base_k: 12
gold_newly_available_but_not_cited: 12
gold_not_in_context: 41
gold_candidate_below_min_sentence_score: 6
gold_candidate_loses_to_wrong_sentences: 35

top20:
gold_in_context: 127
gold_cited: 78
gold_in_context_not_cited: 49
gold_newly_available_after_base_k: 20
gold_newly_available_but_not_cited: 20
gold_not_in_context: 33
gold_candidate_below_min_sentence_score: 11
gold_candidate_loses_to_wrong_sentences: 38
```

报告文件：

```text
artifacts/evidence_selection_analysis_dev.json
```

### Example Observations

#### 1. top-k 扩大带来 gold 文档，但没有带来 gold 引用

```text
top5  gold_in_context: 107, gold_cited: 78
top10 gold_in_context: 119, gold_cited: 78
top20 gold_in_context: 127, gold_cited: 78
```

这说明新增进入上下文的 gold 文档完全没有被当前抽取器利用。继续单纯扩大 `top_k` 不是有效方向。

#### 2. 日志类长句会压倒真正 gold 证据

```text
question_id: DEV_Q002
gold_doc: swg21978390
gold_retrieval_rank: 4
bucket: gold_candidate_loses_to_wrong_sentences
best_gold_candidate_rank: 6
best_gold_candidate_score: 7.7522
best_non_gold_candidate_score: 81.0
selected_doc_ids: swg21598554, swg21673044, swg1IC60317
```

错误文档中的日志长句包含大量 query token，例如错误码、模块名、路径、dump 字段，因此 overlap 分数极高。gold 文档虽然是真正答案来源，但候选句排名被挤到第 6。

#### 3. 相邻主题 FAQ 会抢走答案位置

```text
question_id: DEV_Q008
question: How can I export a private key from DataPower Gateway Appliance?
gold_doc: swg21412061
gold_retrieval_rank: 3
selected_doc_ids: swg21412060, swg21412060, swg21412060
best_gold_candidate_score: 3.0
best_non_gold_candidate_score: 5.0
```

gold 文档回答的是 HSM-enabled DataPower appliance 可以用 `crypto-export` 导出私钥；但相邻 FAQ 文档因为表面关键词更像问题，被抽取器优先引用。

#### 4. 新增进入 top10 的 gold 文档仍然被旧 top1 文档压制

```text
question_id: DEV_Q016
gold_doc: swg21502095
gold_retrieval_rank: 7
gold_newly_available_after_base_k: True
selected_doc_ids: swg21615508, swg21615508, swg21615508
best_gold_candidate_rank: 58
best_gold_candidate_score: 4.0
best_non_gold_candidate_score: 17.0
```

这个案例证明 top10 让 gold 文档进来了，但句子级选择器仍然强烈偏向 top1 错误文档。

### Problems Encountered

- `retrieval_top_k` 扩大后，gold 文档进入上下文的数量增加，但 `gold_cited` 完全没有增加。
- 当前抽取器把所有候选句放在同一个池子里按 overlap 排序，容易让错误文档中的高重合句子挤掉 gold 证据。
- 长日志、错误码、路径、产品模板字段会制造非常高的 overlap score。
- gold 文档有时候答案句更短、更概括，反而在简单 overlap 规则下得分低。

### Root Cause

- 当前 evidence score 使用的是 `overlap_terms_count / log2(retrieval_rank + 1)`，没有 IDF 权重，也没有句子长度、字段噪声、标题匹配、answer-like pattern 等特征。
- 候选句选择直接跨文档竞争，top1 错误文档可以用多个高分句子占满 `max_sentences=3`。
- 检索阶段提升的是文档级召回，但回答阶段没有 section-level / sentence-level 的独立检索机制。

### Solution

- 不再继续盲目扩大 `top_k`。
- 把当前问题定位为 evidence selection bottleneck。
- 下一步应实现一个更合理的 sentence / section selector：
  1. 使用 BM25 / IDF 风格的句子级打分，而不是纯 overlap。
  2. 限制同一文档占用过多答案句。
  3. 过滤日志噪声或降低日志长句权重。
  4. 比较新 selector 对 `gold_cited`、F1、拒答率的影响。

### Why This Choice

- 这个分析把“检索召回问题”和“证据选择问题”分开了。
- 如果不做这一步，可能会误以为 top_k、阈值、Agent workflow 能解决问题。
- 真实结果已经证明：gold 文档进入上下文不是终点，能否选中 gold 证据才是下一阶段关键。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 26 个测试。
- `python scripts\analyze_evidence_selection.py` 已真实运行。
- `artifacts/evidence_selection_analysis_dev.json` 已生成，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- 句子级 BM25 是否能显著提升 `gold_cited`。
- section-level selector 是否比 sentence-level selector 更适合 TechQA 文档。
- 是否应该引入去噪规则，例如降低 dump、trace、路径、十六进制错误码的权重。
- 如果限制每个文档最多贡献一句，是否会提高 gold citation，还是会损失多证据答案。

### Next Step

- 下一步实现一个改进版 evidence selector。
- 建议先做 `BM25SentenceEvidenceSelector`：
  1. 用 answerable dev 问题构建候选句或 section。
  2. 用 IDF 风格分数替代纯 overlap。
  3. 支持每个文档最多选择 N 句。
  4. 和当前 overlap selector 对比 `gold_cited`、`average_token_f1`、`unanswerable_refusal_rate`。

## 2026-07-12 - Stage 9: BM25 Sentence Evidence Selector

### Goal

- 验证改进版 `BM25SentenceEvidenceSelector` 是否真的解决 Stage 8 发现的 evidence selection bottleneck。
- 和旧的 `OverlapSentenceEvidenceSelector` 做同条件对比，而不是只看代码直觉。
- 判断是否可以进入 Agent workflow，还是仍然需要继续修 RAG 证据质量闭环。

### What I Studied

- evidence selector 的改进不能只看 `gold_cited`，还必须同时看答案质量和拒答行为。
- 句子级 BM25 / IDF 打分可以减少纯 overlap 对长日志、路径、dump、trace 文本的偏好。
- 但是更容易引用 gold 文档，不等于抽取出的句子一定更像标准答案。
- 不同 selector 的 evidence score 分布不同，同一个 `min_evidence_score=7.0` 不一定有相同含义。

### What I Built Or Changed

- 新增 evidence selection 抽象：
  `src/ts_rag_agent/application/evidence_selection.py`
- 将原来的 overlap 句子选择逻辑抽成：
  `OverlapSentenceEvidenceSelector`
- 新增：
  `BM25SentenceEvidenceSelector`
- 让 verified RAG 评估脚本支持 `--evidence-selector` 参数：
  `scripts/evaluate_verified_rag.py`
- 让 evidence selection 分析脚本支持 `--evidence-selector` 参数：
  `scripts/analyze_evidence_selection.py`
- 新增 evidence selector 单元测试：
  `tests/test_evidence_selection.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\evaluate_verified_rag.py `
  --evidence-selector overlap `
  --min-evidence-score 7.0 `
  --output artifacts\verified_rag_dev_overlap_m7_report.json

python scripts\analyze_evidence_selection.py `
  --evidence-selector overlap `
  --output artifacts\evidence_selection_analysis_dev_overlap.json

python scripts\evaluate_verified_rag.py `
  --evidence-selector bm25-sentence `
  --min-evidence-score 7.0 `
  --output artifacts\verified_rag_dev_bm25_sentence_m7_report.json

python scripts\analyze_evidence_selection.py `
  --evidence-selector bm25-sentence `
  --output artifacts\evidence_selection_analysis_dev_bm25_sentence.json
```

验证命令结果：

```text
ruff: passed
pytest: 28 passed
```

### Verified RAG Result

固定条件：

```text
dataset: PrimeQA/TechQA dev
retriever: BM25
retrieval_top_k: 5
min_sentence_score: 2.0
min_evidence_score: 7.0
max_citation_rank: 3
```

| Selector | verified gold_doc_citation_rate | verified average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate | newly_refused |
| --- | ---: | ---: | ---: | ---: | ---: |
| overlap | 0.5276 | 0.2268 | 0.2062 | 0.1533 | 55 |
| bm25-sentence | 0.6101 | 0.1887 | 0.0063 | 0.0067 | 1 |

原始未验证回答阶段：

| Selector | original gold_doc_citation_rate | original average_token_f1 |
| --- | ---: | ---: |
| overlap | 0.4875 | 0.2155 |
| bm25-sentence | 0.6125 | 0.1896 |

### Evidence Selection Result

| Selector | top_k | gold_in_context | gold_cited | gold_in_context_not_cited | gold_candidate_loses_to_wrong_sentences | below_min_sentence_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap | 5 | 107 | 78 | 29 | 26 | 3 |
| overlap | 10 | 119 | 78 | 41 | 35 | 6 |
| overlap | 20 | 127 | 78 | 49 | 38 | 11 |
| bm25-sentence | 5 | 107 | 98 | 9 | 9 | 0 |
| bm25-sentence | 10 | 119 | 90 | 29 | 29 | 0 |
| bm25-sentence | 20 | 127 | 87 | 40 | 40 | 0 |

### Problems Encountered

- `BM25SentenceEvidenceSelector` 明显提高了 gold 引用，但 `average_token_f1` 下降。
- `min_evidence_score=7.0` 对 bm25-sentence 几乎不再形成有效拒答门槛：
  `newly_refused` 从 overlap 的 55 降到 bm25-sentence 的 1。
- bm25-sentence 在 top5 下效果最好；top10/top20 的 `gold_cited` 反而下降，说明更多候选文档会重新引入干扰。
- gold 文档被引用更多，并不代表抽取出的 gold 句子一定覆盖标准答案。

### Root Cause

- 新 selector 的分数尺度和旧 overlap selector 不同，继续使用同一个 `min_evidence_score=7.0` 不公平，也不可靠。
- BM25 / IDF 更擅长把 gold 文档中的相关句子推到前面，但它仍然不知道“这个句子是否完整回答了问题”。
- 每个文档最多保留 1 个候选句减少了错误文档霸占答案的情况，但也可能损失同一 gold 文档中多个互补证据句。
- 当前评估的 gold citation 和 token F1 指向了两个不同问题：引用来源更准了，但答案文本仍然不够像标准答案。

### Solution

- 不把 Stage 9 结论写成“整体改进成功”，而是记录为“证据引用明显改善，但答案质量和拒答校准仍有问题”。
- 保留 bm25-sentence 作为新的 evidence selector 候选，因为它把 top5 的 `gold_cited` 从 78 提升到 98。
- 不进入 LangGraph / Agent 阶段，先继续做 evidence selector 的校准和质量分析。
- 下一步需要对 bm25-sentence 单独做阈值扫描，而不是沿用 overlap 的 `min_evidence_score=7.0`。

### Why This Choice

- 如果只看 `gold_doc_citation_rate`，会误判 bm25-sentence 已经全面胜出。
- 如果只看 `average_token_f1`，又会忽略它确实解决了 Stage 8 的一部分证据选择问题。
- 企业级 RAG 项目需要能解释 trade-off：引用正确性、答案质量、误拒率、漏拒率不一定同时改善。
- 这个阶段的价值是定位瓶颈，而不是急着把系统包装成 Agent。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 28 个测试。
- `verified_rag_dev_overlap_m7_report.json` 已生成。
- `verified_rag_dev_bm25_sentence_m7_report.json` 已生成。
- `evidence_selection_analysis_dev_overlap.json` 已生成。
- `evidence_selection_analysis_dev_bm25_sentence.json` 已生成。
- 以上实验产物都在 `artifacts/` 下，并被 `.gitignore` 忽略，不提交到 GitHub。

### What I Still Do Not Understand

- bm25-sentence 为什么提高 gold citation 后，token F1 反而下降。
- bm25-sentence 的合理 `min_evidence_score` 应该是多少。
- top10/top20 下为什么新增 gold 文档后 `gold_cited` 下降，需要检查新增候选文档的干扰类型。
- 是否应该允许同一 gold 文档贡献多个候选句，还是继续保持每文档最多 1 句。

### Next Step

- 对 bm25-sentence 单独做 threshold sweep / calibration。
- 扫描不同的 `min_evidence_score` 和 `max_candidates_per_document`。
- 对比：
  1. `gold_doc_citation_rate`
  2. `average_token_f1`
  3. `answerable_refusal_rate`
  4. `unanswerable_refusal_rate`
  5. `gold_cited`
- 如果 F1 仍然低，下一步不是继续调阈值，而是分析 bm25-sentence 选中的 gold 句子是否缺少答案完整性。

## 2026-07-12 - Stage 10: BM25 Sentence Calibration

### Goal

- 对 `BM25SentenceEvidenceSelector` 单独做阈值校准，而不是继续沿用 overlap selector 的分数阈值。
- 扫描不同的 `min_evidence_score` 和 `max_candidates_per_document`。
- 判断 bm25-sentence 能否在保持 gold citation 提升的同时，把 F1 和拒答行为拉回合理范围。

### What I Studied

- 同一个 `min_evidence_score` 不能跨 selector 直接比较，因为不同 selector 的 score scale 不同。
- `max_candidates_per_document` 控制的是同一篇文档最多贡献多少个候选证据句：
  1. 值小可以避免错误文档霸占多个答案句。
  2. 值大可以让 gold 文档贡献更多互补证据。
  3. 但值大也会引入更多冗余和噪声。
- 高阈值会提高拒答率和剩余答案的 gold citation rate，但这通常是通过拒掉大量样本换来的。

### What I Built Or Changed

- 给 threshold sweep 脚本增加 evidence selector 参数：
  `--evidence-selector`
- 给 threshold sweep 脚本增加每文档候选句上限参数：
  `--max-candidates-per-document`
- 在 sweep 报告中记录：
  1. `evidence_selector`
  2. `max_candidates_per_document`
- 默认输出文件名会带上 selector slug，避免覆盖不同 selector 的报告。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\sweep_verified_rag_thresholds.py `
  --evidence-selector bm25-sentence `
  --retrieval-top-k-values 5 `
  --min-evidence-scores 5,10,15,20,25,30,40,50,60,80,100 `
  --max-citation-ranks 3 `
  --max-candidates-per-document 1 `
  --output artifacts\verified_rag_threshold_sweep_dev_bm25_sentence_mcpd1.json

python scripts\sweep_verified_rag_thresholds.py `
  --evidence-selector bm25-sentence `
  --retrieval-top-k-values 5 `
  --min-evidence-scores 5,10,15,20,25,30,40,50,60,80,100 `
  --max-citation-ranks 3 `
  --max-candidates-per-document 2 `
  --output artifacts\verified_rag_threshold_sweep_dev_bm25_sentence_mcpd2.json

python scripts\sweep_verified_rag_thresholds.py `
  --evidence-selector bm25-sentence `
  --retrieval-top-k-values 5 `
  --min-evidence-scores 5,10,15,20,25,30,40,50,60,80,100 `
  --max-citation-ranks 3 `
  --max-candidates-per-document 3 `
  --output artifacts\verified_rag_threshold_sweep_dev_bm25_sentence_mcpd3.json
```

验证命令结果：

```text
ruff: passed
pytest: 28 passed
```

### Main Results

固定条件：

```text
dataset: PrimeQA/TechQA dev
retriever: BM25
retrieval_top_k: 5
evidence_selector: bm25_sentence
max_sentences: 3
max_citation_rank: 3
```

代表性结果：

| max_candidates_per_document | min_evidence_score | gold_doc_citation_rate | average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate | newly_refused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0.6125 | 0.1896 | 0.0000 | 0.0067 | 0 |
| 1 | 15 | 0.6169 | 0.1897 | 0.0375 | 0.0400 | 11 |
| 1 | 25 | 0.6412 | 0.1905 | 0.1812 | 0.1800 | 55 |
| 2 | 5 | 0.5723 | 0.2029 | 0.0063 | 0.0200 | 3 |
| 2 | 15 | 0.5817 | 0.2042 | 0.0437 | 0.0533 | 14 |
| 2 | 25 | 0.6000 | 0.2046 | 0.1875 | 0.1867 | 57 |
| 3 | 5 | 0.5660 | 0.2077 | 0.0063 | 0.0200 | 3 |
| 3 | 15 | 0.5752 | 0.2089 | 0.0437 | 0.0533 | 14 |
| 3 | 25 | 0.6000 | 0.2113 | 0.1875 | 0.1867 | 57 |

和 Stage 9 overlap baseline 对照：

| Selector | max_candidates_per_document | min_evidence_score | gold_doc_citation_rate | average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap | - | 7 | 0.5276 | 0.2268 | 0.2062 | 0.1533 |
| bm25-sentence | 1 | 15 | 0.6169 | 0.1897 | 0.0375 | 0.0400 |
| bm25-sentence | 3 | 15 | 0.5752 | 0.2089 | 0.0437 | 0.0533 |
| bm25-sentence | 3 | 25 | 0.6000 | 0.2113 | 0.1875 | 0.1867 |

### Problems Encountered

- `max_candidates_per_document=1` 保留了最高的 citation 优势，但 F1 始终偏低。
- `max_candidates_per_document=2/3` 能提升 F1，但会降低 gold citation。
- 高阈值看起来让 gold citation rate 更高，但主要原因是大量拒答，尤其会误拒很多 answerable 问题。
- 即使最佳 F1 配置 `max_candidates_per_document=3, min_evidence_score=25`，F1 也只有 0.2113，仍低于 overlap baseline 的 0.2268。

### Root Cause

- bm25-sentence 更擅长选到 gold 文档，但不一定选到最像标准答案的完整句子。
- 每文档 1 句限制太强，可能只拿到 gold 文档中的局部证据。
- 每文档 2/3 句能补充内容，但也会增加答案冗余，使 token precision 变差。
- 当前 verifier 只看 evidence score、citation rank 和 citation count，不理解答案是否真正覆盖了问题需求。

### Solution

- 不把 bm25-sentence calibration 记录为整体成功。
- 记录为：bm25-sentence 是更好的 citation-oriented selector，但还不是更好的 answer-quality selector。
- 如果目标是引用正确性，可以选 `max_candidates_per_document=1, min_evidence_score=15`。
- 如果目标是综合 F1 和引用，可以暂时把 `max_candidates_per_document=3, min_evidence_score=15` 作为下一阶段分析候选。
- 不继续盲目扫阈值，因为 F1 没有追上 overlap baseline。

### Why This Choice

- 项目需要真实解释 trade-off，而不是挑一个最好看的指标。
- 当前结果说明问题已经从“找不到 gold 文档”转移到“gold 文档中的答案句不够好”。
- 继续堆 Agent workflow 不能解决 evidence sentence 本身质量不足的问题。
- 下一步应该分析 bm25-sentence 选中的 gold 句子和标准答案之间的差距。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 28 个测试。
- `artifacts/verified_rag_threshold_sweep_dev_bm25_sentence_mcpd1.json` 已生成。
- `artifacts/verified_rag_threshold_sweep_dev_bm25_sentence_mcpd2.json` 已生成。
- `artifacts/verified_rag_threshold_sweep_dev_bm25_sentence_mcpd3.json` 已生成。
- 以上实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- bm25-sentence 选中的 gold 句子为什么和标准答案 token F1 仍然偏低。
- gold 文档里是否存在更完整的答案句，只是当前 selector 没选到。
- 是否应该从 sentence-level 改成 section-level，因为 TechQA 答案可能跨句。
- 是否应该引入 answer-aware reranking，而不是继续只用 query-to-sentence scoring。

### Next Step

- 做 Stage 11：BM25 sentence answer gap analysis。
- 对比 bm25-sentence 选中的 gold 句子和标准答案：
  1. token overlap 低的样本有哪些。
  2. gold 文档中是否存在更接近标准答案的句子。
  3. 当前 selector 选错是因为句子太短、太泛、还是答案跨句。
  4. 决定下一步做 section-level selector，还是 answer-aware reranker。

## 2026-07-12 - Stage 11: BM25 Sentence Answer Gap Analysis

### Goal

- 分析为什么 bm25-sentence 提高了 gold citation，但答案 token F1 仍然低。
- 对比当前选中的 evidence sentences 和 gold 文档中更接近标准答案的 sentence/window。
- 判断下一步应该做 section-level selector，还是 answer-aware reranker。

### What I Studied

- citation 正确不等于 answer span 正确。
- query-to-sentence scoring 容易选中标题、症状、日志、trace、错误描述，因为它们和问题关键词高度重合。
- TechQA 的标准答案很多来自 gold 文档中的 resolving / cause / answer 片段，而不是问题复述片段。
- 如果 gold 文档中存在高 F1 的连续句子窗口，但当前 selector 没选中，说明瓶颈是 answer-aware evidence selection。

### What I Built Or Changed

- 新增公共 token F1 工具：
  `src/ts_rag_agent/application/text_metrics.py`
- 将 `rag_answering.py` 里的私有 token F1 逻辑改为复用公共函数。
- 新增 answer gap 分析模块：
  `src/ts_rag_agent/application/answer_gap_analysis.py`
- 新增分析脚本：
  `scripts/analyze_answer_gap.py`
- 新增测试：
  `tests/test_answer_gap_analysis.py`

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\analyze_answer_gap.py `
  --evidence-selector bm25-sentence `
  --retrieval-top-k 5 `
  --max-candidates-per-document 3 `
  --output artifacts\answer_gap_analysis_dev_bm25_sentence_mcpd3.json
```

验证命令结果：

```text
ruff: passed
pytest: 31 passed
```

### Main Results

固定条件：

```text
dataset: PrimeQA/TechQA dev
retriever: BM25
retrieval_top_k: 5
evidence_selector: bm25_sentence
max_candidates_per_document: 3
max_sentences: 3
max_window_sentences: 3
```

结果：

```text
total_answerable_questions: 160
gold_document_available: 160
gold_in_context: 107
selected_gold_citation: 91

average_selected_answer_token_f1: 0.2159
average_best_gold_sentence_token_f1: 0.7349
average_best_gold_window_token_f1: 0.8881
```

Bucket 分布：

| Bucket | Count |
| --- | ---: |
| gold_not_in_context | 53 |
| gold_in_context_not_selected | 16 |
| gold_window_beats_selected_answer | 91 |
| gold_sentence_beats_selected_answer | 0 |
| selected_answer_low_overlap | 0 |
| selected_answer_reasonable_overlap | 0 |
| gold_document_missing | 0 |

### Example Observations

#### 1. 选中了症状，但 gold 文档里的解决方案更接近答案

```text
question_id: DEV_Q002
selected_answer_token_f1: 0.0213
best_gold_sentence_f1: 0.7568
best_gold_window_f1: 0.9333
```

当前 selector 选中了 `PROBLEM(ABSTRACT)`、dump、javacore 片段，但标准答案对应的是：

```text
RESOLVING THE PROBLEM Install the missing libraries ...
```

#### 2. 选中了问题复述，但 gold 文档里的 resolving 段落几乎就是标准答案

```text
question_id: DEV_Q012
selected_answer_token_f1: 0.0
best_gold_sentence_f1: 0.619
best_gold_window_f1: 0.9735
```

当前 selector 选中了错误标题和症状：

```text
Argument list too long
```

但 gold 文档中的 resolving window 更接近标准答案。

#### 3. gold 文档进入上下文后，仍可能被相邻主题或日志句子干扰

```text
question_id: DEV_Q258
selected_answer_token_f1: 0.0308
best_gold_sentence_f1: 1.0
best_gold_window_f1: 1.0
```

gold 文档中存在完全匹配标准答案的句子，但 selector 仍选了告警、workspace、application health 等相邻上下文。

### Problems Encountered

- bm25-sentence 可以找到 gold 文档，但仍倾向于选 query-overlap 最高的句子。
- 这些高 overlap 句子往往是标题、症状、日志、trace 或错误消息，不是答案。
- gold 文档里的答案 span 往往带有 `RESOLVING THE PROBLEM`、`CAUSE`、`ANSWER` 等结构信号。
- 单纯扩大 sentence window 只能说明 gold 文档有答案，但不会自动让 selector 学会选答案段。

### Root Cause

- 当前 selector 是 query-aware，不是 answer-aware。
- 它只知道“这句话像不像问题”，不知道“这句话是不是在回答问题”。
- 标准答案更像 gold 文档中的解决方案段、原因段或结论段，而不是和问题表面词最像的句子。
- 因此，citation-oriented selector 解决了“引用哪篇文档”的一部分问题，但没有解决“引用文档里的哪段答案”的问题。

### Solution

- 不继续盲目调 `min_evidence_score` 或 `max_candidates_per_document`。
- 将下一阶段目标从 sentence scoring 改为 answer-aware span selection。
- 优先考虑 section-level 或 answer-aware reranking，而不是直接进入 Agent。
- 继续保留 bm25-sentence 作为 document/citation 候选来源，但不能把它当最终 answer selector。

### Why This Choice

- Stage 11 证明了 gold 文档里有高质量答案 span：平均 best gold window F1 达到 0.8881。
- 当前 selected answer 平均 F1 只有 0.2159，说明差距主要来自 span selection，而不是数据缺答案。
- 如果现在进入 Agent workflow，只会把错误 evidence 包装成更复杂的流程。
- 更合理的下一步是让系统学会选“回答型片段”，再谈 Agent 编排。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 31 个测试。
- `artifacts/answer_gap_analysis_dev_bm25_sentence_mcpd3.json` 已生成。
- 实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- 使用 section-level selector 能否直接命中 resolving/cause/answer 段。
- 是否需要显式识别文档结构标题，例如 `RESOLVING THE PROBLEM`、`CAUSE`、`ANSWER`。
- answer-aware reranker 是先用规则特征实现，还是接一个小模型/LLM judge。
- 如果用 gold answer 做 oracle 分析，如何避免把 oracle 结果误当作真实可部署方法。

### Next Step

- 做 Stage 12：Answer-aware section/span selector prototype。
- 第一版先不接 LLM，先做规则和结构特征：
  1. 给 resolving/cause/answer 类标题附近的句子加权。
  2. 降低 symptom/problem/trace/dump/error log 类句子的权重。
  3. 在 gold-in-context 的样本上比较 selected answer F1。
  4. 再决定是否做 learned reranker 或 LLM judge。

## 2026-07-12 - Stage 12: Answer-Aware Section/Span Selector Prototype

### Goal

- 做一个不依赖 LLM 的 answer-aware evidence selector 原型。
- 利用 TechQA 文档中的结构信号，把真正回答问题的段落排到更前面。
- 验证它是否能提升 answer quality，而不只是提升 gold citation。

### What I Studied

- TechQA 文档里存在明显的结构字段和段落信号，例如：
  `RESOLVING THE PROBLEM`、`ANSWER`、`CAUSE`、`WORKAROUND`。
- 旧的 bm25-sentence selector 主要是 query-aware：更容易选中问题复述、症状、错误日志、trace 和 dump。
- answer-aware selector 需要区分“这句话像问题”和“这句话在回答问题”。
- 第一版不能用 gold answer 参与打分，否则会变成 oracle，不是可部署方法。

### What I Built Or Changed

- 新增：
  `AnswerAwareBM25SentenceEvidenceSelector`
- 支持稳定 selector 名称：
  `answer-aware`
- 更新 selector factory，使已有脚本都能直接使用：
  `--evidence-selector answer-aware`
- 对答案型结构加权：
  1. `RESOLVING THE PROBLEM`
  2. `ANSWER`
  3. `CAUSE`
  4. `WORKAROUND`
  5. `FIX`
- 对问题复述和日志型结构降权：
  1. `PROBLEM(ABSTRACT)`
  2. `PROBLEM SUMMARY`
  3. `SYMPTOM`
  4. `trace`
  5. `dump`
  6. `exception`
  7. `javacore`
  8. `0SECTION` / `1XHEXC`
- 修复了一个真实规则 bug：最初的正则没有正确命中 `PROBLEM(ABSTRACT)`，导致问题复述仍然排在答案段前面。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\evaluate_verified_rag.py `
  --evidence-selector answer-aware `
  --retrieval-top-k 5 `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --output artifacts\verified_rag_dev_answer_aware_m15_report.json

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --retrieval-top-k 5 `
  --max-candidates-per-document 3 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3.json
```

验证命令结果：

```text
ruff: passed
pytest: 33 passed
```

### Main Results

对比 verified RAG 指标：

| Selector | Phase | gold_doc_citation_rate | average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| overlap | original | 0.4875 | 0.2155 | 0.0000 | 0.0067 |
| overlap | verified | 0.5276 | 0.2268 | 0.2062 | 0.1533 |
| bm25-sentence mcpd=3 | verified, score=15 | 0.5752 | 0.2089 | 0.0437 | 0.0533 |
| bm25-sentence mcpd=3 | verified, score=25 | 0.6000 | 0.2113 | 0.1875 | 0.1867 |
| answer-aware mcpd=3 | original | 0.5625 | 0.2427 | 0.0000 | 0.0000 |
| answer-aware mcpd=3 | verified, score=15 | 0.5696 | 0.2449 | 0.0125 | 0.0267 |

Answer-gap 对比：

| Selector | selected_gold_citation | average_selected_answer_token_f1 | average_best_gold_window_token_f1 |
| --- | ---: | ---: | ---: |
| bm25-sentence mcpd=3 | 91 | 0.2159 | 0.8881 |
| answer-aware mcpd=3 | 90 | 0.2501 | 0.8881 |

### Example Debugging Note

在 `DEV_Q002` 上，旧 bm25-sentence 把 `PROBLEM(ABSTRACT)`、dump、javacore 片段排在真正解决方案前面。

第一次 answer-aware 实现仍失败，因为 `PROBLEM(ABSTRACT)` 没有被正则正确命中，导致问题复述没有被降权。

修复后，gold 文档中的解决方案句：

```text
RESOLVING THE PROBLEM Install the missing libraries ...
```

从候选排名第 11 提升到第 2。

### Problems Encountered

- 只给答案型段落加权不够，还必须强力压低问题复述和日志型文本。
- `RESOLVING THE PROBLEM` 这类信号不只出现在 gold 文档，也会出现在相邻错误文档中。
- answer-aware 的 F1 明显提升，但 selected_gold_citation 从 91 小降到 90，说明结构信号可能把部分非 gold 文档中的答案段推高。
- answer-gap 仍显示 90 个样本里 gold window 明显优于当前答案，说明规则原型还远没到 oracle span。

### Root Cause

- 文档结构信号能帮助找到“回答型文本”，但不能判断该答案段是否对应当前问题。
- 当前 selector 仍然主要是规则加权，没有真正做 query-answer matching。
- TechQA 的答案常常是多个句子组合，单句或固定 3 句抽取仍然粗糙。
- 只靠 handcrafted rules 很难同时兼顾 citation、F1、拒答和跨文档干扰。

### Solution

- 保留 answer-aware selector，因为它是目前 F1 最好的非 LLM 原型。
- 不把它包装成最终方案：它只是证明“结构信号有效”。
- 下一步应该继续做更细的 section/span selection，而不是回到阈值搜索。
- 后续可以把 answer-aware 作为 reranker 的 feature baseline。

### Why This Choice

- answer-aware 是当前第一个超过 overlap baseline F1 的方案：
  0.2449 vs 0.2268。
- 它的 answerable refusal rate 只有 0.0125，明显低于 overlap verified 的 0.2062。
- 它证明 Stage 11 的判断是对的：问题主要在 answer span selection。
- 但 gold window oracle 仍有 0.8881，说明还有巨大上限空间。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 33 个测试。
- `artifacts/verified_rag_dev_answer_aware_m15_report.json` 已生成。
- `artifacts/answer_gap_analysis_dev_answer_aware_mcpd3.json` 已生成。
- 实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- answer-aware 规则提升 F1 的样本主要集中在哪些问题类型。
- 结构信号导致非 gold 文档答案段被推高的样本有多少。
- 是否应该先做 section-level candidate，再在 section 内选句。
- 如果引入 reranker，应该使用规则特征、传统 ML，还是 LLM judge。

### Next Step

- 做 Stage 13：section-level answer span selector。
- 第一版思路：
  1. 按文档结构切 section，而不是只切 sentence。
  2. 对 resolving/cause/answer section 做候选召回。
  3. 在 section 内做 sentence/window selection。
  4. 对比 answer-aware sentence selector 是否继续提升 F1。

## 2026-07-13 - Stage 13: Section-Level Answer Span Selector

### Goal

- 实现一个 section-level answer span selector 原型。
- 从“全篇文档句子打分”升级为“先识别 section，再在 section 内选择句子/window”。
- 验证它是否能缩小 Stage 12 发现的 answer-gap。

### What I Studied

- section/window 选择不等于自然提升 F1。
- answer-like section 在 gold 文档和相邻错误文档中都会出现，因此结构信号需要和 query matching 一起使用。
- 多句 window 可能包含更完整答案，也可能带来更多噪声。
- 对 section-span selector 来说，每个文档最多贡献多少个候选 span 是一个强影响参数。

### What I Built Or Changed

- 新增 `SectionSpanBM25SentenceEvidenceSelector`。
- 支持命令行参数：
  `--evidence-selector section-span`
- 新 selector 会：
  1. 按 TechQA 文档中的 section heading 切分文本。
  2. 在 section 内生成 1 到 3 句连续 window。
  3. 使用 section heading 参与打分，但不把 heading 拼进最终答案文本。
  4. 给 CVE/CVSS、restriction、limitation、enhancement 等答案型 span 加权。
  5. 给 SUMMARY、affected-products 这类背景段降权。
- 更新脚本 help 文案：
  `evaluate_verified_rag.py`
  `analyze_evidence_selection.py`
  `analyze_answer_gap.py`
  `sweep_verified_rag_thresholds.py`
- 新增 section-span 相关测试。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\evaluate_verified_rag.py `
  --evidence-selector section-span `
  --max-candidates-per-document 1 `
  --min-evidence-score 15 `
  --output artifacts\verified_rag_dev_section_span_mcpd1_v2_report.json

python scripts\evaluate_verified_rag.py `
  --evidence-selector section-span `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --output artifacts\verified_rag_dev_section_span_mcpd3_v2_report.json

python scripts\analyze_answer_gap.py `
  --evidence-selector section-span `
  --max-candidates-per-document 1 `
  --output artifacts\answer_gap_analysis_dev_section_span_mcpd1_v2.json
```

验证命令结果：

```text
ruff: passed
pytest: 36 passed
```

### Main Results

对比 verified RAG 指标：

| Selector | Config | gold_doc_citation_rate | average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| answer-aware | mcpd=3, score=15 | 0.5696 | 0.2449 | 0.0125 | 0.0267 |
| section-span v1 | mcpd=1, score=15 | 0.5687 | 0.2129 | 0.0000 | 0.0000 |
| section-span v2 | mcpd=1, score=15 | 0.5750 | 0.2378 | 0.0000 | 0.0000 |
| section-span v2 | mcpd=3, score=15 | 0.4088 | 0.2041 | 0.1437 | 0.1467 |

Answer-gap 结果：

| Selector | Config | selected_gold_citation | average_selected_answer_token_f1 | average_best_gold_window_token_f1 |
| --- | --- | ---: | ---: | ---: |
| answer-aware | mcpd=3 | 90 | 0.2501 | 0.8881 |
| section-span v1 | mcpd=1 | 91 | 0.2181 | 0.8881 |
| section-span v2 | mcpd=1 | 92 | 0.2432 | 0.8881 |

### Problems Encountered

- 第一版 section-span 明显低于 answer-aware：F1 只有 0.2129。
- `max_candidates_per_document=3` 会让错误文档中的 answer-like windows 霸占候选，结果比 `mcpd=1` 更差。
- `max_sentences=1` 没有解决问题，反而导致 gold citation 和拒答表现变差。
- 安全公告类问题中，正确答案常在 `CVEID`、`DESCRIPTION`、`CVSS Base Score` 附近，而不是普通 `RESOLVING THE PROBLEM` section。
- section heading 本身只能说明“这里像答案区”，不能说明“这里回答了当前问题”。

### Root Cause

- section-span v1 太相信 section 结构，忽略了不同问题类型的答案形态。
- TechQA 中有多种答案格式：
  1. resolving/problem-solution 型。
  2. cause/limitation 型。
  3. CVE/CVSS security bulletin 型。
  4. RFE/enhancement 型。
  5. 附件/链接型。
- 同一个错误文档也可能包含高质量 answer section，因此只靠 heading 会推高错误答案段。
- 多 window 拼接会提高召回，但也会降低答案精度。

### Solution

- 保留 section-span 原型，但不把它替代 answer-aware。
- 将 `max_candidates_per_document=1` 作为当前 section-span 的更合理配置。
- 加入少量 answer-span pattern：
  `CVEID`
  `CVSS Base Score`
  `restriction`
  `limitation`
  `enhancement`
- 对 `SUMMARY`、`affected products and versions`、`remediation/fixes` 做背景段降权。

### Why This Choice

- v2 证明 section/span 方向有价值：F1 从 0.2129 提到 0.2378，selected gold citation 从 91 到 92。
- 但它仍没有超过 answer-aware 的 0.2449，所以不能宣称 Stage 13 成功替代 Stage 12。
- 当前最强非 LLM 原型仍然是 answer-aware selector。
- section-span 更适合作为下一版 hybrid selector 或 reranker feature，而不是直接作为主 selector。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 36 个测试。
- `artifacts/verified_rag_dev_section_span_mcpd1_v2_report.json` 已生成。
- `artifacts/verified_rag_dev_section_span_mcpd3_v2_report.json` 已生成。
- `artifacts/answer_gap_analysis_dev_section_span_mcpd1_v2.json` 已生成。
- 实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- section-span v2 提升的样本主要是哪几类问题。
- answer-aware 和 section-span 是否能融合，而不是二选一。
- 是否应该让 section-span 只在特定问题类型上启用，例如 security bulletin 或 limitation 类问题。
- 是否需要一个 learned reranker 来学习 query-answer matching，而不是继续手写规则。

### Next Step

- 不要继续单独堆 section-span 规则。
- 下一步更合理的是做 selector comparison analysis：
  1. 找出 answer-aware 赢、section-span 输的样本。
  2. 找出 section-span 赢、answer-aware 输的样本。
  3. 按问题类型分类，例如 CVE、安全公告、limitation、how-to、附件/链接。
  4. 决定是否做 hybrid selector，还是转向 learned reranker / LLM judge。

## 2026-07-13 - Stage 14: Selector Comparison Analysis

### Goal

- 不再只看整体平均 F1。
- 对比 answer-aware 和 section-span 在每个问题上的胜负。
- 按问题类型拆解，判断 section-span 是否适合做全局主 selector，还是只适合特定类型。

### What I Studied

- 一个 selector 整体平均分较低，不代表它没有价值。
- 如果两个 selector 在不同问题类型上互补，下一步应该考虑 hybrid routing，而不是继续单点调参。
- 只保存 30 条样例的 answer-gap 报告不适合做严肃比较，需要重新生成全量 case 报告。

### What I Built Or Changed

- 新增 selector comparison 分析模块：
  `src/ts_rag_agent/application/selector_comparison_analysis.py`
- 新增脚本：
  `scripts/compare_selectors.py`
- 新增测试：
  `tests/test_selector_comparison_analysis.py`
- 脚本支持：
  1. 读取两个 answer-gap JSON。
  2. 按 question_id 对齐。
  3. 计算 F1 delta。
  4. 判断 baseline win / challenger win / tie。
  5. 基于标题、问题文本、gold answer 做粗粒度 question type 分类。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_full.json

python scripts\analyze_answer_gap.py `
  --evidence-selector section-span `
  --max-candidates-per-document 1 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_section_span_mcpd1_v2_full.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_full.json `
  --challenger-report artifacts\answer_gap_analysis_dev_section_span_mcpd1_v2_full.json `
  --baseline-label answer-aware `
  --challenger-label section-span `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_section_span.json
```

验证命令结果：

```text
ruff: passed
pytest: 38 passed
```

### Main Results

整体胜负：

```text
total_compared: 160
answer-aware wins: 68
section-span wins: 47
ties: 45
avg_f1_delta: -0.007
answer-aware gold citations: 90
section-span gold citations: 92
```

按问题类型统计：

| Question Type | Count | answer-aware wins | section-span wins | ties |
| --- | ---: | ---: | ---: | ---: |
| security_bulletin | 22 | 5 | 15 | 2 |
| limitation_or_restriction | 8 | 0 | 4 | 4 |
| how_to_or_lookup | 14 | 5 | 7 | 2 |
| install_upgrade_config | 26 | 12 | 6 | 8 |
| error_or_log | 24 | 11 | 3 | 10 |
| attachment_or_link | 19 | 5 | 4 | 10 |
| other | 47 | 30 | 8 | 9 |

### Example Observations

- section-span 明显赢在 security bulletin：
  - `DEV_Q098`
  - `DEV_Q129`
  - `DEV_Q182`
  - 这些样本里 `CVEID`、`DESCRIPTION`、`CVSS Base Score` 规则能把答案 span 从 SUMMARY 背景段里拉出来。
- answer-aware 在普通排障和配置类问题上更稳：
  - `DEV_Q071`
  - `DEV_Q120`
  - `DEV_Q106`
  - 这些样本里 section-span 容易选择过长的说明段或步骤段，F1 被稀释。
- section-span 的 gold citation 稍高，但平均 F1 稍低，说明它更常引用 gold 文档，却不一定选中最短、最贴近 gold answer 的 span。

### Problems Encountered

- answer-gap 原始报告默认只保存样例，不是全量结果，因此不能直接做可靠 selector comparison。
- question type 分类目前是规则型粗分类，可能会有误分。
- F1 win margin 的选择会影响胜负统计，本次使用 `0.03` 作为最小胜出差距。
- section-span 的优势集中在安全公告和限制类问题，但它在普通问题上拖累整体平均值。

### Root Cause

- 不同问题类型的答案形态不同：
  1. security bulletin 依赖 `CVEID`、`DESCRIPTION`、`CVSS`。
  2. limitation/restriction 依赖短 cause/limitation span。
  3. 普通 how-to / troubleshooting 更依赖 resolving/problem-solution 结构和具体操作句。
- 单一 selector 很难同时覆盖所有类型。
- section-span 的结构粒度更强，但 query-answer matching 仍然较弱。

### Solution

- 不把 section-span 作为全局替代方案。
- 保留 answer-aware 作为默认主 selector。
- 把 section-span 作为特定问题类型的候选 selector，尤其是：
  1. security bulletin。
  2. limitation_or_restriction。
  3. 部分 how_to_or_lookup。
- 下一步应该做 hybrid selector routing，而不是继续单独调 section-span。

### Why This Choice

- 整体上 answer-aware 仍然赢：68 胜 vs 47 胜。
- 但 section-span 在 security_bulletin 上优势很明显：15 胜 vs 5 胜。
- 这说明项目已经进入“按问题类型路由 selector”的阶段，而不是继续寻找一个万能规则。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 38 个测试。
- `artifacts/answer_gap_analysis_dev_answer_aware_mcpd3_full.json` 已生成。
- `artifacts/answer_gap_analysis_dev_section_span_mcpd1_v2_full.json` 已生成。
- `artifacts/selector_comparison_answer_aware_vs_section_span.json` 已生成。
- 实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- question type 分类是否足够可靠，是否需要单独评估分类准确性。
- section-span 在 security bulletin 上的优势是否来自 CVE/CVSS 规则，还是来自 window 选择。
- hybrid routing 应该按问题类型硬规则切换，还是让两个 selector 都产出候选后做 late fusion。
- 如果未来引入 learned reranker，应不应该把 question type、section type、selector score 都作为特征。

### Next Step

- 做 Stage 15：hybrid selector routing 原型。
- 第一版不要训练模型，先做规则路由：
  1. security_bulletin -> section-span。
  2. limitation_or_restriction -> section-span。
  3. 其他类型 -> answer-aware。
- 跑 verified RAG 和 answer-gap，比较：
  1. 是否超过 answer-aware 的 F1 0.2449。
  2. 是否保持低 answerable refusal。
  3. 是否保留 section-span 在 security bulletin 上的优势。
