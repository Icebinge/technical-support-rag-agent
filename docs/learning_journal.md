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

## 2026-07-13 - Stage 15: Hybrid Selector Routing Prototype

### Goal

- 基于 Stage 14 的 selector comparison，做一个不训练模型的 hybrid selector。
- 对安全公告和限制类问题使用 section-span。
- 对其他问题继续使用 answer-aware。
- 验证 hybrid 是否能超过当前最强非 LLM baseline：answer-aware F1 0.2449。

### What I Studied

- selector routing 不能使用 gold answer，否则会发生 evaluation leakage。
- Stage 14 的 question type analysis 可以使用 gold answer 做错误分析，但 Stage 15 的真实路由只能看 question title/text。
- 如果 hybrid 只是把两个 selector 硬切换，最关键的是不要伤害原本 answer-aware 擅长的普通问题。

### What I Built Or Changed

- 新增 `HybridRoutingEvidenceSelector`。
- 支持命令行参数：
  `--evidence-selector hybrid-routing`
- 新增无泄漏路由函数：
  `classify_question_route`
- 路由规则：
  1. `security_bulletin` -> `SectionSpanBM25SentenceEvidenceSelector`
  2. `limitation_or_restriction` -> `SectionSpanBM25SentenceEvidenceSelector`
  3. 其他类型 -> `AnswerAwareBM25SentenceEvidenceSelector`
- 具体配置：
  - answer-aware: `max_candidates_per_document=3`
  - section-span: `max_candidates_per_document=1`
- 更新脚本 help 文案。
- 新增测试，验证：
  1. security bulletin 走 section-span。
  2. 普通问题走 answer-aware。
  3. 路由分类不读取 gold answer。
  4. factory 可以创建 hybrid selector。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_report.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_full.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_full.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_full.json `
  --baseline-label answer-aware `
  --challenger-label hybrid-routing `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_hybrid_routing.json
```

验证命令结果：

```text
ruff: passed
pytest: 42 passed
```

### Main Results

Verified RAG 对比：

| Selector | gold_doc_citation_rate | average_token_f1 | answerable_refusal_rate | unanswerable_refusal_rate |
| --- | ---: | ---: | ---: | ---: |
| answer-aware | 0.5696 | 0.2449 | 0.0125 | 0.0267 |
| section-span v2 | 0.5750 | 0.2378 | 0.0000 | 0.0000 |
| hybrid-routing | 0.5823 | 0.2715 | 0.0125 | 0.0267 |

Answer-gap：

| Selector | selected_gold_citation | average_selected_answer_token_f1 | average_best_gold_window_token_f1 |
| --- | ---: | ---: | ---: |
| answer-aware | 90 | 0.2501 | 0.8881 |
| section-span v2 | 92 | 0.2432 | 0.8881 |
| hybrid-routing | 92 | 0.2764 | 0.8881 |

Answer-aware vs hybrid-routing comparison：

```text
total_compared: 160
answer-aware wins: 5
hybrid-routing wins: 16
ties: 139
avg_f1_delta: +0.0263
answer-aware gold citations: 90
hybrid-routing gold citations: 92
```

按问题类型：

| Question Type | Count | answer-aware wins | hybrid-routing wins | ties |
| --- | ---: | ---: | ---: | ---: |
| security_bulletin | 22 | 5 | 15 | 2 |
| limitation_or_restriction | 8 | 0 | 1 | 7 |
| error_or_log | 24 | 0 | 0 | 24 |
| install_upgrade_config | 26 | 0 | 0 | 26 |
| how_to_or_lookup | 14 | 0 | 0 | 14 |
| attachment_or_link | 19 | 0 | 0 | 19 |
| other | 47 | 0 | 0 | 47 |

### Problems Encountered

- Stage 14 的 question type 分类使用了 gold answer，因此不能直接复用到在线 selector routing。
- limitation/restriction 类在 analysis 中有 8 条，但真实路由只靠 title/text，能捕捉到的显式 limitation/restriction 更少。
- 当前 hybrid 是硬规则路由，不是 learned routing。
- hybrid 的提升主要来自 security bulletin，对其他类型几乎保持 answer-aware 原样。

### Root Cause

- Security bulletin 的问题标题通常显式包含 `Security Bulletin`、`CVE`、`CVSS`，因此可以无泄漏识别。
- Limitation/restriction 有些只在 gold answer 中体现，问题文本未必显式写出，所以无泄漏路由较难捕捉。
- Answer-aware 本身已经覆盖普通 troubleshooting / configuration 问题，因此 hybrid 只需要在特定类型切换 selector。

### Solution

- 保留一个非常保守的 routing 原型：
  - 明确安全公告 -> section-span。
  - 明确 limitation/restriction -> section-span。
  - 其他全部 -> answer-aware。
- 不引入任何 gold answer 信息。
- 不让 section-span 全局接管，避免破坏普通问题。

### Why This Choice

- Hybrid-routing 首次显著超过当前最强非 LLM baseline：
  - F1: 0.2715 vs 0.2449
  - gold citation: 0.5823 vs 0.5696
  - answerable refusal: 0.0125，与 answer-aware 持平
- 对比分析显示 hybrid 的收益集中在 security bulletin，不会大面积伤害其他类型。
- 这说明 Stage 14 的结论是正确的：不是继续寻找万能 selector，而是按问题类型路由。

### Verification

- `python -m ruff check .` 通过。
- `python -m pytest -q` 通过，当前为 42 个测试。
- `artifacts/verified_rag_dev_hybrid_routing_m15_report.json` 已生成。
- `artifacts/answer_gap_analysis_dev_hybrid_routing_mcpd3_full.json` 已生成。
- `artifacts/selector_comparison_answer_aware_vs_hybrid_routing.json` 已生成。
- 实验产物在 `artifacts/` 下，并被 `.gitignore` 忽略。

### What I Still Do Not Understand

- limitation/restriction 的无泄漏识别是否可以做得更好。
- hybrid routing 是否需要输出 route reason，方便后续调试。
- 是否应该把 selector route 记录进每条 answer-gap case，便于分析每个问题实际走了哪个 selector。
- 是否应该进入 learned reranker，还是继续做 route-aware hard rules。

### Next Step

- 做 Stage 16：route trace and hybrid failure analysis。
- 重点不是继续调 F1，而是让 hybrid 可解释：
  1. 在每条结果里记录 question route。
  2. 统计 route 分布。
  3. 分析 hybrid 仍然输给 answer-aware 的 5 条样本。
  4. 判断是否需要 route override 或 learned reranker。

## 2026-07-13 - Stage 16: Route Trace And Hybrid Failure Analysis

### Goal

- 给 hybrid-routing 增加可解释 trace。
- 在 answer-gap 报告中记录每条问题实际走了哪个 route 和 selector。
- 分析 hybrid-routing 仍然输给 answer-aware 的 5 条样本。
- 判断下一步是否应该做 route override、继续硬规则，还是进入 learned reranker。

### What I Studied

- hybrid 的整体指标提升不等于每个 route 都可靠。
- 如果没有 route trace，很难解释某条样本为什么走了 section-span 或 answer-aware。
- route trace 必须无泄漏：只能依赖 question title/text 和 selector name，不能依赖 gold answer。

### What I Built Or Changed

- 新增 `SelectorRouteTrace`。
- 新增 `trace_selector_route`。
- `HybridRoutingEvidenceSelector` 内部改为复用统一 trace 逻辑。
- `AnswerGapCase` 新增：
  - `question_route`
  - `selected_selector_name`
  - `route_reason`
- `AnswerGapSummary` 新增：
  - `question_route_counts`
  - `selected_selector_counts`
- `selector_comparison_analysis` 新增：
  - baseline/challenger route 字段
  - challenger route 分布
  - challenger selected selector 分布
  - challenger route win counts
- 新增和更新 route trace 相关测试。

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_route_full.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_route_full.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_route_full.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_route_full.json `
  --baseline-label answer-aware `
  --challenger-label hybrid-routing `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_hybrid_routing_route.json
```

验证命令结果：

```text
ruff: passed
pytest: 43 passed
```

### Main Results

Hybrid answer-gap route distribution:

```text
question_route_counts:
install_upgrade_config: 24
limitation_or_restriction: 5
other: 71
how_to_or_lookup: 14
error_or_log: 26
security_bulletin: 20

selected_selector_counts:
answer_aware_bm25_sentence: 135
section_span_bm25_sentence: 25
```

Answer-aware vs hybrid-routing comparison:

```text
total_compared: 160
answer-aware wins: 5
hybrid-routing wins: 16
ties: 139
avg_f1_delta: +0.0263
```

Route-level win counts:

```text
security_bulletin:
answer-aware wins: 5
hybrid-routing wins: 15
ties: 0

limitation_or_restriction:
answer-aware wins: 0
hybrid-routing wins: 1
ties: 4

all other no-leak routes:
ties only
```

### Hybrid Loss Cases

Hybrid still loses to answer-aware on 5 cases, all routed as:

```text
question_route: security_bulletin
selected_selector_name: section_span_bm25_sentence
```

The 5 cases are:

```text
DEV_Q307
DEV_Q089
DEV_Q019
DEV_Q220
DEV_Q260
```

Observed failure patterns:

- Some security bulletin questions ask about affected product/version, but section-span over-prioritizes CVE detail windows.
- Some documents contain translated or reference-link sections; section-span can pick those instead of the actual answer.
- Some questions are security-related but their gold answer is a remediation or post-fix behavior explanation, where answer-aware is better.
- The current route treats all security bulletin questions the same, which is too coarse.

### Problems Encountered

- Stage 15 showed hybrid wins overall, but without route trace it was not clear where wins/losses came from.
- `security_bulletin` is not a single homogeneous type:
  1. vulnerability detail questions
  2. affected product/version questions
  3. remediation/fix questions
  4. post-fix behavior questions
  5. translated/reference bulletin pages
- section-span is good at CVE/CVSS detail extraction but not always good at affected product/remediation answers.

### Root Cause

- The first hybrid route is too coarse:
  `security_bulletin -> section-span`
- It ignores the sub-intent inside security bulletin questions.
- The route classifier sees the question, but it does not yet decide whether the user is asking for vulnerability details, affected products, remediation, or crash/fix behavior.

### Solution

- Keep hybrid-routing as the current best non-LLM baseline.
- Do not replace section-span globally.
- Do not blindly route all security bulletin questions to section-span in the final design.
- Use Stage 16 trace to define narrower route override rules.

### Why This Choice

- Hybrid still has a strong net win: 16 wins vs 5 losses.
- All 5 losses are explainable and localized to one route.
- This means the next improvement should target route refinement, not a full architecture change.

### Verification

- `python -m ruff check .` passed.
- `python -m pytest -q` passed, current test count is 43.
- `artifacts/answer_gap_analysis_dev_answer_aware_mcpd3_route_full.json` generated.
- `artifacts/answer_gap_analysis_dev_hybrid_routing_mcpd3_route_full.json` generated.
- `artifacts/selector_comparison_answer_aware_vs_hybrid_routing_route.json` generated.
- Experiment artifacts are under `artifacts/` and ignored by git.

### What I Still Do Not Understand

- Which security-bulletin sub-intents should stay on section-span.
- Whether affected-products/remediation questions can be detected reliably from title/text alone.
- Whether route override rules will overfit the current dev set.
- Whether a learned reranker should replace route-specific hand rules soon.

### Next Step

- Do Stage 17: security bulletin route refinement.
- First version should stay rule-based and no-leak:
  1. If a security bulletin question asks affected product/version -> answer-aware.
  2. If it asks remediation/fix/update/apply -> answer-aware or a remediation-specific selector.
  3. If it asks CVE/CVSS/vulnerability details -> section-span.
  4. If it mentions crash/post-fix behavior -> answer-aware.
- Re-run verified RAG and comparison against Stage 15 hybrid.

## Stage 17 - Security Bulletin Route Refinement

### Goal

- Refine the coarse Stage 15/16 route:
  `security_bulletin -> section-span`.
- Keep the routing no-leak: only use question title/text, never gold answer.
- Split security bulletin questions into narrower sub-intents:
  1. affected product/version
  2. remediation/fix/update/apply
  3. post-fix crash/error behavior
  4. vulnerability detail

### What I Studied

- Stage 16 showed that all 5 hybrid losses came from security bulletin questions.
- The losses were not the same kind of question:
  - `DEV_Q307` asked affected versions.
  - `DEV_Q260` asked crash/post-fix behavior.
  - `DEV_Q089`, `DEV_Q019`, and `DEV_Q220` still looked like vulnerability-detail or mixed bulletin cases.
- Therefore the next improvement should first make the route more precise instead of replacing hybrid globally.

### What I Built Or Changed

- Added `_is_security_bulletin_question`.
- Added `_classify_security_bulletin_route`.
- Added three explicit security-bulletin sub-intent checks:
  - `_asks_security_post_fix_behavior`
  - `_asks_security_affected_product_or_version`
  - `_asks_security_remediation`
- Replaced the old route:
  `security_bulletin`
  with these more specific routes:
  - `security_bulletin_vulnerability_detail`
  - `security_bulletin_affected_product`
  - `security_bulletin_remediation`
  - `security_bulletin_post_fix_behavior`
- Updated hybrid routing:
  - `security_bulletin_vulnerability_detail` -> section-span
  - `limitation_or_restriction` -> section-span
  - affected product/remediation/post-fix routes -> answer-aware
- Added tests for all three answer-aware security sub-routes.

### Commands And Evidence

```powershell
python -m pytest tests\test_evidence_selection.py -q
python -m ruff check src\ts_rag_agent\application\evidence_selection.py tests\test_evidence_selection.py
python -m pytest -q
python -m ruff check .

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage17_route.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage17_refined_security_route.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage17_route.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage17_refined_security_route.json `
  --baseline-label answer-aware `
  --challenger-label hybrid-routing-stage17 `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_hybrid_routing_stage17_security_route.json
```

Verification:

```text
targeted evidence-selection tests: 15 passed
full pytest: 46 passed
ruff: passed
```

### Main Results

Answer-aware baseline:

```text
average selected-answer F1: 0.2501
gold citation: 90 / 160
```

Stage 17 hybrid:

```text
average selected-answer F1: 0.2768
gold citation: 93 / 160
selected selectors:
  answer_aware_bm25_sentence: 138
  section_span_bm25_sentence: 22
```

Route distribution:

```text
install_upgrade_config: 24
limitation_or_restriction: 5
other: 71
how_to_or_lookup: 14
error_or_log: 26
security_bulletin_vulnerability_detail: 17
security_bulletin_remediation: 1
security_bulletin_post_fix_behavior: 1
security_bulletin_affected_product: 1
```

Answer-aware vs Stage 17 hybrid:

```text
total_compared: 160
answer-aware wins: 3
hybrid wins: 15
ties: 142
avg_f1_delta: +0.0267
baseline gold citation: 90
challenger gold citation: 93
```

Compared with Stage 16:

```text
Stage 16 answer-aware wins: 5
Stage 17 answer-aware wins: 3
Stage 16 hybrid wins: 16
Stage 17 hybrid wins: 15
Stage 16 ties: 139
Stage 17 ties: 142
```

### Problems Encountered

- The first affected-version rule did not catch `DEV_Q307`.
- Root cause: the real question text contains dirty whitespace:
  `what  versions`
  with two spaces.
- The original regex expected a single normal space.
- Fix: change affected product/version patterns to use `\s+`, for example:
  `what\s+versions?`.

### Remaining Loss Cases

Stage 17 still loses to answer-aware on 3 questions:

```text
DEV_Q089
DEV_Q019
DEV_Q220
```

All 3 are still routed as:

```text
security_bulletin_vulnerability_detail -> section-span
```

Observed pattern:

- They are no longer simple route-refinement cases.
- The issue is more about section-span selecting the wrong local window:
  - wrong adjacent CVE
  - translated/reference-link sections
  - remediation snippets near the vulnerability section

### Why This Choice

- Stage 17 improved the failure distribution without using gold answers.
- It reduced answer-aware wins from 5 to 3.
- It slightly improved average selected-answer F1 and gold citation.
- The gain is real but small, so continuing to add route rules would likely become brittle.

### What I Still Do Not Understand

- Whether `DEV_Q019` can be routed correctly from question text alone without adding an overfit rule.
- Whether `DEV_Q089` and `DEV_Q220` should be solved by exact-CVE anchoring inside section-span.
- Whether translated/reference-link sections need an explicit section penalty.

### Next Step

- Do Stage 18: improve section-span precision for security bulletin detail routes.
- Focus on no-leak document-side scoring:
  1. extract CVE IDs from the question,
  2. reward spans containing the requested CVE,
  3. penalize spans dominated by non-requested CVEs,
  4. penalize translated/reference-link sections when they are not the only relevant span.

## Stage 18 - CVE-Anchored Section Span Scoring

### Goal

- Improve section-span precision for security bulletin detail routes.
- Keep the method no-leak:
  - use question title/text,
  - use retrieved document text,
  - do not use gold answer or gold document id.
- Fix the Stage 17 pattern where security detail questions could select:
  - adjacent non-target CVEs,
  - translated/reference-link noise,
  - remediation/link-heavy spans instead of vulnerability details.

### What I Studied

Stage 17 left 3 answer-aware wins:

```text
DEV_Q089
DEV_Q019
DEV_Q220
```

Observed failure details:

- `DEV_Q089`: section-span selected an OpenSSL span from a different document and previously preferred a nearby non-target CVE.
- `DEV_Q019`: section-span selected translated/reference material before the English bulletin span.
- `DEV_Q220`: section-span selected remediation/reference-link spans before the actual `VULNERABILITY DETAILS` span.

The important lesson:

- A security bulletin question containing a CVE is not solved by merely rewarding any `CVEID` or `CVSS` text.
- The selector must check whether the span aligns with the CVE requested by the question.

### What I Built Or Changed

- Added `CVE_PATTERN`.
- Added CVE extraction helpers:
  - `_extract_cves_from_terms`
  - `_extract_cves_from_text`
  - `_first_cve_in_text`
- Added `_section_span_cve_alignment_bonus`.
- Added `_section_span_cve_alignment_multiplier`.
- Added `_section_span_reference_translation_penalty`.
- Updated `SectionSpanBM25SentenceEvidenceSelector` scoring order:
  1. base BM25/answer-aware score
  2. section-answer pattern bonus
  3. requested-CVE alignment bonus
  4. requested-CVE alignment multiplier
  5. background penalty
  6. translated/reference-link penalty
  7. length penalty
- Added tests for:
  - preferring the requested CVE over an adjacent non-target CVE,
  - penalizing translated/reference noise when a clean vulnerability-detail span exists.

### Commands And Evidence

```powershell
python -m pytest tests\test_evidence_selection.py -q
python -m ruff check src\ts_rag_agent\application\evidence_selection.py tests\test_evidence_selection.py
python -m pytest -q
python -m ruff check .

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage18_cve_anchor.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage18_cve_anchor.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --baseline-label answer-aware `
  --challenger-label hybrid-routing-stage18 `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_hybrid_routing_stage18_cve_anchor.json
```

Verification:

```text
targeted evidence-selection tests: 17 passed
full pytest: 48 passed
ruff: passed
```

### Main Results

Answer-aware baseline:

```text
average selected-answer F1: 0.2501
gold citation: 90 / 160
```

Stage 18 hybrid:

```text
average selected-answer F1: 0.2805
gold citation: 95 / 160
selected selectors:
  answer_aware_bm25_sentence: 138
  section_span_bm25_sentence: 22
```

Answer-aware vs Stage 18 hybrid:

```text
total_compared: 160
answer-aware wins: 2
hybrid wins: 16
ties: 142
avg_f1_delta: +0.0304
baseline gold citation: 90
challenger gold citation: 95
```

Security bulletin route-level result:

```text
security_bulletin_vulnerability_detail:
  answer-aware wins: 2
  hybrid wins: 15
```

Compared with Stage 17:

```text
Stage 17 average selected-answer F1: 0.2768
Stage 18 average selected-answer F1: 0.2805

Stage 17 gold citation: 93 / 160
Stage 18 gold citation: 95 / 160

Stage 17 answer-aware wins: 3
Stage 18 answer-aware wins: 2

Stage 17 hybrid wins: 15
Stage 18 hybrid wins: 16
```

### Problems Encountered

- The first translated-reference test put the noisy reference and the clean vulnerability detail in the same sentence.
- That made the test invalid because the selector had no separate span to choose.
- Fix: split the synthetic document into two sentences so the selector can rank the noisy reference span against the clean vulnerability-detail span.

### Remaining Loss Cases

Stage 18 still loses to answer-aware on:

```text
DEV_Q089
DEV_Q019
```

Observed details:

- `DEV_Q089` now selects the requested CVE, but the final selected answer is still weaker than answer-aware because multiple cross-document CVE windows are included.
- `DEV_Q019` now promotes the English target-CVE span above the translated reference span, but the gold answer is actually closer to a summary/affected-runtime statement than to the vulnerability-detail paragraph.

### Why This Choice

- CVE anchoring improves the exact weakness found in Stage 17.
- It is still no-leak and explainable.
- It improves both F1 and gold citation.
- The remaining errors no longer look like simple CVE-selection errors, so adding more CVE rules is unlikely to be the best next move.

### What I Still Do Not Understand

- Whether security-bulletin detail routes should use fewer selected evidence windows than general answer-aware routes.
- Whether retrieval rank or product-title alignment should be part of section-span scoring.
- Whether `DEV_Q019` should be rerouted away from vulnerability detail because its actual answer is closer to an affected-runtime summary.

### Next Step

- Do Stage 19: evidence aggregation / document-aware selection.
- Candidate directions:
  1. for section-span security detail routes, reduce cross-document evidence mixing,
  2. prefer higher-ranked retrieved documents when CVE alignment is similar,
  3. add product-title alignment between question title and document title,
  4. test whether security detail answers should use top-1 or top-2 spans instead of always taking 3.

## Stage 19 - Document-Aware Aggregation Experiment

### Goal

- Investigate whether security-bulletin detail answers should reduce cross-document evidence mixing.
- Test a document-aware aggregation idea:
  - choose one focus document using question CVE, product-title overlap, retrieval rank, and document-title signals,
  - keep evidence windows from that focus document,
  - avoid mixing multiple documents that mention the same CVE but belong to different IBM products.

### What I Studied

Stage 18 left two answer-aware wins:

```text
DEV_Q089
DEV_Q019
```

The observed problem was not pure CVE matching anymore:

- `DEV_Q089` selected the requested CVE, but answer quality was hurt by cross-document CVE windows.
- `DEV_Q019` selected the target CVE, but the gold answer was closer to an affected-runtime summary than to the vulnerability-detail paragraph.
- `DEV_Q220` had improved in Stage 18, but still showed that product/document alignment might matter.

### What I Tried

I implemented an experimental document-focus branch locally:

1. First attempt:
   - run normal section-span with `section_span_mcpd1`,
   - choose a focus document,
   - keep only candidates from that document.
2. Second attempt:
   - use a wider security-detail section-span pass with `mcpd3`,
   - choose a focus document,
   - keep up to three candidates from that document.
3. Bug fix during the experiment:
   - the first version accidentally changed no-CVE security bulletin questions because it used the wider focus selector even when no explicit CVE existed.
   - fixed the experiment so document focus only activates when the question text contains an explicit CVE.

The experiment used no gold answer and no gold document id.

### Commands And Evidence

```powershell
python -m pytest tests\test_evidence_selection.py -q
python -m ruff check src\ts_rag_agent\application\evidence_selection.py tests\test_evidence_selection.py
python -m pytest -q

python scripts\analyze_answer_gap.py `
  --evidence-selector answer-aware `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage19_doc_focus.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage19_doc_focus.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage19_doc_focus.json `
  --baseline-label hybrid-stage18 `
  --challenger-label hybrid-stage19-doc-focus `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_hybrid_stage18_vs_stage19_doc_focus.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_answer_aware_mcpd3_stage19_doc_focus.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage19_doc_focus.json `
  --baseline-label answer-aware `
  --challenger-label hybrid-routing-stage19 `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_answer_aware_vs_hybrid_routing_stage19_doc_focus.json
```

### Main Results

First attempt result:

```text
Stage 19 focus mcpd1 F1: 0.2789
Stage 19 focus mcpd1 gold citation: 94 / 160
```

This was worse than Stage 18:

```text
Stage 18 F1: 0.2805
Stage 18 gold citation: 95 / 160
```

Second attempt after widening focus document candidates:

```text
Stage 19 focus mcpd3 F1 before no-CVE fix: 0.2803
Stage 19 focus mcpd3 gold citation before no-CVE fix: 95 / 160
```

After restricting focus only to questions with explicit CVE:

```text
Stage 19 F1: 0.2805
Stage 19 gold citation: 95 / 160
```

Stage 18 vs Stage 19 after the no-CVE fix:

```text
total_compared: 160
Stage 18 wins: 5
Stage 19 wins: 8
ties: 147
avg_f1_delta: -0.0
Stage 18 gold citation: 95
Stage 19 gold citation: 95
```

Answer-aware vs Stage 19:

```text
answer-aware wins: 2
Stage 19 hybrid wins: 15
ties: 143
avg_f1_delta: +0.0303
gold citation: 95 / 160
```

For comparison, Stage 18 had:

```text
answer-aware wins: 2
Stage 18 hybrid wins: 16
ties: 142
avg_f1_delta: +0.0304
gold citation: 95 / 160
```

### Decision

- The document-focus experiment was not adopted into runtime code.
- Reason:
  - first attempt regressed both F1 and gold citation,
  - second attempt recovered citation but did not beat Stage 18,
  - relative to answer-aware, Stage 19 had one fewer win than Stage 18,
  - the best observed result remained Stage 18.
- Final code was reverted to Stage 18 behavior.
- Only this learning-journal record is kept.

### Problems Encountered

- The naive document-focus idea removed useful complementary evidence.
- `section_span_mcpd1` was too narrow after document filtering.
- `section_span_mcpd3` recovered some coverage, but introduced new local tradeoffs.
- A no-CVE activation bug showed that aggregation rules must have strict entry conditions.

### Root Cause

- Cross-document mixing is sometimes harmful, but not uniformly harmful.
- Some security bulletin answers benefit from multiple windows, even if those windows are not all from the same document.
- A simple product-title/retrieval-rank focus rule is not enough to predict when document filtering will help.

### What I Learned

- Stage 19 is a useful negative result.
- The next improvement should not be another hard document-focus rule.
- The system needs better candidate-level answer quality prediction or a route-specific answer composer, not a blunt document filter.

### Next Step

- Do Stage 20: answer composition / candidate selection analysis.
- Instead of filtering documents, analyze the top selected candidates themselves:
  1. whether repeated near-duplicate CVE spans hurt F1,
  2. whether selected spans should be compressed or deduplicated,
  3. whether security-detail route should choose the best window rather than always concatenating top 3,
  4. whether answer-aware and section-span candidates should be merged before final answer selection.

## Stage 20 - Answer Composition Analysis

### Goal

- Analyze whether the remaining quality gap comes from answer composition rather than retrieval or evidence selection.
- Keep runtime unchanged unless a composition strategy clearly improves both answer quality and citation behavior.
- Distinguish deployable strategies from oracle diagnostics:
  - deployable: top1, top2, dedup-top3
  - oracle only: best single candidate, best prefix, best same-document prefix

### What I Built Or Changed

- Added `src/ts_rag_agent/application/answer_composition_analysis.py`.
- Added `scripts/analyze_answer_composition.py`.
- Added `tests/test_answer_composition_analysis.py`.
- The analysis reads an existing answer-gap JSON report and computes:
  - current top-k answer F1
  - top1 F1
  - top2 F1
  - deduplicated top3 F1
  - best-single oracle F1
  - best-prefix oracle F1
  - best-same-document-prefix oracle F1
  - multi-document answer count
  - duplicate answer count
  - representative gain cases

### Commands And Evidence

```powershell
python -m pytest tests\test_answer_composition_analysis.py -q
python -m ruff check src\ts_rag_agent\application\answer_composition_analysis.py scripts\analyze_answer_composition.py tests\test_answer_composition_analysis.py
python -m pytest -q
python -m ruff check .

python scripts\analyze_answer_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --f1-gain-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\answer_composition_analysis_stage20_hybrid_stage18.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --max-sentences 2 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage20_max_sentences2.json

python scripts\analyze_answer_gap.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --max-sentences 1 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage20_max_sentences1.json

python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage20_max_sentences2.json `
  --baseline-label hybrid-stage18-top3 `
  --challenger-label hybrid-stage20-top2 `
  --f1-win-margin 0.03 `
  --sample-limit-per-bucket 30 `
  --output artifacts\selector_comparison_hybrid_stage18_top3_vs_stage20_top2.json
```

Verification:

```text
new tests: 2 passed
full pytest: 50 passed
ruff: passed
```

### Main Composition Results

Input report:

```text
artifacts/answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json
```

Composition analysis:

```text
total cases: 160
current top3 average F1: 0.2805
top1 average F1: 0.2600
top2 average F1: 0.2826
dedup-top3 average F1: 0.2820
best-single oracle F1: 0.3847
best-prefix oracle F1: 0.3553
best-same-document-prefix oracle F1: 0.3864
average best-prefix oracle gain: +0.0749
```

Case counts:

```text
top1 beats current: 52
top2 beats current: 56
dedup-top3 beats current: 7
best-single oracle beats current: 88
best-prefix oracle beats current: 78
best-same-document-prefix oracle beats current: 89
multi-document answer count: 151
duplicate answer count: 6
```

Oracle strategy counts:

```text
best_single: 106
best_prefix: 53
best_same_doc_prefix: 1
```

### Deployable Top-K Experiment

Because top2 was deployable and had a higher analysis F1 than current top3, I reran answer-gap with:

```text
max_sentences = 2
```

Result:

```text
top3 F1: 0.2805
top3 gold citation: 95 / 160

top2 F1: 0.2814
top2 gold citation: 72 / 160
```

Comparison:

```text
top3 wins: 47
top2 wins: 55
ties: 58
avg F1 delta: +0.0009
top3 gold citation: 95
top2 gold citation: 72
```

I also tested:

```text
max_sentences = 1
```

Result:

```text
top1 F1: 0.2565
top1 gold citation: 48 / 160
```

### Decision

- Do not change runtime default from top3 to top2.
- Reason:
  - top2 gives a small F1 improvement,
  - but it loses 23 gold citations,
  - citation faithfulness is a core RAG metric and should not be traded away for a tiny F1 gain.
- Keep Stage 18 runtime behavior.
- Keep Stage 20 analysis tool because it gives a clear diagnostic view of composition bottlenecks.

### What I Learned

- The dominant bottleneck is not just “too many documents”.
- Many cases need better final candidate selection:
  - top1 is often best for concise factual answers,
  - top3 often preserves citation coverage,
  - top2 is a tempting middle ground but hurts citation too much,
  - dedup helps only a small number of cases.
- The oracle gap is large:
  - current top3 F1: 0.2805
  - best-single oracle F1: 0.3847
  - best-same-document-prefix oracle F1: 0.3864
- That means the next useful model-free stage should predict which candidate subset to use, not blindly shrink all answers.

### Remaining Questions

- Can we predict when top1/top2 is better without using gold answers?
- Can citation coverage be preserved while removing unrelated second/third candidates?
- Should answer composition be route-specific?
- Should we merge answer-aware and section-span candidates before composition?

### Next Step

- Do Stage 21: route-aware answer composition policy.
- Candidate design:
  1. keep top3 as default for citation-sensitive routes,
  2. use top1 for concise direct-answer routes only when the first candidate has strong answer-section signals,
  3. apply conservative near-duplicate removal,
  4. test policy against Stage 18 top3 and reject it unless it improves F1 without large citation loss.

## 2026-07-13 - Stage 21: Route-Aware Answer Composition Policy

### Goal

- Do not blindly change `max_sentences` from 3 to 2 after Stage 20.
- Build a route-aware composition policy that only shortens answers in low-risk
  direct-answer cases.
- Preserve citation-sensitive behavior for security bulletin and limitation
  routes.
- Record the policy as a reproducible analysis first, not as a default runtime
  behavior.

### What I Studied

- Stage 20 showed that global top2 is not safe:
  - it slightly improves F1,
  - but it loses too many gold citations.
- A better policy needs to decide per question route.
- Route-aware composition should use only deployable signals:
  - question route,
  - candidate score,
  - retrieval rank,
  - answer/action wording in the selected sentence,
  - duplicate sentence similarity.
- It must not use `candidate_token_f1` or gold answer labels for policy
  decisions. Those are only used after the policy runs to evaluate quality.

### What I Built Or Changed

- Added route-aware composition policy module:
  `src/ts_rag_agent/application/route_aware_composition_policy.py`
- Added CLI analysis script:
  `scripts/analyze_route_aware_composition.py`
- Added tests:
  `tests/test_route_aware_composition_policy.py`

Policy behavior:

```text
Policy name:
route_aware_top1_direct_otherwise_top3

Direct-answer routes eligible for top1:
how_to_or_lookup
install_upgrade_config

Citation-sensitive routes kept at top3:
security_bulletin_vulnerability_detail
security_bulletin_affected_product
security_bulletin_remediation
security_bulletin_post_fix_behavior
limitation_or_restriction
```

Default top1 gate:

```text
strong_first_score_min: 100.0
strong_first_score_ratio_min: 1.15
strong_first_score_margin_min: 20.0
max_top1_retrieval_rank: 3
duplicate_threshold: 0.96
```

The top1 decision also requires an answer/action signal in the first sentence,
such as `install`, `configure`, `use`, `set`, `apply`, `download`, `fix`,
`solution`, or `workaround`.

### Commands And Evidence

```powershell
python -m ruff check .
python -m pytest -q

python scripts\analyze_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --output artifacts\route_aware_composition_policy_stage21_hybrid_stage18.json
```

Validation:

```text
ruff: passed
pytest: 52 passed
```

Stage 21 report:

```text
artifacts/route_aware_composition_policy_stage21_hybrid_stage18.json
```

Aggregate result:

```text
total_cases: 160
baseline F1: 0.2808
policy F1: 0.2851
average F1 delta: +0.0043

baseline gold citation: 95 / 160
policy gold citation: 94 / 160
citation delta: -1

changed_answer_count: 10
f1_improved_count: 5
f1_regressed_count: 4
citation_lost_count: 1
citation_gained_count: 0
```

Strategy counts:

```text
keep_top3_default: 125
keep_top3_citation_sensitive: 25
top1_direct_strong_signal: 10
```

Route-level result:

```text
how_to_or_lookup:
  cases: 14
  baseline F1: 0.2508
  policy F1: 0.2854
  F1 delta: +0.0346
  citation delta: 0
  top1_direct_strong_signal: 2

install_upgrade_config:
  cases: 24
  baseline F1: 0.2064
  policy F1: 0.2148
  F1 delta: +0.0084
  citation delta: -1
  top1_direct_strong_signal: 8

security bulletin routes:
  policy kept top3
  citation delta: 0
```

The policy passed the configured offline gate:

```text
accepted_for_runtime_experiment: true
reason: F1 gain meets threshold and citation loss is bounded
```

### Problems Encountered

- The first Stage 21 report showed F1 changes even for routes where the policy
  kept top3.
- That was not a model improvement. It was an analysis bug.
- The baseline F1 was read from the old answer-gap report, while policy F1 was
  recomputed from candidate sentences with the current metric function.
- This mixed two measurement paths and made unchanged answers look changed.

### Root Cause

- Stage 20 stored `selected_answer_token_f1` in the answer-gap report.
- Stage 21 initially used that stored value for baseline F1.
- Policy F1 was freshly computed from the selected candidate sentences.
- Even small rounding or metric-path differences are enough to create false
  deltas.

### Solution

- Changed Stage 21 analysis so baseline and policy F1 are both recomputed from
  the same candidate sentences.
- Reran the report.
- After the fix, all unchanged `keep_top3` routes had zero F1 delta.

### Why This Choice

- The policy needs to be evaluated with a fair counterfactual:
  - baseline answer = current selected candidates joined together,
  - policy answer = policy-selected candidates joined together,
  - both scored with the same `token_f1` function.
- This avoids overclaiming improvements from stale or rounded metrics.
- The policy is deliberately conservative:
  - only 10 of 160 answers changed,
  - security and limitation routes stayed top3,
  - only one gold citation was lost.

### Key Case Notes

Best positive case:

```text
question_id: DEV_Q008
route: how_to_or_lookup
question: How can I export a private key from DataPower Gateway Appliance?
baseline F1: 0.3939
policy F1: 0.9630
citation delta: 0
```

Main risk case:

```text
question_id: DEV_Q202
route: install_upgrade_config
question: Why is installation manager cores when try to install netcool using GUI mode in AIX 7.1?
baseline F1: 0.3178
policy F1: 0.1455
citation delta: -1
```

This single case explains the current citation loss.

### Decision

- Do not change the default runtime yet.
- Accept the policy only as a candidate for a runtime experiment.
- The offline result is promising because it improves F1 with only one citation
  loss, but it still needs an end-to-end run through the real answer generator
  and verifier path.

### What I Learned

- Route-aware composition is much safer than global top1/top2.
- `how_to_or_lookup` benefits most from selective top1 shortening.
- `install_upgrade_config` has some benefit but also contains the only citation
  loss, so it needs stricter gating or extra risk checks.
- Metric-path consistency matters. Even a small difference between stored metric
  values and recomputed metric values can create fake gains or losses.

### Remaining Questions

- Should `install_upgrade_config` require an even stronger top1 gate than
  `how_to_or_lookup`?
- Can the policy detect cases like `DEV_Q202` without using gold labels?
- Should the next runtime experiment keep this exact gate, or split thresholds
  by route?
- Does the end-to-end verified RAG report preserve the offline gain once the
  policy is integrated into the answer generator?

### Next Step

- Do Stage 22: optional runtime integration of the route-aware composition
  policy.
- Add a non-default composition policy option to the answer generator and
  evaluation script.
- Run end-to-end verified RAG with:
  1. Stage 18 top3 baseline,
  2. Stage 21 route-aware policy.
- Keep the policy disabled by default unless the end-to-end report confirms:
  1. F1 improves,
  2. gold citation loss stays bounded,
  3. verification behavior does not regress.

## Stage 22 - Route-Aware Composition Runtime Experiment

### 目标

- 把 Stage 21 的 route-aware composition policy 接入真实 runtime。
- 保持默认运行策略不变，只有显式传入参数时才启用 route-aware。
- 用完整 verified RAG end-to-end 路径对比当前 top-k baseline 和 route-aware policy。

### 我学习和确认了什么

- Stage 21 的 policy 原本只在离线 answer-gap report 上工作。
- Runtime 真正拼答案的位置是
  `ExtractiveAnswerGenerator.generate()`。
- 如果只在评估脚本里临时改候选句数量，会让 runtime、分析脚本和测试路径分叉。
- 所以本阶段把“候选句排序”和“最终选几句组成答案”拆成独立的
  answer composition 层。

### 本阶段改动

- 新增 `src/ts_rag_agent/application/answer_composition.py`：
  - `AnswerCompositionPolicy`
  - `TopKAnswerCompositionPolicy`
  - `RouteAwareAnswerCompositionPolicy`
  - `create_answer_composition_policy`
- 更新 `ExtractiveAnswerGenerator`：
  - 新增 `composition_policy` 参数；
  - 默认仍然使用 `TopKAnswerCompositionPolicy`；
  - 新增统一的 `select_answer_candidates()`。
- 更新 `AnswerGapAnalyzer`：
  - 改为调用 generator 的统一候选选择方法；
  - 避免分析路径和 runtime 路径使用不同的最终候选选择逻辑。
- 更新 `scripts/evaluate_verified_rag.py`：
  - 新增 `--composition-policy`；
  - 可选值包括 `top-k` 和 `route-aware`；
  - JSON report 的 `rag` 字段会记录实际 composition policy。
- 更新测试：
  - 覆盖 route-aware runtime 接入；
  - 覆盖未知 composition policy 报错。

### 命令和真实结果

```powershell
python -m ruff check .
python -m pytest -q
```

验证结果：

```text
ruff: passed
pytest: 54 passed
```

当前代码下的 top-k baseline：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy top-k `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage22_topk_report.json
```

结果：

```text
composition_policy: top_k
original average_token_f1: 0.2729
original gold_doc_citation_rate: 0.5938
original gold citation count: 95 / 160

verified average_token_f1: 0.2755
verified gold_doc_citation_rate: 0.6013
verified gold citation count: 95 / 158 generated answerable

answerable_refusal_rate: 0.0125
unanswerable_refusal_rate: 0.0267
newly_refused: 6
```

Route-aware runtime experiment：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage22_route_aware_report.json
```

结果：

```text
composition_policy: route_aware_top1_direct_otherwise_top3
original average_token_f1: 0.2769
original gold_doc_citation_rate: 0.5875
original gold citation count: 94 / 160

verified average_token_f1: 0.2795
verified gold_doc_citation_rate: 0.5949
verified gold citation count: 94 / 158 generated answerable

answerable_refusal_rate: 0.0125
unanswerable_refusal_rate: 0.0267
newly_refused: 6
```

端到端对比：

```text
verified average_token_f1 delta: +0.0040
verified gold citation count delta: -1
answerable_refusal_rate delta: 0.0000
unanswerable_refusal_rate delta: 0.0000
newly_refused delta: 0
```

### 事实边界

- Stage 18 的 `0.2805` 是 answer-gap selected-answer F1，不是完整
  verified RAG report 的 verified F1。
- Stage 22 的对比使用同一份当前代码重跑出的两个完整 verified RAG report：
  - `artifacts/verified_rag_dev_hybrid_routing_m15_stage22_topk_report.json`
  - `artifacts/verified_rag_dev_hybrid_routing_m15_stage22_route_aware_report.json`
- 旧文件 `artifacts/verified_rag_dev_hybrid_routing_m15_report.json`
  缺少 `composition_policy` 字段，而且指标低于当前代码重跑结果；
  因此本阶段没有把它当作 Stage 22 的直接对照数据。
- `artifacts/*` 被 `.gitignore` 忽略，本阶段报告文件是本地实验产物，不会随提交进入仓库。

### 遇到的问题

- 初看旧 verified RAG artifact 时，它的 F1 和 gold citation 都低于本轮 top-k 重跑结果。
- 这不是 route-aware policy 带来的变化，因为旧 artifact 缺少 composition policy 字段，也不是本轮同代码生成。

### 原因

- Stage 22 修改了 report schema，新增了 `composition_policy`。
- 旧 report 不能证明当前默认 top-k 路径在新代码下的表现。
- 为了避免混用不同时间、不同 schema、不同代码状态的结果，本阶段必须重跑 top-k baseline。

### 处理方式

- 用当前代码重新生成 top-k baseline。
- 再用同一套参数只切换 `--composition-policy route-aware`。
- 只比较这两份同代码、同参数、同时间段生成的 report。

### 结论

- Route-aware runtime policy 端到端保留了 Stage 21 的 F1 小幅收益：
  verified F1 提升 `+0.0040`。
- Citation 风险也被保留下来：
  verified gold citation 少 1 个。
- 验证器行为没有变差：
  `answerable_refusal_rate`、`unanswerable_refusal_rate`、`newly_refused`
  都没有变化。
- 因为 gold citation 仍然少 1 个，本阶段不把 route-aware 改成默认策略。
- 当前结论是：route-aware 可以作为显式 runtime 实验参数保留，但默认仍然使用 top-k。

### 我学到的

- 一个离线 policy 即使通过 answer-gap 分析，也必须通过真实 generator 和 verifier 链路验证。
- 加 runtime 参数时，默认行为必须用当前代码重跑，而不是引用旧 artifact。
- F1 提升和 citation 损失可以同时发生；这类策略不能只看一个指标。
- 把 answer composition 独立封装后，后续可以继续做 route-specific 阈值，而不需要改检索器或 evidence selector。

### 下一步

- 做 Stage 23：分析 route-aware 少掉的那个 gold citation 是否仍然集中在
  `DEV_Q202` 或同类 `install_upgrade_config` 问题。
- 如果仍然是 install/config 路线导致 citation 损失，再评估是否需要一个更严格的
  route-specific top1 gate。
- 在确认之前，继续保持默认 `--composition-policy top-k`。

## Stage 23 - Strict Install Route Gate For Route-Aware Composition

### 目标

- 精确定位 Stage 22 route-aware runtime 少掉的那个 gold citation。
- 判断 citation loss 是否仍然来自 Stage 21 已知风险样本 `DEV_Q202`。
- 如果风险集中在 `install_upgrade_config`，设计一个不使用 gold label 的更严格
  route-specific top1 gate。
- 继续保持默认 runtime 为 `top-k`，只优化显式 `route-aware` 参数。

### 我先确认了什么

- 仓库起始状态：

```text
git: main...origin/main clean
```

- Stage 22 默认 report 只保存部分 sample，不足以逐题定位 citation regression。
- 因此本阶段先重跑 full-sample report，而不是直接引用旧 report 的聚合数。

### 全量重跑命令

Top-k baseline：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy top-k `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage23_topk_full_report.json
```

Stage 22 route-aware：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage23_route_aware_full_report.json
```

重跑结果复现了 Stage 22：

```text
top-k verified average_token_f1: 0.2755
top-k verified gold citation count: 95 / 158 generated answerable

route-aware verified average_token_f1: 0.2795
route-aware verified gold citation count: 94 / 158 generated answerable
```

### Citation Regression 定位结果

逐题比对后，citation loss 只来自一个问题，且 original 和 verified 都一样：

```text
question_id: DEV_Q202
route: install_upgrade_config
question: Why is installation manager cores when try to install netcool using GUI mode in AIX 7.1?
gold_doc: swg21631478

top-k docs: swg21661861, swg21631478, swg27043142
route-aware docs: swg21661861

top-k F1: 0.3091
route-aware F1: 0.1429
```

这和 Stage 21 的已知风险样本一致。

### 反事实分析

本阶段比较了几个只使用 runtime 可见特征的候选策略：

```text
current route-aware:
  verified average_token_f1: 0.2795
  verified gold citation count: 94

how_to_only:
  verified average_token_f1: 0.2785
  verified gold citation count: 95

install stricter margin gate:
  verified average_token_f1: 0.2815
  verified gold citation count: 95
```

关键观察：

- `how_to_or_lookup` 的 top1 保留了最大的 F1 收益。
- `install_upgrade_config` 不能继续只用原来的 ratio/margin OR 条件。
- 坏例子 `DEV_Q202` 的 first-vs-second margin 只有约 `20.7`。
- 另一个明显 F1 回退样本 `DEV_Q016` 的 margin 约 `35.8`。
- dev 上带来正收益的 install 样本 margin 都在约 `47.4` 以上。

因此采用：

```text
install_upgrade_score_margin_min: 45.0
```

事实边界：

- `45.0` 是基于当前 dev split 和当前 selector score scale 的实验门槛。
- 它不是外部业务规则，也不能直接迁移到不同 selector 或不同 score scale。
- 它不使用 gold answer 或 gold document id；只看候选句分数、rank、route 和答案信号。

### 本阶段代码改动

- 更新 `RouteAwareCompositionPolicy`：
  - policy name 改为
    `route_aware_top1_direct_strict_install_otherwise_top3`；
  - `how_to_or_lookup` 仍使用原始强 top1 gate；
  - `install_upgrade_config` 改用更严格的
    `install_upgrade_score_margin_min=45.0`；
  - citation-sensitive routes 仍保持 top3。
- 更新 `scripts/analyze_route_aware_composition.py`：
  - 新增 `--install-upgrade-score-margin-min`；
  - JSON report 记录该参数。
- 更新 `create_answer_composition_policy`：
  - `route-aware` 仍指向当前 route-aware policy；
  - 保留旧 policy name alias，避免旧命令名无法创建策略。
- 更新测试：
  - 增加 install/config stricter margin gate 测试。

### 离线验证

命令：

```powershell
python scripts\analyze_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --output artifacts\route_aware_composition_policy_stage23_strict_install_hybrid_stage18.json
```

结果：

```text
baseline F1: 0.2808
policy F1: 0.2869
F1 delta: +0.0061

baseline gold citation: 95 / 160
policy gold citation: 95 / 160
citation delta: 0

changed_answer_count: 6
f1_improved_count: 5
f1_regressed_count: 1
citation_lost_count: 0
```

Route-level result：

```text
how_to_or_lookup:
  F1 delta: +0.0346
  citation delta: 0
  top1_direct_strong_signal: 2

install_upgrade_config:
  F1 delta: +0.0202
  citation delta: 0
  top1_direct_strong_signal: 4
```

### End-To-End Verified RAG 验证

命令：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage23_strict_install_report.json
```

结果：

```text
composition_policy: route_aware_top1_direct_strict_install_otherwise_top3

original average_token_f1: 0.2789
original gold_doc_citation_rate: 0.5938
original gold citation count: 95 / 160

verified average_token_f1: 0.2815
verified gold_doc_citation_rate: 0.6013
verified gold citation count: 95 / 158 generated answerable

answerable_refusal_rate: 0.0125
unanswerable_refusal_rate: 0.0267
newly_refused: 6
```

Against top-k baseline：

```text
verified average_token_f1 delta: +0.0060
verified gold citation count delta: 0
answerable_refusal_rate delta: 0
unanswerable_refusal_rate delta: 0
newly_refused delta: 0
```

逐题检查：

```text
citation losses: 0
citation gains: 0
changed verified answers: 16 total
changed answerable answers: 6
```

Answerable changed cases：

```text
DEV_Q008 how_to_or_lookup          F1 delta +0.5518
DEV_Q022 install_upgrade_config   F1 delta +0.0326
DEV_Q204 install_upgrade_config   F1 delta +0.0553
DEV_Q257 install_upgrade_config   F1 delta +0.0764
DEV_Q296 install_upgrade_config   F1 delta +0.3203
DEV_Q302 how_to_or_lookup          F1 delta -0.0793
```

### 测试

```powershell
python -m ruff check .
python -m pytest -q
```

结果：

```text
ruff: passed
pytest: 55 passed
```

### 结论

- `DEV_Q202` 的 citation loss 被 strict install gate 修复。
- 新 route-aware policy 在 dev verified RAG 上同时满足：
  - F1 提升；
  - gold citation 不下降；
  - verifier refusal 行为不变。
- 默认 runtime 仍然不改，继续是 `top-k`。
- `route-aware` 现在是更强的实验参数，但是否提升为默认策略需要下一阶段跨 split 或更多数据复验。

### 我学到的

- route-aware 不是简单地按 route 开关 top1；同一个 direct-answer route 内部也有不同风险。
- `install_upgrade_config` 的问题更容易出现“第一句看起来强，但第二/第三句保留 gold citation”的情况。
- 离线 answer-gap、full-sample report 和 verified RAG 三层都要对齐，否则容易把局部收益误判成可默认化策略。
- 引入 route-specific threshold 时必须记录 score scale 和数据边界；否则后面换 selector 会误用旧门槛。

### 下一步

- 做 Stage 24：对 strict route-aware policy 做更强的稳健性验证。
- 优先选项：
  1. 在 train split 上跑同参数 verified RAG；
  2. 或者先做 dev 的 route-level quality report，分析 unanswerable changed cases 是否有隐性风险。
- 在 Stage 24 完成前，不把 `route-aware` 设为默认 runtime 策略。

## Stage 24 - Cross-Validated Route-Aware Robustness Check

### 目标

- 按用户要求引入交叉验证，而不是继续只看单次 dev 全量结果。
- 验证 Stage 23 的 `install_upgrade_score_margin_min=45.0` 是否稳定。
- 判断 strict route-aware policy 是否已经足够稳，可以进入默认化讨论。

### 交叉验证口径

- 使用 answer-gap report 做离线 k-fold CV。
- Runtime policy 仍然不使用 gold label。
- CV 层使用 gold answer / gold document 来评估候选 threshold 的 F1 和 citation。
- 每折流程：
  1. 将 answerable cases 按 `question_id` 排序后 deterministic round-robin 分成 5 折。
  2. 用 4 折训练候选 install margin。
  3. 训练折要求 `citation_delta >= 0`。
  4. 在满足 citation 约束的候选中优先最大化 F1。
  5. 用剩余 1 折验证被选 threshold。

事实边界：

- 这是阈值稳健性分析，不是模型训练。
- CV 分析可以使用 gold label；runtime policy 不能使用 gold label。
- 目前 CV 只调 `install_upgrade_score_margin_min`，没有同时调 how-to gate。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/route_aware_composition_cv.py`
  - `cross_validate_route_aware_composition_policy`
  - `route_aware_cv_result_to_dict`
  - deterministic k-fold 切分
  - train-fold threshold selection
  - aggregate validation summary
- 新增 `scripts/cross_validate_route_aware_composition.py`
  - 支持 `--fold-count`
  - 支持 `--install-upgrade-score-margin-grid`
  - 输出 JSON CV report
- 新增 `tests/test_route_aware_composition_cv.py`
  - 覆盖 citation-preserving threshold selection
  - 覆盖非法 fold count

### Dev CV

自动选 threshold CV：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --fold-count 5 `
  --output artifacts\route_aware_composition_cv_stage24_dev_hybrid_stage18.json
```

结果：

```text
total_validation_cases: 160
average_baseline_f1: 0.2808
average_policy_f1: 0.2849
average_f1_delta: +0.0041
baseline gold citation: 95
policy gold citation: 95
citation_delta: 0
selected_margin_counts:
  45.0: 4
  50.0: 1
```

固定 `45.0` CV：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --fold-count 5 `
  --install-upgrade-score-margin-grid 45 `
  --output artifacts\route_aware_composition_cv_stage24_dev_fixed45_hybrid_stage18.json
```

结果：

```text
total_validation_cases: 160
average_baseline_f1: 0.2808
average_policy_f1: 0.2869
average_f1_delta: +0.0061
baseline gold citation: 95
policy gold citation: 95
citation_delta: 0
selected_margin_counts:
  45.0: 5
```

Dev 结论：

- `45.0` 在 dev 上通过 fixed-threshold CV。
- 自动选 threshold CV 有一折因为训练折并列选择了 `50.0`，导致少拿一个验证收益。
- 这说明自动调参流程本身还不够成熟，但 dev 数据仍支持 `45.0`。

### Train Answer-Gap Source

为了避免只在 dev 上循环确认，本阶段生成 train split 的 answer-gap source：

```powershell
python scripts\analyze_answer_gap.py `
  --split train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json
```

结果：

```text
total_answerable_questions: 450
gold_in_context: 271
selected_gold_citation: 232
average_selected_answer_token_f1: 0.2596

route counts:
install_upgrade_config: 67
how_to_or_lookup: 41
security_bulletin_vulnerability_detail: 70
error_or_log: 93
other: 170
```

### Train CV

自动选 threshold CV：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --fold-count 5 `
  --output artifacts\route_aware_composition_cv_stage24_train_auto_hybrid.json
```

结果：

```text
total_validation_cases: 450
average_baseline_f1: 0.2609
average_policy_f1: 0.2600
average_f1_delta: -0.0009
baseline gold citation: 232
policy gold citation: 228
citation_delta: -4
citation_lost_count: 4
selected_margin_counts:
  60.0: 5
```

固定 `45.0` CV：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --fold-count 5 `
  --install-upgrade-score-margin-grid 45 `
  --output artifacts\route_aware_composition_cv_stage24_train_fixed45_hybrid.json
```

结果：

```text
total_validation_cases: 450
average_baseline_f1: 0.2609
average_policy_f1: 0.2590
average_f1_delta: -0.0019
baseline gold citation: 232
policy gold citation: 225
citation_delta: -7
citation_lost_count: 7
selected_margin_counts:
  45.0: 5
```

额外 margin 检查：

```text
margin 60.0:  F1 delta -0.0009, citation_delta -4
margin 70.0:  F1 delta -0.0012, citation_delta -4
margin 80.0:  F1 delta -0.0012, citation_delta -4
margin 100.0: F1 delta -0.0013, citation_delta -4
margin 999.0: F1 delta -0.0013, citation_delta -4
```

解释：

- 即使把 install top1 几乎关掉，train 仍有 `-4` citation。
- 所以 train 上的主要新风险不是 install，而是 `how_to_or_lookup`。

How-to citation loss cases：

```text
TRAIN_Q188 how_to_or_lookup F1 delta -0.5224
TRAIN_Q467 how_to_or_lookup F1 delta -0.3469
TRAIN_Q384 how_to_or_lookup F1 delta -0.2217
TRAIN_Q075 how_to_or_lookup F1 delta +0.2074
```

Route-level train no-install-top1 report：

```text
how_to_or_lookup:
  cases: 41
  baseline F1: 0.3007
  policy F1: 0.2851
  F1 delta: -0.0156
  baseline gold citation: 25
  policy gold citation: 21
  citation_delta: -4
```

### Train End-To-End Verified RAG

Top-k baseline：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy top-k `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_m15_stage24_topk_report.json
```

Strict route-aware：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_m15_stage24_route_aware_report.json
```

结果：

```text
top-k verified average_token_f1: 0.2557
top-k verified gold_doc_citation_rate: 0.5249
top-k verified gold citation count: about 232 / 442 generated answerable

route-aware verified average_token_f1: 0.2539
route-aware verified gold_doc_citation_rate: 0.5090
route-aware verified gold citation count: about 225 / 442 generated answerable

verified F1 delta: -0.0018
verified gold citation count delta: -7
answerable_refusal_rate delta: 0
unanswerable_refusal_rate delta: 0
newly_refused delta: 0
```

### 测试

```powershell
python -m ruff check .
python -m pytest -q
```

结果：

```text
ruff: passed
pytest: 57 passed
```

### 结论

- Stage 24 的交叉验证没有支持 route-aware 默认化。
- Dev 上 `45.0` 看起来稳定，但 train CV 和 train verified RAG 都显示回退。
- Stage 23 的 strict install gate 修复了 dev 上的 `DEV_Q202`，但没有解决 train 上的 how-to top1 风险。
- 当前正确决策是：
  - 保留 `route-aware` 作为实验参数；
  - 默认 runtime 继续使用 `top-k`；
  - 不把 Stage 23 的 dev 成功外推成全局策略成功。

### 我学到的

- 交叉验证很必要：它直接推翻了“dev 上 +0.0060 且 citation 不降，所以可以默认化”的诱惑。
- `how_to_or_lookup` 不是天然安全的 direct-answer route；在 train 上，强 top1 也可能丢掉 gold citation。
- 只调 `install_upgrade_config` 会把风险转移到未处理的 how-to route。
- 自动阈值选择本身也需要被验证；CV fold 内的并列选择可能影响验证收益。

### 下一步

- 做 Stage 25：重新设计 how-to route gate。
- 候选方向：
  1. 暂时关闭 `how_to_or_lookup` top1，只保留 strict install top1；
  2. 对 how-to 增加独立 CV 网格，而不是沿用全局 ratio/margin；
  3. 引入非 gold 的文档一致性信号，例如 top1 文档是否和 top2/top3 文档同主题、标题是否强匹配问题。
- 在 Stage 25 前，`route-aware` 不能进入默认 runtime。

## Stage 25 - Disable How-To Top1 And Keep Only Strict Install Top1

### 目标

- 重新设计 Stage 24 暴露风险的 `how_to_or_lookup` route gate。
- 判断 how-to 是否能通过更高 margin/ratio 变安全。
- 如果不能，就关闭 how-to top1，只保留更保守的 install/config top1。

### 起始状态

```text
git: main...origin/main clean
```

Stage 24 结论：

- Dev 上 strict route-aware 表现好。
- Train CV 和 train verified RAG 都回退。
- 即使把 install top1 几乎关掉，train 上仍有 how-to citation loss。

### 离线候选策略分析

本阶段先不改代码，直接用 dev/train answer-gap artifacts 做反事实分析：

- current strict：Stage 24 策略，how-to 仍允许 top1，install margin `45.0`
- install only：关闭 how-to top1，install margin `45.0`
- how-to margin grid：`20, 30, 45, 60, 80, 100, 120`
- how-to ratio grid：`1.15, 1.3, 1.5, 1.8, 2.0, 2.5`
- install margin grid with how-to disabled：`45, 60, 80, 100, 120, 150, 999`

关键结果：

```text
dev current strict:
  F1 delta: +0.0061
  citation_delta: 0

dev install-only margin 60:
  F1 delta: +0.0002
  citation_delta: 0

train current strict:
  F1 delta: -0.0022
  citation_delta: -7

train install-only margin 45:
  F1 delta: -0.0008
  citation_delta: -3

train install-only margin 60:
  F1 delta: +0.0002
  citation_delta: 0

train install-only margin 100:
  F1 delta: 0.0000
  citation_delta: 0
```

How-to 结论：

- 提高 how-to margin/ratio 不能可靠解决风险。
- Train 里存在 high-margin how-to false positive，例如：

```text
TRAIN_Q384
route: how_to_or_lookup
first-vs-second margin: about 100.6
ratio: about 1.977
gold in top3: true
gold top1: false
```

这说明单靠更高分数差会继续放过一部分 how-to citation loss。

### 本阶段代码改动

- 更新 `RouteAwareCompositionPolicy`：
  - policy name 改为 `route_aware_strict_install_top1_otherwise_top3`
  - `enable_how_to_top1=False`
  - `install_upgrade_score_margin_min=60.0`
  - how-to top1 只能显式启用，默认关闭
- 更新 `scripts/analyze_route_aware_composition.py`：
  - 新增 `--enable-how-to-top1/--disable-how-to-top1`
  - 默认 `--disable-how-to-top1`
  - 默认 install margin 改为 `60.0`
- 更新 `route_aware_composition_cv.py`：
  - 默认 install margin grid 改为 `45, 50, 60, 70, 80, 100`
  - CV 默认禁用 how-to top1
- 更新 `create_answer_composition_policy`：
  - 保留旧 route-aware policy name alias，避免旧报告里的名字无法创建策略
- 更新测试：
  - 覆盖 how-to 默认不再 top1
  - 覆盖 how-to 必须显式启用才会 top1
  - runtime route-aware 测试改用足够强的 install margin

### 离线验证

Dev：

```powershell
python scripts\analyze_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --output artifacts\route_aware_composition_policy_stage25_dev_strict_install_only.json
```

结果：

```text
average_baseline_f1: 0.2808
average_policy_f1: 0.2810
average_f1_delta: +0.0002
baseline gold citation: 95
policy gold citation: 95
citation_delta: 0
changed_answer_count: 1
f1_improved_count: 1
f1_regressed_count: 0
citation_lost_count: 0
```

Train：

```powershell
python scripts\analyze_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --output artifacts\route_aware_composition_policy_stage25_train_strict_install_only.json
```

结果：

```text
average_baseline_f1: 0.2609
average_policy_f1: 0.2614
average_f1_delta: +0.0005
baseline gold citation: 232
policy gold citation: 232
citation_delta: 0
changed_answer_count: 5
f1_improved_count: 5
f1_regressed_count: 0
citation_lost_count: 0
```

### CV 验证

Dev：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --fold-count 5 `
  --output artifacts\route_aware_composition_cv_stage25_dev_strict_install_only.json
```

结果：

```text
average_baseline_f1: 0.2808
average_policy_f1: 0.2818
average_f1_delta: +0.0010
baseline gold citation: 95
policy gold citation: 95
citation_delta: 0
citation_lost_count: 0
selected_margin_counts:
  45.0: 4
  50.0: 1
```

Train：

```powershell
python scripts\cross_validate_route_aware_composition.py `
  --answer-gap-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --fold-count 5 `
  --output artifacts\route_aware_composition_cv_stage25_train_strict_install_only.json
```

结果：

```text
average_baseline_f1: 0.2609
average_policy_f1: 0.2612
average_f1_delta: +0.0003
baseline gold citation: 232
policy gold citation: 232
citation_delta: 0
citation_lost_count: 0
selected_margin_counts:
  60.0: 4
  70.0: 1
```

### End-To-End Verified RAG

Dev route-aware：

```powershell
python scripts\evaluate_verified_rag.py `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_m15_stage25_route_aware_report.json
```

Result：

```text
composition_policy: route_aware_strict_install_top1_otherwise_top3

verified average_token_f1: 0.2757
verified gold_doc_citation_rate: 0.6013
verified gold citation count: 95 / 158 generated answerable
answerable_refusal_rate: 0.0125
unanswerable_refusal_rate: 0.0267
newly_refused: 6
```

Compared with Stage 23 top-k report:

```text
top-k verified average_token_f1: 0.2755
route-aware verified average_token_f1: 0.2757
F1 delta: +0.0002
gold citation delta: 0
```

Train route-aware：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy route-aware `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_m15_stage25_route_aware_report.json
```

Result：

```text
composition_policy: route_aware_strict_install_top1_otherwise_top3

verified average_token_f1: 0.2562
verified gold_doc_citation_rate: 0.5249
verified gold citation count: about 232 / 442 generated answerable
answerable_refusal_rate: 0.0178
unanswerable_refusal_rate: 0.0333
newly_refused: 13
```

Compared with Stage 24 top-k report:

```text
top-k verified average_token_f1: 0.2557
route-aware verified average_token_f1: 0.2562
F1 delta: +0.0005
gold citation delta: 0
```

逐题 verified 对比：

```text
dev citation losses: 0
dev citation gains: 0
dev changed answers: 4

train citation losses: 0
train citation gains: 0
train changed answers: 8
```

Changed answer routes：

```text
dev changed answerable:
  DEV_Q022 install_upgrade_config F1 delta +0.0326

train changed answerable:
  TRAIN_Q013 install_upgrade_config F1 delta +0.1175
  TRAIN_Q020 install_upgrade_config F1 delta +0.0781
  TRAIN_Q136 install_upgrade_config F1 delta +0.0203
  TRAIN_Q160 install_upgrade_config F1 delta +0.0101
  TRAIN_Q345 install_upgrade_config F1 delta +0.0068
```

### 测试

```powershell
python -m ruff check .
python -m pytest -q
```

结果：

```text
ruff: passed
pytest: 58 passed
```

### 结论

- Stage 25 解决了 Stage 24 暴露的 how-to citation risk：
  how-to top1 默认关闭后，dev/train verified 都没有 citation loss。
- 新 route-aware policy 的收益非常小：
  - dev verified F1 `+0.0002`
  - train verified F1 `+0.0005`
- 这说明它更像“安全的微调实验参数”，不是足够有价值的默认策略。
- 默认 runtime 仍然保持 `top-k`。
- `route-aware` 可以保留，但当前不值得默认化。

### 我学到的

- 数据驱动的下一步不一定是继续加复杂规则；有时是关闭不稳定分支。
- How-to 问题表面上像 direct answer，但 top1 文档不一定是 gold 文档。
- CV 让策略从“dev 上很好”变成“跨 split 更诚实”；收益变小了，但风险也被压下来了。
- 当收益只有 `+0.0002` 到 `+0.0005` 时，默认化的工程价值不足。

### 下一步

- 做 Stage 26：决定 route-aware 的定位。
- 可选方向：
  1. 保留为实验参数，转向更高收益的 evidence selector / retrieval 改进；
  2. 分析 install-only changed cases，看看是否能形成更有价值的 install-specific policy；
  3. 回到 answer-gap 的最大痛点：gold document in context but selected answer weak，继续改 evidence selection 而不是 composition。
- 当前建议：不要继续在 route-aware composition 上做小阈值调参，转向 evidence selection 或 retrieval。

## Stage 26 - Answer-Gap Priority Analysis

### 目标

- 收口 route-aware composition 的阶段性结论。
- 不继续做低收益阈值微调。
- 回到更有收益空间的 answer-gap / evidence-selection 问题。
- 用 dev + train 的 answer-gap reports 统一排序，找下一阶段最值得改的 route/bucket。

### 起始状态

```text
git: main...origin/main clean
```

Stage 25 结论：

- route-aware 已经变成较安全的实验参数。
- 但 verified F1 提升只有：
  - dev: `+0.0002`
  - train: `+0.0005`
- 默认 runtime 继续使用 `top-k`。
- 继续调 route-aware composition 的收益空间很小。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/answer_gap_priority_analysis.py`
  - 跨多个 answer-gap report 聚合 route / bucket / route-bucket。
  - 统计事实字段：
    - total cases
    - gold in context
    - selected gold citation
    - gold-in-context-not-selected
    - gold-window-beats-selected
    - average selected F1
    - average best gold window F1
    - average gold-window gap
  - 生成代表样本。
  - 生成 heuristic priority score。
- 新增 `scripts/analyze_answer_gap_priorities.py`
  - 支持多个 answer-gap report。
  - 支持 `--min-cases`。
  - 支持 `--sample-limit-per-group`。
- 新增 `tests/test_answer_gap_priority_analysis.py`
  - 覆盖 route/bucket 排序。
  - 覆盖缺少 `cases` 的错误输入。

### 事实边界

- `priority_score` 是启发式排序指标，不是模型指标。
- 它只用来帮助决定下一阶段看哪里。
- 真实事实字段仍然是 count、F1、gold-in-context、selected-gold-citation 等。

### 命令

```powershell
python scripts\analyze_answer_gap_priorities.py `
  --answer-gap-reports artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json,artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --min-cases 3 `
  --sample-limit-per-group 5 `
  --output artifacts\answer_gap_priority_stage26_dev_train_hybrid.json
```

### 结果

Total cases:

```text
610 answerable cases
source reports: dev + train
```

Top route priorities:

```text
other:
  total_cases: 241
  gold_in_context_count: 143
  selected_gold_citation_count: 112
  gold_in_context_not_selected_count: 31
  gold_window_beats_selected_count: 112
  average_selected_answer_f1: 0.2322
  average_best_gold_window_f1: 0.8857
  average_gold_window_f1_gap: 0.6535

error_or_log:
  total_cases: 119
  gold_in_context_count: 64
  selected_gold_citation_count: 51
  gold_in_context_not_selected_count: 13
  gold_window_beats_selected_count: 51
  average_gold_window_f1_gap: 0.6813

security_bulletin_vulnerability_detail:
  total_cases: 87
  gold_in_context_count: 78
  selected_gold_citation_count: 78
  gold_window_beats_selected_count: 77
  average_gold_window_f1_gap: 0.5068
```

Top route-bucket priorities:

```text
other::gold_window_beats_selected_answer:
  total_cases: 112
  gold_in_context_count: 112
  selected_gold_citation_count: 112
  average_selected_answer_f1: 0.3172
  average_best_gold_window_f1: 0.8953
  average_gold_window_f1_gap: 0.5781

security_bulletin_vulnerability_detail::gold_window_beats_selected_answer:
  total_cases: 77
  gold_in_context_count: 77
  selected_gold_citation_count: 77
  average_selected_answer_f1: 0.4590
  average_best_gold_window_f1: 0.9415
  average_gold_window_f1_gap: 0.4825

other::gold_in_context_not_selected:
  total_cases: 31
  gold_in_context_count: 31
  selected_gold_citation_count: 0
  average_selected_answer_f1: 0.1628
  average_best_gold_window_f1: 0.8740
  average_gold_window_f1_gap: 0.7112
```

Representative cases for the top priority:

```text
TRAIN_Q374
route: other
bucket: gold_window_beats_selected_answer
selected F1: 0.0494
best gold window F1: 0.9921
gold retrieval rank: 1
selected gold candidate count: 3

TRAIN_Q379
route: other
bucket: gold_window_beats_selected_answer
selected F1: 0.0200
best gold window F1: 0.9565
gold retrieval rank: 2
selected gold candidate count: 1

DEV_Q134
route: other
bucket: gold_window_beats_selected_answer
selected F1: 0.1010
best gold window F1: 1.0000
gold retrieval rank: 1
selected gold candidate count: 1
```

### 解释

- 最大痛点不是单纯 retrieval。
- 很多 case 已经满足：
  - gold document is in context；
  - selected answer already cites gold document；
  - 但 selected sentence/window 与 gold answer 的 token F1 很低。
- 这说明下一阶段应优先改 evidence selection 的“gold 文档内部选句/选窗口能力”。
- `other` route 的问题最大，说明只继续做 security/install/how-to 的手写 route policy 不够。

### 测试

```powershell
python -m ruff check .
python -m pytest -q
```

结果：

```text
ruff: passed
pytest: 60 passed
```

### 结论

- Stage 26 把方向从 composition 阈值调参切回 evidence selection。
- 下一阶段最值得做的是：
  `other::gold_window_beats_selected_answer`
- 这类问题的 gold 文档通常已经被检索并引用，但候选句/窗口不够 answer-like。
- 继续做 route-aware composition 的边际收益不高。

### 我学到的

- 只看 gold citation 会误导判断：有些答案已经引用 gold 文档，但句子选得很差。
- `gold_window_beats_selected_answer` 比 `gold_in_context_not_selected` 更像“选句质量”问题。
- route 分类为 `other` 并不意味着低价值；它反而是最大的损失池。
- 下一阶段需要让 selector 更会在一个已知相关文档内部选出 answer span，而不是继续缩短最终答案。

### 下一步

- 做 Stage 27：针对 `other::gold_window_beats_selected_answer` 设计 answer-window selector 改进。
- 候选方向：
  1. 在 `other` route 上引入更强的 section/window answer-pattern scoring；
  2. 对 selected gold document 内部做 second-pass window rerank；
  3. 用 `best_gold_window` 代表样本分析哪些非 gold 特征可以帮助 selector 找到答案窗口。
- 不建议下一步继续调 route-aware composition。

## Stage 27 - Answer-Window Selector Experiment

### 目标

- 针对 Stage 26 找出的最高优先级问题：`other::gold_window_beats_selected_answer`。
- 验证一个不依赖 gold answer、不依赖 LLM judge 的 answer-window selector 是否能改善普通 `other` route 的选窗质量。
- 保持 Stage 18/25 的 `hybrid-routing` baseline 不变，新 selector 只作为可选实验入口。
- 使用 dev + train split 做跨 split 检查，避免只看单一 split 后误判。

### 起始状态

```text
git: main...origin/main clean
```

Stage 26 结论：

- 最大损失池不是 retrieval 缺失，而是 gold 文档已经在 context 内、甚至已经被引用，但 selected answer 没选中更贴近 gold answer 的连续句子窗口。
- 下一步应优先尝试 answer-window / local window reranking，而不是继续微调 route-aware composition。

### 本阶段新增内容

- 新增 `AnswerWindowBM25SentenceEvidenceSelector`：
  - 复用现有 BM25 + answer-aware scoring；
  - 在文档 section 内构造连续句子窗口；
  - 增加通用 action / resolution / procedure 特征；
  - 增加 compact window multiplier 和长度惩罚。
- 新增 `_collect_section_window_rows`：
  - 抽出 section-span 和 answer-window 共用的 window candidate 收集逻辑；
  - 避免 section-span 与 answer-window 重复维护同一套窗口枚举代码。
- 新增 `HybridWindowRoutingEvidenceSelector`：
  - `security_bulletin_vulnerability_detail` 和 `limitation_or_restriction` 仍走 section-span；
  - `other` route 走 answer-window；
  - 其他 route 保持 answer-aware；
  - 默认不替换现有 `hybrid-routing`。
- 更新 selector factory 和脚本 help：
  - `answer-window`
  - `hybrid-window-routing`
- 新增测试：
  - answer-window 可以选中 compact answer window；
  - factory 可以创建 answer-window；
  - hybrid-window-routing 会把 `other` route 解释为 answer-window；
  - factory 可以创建 hybrid-window-routing。

### 事实边界

- 本阶段没有使用 gold answer 参与 runtime selector。
- `best_gold_window` 只用于离线诊断和评估，不参与候选排序。
- `hybrid-window-routing` 是可选实验参数，不是默认 runtime。
- 当前结果不能被描述为成功晋升，因为 dev split 明显下降。

### 命令

正式 dev report：

```powershell
python scripts\analyze_answer_gap.py `
  --split dev `
  --evidence-selector hybrid-window-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_hybrid_window_routing_mcpd3_stage27.json
```

dev selector comparison：

```powershell
python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --challenger-report artifacts\answer_gap_analysis_dev_hybrid_window_routing_mcpd3_stage27.json `
  --baseline-label hybrid-stage18 `
  --challenger-label hybrid-window-stage27 `
  --sample-limit-per-bucket 50 `
  --output artifacts\selector_comparison_hybrid_vs_hybrid_window_stage27_dev.json
```

正式 train report：

```powershell
python scripts\analyze_answer_gap.py `
  --split train `
  --evidence-selector hybrid-window-routing `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_train_hybrid_window_routing_mcpd3_stage27.json
```

train selector comparison：

```powershell
python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --challenger-report artifacts\answer_gap_analysis_train_hybrid_window_routing_mcpd3_stage27.json `
  --baseline-label hybrid-stage24-train `
  --challenger-label hybrid-window-stage27-train `
  --sample-limit-per-bucket 50 `
  --output artifacts\selector_comparison_hybrid_vs_hybrid_window_stage27_train.json
```

### 结果

Dev answer-gap：

```text
baseline F1: 0.2805
hybrid-window F1: 0.2495
F1 delta: -0.0310

baseline gold citation: 95 / 160
hybrid-window gold citation: 99 / 160
citation delta: +4

baseline gold_window_beats_selected_answer: 95
hybrid-window gold_window_beats_selected_answer: 98
```

Dev selector comparison：

```text
baseline wins: 39
hybrid-window wins: 14
ties: 107
avg F1 delta: -0.0310

other route:
  baseline wins: 39
  hybrid-window wins: 14
  ties: 18
```

Train answer-gap：

```text
baseline F1: 0.2596
hybrid-window F1: 0.2668
F1 delta: +0.0072

baseline gold citation: 232 / 450
hybrid-window gold citation: 240 / 450
citation delta: +8

baseline gold_window_beats_selected_answer: 231
hybrid-window gold_window_beats_selected_answer: 238
```

Train selector comparison：

```text
baseline wins: 50
hybrid-window wins: 71
ties: 329
avg F1 delta: +0.0072

other route:
  baseline wins: 50
  hybrid-window wins: 71
  ties: 49
```

### 解释

- answer-window 在 train split 上有小幅提升，但在 dev split 上明显下降。
- 这说明当前 answer-window 规则不是稳定泛化改进，更像对 train 分布有收益、对 dev 分布有伤害。
- dev 下降的主要原因不是 citation 变差：
  - dev citation 从 95 增加到 99；
  - 但 selected answer F1 从 0.2805 降到 0.2495。
- 这说明 answer-window 更常引用 gold 文档，但窗口文本更长或更偏上下文，导致 token F1 被稀释。
- 负例样本显示，answer-window 会把相邻但不够 answer-like 的解释、检查项或重复窗口带进 top candidates。
- 因此，不能把 `other` route 整体切到 answer-window。

### 问题与原因

- 问题 1：`other` route 太杂。
  - 它包含 support guide、配置说明、限制说明、定义解释、排障步骤等多种答案形态。
  - 单一 answer-window scoring 不能稳定覆盖这些形态。
- 问题 2：窗口扩展增加 citation，但不一定增加答案质量。
  - 更长的窗口更容易覆盖 gold 文档。
  - 但也更容易混入非答案上下文，F1 反而下降。
- 问题 3：当前 selector 只知道 query 和 document text，不知道“已选 gold document 内哪一小段最像最终答案”。
  - 这意味着单纯 route-level 切换不够细。

### 修正与处理

- 保留 `answer-window` 和 `hybrid-window-routing` 作为可选实验入口。
- 不把它们设为默认 runtime。
- 不替换 Stage 18/25 的 `hybrid-routing` baseline。
- 将 `hybrid-window-routing` 的 answer-window 每文档候选数设为 1，而不是 3：
  - dev 探针显示 mcpd=1 比 mcpd=3 更少伤害 citation 和 F1；
  - 但 mcpd=1 仍未达到晋升条件。

### 测试

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 64 passed
```

### 结论

- Stage 27 是一个负实验，不应晋升 runtime 默认策略。
- answer-window 的方向有价值，但不能用 route-level 全量切换解决。
- 当前更可靠的结论是：
  - `other` route 需要进一步细分；
  - 或者需要 second-pass local rerank，只在已选高置信文档内部做窗口替换；
  - 不应把全部 `other` 问题直接交给 answer-window。

### 我学到的

- citation 增加不等于答案质量提高。
- 连续窗口能提升召回，但会放大“答案稀释”问题。
- dev/train 分歧非常重要：train 小涨不能掩盖 dev 明显下降。
- 对 `other` route 的改进不能继续用粗路由，要进入更细的 question subtype 或 candidate-level rerank。

### 下一步

- 做 Stage 28：分析 Stage 27 的 other route 胜负样本，设计更细的 gating 或 second-pass rerank。
- 优先比较两条路线：
  1. `other` subtype classifier：只让少数 procedure/support-guide 型问题使用 answer-window。
  2. local second-pass rerank：先沿用 hybrid-routing 选中候选，再在候选所在文档/section 内寻找更紧凑窗口。
- Stage 28 开始前不应改默认 selector。

## Stage 28 - Other-Route Window Outcome Analysis

### 目标

- 接着 Stage 27 的负实验，分析 `other` route 中 answer-window 的胜负样本。
- 判断是否存在稳定的 `other` subtype，可以安全地做 subtype gating。
- 如果 subtype gating 不稳定，就把下一步转向 candidate-level / local second-pass rerank。
- 继续使用 dev + train split 做跨 split 检查，不只看整体平均值。

### 起始状态

```text
git: main...origin/main clean
```

Stage 27 结论：

- `hybrid-window-routing` 在 train 上小涨，但在 dev 上明显下降。
- 直接把全部 `other` route 交给 answer-window 不稳定。
- 需要更细的 gating 或 second-pass rerank。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/other_route_window_outcome_analysis.py`
  - 读取 baseline/challenger 的 answer-gap reports。
  - 只分析 challenger route 为 `other` 的样本。
  - 用 runtime 可见字段 `question_title` / `question_text` 做 subtype 分类。
  - 统计每个 subtype 的：
    - total cases
    - baseline wins
    - challenger wins
    - ties
    - average F1 delta
    - gold citation delta
    - recommendation
    - representative cases
  - 同时生成 overall summary 和 source-level summary。
- 新增 `scripts/analyze_other_route_window_outcomes.py`
  - 支持多个 baseline/challenger report pair。
  - 支持 `--source-labels`、`--min-cases`、`--sample-limit-per-subtype`。
- 新增 `tests/test_other_route_window_outcome_analysis.py`
  - 验证 subtype 分类只用 runtime-visible text。
  - 验证 dev/train 分歧不会被误判为 stable candidate。
  - 验证真正跨 source 都正向的 subtype 才会进入 `stable_answer_window_subtypes`。

### 事实边界

- 本阶段没有修改默认 runtime selector。
- 本阶段没有新增 fallback 策略。
- subtype classifier 没有使用 gold answer、selected answer 或离线 best_gold_window。
- 该分析是 heuristic diagnostic，不是模型指标，也不是最终 gating policy。

### 命令

```powershell
python scripts\analyze_other_route_window_outcomes.py `
  --baseline-reports artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json,artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --challenger-reports artifacts\answer_gap_analysis_dev_hybrid_window_routing_mcpd3_stage27.json,artifacts\answer_gap_analysis_train_hybrid_window_routing_mcpd3_stage27.json `
  --source-labels dev,train `
  --min-cases 3 `
  --sample-limit-per-subtype 5 `
  --output artifacts\other_route_window_outcome_stage28_dev_train.json
```

### 结果

总体：

```text
total other-route cases: 241
dev cases: 71
train cases: 170
stable_answer_window_subtypes: []
```

Overall subtype summary：

```text
support_or_download:
  cases: 16
  baseline wins: 3
  answer-window wins: 6
  ties: 7
  avg F1 delta: +0.0198
  recommendation: candidate_answer_window

capability_or_support:
  cases: 49
  baseline wins: 18
  answer-window wins: 18
  ties: 13
  avg F1 delta: +0.0100
  recommendation: mixed_or_insufficient

configuration_or_property:
  cases: 18
  baseline wins: 9
  answer-window wins: 7
  ties: 2
  avg F1 delta: +0.0077
  recommendation: mixed_or_insufficient

procedure_or_change:
  cases: 60
  baseline wins: 25
  answer-window wins: 25
  ties: 10
  avg F1 delta: +0.0056
  recommendation: mixed_or_insufficient

general_other:
  cases: 53
  baseline wins: 23
  answer-window wins: 13
  ties: 17
  avg F1 delta: -0.0266
  recommendation: keep_baseline

failure_or_behavior:
  cases: 43
  baseline wins: 9
  answer-window wins: 16
  ties: 18
  avg F1 delta: -0.0269
  recommendation: mixed_or_insufficient
```

Source-level split：

```text
support_or_download:
  dev:   avg F1 delta -0.0067, keep_baseline
  train: avg F1 delta +0.0319, candidate_answer_window

capability_or_support:
  dev:   avg F1 delta -0.0715, keep_baseline
  train: avg F1 delta +0.0365, candidate_answer_window

configuration_or_property:
  dev:   avg F1 delta -0.0358, keep_baseline
  train: avg F1 delta +0.0425, candidate_answer_window

procedure_or_change:
  dev:   avg F1 delta -0.0535, keep_baseline
  train: avg F1 delta +0.0290, candidate_answer_window

failure_or_behavior:
  dev:   avg F1 delta -0.0762, keep_baseline
  train: avg F1 delta +0.0023, candidate_answer_window

general_other:
  dev:   avg F1 delta -0.1153, keep_baseline
  train: avg F1 delta -0.0006, keep_baseline
```

### 解释

- 没有任何 subtype 同时在 dev 和 train 上稳定支持 answer-window。
- `support_or_download` 是 overall 最好的 subtype，但它仍然是：
  - dev 负向；
  - train 正向；
  - 因此不能作为可靠 gating 规则。
- 多个 subtype 出现相同形态：
  - dev 上 keep baseline；
  - train 上 candidate answer-window。
- 这说明 Stage 27 的 train 小涨很可能来自 train 分布，而不是稳定可迁移的 subtype 规则。
- `general_other` 在 dev/train 都不支持 answer-window，应该明确保留 baseline。

### 问题与原因

- 问题 1：只靠 question subtype 仍然太粗。
  - 同一个 subtype 内部既有 answer-window 大赢样本，也有 answer-window 大输样本。
- 问题 2：subtype gating 无法判断候选窗口是否混入了相邻噪声。
  - 它只能判断问题形态，不能判断候选证据质量。
- 问题 3：Stage 26 暴露的是“已选文档内部选窗差”，不是“某类问题一定该换 selector”。
  - 因此 candidate-level / local rerank 比 route-level / subtype-level gating 更贴近问题根因。

### 修正与处理

- 不做 subtype gating runtime 改动。
- 保留 Stage 28 分析工具，作为以后评估 gating 规则的离线工具。
- 将下一步方向从 subtype classifier 转向 local second-pass rerank。

### 测试

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 67 passed
```

### 结论

- Stage 28 否定了“直接做 other subtype gating”的路线。
- 当前证据不支持让任何 `other` subtype 稳定切到 answer-window。
- 下一步应该做更细粒度的 candidate-level local rerank：
  - 先沿用当前 `hybrid-routing` 产生候选；
  - 再只在候选所在的高置信文档或 section 内寻找更紧凑、更 answer-like 的窗口；
  - 用 dev/train 验证是否能减少 `gold_window_beats_selected_answer`，且不牺牲 selected F1。

### 我学到的

- overall 小幅正收益仍可能被 split-level 分歧否定。
- “问题类型看起来像步骤/下载/支持”不等于 answer-window 一定更好。
- 这个项目当前更需要候选质量判断，而不是继续加粗粒度路由。
- 数据分析工具本身也要防止 sample-limited comparison report 带来的偏差，所以 Stage 28 改为直接读完整 answer-gap reports。

### 下一步

- 做 Stage 29：实现 local second-pass window rerank 的离线原型。
- 约束：
  - 不改变默认 runtime；
  - 不使用 gold answer；
  - 只在 baseline 已选候选的 document / section 范围内 rerank；
  - dev 不能下降，train 不能只靠单 split 偶然上涨。

## Stage 29 - Local Second-Pass Window Rerank Prototype

### 目标

- 接着 Stage 28 的结论，验证 candidate-level local rerank 是否比 subtype gating 更稳。
- 原型只处理 `other` route。
- 原型只围绕 baseline 已选候选的局部文档窗口生成替换候选。
- 不使用 gold answer、best_gold_window 或 LLM judge。
- 不改变默认 runtime selector。

### 起始状态

```text
git: main...origin/main clean
```

Stage 28 结论：

- 没有任何 `other` subtype 在 dev/train 上稳定支持 answer-window。
- subtype gating 不应进入 runtime。
- 下一步应验证更细粒度的 local second-pass rerank。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/local_window_rerank.py`
  - 新增 `LocalWindowRerankEvidenceSelector`。
  - 默认包装当前 `HybridRoutingEvidenceSelector`。
  - 只对 `question_route == "other"` 的前 3 个 baseline candidates 做局部窗口替换。
  - 在候选句子所在文档的相邻句子范围内生成最多 3 句窗口。
  - 用 query overlap、anchor coverage、answer signal、compactness 和 noise penalty 做局部窗口排序。
  - 保留原候选分数和候选顺序，只替换候选文本。
- 更新 selector factory：
  - 新增 `local-window-rerank`
  - 新增 `local_window_rerank`
- 更新 `trace_selector_route`
  - `other` route 记录为 `local_window_rerank`
  - 非 `other` route 记录为保留 baseline `hybrid_routing`
- 更新脚本 help：
  - `analyze_answer_gap.py`
  - `analyze_evidence_selection.py`
  - `evaluate_verified_rag.py`
  - `sweep_verified_rag_thresholds.py`
- 新增 `tests/test_local_window_rerank.py`
  - 覆盖局部窗口扩展。
  - 覆盖非目标 route 保持 base selector。
  - 覆盖 factory 创建。
  - 覆盖 trace 解释。

### 事实边界

- 本阶段没有把 `local-window-rerank` 设为默认 selector。
- 本阶段没有使用 gold answer 参与 rerank。
- 本阶段没有使用离线 `best_gold_window` 参与 rerank。
- 本阶段没有引入 LLM judge。
- 本阶段的 local rerank 是强制替换式原型，不包含“替换安全门控”。

### 命令

Dev answer-gap：

```powershell
python scripts\analyze_answer_gap.py `
  --split dev `
  --evidence-selector local-window-rerank `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_dev_local_window_rerank_mcpd3_stage29.json
```

Dev selector comparison：

```powershell
python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_dev_hybrid_routing_mcpd3_stage18_cve_anchor.json `
  --challenger-report artifacts\answer_gap_analysis_dev_local_window_rerank_mcpd3_stage29.json `
  --baseline-label hybrid-stage18 `
  --challenger-label local-window-stage29 `
  --sample-limit-per-bucket 50 `
  --output artifacts\selector_comparison_hybrid_vs_local_window_stage29_dev.json
```

Train answer-gap：

```powershell
python scripts\analyze_answer_gap.py `
  --split train `
  --evidence-selector local-window-rerank `
  --max-candidates-per-document 3 `
  --sample-limit 1000 `
  --output artifacts\answer_gap_analysis_train_local_window_rerank_mcpd3_stage29.json
```

Train selector comparison：

```powershell
python scripts\compare_selectors.py `
  --baseline-report artifacts\answer_gap_analysis_train_hybrid_routing_mcpd3_stage24_cv_source.json `
  --challenger-report artifacts\answer_gap_analysis_train_local_window_rerank_mcpd3_stage29.json `
  --baseline-label hybrid-stage24-train `
  --challenger-label local-window-stage29-train `
  --sample-limit-per-bucket 50 `
  --output artifacts\selector_comparison_hybrid_vs_local_window_stage29_train.json
```

### 结果

Dev answer-gap：

```text
baseline F1: 0.2805
local-window F1: 0.2410
F1 delta: -0.0395

baseline gold citation: 95 / 160
local-window gold citation: 95 / 160
citation delta: 0

baseline gold_window_beats_selected_answer: 95
local-window gold_window_beats_selected_answer: 95
```

Dev selector comparison：

```text
baseline wins: 47
local-window wins: 10
ties: 103
avg F1 delta: -0.0395

other route:
  baseline wins: 47
  local-window wins: 10
  ties: 14
```

Train answer-gap：

```text
baseline F1: 0.2596
local-window F1: 0.2497
F1 delta: -0.0099

baseline gold citation: 232 / 450
local-window gold citation: 232 / 450
citation delta: 0

baseline gold_window_beats_selected_answer: 231
local-window gold_window_beats_selected_answer: 230
```

Train selector comparison：

```text
baseline wins: 73
local-window wins: 33
ties: 344
avg F1 delta: -0.0099

other route:
  baseline wins: 73
  local-window wins: 33
  ties: 64
```

### 解释

- 强制 local window replacement 在 dev/train 都下降。
- citation 没有变化：
  - dev citation delta: 0
  - train citation delta: 0
- 说明问题不是引用了错误文档，而是同一文档/局部窗口中混入了不该进入最终答案的上下文。
- 赢例通常是 local window 补全了相邻答案句，例如 support guide、步骤补全、属性说明。
- 负例更常见，通常是 local window 把以下内容带入答案：
  - `QUESTION`
  - `SYMPTOM`
  - `DIAGNOSING THE PROBLEM`
  - 相邻但不同问题的解释段
  - 过长的配置说明或背景说明
- 这和 Stage 27 的失败原因一致：窗口扩展会提高局部召回，但更容易稀释答案。

### 问题与原因

- 问题 1：只要强制替换，就无法区分“补全答案”和“引入噪声”。
- 问题 2：局部窗口排序仍然只看 query/anchor/answer-like 信号，不知道替换后是否更贴近最终答案。
- 问题 3：保留候选顺序但替换文本，会让 citation 不变、F1 下降；这说明 rerank 的核心不是引用选择，而是替换安全性。

### 修正与处理

- 保留 `local-window-rerank` 作为可选实验 selector。
- 不晋升默认 runtime。
- 不继续在强制替换版本上调权重。
- 下一步如果继续这个方向，必须先设计替换安全门控，而不是继续扩大窗口。

### 测试

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 71 passed
```

### 结论

- Stage 29 是负实验。
- `local-window-rerank` 原型证明：candidate-level local rerank 比 route-level answer-window 更精细，但“强制替换”仍然不可用。
- 当前不能把 local second-pass window rerank 作为默认或推荐 runtime 策略。

### 我学到的

- 不改变 citation 也可能显著降低 F1。
- evidence selection 的下一层问题不是“找更多局部窗口”，而是“判断什么时候不能替换原句”。
- candidate-level 比 route-level 更接近根因，但还缺少替换安全门控。
- 继续调 score 权重很可能只是局部优化，不能解决强制替换的结构问题。

### 下一步

- 做 Stage 30 前需要先确认是否引入“替换安全门控”。
- 可选方向：
  1. 保守门控：只有 local window 同时满足 answer-signal 增强、长度不过长、且不包含 problem/question/symptom heading 时才替换。
  2. 分析型 gate search：离线扫描多组门控条件，先找 dev/train 都不下降的条件，再决定是否实现。
  3. 暂停窗口替换路线，转向 retrieval/reranker 层，重新分析 `gold_in_context_not_selected`。
- 当前建议：先做方向 2，只做离线 gate search，不改默认 runtime。

## Stage 30 - Local-Window Replacement Gate Search

### 目标

- 接着 Stage 29 的负实验，验证是否存在“替换安全门控”。
- 只做离线 gate search，不改默认 runtime。
- 使用完整候选文本重新跑 dev/train，而不是只读取 answer-gap JSON 里的截断句子。
- gate 特征只能使用 runtime 可见信息：
  - baseline candidate text
  - forced local-window candidate text
  - question route
  - citation document id
- F1 只用于离线评估，不参与 gate 判断。

### 起始状态

```text
git: main...origin/main clean
```

Stage 29 结论：

- 强制 local-window replacement 在 dev/train 都下降。
- citation 不变但 F1 下降，说明问题是同一文档内部窗口替换不安全。
- 下一步必须先做 gate search，而不是继续扩大窗口或调权重。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/local_window_gate_search.py`
  - 定义 `LocalWindowGateConfig`。
  - 定义 5 组候选 gate：
    - `strict_answer_gain_no_heading`
    - `moderate_answer_gain_no_problem`
    - `compact_same_signal_no_heading`
    - `answer_heading_gain`
    - `shorter_same_signal_no_problem`
  - 支持 candidate-level gate：
    - token 长度限制
    - sentence 数限制
    - added token 限制
    - length ratio 限制
    - anchor coverage 限制
    - answer-signal delta 限制
    - problem/question heading noise 阻断
    - noise growth 阻断
  - 输出每个 gate 在每个 split 和 overall 的：
    - average F1
    - delta vs baseline
    - delta vs forced local-window
    - changed cases
    - replacement count
    - gold citation delta
    - win counts
  - stable gate candidate 条件：
    - 每个 split average delta vs baseline 不为负；
    - 每个 split citation 不下降；
    - 所有 split 总计至少发生过替换。
- 新增 `scripts/analyze_local_window_gate_search.py`
  - 重新加载 TechQA dev/train。
  - 重新建立 BM25 index。
  - 对每个 answerable question 重新生成：
    - baseline `hybrid-routing` candidates；
    - forced `local-window-rerank` candidates；
    - gated candidates。
  - 避免使用报告中被截断的 candidate sentence。
- 新增 `tests/test_local_window_gate_search.py`
  - 验证安全窗口可以通过 gate。
  - 验证 problem/symptom noise 会被阻断。
  - 验证非 `other` route 不替换。
  - 验证 stable candidate 必须跨 split 不下降且至少有实际替换。

### 事实边界

- 本阶段没有修改默认 selector。
- 本阶段没有实现 runtime gate 参数。
- 本阶段没有使用 gold answer 参与 gate 判断。
- 本阶段没有引入 LLM judge。
- `stable_gate_candidates` 是离线搜索结果，不等于已经可以默认上线。

### 命令

```powershell
python scripts\analyze_local_window_gate_search.py `
  --splits dev,train `
  --max-candidates-per-document 3 `
  --output artifacts\local_window_gate_search_stage30_dev_train.json
```

### 结果

Baseline 与 forced local-window：

```text
dev baseline F1: 0.2805
dev forced local-window F1: 0.2410

train baseline F1: 0.2596
train forced local-window F1: 0.2497
```

Stable gate candidates：

```text
stable_gate_candidates:
  - strict_answer_gain_no_heading
```

`strict_answer_gain_no_heading`：

```text
dev:
  F1: 0.2805
  delta vs baseline: +0.0000
  changed cases: 0
  replacements: 0
  wins: tie 160

train:
  F1: 0.2597
  delta vs baseline: +0.0001
  changed cases: 3
  replacements: 3
  wins: tie 450

overall:
  F1: 0.2651
  delta vs baseline: +0.0000
  changed cases: 3
  replacements: 3
  wins: tie 610
```

`moderate_answer_gain_no_problem`：

```text
dev:
  F1: 0.2781
  delta vs baseline: -0.0024
  changed cases: 13
  wins: baseline 4, tie 156

train:
  F1: 0.2606
  delta vs baseline: +0.0010
  changed cases: 31
  wins: gated 8, baseline 4, tie 438

overall:
  F1: 0.2652
  delta vs baseline: +0.0001
  changed cases: 44
  wins: gated 8, baseline 8, tie 594
```

其他 gate：

```text
answer_heading_gain:
  dev delta: -0.0014
  train delta: +0.0003

compact_same_signal_no_heading:
  dev delta: -0.0017
  train delta: +0.0002

shorter_same_signal_no_problem:
  dev delta: +0.0000
  train delta: +0.0000
  changed cases: 0
```

### 解释

- Stage 30 找到了一个非降 gate：`strict_answer_gain_no_heading`。
- 但它的实际工程价值很弱：
  - dev 完全没有替换；
  - train 只替换 3 个 case；
  - 所有 case 在 `f1_win_margin=0.03` 下仍然都是 tie；
  - overall average delta 四舍五入后仍是 `+0.0000`。
- `moderate_answer_gain_no_problem` 虽然 overall 看起来最高，且 train 有 `+0.0010`，但 dev 下降 `-0.0024`，不能作为稳定 gate。
- 这说明：
  - gate 变宽会重新引入噪声窗口；
  - gate 变严会几乎不替换；
  - local-window 路线目前处于“安全但无收益”或“有替换但不稳定”的两难。

### 问题与原因

- 问题 1：安全门控和收益之间冲突明显。
  - 严格 gate 防住了噪声，但几乎不触发。
  - 宽松 gate 触发更多，但 dev 下降。
- 问题 2：当前特征仍然只看文本形态。
  - 它能识别明显的 heading noise；
  - 但不能稳定判断“替换后是否更接近最终答案”。
- 问题 3：Stage 26 的 gap 很大，但局部规则很难吃到这部分收益。
  - 这说明更可能需要学习式 reranker 或更强的 answer-candidate scorer，而不是继续手写窗口规则。

### 修正与处理

- 保留 gate search 工具作为离线诊断工具。
- 不实现 runtime gate。
- 不把 `strict_answer_gain_no_heading` 晋升为 runtime 参数。
- 不继续扩大 local-window gate 的规则集合，避免继续局部调参。

### 测试

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 76 passed
```

### 结论

- Stage 30 找到了“非降但几乎无收益”的 gate。
- 这不足以支撑 runtime 实现。
- local-window replacement 路线目前应暂时收口：
  - 强制替换失败；
  - subtype gating 失败；
  - safety gate 只有极小 no-op 级收益。

### 我学到的

- “不下降”不等于“值得工程化”。
- 如果一个 gate 的安全性来自几乎不触发，它不能解决 Stage 26 的大 gap。
- 手写规则已经接近瓶颈，继续调 threshold 可能只是围着噪声打转。
- answer-gap 的最大空间仍在，但需要更强的 candidate scoring，而不是继续扩大固定规则。

### 下一步

- Stage 31 应该先在两条路线之间做选择：
  1. 收口 local-window 规则路线，转向 learned / feature-based candidate reranker 的离线数据集构建。
  2. 暂停 evidence selection，回到 retrieval 层重新看 `gold_in_context_not_selected`。
- 当前建议：选择方向 1。
  - 用现有 answer-gap / best_gold_window 生成训练样本；
  - 先做离线 feature dataset，不训练模型；
  - 明确区分 runtime-visible features 和 gold-derived labels。

## Stage 31 - Candidate Reranker Feature Dataset

### 目标

- 接着 Stage 30 的结论，收口手写 local-window 规则路线。
- 转向 learned / feature-based candidate reranker 的离线数据集构建。
- 本阶段只构建数据集，不训练模型。
- 明确区分：
  - runtime-visible features：未来 reranker 推理时可用；
  - gold-derived labels：只用于离线训练/评估，不允许进入 runtime features；
  - metadata：只用于人工检查，不作为模型数值特征。

### 起始状态

```text
git: main...origin/main clean
```

Stage 30 结论：

- `strict_answer_gain_no_heading` 是非降 gate，但几乎无收益。
- 宽松 gate 会重新引入 dev 下降。
- 继续调手写 gate 的收益很有限。
- 下一步应该构建 candidate-level feature dataset，为后续 learned / feature-based reranker 做准备。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/candidate_reranker_dataset.py`
  - 新增 `CandidateRerankerDatasetRow`
  - 新增 `CandidateRerankerQuestionSummary`
  - 新增 `CandidateRerankerDatasetSummary`
  - 新增 `CandidateRerankerDatasetBuild`
  - 支持从 selector ranked candidates 构建候选级别数据集。
- 新增 `scripts/build_candidate_reranker_dataset.py`
  - 支持 `--splits`
  - 支持 `--evidence-selector`
  - 支持 `--max-candidates-per-document`
  - 支持 `--candidate-limit`
  - 支持 `--min-candidate-score`
  - 输出 JSONL dataset 和 summary JSON。
- 新增 `tests/test_candidate_reranker_dataset.py`
  - 验证 runtime features 与 gold labels 分离。
  - 验证 question-level oracle gain summary。
  - 验证 min score 和 candidate limit 生效。

### 数据结构边界

每条 row 分为三块：

```text
runtime_features:
  未来 runtime 可用。
  不包含 gold answer、gold document、candidate token F1、best candidate label。

gold_labels:
  只用于离线训练/评估。
  包含 candidate_token_f1、is_gold_document、is_best_candidate_for_question 等。

metadata:
  只用于人工检查。
  包含 question title、document id/title、截断后的 candidate sentence。
```

本阶段 runtime features 包含：

```text
selector_name
question_route
retrieval_rank
retrieval_score
candidate_score
candidate_token_count
candidate_sentence_count
question_token_count
query_term_count
query_overlap_count
query_overlap_ratio
candidate_query_coverage_ratio
title_query_overlap_count
title_query_overlap_ratio
answer_signal_score
problem_noise_score
has_answer_heading
has_problem_heading
has_question_heading
has_url
has_trace_noise
symbol_ratio
```

gold labels 包含：

```text
candidate_token_f1
is_gold_document
is_best_candidate_for_question
best_candidate_token_f1_for_question
f1_gap_to_best_candidate
```

### 命令

```powershell
python scripts\build_candidate_reranker_dataset.py `
  --splits dev,train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --candidate-limit 25 `
  --min-candidate-score 2.0 `
  --output artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --summary-output artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
```

### 结果

```text
selector: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
answerable questions: 610
dataset rows: 8060

dev questions: 160
train questions: 450

dev rows: 2164
train rows: 5896

average rows per question: 13.2131
average top candidate token F1: 0.2361
average best candidate token F1: 0.4292
average oracle gain vs top candidate: 0.1931

questions with gold-document candidate: 378
gold-document candidate rows: 963
```

Rows by route：

```text
other: 3562
error_or_log: 1762
install_upgrade_config: 1353
how_to_or_lookup: 814
security_bulletin_vulnerability_detail: 424
limitation_or_restriction: 55
security_bulletin_post_fix_behavior: 45
security_bulletin_remediation: 30
security_bulletin_affected_product: 15
```

Artifacts：

```text
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
  size: 12,092,764 bytes

artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
  size: 217,743 bytes
```

### 解释

- 当前 top candidate 平均 F1 是 `0.2361`。
- 同一个 candidate pool 内的 oracle best 平均 F1 是 `0.4292`。
- 平均 oracle gain 是 `+0.1931`。
- 这说明 candidate pool 内确实经常已经存在更好的候选，问题是排序没有选中。
- 这比继续扩写 local-window 规则更适合作为 learned / feature-based reranker 的输入。
- 378 / 610 个 answerable question 的候选池中出现了 gold document candidate。
- 这说明 candidate reranker 仍受 retrieval/candidate pool 上限约束：
  - 如果 gold document 不在候选池，reranker 也无法凭空修复。

### 事实边界

- 本阶段没有训练模型。
- 本阶段没有改默认 runtime。
- 本阶段没有把 `gold_labels` 混入 `runtime_features`。
- 输出 dataset artifacts 没有纳入 git。
- `metadata` 里的文本是人工检查字段，不是当前 feature contract 的训练数值字段。

### 问题与原因

- 问题 1：候选池 oracle 很高，但 top candidate 明显偏低。
  - 这证明 reranking 有空间。
- 问题 2：仍有 232 个 answerable question 没有 gold-document candidate。
  - 这部分不是 reranker 能单独解决的。
- 问题 3：当前 feature set 仍是手写结构化特征。
  - 下一步训练前需要先检查 label 分布和 feature 泄漏风险。

### 修正与处理

- 明确 dataset contract：
  - `runtime_features` 可用于模型；
  - `gold_labels` 只用于训练/评估；
  - `metadata` 只用于检查。
- 先输出 JSONL 和 summary，不把训练逻辑混进本阶段。
- 后续训练模型之前，必须先做 dataset audit。

### 测试

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 79 passed
```

### 结论

- Stage 31 成功构建了 candidate-reranker 离线 feature dataset。
- 数据显示 candidate pool 内存在明显 oracle gain：
  `0.4292 - 0.2361 = +0.1931`
- 这支持下一阶段继续走 feature-based reranker 路线。
- 但还不能直接训练模型；需要先审计标签分布、split 差异和泄漏风险。

### 我学到的

- 继续手写窗口规则的收益有限，但候选池内部仍有很大排序空间。
- “候选池里有好答案”和“当前 selector 选中好答案”是两个不同问题。
- learned reranker 的第一步不是训练，而是把 feature/label 边界做干净。
- 数据集 contract 本身就是工程资产，能防止后续把 gold label 不小心带进 runtime。

### 下一步

- 做 Stage 32：candidate-reranker dataset audit。
- 审计内容：
  1. label 分布；
  2. best candidate rank 分布；
  3. route-level oracle gain；
  4. dev/train 差异；
  5. runtime feature 是否含有 label leakage 风险。
- 审计通过后，再考虑训练一个简单 baseline reranker。

## Stage 32 - Candidate Reranker Dataset Audit with Visualizations

### 目标

- 接着 Stage 31 的 candidate-reranker feature dataset，先做训练前审计。
- 本阶段不训练模型、不改 runtime，只验证数据是否适合进入 baseline reranker 实验。
- 审计重点：
  1. candidate label F1 分布；
  2. best candidate rank 分布；
  3. route-level oracle gain；
  4. dev/train split 差异；
  5. `runtime_features` 是否有明显 label leakage 风险；
  6. 输出可复跑的 JSON 审计报告和 SVG 可视化结果。

### 起始状态

```text
git: main...origin/main clean

Stage 31 dataset:
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
  rows: 8060
  unique row questions: 610

Stage 31 summary:
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json
  question_summaries: 610
  zero-candidate questions: 0
```

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/candidate_reranker_dataset_audit.py`
  - `CandidateRerankerQuestionAudit`
  - `F1DistributionBucket`
  - `RankDistributionBucket`
  - `RouteOracleGainSummary`
  - `SplitAuditSummary`
  - `FeatureLeakageAudit`
  - `DatasetConsistencyAudit`
  - `CandidateRerankerDatasetAudit`
  - JSONL / summary 读取函数
  - dataset audit 主函数
  - SVG bar chart 生成函数
- 新增 `scripts/audit_candidate_reranker_dataset.py`
  - 要求同时传入 Stage 31 JSONL 和 summary JSON。
  - 输出 audit JSON。
  - 输出 4 个 SVG 可视化图表。
- 新增 `tests/test_candidate_reranker_dataset_audit.py`
  - 验证 label/rank/route/split 汇总。
  - 验证 consistency mismatch 会被如实报告。
  - 验证 obvious runtime feature label leakage 能被识别。
  - 验证 `title_query_overlap_count`、`candidate_sentence_count` 这类聚合特征不会被误判为原始文本泄漏。

### 命令

```powershell
python scripts\audit_candidate_reranker_dataset.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\candidate_reranker_dataset_stage32_audit.json `
  --visualization-dir artifacts\candidate_reranker_dataset_stage32_visuals
```

### 结果

基础规模：

```text
total_rows: 8060
total_questions: 610
```

Consistency audit：

```text
summary_total_rows: 8060
actual_total_rows: 8060
total_rows_match: true

summary_total_questions: 610
actual_question_summary_count: 610
row_question_count: 610
total_questions_match: true

row_questions_without_summary_count: 0
summary_questions_without_rows_count: 0
rows_by_split_match: true
rows_by_route_match: true
```

Candidate token F1 label 分布：

```text
0.00-0.05: 1807 / 8060 = 22.42%
0.05-0.10: 1758 / 8060 = 21.81%
0.10-0.20: 2702 / 8060 = 33.52%
0.20-0.40: 1302 / 8060 = 16.15%
0.40-0.60: 297 / 8060 = 3.68%
0.60-0.80: 106 / 8060 = 1.32%
0.80-1.00: 88 / 8060 = 1.09%
```

Best candidate token F1 分布：

```text
0.00-0.05: 4 / 610 = 0.66%
0.05-0.10: 5 / 610 = 0.82%
0.10-0.20: 111 / 610 = 18.20%
0.20-0.40: 235 / 610 = 38.52%
0.40-0.60: 99 / 610 = 16.23%
0.60-0.80: 69 / 610 = 11.31%
0.80-1.00: 87 / 610 = 14.26%
```

Best candidate rank 分布：

```text
rank_1: 113 / 610 = 18.52%
rank_2: 81 / 610 = 13.28%
rank_3: 68 / 610 = 11.15%
rank_4_5: 91 / 610 = 14.92%
rank_6_10: 152 / 610 = 24.92%
rank_11_25: 105 / 610 = 17.21%
missing: 0 / 610 = 0.00%
```

Split summary：

```text
dev:
  question_count: 160
  average_oracle_gain_vs_top_candidate: +0.2125
  best_rank_1_rate: 21.88%
  gold_document_candidate_rate: 66.87%

train:
  question_count: 450
  average_oracle_gain_vs_top_candidate: +0.1861
  best_rank_1_rate: 17.33%
  gold_document_candidate_rate: 60.22%
```

Route-level oracle gain：

```text
security_bulletin_post_fix_behavior:
  n: 3
  average_oracle_gain_vs_top_candidate: +0.3409
  positive_oracle_gain_rate: 100.00%
  gold_document_candidate_rate: 66.67%

other:
  n: 241
  average_oracle_gain_vs_top_candidate: +0.2185
  positive_oracle_gain_rate: 83.40%
  gold_document_candidate_rate: 59.34%

error_or_log:
  n: 119
  average_oracle_gain_vs_top_candidate: +0.2167
  positive_oracle_gain_rate: 86.55%
  gold_document_candidate_rate: 53.78%

install_upgrade_config:
  n: 91
  average_oracle_gain_vs_top_candidate: +0.1836
  positive_oracle_gain_rate: 76.92%
  gold_document_candidate_rate: 51.65%

how_to_or_lookup:
  n: 55
  average_oracle_gain_vs_top_candidate: +0.1629
  positive_oracle_gain_rate: 78.18%
  gold_document_candidate_rate: 65.45%

limitation_or_restriction:
  n: 11
  average_oracle_gain_vs_top_candidate: +0.1523
  positive_oracle_gain_rate: 72.73%
  gold_document_candidate_rate: 45.45%

security_bulletin_vulnerability_detail:
  n: 87
  average_oracle_gain_vs_top_candidate: +0.1241
  positive_oracle_gain_rate: 77.01%
  gold_document_candidate_rate: 89.66%

security_bulletin_remediation:
  n: 2
  average_oracle_gain_vs_top_candidate: +0.0741
  positive_oracle_gain_rate: 100.00%
  gold_document_candidate_rate: 100.00%

security_bulletin_affected_product:
  n: 1
  average_oracle_gain_vs_top_candidate: +0.0000
  positive_oracle_gain_rate: 0.00%
  gold_document_candidate_rate: 100.00%
```

Feature leakage audit：

```text
suspicious_runtime_feature_keys: []
text_like_runtime_feature_keys: []
non_scalar_runtime_feature_keys: []
label_leakage_detected_from_keys: false
```

注意：这个 leakage audit 是静态 key/value-shape 审计，只能说明没有发现明显的 label key、raw text/id 字段或非标量字段混入 `runtime_features`。它不是统计意义上的因果安全证明。

Visualization artifacts：

```text
artifacts/candidate_reranker_dataset_stage32_visuals/candidate_label_f1_distribution.svg
artifacts/candidate_reranker_dataset_stage32_visuals/best_candidate_rank_distribution.svg
artifacts/candidate_reranker_dataset_stage32_visuals/route_oracle_gain.svg
artifacts/candidate_reranker_dataset_stage32_visuals/split_oracle_gain.svg
```

Audit JSON：

```text
artifacts/candidate_reranker_dataset_stage32_audit.json
```

上述 artifact 都是本地生成结果，没有纳入 git。

### 解释

- Candidate row 的 label 分布非常偏低：
  - F1 `< 0.20` 的 candidate row 有 `6267 / 8060 = 77.75%`。
  - F1 `>= 0.60` 的 candidate row 只有 `194 / 8060 = 2.41%`。
- 但 question-level best candidate 明显更好：
  - best candidate F1 `>= 0.40` 的 question 有 `255 / 610 = 41.80%`。
  - best candidate F1 `>= 0.80` 的 question 有 `87 / 610 = 14.26%`。
- 当前 selector 已经把 best candidate 放在 rank 1 的比例只有 `18.52%`。
- 但 best candidate 在 rank 1-10 的比例是：
  `113 + 81 + 68 + 91 + 152 = 505 / 610 = 82.79%`
- 这说明 Stage 31 的 candidate pool 对 reranker 是有意义的：
  - 大多数 case 的 best candidate 没有离候选前列太远；
  - 但当前 top1 排序明显不够。
- dev 和 train 方向一致：
  - dev oracle gain 更高，gold-document candidate rate 也更高；
  - train 仍有 `+0.1861` 的平均 oracle gain，不是只有 dev 有空间。
- route-level 看，`other` 和 `error_or_log` 是高样本、高 gain 的主要空间。
- 少样本 route 的 gain 不能过度解释，例如：
  - `security_bulletin_post_fix_behavior` 只有 3 个 question；
  - `security_bulletin_remediation` 只有 2 个 question；
  - `security_bulletin_affected_product` 只有 1 个 question。

### 问题与原因

- 问题 1：候选 row 的正样本很稀疏。
  - 高 F1 candidate 很少，后续训练 baseline reranker 需要按 question 分组评估，而不能只看 row-level accuracy。
- 问题 2：best candidate 大多不在 rank 1。
  - 这说明当前 selector 的排序信号不足，但 candidate pool 中仍存在可学习空间。
- 问题 3：route 之间样本量差异很大。
  - 大 route 可以作为主要训练信号，小 route 只能作为诊断维度，不能单独调参过拟合。
- 问题 4：第一次 leakage 审计规则过严。
  - 初始规则把 `candidate_sentence_count`、`title_query_overlap_count`、`title_query_overlap_ratio` 这类聚合数值特征误判为 text-like runtime 风险。
  - 原因是规则粗暴匹配了 `sentence` 和 `title` 字符串，没有区分 raw text/id 字段和派生聚合特征。

### 修正与处理

- 修正 leakage 规则：
  - 只把明显的 raw text/id 字段视为 text-like 风险，例如 `candidate_sentence`、`document_title`、`document_id`、`question_title`、`question_id` 等。
  - 不把 `*_count`、`*_ratio` 这类聚合数值特征误判为原始文本泄漏。
- 重跑 Stage 32 audit。
- 最终真实结果是：
  - `label_leakage_detected_from_keys: false`
  - consistency 全部通过。

### 测试

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_dataset_audit.py `
  scripts\audit_candidate_reranker_dataset.py `
  tests\test_candidate_reranker_dataset_audit.py

pytest -q tests\test_candidate_reranker_dataset_audit.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 82 passed
```

### 结论

- Stage 32 审计通过，可以进入 baseline reranker 实验。
- 当前 dataset/summary 一致，没有发现明显 runtime feature label leakage。
- 可视化结果已经生成，能直观看到：
  - candidate labels 严重偏低；
  - best candidate rank 大量落在 rank 2-10；
  - route oracle gain 主要集中在 `other`、`error_or_log`、`install_upgrade_config` 等较大 route；
  - dev/train 都存在正向 reranking 空间。
- 仍然不能把这个阶段解释为模型收益：
  - 本阶段没有训练模型；
  - 没有 end-to-end reranker 结果；
  - 只有 candidate pool 的 oracle 上限与数据质量审计。

### 我学到的

- 做 learned reranker 前，最重要的不是马上训练，而是确认数据边界和评估单位。
- row-level label 很偏时，row accuracy 很容易误导；更应该按 question 评估“最终选中的 candidate F1 是否提升”。
- 静态 leakage 审计也要小心误报：raw text/id 和派生数值特征必须区分。
- 可视化很有帮助，rank 分布图比单个 oracle gain 数字更能说明为什么 reranker 有必要。

### 下一步

- 做 Stage 33：cross-validated baseline candidate reranker。
- 建议路线：
  1. 以 question 为 group 做 deterministic k-fold cross-validation；
  2. 只使用 `runtime_features`；
  3. label 先使用 `is_best_candidate_for_question` 或 `candidate_token_f1` 派生目标；
  4. 指标按 question 汇总：
     - selected candidate average token F1；
     - delta vs original top candidate；
     - selected rank distribution；
     - route-level delta；
     - dev/train 或 fold-level stability。
- Stage 33 仍先做离线 baseline，不改 runtime。

## Stage 33 - Cross-Validated Baseline Candidate Reranker

### 目标

- 接着 Stage 32 的 dataset audit，训练前先做一个可复跑的离线 baseline reranker。
- 本阶段只做 candidate-level reranking CV，不改 runtime，不接入 end-to-end RAG。
- Stage 32 的下一步里有两个可选 target：
  - `is_best_candidate_for_question`
  - `candidate_token_f1`
- 为了不擅自押一个方向，本阶段同时比较两个 baseline：
  1. `logistic_best_candidate`
     - sklearn `LogisticRegression`
     - target: `is_best_candidate_for_question`
  2. `ridge_candidate_token_f1`
     - sklearn `Ridge`
     - target: `candidate_token_f1`
- 两个模型都只使用 `row.runtime_features`。
- 评估单位按 question 聚合，而不是 row-level accuracy。

### 起始状态

```text
git: main...origin/main clean
scikit-learn: 1.7.2

input:
artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
  rows: 8060
  questions: 610
```

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/candidate_reranker_cv.py`
  - `CandidateRerankerExample`
  - `CandidateRerankerSelection`
  - `CandidateRerankerEvaluationMetrics`
  - `CandidateRerankerFoldResult`
  - `CandidateRerankerSegmentMetrics`
  - `CandidateRerankerModelCVResult`
  - `CandidateRerankerCVResult`
  - `CandidateRerankerScorer`
  - `LogisticBestCandidateScorer`
  - `RidgeTokenF1Scorer`
  - deterministic question-level k-fold CV
  - route/split/fold aggregate metrics
  - SVG visualization output
- 新增 `scripts/cross_validate_candidate_reranker.py`
  - 读取 Stage 31 JSONL。
  - 支持 `--fold-count`。
  - 支持 `--models`。
  - 输出 CV JSON report。
  - 输出 SVG charts。
- 新增 `tests/test_candidate_reranker_cv.py`
  - 验证可学习信号下两个 baseline 都能提升。
  - 验证非法 fold count 会失败。
  - 验证未知 model name 会失败。
  - 验证 SVG 图表输出。

### 特征边界

本阶段模型拟合和打分只使用：

```text
row.runtime_features
```

没有使用：

```text
row.gold_labels
row.metadata
row.candidate_rank
```

说明：

- `gold_labels` 只作为训练 target 和评估 label。
- `metadata` 不进入模型。
- 顶层 `candidate_rank` 不作为模型特征，只用于：
  - 原始 top candidate baseline；
  - tie-break；
  - selected rank 分布统计。

### 命令

```powershell
python scripts\cross_validate_candidate_reranker.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --fold-count 5 `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --output artifacts\candidate_reranker_stage33_cv.json `
  --visualization-dir artifacts\candidate_reranker_stage33_visuals
```

### 结果

Best model：

```text
best_model_name: logistic_best_candidate
selection_metric:
  max aggregate average_delta_vs_top_candidate,
  then oracle_gap_closed_rate,
  then selected_best_candidate_rate
```

Model comparison：

```text
baseline original top candidate average F1: 0.2361
oracle best candidate average F1: 0.4292

logistic_best_candidate:
  target: is_best_candidate_for_question
  selected average F1: 0.2590
  average delta vs top candidate: +0.0229
  oracle gap closed: 11.85%
  selected best candidate rate: 22.30%
  selected gold-document candidate rate: 40.66%
  improved / regressed / tied: 213 / 147 / 250

ridge_candidate_token_f1:
  target: candidate_token_f1
  selected average F1: 0.2558
  average delta vs top candidate: +0.0197
  oracle gap closed: 10.19%
  selected best candidate rate: 20.98%
  selected gold-document candidate rate: 41.31%
  improved / regressed / tied: 182 / 137 / 291
```

`logistic_best_candidate` fold results：

```text
fold 0:
  selected average F1: 0.2532
  delta: +0.0264
  oracle gap closed: 12.84%
  selected best rate: 19.67%

fold 1:
  selected average F1: 0.2493
  delta: +0.0119
  oracle gap closed: 6.21%
  selected best rate: 18.85%

fold 2:
  selected average F1: 0.2624
  delta: +0.0150
  oracle gap closed: 7.93%
  selected best rate: 24.59%

fold 3:
  selected average F1: 0.2585
  delta: +0.0356
  oracle gap closed: 17.84%
  selected best rate: 21.31%

fold 4:
  selected average F1: 0.2714
  delta: +0.0255
  oracle gap closed: 14.23%
  selected best rate: 27.05%
```

`logistic_best_candidate` selected rank distribution：

```text
rank_1: 241
rank_2: 108
rank_3: 79
rank_4_5: 65
rank_6_10: 71
rank_11_plus: 46
```

Split metrics for `logistic_best_candidate`：

```text
train:
  question_count: 450
  baseline average F1: 0.2289
  selected average F1: 0.2551
  delta: +0.0262
  oracle gap closed: 14.08%
  selected best rate: 21.33%

dev:
  question_count: 160
  baseline average F1: 0.2565
  selected average F1: 0.2700
  delta: +0.0135
  oracle gap closed: 6.35%
  selected best rate: 25.00%
```

Route metrics for `logistic_best_candidate`：

```text
security_bulletin_post_fix_behavior:
  n: 3
  delta: +0.0807
  oracle gap closed: 23.68%
  selected best rate: 33.33%

error_or_log:
  n: 119
  delta: +0.0602
  oracle gap closed: 27.78%
  selected best rate: 26.05%

install_upgrade_config:
  n: 91
  delta: +0.0368
  oracle gap closed: 20.05%
  selected best rate: 24.18%

security_bulletin_vulnerability_detail:
  n: 87
  delta: +0.0235
  oracle gap closed: 18.94%
  selected best rate: 31.03%

other:
  n: 241
  delta: +0.0092
  oracle gap closed: 4.20%
  selected best rate: 15.77%

security_bulletin_remediation:
  n: 2
  delta: +0.0000
  oracle gap closed: 0.00%
  selected best rate: 0.00%

limitation_or_restriction:
  n: 11
  delta: -0.0011
  oracle gap closed: -0.72%
  selected best rate: 36.36%

how_to_or_lookup:
  n: 55
  delta: -0.0092
  oracle gap closed: -5.66%
  selected best rate: 23.64%

security_bulletin_affected_product:
  n: 1
  delta: -0.5334
  oracle gap closed: 0.00%
  selected best rate: 0.00%
```

Visualization artifacts：

```text
artifacts/candidate_reranker_stage33_visuals/candidate_reranker_model_delta.svg
artifacts/candidate_reranker_stage33_visuals/candidate_reranker_model_gap_closed.svg
artifacts/candidate_reranker_stage33_visuals/candidate_reranker_best_model_route_delta.svg
artifacts/candidate_reranker_stage33_visuals/candidate_reranker_best_model_selected_rank.svg
```

CV JSON：

```text
artifacts/candidate_reranker_stage33_cv.json
```

上述 artifact 都是本地生成结果，没有纳入 git。

### 解释

- Stage 33 证明：只用 Stage 31 的 `runtime_features`，确实能学到一点 reranking 信号。
- `logistic_best_candidate` 比原始 top candidate 提高：

```text
0.2590 - 0.2361 = +0.0229
```

- 但它只关闭了 Stage 31 oracle gap 的 `11.85%`：

```text
oracle gap: 0.4292 - 0.2361 = +0.1931
model gain: +0.0229
gap closed: 11.85%
```

- 所以这是一个正向 baseline，但不是强 reranker。
- `ridge_candidate_token_f1` 也提升了，但低于 logistic：

```text
0.2558 - 0.2361 = +0.0197
```

- `logistic_best_candidate` 的 fold delta 全部为正：
  - 最低 fold: `+0.0119`
  - 最高 fold: `+0.0356`
  - 说明正向提升不是只来自单个 fold。
- 但 route-level 有明显分化：
  - `error_or_log`、`install_upgrade_config`、`security_bulletin_vulnerability_detail` 是主要正收益 route；
  - `how_to_or_lookup` 出现负收益；
  - 极小样本 route 不能过度解释。
- selected rank 分布显示模型不是只保守选择 rank 1：
  - rank 1 选择 241 个；
  - rank 2-10 选择 323 个；
  - rank 11+ 选择 46 个。
- 这说明模型确实在重排候选，但重排仍会带来 147 个 F1 regression。

### 问题与原因

- 问题 1：baseline 有提升，但离 oracle 很远。
  - 当前 runtime features 仍是浅层手写特征，缺少语义匹配能力。
- 问题 2：regression 数量不少。
  - `logistic_best_candidate` 有 147 个 regressed question。
  - 说明直接替换 top candidate 还不适合作为默认 runtime 行为。
- 问题 3：route-level 不稳定。
  - `how_to_or_lookup` 为负收益。
  - 少样本 security sub-routes 不能单独作为策略依据。
- 问题 4：selected gold-document candidate rate 只有 40.66%。
  - 这低于 Stage 31 中 gold-document candidate 出现在候选池的比例。
  - 说明模型没有充分学会把 gold-document candidate 选出来。

### 修正与处理

- 不接入 runtime。
- 不把 `logistic_best_candidate` 作为默认策略。
- 保留 CV 工具，作为后续特征和模型改进的离线评估基线。
- 下一阶段应该先分析 regression case 和 feature contribution，而不是急着做端到端接入。

### 测试

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_cv.py `
  scripts\cross_validate_candidate_reranker.py `
  tests\test_candidate_reranker_cv.py

pytest -q tests\test_candidate_reranker_cv.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 85 passed
```

### 结论

- Stage 33 成功建立了 cross-validated baseline candidate reranker。
- 最佳 baseline 是 `logistic_best_candidate`：

```text
average F1: 0.2361 -> 0.2590
delta: +0.0229
oracle gap closed: 11.85%
```

- 这说明 learned reranker 方向成立，但当前 baseline 不够稳，不能进入 runtime。
- 下一步应从“为什么 regression”入手，而不是继续盲目换模型。

### 我学到的

- 一个 baseline 只要能在 grouped CV 下稳定正收益，就能证明方向不是空想。
- 但正收益不等于可上线：regression count 和 route-level negative cases 同样重要。
- `is_best_candidate_for_question` 这个分类目标在当前数据上略优于直接回归 `candidate_token_f1`。
- question-level CV 比 row-level 训练指标更贴近最终使用方式。

### 下一步

- 做 Stage 34：candidate-reranker regression/error analysis。
- 重点分析：
  1. `logistic_best_candidate` 的 improved/regressed/tied case；
  2. route-level regression 来源，尤其 `how_to_or_lookup`；
  3. selected rank 太深的 case；
  4. gold-document candidate 存在但没选中的 case；
  5. 当前 runtime features 是否缺少关键语义信号。
- Stage 34 仍然只做离线分析，不改 runtime。

## Stage 34 - Candidate Reranker Regression/Error Analysis

### 目标

- 接着 Stage 33 的结论，分析 `logistic_best_candidate` 为什么虽然平均提升，但仍有不少 regression。
- 本阶段只做离线 grouped-CV error analysis，不改 runtime。
- 分析重点：
  1. improved / regressed / tied case 分布；
  2. route-level regression 来源，尤其 `how_to_or_lookup`；
  3. selected rank 太深的 case；
  4. gold-document candidate 存在但没选中的 case；
  5. improved 与 regressed case 的 runtime feature 差异。

### 起始状态

```text
git: main...origin/main clean

Stage 33 best model:
logistic_best_candidate

Stage 33 headline:
baseline top candidate F1: 0.2361
selected average F1: 0.2590
delta: +0.0229
oracle gap closed: 11.85%
improved / regressed / tied: 213 / 147 / 250
```

### 本阶段新增内容

- 更新 `src/ts_rag_agent/application/candidate_reranker_cv.py`
  - 新增 `cross_validated_candidate_reranker_selections`
  - 用于重新跑同样的 deterministic grouped CV，并保留每个 validation question 的 selected candidate。
- 新增 `src/ts_rag_agent/application/candidate_reranker_error_analysis.py`
  - `CandidateSnapshot`
  - `CandidateRerankerErrorCase`
  - `ErrorOutcomeSummary`
  - `SegmentErrorSummary`
  - `FeatureContrastSummary`
  - `CandidateRerankerErrorAnalysisResult`
  - 支持 outcome、route、split、rank、gold miss、feature contrast 和 sample buckets 分析。
- 新增 `scripts/analyze_candidate_reranker_errors.py`
  - 读取 Stage 31 JSONL。
  - 重新按 Stage 33 的 grouped CV 方式训练/验证。
  - 输出 error-analysis JSON。
- 新增 `tests/test_candidate_reranker_error_analysis.py`
  - 验证 grouped-CV case analysis。
  - 验证 unknown model 明确失败。

### 命令

```powershell
python scripts\analyze_candidate_reranker_errors.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --fold-count 5 `
  --sample-limit 10 `
  --output artifacts\candidate_reranker_stage34_error_analysis.json
```

### 结果

Overall summary：

```text
question_count: 610
improved_count: 213
regressed_count: 147
tied_count: 250

improved_rate: 34.92%
regressed_rate: 24.10%
tied_rate: 40.98%

average_delta_vs_top_candidate: +0.0229
average_improvement_delta: +0.1819
average_regression_delta: -0.1686

selected_missed_gold_document_count: 130
selected_missed_gold_document_rate: 21.31%

selected_missed_oracle_best_count: 474
selected_missed_oracle_best_rate: 77.70%

selected_deep_rank_count: 117
selected_deep_rank_rate: 19.18%
```

Route-level regression：

```text
security_bulletin_affected_product:
  n: 1
  improved / regressed / tied: 0 / 1 / 0
  regressed_rate: 100.00%
  average_delta: -0.5334

security_bulletin_post_fix_behavior:
  n: 3
  improved / regressed / tied: 1 / 2 / 0
  regressed_rate: 66.67%
  average_delta: +0.0807
  deep_rank_rate: 100.00%

how_to_or_lookup:
  n: 55
  improved / regressed / tied: 13 / 18 / 24
  regressed_rate: 32.73%
  average_delta: -0.0092
  selected_missed_gold_document_rate: 27.27%
  deep_rank_rate: 23.64%

other:
  n: 241
  improved / regressed / tied: 88 / 67 / 86
  regressed_rate: 27.80%
  average_delta: +0.0092
  selected_missed_gold_document_rate: 23.65%
  deep_rank_rate: 24.48%

error_or_log:
  n: 119
  improved / regressed / tied: 52 / 28 / 39
  regressed_rate: 23.53%
  average_delta: +0.0602
  selected_missed_gold_document_rate: 18.49%
  deep_rank_rate: 22.69%

install_upgrade_config:
  n: 91
  improved / regressed / tied: 37 / 19 / 35
  regressed_rate: 20.88%
  average_delta: +0.0368
  selected_missed_gold_document_rate: 19.78%
  deep_rank_rate: 16.48%

limitation_or_restriction:
  n: 11
  improved / regressed / tied: 2 / 2 / 7
  regressed_rate: 18.18%
  average_delta: -0.0011
  selected_missed_gold_document_rate: 36.36%

security_bulletin_vulnerability_detail:
  n: 87
  improved / regressed / tied: 20 / 10 / 57
  regressed_rate: 11.49%
  average_delta: +0.0235
  selected_missed_gold_document_rate: 14.94%

security_bulletin_remediation:
  n: 2
  improved / regressed / tied: 0 / 0 / 2
  regressed_rate: 0.00%
  average_delta: +0.0000
```

Split-level：

```text
dev:
  n: 160
  improved / regressed / tied: 55 / 44 / 61
  regressed_rate: 27.50%
  average_delta: +0.0135
  selected_missed_gold_document_rate: 26.87%
  deep_rank_rate: 21.88%

train:
  n: 450
  improved / regressed / tied: 158 / 103 / 189
  regressed_rate: 22.89%
  average_delta: +0.0262
  selected_missed_gold_document_rate: 19.33%
  deep_rank_rate: 18.22%
```

Selected-rank regression：

```text
rank_1:
  n: 241
  improved / regressed / tied: 0 / 0 / 241
  average_delta: +0.0000

rank_2:
  n: 108
  improved / regressed / tied: 68 / 37 / 3
  regressed_rate: 34.26%
  average_delta: +0.1006

rank_3:
  n: 79
  improved / regressed / tied: 49 / 27 / 3
  regressed_rate: 34.18%
  average_delta: +0.0916

rank_4_5:
  n: 65
  improved / regressed / tied: 40 / 23 / 2
  regressed_rate: 35.38%
  average_delta: -0.0020

rank_6_10:
  n: 71
  improved / regressed / tied: 36 / 35 / 0
  regressed_rate: 49.30%
  average_delta: -0.0394
  selected_missed_gold_document_rate: 30.99%

rank_11_plus:
  n: 46
  improved / regressed / tied: 20 / 25 / 1
  regressed_rate: 54.35%
  average_delta: -0.0264
  selected_missed_gold_document_rate: 39.13%
```

Feature contrast top signals：

```text
candidate_token_count:
  improved selected-baseline mean: +17.1596
  regressed selected-baseline mean: +37.3605
  regressed_minus_improved: +20.2009

retrieval_score:
  improved selected-baseline mean: +17.9104
  regressed selected-baseline mean: +9.1087
  regressed_minus_improved: -8.8017

candidate_score:
  improved selected-baseline mean: -43.8597
  regressed selected-baseline mean: -50.6369
  regressed_minus_improved: -6.7772

query_overlap_count:
  improved selected-baseline mean: -1.0563
  regressed selected-baseline mean: -0.1973
  regressed_minus_improved: +0.8590

answer_signal_score:
  improved selected-baseline mean: -0.3587
  regressed selected-baseline mean: -0.6912
  regressed_minus_improved: -0.3325

title_query_overlap_count:
  improved selected-baseline mean: +1.0751
  regressed selected-baseline mean: +0.7687
  regressed_minus_improved: -0.3064
```

Largest regression samples：

```text
DEV_Q008:
  route: how_to_or_lookup
  delta: -0.9322
  baseline rank/F1: 1 / 0.9630
  selected rank/F1: 8 / 0.0308
  oracle rank/F1: 1 / 0.9630
  selected_missed_gold_document: true

DEV_Q155:
  route: other
  delta: -0.9209
  baseline rank/F1: 1 / 0.9697
  selected rank/F1: 3 / 0.0488
  oracle rank/F1: 1 / 0.9697
  selected_missed_gold_document: true

TRAIN_Q255:
  route: error_or_log
  delta: -0.9151
  baseline rank/F1: 1 / 0.9655
  selected rank/F1: 9 / 0.0504
  oracle rank/F1: 1 / 0.9655
  selected_missed_gold_document: true
```

Largest improvement samples：

```text
TRAIN_Q415:
  route: error_or_log
  delta: +0.9143
  baseline rank/F1: 1 / 0.0000
  selected rank/F1: 2 / 0.9143
  oracle rank/F1: 2 / 0.9143

TRAIN_Q548:
  route: other
  delta: +0.8960
  baseline rank/F1: 1 / 0.0851
  selected rank/F1: 7 / 0.9811
  oracle rank/F1: 7 / 0.9811

TRAIN_Q188:
  route: how_to_or_lookup
  delta: +0.8620
  baseline rank/F1: 1 / 0.1176
  selected rank/F1: 2 / 0.9796
  oracle rank/F1: 2 / 0.9796
```

Artifact：

```text
artifacts/candidate_reranker_stage34_error_analysis.json
size: 309,040 bytes
```

该 artifact 是本地生成结果，没有纳入 git。

### 解释

- Stage 34 说明 Stage 33 baseline 的主要风险不是“完全不会选更好候选”，而是“有时会跳过已经很好的 top1”。
- 最大退化样例里，baseline rank 1 本来已经接近完美：
  - `DEV_Q008`: top1 F1 `0.9630`
  - `DEV_Q155`: top1 F1 `0.9697`
  - `TRAIN_Q255`: top1 F1 `0.9655`
- 但模型选择了 rank 3、8、9 等更深候选，导致 F1 接近 0。
- selected rank 越深，风险明显越高：
  - rank 2/3 虽然有 regression，但平均仍为正；
  - rank 6-10 和 rank 11+ 的平均 delta 都为负；
  - rank 11+ 的 regressed rate 达到 `54.35%`。
- `how_to_or_lookup` 是 Stage 34 的重点风险 route：
  - 平均 delta 是 `-0.0092`；
  - regressed rate 是 `32.73%`；
  - gold-document miss rate 是 `27.27%`。
- feature contrast 暗示一个风险：
  - regression case 中，selected candidate 相对 baseline 更长；
  - 但 retrieval_score、candidate_score、answer_signal_score 的下降更明显；
  - 当前浅层特征不能可靠判断“较长且排名更深的候选是否真的更接近答案”。
- 130 个 selected_missed_gold_document case 说明：
  - gold document 候选池中存在时，模型仍经常没有选中；
  - 这不是 retrieval 缺失，而是 reranker 选择错误。

### 问题与原因

- 问题 1：模型缺少 top1 保护机制。
  - 当 top1 本来非常好时，模型仍可能跳到深 rank 候选。
- 问题 2：深 rank 选择风险高。
  - rank 6-10 和 rank 11+ 平均收益都为负。
- 问题 3：`how_to_or_lookup` route 不适合直接套用当前 learned reranker。
  - 它在 Stage 33 就是负收益，Stage 34 进一步确认 regression 来源明显。
- 问题 4：当前 runtime features 缺少语义判别能力。
  - 现有特征主要是 overlap、heading、score、长度。
  - 它们不足以稳定区分“更长的候选是否更完整”与“更长的候选是否只是噪声”。

### 修正与处理

- 不接入 runtime。
- 不把 Stage 33 的 `logistic_best_candidate` 直接作为候选替换策略。
- 保留 Stage 34 error-analysis 工具，作为后续策略约束和特征改进的依据。
- 下一阶段不应盲目换模型，而应该先做 constrained reranker policy search：
  - 限制 selected rank；
  - 对 route 做门控；
  - 对 top1 强信号做保护；
  - 再用 grouped CV 验证是否能保留收益并减少 regression。

### 测试

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_cv.py `
  src\ts_rag_agent\application\candidate_reranker_error_analysis.py `
  scripts\analyze_candidate_reranker_errors.py `
  tests\test_candidate_reranker_error_analysis.py

pytest -q tests\test_candidate_reranker_error_analysis.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 87 passed
```

### 结论

- Stage 34 找到了 Stage 33 baseline 不能 runtime 化的核心原因：
  - regression 数量 `147 / 610`；
  - deep-rank selection 风险高；
  - top1 本来很强时仍会被替换；
  - `how_to_or_lookup` route 有负收益；
  - gold-document candidate 存在时仍有 130 个没选中。
- learned reranker 方向仍成立，但需要加约束，不能直接替换 top candidate。

### 我学到的

- 平均 F1 提升会掩盖严重退化个案。
- reranker 的第一版工程化目标不是“尽量多替换”，而是“只在高把握场景替换”。
- top1 保护、rank 上限和 route gate 是下一阶段最重要的安全约束。
- 样例 case 比 aggregate 指标更能解释为什么不能上线。

### 下一步

- 做 Stage 35：constrained candidate-reranker policy search。
- 搜索方向：
  1. max selected rank，例如只允许 rank 2-3 或 rank 2-5；
  2. route gate，例如先排除 `how_to_or_lookup`；
  3. top1 protection，例如 top1 candidate F1 proxy 很强时不替换；
  4. score margin gate，例如模型分数必须明显高于 top1 才替换；
  5. gold-document miss / regression-aware summary。
- Stage 35 仍然只做离线分析，不改 runtime。

## Stage 35 - Constrained Candidate Reranker Policy Search

### 目标

- 接着 Stage 34 的 regression/error analysis，搜索一个离线 constrained policy。
- 本阶段只做分析，不改 runtime。
- 核心问题：
  - Stage 33 的 unconstrained `logistic_best_candidate` 平均 F1 有提升；
  - 但 Stage 34 发现它会产生 147 个 regression，并且 deep-rank selection 风险高。
- 本阶段要验证：
  1. 限制 selected rank 是否能减少 regression；
  2. route gate 是否能缓解 `how_to_or_lookup` 的负收益；
  3. model score margin 是否能过滤低把握替换；
  4. top1 candidate score protection 是否有帮助；
  5. 是否存在“比 unconstrained 更高收益、更低 regression”的离线策略。

### 起始状态

```text
git: main...origin/main clean

Stage 33 unconstrained model:
logistic_best_candidate

Stage 34 key risk:
improved / regressed / tied: 213 / 147 / 250
average delta: +0.0229
selected deep rank: 117 / 610 = 19.18%
how_to_or_lookup average delta: -0.0092
```

### 本阶段新增内容

- 更新 `src/ts_rag_agent/application/candidate_reranker_cv.py`
  - `CandidateRerankerSelection` 增加：
    - `baseline_model_score`
    - `score_margin_vs_top_candidate`
    - `baseline_is_gold_document`
    - `baseline_is_oracle_best_f1`
- 新增 `src/ts_rag_agent/application/candidate_reranker_policy_search.py`
  - `CandidateRerankerPolicyConfig`
  - `CandidateRerankerPolicyDecision`
  - `CandidateRerankerPolicyMetrics`
  - `CandidateRerankerPolicyMetricsBySegment`
  - `CandidateRerankerPolicyEvaluation`
  - `CandidateRerankerPolicySearchResult`
  - 支持 rank / route / score margin / top1 score protection 的离线搜索。
- 新增 `scripts/search_candidate_reranker_policy.py`
  - 读取 Stage 31 JSONL。
  - 重新跑 Stage 33 grouped CV selections。
  - 搜索 constrained policy grid。
  - 输出 JSON report。
- 新增 `tests/test_candidate_reranker_policy_search.py`
  - 验证 rank constraint 能找到正收益 policy。
  - 验证非法搜索 grid 会明确失败。

### 搜索空间

```text
model: logistic_best_candidate
fold_count: 5
policy_count: 500

max_selected_rank_grid:
  2, 3, 5, 10, 25

min_score_margin_grid:
  0.0, 0.05, 0.1, 0.2, 0.3

protect_top1_candidate_score_min_grid:
  none, 90.0, 110.0, 140.0, 170.0

blocked_route_sets:
  []
  [how_to_or_lookup]
  [how_to_or_lookup, limitation_or_restriction]
  [how_to_or_lookup, security_bulletin_affected_product]
```

说明：

- `protect_top1_candidate_score_min_grid` 使用 Stage 31 top1 candidate score 的真实分位附近值：
  - P25 约 `88.67`
  - P50 约 `110.28`
  - P75 约 `136.54`
  - P90 约 `170.15`
- `min_score_margin_grid` 使用 Stage 33 replacement candidates 的真实 model score margin 分布附近值：
  - P25 约 `0.064`
  - P50 约 `0.139`
  - P75 约 `0.227`
  - P90 约 `0.309`

### 命令

```powershell
python scripts\search_candidate_reranker_policy.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --fold-count 5 `
  --output artifacts\candidate_reranker_stage35_policy_search.json
```

### 结果

Unconstrained model：

```text
policy_average_token_f1: 0.2590
average_delta_vs_top_candidate: +0.0229
oracle_gap_closed_rate: 11.85%
replacement_count: 369
replacement_rate: 60.49%
improved / regressed / tied: 213 / 147 / 250
regressed_rate: 24.10%
final_missed_gold_document_count: 130
final_deep_rank_count: 117
```

Best average-delta policy：

```text
name:
rank_lte_5__margin_gte_0.05__top1_score_protect_none__blocked_how_to_or_lookup+security_bulletin_affected_product

constraints:
  max_selected_rank: 5
  min_score_margin_vs_top_candidate: 0.05
  blocked_routes:
    - how_to_or_lookup
    - security_bulletin_affected_product
  protect_top1_candidate_score_min: none

policy_average_token_f1: 0.2708
average_delta_vs_top_candidate: +0.0347
oracle_gap_closed_rate: 17.97%
replacement_count: 188
replacement_rate: 30.82%
improved / regressed / tied: 126 / 55 / 429
regressed_rate: 9.02%
regression_reduction_vs_unconstrained: 92
final_missed_gold_document_count: 130
final_deep_rank_count: 0
final_oracle_best_count: 138
```

对比 unconstrained：

```text
average F1:
  unconstrained: 0.2590
  constrained:   0.2708
  delta:         +0.0118

delta vs top candidate:
  unconstrained: +0.0229
  constrained:   +0.0347

regressed_count:
  unconstrained: 147
  constrained:   55
  reduction:      92

deep-rank selections:
  unconstrained: 117
  constrained:   0
```

Decision reason counts for best policy：

```text
accepted: 188
model_selected_top_candidate: 241
route_blocked: 32
score_margin_below_min: 74
selected_rank_exceeds_limit: 117
```

Best policy route metrics：

```text
error_or_log:
  n: 119
  replacements: 47
  average_delta: +0.0708
  regressed_count: 10
  regressed_rate: 8.40%

install_upgrade_config:
  n: 91
  replacements: 33
  average_delta: +0.0404
  regressed_count: 8
  regressed_rate: 8.79%

security_bulletin_vulnerability_detail:
  n: 87
  replacements: 17
  average_delta: +0.0290
  regressed_count: 5
  regressed_rate: 5.75%

other:
  n: 241
  replacements: 88
  average_delta: +0.0268
  regressed_count: 31
  regressed_rate: 12.86%

limitation_or_restriction:
  n: 11
  replacements: 3
  average_delta: +0.0068
  regressed_count: 1
  regressed_rate: 9.09%

how_to_or_lookup:
  n: 55
  replacements: 0
  average_delta: +0.0000
  regressed_count: 0

security_bulletin_affected_product:
  n: 1
  replacements: 0
  average_delta: +0.0000
  regressed_count: 0
```

Best policy selected-rank metrics：

```text
rank_2:
  n: 80
  average_delta: +0.1317
  regressed_count: 26
  regressed_rate: 32.50%

rank_3:
  n: 58
  average_delta: +0.1465
  regressed_count: 14
  regressed_rate: 24.14%

rank_4_5:
  n: 50
  average_delta: +0.0425
  regressed_count: 15
  regressed_rate: 30.00%

rank_1:
  n: 422
  replacements: 0
  average_delta: +0.0000
```

Best regression-reduction policy：

```text
name:
rank_lte_2__margin_gte_0.3__top1_score_protect_140__blocked_how_to_or_lookup

policy_average_token_f1: 0.2388
average_delta_vs_top_candidate: +0.0027
oracle_gap_closed_rate: 1.40%
replacement_count: 5
regressed_count: 0
regression_reduction_vs_unconstrained: 147
final_missed_gold_document_count: 183
```

解释：

- 这个 policy 虽然 0 regression，但只替换 5 个 question。
- 平均收益只有 `+0.0027`。
- final missed gold-document count 从 130 增加到 183。
- 因此它是“过度保守”的诊断点，不适合作为主要方向。

Top policies 摘要：

```text
rank<=5, margin>=0.05, block how_to + affected_product:
  delta: +0.0347, replacements: 188, regressions: 55

rank<=5, margin>=0.05, block how_to:
  delta: +0.0338, replacements: 189, regressions: 56

rank<=5, margin>=0.05, block how_to + limitation:
  delta: +0.0337, replacements: 186, regressions: 55

rank<=5, margin>=0.05, block none:
  delta: +0.0331, replacements: 206, regressions: 67

rank<=3, margin>=0.05, block how_to:
  delta: +0.0312, replacements: 138, regressions: 40
```

Artifact：

```text
artifacts/candidate_reranker_stage35_policy_search.json
size: 186,808 bytes
```

该 artifact 是本地生成结果，没有纳入 git。

### 解释

- Stage 35 找到了比 unconstrained reranker 更好的 constrained policy。
- 最优平均收益 policy 同时做到：
  - 平均 F1 更高；
  - regression 更少；
  - deep-rank selection 清零。
- 关键有效约束是：
  1. `max_selected_rank <= 5`
  2. `score_margin >= 0.05`
  3. block `how_to_or_lookup`
  4. block `security_bulletin_affected_product`
- `top1_candidate_score_protect` 没有进入最优平均收益 policy。
- 这说明 Stage 34 的风险主要不是“top1 score 高就保护”，而是：
  - 深 rank 替换风险高；
  - `how_to_or_lookup` route 对当前 reranker 不友好；
  - 低 score margin 的替换把噪声带进来了。
- 但这个 policy 仍然不能直接 runtime 化：
  - 它仍有 55 个 regression；
  - final missed gold-document count 没有下降；
  - 还没有 end-to-end RAG 评估；
  - 当前只是 candidate-level F1。

### 问题与原因

- 问题 1：约束后仍有 55 个 regression。
  - rank 2-5 的替换仍有风险。
- 问题 2：gold-document miss 没有改善。
  - constrained policy 的 final missed gold-document count 仍是 130。
  - 说明当前约束主要减少坏替换，不是更会找 gold document。
- 问题 3：`security_bulletin_affected_product` 只有 1 个样本。
  - 它进入最优 policy 可能带有样本偶然性。
  - `block how_to` 的 policy 与最佳 policy 只差 `0.0009` average delta。
- 问题 4：过度保守 policy 没有工程价值。
  - 0 regression policy 只替换 5 个 case，平均收益太小。

### 修正与处理

- 不接入 runtime。
- 不把 Stage 35 best policy 改成默认策略。
- 保留 constrained policy search 作为离线工具。
- 后续需要做更稳健的验证：
  - 对最佳 policy 做 fold-level / route-level stability 分析；
  - 对是否 block `security_bulletin_affected_product` 做样本量敏感性检查；
  - 再考虑 end-to-end answer 级实验。

### 测试

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_cv.py `
  src\ts_rag_agent\application\candidate_reranker_policy_search.py `
  scripts\search_candidate_reranker_policy.py `
  tests\test_candidate_reranker_policy_search.py

pytest -q tests\test_candidate_reranker_policy_search.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 89 passed
```

### 结论

- Stage 35 成功找到一个离线 constrained policy：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
block security_bulletin_affected_product
```

- 它比 unconstrained baseline 更好：

```text
unconstrained F1: 0.2590
constrained F1:   0.2708

unconstrained delta: +0.0229
constrained delta:   +0.0347

unconstrained regressions: 147
constrained regressions:   55
```

- 这个结果支持继续向 “guarded reranker” 方向推进。
- 但它仍然只是 candidate-level 离线结果，不是 runtime 或 end-to-end 结论。

### 我学到的

- 一个弱 reranker 不一定要直接丢掉；加约束后可能更有价值。
- rank 上限和 margin gate 比单纯 top1 score protection 更有效。
- route gate 要谨慎：大样本负收益 route 可以先 block，小样本 route 需要做敏感性检查。
- 0 regression 不等于最好，太保守会让收益几乎消失。

### 下一步

- 做 Stage 36：constrained policy stability analysis。
- 重点：
  1. 比较最佳 policy 与 `block how_to only` policy 的差异；
  2. 看 fold-level stability；
  3. 看 route-level regression 是否稳定下降；
  4. 检查 `security_bulletin_affected_product` 这种 1-sample route 是否应从策略中移除；
  5. 决定是否进入 end-to-end answer-level 实验。
- Stage 36 仍然只做离线分析，不改 runtime。

## Stage 36 - Constrained Policy Stability Analysis

### 目标

- 接着 Stage 35 的 constrained policy search，检查最佳策略是否稳定。
- 本阶段只做离线 stability analysis，不改 runtime。
- 重点比较：
  1. Stage 35 best policy：
     - rank <= 5
     - score margin >= 0.05
     - block `how_to_or_lookup`
     - block `security_bulletin_affected_product`
  2. 更简单的 challenger policy：
     - rank <= 5
     - score margin >= 0.05
     - block `how_to_or_lookup`
- 核心问题：
  - `security_bulletin_affected_product` 只有 1 个样本；
  - 它进入 Stage 35 best policy 是否只是样本偶然性；
  - 下一步是否应该用更简单、更稳定的 policy 进入后续验证。

### 起始状态

```text
git: main...origin/main clean

Stage 35 best policy:
rank <= 5
score margin >= 0.05
block how_to_or_lookup
block security_bulletin_affected_product

Stage 35 best result:
F1: 0.2708
delta: +0.0347
regressions: 55
replacements: 188
deep-rank selections: 0
```

### 本阶段新增内容

- 更新 `src/ts_rag_agent/application/candidate_reranker_policy_search.py`
  - 新增 `evaluate_candidate_reranker_policy_from_selections`
  - 新增 `candidate_reranker_policy_decisions_from_selections`
  - 新增 `summarize_candidate_reranker_policy_decisions`
  - 目的是让 Stage 36 不复制 Stage 35 的 policy decision 逻辑。
- 新增 `src/ts_rag_agent/application/candidate_reranker_policy_stability.py`
  - `CandidateRerankerPolicyDelta`
  - `CandidateRerankerPolicyFoldStability`
  - `CandidateRerankerPolicyRouteComparison`
  - `CandidateRerankerPolicyStabilityResult`
  - 固定比较 Stage 35 best policy 和 challenger policy。
- 新增 `scripts/analyze_candidate_reranker_policy_stability.py`
  - 读取 Stage 31 JSONL。
  - 重新跑 `logistic_best_candidate` grouped-CV selections。
  - 输出 primary/challenger 的 aggregate、fold-level、route-level stability JSON。
- 新增 `tests/test_candidate_reranker_policy_stability.py`
  - 验证 fixed policy comparison。
  - 验证 fold metrics 和 route comparisons。
  - 验证结果可序列化。

### 命令

```powershell
python scripts\analyze_candidate_reranker_policy_stability.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --fold-count 5 `
  --output artifacts\candidate_reranker_stage36_policy_stability.json
```

### 结果

Primary policy：

```text
name:
rank_lte_5__margin_gte_0.05__top1_score_protect_none__blocked_how_to_or_lookup+security_bulletin_affected_product

policy_average_token_f1: 0.2708
average_delta_vs_top_candidate: +0.0347
oracle_gap_closed_rate: 17.97%
replacement_count: 188
regressed_count: 55
final_missed_gold_document_count: 130
final_deep_rank_count: 0
```

Challenger policy：

```text
name:
rank_lte_5__margin_gte_0.05__top1_score_protect_none__blocked_how_to_or_lookup

policy_average_token_f1: 0.2699
average_delta_vs_top_candidate: +0.0338
oracle_gap_closed_rate: 17.51%
replacement_count: 189
regressed_count: 56
final_missed_gold_document_count: 130
final_deep_rank_count: 0
```

Primary vs challenger：

```text
average_delta_difference: +0.0009
policy_average_f1_difference: +0.0009
oracle_gap_closed_difference: +0.0046
replacement_count_difference: -1
regressed_count_difference: -1
final_missed_gold_document_count_difference: 0
final_deep_rank_count_difference: 0
```

Fold-level metrics：

```text
challenger block_how_to_only:
  fold 0: delta +0.0218, F1 0.2487, replacements 33, regressions 9
  fold 1: delta +0.0254, F1 0.2627, replacements 37, regressions 13
  fold 2: delta +0.0358, F1 0.2831, replacements 43, regressions 9
  fold 3: delta +0.0470, F1 0.2699, replacements 45, regressions 15
  fold 4: delta +0.0391, F1 0.2851, replacements 31, regressions 10

primary block_how_to_and_affected_product:
  fold 0: delta +0.0218, F1 0.2487, replacements 33, regressions 9
  fold 1: delta +0.0254, F1 0.2627, replacements 37, regressions 13
  fold 2: delta +0.0358, F1 0.2831, replacements 43, regressions 9
  fold 3: delta +0.0470, F1 0.2699, replacements 45, regressions 15
  fold 4: delta +0.0435, F1 0.2894, replacements 30, regressions 9
```

解释：

- 两个 policy 在 fold 0-3 完全相同。
- 差异只出现在 fold 4。
- primary 比 challenger 少替换 1 个 case，少 1 个 regression，delta 多 `+0.0044` in fold 4。

Route-level comparison：

```text
security_bulletin_affected_product:
  question_count: 1
  primary average_delta: +0.0000
  challenger average_delta: -0.5334
  average_delta_difference: +0.5334
  primary replacements/regressions: 0 / 0
  challenger replacements/regressions: 1 / 1

all other routes:
  average_delta_difference: +0.0000
  replacement_count_difference: 0
  regressed_count_difference: 0
```

Findings：

```text
1. Primary policy improves average delta by at most 0.001 over the simpler challenger.
2. Primary policy reduces regression count by 1 versus the simpler challenger.
3. security_bulletin_affected_product has only one question.
4. Both policies eliminate deep-rank selections.
```

Artifact：

```text
artifacts/candidate_reranker_stage36_policy_stability.json
size: 33,162 bytes
```

该 artifact 是本地生成结果，没有纳入 git。

### 解释

- Stage 36 说明 Stage 35 best policy 的额外优势完全来自 `security_bulletin_affected_product` 这个 1-sample route。
- primary 比 challenger 只多：
  - `+0.0009` average delta；
  - 少 1 个 regression；
  - 少 1 个 replacement。
- 这不是足够稳定的策略证据。
- `block how_to_or_lookup` 的主体约束是稳定的：
  - 两个 policy 都 block 它；
  - 两者都把 deep-rank selections 清零；
  - 两者 fold-level delta 全部为正。
- `security_bulletin_affected_product` 应该被视为样本量敏感项，不应在下一步主策略中写死。

### 问题与原因

- 问题 1：Stage 35 的 best policy 带有 1-sample route 敏感性。
  - `security_bulletin_affected_product` 只有 1 个 question。
  - block 它带来的收益是单个 case 贡献。
- 问题 2：primary 与 challenger 的整体差异太小。
  - +0.0009 average delta 不足以证明额外 route gate 稳定。
- 问题 3：gold-document miss 没有改善。
  - 两个 policy 的 final_missed_gold_document_count 都是 130。
- 问题 4：仍不能 runtime 化。
  - 两个 policy 仍有 55/56 个 regression。
  - 当前仍是 candidate-level F1，不是 answer-level end-to-end。

### 修正与处理

- 不接入 runtime。
- 不把 `security_bulletin_affected_product` block 写入下一步主策略。
- 下一步主候选改为更简单的 challenger：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
```

- Stage 35 best policy 可作为 sensitivity 对照保留，但不作为主策略。

### 测试

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_policy_search.py `
  src\ts_rag_agent\application\candidate_reranker_policy_stability.py `
  scripts\analyze_candidate_reranker_policy_stability.py `
  tests\test_candidate_reranker_policy_stability.py

pytest -q tests\test_candidate_reranker_policy_stability.py
```

结果：

```text
ruff: passed
pytest: 1 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 90 passed
```

### 结论

- Stage 36 结论：
  - Stage 35 best policy 的额外 `security_bulletin_affected_product` block 不够稳定；
  - `block how_to_or_lookup` 是更稳妥的主策略；
  - rank <= 5 和 margin >= 0.05 仍然保留；
  - 当前结果支持进入下一步 answer-level 离线实验，但仍不能改 runtime。

推荐下一步主候选：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
```

对照候选：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
block security_bulletin_affected_product
```

### 我学到的

- 网格搜索的 best policy 不一定是最稳的 policy。
- 当收益来自 1 个样本 route 时，应该优先选择更简单策略。
- fold-level stability 能揭示“只有某一 fold 有差异”的情况。
- 候选级策略进入 answer-level 之前，必须先把这种样本量敏感性拆出来。

### 下一步

- 做 Stage 37：answer-level offline guarded reranker experiment。
- 目标：
  1. 使用 Stage 36 主候选 policy；
  2. 以 Stage 35 best policy 作为 sensitivity 对照；
  3. 在 answer-level / verified RAG pipeline 上离线评估；
  4. 比较 baseline top candidate answer 与 guarded reranker answer；
  5. 重点看 average answer token F1、citation/gold-document、regression count。
- Stage 37 仍先做可选离线实验，不改默认 runtime。

## Stage 37 - Guarded Candidate Reranker Answer-Level Proxy Experiment

### 目标

- 接着 Stage 36，把 guarded candidate reranker 放到 answer-level proxy 上看效果。
- 本阶段仍然是离线实验，不接入默认 runtime，也不修改 verified RAG 默认策略。
- 主策略使用 Stage 36 结论中更稳妥的配置：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
```

- sensitivity 对照继续保留 Stage 35 best policy：

```text
rank <= 5
score margin >= 0.05
block how_to_or_lookup
block security_bulletin_affected_product
```

### 边界说明

- 本阶段不是完整 runtime end-to-end verified RAG 实验。
- 本阶段有两个 answer-level proxy mode：
  1. `single_candidate_answer`
     - 使用 Stage 31 candidate dataset 中已经计算好的 `candidate_token_f1` 标签；
     - 对比原 top candidate answer 和 guarded reranker 选出的 single candidate answer。
  2. `top3_leading_candidate_rewrite`
     - 使用本地 PrimeQA gold answer；
     - 用 candidate dataset metadata 中保存的候选句文本重算 top3 answer token F1；
     - 做法是把 guarded reranker 选中的候选放到 top3 首位，再补原 top candidates；
     - 这是 metadata sentence proxy，不等同于真实 verified RAG runtime 结果。

### 起始状态

```text
git: main...origin/main clean
latest committed stage: Stage 36
Stage 36 main candidate:
rank <= 5
score margin >= 0.05
block how_to_or_lookup
```

Stage 36 结论是：

- Stage 35 best policy 多出来的收益来自 `security_bulletin_affected_product` 这个 1-sample route；
- 因此 Stage 37 主策略不把这个 route 写死为 block；
- 它只作为 sensitivity 对照保留。

### 本阶段新增内容

- 新增 `src/ts_rag_agent/application/guarded_candidate_reranker_answer_experiment.py`
  - 复用 grouped-CV candidate reranker selections；
  - 复用 constrained policy decision；
  - 新增 single-candidate answer proxy；
  - 新增 top-k leading-candidate rewrite proxy；
  - 输出 aggregate、route、split、sample cases；
  - 输出 SVG 可视化。
- 新增 `scripts/evaluate_guarded_candidate_reranker_answers.py`
  - 读取 Stage 31 JSONL candidate dataset；
  - 读取本地 PrimeQA dev/train gold answers；
  - 输出 Stage 37 JSON artifact；
  - 输出 Stage 37 SVG visualization directory。
- 新增 `tests/test_guarded_candidate_reranker_answer_experiment.py`
  - 验证两个 proxy mode；
  - 验证结果可序列化；
  - 验证 SVG 可视化真实写出。

### 命令

```powershell
python scripts\evaluate_guarded_candidate_reranker_answers.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --fold-count 5 `
  --splits dev,train `
  --max-answer-candidates 3 `
  --sample-limit 20 `
  --output artifacts\candidate_reranker_stage37_answer_experiment.json `
  --visualization-dir artifacts\candidate_reranker_stage37_answer_visuals
```

### 结果

Main policy, `single_candidate_answer`：

```text
baseline average answer token F1: 0.2361
policy average answer token F1:   0.2699
delta:                            +0.0338
oracle gap closed:                17.51%
replacement count:                189 / 610
regressed count:                  56 / 610
gold-document citation count:      192 -> 248
citation delta:                   +56
```

Main policy, `top3_leading_candidate_rewrite`：

```text
baseline average answer token F1: 0.2655
policy average answer token F1:   0.2665
delta:                            +0.0010
oracle gap closed:                1.80%
replacement count:                189 / 610
improved count:                   26 / 610
regressed count:                  22 / 610
gold-document citation count:      327 -> 327
citation delta:                   0
citation lost / gained:           4 / 4
```

Sensitivity policy, `single_candidate_answer`：

```text
policy average answer token F1:   0.2708
delta:                            +0.0347
replacement count:                188 / 610
regressed count:                  55 / 610
gold-document citation count:      192 -> 248
citation delta:                   +56
```

Sensitivity policy, `top3_leading_candidate_rewrite`：

```text
policy average answer token F1:   0.2667
delta:                            +0.0012
replacement count:                188 / 610
regressed count:                  21 / 610
gold-document citation count:      327 -> 327
citation delta:                   0
citation lost / gained:           4 / 4
```

Main vs sensitivity：

```text
single_candidate_answer:
  policy_average_f1_difference: -0.0009
  average_delta_difference:     -0.0009
  regressed_count_difference:   +1
  gold_citation_count_difference: 0

top3_leading_candidate_rewrite:
  policy_average_f1_difference: -0.0002
  average_delta_difference:     -0.0002
  regressed_count_difference:   +1
  gold_citation_count_difference: 0
```

Top3 route-level 现象：

```text
other:                                  +0.0018
error_or_log:                            +0.0012
install_upgrade_config:                  +0.0010
security_bulletin_vulnerability_detail:  +0.0007
how_to_or_lookup:                        +0.0000
security_bulletin_affected_product:      -0.0941, n = 1
```

Sensitivity policy 在 `security_bulletin_affected_product` 上保持 baseline：

```text
security_bulletin_affected_product:
  question_count: 1
  average_delta_vs_baseline: +0.0000
  replacement_count: 0
  regressed_count: 0
```

### 可视化结果

本阶段新增 SVG 可视化：

```text
artifacts/candidate_reranker_stage37_answer_visuals/guarded_answer_policy_delta.svg
artifacts/candidate_reranker_stage37_answer_visuals/guarded_answer_main_route_delta.svg
artifacts/candidate_reranker_stage37_answer_visuals/guarded_answer_main_citation_delta.svg
```

Stage 37 主 JSON artifact：

```text
artifacts/candidate_reranker_stage37_answer_experiment.json
```

说明：以上 artifact 都在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：single-candidate proxy 看起来提升明显，但 top3 proxy 提升很小。
  - 原因：当答案由 top3 多句组成时，替换或前置一个候选只影响一部分 answer text；
  - 原 top3 已经包含不少有效证据，因此 reranker 的 single-candidate 收益被组合答案稀释。
- 问题 2：top3 proxy 仍有 regression。
  - main policy top3 regression 是 22 个；
  - sensitivity top3 regression 是 21 个；
  - regression 没有被当前 rank/margin/route gate 完全消除。
- 问题 3：`security_bulletin_affected_product` 仍然是 1-sample 敏感点。
  - main policy 在该 route 上 top3 delta 是 `-0.0941`；
  - sensitivity 通过 block 它避免了这个 regression；
  - 但 n=1 仍不足以证明应该把它写死进主策略。
- 问题 4：top3 citation count 总量没变，但有 citation exchange。
  - citation lost 是 4；
  - citation gained 是 4；
  - 总量为 0 不代表没有个案风险。

### 修正与处理

- 不接入 runtime。
- 不把 sensitivity policy 设为默认。
- 不把 `security_bulletin_affected_product` 的 1 个样本作为稳定主策略依据。
- 保留 Stage 36 main policy 作为继续分析对象。
- 下一步重点从 aggregate 指标转向 changed case 级别：
  - 分析 22 个 top3 regression；
  - 分析 4 个 citation lost；
  - 分析 4 个 citation gained；
  - 判断是否存在可解释、可泛化的 stricter gate。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\guarded_candidate_reranker_answer_experiment.py `
  scripts\evaluate_guarded_candidate_reranker_answers.py `
  tests\test_guarded_candidate_reranker_answer_experiment.py

pytest -q tests\test_guarded_candidate_reranker_answer_experiment.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

### 结论

- Stage 37 说明：
  - guarded reranker 在 single-candidate answer proxy 上仍有明显提升；
  - 但放到 top3 answer proxy 后，主策略只提升 `+0.0010`；
  - sensitivity 只比 main 多 `+0.0002` top3 delta，仍然主要受 1-sample route 影响；
  - 当前证据不足以接入 runtime；
  - 下一步应该做 top3 changed-case error analysis，而不是继续直接推进 runtime。

### 我学到的

- 候选级 F1 提升不能直接等价为多句答案级提升。
- answer composition 会稀释 single-candidate reranking 的收益。
- citation 总量不变时，仍然要检查 citation lost 和 citation gained 的交换。
- 1-sample route 即使能改善 headline metric，也不能作为稳定策略依据。
- 可视化能更快暴露“整体有小收益，但 route 级别很薄”的事实。

### 下一步

- 做 Stage 38：top3 guarded answer changed-case error analysis。
- 目标：
  1. 读取 Stage 37 artifact；
  2. 专门分析 main policy 的 22 个 top3 regression；
  3. 专门分析 4 个 citation lost 和 4 个 citation gained；
  4. 按 route、selected rank、score margin、candidate score、document transition 做归因；
  5. 判断是否存在可泛化的 stricter gate；
  6. 继续不改 runtime，除非后续端到端证据足够。

## Stage 38 - Top3 Guarded Answer Changed-Case Error Analysis

### 目标

- 接着 Stage 37，专门分析 `top3_leading_candidate_rewrite` 的 changed cases。
- 本阶段重点不是再看 headline average F1，而是解释：
  1. 22 个 top3 regression 来自哪里；
  2. 4 个 citation lost 和 4 个 citation gained 发生在什么条件下；
  3. selected rank、model score margin、candidate score、document transition 是否能形成更稳的 gate；
  4. 是否存在可以进入下一步 policy 实验的 stricter gate。
- 本阶段仍然是离线分析，不改 runtime。

### 起始状态

```text
git: main...origin/main clean
latest committed stage: Stage 37

Stage 37 main top3 proxy:
baseline F1: 0.2655
policy F1:   0.2665
delta:       +0.0010
regressions: 22 / 610
citation lost / gained: 4 / 4
```

### 边界说明

- Stage 37 JSON 只保存 aggregate 和 sample cases，不保存完整 610 个 case。
- 因此 Stage 38 的做法是：
  1. 读取 Stage 37 report；
  2. 校验 Stage 37 main policy、mode、aggregate 指标；
  3. 从 Stage 31 candidate dataset 和本地 PrimeQA gold answers 复算完整 610 个 top3 cases；
  4. 再做 changed-case attribution。
- 这避免把 sample cases 当作全量事实。
- gate audit 是 post-hoc analysis，不是 runtime 策略变更。

### 本阶段新增内容

- 更新 `src/ts_rag_agent/application/guarded_candidate_reranker_answer_experiment.py`
  - 新增 `build_topk_leading_candidate_answer_cases_from_decisions`
  - 目的：让 Stage 38 可以复用 Stage 37 的 top-k case 构造逻辑，而不是复制一份。
- 新增 `src/ts_rag_agent/application/guarded_candidate_reranker_changed_case_analysis.py`
  - 复算 full top3 cases；
  - 校验 Stage 37 aggregate；
  - 输出 changed-case metrics；
  - 输出 regression/citation loss/citation gain cases；
  - 按 route、rank bucket、model margin bucket、candidate score bucket、document transition 归因；
  - 做 stricter gate post-hoc audit；
  - 输出 SVG 可视化。
- 新增 `scripts/analyze_guarded_candidate_changed_cases.py`
  - 读取 Stage 31 dataset；
  - 读取 Stage 37 report；
  - 读取本地 PrimeQA dev/train gold answers；
  - 输出 Stage 38 JSON 和 SVG。
- 新增 `tests/test_guarded_candidate_reranker_changed_case_analysis.py`
  - 验证 changed-case attribution；
  - 验证 Stage 37 report 一致性校验；
  - 验证 SVG 写出。

### 命令

```powershell
python scripts\analyze_guarded_candidate_changed_cases.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage37-report artifacts\candidate_reranker_stage37_answer_experiment.json `
  --model logistic_best_candidate `
  --fold-count 5 `
  --splits dev,train `
  --max-answer-candidates 3 `
  --sample-limit 50 `
  --output artifacts\candidate_reranker_stage38_changed_case_analysis.json `
  --visualization-dir artifacts\candidate_reranker_stage38_changed_case_visuals
```

### 结果

复核后的 main top3 proxy：

```text
question_count: 610
changed_case_count: 189
average_delta_vs_baseline: +0.0010
improved_count: 26
regressed_count: 22
citation_lost_count: 4
citation_gained_count: 4
gold_citation_delta: 0
```

Regression route summary：

```text
other:                                  8 cases, avg delta -0.0602
error_or_log:                            6 cases, avg delta -0.0194
install_upgrade_config:                  5 cases, avg delta -0.0488
security_bulletin_vulnerability_detail:  2 cases, avg delta -0.0037
security_bulletin_affected_product:      1 case,  avg delta -0.0941
```

Rank bucket summary：

```text
rank_2: 80 changed, 0 improved, 0 regressed, avg delta +0.0000
rank_3: 58 changed, 0 improved, 0 regressed, avg delta +0.0000
rank_4: 32 changed, 15 improved, 15 regressed, avg delta +0.0106
rank_5: 19 changed, 11 improved, 7 regressed, avg delta +0.0155
```

解释：

- `rank_2` 和 `rank_3` 在 top3 proxy 中只是重排原 top3，token F1 和 citation 基本不变。
- 真正影响 answer text 的 case 来自 `rank_4` 和 `rank_5`。
- 22 / 22 个 regression 都来自 rank 4/5 replacement。

Model score margin bucket summary：

```text
margin 0.05-0.10: 41 changed, avg delta -0.0019, regressions 6
margin 0.10-0.20: 65 changed, avg delta +0.0029, regressions 8
margin 0.20-0.40: 78 changed, avg delta +0.0068, regressions 7
margin 0.40+:      5 changed, avg delta -0.0013, regressions 1
```

Candidate score bucket summary：

```text
score < 60:    48 changed, avg delta -0.0025, regressions 13, citation lost 2, citation gained 0
score 60-90:   77 changed, avg delta +0.0057, regressions 7,  citation lost 2, citation gained 4
score 90-120:  46 changed, avg delta +0.0060, regressions 2,  citation lost 0, citation gained 0
score 120+:    18 changed, avg delta +0.0018, regressions 0,  citation lost 0, citation gained 0
```

Document transition summary：

```text
new_non_gold_leading_document:       88 changed, avg delta +0.0021, regressions 14, citation lost 3
new_gold_leading_document:           67 changed, avg delta +0.0072, regressions 2,  citation gained 4
same_leading_document:               23 changed, avg delta +0.0081, regressions 2,  citation lost 1
gold_to_non_gold_leading_document:   11 changed, avg delta -0.0208, regressions 4
```

Post-hoc stricter gate audit：

```text
current main policy:
  delta +0.0010
  regressions 22
  citation lost/gained 4/4
  gold citation delta 0

rank_lte_3:
  blocked replacements: 51
  delta: +0.0000
  regressions: 0
  citation lost/gained: 0/0
  note: removes all real top3 effects, so it is not useful as a positive policy.

model_margin_gte_0.10:
  blocked replacements: 41
  delta: +0.0012
  regressions: 16
  citation lost/gained: 3/3
  gold citation delta: 0

model_margin_gte_0.20:
  blocked replacements: 106
  delta: +0.0009
  regressions: 8
  citation lost/gained: 2/3
  gold citation delta: +1

candidate_score_gte_60:
  blocked replacements: 48
  delta: +0.0012
  regressions: 9
  citation lost/gained: 2/4
  gold citation delta: +2

candidate_score_gte_90:
  blocked replacements: 125
  delta: +0.0005
  regressions: 2
  citation lost/gained: 0/0
  gold citation delta: 0

same_leading_document_only:
  blocked replacements: 166
  delta: +0.0003
  regressions: 2
  citation lost/gained: 1/0
  gold citation delta: -1
```

### 可视化结果

本阶段新增 SVG 可视化：

```text
artifacts/candidate_reranker_stage38_changed_case_visuals/stage38_changed_case_outcomes.svg
artifacts/candidate_reranker_stage38_changed_case_visuals/stage38_regressions_by_route.svg
artifacts/candidate_reranker_stage38_changed_case_visuals/stage38_gate_audit_delta.svg
artifacts/candidate_reranker_stage38_changed_case_visuals/stage38_gate_audit_regressions.svg
```

Stage 38 主 JSON artifact：

```text
artifacts/candidate_reranker_stage38_changed_case_analysis.json
```

说明：以上 artifact 都在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：Stage 37 artifact 没有保存完整 case list。
  - 原因：Stage 37 为了避免 JSON 过大，只保存 aggregate 和 sample cases；
  - 修正：Stage 38 读取 Stage 37 report 做一致性校验，再从 dataset 复算 full cases。
- 问题 2：rank gate 容易产生误读。
  - `rank_lte_3` 能把 regression 清零；
  - 但它也把 top3 proxy 的真实文本变化基本清零，delta 变成 `+0.0000`；
  - 因此它不是正向策略，只是说明风险集中在 rank 4/5。
- 问题 3：model margin 不是唯一解释变量。
  - `margin_0.20_0.40` 仍有 7 个 regression；
  - 高 margin 也可能选错，因为模型 confidence 不等于 answer utility。
- 问题 4：candidate score 更有解释力。
  - `score < 60` 的 changed cases 平均 delta 为负；
  - candidate_score_gte_60 同时提升 delta、减少 regression、改善 gold citation delta；
  - 但这仍是 post-hoc evidence，需要下一步独立评估。

### 修正与处理

- 不接入 runtime。
- 不把任何 Stage 38 gate 直接设为默认。
- 把 `candidate_score_gte_60` 标记为下一步候选 gate，而不是结论。
- 下一步需要把 candidate score gate 放回 policy evaluation：
  - 复用 grouped-CV selections；
  - 加入 `selected_candidate_score >= 60` 约束；
  - 同时比较 main policy、model_margin_gte_0.10、candidate_score_gte_60、candidate_score_gte_90；
  - 观察 top3 proxy 和 single-candidate proxy 是否仍稳定。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\guarded_candidate_reranker_changed_case_analysis.py `
  tests\test_guarded_candidate_reranker_changed_case_analysis.py

pytest -q tests\test_guarded_candidate_reranker_changed_case_analysis.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 94 passed
```

### 结论

- Stage 38 的核心结论：
  - top3 regression 集中在 rank 4/5 replacement；
  - 低 candidate score 是更强的风险信号；
  - `candidate_score_gte_60` 是当前最值得进入下一步验证的 post-hoc gate；
  - 它在 Stage 38 audit 中把 regression 从 22 降到 9，并把 delta 从 `+0.0010` 提到 `+0.0012`；
  - 但它仍是后验分析结果，不能直接接 runtime。

### 我学到的

- 对 top3 answer 来说，rank 2/3 的“替换”很多时候只是重排，不会改变 token F1 或 citation。
- regression 不只看 model margin，candidate 本身的 evidence score 更有解释力。
- citation 总量为 0 delta 时，case-level citation lost/gained 仍然必须拆开看。
- post-hoc gate 可以提出候选，但必须经过独立 policy evaluation 才能进入下一阶段。
- 记录 artifact 结构的限制很重要，否则下一步会误把 sample 当作 full data。

### 下一步

- 做 Stage 39：candidate-score guarded policy evaluation。
- 目标：
  1. 把 `selected_candidate_score >= 60` 做成候选约束；
  2. 对比 Stage 36 main policy、`model_margin_gte_0.10`、`candidate_score_gte_60`、`candidate_score_gte_90`；
  3. 同时输出 single-candidate proxy 和 top3 proxy；
  4. 重点看 delta、regression count、citation lost/gained、gold citation delta；
  5. 如果候选 gate 稳定，再考虑进入 verified RAG dev split runtime-style 实验；
  6. 继续不改默认 runtime。

## Stage 39 - Candidate-score guarded policy evaluation

真实执行时间：2026-07-13

### 目标

- 把 Stage 38 发现的 `candidate_score_gte_60` 从 post-hoc audit 变成正式离线 policy constraint。
- 对比以下固定策略：
  - `stage36_main`
  - `model_margin_gte_0.10`
  - `candidate_score_gte_60`
  - `candidate_score_gte_90`
- 同时评估：
  - `single_candidate_answer`
  - `top3_leading_candidate_rewrite`
- 继续保持边界：
  - 不改 runtime 默认策略；
  - 不使用 held-out test set；
  - 本阶段仍是 `dev,train` grouped-CV 离线评估，不是最终测试集结论。

### 变更内容

- 扩展 `CandidateRerankerPolicyConfig`：
  - 新增 `min_selected_candidate_score`；
  - 新增配置合法性校验；
  - 当 selected candidate 的 runtime `candidate_score` 低于门槛时，记录 `selected_runtime_candidate_score_below_min`。
- 从 guarded answer experiment 中抽出可复用的 case builder 和 summarizer，供 Stage 39 复用。
- 新增 `candidate_score_guarded_policy_evaluation.py`：
  - 固定策略集合；
  - grouped-CV selection 复用；
  - single/top-k 两种 answer proxy；
  - policy vs main delta；
  - route 和 selected-rank 分组指标；
  - SVG 可视化输出。
- 新增脚本 `scripts/evaluate_candidate_score_guarded_policies.py`。
- 新增测试：
  - candidate-score gate 会阻止低 runtime candidate score 的 replacement；
  - Stage 39 固定策略结果可序列化；
  - SVG 可视化可写出。

### 实验命令

```powershell
python scripts\evaluate_candidate_score_guarded_policies.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --fold-count 5 `
  --splits dev,train `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage39_candidate_score_policy_evaluation.json `
  --visualization-dir artifacts\candidate_reranker_stage39_policy_visuals
```

### 结果

`single_candidate_answer`：

```text
stage36_main:
  F1 0.2699, delta +0.0338, replacements 189, regressions 56, citation lost/gained 11/67, gold citation delta +56

model_margin_gte_0.10:
  F1 0.2647, delta +0.0286, replacements 148, regressions 42, citation lost/gained 7/56, gold citation delta +49

candidate_score_gte_60:
  F1 0.2679, delta +0.0318, replacements 141, regressions 40, citation lost/gained 4/60, gold citation delta +56

candidate_score_gte_90:
  F1 0.2529, delta +0.0168, replacements 64, regressions 15, citation lost/gained 2/23, gold citation delta +21
```

`top3_leading_candidate_rewrite`：

```text
stage36_main:
  F1 0.2665, delta +0.0010, replacements 189, regressions 22, citation lost/gained 4/4, gold citation delta 0

model_margin_gte_0.10:
  F1 0.2666, delta +0.0012, replacements 148, regressions 16, citation lost/gained 3/3, gold citation delta 0

candidate_score_gte_60:
  F1 0.2667, delta +0.0012, replacements 141, regressions 9, citation lost/gained 2/4, gold citation delta +2

candidate_score_gte_90:
  F1 0.2660, delta +0.0005, replacements 64, regressions 2, citation lost/gained 0/0, gold citation delta 0
```

相对 Stage 36 main policy 的关键差异：

```text
candidate_score_gte_60 / single:
  delta diff -0.0020, regressions -16, gold citation delta diff 0

candidate_score_gte_60 / top3:
  delta diff +0.0002, regressions -13, gold citation delta diff +2

candidate_score_gte_90 / top3:
  delta diff -0.0005, regressions -20, gold citation delta diff 0
```

### 可视化结果

本阶段新增 4 个 SVG：

```text
artifacts/candidate_reranker_stage39_policy_visuals/stage39_single_candidate_policy_delta.svg
artifacts/candidate_reranker_stage39_policy_visuals/stage39_top3_policy_delta.svg
artifacts/candidate_reranker_stage39_policy_visuals/stage39_top3_policy_regressions.svg
artifacts/candidate_reranker_stage39_policy_visuals/stage39_top3_policy_citation_exchange.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage39_candidate_score_policy_evaluation.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：Stage 38 的 `candidate_score_gte_60` 只是 post-hoc audit。
  - 原因：Stage 38 是 changed-case 分析，不是统一 policy evaluation。
  - 处理：Stage 39 把它放回 `CandidateRerankerPolicyConfig`，作为正式离线策略约束重跑。
- 问题 2：单看 top3 proxy 容易过度乐观。
  - 原因：top3 rewrite 中候选替换经常只改变 leading sentence，不一定改变答案整体 token F1。
  - 处理：同时报告 single-candidate proxy；`candidate_score_gte_60` 在 top3 更好，但 single delta 比 main 低 `0.0020`。
- 问题 3：`candidate_score_gte_90` 看起来 regression 最低，但收益也被压低。
  - 原因：门槛太高会过度阻止 replacement。
  - 处理：不把最低 regression 当成唯一目标，同时观察 delta 和 citation。
- 问题 4：可视化函数最初写死 `top3` 显示名。
  - 原因：Stage 39 当前固定跑 top3，但模块参数支持不同 top-k。
  - 处理：改为从 mode name 动态生成显示名，避免以后 `top1/top5` 实验时标题不真实。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\candidate_score_guarded_policy_evaluation.py `
  scripts\evaluate_candidate_score_guarded_policies.py `
  tests\test_candidate_score_guarded_policy_evaluation.py

pytest -q tests\test_candidate_score_guarded_policy_evaluation.py `
  tests\test_candidate_reranker_policy_search.py
```

结果：

```text
ruff: passed
pytest: 5 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 97 passed
```

### 结论

- `candidate_score_gte_60` 是当前最均衡的候选策略：
  - top3 proxy delta 从 `+0.0010` 提到 `+0.0012`；
  - top3 regression 从 `22` 降到 `9`；
  - top3 gold citation delta 从 `0` 提到 `+2`；
  - single-candidate gold citation delta 仍保持 `+56`；
  - 但 single-candidate delta 比 main 少 `0.0020`。
- `candidate_score_gte_90` 不适合作为下一步主候选：
  - top3 regression 最低，为 `2`；
  - 但 top3 delta 降到 `+0.0005`；
  - single delta 也降到 `+0.0168`，说明过度保守。
- 这仍不是测试集结论，也不是 runtime 结论。
- 不能把 `candidate_score_gte_60` 直接设为默认策略；它只能进入下一阶段更严格的 split-respecting 验证。

### 我学到的

- 一个 gate 是否好，不能只看 regression count；过度保守会把有效 replacement 一起挡掉。
- `candidate_score_gte_60` 的价值在于同时减少 regression、保持 citation 收益，并且不牺牲 single gold citation delta。
- top3 proxy 是更接近答案组合行为的指标，但 single-candidate proxy 能提醒我们不要忽略候选本身质量的损失。
- 每次把 post-hoc 发现升级为 policy，都必须重跑固定策略评估，而不是沿用 audit 结果。
- “还没用测试集”必须明确记录；否则后面容易把 dev/train CV 结果误读成最终泛化结果。

### 下一步

- 做 Stage 40：split-respecting train-to-dev candidate-score policy validation。
- 目标：
  1. 只在 train split 上训练/选择候选策略；
  2. 在 dev split 上评估固定策略；
  3. 对比 `stage36_main` 和 `candidate_score_gte_60`；
  4. 继续同时报告 single-candidate proxy 和 top3 proxy；
  5. 明确保持 held-out test set 不使用；
  6. 只有 train-to-dev 边界验证也稳定后，才讨论是否进入最终 test set 一次性评估或 runtime dev-style end-to-end。

## Stage 40 - Split-respecting train-to-dev candidate-score policy validation

真实执行时间：2026-07-13

### 目标

- 把 Stage 39 的 `dev,train` grouped-CV 结果推进到更严格的 split-respecting 验证。
- 分成两块做：
  1. `train` only grouped-CV：只在 train split 内观察固定策略；
  2. train-to-dev holdout：candidate reranker 只用 train split 拟合，再只在 dev split 上评估固定策略。
- 继续保持边界：
  - 不使用 held-out test set；
  - 不改 runtime 默认策略；
  - 不把 holdout 结果包装成最终泛化结论。

### 变更内容

- 在 `candidate_reranker_cv.py` 新增 `split_validated_candidate_reranker_selections()`：
  - 明确训练 split 和验证 split 必须不同；
  - 明确拒绝空 split；
  - 训练时只使用 `train_split`；
  - selection 只输出 `validation_split` 的 question。
- 把 Stage 39 的 candidate-score policy evaluation 抽象为可复用接口：
  - 新增 `evaluate_candidate_score_guarded_policies_from_selections()`；
  - 结果中新增 `selection_scope`、`train_split`、`evaluation_split`、question count 等字段；
  - SVG 写出支持自定义 artifact prefix 和 title prefix。
- 新增 `candidate_score_guarded_policy_split_validation.py`：
  - 统一输出 train-only CV 与 train-to-dev holdout；
  - 复用同一套 single/top3 proxy metrics；
  - 复用同一套固定策略集合。
- 新增脚本 `scripts/evaluate_candidate_score_guarded_policy_split_validation.py`。
- 新增测试：
  - train-to-dev selection 只输出 dev 问题；
  - 同 split 会报错；
  - Stage 40 split validation 可序列化；
  - Stage 40 SVG 可写出；
  - 显式空 policy list 会报错，避免隐式回到默认策略。

### 实验命令

```powershell
python scripts\evaluate_candidate_score_guarded_policy_split_validation.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --train-fold-count 5 `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage40_candidate_score_split_validation.json `
  --visualization-dir artifacts\candidate_reranker_stage40_split_validation_visuals
```

真实数据规模：

```text
train question count: 450
dev evaluation question count: 160
held-out test set: not used
```

### Train-only CV 结果

`single_candidate_answer`：

```text
stage36_main:
  F1 0.2581, delta +0.0293, replacements 124, regressions 34, citation lost/gained 6/42, gold citation delta +36

model_margin_gte_0.10:
  F1 0.2516, delta +0.0228, replacements 96, regressions 26, citation lost/gained 4/34, gold citation delta +30

candidate_score_gte_60:
  F1 0.2576, delta +0.0288, replacements 97, regressions 24, citation lost/gained 4/40, gold citation delta +36

candidate_score_gte_90:
  F1 0.2445, delta +0.0157, replacements 49, regressions 11, citation lost/gained 2/16, gold citation delta +14
```

`top3_leading_candidate_rewrite`：

```text
stage36_main:
  F1 0.2612, delta +0.0011, replacements 124, regressions 14, citation lost/gained 1/3, gold citation delta +2

model_margin_gte_0.10:
  F1 0.2612, delta +0.0011, replacements 96, regressions 9, citation lost/gained 0/2, gold citation delta +2

candidate_score_gte_60:
  F1 0.2609, delta +0.0008, replacements 97, regressions 8, citation lost/gained 1/3, gold citation delta +2

candidate_score_gte_90:
  F1 0.2608, delta +0.0006, replacements 49, regressions 2, citation lost/gained 0/0, gold citation delta 0
```

Train-only CV 观察：

- `model_margin_gte_0.10` 和 `stage36_main` 在 rounded top3 delta 上同为 `+0.0011`，但 `model_margin_gte_0.10` regression 更少。
- `candidate_score_gte_60` 在 train-only CV 的 top3 delta 是 `+0.0008`，低于 main 和 margin gate；
- 但它把 top3 regression 从 `14` 降到 `8`，single gold citation delta 仍保持 `+36`。

### Train-to-dev holdout 结果

`single_candidate_answer`：

```text
stage36_main:
  F1 0.2928, delta +0.0363, replacements 42, regressions 12, citation lost/gained 3/19, gold citation delta +16

model_margin_gte_0.10:
  F1 0.2879, delta +0.0315, replacements 35, regressions 12, citation lost/gained 2/16, gold citation delta +14

candidate_score_gte_60:
  F1 0.2952, delta +0.0388, replacements 29, regressions 8, citation lost/gained 0/16, gold citation delta +16

candidate_score_gte_90:
  F1 0.2746, delta +0.0181, replacements 12, regressions 3, citation lost/gained 0/7, gold citation delta +7
```

`top3_leading_candidate_rewrite`：

```text
stage36_main:
  F1 0.2791, delta -0.0014, replacements 42, regressions 8, citation lost/gained 2/1, gold citation delta -1

model_margin_gte_0.10:
  F1 0.2778, delta -0.0028, replacements 35, regressions 7, citation lost/gained 2/0, gold citation delta -2

candidate_score_gte_60:
  F1 0.2813, delta +0.0008, replacements 29, regressions 1, citation lost/gained 0/0, gold citation delta 0

candidate_score_gte_90:
  F1 0.2806, delta +0.0001, replacements 12, regressions 0, citation lost/gained 0/0, gold citation delta 0
```

Holdout 关键差异：

```text
candidate_score_gte_60 vs stage36_main / top3:
  delta: -0.0014 -> +0.0008
  regressions: 8 -> 1
  citation lost/gained: 2/1 -> 0/0
  gold citation delta: -1 -> 0

candidate_score_gte_60 vs stage36_main / single:
  delta: +0.0363 -> +0.0388
  regressions: 12 -> 8
  citation lost/gained: 3/19 -> 0/16
  gold citation delta: +16 -> +16
```

### 可视化结果

本阶段新增 8 个 SVG：

```text
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_train_cv_single_candidate_policy_delta.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_train_cv_top3_policy_delta.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_train_cv_top3_policy_regressions.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_train_cv_top3_policy_citation_exchange.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_holdout_single_candidate_policy_delta.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_holdout_top3_policy_delta.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_holdout_top3_policy_regressions.svg
artifacts/candidate_reranker_stage40_split_validation_visuals/stage40_holdout_top3_policy_citation_exchange.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage40_candidate_score_split_validation.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：Stage 39 的 `candidate_score_gte_60` 阈值来自 `dev,train` grouped-CV 后的分析。
  - 原因：Stage 38/39 都还没有严格切开 train/dev 边界。
  - 处理：Stage 40 增加 train-only CV 与 train-to-dev holdout，明确不使用 test set。
- 问题 2：Train-only CV 与 dev holdout 的策略排序不完全一致。
  - Train-only CV 中 `model_margin_gte_0.10` 的 top3 rounded delta 与 main 同为 `+0.0011`，且 regression 更少；
  - Dev holdout 中 `candidate_score_gte_60` 最强，top3 delta 变为正，regression 从 `8` 降到 `1`。
  - 这说明 split 边界下仍存在分布差异，不能只凭一个 aggregate 排名直接改 runtime。
- 问题 3：显式空 policy list 最初会因为 Python 的 truthy 逻辑退回默认 policy。
  - 原因：`policies or default_stage39_policy_specs()` 会把空列表当作未传参。
  - 处理：改成 `policies is None` 才使用默认策略；显式空策略会报错。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\candidate_reranker_cv.py `
  src\ts_rag_agent\application\candidate_score_guarded_policy_evaluation.py `
  src\ts_rag_agent\application\candidate_score_guarded_policy_split_validation.py `
  scripts\evaluate_candidate_score_guarded_policy_split_validation.py `
  tests\test_candidate_reranker_cv.py `
  tests\test_candidate_score_guarded_policy_evaluation.py `
  tests\test_candidate_score_guarded_policy_split_validation.py

pytest -q tests\test_candidate_reranker_cv.py `
  tests\test_candidate_score_guarded_policy_evaluation.py `
  tests\test_candidate_score_guarded_policy_split_validation.py
```

结果：

```text
ruff: passed
pytest: 10 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 102 passed
```

### 结论

- `candidate_score_gte_60` 在 dev holdout 上是当前最强候选：
  - top3 delta 从 main 的 `-0.0014` 提到 `+0.0008`；
  - top3 regression 从 `8` 降到 `1`；
  - top3 citation lost 从 `2` 降到 `0`；
  - single-candidate delta 也比 main 高 `+0.0025`；
  - single-candidate gold citation delta 保持 `+16`。
- 但 train-only CV 并没有把它排成唯一明显第一：
  - `model_margin_gte_0.10` 在 train-only CV 的 top3 rounded delta 与 main 同为 `+0.0011`；
  - `candidate_score_gte_60` 的 train-only top3 delta 是 `+0.0008`。
- 因此 Stage 40 的结论是：
  - `candidate_score_gte_60` 可以进入更细的 dev holdout changed-case 审计；
  - 不能直接设为 runtime 默认；
  - 不能使用 held-out test set 做反复调参。

### 我学到的

- 严格切开 train/dev 后，aggregate ranking 可能变化；这比继续在 dev+train CV 上打磨更接近真实风险。
- 只看 train-only CV 可能错过 dev holdout 上更稳的策略，但只看 dev holdout 又会带来新的调参风险。
- candidate score gate 的价值不是“收益最大”，而是把负 top3 delta、regression 和 citation loss 同时压住。
- 显式参数和默认参数要分清楚；空 policy list 不应该被当作“使用默认”。
- 目前仍然没有使用测试集，这个边界必须一直写清楚。

### 下一步

- 做 Stage 41：dev holdout changed-case audit for `candidate_score_gte_60`。
- 目标：
  1. 只分析 Stage 40 dev holdout 中 `candidate_score_gte_60` 与 `stage36_main` 的 changed cases；
  2. 找出 `candidate_score_gte_60` 留下的 1 个 top3 regression；
  3. 拆 route、rank、document transition、candidate score bucket；
  4. 输出 JSON 与 SVG 可视化；
  5. 判断 holdout 收益是否集中在少数偶然 case，还是有稳定模式；
  6. 继续不使用 held-out test set，不改 runtime 默认策略。

## Stage 41 - Dev holdout changed-case audit for candidate_score_gte_60

真实执行时间：2026-07-13

### 目标

- 只审计 Stage 40 train-to-dev holdout 的 dev 结果。
- 对比：
  - `stage36_main`
  - `candidate_score_gte_60`
- changed case 定义：
  - 在同一批 train-to-dev holdout selections 上，两种 policy 生成的 top-k candidate list 不同。
- 重点回答：
  1. `candidate_score_gte_60` 为什么能把 top3 regression 从 `8` 降到 `1`；
  2. 它相对 main 变差的 case 是哪些；
  3. 剩下的 1 个 top3 regression 是什么；
  4. 是否可以进入 runtime 实验前的更细 gate 设计。
- 边界：
  - 不使用 held-out test set；
  - 不改 runtime 默认策略；
  - 不把 dev holdout changed-case 审计当成最终测试结论。

### 变更内容

- 新增 `candidate_score_holdout_changed_case_audit.py`：
  - 重建 train-to-dev holdout selections；
  - 重算 `stage36_main` 与 `candidate_score_gte_60` 的 top-k cases；
  - 校验重算指标与 Stage 40 JSON report 一致；
  - 输出 policy 间 changed cases；
  - 输出 `candidate_score_gte_60` 的 residual regression cases；
  - 拆分 route、rank、blocked candidate score bucket、document transition。
- 新增脚本 `scripts/analyze_candidate_score_holdout_changed_cases.py`：
  - 输入 Stage 31 candidate dataset；
  - 输入 Stage 40 split-validation JSON；
  - 输出 Stage 41 JSON 与 SVG。
- 新增测试：
  - 能找到 changed cases；
  - 能找到 residual regression cases；
  - 能写出 SVG；
  - Stage 40 report 与重算结果不一致时会报错。

### 实验命令

```powershell
python scripts\analyze_candidate_score_holdout_changed_cases.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage40-report artifacts\candidate_reranker_stage40_candidate_score_split_validation.json `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --max-answer-candidates 3 `
  --sample-limit 50 `
  --output artifacts\candidate_reranker_stage41_candidate_score_holdout_changed_cases.json `
  --visualization-dir artifacts\candidate_reranker_stage41_holdout_changed_case_visuals
```

### 总体结果

```text
question count: 160
changed cases vs stage36_main: 13
changed case rate: 0.0813

stage36_main top3:
  delta -0.0014
  regressions 8
  citation lost 2
  gold citation delta -1

candidate_score_gte_60 top3:
  delta +0.0008
  regressions 1
  citation lost 0
  gold citation delta 0

candidate_score_gte_60 average delta vs main:
  +0.0022
```

Changed cases 中，`candidate_score_gte_60` 相对 main：

```text
better: 7
tied:   3
worse:  3
```

### 分组结果

Route：

```text
other:
  cases 7, avg delta vs main +0.0273, better 3, worse 2

install_upgrade_config:
  cases 3, avg delta vs main +0.0231, better 2, worse 1

error_or_log:
  cases 2, avg delta vs main +0.0018, better 1, worse 0

security_bulletin_affected_product:
  cases 1, avg delta vs main +0.0941, better 1, worse 0
```

Blocked candidate score bucket：

```text
score_lt_60:
  cases 13, avg delta vs main +0.0275, better 7, worse 3
```

Rank bucket：

```text
rank_4:
  cases 6, avg delta vs main +0.0434, better 4, worse 2

rank_5:
  cases 4, avg delta vs main +0.0244, better 3, worse 1

rank_3:
  cases 3, avg delta vs main +0.0000, better 0, worse 0
```

Document transition：

```text
new_non_gold_leading_document:
  cases 4, avg delta vs main +0.0120, better 3, worse 1

new_gold_leading_document:
  cases 3, avg delta vs main +0.0994, better 3, worse 0

same_leading_document:
  cases 3, avg delta vs main +0.0292, better 1, worse 1

gold_to_non_gold_leading_document:
  cases 3, avg delta vs main -0.0253, better 0, worse 1
```

### 关键 case

`candidate_score_gte_60` 相对 main 变差的 3 个 changed cases：

```text
DEV_Q119:
  route: other
  candidate delta vs main: -0.0758
  main delta vs baseline: +0.0758
  candidate_score_gte_60 delta vs baseline: +0.0000
  blocked candidate rank: 4
  blocked candidate score: 13.0000
  transition: gold_to_non_gold_leading_document
  main_gold_cited: True
  candidate_gold_cited: False

DEV_Q282:
  route: other
  candidate delta vs main: -0.0065
  main delta vs baseline: +0.0065
  candidate_score_gte_60 delta vs baseline: +0.0000
  blocked candidate rank: 4
  blocked candidate score: 25.8044
  transition: same_leading_document
  main_gold_cited: False
  candidate_gold_cited: True

DEV_Q201:
  route: install_upgrade_config
  candidate delta vs main: -0.0028
  main delta vs baseline: +0.0028
  candidate_score_gte_60 delta vs baseline: +0.0000
  blocked candidate rank: 5
  blocked candidate score: 56.8104
  transition: new_non_gold_leading_document
  main_gold_cited: False
  candidate_gold_cited: False
```

`candidate_score_gte_60` 剩下的 1 个 residual top3 regression：

```text
DEV_Q261:
  route: other
  delta vs baseline: -0.0573
  leading rank: 4
  leading candidate score: 63.6936
  score bucket: score_60_90
  document transition: new_non_gold_leading_document
  citation delta: 0
```

### 可视化结果

本阶段新增 5 个 SVG：

```text
artifacts/candidate_reranker_stage41_holdout_changed_case_visuals/stage41_candidate_vs_main_outcomes.svg
artifacts/candidate_reranker_stage41_holdout_changed_case_visuals/stage41_changed_cases_by_route.svg
artifacts/candidate_reranker_stage41_holdout_changed_case_visuals/stage41_changed_cases_by_blocked_score.svg
artifacts/candidate_reranker_stage41_holdout_changed_case_visuals/stage41_changed_cases_by_document_transition.svg
artifacts/candidate_reranker_stage41_holdout_changed_case_visuals/stage41_residual_regression_routes.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage41_candidate_score_holdout_changed_cases.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：只看 Stage 40 aggregate 会误以为 `candidate_score_gte_60` 没有明显副作用。
  - 原因：它确实把 regression 和 citation loss 压下来了；
  - 但 changed-case 审计显示仍有 3 个 case 比 main 差，其中 `DEV_Q119` 还涉及 main gold citation 被挡掉。
- 问题 2：`score_lt_60` 不是绝对坏信号。
  - 13 个 blocked cases 中有 7 个变好、3 个持平、3 个变差；
  - 说明低 score 是风险信号，不是充分拒绝条件。
- 问题 3：残留 regression 已经不属于 score < 60。
  - `DEV_Q261` 的 candidate score 是 `63.6936`；
  - 说明简单把门槛从 60 提到更高可能会挡住更多风险，但也会继续牺牲有效 replacement。
- 问题 4：document transition 方向很关键。
  - `new_gold_leading_document` 全部为正；
  - `gold_to_non_gold_leading_document` 平均为负；
  - 这提示下一步应该引入 document/citation-aware gate，而不是单独调 candidate_score 阈值。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\candidate_score_holdout_changed_case_audit.py `
  scripts\analyze_candidate_score_holdout_changed_cases.py `
  tests\test_candidate_score_holdout_changed_case_audit.py

pytest -q tests\test_candidate_score_holdout_changed_case_audit.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 105 passed
```

### 结论

- `candidate_score_gte_60` 的 holdout 收益不是单个偶然 case：
  - 13 个 changed cases 中 7 个更好；
  - regression 从 `8` 降到 `1`；
  - citation loss 从 `2` 降到 `0`。
- 但它不是可直接 runtime 化的最终策略：
  - 3 个 changed cases 比 main 差；
  - `DEV_Q119` 显示低 score 候选仍可能是 gold-leading improvement；
  - residual regression `DEV_Q261` 显示 score >= 60 仍可能选错。
- 下一步不应该继续单调提高 candidate_score 阈值。
- 更合理的方向是做一个 citation/document-aware guard：
  - 保留 `candidate_score_gte_60` 的风险控制；
  - 对 `new_gold_leading_document` 低 score 候选更谨慎地放行；
  - 对 `gold_to_non_gold_leading_document` 和 `new_non_gold_leading_document` 加强阻断。

### 我学到的

- changed-case audit 能把 aggregate 的“看起来很好”拆成可解释的收益和副作用。
- 低 candidate score 是强风险信号，但不是绝对坏信号；它需要和 document transition、gold citation proxy 一起看。
- runtime gate 不能只做单阈值调参，否则会在 `DEV_Q119` 这类 case 上误杀。
- residual regression 的定位比总体 regression count 更重要；`DEV_Q261` 已经说明下一步风险不是 `score_lt_60`，而是 rank 4 + score 60-90 + new non-gold leading document。
- 继续保留测试集不使用是必要的，否则会把探索性阈值设计污染成测试集调参。

### 下一步

- 做 Stage 42：citation/document-aware holdout guard design。
- 目标：
  1. 基于 Stage 41 changed-case evidence 设计候选 guard，不接 runtime；
  2. 比较 `candidate_score_gte_60` 与 document-transition-aware variants；
  3. 特别保护 `new_gold_leading_document` 的有效低分替换；
  4. 阻断 `gold_to_non_gold_leading_document` 和高风险 `new_non_gold_leading_document`；
  5. 单独观察 `DEV_Q119`、`DEV_Q201`、`DEV_Q261` 是否被正确处理；
  6. 继续只在 train/dev 范围内做离线设计，不使用 held-out test set，不改 runtime 默认策略。

## Stage 42 - Citation/document-aware holdout guard design

真实执行时间：2026-07-13

### 目标

- 基于 Stage 41 的 dev holdout changed-case evidence，设计 citation/document-aware guard 候选。
- 比较：
  - `stage36_main`
  - `candidate_score_gte_60`
  - `score60_or_new_gold`
  - `citation_preserving_score60`
  - `document_risk_v1`
  - `citation_doc_risk_v1`
- 单独跟踪：
  - `DEV_Q119`
  - `DEV_Q201`
  - `DEV_Q261`
- 继续保持边界：
  - 不使用 held-out test set；
  - 不改 runtime 默认策略；
  - 使用 gold/citation/document-transition 的 guard 必须标注为 `diagnostic_offline_only`，不能说成 runtime 可用策略。

### Guard 设计

```text
stage36_main:
  Stage 36 main policy baseline.

candidate_score_gte_60:
  runtime-feature-only reference.
  Reject selected candidate score < 60.

score60_or_new_gold:
  diagnostic_offline_only_gold_document_transition.
  Reject score < 60 unless replacement moves leading document from non-gold to gold.

citation_preserving_score60:
  diagnostic_offline_only_gold_citation.
  Keep citation-gaining replacements, reject citation-losing replacements, otherwise require score >= 60.

document_risk_v1:
  diagnostic_offline_only_gold_document_transition.
  Keep new-gold replacements, block gold-to-non-gold replacements, require score >= 90 for rank>=4 new-non-gold replacements.

citation_doc_risk_v1:
  diagnostic_offline_only_gold_citation_and_document_transition.
  Keep citation gains, block citation losses, and block rank>=4 new-non-gold replacements below score 90.
```

### 变更内容

- 新增 `document_aware_guard_design.py`：
  - 重建 train-to-dev holdout selections；
  - 重算 Stage 41 main 与 `candidate_score_gte_60` 指标；
  - 校验 Stage 41 JSON report 与重算结果一致；
  - 对多种 guard 候选进行离线评估；
  - 输出相对 main 与相对 `candidate_score_gte_60` 的 delta；
  - 输出 probe case 的逐策略处理结果；
  - 输出 SVG 可视化。
- 新增脚本 `scripts/evaluate_document_aware_guards.py`。
- 新增测试：
  - guard labels 与 feature scope 正确；
  - diagnostic-only guard 被明确标注；
  - probe cases 可输出；
  - Stage 41 report 不一致时会报错；
  - SVG 可写出。

### 实验命令

```powershell
python scripts\evaluate_document_aware_guards.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage41-report artifacts\candidate_reranker_stage41_candidate_score_holdout_changed_cases.json `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage42_document_aware_guard_design.json `
  --visualization-dir artifacts\candidate_reranker_stage42_document_aware_guard_visuals
```

### 结果

```text
stage36_main:
  F1 0.2791
  delta -0.0014
  replacements 42
  regressions 8
  citation lost/gained 2/1
  gold citation delta -1

candidate_score_gte_60:
  F1 0.2813
  delta +0.0008
  replacements 29
  regressions 1
  citation lost/gained 0/0
  gold citation delta 0

score60_or_new_gold:
  F1 0.2818
  delta +0.0013
  replacements 32
  regressions 1
  citation lost/gained 0/1
  gold citation delta +1

citation_preserving_score60:
  F1 0.2818
  delta +0.0013
  replacements 30
  regressions 1
  citation lost/gained 0/1
  gold citation delta +1

document_risk_v1:
  F1 0.2811
  delta +0.0006
  replacements 28
  regressions 0
  citation lost/gained 0/1
  gold citation delta +1

citation_doc_risk_v1:
  F1 0.2811
  delta +0.0006
  replacements 26
  regressions 0
  citation lost/gained 0/1
  gold citation delta +1
```

相对 `candidate_score_gte_60`：

```text
score60_or_new_gold:
  delta diff +0.0005
  regressions diff 0
  citation lost diff 0
  gold citation delta diff +1
  replacements diff +3

citation_preserving_score60:
  delta diff +0.0005
  regressions diff 0
  citation lost diff 0
  gold citation delta diff +1
  replacements diff +1

document_risk_v1:
  delta diff -0.0002
  regressions diff -1
  citation lost diff 0
  gold citation delta diff +1
  replacements diff -1

citation_doc_risk_v1:
  delta diff -0.0002
  regressions diff -1
  citation lost diff 0
  gold citation delta diff +1
  replacements diff -3
```

### Probe case 结果

`DEV_Q119`：

```text
stage36_main:
  replace, delta +0.0758, gold cited true, citation delta +1

candidate_score_gte_60:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0

score60_or_new_gold:
  replace, delta +0.0758, gold cited true, citation delta +1

document_risk_v1:
  replace, delta +0.0758, gold cited true, citation delta +1
```

解释：`candidate_score_gte_60` 误杀了一个低分但 new-gold 的有效替换。

`DEV_Q201`：

```text
stage36_main:
  replace, delta +0.0028, gold cited false, citation delta 0

candidate_score_gte_60:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0

score60_or_new_gold:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0

document_risk_v1:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0
```

解释：`DEV_Q201` 是低分 new-non-gold 小幅正收益，document-aware guard 仍会挡掉它。

`DEV_Q261`：

```text
candidate_score_gte_60:
  replace, delta -0.0573, gold cited false, citation delta 0

score60_or_new_gold:
  replace, delta -0.0573, gold cited false, citation delta 0

document_risk_v1:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0

citation_doc_risk_v1:
  keep top candidate, delta +0.0000, gold cited false, citation delta 0
```

解释：`DEV_Q261` 的风险不是 score < 60，而是 `rank 4 + score 60-90 + new_non_gold_leading_document`。

### 可视化结果

本阶段新增 5 个 SVG：

```text
artifacts/candidate_reranker_stage42_document_aware_guard_visuals/stage42_guard_delta.svg
artifacts/candidate_reranker_stage42_document_aware_guard_visuals/stage42_guard_regressions.svg
artifacts/candidate_reranker_stage42_document_aware_guard_visuals/stage42_guard_citation_loss.svg
artifacts/candidate_reranker_stage42_document_aware_guard_visuals/stage42_guard_changed_vs_score60.svg
artifacts/candidate_reranker_stage42_document_aware_guard_visuals/stage42_probe_delta.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage42_document_aware_guard_design.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：最高 F1 delta 的 guard 仍是 offline diagnostic。
  - `score60_or_new_gold` 达到 `+0.0013`，高于 `candidate_score_gte_60` 的 `+0.0008`；
  - 但它依赖 gold-document transition，runtime 不可直接获得。
- 问题 2：最低 regression 的 guard 牺牲了一点 delta。
  - `document_risk_v1` 和 `citation_doc_risk_v1` 把 regression 降到 `0`；
  - 但 delta 从 `candidate_score_gte_60` 的 `+0.0008` 降到 `+0.0006`。
- 问题 3：`DEV_Q201` 说明过度阻断会损失小幅正收益。
  - 它是 low-score new-non-gold，但 main 有 `+0.0028`；
  - document-risk guard 会选择阻断，换来更少风险。
- 问题 4：当前最佳方向还不能 runtime 化。
  - 这些诊断型 guard 证明 document/citation signal 有用；
  - 下一步需要寻找 runtime 可用 proxy，例如 document-score relation、rank、score bucket、query coverage、title overlap 等，而不是直接用 gold labels。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\document_aware_guard_design.py `
  scripts\evaluate_document_aware_guards.py `
  tests\test_document_aware_guard_design.py

pytest -q tests\test_document_aware_guard_design.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 108 passed
```

### 结论

- `candidate_score_gte_60` 仍是当前最好的 runtime-feature-only reference：
  - delta `+0.0008`
  - regressions `1`
  - citation loss `0`
  - gold citation delta `0`
- `score60_or_new_gold` 是最佳离线诊断 guard：
  - delta `+0.0013`
  - regressions `1`
  - gold citation delta `+1`
  - 但它使用 gold-document transition，不能作为 runtime policy。
- `document_risk_v1` 和 `citation_doc_risk_v1` 是最低风险诊断 guard：
  - regressions `0`
  - citation loss `0`
  - gold citation delta `+1`
  - 但 delta 低于 `candidate_score_gte_60`。
- 下一步应把 diagnostic guard 转换成 runtime-available proxy search，而不是直接接入 runtime。

### 我学到的

- 离线 gold/citation signal 很适合解释方向，但必须和 runtime feature 分开，否则容易造成数据泄漏。
- `DEV_Q119` 证明低 candidate score 也可能是有效 new-gold 替换；
- `DEV_Q261` 证明 score >= 60 仍可能存在 rank 4 new-non-gold regression；
- 选择 guard 时需要明确目标：更高 delta 和更低 regression 不一定同时最优。
- Stage 42 的价值不是“找到可上线策略”，而是把下一步 runtime proxy 搜索的目标刻画清楚。

### 下一步

- 做 Stage 43：runtime-available document-risk proxy search。
- 目标：
  1. 不使用 gold/citation label 做决策；
  2. 从 runtime features 中寻找 document-risk proxy；
  3. 对比 `candidate_score_gte_60`、rank/score 联合 gate、coverage/title-overlap gate；
  4. 重点观察是否能处理 `DEV_Q261`，同时不误杀太多类似 `DEV_Q119` 的有效替换；
  5. 继续只在 train/dev 范围内做离线验证；
  6. 不使用 held-out test set，不改 runtime 默认策略。

## Stage 43 - Runtime-available document-risk proxy search

### 本阶段目标

- 把 Stage 42 的 document/citation 诊断结论转换成 runtime 可获得特征上的 proxy 搜索。
- 决策特征只允许使用 candidate rank、candidate score、document id、query coverage、title overlap、answer/problem signal 等 runtime features。
- gold answer、gold document、citation label 只用于离线评估，不能进入 guard 决策。
- 继续只使用 train/dev，不使用 held-out test set。
- 不改变 runtime 默认策略。

### 新增内容

- 新增 `src/ts_rag_agent/application/runtime_document_risk_proxy_search.py`
  - 复用 train -> dev split validation selection；
  - 校验 Stage 42 报告与当前重算的 `stage36_main`、`candidate_score_gte_60` 指标一致；
  - 比较 7 个 runtime-only guard：
    - `stage36_main`
    - `candidate_score_gte_60`
    - `rank4_score90`
    - `rank5_score90`
    - `score60_or_title3`
    - `title_rescue_rank4_score90`
    - `coverage_preserving_score60`
  - 输出 guard metrics、相对 main/score60 delta、probe case 和 findings。
- 新增 `scripts/evaluate_runtime_document_risk_proxies.py`
  - 可复跑 Stage 43；
  - 输出 JSON；
  - 输出 SVG 可视化。
- 新增 `tests/test_runtime_document_risk_proxy_search.py`
  - 覆盖 runtime-only feature scope；
  - 覆盖 Stage 42 报告一致性校验；
  - 覆盖 SVG 写出。

### 实验命令

```powershell
python scripts\evaluate_runtime_document_risk_proxies.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage42-report artifacts\candidate_reranker_stage42_document_aware_guard_design.json `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage43_runtime_document_risk_proxy_search.json `
  --visualization-dir artifacts\candidate_reranker_stage43_runtime_proxy_visuals
```

### 实验结果

```text
stage36_main:
  F1 0.2791
  delta -0.0014
  replacements 42
  regressions 8
  citation lost 2
  gold citation delta -1

candidate_score_gte_60:
  F1 0.2813
  delta +0.0008
  replacements 29
  regressions 1
  citation lost 0
  gold citation delta 0

rank4_score90:
  F1 0.2806
  delta +0.0001
  replacements 25
  regressions 0
  citation lost 0
  gold citation delta 0
  changed vs score60 4

rank5_score90:
  F1 0.2803
  delta -0.0002
  replacements 26
  regressions 1
  citation lost 0
  gold citation delta 0
  changed vs score60 3

score60_or_title3:
  F1 0.2793
  delta -0.0012
  replacements 38
  regressions 6
  citation lost 2
  gold citation delta -1

title_rescue_rank4_score90:
  F1 0.2790
  delta -0.0015
  replacements 32
  regressions 3
  citation lost 2
  gold citation delta -1
  changed vs score60 11

coverage_preserving_score60:
  F1 0.2805
  delta +0.0000
  replacements 24
  regressions 0
  citation lost 0
  gold citation delta 0
  changed vs score60 5
```

### Probe case 观察

`DEV_Q119`：

```text
candidate_score_gte_60:
  keep top candidate
  delta +0.0000
  gold cited false

title_rescue_rank4_score90:
  replace rank 4, score 13.0
  delta +0.0758
  gold cited true
  citation delta +1
```

解释：`title_rescue_rank4_score90` 能救回 Stage 42 识别出的低分有效替换，但它在全体 dev 上引入了更多回归和 citation loss。

`DEV_Q261`：

```text
candidate_score_gte_60:
  replace rank 4, score 63.6936
  same leading document false
  delta -0.0573

rank4_score90:
  keep top candidate
  delta +0.0000

coverage_preserving_score60:
  keep top candidate
  delta +0.0000
```

解释：`rank4_score90` 和 `coverage_preserving_score60` 能挡住这个 residual regression，但会额外挡掉一些 score60 原本保留的正收益替换。

### 可视化结果

本阶段新增 5 个 SVG：

```text
artifacts/candidate_reranker_stage43_runtime_proxy_visuals/stage43_probe_delta.svg
artifacts/candidate_reranker_stage43_runtime_proxy_visuals/stage43_runtime_proxy_changed_vs_score60.svg
artifacts/candidate_reranker_stage43_runtime_proxy_visuals/stage43_runtime_proxy_citation_loss.svg
artifacts/candidate_reranker_stage43_runtime_proxy_visuals/stage43_runtime_proxy_delta.svg
artifacts/candidate_reranker_stage43_runtime_proxy_visuals/stage43_runtime_proxy_regressions.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage43_runtime_document_risk_proxy_search.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：没有找到同时优于 `candidate_score_gte_60` 的 runtime proxy。
  - `candidate_score_gte_60` 的平均 delta 最高：`+0.0008`；
  - `rank4_score90` 和 `coverage_preserving_score60` 可以把 regression 降到 `0`，但 delta 分别降到 `+0.0001` 和 `+0.0000`。
- 问题 2：title overlap 不能直接替代 gold-document transition。
  - `title_rescue_rank4_score90` 救回了 `DEV_Q119`；
  - 但全局 delta 降到 `-0.0015`，regressions 升到 `3`，citation lost 升到 `2`。
- 问题 3：runtime document id 只能判断是否换文档，不能判断新文档是否 gold。
  - Stage 42 的最佳诊断信息来自 gold/citation label；
  - Stage 43 的 runtime-only proxy 无法无损复现这个信号。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\runtime_document_risk_proxy_search.py `
  scripts\evaluate_runtime_document_risk_proxies.py `
  tests\test_runtime_document_risk_proxy_search.py

pytest -q tests\test_runtime_document_risk_proxy_search.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 111 passed
```

### 结论

- `candidate_score_gte_60` 仍是当前最好的 runtime-only reference。
- `rank4_score90` 和 `coverage_preserving_score60` 是更保守的零回归候选，但收益低于 `candidate_score_gte_60`。
- `title_rescue_rank4_score90` 虽然能处理 `DEV_Q119`，但整体质量更差，不能进入下一步 runtime 默认策略。
- Stage 43 没有使用 held-out test set，也没有改变 runtime 默认行为。

### 我学到的

- Stage 42 的 gold/citation 诊断信号方向是对的，但 runtime proxy 的表达能力不足以直接复现。
- `DEV_Q119` 和 `DEV_Q261` 之间存在真实冲突：救低分有效替换和拦截中分深 rank 换文档回归不是同一个简单规则能同时做好。
- 如果只看 dev 结果继续调阈值，容易把 dev 当成训练集；下一步应把 proxy guard family 放回 train-only cross-validation。
- runtime feature 的价值需要通过分层和交叉验证确认，不能只靠单个 probe case 决策。

### 下一步

- 做 Stage 44：train-only cross-validation for runtime proxy guard selection。
- 目标：
  1. 在 train split 内对 runtime-only guard family 做交叉验证；
  2. 选择规则只看 train-CV aggregate，不看 dev 的最优标签；
  3. 再用 dev holdout 做一次确认；
  4. 继续不使用 held-out test set；
  5. 继续不改变 runtime 默认策略。

## Stage 44 - Train-only CV selection for runtime proxy guards

### 本阶段目标

- 把 Stage 43 的 runtime-only guard family 放回 train split 内做 grouped cross-validation。
- guard 选择只看 train-CV aggregate，不看 dev 的最佳结果。
- dev holdout 只作为确认，并校验它与 Stage 43 冻结报告一致。
- 继续不使用 held-out test set。
- 继续不改变 runtime 默认策略。

### 新增内容

- 更新 `src/ts_rag_agent/application/runtime_document_risk_proxy_search.py`
  - 暴露 `default_runtime_document_risk_guard_specs()`；
  - 暴露 `evaluate_runtime_document_risk_guards_from_main_decisions()`；
  - Stage 43 搜索函数复用这条公共评估路径，避免 Stage 44 复制 guard 逻辑。
- 新增 `src/ts_rag_agent/application/runtime_document_risk_proxy_cv.py`
  - train-only grouped CV 评估 runtime proxy guard family；
  - train-CV 选择 guard；
  - dev holdout 只确认；
  - 校验 Stage 43 report 中的 holdout metrics 与当前重算一致。
- 新增 `scripts/cross_validate_runtime_document_risk_proxies.py`
  - 可复跑 Stage 44；
  - 输出 JSON 和 SVG。
- 新增 `tests/test_runtime_document_risk_proxy_cv.py`
  - 覆盖 train-CV 选择；
  - 覆盖 Stage 43 report mismatch 失败；
  - 覆盖 SVG 写出。

### 实验命令

```powershell
python scripts\cross_validate_runtime_document_risk_proxies.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage43-report artifacts\candidate_reranker_stage43_runtime_document_risk_proxy_search.json `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --train-fold-count 5 `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage44_runtime_proxy_train_cv_selection.json `
  --visualization-dir artifacts\candidate_reranker_stage44_runtime_proxy_cv_visuals
```

### Train-CV 结果

按 Stage 44 当前选择规则：

```text
selection metric:
  max train-CV average_delta_vs_baseline,
  then fewer regressions,
  then fewer citation losses,
  then higher gold citation delta,
  then higher policy F1,
  then label
```

train-CV 排序核心结果：

```text
stage36_main:
  delta +0.0011
  regressions 14
  citation lost 1
  gold citation delta +2

score60_or_title3:
  delta +0.0009
  regressions 11
  citation lost 1
  gold citation delta +2

title_rescue_rank4_score90:
  delta +0.0008
  regressions 3
  citation lost 0
  gold citation delta 0

candidate_score_gte_60:
  delta +0.0008
  regressions 8
  citation lost 1
  gold citation delta +2

rank4_score90:
  delta +0.0007
  regressions 2
  citation lost 0
  gold citation delta 0

coverage_preserving_score60:
  delta +0.0006
  regressions 0
  citation lost 0
  gold citation delta +1

rank5_score90:
  delta +0.0005
  regressions 8
  citation lost 1
  gold citation delta 0
```

Stage 44 的纯 train-CV delta 选择结果：

```text
selected_guard_label: stage36_main
train-CV delta: +0.0011
train-CV regressions: 14
train-CV citation lost: 1
```

### Dev holdout 确认

dev holdout 排序核心结果：

```text
candidate_score_gte_60:
  delta +0.0008
  regressions 1
  citation lost 0
  gold citation delta 0

rank4_score90:
  delta +0.0001
  regressions 0
  citation lost 0
  gold citation delta 0

coverage_preserving_score60:
  delta +0.0000
  regressions 0
  citation lost 0
  gold citation delta 0

rank5_score90:
  delta -0.0002
  regressions 1
  citation lost 0
  gold citation delta 0

score60_or_title3:
  delta -0.0012
  regressions 6
  citation lost 2
  gold citation delta -1

stage36_main:
  delta -0.0014
  regressions 8
  citation lost 2
  gold citation delta -1

title_rescue_rank4_score90:
  delta -0.0015
  regressions 3
  citation lost 2
  gold citation delta -1
```

结论：train-CV 纯 delta 选择的 `stage36_main` 在 dev holdout 上表现很差：

```text
stage36_main holdout:
  delta -0.0014
  regressions 8
  citation lost 2
  gold citation delta -1
```

### 可视化结果

本阶段新增 5 个 SVG：

```text
artifacts/candidate_reranker_stage44_runtime_proxy_cv_visuals/stage44_train_cv_runtime_proxy_delta.svg
artifacts/candidate_reranker_stage44_runtime_proxy_cv_visuals/stage44_train_cv_runtime_proxy_regressions.svg
artifacts/candidate_reranker_stage44_runtime_proxy_cv_visuals/stage44_holdout_runtime_proxy_delta.svg
artifacts/candidate_reranker_stage44_runtime_proxy_cv_visuals/stage44_holdout_runtime_proxy_regressions.svg
artifacts/candidate_reranker_stage44_runtime_proxy_cv_visuals/stage44_selected_guard_train_vs_holdout.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage44_runtime_proxy_train_cv_selection.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：纯 `average_delta_vs_baseline` 的 train-CV 选择不稳定。
  - train-CV 选出 `stage36_main`；
  - dev holdout 最好的是 `candidate_score_gte_60`；
  - 二者方向相反，说明 train-CV 纯收益目标没有很好反映 dev 风险。
- 问题 2：train-CV 中高收益通常伴随较高 regression。
  - `stage36_main` delta 最高，但 regressions 是 `14`；
  - `score60_or_title3` delta 第二，但 regressions 是 `11`；
  - `coverage_preserving_score60` regressions 是 `0`，但 delta 只有 `+0.0006`。
- 问题 3：当前 selection metric 没有把 risk constraint 放在主目标。
  - regression 和 citation loss 只是 tie-break；
  - 当 delta 有细微差距时，高风险策略仍会胜出。

### 修正与处理

- 没有把 Stage 44 选择结果接入 runtime。
- 没有把 `stage36_main` 重新包装成推荐策略。
- 把 Stage 44 结论定位为：train-CV 已接入，但纯收益选择目标不够，需要下一步做 risk-aware objective。
- dev holdout 仍只用于确认和解释，不反向修改本阶段已声明的 train-CV 选择结果。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\runtime_document_risk_proxy_search.py `
  src\ts_rag_agent\application\runtime_document_risk_proxy_cv.py `
  scripts\cross_validate_runtime_document_risk_proxies.py `
  tests\test_runtime_document_risk_proxy_search.py `
  tests\test_runtime_document_risk_proxy_cv.py

pytest -q tests\test_runtime_document_risk_proxy_search.py `
  tests\test_runtime_document_risk_proxy_cv.py
```

结果：

```text
ruff: passed
pytest: 6 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 114 passed
```

### 结论

- Stage 44 已经把交叉验证接入 runtime-only proxy guard family。
- 但纯 train-CV delta selection 选出了 `stage36_main`，它在 dev 上表现差，因此不能作为 runtime 推荐。
- `candidate_score_gte_60` 仍是 dev holdout 最好结果，但它不是本阶段 train-CV 纯收益选择的产物。
- `coverage_preserving_score60` 和 `rank4_score90` 仍是零回归候选，但收益较低。
- 本阶段没有使用 held-out test set，也没有改变 runtime 默认行为。

### 我学到的

- “使用交叉验证”不等于“选择目标已经正确”。
- 当可用样本不大、delta 差距很小时，纯平均收益目标会偏向高替换率、高 regression 的策略。
- guard 选择应该把风险约束前置，而不是只作为收益相同后的 tie-break。
- dev holdout 的价值在这里不是调参，而是暴露 train-CV 选择目标与实际风险之间的不一致。

### 下一步

- 做 Stage 45：risk-aware train-CV objective for runtime proxy guards。
- 目标：
  1. 在 train-CV 内定义明确的风险约束，例如最大 regression、最大 citation loss、gold citation delta 下限；
  2. 在满足风险约束的候选中再最大化 delta；
  3. 同时保留 `candidate_score_gte_60`、`rank4_score90`、`coverage_preserving_score60` 作为固定对照；
  4. dev holdout 仍只做确认，不用于选择规则调参；
  5. 继续不使用 held-out test set；
  6. 继续不改变 runtime 默认策略。

## Stage 45 - Risk-aware train-CV objective for runtime proxy guards

### 本阶段目标

- 在 Stage 44 的 train-CV guard family 上，把风险约束前置。
- 先筛掉不满足风险约束的 guard，再在可行 guard 中最大化 train-CV delta。
- dev holdout 继续只做确认，不参与选择。
- 继续不使用 held-out test set。
- 继续不改变 runtime 默认策略。

### 新增内容

- 新增 `src/ts_rag_agent/application/runtime_document_risk_risk_aware_selection.py`
  - 复用 Stage 44 的 train-CV / dev holdout guard evaluations；
  - 校验 Stage 44 report 与当前重算结果一致；
  - 实现 4 组 risk-aware objective；
  - 输出每组 objective 的可行 guard、选中 guard、train-CV metrics 和 dev holdout metrics；
  - 输出 SVG 可视化。
- 新增 `scripts/select_risk_aware_runtime_document_risk_proxy.py`
  - 可复跑 Stage 45；
  - 输入 Stage 43 和 Stage 44 冻结报告；
  - 输出 JSON 和 SVG。
- 新增 `tests/test_runtime_document_risk_risk_aware_selection.py`
  - 覆盖 primary objective 的 train-CV 风险约束；
  - 覆盖 Stage 44 report mismatch 失败；
  - 覆盖 SVG 写出。

### 实验命令

```powershell
python scripts\select_risk_aware_runtime_document_risk_proxy.py `
  --dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage43-report artifacts\candidate_reranker_stage43_runtime_document_risk_proxy_search.json `
  --stage44-report artifacts\candidate_reranker_stage44_runtime_proxy_train_cv_selection.json `
  --model logistic_best_candidate `
  --train-split train `
  --evaluation-split dev `
  --train-fold-count 5 `
  --max-answer-candidates 3 `
  --output artifacts\candidate_reranker_stage45_runtime_proxy_risk_aware_selection.json `
  --visualization-dir artifacts\candidate_reranker_stage45_runtime_proxy_risk_aware_visuals
```

### Objective 设计

本阶段比较 4 组 objective：

```text
score60_risk_parity:
  train-CV regressions <= candidate_score_gte_60
  train-CV citation_lost <= candidate_score_gte_60
  train-CV gold_citation_delta >= candidate_score_gte_60
  then maximize train-CV delta

no_citation_loss:
  train-CV citation_lost <= 0
  train-CV gold_citation_delta >= 0
  then maximize train-CV delta

low_regression_no_citation_loss:
  train-CV regressions <= 2
  train-CV citation_lost <= 0
  train-CV gold_citation_delta >= 0
  then maximize train-CV delta

zero_regression_positive_gold:
  train-CV regressions <= 0
  train-CV citation_lost <= 0
  train-CV gold_citation_delta >= 1
  then maximize train-CV delta
```

Primary objective 是 `score60_risk_parity`，原因：

- `candidate_score_gte_60` 是 Stage 43/44 里最稳定的 runtime-only reference；
- Stage 44 已证明纯 delta 会选出高风险的 `stage36_main`；
- 因此 Stage 45 先要求“不比 score60 风险更差”，再最大化 delta。

### 实验结果

`score60_risk_parity`：

```text
selected: candidate_score_gte_60
constraint: regressions <= 8; citation_lost <= 1; gold_citation_delta >= 2

train-CV:
  delta +0.0008
  regressions 8
  citation lost 1
  gold citation delta +2

dev holdout:
  delta +0.0008
  regressions 1
  citation lost 0
  gold citation delta 0
```

`no_citation_loss`：

```text
selected: title_rescue_rank4_score90
constraint: citation_lost <= 0; gold_citation_delta >= 0

train-CV:
  delta +0.0008
  regressions 3
  citation lost 0
  gold citation delta 0

dev holdout:
  delta -0.0015
  regressions 3
  citation lost 2
  gold citation delta -1
```

`low_regression_no_citation_loss`：

```text
selected: rank4_score90
constraint: regressions <= 2; citation_lost <= 0; gold_citation_delta >= 0

train-CV:
  delta +0.0007
  regressions 2
  citation lost 0
  gold citation delta 0

dev holdout:
  delta +0.0001
  regressions 0
  citation lost 0
  gold citation delta 0
```

`zero_regression_positive_gold`：

```text
selected: coverage_preserving_score60
constraint: regressions <= 0; citation_lost <= 0; gold_citation_delta >= 1

train-CV:
  delta +0.0006
  regressions 0
  citation lost 0
  gold citation delta +1

dev holdout:
  delta +0.0000
  regressions 0
  citation lost 0
  gold citation delta 0
```

### 可视化结果

本阶段新增 5 个 SVG：

```text
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_visuals/stage45_objective_feasible_guard_count.svg
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_visuals/stage45_objective_holdout_delta.svg
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_visuals/stage45_objective_holdout_regressions.svg
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_visuals/stage45_objective_train_delta.svg
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_visuals/stage45_train_guard_delta_under_risk_view.svg
```

完整 JSON artifact：

```text
artifacts/candidate_reranker_stage45_runtime_proxy_risk_aware_selection.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：只约束 train-CV citation loss 不够。
  - `no_citation_loss` 在 train-CV 上 citation loss 为 `0`；
  - 但它选择的 `title_rescue_rank4_score90` 在 dev 上 citation lost 变成 `2`，delta 变成 `-0.0015`。
- 问题 2：越严格的风险约束，delta 越低。
  - `rank4_score90` 和 `coverage_preserving_score60` 都能在 dev 上做到 `0` regression；
  - 但它们的 dev delta 只有 `+0.0001` 和 `+0.0000`。
- 问题 3：Stage 45 的 primary objective 不是凭 dev 选出来的。
  - `candidate_score_gte_60` 是在 train-CV 风险不劣于自身 reference 的约束下被选中；
  - dev holdout 只是确认它没有出现 Stage 44 那种反向失效。

### 修正与处理

- 没有改变 runtime 默认策略。
- 没有使用 held-out test set。
- 没有把 `title_rescue_rank4_score90` 作为推荐策略，即使它满足 train-CV no-citation-loss objective。
- 把 `candidate_score_gte_60` 提升为“可进入 runtime optional end-to-end 实验”的候选，而不是直接作为默认策略。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\runtime_document_risk_proxy_search.py `
  src\ts_rag_agent\application\runtime_document_risk_proxy_cv.py `
  src\ts_rag_agent\application\runtime_document_risk_risk_aware_selection.py `
  scripts\cross_validate_runtime_document_risk_proxies.py `
  scripts\select_risk_aware_runtime_document_risk_proxy.py `
  tests\test_runtime_document_risk_proxy_search.py `
  tests\test_runtime_document_risk_proxy_cv.py `
  tests\test_runtime_document_risk_risk_aware_selection.py

pytest -q tests\test_runtime_document_risk_proxy_search.py `
  tests\test_runtime_document_risk_proxy_cv.py `
  tests\test_runtime_document_risk_risk_aware_selection.py
```

结果：

```text
ruff: passed
pytest: 9 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 117 passed
```

### 结论

- Stage 45 的 primary risk-aware objective 选择了 `candidate_score_gte_60`。
- 这次选择不是 dev 反向调参，而是 train-CV risk parity objective 的结果。
- dev holdout 确认 `candidate_score_gte_60` 仍是当前最稳的 runtime-only candidate：
  - delta `+0.0008`
  - regressions `1`
  - citation lost `0`
- 更严格的零回归策略可以降低风险，但收益几乎归零。
- 本阶段仍然只是离线选择与确认，没有改 runtime 默认行为。

### 我学到的

- 风险约束需要覆盖 regression、citation loss 和 gold citation delta，单看 citation loss 不够。
- `title_rescue_rank4_score90` 的失败说明 train-CV 的单一风险维度可能在 dev 上迁移不稳。
- `candidate_score_gte_60` 的价值不是最高 train-CV delta，而是风险/收益平衡稳定。
- 下一步如果要继续推进，应该进入可选 runtime end-to-end，而不是继续只在离线 proxy 上调阈值。

### 下一步

- 做 Stage 46：把 `candidate_score_gte_60` 作为可选 runtime guard 做 end-to-end verified RAG 实验。
- 目标：
  1. 只增加可选参数或实验入口，不改变默认 runtime；
  2. 用完整 verified RAG end-to-end 指标确认是否超过当前 baseline；
  3. 对比 Stage 18 / 当前默认策略的 F1、gold citation、changed answers；
  4. 继续不使用 held-out test set；
  5. 如果端到端没有通过，不进入默认策略。

## Stage 46 - Candidate-Score Guarded Runtime Reranker End-to-End

日期：2026-07-14

### 目标

- 把 Stage 45 选出的 `candidate_score_gte_60` 做成可选 runtime answer-composition policy。
- 用完整 verified RAG end-to-end 流程对比当前 `top_k` baseline。
- 继续只使用 `dev` 和既有 train/dev candidate-reranker dataset，不使用 held-out test set。
- 生成可复查的 comparison JSON 和 SVG 可视化。
- 不改变默认 runtime 策略。

### 实现内容

- 新增 `CandidateScoreGuardedRerankerCompositionPolicy`：
  - 训练阶段使用 Stage 31 candidate-reranker dataset 的 `train` split；
  - runtime 阶段只用 `runtime_features`，不读取 gold answer 或 gold document label；
  - 采用 Stage 36/39 的主约束：
    - model 选中 rank 1 时保持 top candidate；
    - 阻断 `how_to_or_lookup` route；
    - 选中 rank 必须不超过 5；
    - model score margin 必须不低于 `0.05`；
    - 被选 candidate 的 evidence score 必须 `>= 60`。
- 在 `candidate_reranker_dataset.py` 中公开 `build_candidate_runtime_features()`，避免 runtime policy 依赖私有函数。
- 扩展 `scripts/evaluate_verified_rag.py`：
  - 新增 `--composition-policy candidate-score-guarded-reranker`；
  - 新增 `--candidate-reranker-dataset`；
  - 新增 `--candidate-reranker-model`；
  - 新增 `--candidate-reranker-train-split`；
  - report 中记录 `candidate_reranker` 配置，方便追溯。
- 新增 `scripts/compare_verified_rag_reports.py` 和 `verified_rag_report_comparison.py`：
  - 对比 metrics delta；
  - 用完整 samples 精确计算 gold citation count；
  - 统计 changed answers；
  - 统计 answerable verified F1 improved/regressed/tied cases；
  - 生成 SVG 可视化。

### 实验命令

baseline：

```powershell
python scripts\evaluate_verified_rag.py `
  --split dev `
  --retrieval-top-k 5 `
  --max-sentences 3 `
  --min-sentence-score 2.0 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --composition-policy top-k `
  --min-evidence-score 15.0 `
  --max-citation-rank 3 `
  --min-citations 1 `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json
```

candidate-score guarded reranker：

```powershell
python scripts\evaluate_verified_rag.py `
  --split dev `
  --retrieval-top-k 5 `
  --max-sentences 3 `
  --min-sentence-score 2.0 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --composition-policy candidate-score-guarded-reranker `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --candidate-reranker-model logistic_best_candidate `
  --candidate-reranker-train-split train `
  --min-evidence-score 15.0 `
  --max-citation-rank 3 `
  --min-citations 1 `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_candidate_score_guarded_reranker_report.json
```

comparison 和可视化：

```powershell
python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_candidate_score_guarded_reranker_report.json `
  --output artifacts\verified_rag_stage46_candidate_score_guarded_comparison.json `
  --visualization-dir artifacts\verified_rag_stage46_candidate_score_guarded_visuals
```

### 实验结果

baseline `top_k`：

```text
original F1: 0.2729
verified F1: 0.2755
original gold citation: 95 / 160
verified gold citation: 95 / 158
verified answerable refusals: 2
verified unanswerable refusals: 4
newly refused: 6
```

candidate-score guarded reranker：

```text
original F1: 0.2737
verified F1: 0.2763
original gold citation: 95 / 160
verified gold citation: 95 / 158
verified answerable refusals: 2
verified unanswerable refusals: 2
newly refused: 4
```

delta：

```text
original F1 delta: +0.0008
verified F1 delta: +0.0008
original gold citation delta: 0
verified gold citation delta: 0
verified generated answerable delta: 0
verified answerable refusal delta: 0
verified unanswerable refusal delta: -2
newly refused delta: -2
```

changed answers：

```text
original changed answers: 60 / 310
original changed answerable: 29 / 160
original changed unanswerable: 31 / 150

verified changed answers: 59 / 310
verified changed answerable: 28 / 160
verified changed unanswerable: 31 / 150
```

answerable verified F1 outcomes：

```text
comparable generated answerable: 158
improved: 4
regressed: 1
changed but tied: 23
average delta over comparable: +0.000807
```

两个 unanswerable 风险样本从 baseline 的拒答变成 candidate 的回答：

```text
DEV_Q083 - How do you configure the IBM HTTP Server 7 to 8.5 on z/OS for Host On-Demand V11?
DEV_Q203 - Why WebSphere Application Server transactions are rolled back with these StaleConnectionExceptions?
```

### 可视化结果

本阶段新增 3 个 SVG：

```text
artifacts/verified_rag_stage46_candidate_score_guarded_visuals/verified_rag_metric_deltas.svg
artifacts/verified_rag_stage46_candidate_score_guarded_visuals/verified_rag_changed_answers.svg
artifacts/verified_rag_stage46_candidate_score_guarded_visuals/verified_rag_answerable_f1_outcomes.svg
```

完整 comparison artifact：

```text
artifacts/verified_rag_stage46_candidate_score_guarded_comparison.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：F1 提升很小，但 changed answers 很多。
  - verified F1 只提升 `+0.0008`；
  - 但 verified changed answers 达到 `59 / 310`；
  - 其中 `31` 个是 unanswerable 问题。
- 问题 2：unanswerable refusal 下降了。
  - baseline verified unanswerable refusals 是 `4`；
  - candidate-score guarded reranker 是 `2`；
  - 这表示策略在两个 unanswerable 样本上变得更不保守。
- 问题 3：gold citation 没有损失，但也没有增加。
  - verified gold citation 仍是 `95 / 158`；
  - citation gained 和 citation lost 都是空列表。

### 修正与处理

- 没有改变默认 runtime。
- 没有使用 held-out test set。
- 没有把 candidate-score guarded reranker 设置为默认策略。
- 把本阶段定位为“可选 runtime 实验通过 F1/citation 基础门槛，但默认化前必须继续做风险分析”。
- 新增 comparison 工具，避免后续只靠人工读 report。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\verified_rag_report_comparison.py `
  scripts\compare_verified_rag_reports.py `
  tests\test_verified_rag_report_comparison.py

pytest -q tests\test_verified_rag_report_comparison.py `
  tests\test_candidate_score_guarded_composition_policy.py `
  tests\test_candidate_reranker_dataset.py
```

结果：

```text
ruff: passed
pytest: 7 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 121 passed
```

### 结论

- Stage 46 端到端 verified RAG 结果显示 candidate-score guarded reranker 在 dev 上：
  - verified F1 小幅提升 `+0.0008`；
  - verified gold citation 不变；
  - answerable refusal 不变；
  - 但 unanswerable refusal 少了 `2`；
  - changed answers 达到 `59 / 310`。
- 因此它可以保留为可选实验入口，但不能进入默认 runtime。
- 当前主要风险不是 citation loss，而是 answer rewrite 面太宽和 unanswerable 保守性下降。

### 我学到的

- 离线 candidate-level gain 进入 end-to-end 后会被 verifier 大幅压缩，最终 F1 只剩 `+0.0008`。
- gold citation 不掉不代表策略就安全，因为 unanswerable refusal 可能下降。
- changed answer count 是 runtime 默认化前必须纳入的风险指标。
- 完整 sample report 很重要；如果只看 aggregate metrics，会漏掉 `DEV_Q083` 和 `DEV_Q203` 这类拒答退化。

### 下一步

- 做 Stage 47：专门分析 Stage 46 的 changed-answer 和 unanswerable-risk cases。
- 目标：
  1. 聚焦 `DEV_Q083`、`DEV_Q203` 两个 unanswerable refusal regression；
  2. 分析 59 个 verified changed answers 的 route、rank、score、citation 分布；
  3. 判断是否能设计更窄的 runtime gate，降低 changed-answer 面和 unanswerable 风险；
  4. 仍然不使用 held-out test set；
  5. 在没有解决风险前，不把 candidate-score guarded reranker 默认化。

## Stage 47 - Changed-Answer And Unanswerable-Risk Analysis

日期：2026-07-14

### 目标

- 专门分析 Stage 46 的 `59 / 310` 个 verified changed answers。
- 聚焦两个 unanswerable refusal regression：
  - `DEV_Q083`
  - `DEV_Q203`
- 统计 changed answers 的 route、outcome、citation rank、evidence score 分布。
- 生成可复查 JSON 和 SVG 可视化。
- 继续不使用 held-out test set，不改变默认 runtime。

### 实现内容

- 新增 `verified_rag_changed_answer_risk_analysis.py`：
  - 读取 baseline/candidate verified RAG report；
  - 读取 PrimeQA question metadata；
  - 用现有 `classify_question_route()` 计算 route；
  - 逐题识别 verified answer 是否变化；
  - 统计 answerable F1 improved/regressed/tied；
  - 统计 unanswerable refusal regression；
  - 统计 candidate citations 的 best/worst rank、score bucket、是否含 out-of-rank citation。
- 新增 `scripts/analyze_verified_rag_changed_answer_risk.py`：
  - 输出 Stage 47 JSON；
  - 输出 SVG 可视化。
- 新增 `svg_charts.py`：
  - 把 Stage 46 comparison 模块里的横向柱图渲染抽成共享工具；
  - Stage 46 comparison 和 Stage 47 risk analysis 共同复用。

### 实验命令

```powershell
python scripts\analyze_verified_rag_changed_answer_risk.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_candidate_score_guarded_reranker_report.json `
  --output artifacts\verified_rag_stage47_changed_answer_risk_analysis.json `
  --visualization-dir artifacts\verified_rag_stage47_changed_answer_risk_visuals
```

### 分析结果

总体：

```text
changed verified answers: 59
changed answerable: 28
changed unanswerable: 31
unanswerable refusal regressions: 2
answerable improved: 4
answerable regressed: 1
answerable tied changed: 23
candidate has out-of-rank citation: 44
unanswerable regressions with out-of-rank citation: 2
```

route distribution：

```text
other: 24
error_or_log: 17
install_upgrade_config: 12
limitation_or_restriction: 3
security_bulletin_vulnerability_detail: 3
```

outcome distribution：

```text
answerable_f1_improved: 4
answerable_f1_regressed: 1
answerable_f1_tied_changed: 23
unanswerable_answer_changed: 29
unanswerable_refusal_regression: 2
```

candidate citation rank：

```text
best_rank:
  rank_1: 51
  rank_2: 6
  rank_3: 2

worst_rank:
  rank_1: 1
  rank_2: 5
  rank_3: 9
  rank_4_5: 44
```

out-of-rank by outcome：

```text
answerable_f1_improved: 2 / 4
answerable_f1_regressed: 1 / 1
answerable_f1_tied_changed: 19 / 23
unanswerable_answer_changed: 20 / 29
unanswerable_refusal_regression: 2 / 2
```

evidence score：

```text
max_evidence_score:
  score_60_99: 13
  score_gte_100: 46

min_evidence_score:
  score_15_59: 5
  score_60_99: 42
  score_gte_100: 12
```

两个 unanswerable refusal regression：

```text
DEV_Q083
route: install_upgrade_config
baseline reasons: citation_rank_too_low
candidate reasons: verified
candidate citation ranks: 1, 4, 4
candidate evidence scores: 72.408, 101.9876, 99.4177

DEV_Q203
route: error_or_log
baseline reasons: citation_rank_too_low
candidate reasons: verified
candidate citation ranks: 3, 4, 5
candidate evidence scores: 84.1343, 139.5853, 128.1817
```

观察性 rank-contained gate 影响：

```text
如果要求 candidate answer 的所有 citation retrieval_rank <= verifier max rank 3：
  会挡住 changed cases: 44 / 59
  会挡住 unanswerable refusal regressions: 2 / 2
```

注意：这只是基于已保存 report 输出的观察性分析，不是已验证 runtime gate。

### 可视化结果

本阶段新增 4 个 SVG：

```text
artifacts/verified_rag_stage47_changed_answer_risk_visuals/stage47_changed_by_route.svg
artifacts/verified_rag_stage47_changed_answer_risk_visuals/stage47_changed_by_outcome.svg
artifacts/verified_rag_stage47_changed_answer_risk_visuals/stage47_out_of_rank_by_outcome.svg
artifacts/verified_rag_stage47_changed_answer_risk_visuals/stage47_candidate_worst_rank.svg
```

完整 JSON artifact：

```text
artifacts/verified_rag_stage47_changed_answer_risk_analysis.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：风险不是单一 route 导致。
  - `other`、`error_or_log`、`install_upgrade_config` 都有明显 changed answers；
  - 两个 unanswerable refusal regression 分别来自 `install_upgrade_config` 和 `error_or_log`。
- 问题 2：candidate 通过 verifier 的方式过于宽松。
  - verifier 当前只要求 best citation rank `<= 3`；
  - candidate answer 可以同时携带 rank4/rank5 citations；
  - Stage 47 中 `44 / 59` 个 changed candidate answers 含 out-of-rank citation。
- 问题 3：out-of-rank 信号能覆盖两个拒答退化，但会挡掉很多其它 changed cases。
  - 它会挡住两个 unanswerable refusal regression；
  - 但也会挡住 `2 / 4` 个 answerable improved cases；
  - 所以不能直接声明它就是默认策略。

### 修正与处理

- 没有改变默认 runtime。
- 没有使用 held-out test set。
- 没有把 rank-contained 条件直接加入默认策略。
- 将 rank-contained 条件记录为下一阶段需要真实端到端验证的候选 guard。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\svg_charts.py `
  src\ts_rag_agent\application\verified_rag_report_comparison.py `
  src\ts_rag_agent\application\verified_rag_changed_answer_risk_analysis.py `
  scripts\analyze_verified_rag_changed_answer_risk.py `
  tests\test_verified_rag_report_comparison.py `
  tests\test_verified_rag_changed_answer_risk_analysis.py

pytest -q tests\test_verified_rag_report_comparison.py `
  tests\test_verified_rag_changed_answer_risk_analysis.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 123 passed
```

### 结论

- Stage 46 的主要风险不是 gold citation loss，而是 candidate answer 太容易通过 verifier。
- 两个 unanswerable refusal regression 都符合相同模式：
  - baseline 因 `citation_rank_too_low` 拒答；
  - candidate 加入一个 rank<=3 citation 后通过；
  - 但最终 answer 仍携带 rank4/rank5 citations。
- `all citations rank <= 3` 是一个强信号：
  - 能覆盖 `2 / 2` 个 unanswerable refusal regression；
  - 但会影响 `44 / 59` 个 changed cases；
  - 因此必须进入下一阶段 end-to-end 验证，不能直接默认化。

### 我学到的

- 只看 best citation rank 会隐藏“答案中仍混入低 rank 文档”的风险。
- changed answer 风险主要来自 answer composition 放宽，而不是 route classifier 某一类失效。
- 对 unanswerable 问题来说，“通过 verifier”本身也可能是坏事，因为 gold label 是不可回答。
- 可视化 route/outcome/rank 分布比单看两个样本更能解释为什么 Stage 46 不能默认化。

### 下一步

- 做 Stage 48：把 rank-contained 条件做成可选 runtime guard 并重跑 end-to-end verified RAG。
- 目标：
  1. candidate-score guarded reranker 只有在最终 selected citations 全部 `retrieval_rank <= 3` 时才允许改写；
  2. 对比 Stage 46 candidate 和 Stage 46 top-k baseline；
  3. 重点检查是否消除 `DEV_Q083`、`DEV_Q203` 两个 unanswerable refusal regression；
  4. 同时量化 answerable improved/regressed 的损失；
  5. 仍然不使用 held-out test set；
  6. 不改变默认 runtime。

## Stage 48 - Rank-Contained Runtime Guard End-to-End

日期：2026-07-14

### 目标

- 把 Stage 47 的观察性 rank-contained 条件做成可选 runtime guard。
- 新 guard 只在最终 selected citations 全部满足 `retrieval_rank <= max_citation_rank` 时才允许 candidate reranker 改写。
- 用完整 verified RAG end-to-end 对比：
  - Stage 46 top-k baseline；
  - Stage 46 candidate-score guarded reranker；
  - Stage 48 rank-contained guarded reranker。
- 重点确认 `DEV_Q083`、`DEV_Q203` 两个 unanswerable refusal regression 是否消失。
- 继续不使用 held-out test set，不改变默认 runtime。

### 实现内容

- 扩展 `CandidateScoreGuardedRerankerCompositionPolicy`：
  - 新增 `rank_contained_max_retrieval_rank` 参数；
  - 保持原 `candidate_score_gte_60_guarded_reranker` 行为不变；
  - 新增 rank-contained 模式的 policy name：
    `candidate_score_gte_60_rank_contained_guarded_reranker`；
  - 当 candidate reranker 选中的最终候选集合里任一 citation 的 retrieval rank 超过阈值时，拒绝改写并保持 top-k。
- 扩展 `scripts/evaluate_verified_rag.py`：
  - 新增 policy alias：`candidate-score-rank-contained-reranker`；
  - rank-contained 阈值直接使用同次评估的 `--max-citation-rank`；
  - report 中记录：
    - `runtime_guard`
    - `rank_contained_max_retrieval_rank`
- 新增单测覆盖 rank-contained block 行为。

### 实验命令

rank-contained runtime：

```powershell
python scripts\evaluate_verified_rag.py `
  --split dev `
  --retrieval-top-k 5 `
  --max-sentences 3 `
  --min-sentence-score 2.0 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --composition-policy candidate-score-rank-contained-reranker `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --candidate-reranker-model logistic_best_candidate `
  --candidate-reranker-train-split train `
  --min-evidence-score 15.0 `
  --max-citation-rank 3 `
  --min-citations 1 `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json
```

对比 top-k：

```powershell
python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage48_rank_contained_vs_topk_comparison.json `
  --visualization-dir artifacts\verified_rag_stage48_rank_contained_vs_topk_visuals
```

对比 Stage 46 candidate：

```powershell
python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_candidate_score_guarded_reranker_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage48_rank_contained_vs_stage46_candidate_comparison.json `
  --visualization-dir artifacts\verified_rag_stage48_rank_contained_vs_stage46_candidate_visuals
```

changed-answer risk：

```powershell
python scripts\analyze_verified_rag_changed_answer_risk.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage48_rank_contained_changed_answer_risk_analysis.json `
  --visualization-dir artifacts\verified_rag_stage48_rank_contained_changed_answer_risk_visuals
```

### 实验结果

Stage 46 top-k baseline：

```text
verified F1: 0.2755
verified gold citation: 95 / 158
verified answerable refusals: 2
verified unanswerable refusals: 4
newly refused: 6
DEV_Q083: refused, citation_rank_too_low
DEV_Q203: refused, citation_rank_too_low
```

Stage 46 candidate-score guarded reranker：

```text
verified F1: 0.2763
verified gold citation: 95 / 158
verified answerable refusals: 2
verified unanswerable refusals: 2
newly refused: 4
DEV_Q083: verified
DEV_Q203: verified
```

Stage 48 rank-contained guarded reranker：

```text
verified F1: 0.2763
verified gold citation: 95 / 158
verified answerable refusals: 2
verified unanswerable refusals: 4
newly refused: 6
DEV_Q083: refused, citation_rank_too_low
DEV_Q203: refused, citation_rank_too_low
```

相对 top-k：

```text
verified F1 delta: +0.0008
verified gold citation delta: 0
verified generated answerable delta: 0
verified answerable refusal delta: 0
verified unanswerable refusal delta: 0
newly refused delta: 0
changed verified answers: 15 / 310
answerable F1 outcomes:
  improved: 2
  regressed: 0
  changed but tied: 4
```

相对 Stage 46 candidate：

```text
verified F1 delta: 0.0000
verified gold citation delta: 0
verified generated answerable delta: 0
verified answerable refusal delta: 0
verified unanswerable refusal delta: +2
newly refused delta: +2
changed verified answers: 44 / 310
```

Stage 48 changed-answer risk：

```text
changed verified answers: 15
changed answerable: 6
changed unanswerable: 9
unanswerable refusal regressions: 0
answerable improved: 2
answerable regressed: 0
answerable tied changed: 4
candidate has out-of-rank citation: 0
```

### 可视化结果

本阶段新增可视化目录：

```text
artifacts/verified_rag_stage48_rank_contained_vs_topk_visuals
artifacts/verified_rag_stage48_rank_contained_vs_stage46_candidate_visuals
artifacts/verified_rag_stage48_rank_contained_changed_answer_risk_visuals
```

完整 JSON artifacts：

```text
artifacts/verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json
artifacts/verified_rag_stage48_rank_contained_vs_topk_comparison.json
artifacts/verified_rag_stage48_rank_contained_vs_stage46_candidate_comparison.json
artifacts/verified_rag_stage48_rank_contained_changed_answer_risk_analysis.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- Stage 48 修掉了 Stage 46 的两个 unanswerable refusal regression。
  - `DEV_Q083` 和 `DEV_Q203` 都回到拒答；
  - 拒答原因仍是 `citation_rank_too_low`。
- rank-contained guard 大幅降低 changed answer 面。
  - Stage 46 candidate vs top-k：`59 / 310`；
  - Stage 48 rank-contained vs top-k：`15 / 310`。
- 当前 dev 结果没有看到 F1 损失。
  - Stage 48 verified F1 仍是 `0.2763`；
  - 相对 top-k 仍是 `+0.0008`；
  - 相对 Stage 46 candidate 是 `0.0000`。

### 修正与处理

- 没有改变默认 runtime。
- 没有使用 held-out test set。
- 没有把 rank-contained 策略默认化。
- 将 rank-contained guarded reranker 保留为可选 runtime policy，并进入下一阶段 train split 稳定性验证。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\candidate_score_guarded_composition_policy.py `
  scripts\evaluate_verified_rag.py `
  tests\test_candidate_score_guarded_composition_policy.py

pytest -q tests\test_candidate_score_guarded_composition_policy.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 124 passed
```

### 结论

- Stage 48 在 dev verified RAG 上达到了本阶段目标：
  - 保留 Stage 46 的 F1 提升；
  - 修复 `DEV_Q083`、`DEV_Q203` 两个 unanswerable refusal regression；
  - gold citation 不损失；
  - changed answers 从 `59` 降到 `15`；
  - candidate out-of-rank citation 从 `44` 降到 `0`。
- 但这仍然只是 dev split 结果。
- 按前面 Stage 23/25 的教训，不能只凭 dev 成功就默认化；必须检查 train split 是否稳定。

### 我学到的

- Stage 47 的观察性信号经 runtime 验证后确实有效：rank-contained guard 能消除这次的两个 unanswerable refusal regression。
- “所有 selected citations rank<=3” 比 route-based block 更直接命中风险来源。
- 更窄 guard 不一定牺牲 aggregate F1；在本次 dev 上反而保留了 Stage 46 的 F1。
- 但 train split 可能暴露新的 route 或 citation trade-off，仍需下一阶段验证。

### 下一步

- 做 Stage 49：在 train split 上验证 rank-contained guarded reranker 的稳定性。
- 目标：
  1. 同参数跑 train top-k baseline；
  2. 同参数跑 train rank-contained guarded reranker；
  3. 对比 F1、gold citation、answerable/unanswerable refusal、changed answers；
  4. 分析 train changed-answer risk；
  5. 仍然不使用 held-out test set；
  6. train 也稳定后，再考虑是否进入默认化评审。

## Stage 49 - Train Split Stability Check For Rank-Contained Guard

日期：2026-07-14

### 目标

- 在 train split 上验证 Stage 48 的 rank-contained guarded reranker 是否稳定。
- 同参数跑：
  - train top-k baseline；
  - train rank-contained guarded reranker。
- 对比 F1、gold citation、answerable/unanswerable refusal、changed answers。
- 分析 train changed-answer risk。
- 继续不使用 held-out test set，不改变默认 runtime。

### 重要边界

- 本阶段的 rank-contained reranker 仍使用 `candidate_reranker_train_split=train` 训练。
- 因此 train split 评估是“训练集上的稳定性/风险检查”，不是 held-out 泛化验证。
- 本阶段不能被解释成测试集通过，也不能直接支持默认化。

### 实验命令

train top-k baseline：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --retrieval-top-k 5 `
  --max-sentences 3 `
  --min-sentence-score 2.0 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --composition-policy top-k `
  --min-evidence-score 15.0 `
  --max-citation-rank 3 `
  --min-citations 1 `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json
```

train rank-contained guarded reranker：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --retrieval-top-k 5 `
  --max-sentences 3 `
  --min-sentence-score 2.0 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --composition-policy candidate-score-rank-contained-reranker `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --candidate-reranker-model logistic_best_candidate `
  --candidate-reranker-train-split train `
  --min-evidence-score 15.0 `
  --max-citation-rank 3 `
  --min-citations 1 `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json
```

comparison：

```powershell
python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage49_train_rank_contained_vs_topk_comparison.json `
  --visualization-dir artifacts\verified_rag_stage49_train_rank_contained_vs_topk_visuals
```

changed-answer risk：

```powershell
python scripts\analyze_verified_rag_changed_answer_risk.py `
  --baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage49_train_rank_contained_changed_answer_risk_analysis.json `
  --visualization-dir artifacts\verified_rag_stage49_train_rank_contained_changed_answer_risk_visuals
```

### 实验结果

train top-k baseline：

```text
verified F1: 0.2557
verified gold citation: 232 / 442
verified gold citation rate: 0.5249
verified answerable refusals: 8
verified unanswerable refusals: 5
newly refused: 13
```

train rank-contained guarded reranker：

```text
verified F1: 0.2565
verified gold citation: 231 / 442
verified gold citation rate: 0.5226
verified answerable refusals: 8
verified unanswerable refusals: 5
newly refused: 13
```

delta：

```text
verified F1 delta: +0.0008
verified gold citation delta: -1
verified gold citation rate delta: -0.0023
verified generated answerable delta: 0
verified answerable refusal delta: 0
verified unanswerable refusal delta: 0
newly refused delta: 0
```

changed-answer summary：

```text
changed verified answers: 39 / 600
changed answerable: 29 / 450
changed unanswerable: 10 / 150
unanswerable refusal regressions: 0
answerable improved: 6
answerable regressed: 2
answerable tied changed: 21
candidate has out-of-rank citation: 0
```

route distribution：

```text
error_or_log: 12
install_upgrade_config: 11
other: 11
security_bulletin_vulnerability_detail: 5
```

answerable F1 improved cases：

```text
TRAIN_Q517: +0.204784
TRAIN_Q353: +0.107261
TRAIN_Q027: +0.074552
TRAIN_Q515: +0.029583
TRAIN_Q124: +0.026721
TRAIN_Q219: +0.017297
```

answerable F1 regressed cases：

```text
TRAIN_Q310: -0.079820
TRAIN_Q496: -0.011322
```

gold citation loss：

```text
verified citation gained: none
verified citation lost: TRAIN_Q496
```

`TRAIN_Q496` detail：

```text
question:
Why am I getting an SSL Key Exception (RSA premaster secret error) when trying to create a syndication pair?

gold doc:
swg21663373

baseline citations:
swg21587102 rank 1
swg21587102 rank 1
swg21663373 rank 4

rank-contained candidate citations:
swg21971637 rank 2
swg21587102 rank 1
swg21587102 rank 1
```

说明：Stage 48 的 rank-contained guard 解决了 out-of-rank citation 风险，但 `TRAIN_Q496` 暴露了另一类风险：同主题、rank 更高的非 gold 文档替换掉了原本的 gold doc。

### 可视化结果

本阶段新增可视化目录：

```text
artifacts/verified_rag_stage49_train_rank_contained_vs_topk_visuals
artifacts/verified_rag_stage49_train_rank_contained_changed_answer_risk_visuals
```

完整 JSON artifacts：

```text
artifacts/verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json
artifacts/verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json
artifacts/verified_rag_stage49_train_rank_contained_vs_topk_comparison.json
artifacts/verified_rag_stage49_train_rank_contained_changed_answer_risk_analysis.json
```

说明：以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：train split 上仍有 F1 收益，但 gold citation 不稳定。
  - verified F1 提升 `+0.0008`；
  - verified gold citation 损失 `1` 个。
- 问题 2：rank-contained guard 只约束 retrieval rank，不保证 gold document preservation。
  - `TRAIN_Q496` 中 baseline 虽然引用了 rank4 gold doc；
  - rank-contained candidate 为了满足 rank<=3，选择了 rank2 的相似 SSLKeyException 文档；
  - 结果 candidate answer 不再引用 gold doc。
- 问题 3：没有 unanswerable refusal regression，但仍存在 answerable regression。
  - unanswerable refusal regressions 为 `0`；
  - answerable F1 regressed 为 `2`；
  - 其中 `TRAIN_Q496` 同时是 F1 regression 和 citation loss。

### 修正与处理

- 没有改变默认 runtime。
- 没有使用 held-out test set。
- 没有把 rank-contained guarded reranker 默认化。
- 将 Stage 49 结论定为：rank-contained guard 通过 unanswerable-risk 检查，但未通过 gold-citation stability 检查。

### 测试

本阶段没有修改代码，只新增本地 ignored artifacts 并更新学习日志。

沿用当前代码状态的全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 124 passed
```

### 结论

- Stage 49 在 train split 上确认：
  - rank-contained guard 继续带来 F1 小幅收益；
  - 没有新增 answerable/unanswerable refusal；
  - 没有 out-of-rank citation；
  - 没有 unanswerable refusal regression；
  - 但损失了 1 个 verified gold citation。
- 因此 rank-contained guarded reranker 不能进入默认化评审。
- 下一步应围绕 `TRAIN_Q496` 这种“gold doc 被 rank<=3 相似文档替换”的风险设计 preservation guard。

### 我学到的

- rank-contained guard 解决的是 citation rank 风险，不解决 document identity / gold citation preservation 风险。
- train split 暴露了 dev 没暴露的问题：candidate 可以用 rank 更高的相似非 gold 文档替代原 gold 文档。
- F1 小幅收益不足以覆盖 gold citation loss；默认化门槛必须同时看 citation stability。
- 训练集检查仍有价值，因为它能揭示 policy 的风险模式，但不能替代 held-out test。

### 下一步

- 做 Stage 50：分析并设计 gold-citation preservation guard。
- 目标：
  1. 聚焦 `TRAIN_Q496` 和两个 answerable F1 regression；
  2. 判断是否能用 runtime-only 信号识别“相似非 gold 文档替换 gold doc”的风险；
  3. 评估候选 guard 是否能保持 Stage 48 dev 收益；
  4. 继续不使用 held-out test set；
  5. 在 gold citation 稳定前，不进入默认化评审。

## 2026-07-14 Stage 50：gold-citation preservation guard 离线设计与可视化分析

### 本阶段目标

Stage 49 发现 rank-contained guarded reranker 虽然仍有 train F1 小幅收益，但在 `TRAIN_Q496` 上损失了 1 个 verified gold citation。本阶段目标不是直接修改默认 runtime，而是先做离线 guard 设计：

1. 用 dev/train 既有报告模拟若干 runtime-only preservation guard；
2. 判断能否拦住 `TRAIN_Q496` 这类“baseline 引用了 max citation rank 之外的文档，candidate 因 rank-contained 约束改引 rank 内相似文档”的风险；
3. 给出可视化结果，方便比较 guard 的 F1、gold citation 和 blocked changed-answer 代价；
4. 继续不使用 held-out test set。

### 新增内容

新增离线分析模块：

```text
src/ts_rag_agent/application/gold_citation_preservation_guard_analysis.py
```

新增命令行脚本：

```text
scripts/analyze_gold_citation_preservation_guards.py
```

新增测试：

```text
tests/test_gold_citation_preservation_guard_analysis.py
```

本阶段分析的 guard 候选：

```text
candidate_as_is
preserve_all_baseline_docs
preserve_baseline_out_of_rank_docs
preserve_baseline_out_of_rank_score_gte_60
preserve_baseline_score_gte_80_docs
```

重要边界：

- guard 判定信号只使用 runtime 可见信息，例如 baseline/candidate citation document id、citation rank、retrieval score；
- F1 和 gold citation 指标使用 gold labels 做离线评估；
- 因此本阶段是 offline design / simulation，不是 runtime 默认策略变更。

### 实验命令

```powershell
python scripts\analyze_gold_citation_preservation_guards.py `
  --dev-baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --dev-candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --train-baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --train-candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json `
  --output artifacts\verified_rag_stage50_gold_citation_preservation_guard_analysis.json `
  --visualization-dir artifacts\verified_rag_stage50_gold_citation_preservation_guard_visuals
```

### 实验结果

dev split：

```text
candidate_as_is:
  F1 delta: +0.0008
  gold citation delta: 0
  changed answers: 15

preserve_baseline_out_of_rank_docs:
  blocked changed answers: 2
  blocked ids: DEV_Q052, DEV_Q095
  blocked gold citation loss: 0
  blocked answerable improvement: 1
  blocked answerable regression: 0
  F1 delta: +0.0005
  gold citation delta: 0
  changed answers: 13
```

train split：

```text
candidate_as_is:
  F1 delta: +0.0008
  gold citation delta: -1
  changed answers: 39

preserve_baseline_out_of_rank_docs:
  blocked changed answers: 6
  blocked ids: TRAIN_Q124, TRAIN_Q167, TRAIN_Q219, TRAIN_Q354, TRAIN_Q464, TRAIN_Q496
  blocked gold citation loss: 1
  blocked answerable improvement: 2
  blocked answerable regression: 1
  F1 delta: +0.0008
  gold citation delta: 0
  changed answers: 33
```

候选 guard 横向比较摘要：

```text
DEV:
candidate_as_is                         F1 +0.0008, gold delta  0, blocked 0
preserve_all_baseline_docs              F1 +0.0005, gold delta  0, blocked 2
preserve_baseline_out_of_rank_docs      F1 +0.0005, gold delta  0, blocked 2
preserve_baseline_out_of_rank_score_gte_60 F1 +0.0005, gold delta  0, blocked 2
preserve_baseline_score_gte_80_docs     F1 +0.0005, gold delta  0, blocked 1

TRAIN:
candidate_as_is                         F1 +0.0008, gold delta -1, blocked 0
preserve_all_baseline_docs              F1 +0.0007, gold delta  0, blocked 7
preserve_baseline_out_of_rank_docs      F1 +0.0008, gold delta  0, blocked 6
preserve_baseline_out_of_rank_score_gte_60 F1 +0.0008, gold delta  0, blocked 6
preserve_baseline_score_gte_80_docs     F1 +0.0007, gold delta  0, blocked 6
```

### 可视化结果

本阶段生成了 6 个 SVG 图：

```text
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_dev_f1_delta.svg
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_dev_gold_citation_delta.svg
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_dev_blocked_changed_count.svg
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_train_f1_delta.svg
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_train_gold_citation_delta.svg
artifacts/verified_rag_stage50_gold_citation_preservation_guard_visuals/stage50_train_blocked_changed_count.svg
```

完整 JSON artifact：

```text
artifacts/verified_rag_stage50_gold_citation_preservation_guard_analysis.json
```

以上 artifact 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：Stage 48/49 的 rank-contained guard 只约束 citation rank，不保护 baseline 已引用文档的 identity。
  - `TRAIN_Q496` 中 baseline 引用了 rank4 的 gold doc `swg21663373`；
  - candidate 改为引用 rank2 的相似非 gold doc `swg21971637`；
  - 结果 candidate 满足 rank-contained，但损失了 verified gold citation。
- 问题 2：保护 baseline out-of-rank docs 会有收益代价。
  - dev 上会拦掉 `DEV_Q052` 的 answerable improvement；
  - train 上会拦掉 `TRAIN_Q124`、`TRAIN_Q219` 两个 answerable improvement；
  - 同时也会拦掉 `TRAIN_Q496` 的 answerable regression 和 gold citation loss。
- 问题 3：更宽的 `preserve_all_baseline_docs` 没有明显更优。
  - train 上同样恢复 gold citation；
  - 但 blocked changed answers 更多，F1 delta 低于 `preserve_baseline_out_of_rank_docs`。

### 修正与处理

- 新增 offline guard analysis，而不是直接改默认 runtime。
- 新增可视化输出，避免只依赖文字表格判断。
- 新增单元测试，覆盖“candidate 丢失 protected gold citation 时，guard 用 baseline 样本回退并恢复 gold citation”的核心行为。
- 明确记录 runtime-only guard 信号与 gold-label offline evaluation 的边界。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\gold_citation_preservation_guard_analysis.py scripts\analyze_gold_citation_preservation_guards.py tests\test_gold_citation_preservation_guard_analysis.py
pytest -q tests\test_gold_citation_preservation_guard_analysis.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 126 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\verified_rag_stage50_gold_citation_preservation_guard_analysis.json > $null
```

结果：通过。

### 结论

- `preserve_baseline_out_of_rank_docs` 是当前最值得进入 runtime 实验的 guard 候选。
- 它在 train 上恢复了 `TRAIN_Q496` 的 gold citation loss，使 gold citation delta 从 `-1` 回到 `0`。
- 它没有降低 train 上四舍五入后的 F1 delta，仍为 `+0.0008`。
- 它在 dev 上会把 F1 delta 从 `+0.0008` 降到 `+0.0005`，代价是真实存在的，不能忽略。
- 因此本阶段只证明该 guard 候选值得进入下一步 runtime end-to-end 实验，不证明它可以默认化。
- 本阶段没有使用 held-out test set，没有修改默认 runtime 策略。

### 我学到的

- citation rank 合规和 gold document preservation 是两个不同风险面，不能用一个 guard 互相替代。
- train split 的价值在于暴露策略风险模式；但任何结论都只能指导下一步 dev/train runtime 实验，不能替代最终测试集评估。
- 对 RAG 策略而言，changed answers 数量本身不是好坏指标，必须拆开看 improvement、regression、unanswerable refusal、gold citation loss。
- 可视化对这种多指标比较很有帮助，尤其能快速看出更宽 guard 的 blocked 成本。

### 下一步

- 做 Stage 51：把 `preserve_baseline_out_of_rank_docs` 做成可选 runtime 参数。
- 目标：
  1. 不改变默认策略；
  2. 在 runtime 中组合 rank-contained reranker 与 baseline out-of-rank doc preservation guard；
  3. 跑 dev/train end-to-end verified RAG；
  4. 对比 Stage 46/49 top-k baseline 与 Stage 48/49 rank-contained candidate；
  5. 继续不使用 held-out test set；
  6. 若 runtime 结果仍满足 F1、citation、refusal 三类稳定性，再进入下一阶段评审。

## 2026-07-14 Stage 51：rank-contained preservation guard runtime 实验

### 本阶段目标

Stage 50 的离线模拟显示 `preserve_baseline_out_of_rank_docs` 可以恢复 train 上 `TRAIN_Q496` 的 gold citation loss。本阶段目标是把该 guard 接入真实 runtime，但仍保持默认策略不变：

1. 新增非默认 composition policy；
2. 在 runtime 中组合 `candidate_score_gte_60`、rank-contained guard 和 baseline out-of-rank document preservation guard；
3. 跑 dev/train end-to-end verified RAG；
4. 对比 top-k baseline 与 Stage 48/49 rank-contained candidate；
5. 继续不使用 held-out test set。

### 新增与修改

更新 runtime policy：

```text
src/ts_rag_agent/application/candidate_score_guarded_composition_policy.py
```

新增能力：

- `preserve_baseline_out_of_rank_docs` 可选参数；
- 当 baseline top-k answer candidates 中存在 `retrieval_rank > max_citation_rank` 的 document，且 candidate rewrite 会丢掉这些 document 时，阻止 rewrite，保留 baseline top-k candidates；
- 该 guard 必须和 `rank_contained_max_retrieval_rank` 一起使用，否则直接报错；
- decision trace 记录：
  - `preserve_baseline_out_of_rank_docs`
  - `protected_baseline_out_of_rank_document_ids`
  - `dropped_protected_baseline_out_of_rank_document_ids`

更新 runtime CLI：

```text
scripts/evaluate_verified_rag.py
```

新增非默认 policy alias：

```text
candidate-score-rank-contained-preserve-baseline-out-of-rank-reranker
```

report 中实际 policy name：

```text
candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
```

report 中 runtime guard：

```text
candidate_score_gte_60_all_selected_citations_rank_lte_max_citation_rank_preserve_baseline_out_of_rank_docs
```

更新测试：

```text
tests/test_candidate_score_guarded_composition_policy.py
```

新增覆盖：

- candidate rewrite 丢掉 baseline out-of-rank document 时会被 preservation guard 拦截；
- 没有 protected document 时不会误拦；
- 启用 preservation 但没有 rank limit 会报错；
- 如果 rewrite 已被 score guard 提前拦截，trace 不会误报 protected document 被 dropped。

### 实验命令

dev：

```powershell
python scripts\evaluate_verified_rag.py `
  --split dev `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy candidate-score-rank-contained-preserve-baseline-out-of-rank-reranker `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --candidate-reranker-model logistic_best_candidate `
  --candidate-reranker-train-split train `
  --sample-limit 1000 `
  --output artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json
```

train：

```powershell
python scripts\evaluate_verified_rag.py `
  --split train `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --min-evidence-score 15 `
  --composition-policy candidate-score-rank-contained-preserve-baseline-out-of-rank-reranker `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --candidate-reranker-model logistic_best_candidate `
  --candidate-reranker-train-split train `
  --sample-limit 1000 `
  --output artifacts\verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json
```

说明：train 第一次使用 120 秒超时时间运行时超时且未生成报告；随后使用 300 秒超时时间重跑成功。后续又在 trace 修正后重新跑了 dev/train，最终 artifacts 与当前代码一致。

comparison：

```powershell
python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_dev_preserve_vs_topk_comparison.json `
  --visualization-dir artifacts\verified_rag_stage51_dev_preserve_vs_topk_visuals

python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_dev_preserve_vs_rank_contained_comparison.json `
  --visualization-dir artifacts\verified_rag_stage51_dev_preserve_vs_rank_contained_visuals

python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_train_preserve_vs_topk_comparison.json `
  --visualization-dir artifacts\verified_rag_stage51_train_preserve_vs_topk_visuals

python scripts\compare_verified_rag_reports.py `
  --baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json `
  --candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_train_preserve_vs_rank_contained_comparison.json `
  --visualization-dir artifacts\verified_rag_stage51_train_preserve_vs_rank_contained_visuals
```

changed-answer risk：

```powershell
python scripts\analyze_verified_rag_changed_answer_risk.py `
  --baseline-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_dev_preserve_changed_answer_risk_analysis.json `
  --visualization-dir artifacts\verified_rag_stage51_dev_preserve_changed_answer_risk_visuals

python scripts\analyze_verified_rag_changed_answer_risk.py `
  --baseline-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --output artifacts\verified_rag_stage51_train_preserve_changed_answer_risk_analysis.json `
  --visualization-dir artifacts\verified_rag_stage51_train_preserve_changed_answer_risk_visuals
```

### 实验结果

dev top-k baseline：

```text
verified F1: 0.2755
verified gold citation: 95 / 158
answerable refusals: 2
unanswerable refusals: 4
newly refused: 6
```

dev Stage 48 rank-contained：

```text
verified F1: 0.2763
verified gold citation: 95 / 158
answerable refusals: 2
unanswerable refusals: 4
newly refused: 6
```

dev Stage 51 preservation runtime：

```text
verified F1: 0.2760
verified gold citation: 95 / 158
answerable refusals: 2
unanswerable refusals: 4
newly refused: 6
```

dev Stage 51 vs top-k：

```text
verified F1 delta: +0.0005
verified gold citation delta: 0
newly refused delta: 0
changed verified answers: 13
answerable improved: 1
answerable regressed: 0
answerable tied changed: 4
changed unanswerable: 8
unanswerable refusal regressions: 0
candidate out-of-rank citation: 0
```

dev Stage 51 vs Stage 48 rank-contained：

```text
verified F1 delta: -0.0003
verified gold citation delta: 0
newly refused delta: 0
changed verified answers: 2
answerable improved: 0
answerable regressed: 1
answerable tied changed: 0
```

train top-k baseline：

```text
verified F1: 0.2557
verified gold citation: 232 / 442
answerable refusals: 8
unanswerable refusals: 5
newly refused: 13
```

train Stage 49 rank-contained：

```text
verified F1: 0.2565
verified gold citation: 231 / 442
answerable refusals: 8
unanswerable refusals: 5
newly refused: 13
```

train Stage 51 preservation runtime：

```text
verified F1: 0.2565
verified gold citation: 232 / 442
answerable refusals: 8
unanswerable refusals: 5
newly refused: 13
```

train Stage 51 vs top-k：

```text
verified F1 delta: +0.0008
verified gold citation delta: 0
newly refused delta: 0
changed verified answers: 33
answerable improved: 4
answerable regressed: 1
answerable tied changed: 21
changed unanswerable: 7
unanswerable refusal regressions: 0
candidate out-of-rank citation: 0
```

train Stage 51 vs Stage 49 rank-contained：

```text
verified F1 delta: 0.0000
verified gold citation delta: +1
newly refused delta: 0
changed verified answers: 6
answerable improved: 1
answerable regressed: 2
answerable tied changed: 0
```

关键样本：

```text
TRAIN_Q496:
  Stage 49 rank-contained verified docs:
    swg21971637
    swg21587102
    swg21587102

  Stage 51 preservation verified docs:
    swg21587102
    swg21587102
    swg21663373

  result:
    gold doc swg21663373 restored
```

相对 Stage 49 rank-contained，Stage 51 的代价：

```text
TRAIN_Q124: answerable F1 delta -0.026721
TRAIN_Q219: answerable F1 delta -0.017297
```

相对 top-k，Stage 51 的 train 仍保留的 answerable improvement：

```text
TRAIN_Q027
TRAIN_Q353
TRAIN_Q515
TRAIN_Q517
```

相对 top-k，Stage 51 的 train answerable regression：

```text
TRAIN_Q310
```

### 可视化结果

本阶段生成的对比可视化目录：

```text
artifacts/verified_rag_stage51_dev_preserve_vs_topk_visuals
artifacts/verified_rag_stage51_dev_preserve_vs_rank_contained_visuals
artifacts/verified_rag_stage51_train_preserve_vs_topk_visuals
artifacts/verified_rag_stage51_train_preserve_vs_rank_contained_visuals
```

每个 comparison 目录包含：

```text
verified_rag_metric_deltas.svg
verified_rag_changed_answers.svg
verified_rag_answerable_f1_outcomes.svg
```

changed-answer risk 可视化目录：

```text
artifacts/verified_rag_stage51_dev_preserve_changed_answer_risk_visuals
artifacts/verified_rag_stage51_train_preserve_changed_answer_risk_visuals
```

每个 risk 目录包含：

```text
stage47_changed_by_route.svg
stage47_changed_by_outcome.svg
stage47_out_of_rank_by_outcome.svg
stage47_candidate_worst_rank.svg
```

完整 JSON artifacts：

```text
artifacts/verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json
artifacts/verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json
artifacts/verified_rag_stage51_dev_preserve_vs_topk_comparison.json
artifacts/verified_rag_stage51_dev_preserve_vs_rank_contained_comparison.json
artifacts/verified_rag_stage51_train_preserve_vs_topk_comparison.json
artifacts/verified_rag_stage51_train_preserve_vs_rank_contained_comparison.json
artifacts/verified_rag_stage51_dev_preserve_changed_answer_risk_analysis.json
artifacts/verified_rag_stage51_train_preserve_changed_answer_risk_analysis.json
```

以上 artifacts 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：Stage 49 的 rank-contained guard 能防止 candidate out-of-rank citation，但不能防止 candidate 用 rank 内相似文档替换 baseline 中原本的 out-of-rank gold doc。
  - Stage 51 通过 preservation guard 恢复了 `TRAIN_Q496` 的 gold doc `swg21663373`。
- 问题 2：preservation guard 不是免费收益。
  - dev 相对 Stage 48 rank-contained 损失 `0.0003` F1；
  - train 相对 Stage 49 rank-contained 牺牲了 `TRAIN_Q124`、`TRAIN_Q219` 两个 answerable improvement；
  - 但 train gold citation delta 从 Stage 49 的 `-1` 修复到 `0`。
- 问题 3：首次 train end-to-end 命令 120 秒超时。
  - 原因是 train split 600 题生成验证耗时约 178 秒；
  - 处理方式是使用 300 秒超时重跑，并在 trace 修正后再次重跑，最终报告有效。
- 问题 4：trace 细节最初会在更早 guard 已经拦截时误显示 protected docs dropped。
  - 已修正为：如果 score/route/rank guard 已经保留 baseline，则 dropped protected docs 为空；
  - 新增测试覆盖该诊断边界。

### 修正与处理

- 新增非默认 runtime policy，不改变 `top-k` 默认策略。
- 保持 `candidate-score-rank-contained-reranker` 原有行为不变。
- 新增 preservation guard 只在显式 policy alias 下启用。
- report 中写入明确 runtime guard，方便后续追踪。
- 没有使用 held-out test set。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\candidate_score_guarded_composition_policy.py scripts\evaluate_verified_rag.py tests\test_candidate_score_guarded_composition_policy.py
pytest -q tests\test_candidate_score_guarded_composition_policy.py
```

结果：

```text
ruff: passed
pytest: 7 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 130 passed
```

artifact JSON 校验：8 个 Stage 51 JSON artifacts 均可正常解析。

### 结论

- Stage 51 runtime 实验通过 dev/train end-to-end 检查：
  - dev：相对 top-k F1 `+0.0005`，gold citation delta `0`；
  - train：相对 top-k F1 `+0.0008`，gold citation delta `0`；
  - dev/train 均无新增 refusal；
  - dev/train 均无 unanswerable refusal regression；
  - dev/train candidate out-of-rank citation 均为 `0`。
- 相对 Stage 49 rank-contained，Stage 51 成功修复 train 的 1 个 gold citation loss。
- 代价是真实存在的：dev F1 比 Stage 48 rank-contained 低 `0.0003`，train 牺牲两个 rank-contained 带来的 answerable improvement。
- 因此 Stage 51 可以进入下一步默认化前评审，但仍不能直接默认化。
- 本阶段没有使用 held-out test set，没有改变默认 runtime 策略。

### 我学到的

- citation-risk guard 需要同时看 rank、document identity、changed-answer outcome，单独看 rank-contained 会漏掉“rank 内相似文档替换 gold doc”的风险。
- runtime trace 本身也是实验资产；trace 如果误报 dropped docs，会污染下一阶段分析。
- preservation guard 的收益更像“修复 citation stability”，不是扩大 F1 上限；它适合进入默认化评审，但不应该因为 train gold citation 修复就直接晋升。
- 对这种候选策略，dev 的轻微 F1 损失和 train 的 citation 修复需要在同一个评审门槛下讨论。

### 下一步

- 做 Stage 52：默认化前评审与 held-out test 使用协议冻结。
- 目标：
  1. 汇总 Stage 46-51 的 dev/train 指标、风险、可视化和 artifact；
  2. 明确默认化门槛：F1、gold citation、refusal、unanswerable regression、changed-answer 风险；
  3. 决定是否把 Stage 51 policy 作为“唯一待测候选”；
  4. 在使用 held-out test set 前冻结命令、参数、候选策略和验收标准；
  5. 继续不在未冻结协议前使用 held-out test set。

## 2026-07-14 Stage 52：默认化前评审与 held-out test 协议冻结

### 本阶段目标

Stage 51 已经把 rank-contained preservation guard 做成非默认 runtime 参数，并在 dev/train 上通过 end-to-end verified RAG 检查。本阶段目标是做默认化前评审，而不是直接默认化：

1. 汇总 Stage 46-51 的 dev/train 证据；
2. 明确 Stage 51 是否可以作为唯一 held-out test 候选；
3. 固定 held-out test 前的候选策略、参数和验收标准；
4. 明确禁止在看到 held-out 结果后继续调参；
5. 不加载、不评估 held-out 数据。

### 新增内容

新增评审模块：

```text
src/ts_rag_agent/application/defaultization_readiness_review.py
```

新增 CLI：

```text
scripts/review_defaultization_readiness.py
```

新增测试：

```text
tests/test_defaultization_readiness_review.py
```

模块能力：

- 读取 dev/train top-k baseline、rank-contained candidate、Stage 51 candidate 和 changed-answer risk report；
- 使用完整 `samples` 精确计算 verified gold citation count；
- 生成候选 readiness checks；
- 生成 rank-contained safety checks；
- 写入 held-out test protocol；
- 输出 SVG 可视化。

### 事实边界

- 本阶段没有使用 held-out test set。
- 本阶段只读取已有 dev/train artifacts 和项目文档中的数据策略。
- 本阶段没有读取 `data/raw/nvidia_techqa_rag_eval/train.json` 的样本内容。
- 本阶段没有改变默认 runtime。
- 本阶段没有新增任何调参。

项目已有数据策略确认：

```text
PrimeQA/TechQA: training and development source
nvidia/TechQA-RAG-Eval: final evaluation set
```

因此 Stage 52 将 NVIDIA TechQA-RAG-Eval 固定为后续 held-out evaluation 来源，而不是把 PrimeQA validation 当成默认化测试集。

### 评审命令

```powershell
python scripts\review_defaultization_readiness.py `
  --dev-topk-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage46_topk_report.json `
  --dev-rank-contained-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage48_rank_contained_reranker_report.json `
  --dev-candidate-report artifacts\verified_rag_dev_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --dev-candidate-risk-report artifacts\verified_rag_stage51_dev_preserve_changed_answer_risk_analysis.json `
  --train-topk-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_topk_report.json `
  --train-rank-contained-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage49_rank_contained_reranker_report.json `
  --train-candidate-report artifacts\verified_rag_train_hybrid_routing_mcpd3_stage51_rank_contained_preserve_out_of_rank_report.json `
  --train-candidate-risk-report artifacts\verified_rag_stage51_train_preserve_changed_answer_risk_analysis.json `
  --output artifacts\verified_rag_stage52_defaultization_readiness_review.json `
  --visualization-dir artifacts\verified_rag_stage52_defaultization_readiness_visuals
```

### 评审结果

总体结论：

```text
candidate_passes_dev_train_readiness: true
rank_contained_passes_dev_train_safety: false
unique_heldout_candidate:
  candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
default_runtime_change:
  not_allowed_before_heldout_evaluation
status:
  stage51_is_unique_candidate_for_single_heldout_evaluation
```

dev readiness：

```text
candidate checks: 8 / 8 passed
F1 delta vs top-k: +0.0005
gold citation delta vs top-k: 0
generated answerable delta: 0
answerable refusal delta: 0
unanswerable refusal delta: 0
newly refused delta: 0
unanswerable refusal regressions: 0
candidate out-of-rank citation: 0
changed verified answers: 13
```

train readiness：

```text
candidate checks: 8 / 8 passed
F1 delta vs top-k: +0.0008
gold citation delta vs top-k: 0
generated answerable delta: 0
answerable refusal delta: 0
unanswerable refusal delta: 0
newly refused delta: 0
unanswerable refusal regressions: 0
candidate out-of-rank citation: 0
changed verified answers: 33
```

rank-contained candidate safety：

```text
dev rank-contained vs top-k:
  F1 delta: +0.0008
  gold citation delta: 0

train rank-contained vs top-k:
  F1 delta: +0.0008
  gold citation delta: -1
```

因此 Stage 48/49 的 rank-contained candidate 被排除为 held-out 候选，原因是 train gold citation safety check 未通过。

### 冻结的 held-out test 协议

held-out dataset：

```text
nvidia/TechQA-RAG-Eval
```

本地路径：

```text
data/raw/nvidia_techqa_rag_eval/train.json
data/raw/nvidia_techqa_rag_eval/corpus.zip
```

冻结候选策略：

```text
candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
```

冻结 runtime 参数：

```text
retrieval_top_k: 5
evidence_selector: hybrid-routing
max_candidates_per_document: 3
max_sentences: 3
min_sentence_score: 2.0
min_evidence_score: 15
max_citation_rank: 3
min_citations: 1
composition_policy: candidate-score-rank-contained-preserve-baseline-out-of-rank-reranker
candidate_reranker_dataset: artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl
candidate_reranker_model: logistic_best_candidate
candidate_reranker_train_split: train
```

held-out 前必须完成：

```text
1. 实现或复用 NVIDIA TechQA-RAG-Eval evaluator，且不能改变冻结 runtime 参数。
2. 运行并保存 NVIDIA questions 与 PrimeQA train/dev 调参数据的 leakage report。
3. 在同一个 evaluator 下只运行一次 top-k baseline 和一次冻结 Stage 51 candidate。
```

held-out 验收标准：

```text
heldout verified F1 delta vs top-k >= 0
heldout verified gold citation delta vs top-k >= 0
heldout newly refused delta vs top-k <= 0
heldout answerable refusal delta vs top-k <= 0
heldout unanswerable refusal delta vs top-k <= 0
heldout unanswerable refusal regressions == 0
heldout candidate out-of-rank citation count == 0
leakage report has no unhandled train/dev overlap with the held-out rows
```

冻结后禁止：

```text
1. 看到 held-out 结果后继续调阈值、prompt、selector、reranker model 或 guard logic。
2. 在 held-out 上比较多个新候选后挑最好者。
3. 用 NVIDIA TechQA-RAG-Eval 训练或重新拟合 candidate reranker。
4. held-out 验收标准通过前修改默认 runtime。
```

### 可视化结果

本阶段生成：

```text
artifacts/verified_rag_stage52_defaultization_readiness_visuals/stage52_verified_f1_by_policy.svg
artifacts/verified_rag_stage52_defaultization_readiness_visuals/stage52_gold_citation_delta_vs_topk.svg
artifacts/verified_rag_stage52_defaultization_readiness_visuals/stage52_changed_answer_risk.svg
artifacts/verified_rag_stage52_defaultization_readiness_visuals/stage52_readiness_pass_count.svg
```

完整 JSON artifact：

```text
artifacts/verified_rag_stage52_defaultization_readiness_review.json
```

以上 artifacts 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 问题与原因

- 问题 1：rank-contained candidate 有 F1 收益，但 train gold citation delta 为 `-1`。
  - 原因是 Stage 49 中 `TRAIN_Q496` 的 gold doc 被 rank 内相似文档替换；
  - Stage 52 因此不允许 rank-contained 直接进入 held-out。
- 问题 2：Stage 51 candidate 虽然通过 dev/train readiness，但仍不能默认化。
  - 原因是 dev/train 不是最终 held-out evaluation；
  - Stage 51 只能成为唯一待测候选，不能成为默认策略。
- 问题 3：当前 verified RAG evaluator 主要围绕 PrimeQA train/dev split。
  - 下一步需要实现或复用 NVIDIA evaluator；
  - 但实现 evaluator 不能改变已冻结 runtime 参数。

### 修正与处理

- 将默认化前评审工程化为可复跑脚本，而不是手写一次性结论。
- 将 held-out test protocol 写入 JSON artifact，后续不能随意改参数。
- 将 default runtime change 明确标记为 `not_allowed_before_heldout_evaluation`。
- 将 Stage 51 明确为唯一 held-out candidate。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\defaultization_readiness_review.py scripts\review_defaultization_readiness.py tests\test_defaultization_readiness_review.py
pytest -q tests\test_defaultization_readiness_review.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 134 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\verified_rag_stage52_defaultization_readiness_review.json > $null
```

结果：通过。

### 结论

- Stage 52 完成默认化前评审。
- Stage 51 policy 是唯一进入 held-out evaluation 的候选。
- Stage 48/49 rank-contained policy 因 train gold citation delta `-1` 被排除。
- held-out test 协议已经冻结。
- 默认 runtime 仍然不变。
- 本阶段没有使用 held-out test set。

### 我学到的

- “可以进入 held-out”与“可以默认化”是两个不同门槛，中间必须有协议冻结。
- 对最终测试集，最重要的不是多跑几次，而是在第一次运行前固定候选、参数、验收标准和禁止事项。
- 评审脚本应该使用 exact sample-level gold citation count，而不是用四舍五入后的 citation rate 反推。
- NVIDIA TechQA-RAG-Eval 在本项目里是 held-out evaluation 来源；PrimeQA validation 不应被临时替代成默认化测试集。

### 下一步

- 做 Stage 53：NVIDIA held-out evaluator 与 leakage report。
- 目标：
  1. 不改变 Stage 52 冻结的 runtime 参数；
  2. 实现或复用 NVIDIA TechQA-RAG-Eval evaluator；
  3. 先生成 train/dev 与 NVIDIA held-out 的 leakage report；
  4. 若 leakage 无未处理重叠，再按冻结协议运行 top-k baseline 和 Stage 51 candidate；
  5. 只运行冻结候选，不新增候选、不调参；
  6. 根据 Stage 52 验收标准判断是否允许进入默认化决策。

## 2026-07-14 Stage 53：NVIDIA held-out leakage audit 与评估阻断

### 本阶段目标

Stage 52 冻结协议要求：在运行任何 held-out 指标前，必须先生成 NVIDIA TechQA-RAG-Eval 与 PrimeQA train/dev 的 leakage report。本阶段严格按这个顺序执行：

1. 不改变 Stage 52 冻结的 runtime 参数；
2. 先做 NVIDIA 与 PrimeQA train/dev 的问题文本泄漏检查；
3. 如果无未处理重叠，再运行 held-out top-k baseline 和 Stage 51 candidate；
4. 如果发现未处理重叠，则停止，不运行 held-out 指标。

### 新增内容

新增泄漏分析模块：

```text
src/ts_rag_agent/application/heldout_leakage_analysis.py
```

新增 CLI：

```text
scripts/analyze_nvidia_heldout_leakage.py
```

新增测试：

```text
tests/test_heldout_leakage_analysis.py
```

同步修正文档：

```text
docs/data_strategy.md
```

将 NVIDIA TechQA-RAG-Eval 从“当前可直接使用的 final evaluation set”修正为“原计划 final evaluation set，但 Stage 53 已证明 `train.json` 与 PrimeQA train/dev 完全重叠，当前不能作为 held-out defaultization test”。

模块能力：

- 对 question text 做稳定归一化：
  - lowercase；
  - 只保留 ASCII alnum token；
  - 合并空白；
- 检查 exact overlap；
- 检查非 exact 的 near-duplicate overlap；
- 区分：
  - `exact_overlap_count`：发生 exact overlap 的 held-out question 数；
  - `exact_overlap_pair_count`：held-out 与 development 的 overlap pair 数；
- 输出 `heldout_usable_without_exclusions`；
- 输出阻断决策；
- 生成 SVG 可视化。

### 实验命令

```powershell
python scripts\analyze_nvidia_heldout_leakage.py `
  --nvidia-samples data\raw\nvidia_techqa_rag_eval\train.json `
  --primeqa-train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json `
  --primeqa-dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json `
  --output artifacts\nvidia_heldout_leakage_stage53.json `
  --visualization-dir artifacts\nvidia_heldout_leakage_stage53_visuals `
  --near-duplicate-threshold 0.9 `
  --sample-limit 20
```

### 结果

正式 leakage report：

```text
heldout_questions: 910
development_questions: 910
PrimeQA/TechQA::train: 600
PrimeQA/TechQA::dev: 310
exact_overlap_count: 910
exact_overlap_pair_count: 974
near_duplicate_overlap_count: 0
near_duplicate_overlap_pair_count: 0
unhandled_overlap_count: 910
heldout_questions_without_detected_overlap: 0
heldout_usable_without_exclusions: false
```

阻断决策：

```text
blocked: proposed held-out source overlaps with development data; do not run held-out evaluation metrics.
```

解释：

- NVIDIA TechQA-RAG-Eval `train.json` 的 910 个 question 全部与 PrimeQA train/dev exact overlap。
- `exact_overlap_pair_count` 是 974，大于 910，是因为 PrimeQA train/dev 中存在归一化后重复的问题文本，导致一个 NVIDIA question 可能匹配多个 development row。
- 因为 `unhandled_overlap_count = 910`，Stage 52 的 held-out 验收前置条件失败。

### 可视化结果

本阶段生成：

```text
artifacts/nvidia_heldout_leakage_stage53_visuals/stage53_heldout_overlap_counts.svg
artifacts/nvidia_heldout_leakage_stage53_visuals/stage53_development_source_counts.svg
```

完整 JSON artifact：

```text
artifacts/nvidia_heldout_leakage_stage53.json
```

以上 artifacts 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 事实边界

- 本阶段读取了 NVIDIA `train.json` 的问题字段用于 leakage audit。
- 本阶段没有运行 NVIDIA held-out top-k baseline。
- 本阶段没有运行 NVIDIA held-out Stage 51 candidate。
- 本阶段没有生成任何 held-out answer-quality metric。
- 本阶段没有改变默认 runtime。
- 本阶段没有新增候选策略、没有调参。
- 本阶段没有实现 NVIDIA evaluator 的质量评估路径，因为 leakage audit 已经在评估前置步骤阻断，继续实现并运行指标会制造误导性“held-out”结果。

### 问题与原因

- 问题 1：Stage 52 假设 NVIDIA TechQA-RAG-Eval 可作为 held-out evaluation source，但 Stage 53 发现它与 PrimeQA train/dev 完全重叠。
  - 910/910 NVIDIA questions 与 PrimeQA train/dev exact overlap；
  - 因此它不能作为当前开发过程的独立 held-out test set。
- 问题 2：项目早期数据策略把 NVIDIA 标成 final evaluation set，但没有先完成 overlap exclusion。
  - Stage 53 证明这个假设不成立；
  - 后续必须修正 evaluation strategy。
- 问题 3：第一次正式 report 里把 exact overlap pair count 当成 unique held-out question count，导致 without-overlap 为负数。
  - 原因是一个 held-out question 可以匹配多个 development duplicate row；
  - 已修正为同时报告 question count 与 pair count；
  - 新增测试覆盖该口径。

### 修正与处理

- 建立独立 leakage audit 工具，而不是手写一次性脚本。
- 修正 exact overlap 计数口径：
  - `exact_overlap_count = unique held-out question count`；
  - `exact_overlap_pair_count = heldout-development pair count`。
- 修正 `docs/data_strategy.md`，避免后续继续把 NVIDIA `train.json` 当作当前可用 held-out。
- 按 Stage 52 协议停止，不运行 held-out metrics。
- 将 NVIDIA held-out evaluation 标记为 blocked，而不是用重叠数据生成误导性分数。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\heldout_leakage_analysis.py scripts\analyze_nvidia_heldout_leakage.py tests\test_heldout_leakage_analysis.py
pytest -q tests\test_heldout_leakage_analysis.py
```

结果：

```text
ruff: passed
pytest: 6 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 140 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\nvidia_heldout_leakage_stage53.json > $null
```

结果：通过。

### 结论

- Stage 53 完成 leakage audit。
- NVIDIA TechQA-RAG-Eval `train.json` 不能作为当前项目的独立 held-out test set。
- Stage 52 冻结协议中的 held-out evaluation 被正确阻断。
- 没有运行任何 held-out quality metric。
- Stage 51 仍然只是 dev/train 上通过 readiness 的候选，不能默认化。
- 默认 runtime 仍然不变。

### 我学到的

- “数据集名字不同”不等于“评估样本独立”。
- 在 RAG 项目里，held-out 前置 leakage audit 不是形式流程；它可以直接阻止错误结论。
- overlap report 必须区分 question-level count 和 pair-level count，否则重复 development rows 会让指标解释混乱。
- 被阻断也是有效阶段结果，因为它保护了实验可信度。

### 下一步

- 做 Stage 54：修正 evaluation strategy，重新定义真实可用的最终评估方案。
- 目标：
  1. 更新数据策略，明确 NVIDIA TechQA-RAG-Eval 与 PrimeQA train/dev 完全重叠，不能作为当前 held-out；
  2. 盘点可选评估路径：
     - 引入真正外部数据集；
     - 从现有数据重新划分 leak-safe split 并重新训练/重跑全部 dev 流程；
     - 保留 Stage 51 为 dev/train candidate，但不默认化；
  3. 不运行任何伪 held-out 指标；
  4. 在用户确认新评估路径前，不改变默认 runtime。

## 2026-07-14 Stage 54：evaluation strategy review 与下一路径选项冻结

### 本阶段目标

Stage 53 已经证明 NVIDIA TechQA-RAG-Eval `train.json` 与 PrimeQA train/dev 完全重叠，当前不能作为 held-out defaultization test。本阶段目标是修正 evaluation strategy，而不是继续跑任何伪 held-out 指标：

1. 将 Stage 53 的 leakage 阻断转成正式 evaluation strategy review；
2. 明确 NVIDIA `train.json` 路径被拒绝；
3. 盘点后续可选路径，并标记哪些需要用户确认；
4. 保持 Stage 51 candidate 为非默认候选；
5. 不改变默认 runtime。

### 新增内容

新增策略评审模块：

```text
src/ts_rag_agent/application/evaluation_strategy_review.py
```

新增 CLI：

```text
scripts/review_evaluation_strategy.py
```

新增测试：

```text
tests/test_evaluation_strategy_review.py
```

新增版本化策略文档：

```text
docs/evaluation_strategy.md
```

同步更新：

```text
docs/data_strategy.md
```

更新内容是把当前 evaluation path options 明确指向 `docs/evaluation_strategy.md`，避免后续只看 data strategy 时误把 NVIDIA `train.json` 当作可用 held-out。

### 评审命令

```powershell
python scripts\review_evaluation_strategy.py `
  --readiness-review artifacts\verified_rag_stage52_defaultization_readiness_review.json `
  --leakage-report artifacts\nvidia_heldout_leakage_stage53.json `
  --output artifacts\evaluation_strategy_stage54_review.json `
  --visualization-dir artifacts\evaluation_strategy_stage54_visuals
```

### 评审结果

当前事实：

```text
stage51_candidate_policy:
  candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
stage51_dev_train_readiness_passed: true
nvidia_train_json_blocked_as_heldout: true
nvidia_exact_overlap_questions: 910
nvidia_heldout_questions: 910
nvidia_unhandled_overlap_questions: 910
default_runtime_policy: unchanged
```

拒绝路径：

```text
use_nvidia_train_json_as_current_heldout: rejected
reason:
  Stage 53 found exact normalized overlap for all NVIDIA rows against PrimeQA train/dev.
```

可选路径 1：

```text
label: external_independent_eval_set
status: available_after_user_confirmation
validity_score: 3
effort_score: 2
can_support_defaultization: true
keeps_stage51_candidate_frozen: true
requires_full_pipeline_rerun: false
```

说明：这是最干净的默认化路径。它保留 Stage 51 frozen candidate，但必须先找到或构造真正没有进入 PrimeQA train/dev 开发循环的外部评估源，并先做 license/schema/leakage audit。

可选路径 2：

```text
label: rebuild_leak_safe_primeqa_split
status: available_after_user_confirmation
validity_score: 2
effort_score: 3
can_support_defaultization: true
keeps_stage51_candidate_frozen: false
requires_full_pipeline_rerun: true
```

说明：这条路径不需要外部数据，但会让当前 Stage 31-53 的 model-selection evidence 失效。需要重新划分 split、重建 candidate-reranker dataset、重新跑 dev 流程，再冻结新协议。

可选路径 3：

```text
label: freeze_without_defaultization
status: available_after_user_confirmation
validity_score: 1
effort_score: 1
can_support_defaultization: false
keeps_stage51_candidate_frozen: true
requires_full_pipeline_rerun: false
```

说明：这是最低风险的停止路径。Stage 51 保留为非默认研究候选，默认 runtime 继续保持 top-k，但不能进入默认化。

### 决策边界

Stage 54 没有替用户擅自选择下一条路线。报告中只给出推荐方向：

```text
requires_user_confirmation: true
recommended_for_confirmation: external_independent_eval_set
no_action_without_confirmation:
  Do not change the default runtime, do not run pseudo-held-out metrics,
  and do not tune the Stage 51 candidate.
```

因此下一阶段必须先确认一个路径：

```text
if external_independent_eval_set:
  Stage 55: external dataset discovery and schema fit audit

if rebuild_leak_safe_primeqa_split:
  Stage 55: split redesign plan and full rerun cost estimate

if freeze_without_defaultization:
  Stage 55: package current candidate as non-default research result
```

### 可视化结果

本阶段生成：

```text
artifacts/evaluation_strategy_stage54_visuals/stage54_option_validity_score.svg
artifacts/evaluation_strategy_stage54_visuals/stage54_option_effort_score.svg
artifacts/evaluation_strategy_stage54_visuals/stage54_option_defaultization_support.svg
artifacts/evaluation_strategy_stage54_visuals/stage54_blocked_nvidia_overlap.svg
```

完整 JSON artifact：

```text
artifacts/evaluation_strategy_stage54_review.json
```

以上 artifacts 均在本地 `artifacts/` 下，按 `.gitignore` 规则不纳入 git。

### 事实边界

- 本阶段没有运行任何 held-out 指标。
- 本阶段没有读取新的外部数据集。
- 本阶段没有选择下一条 evaluation path。
- 本阶段没有改默认 runtime。
- 本阶段没有调 Stage 51 candidate 参数。
- 本阶段只是把 Stage 53 的阻断结果转成策略评审和用户确认选项。

### 问题与原因

- 问题 1：原先的 final evaluation source 已经被 leakage audit 阻断。
  - NVIDIA `train.json` 与 PrimeQA train/dev 910/910 exact overlap；
  - 继续使用它会制造伪 held-out 结果。
- 问题 2：Stage 51 通过 dev/train readiness，但不能 default。
  - 缺少独立 final evaluation source；
  - 因此只能保持为 non-default candidate。
- 问题 3：下一步不再是纯工程实现，而是 evaluation design 选择。
  - 外部数据、重划 split、停止默认化三条路的成本和意义不同；
  - 按项目指令，不能由我擅自选择。

### 修正与处理

- 新增可复跑 strategy review 工具。
- 新增 `docs/evaluation_strategy.md`，把当前策略边界版本化。
- 在 `docs/data_strategy.md` 中链接当前 evaluation strategy。
- 把 NVIDIA 当前 held-out 路径明确列入 rejected path。
- 把三个后续选项列为 `available_after_user_confirmation`。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\evaluation_strategy_review.py scripts\review_evaluation_strategy.py tests\test_evaluation_strategy_review.py
pytest -q tests\test_evaluation_strategy_review.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 143 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\evaluation_strategy_stage54_review.json > $null
```

结果：通过。

### 结论

- Stage 54 完成 evaluation strategy review。
- NVIDIA `train.json` 继续保持 rejected，不允许作为当前 held-out。
- Stage 51 继续保持 non-default candidate。
- 默认 runtime 不变。
- 下一步不能自动继续跑指标，必须先确认 evaluation path。

### 我学到的

- 当最终评估源失效时，继续写 evaluator 反而可能让项目偏离事实；先修正 strategy 才是正确顺序。
- 评估策略选择会改变项目成本和结论性质，必须显式让用户确认。
- 外部独立评估集是最干净路径，但仍然需要 license、schema、leakage audit。
- 重新划分现有数据是可行路径，但它会推翻旧 split 上的全部 candidate-selection 证据。

### 下一步

- 做 Stage 55，但需要先确认路径：
  1. `external_independent_eval_set`：做外部数据集发现与 schema fit audit；
  2. `rebuild_leak_safe_primeqa_split`：做 leak-safe split 重设计与全流程重跑成本评估；
  3. `freeze_without_defaultization`：把 Stage 51 打包为非默认研究结果并停止默认化流程。
- 在确认前：
  - 不改默认 runtime；
  - 不跑伪 held-out；
  - 不调 Stage 51 candidate。

## 2026-07-14 Stage 55：external evaluation dataset discovery 与 MSQA 候选冻结

### 本阶段目标

Stage 54 已经把下一步分成三条路径，并要求用户先确认。用户确认走推荐路线：

```text
external_independent_eval_set
```

因此本阶段目标不是跑指标，而是找新的外部独立测试集候选，并完成第一轮 schema/license/citation/leakage 风险审计。

本阶段必须遵守的边界：

1. 不把任何候选数据集直接当作 held-out test；
2. 不运行 Stage 51 candidate 指标；
3. 不运行 top-k baseline 对比；
4. 不改默认 runtime；
5. 不调 Stage 51 candidate；
6. 所有数据集结论必须基于公开来源或本地真实产物，不能臆造。

### 新增内容

新增 Stage 55 发现与审计模块：

```text
src/ts_rag_agent/application/external_eval_dataset_discovery.py
```

新增命令行脚本：

```text
scripts/discover_external_eval_datasets.py
```

新增测试：

```text
tests/test_external_eval_dataset_discovery.py
```

新增文档：

```text
docs/external_eval_datasets.md
```

更新文档：

```text
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 外部来源检查

本阶段通过公开来源检查了以下候选：

```text
microsoft_msqa
multidoc2dial
doc2dial
stackexchange_dumps
natural_questions
msdialog
```

主要来源包括：

```text
https://github.com/microsoft/Microsoft-Q-A-MSQA-
https://aclanthology.org/2023.emnlp-industry.29/
https://cdla.dev/permissive-2-0/
https://huggingface.co/datasets/IBM/doc2dial
https://huggingface.co/datasets/IBM/multidoc2dial
https://ciir.cs.umass.edu/downloads/msdialog/
https://archive.org/download/stackexchange
https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/
https://github.com/google-research-datasets/natural-questions
```

事实边界：

- 本阶段查阅了公开网页、GitHub API 和少量公开 raw 文本。
- 本阶段没有把 MSQA `msqa-32k.csv` 下载到 `data/raw/`。
- 本阶段没有解析 MSQA 32k 行数据。
- 本阶段没有运行任何模型质量评估。
- 本阶段没有运行任何 leakage audit。
- 本阶段没有产生 held-out 分数。

### 审计规则

新增的 `fit_score` 是我根据项目目标生成的审计适配分，不是模型效果分。

计算规则：

```text
domain_fit * 2
+ schema_fit
+ citation_fit
+ answerability_fit
+ license_fit
+ independence_fit
```

其中 `domain_fit` 加权两倍，因为当前项目目标是技术支持 RAG 默认化判断，不是泛化 QA benchmark。

### Stage 55 结果

脚本命令：

```powershell
python scripts\discover_external_eval_datasets.py --output artifacts\external_eval_dataset_discovery_stage55.json --visualization-dir artifacts\external_eval_dataset_discovery_stage55_visuals
```

结果摘要：

```text
recommended_candidate: microsoft_msqa
recommended_candidate_name: Microsoft Q&A (MSQA)
fit_score: 17
can_run_final_metrics_now: false
default_runtime_policy: unchanged
recommended_next_stage: Stage 56: MSQA local schema probe, source-link coverage audit, and PrimeQA leakage audit protocol
```

候选排序：

```text
microsoft_msqa: fit_score 17, status recommended_for_stage56_schema_probe
multidoc2dial: fit_score 15, status secondary_document_grounded_reference
natural_questions: fit_score 15, status control_benchmark_only
doc2dial: fit_score 14, status secondary_document_grounded_reference
stackexchange_dumps: fit_score 11, status manual_derivation_candidate_only
msdialog: fit_score 10, status blocked_until_access_and_license_confirmation
```

结论：

MSQA 是当前最好的外部测试集候选，但还不是可直接使用的 held-out test set。

原因：

- 优点：
  - 技术支持 / 企业 IT QA 域匹配最好；
  - 来源是 Microsoft Q&A；
  - README 记录为 32,252 行；
  - 有 question、accepted answer、tags、URL 和 `test_id.txt`；
  - 数据集许可证公开列为 CDLA-Permissive-2.0。
- 风险：
  - accepted-answer-only 过滤导致没有原生 unanswerable rows；
  - source-link/citation 覆盖率还没有本地统计；
  - 还没有下载并解析 CSV；
  - 还没有与 PrimeQA train/dev 做 exact/near-duplicate leakage audit。

因此本阶段只把 MSQA 冻结为 Stage 56 schema probe 候选，不允许直接跑指标。

### 可视化结果

本阶段生成：

```text
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_fit_score.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_domain_fit.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_citation_fit.svg
artifacts/external_eval_dataset_discovery_stage55_visuals/stage55_candidate_effort_score.svg
```

完整 JSON artifact：

```text
artifacts/external_eval_dataset_discovery_stage55.json
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：新的测试集候选不能只看是否“技术支持相关”。
  - 原因：当前 runtime 选择依赖 citation 与 answerability 行为；
  - 只有问答对不等于可做 citation-preserving RAG 评估。
- 问题 2：MSQA 虽然域匹配强，但没有原生 no-answer rows。
  - 原因：MSQA README 说明过滤掉没有 accepted answer 的样本；
  - 所以后续如果使用 MSQA，不能直接复用当前 unanswerable refusal 指标。
- 问题 3：MSDialog 域也匹配，但当前不能直接用。
  - 原因：公开页要求联系 CIIR 获取访问权限，且公开页未确认可再分发许可证；
  - 因此必须标记为 blocked，而不是当成可用测试集。

### 修正与处理

- 新增结构化候选审计模型，避免后续只凭人工印象选择测试集。
- 把 MSQA 标为 `recommended_for_stage56_schema_probe`，不是 held-out。
- 把 MSDialog 标为 `blocked_until_access_and_license_confirmation`。
- 把 Natural Questions 标为 `control_benchmark_only`，避免把通用 QA 当技术支持测试。
- 把 Stack Exchange dumps 标为 `manual_derivation_candidate_only`，因为它需要额外构造协议和署名处理。
- 在 `docs/evaluation_strategy.md` 中把 Stage 54 的“等待用户确认”更新为“外部路线已确认，但指标仍阻断”。
- 在 `docs/data_strategy.md` 中新增 MSQA 候选边界。
- 新增 `docs/external_eval_datasets.md` 保存外部数据集发现快照。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\external_eval_dataset_discovery.py scripts\discover_external_eval_datasets.py tests\test_external_eval_dataset_discovery.py
pytest -q tests\test_external_eval_dataset_discovery.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

全量验证：

```powershell
ruff check .
pytest -q
```

结果：

```text
ruff: passed
pytest: 147 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\external_eval_dataset_discovery_stage55.json > $null
```

结果：通过。

### 结论

- Stage 55 完成 external evaluation dataset discovery。
- 新测试集候选是 MSQA，但只进入 Stage 56 schema/source-link/leakage probe。
- 本阶段没有使用 MSQA 作为测试集。
- 本阶段没有运行 held-out 指标。
- 默认 runtime 不变。
- Stage 51 candidate 继续保持 frozen non-default candidate。

### 我学到的

- 找测试集不是找“看起来像”的数据，而是要同时满足域、schema、citation、license、leakage 与 answerability 边界。
- 外部数据集即使域匹配，也不能跳过 source-link coverage audit；否则 citation-preserving policy 仍然无法诚实评估。
- MSQA 的 accepted-answer 设计适合技术支持 QA，但对 refusal/unanswerable 测试不够，需要后续单独处理。
- 审计分可以帮助排序，但必须明确它是生成的适配分，不是模型效果。

### 下一步

- 做 Stage 56：MSQA local schema probe、source-link coverage audit 与 PrimeQA leakage audit protocol。
- Stage 56 仍然不直接跑最终质量指标，先完成：
  1. 记录 MSQA 下载 URL、文件大小和 checksum；
  2. 采样解析 CSV header 与少量样本；
  3. 统计 URL/source-link 覆盖率；
  4. 统计 `learn.microsoft.com` 文档链接覆盖率；
  5. 设计 MSQA 与 PrimeQA train/dev 的 exact/near-duplicate leakage audit。

## 2026-07-14 Stage 56：MSQA 本地下载、schema probe 与 source-link 检查

### 目标

按 Stage 55 的推荐路线，把 Microsoft Q&A / MSQA 数据集真实下载到本地，并先做本地 schema/source-link/leakage 前置检查。

本阶段边界：

- 下载并解析 MSQA；
- 记录真实来源、commit、文件大小和 SHA256；
- 检查 CSV header、行数、缺失字段、split 分布、source-link 覆盖率；
- 对 PrimeQA train/dev 做 exact normalized question overlap 预检查；
- 生成 JSON artifact 与可视化结果；
- 不运行 RAG end-to-end 指标；
- 不比较 top-k 和 Stage 51；
- 不改变默认 runtime；
- 不把 MSQA 宣称为已经可用的 held-out test set。

### 新增与修改

新增 Stage 56 probe 模块：

```text
src/ts_rag_agent/application/msqa_schema_probe.py
```

新增 CLI：

```text
scripts/probe_msqa_dataset.py
```

新增测试：

```text
tests/test_msqa_schema_probe.py
```

新增文档：

```text
docs/msqa_schema_probe.md
```

更新文档：

```text
docs/data_strategy.md
docs/evaluation_strategy.md
docs/external_eval_datasets.md
```

### 下载事实

下载命令：

```powershell
git clone --depth 1 --filter=blob:none https://github.com/microsoft/Microsoft-Q-A-MSQA-.git data\raw\msqa_repo
```

真实本地路径：

```text
data/raw/msqa_repo/
```

该路径被 `.gitignore` 忽略，不纳入 git。

仓库 HEAD：

```text
4be7e0376f3fa2ee8cbaa90644bd0eeb291c43f4
```

文件 fingerprint：

```text
data/raw/msqa_repo/data/msqa-32k.csv
bytes: 106085656
sha256: 38839bb6234a2195216762ee7827e770d9d6de2eb49f913a365cc04dbdc20d93

data/raw/msqa_repo/data/test_id.txt
bytes: 4343
sha256: 644fa017e8abcabef91eacf3ef10f58b9118e959b9c6e4497779270183bc4ba1

data/raw/msqa_repo/README.md
bytes: 16664
sha256: 2dad24449874c91d29089a97e1381af3eb95f4f4c4fa07dfd5f5fe8acf92b0ac
```

### 运行命令

```powershell
python scripts\probe_msqa_dataset.py --msqa-csv data\raw\msqa_repo\data\msqa-32k.csv --test-id-file data\raw\msqa_repo\data\test_id.txt --readme data\raw\msqa_repo\README.md --repo-dir data\raw\msqa_repo --primeqa-train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json --primeqa-dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json --output artifacts\msqa_schema_probe_stage56.json --visualization-dir artifacts\msqa_schema_probe_stage56_visuals
```

### Stage 56 结果

Schema：

```text
row_count: 32236
README row count claim: 32252
row_count_delta_vs_readme_claim: -16
field_count: 29
unique_question_ids: 32236
duplicate_question_id_rows: 0
duplicate_normalized_question_rows: 11
malformed_row_count: 0
```

必需字段缺失：

```text
QuestionId: 0
AnswerId: 0
QuestionText: 0
AnswerText: 0
ProcessedAnswerText: 0
Url: 0
Split: 0
```

答案字段候选：

```text
AnswerText: 32236 / 32236 available
ProcessedAnswerText: 32236 / 32236 available
DoubleProcessedAnswerText: 32160 / 32236 available
```

Split：

```text
NNN: 21225
train: 7710
test: 3301
```

Domain flags：

```text
IsAzure=True: 11024
IsM365=True: 6912
IsOther=True: 15904
```

Source-link coverage：

```text
rows_with_row_url: 32236 / 32236 (100.0%)
rows_with_learn_answers_url: 32236 / 32236 (100.0%)
rows_with_question_text_link: 5307 / 32236 (16.463%)
rows_with_answer_text_link: 22024 / 32236 (68.321%)
rows_with_processed_answer_link: 19924 / 32236 (61.807%)
rows_with_double_processed_answer_link: 17048 / 32236 (52.885%)
rows_with_processed_answer_learn_link: 10929 / 32236 (33.903%)
rows_with_processed_answer_azure_docish_link: 4343 / 32236 (13.473%)
```

`test_id.txt`：

```text
test_id_count: 588
unique_test_id_count: 588
duplicate_test_id_count: 0
test_ids_found_in_csv: 587
test_ids_missing_from_csv_count: 1
test_ids_missing_from_csv: 699708
found_test_id_split_counts: test=587
```

PrimeQA exact leakage precheck：

```text
PrimeQA train questions: 600
PrimeQA dev questions: 310
exact_overlap_pair_count: 0
exact_overlap_msqa_question_count: 0
passed_exact_precheck: true
```

没有在 Stage 56 运行：

```text
near_duplicate_token_jaccard_search
semantic_duplicate_search
answer_or_document_overlap_search
```

### 可视化结果

```text
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_split_distribution.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_source_link_coverage.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_domain_flags.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_test_id_coverage.svg
artifacts/msqa_schema_probe_stage56_visuals/stage56_msqa_primeqa_exact_overlap.svg
```

完整 JSON artifact：

```text
artifacts/msqa_schema_probe_stage56.json
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：README 摘要行数是 32,252，但本地 CSV 实际解析为 32,236。
  - 原因：当前只确认到本地文件真实解析结果与 README 摘要不一致；本阶段不倒填、不修正上游数字。
  - 处理：在报告中同时记录 README claim 和本地 row count，并把差值 `-16` 标为 metrics 前阻塞项之一。
- 问题 2：`test_id.txt` 有 588 个 ID，但只有 587 个能在 CSV 中找到。
  - 缺失 ID：`699708`。
  - 原因：上游 split 流程还包含 Azure/长度过滤，本项目不能直接把 `test_id.txt` 当作最终 split。
  - 处理：记录真实覆盖率，不冻结评估 split。
- 问题 3：`DoubleProcessedAnswerText` 有 76 行缺失。
  - 原因：本地 CSV 字段本身缺失。
  - 处理：Stage 56 不实现任何答案字段兜底；下一步必须显式选择 adapter 的 answer field。
- 问题 4：exact overlap 为 0，但这还不是完整 leakage 结论。
  - 原因：near-duplicate、semantic duplicate、answer/document overlap 都没有运行。
  - 处理：只称为 exact-overlap precheck pass，不宣称 leakage fully passed。

### 修正与处理

- 将 MSQA 从 Stage 55 的“候选发现”推进到 Stage 56 的“本地 schema probe 完成”。
- 新增可复跑 probe 脚本，避免只保留一次性临时命令输出。
- 文档中明确区分 README claim、本地 CSV parse result、exact precheck 和尚未运行的 near-duplicate leakage。
- `docs/evaluation_strategy.md` 继续阻止任何 pseudo-held-out metrics。
- `docs/data_strategy.md` 记录 MSQA 已下载但仍不是 held-out。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_schema_probe.py scripts\probe_msqa_dataset.py tests\test_msqa_schema_probe.py
pytest -q tests\test_msqa_schema_probe.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

artifact JSON 校验：

```powershell
python -m json.tool artifacts\msqa_schema_probe_stage56.json > $null
```

结果：通过。

### 结论

- Stage 56 完成 MSQA 本地下载与 schema/source-link probe。
- MSQA 本地 CSV 可解析，字段完整性基本可用。
- 行级 Microsoft Learn Q&A URL 覆盖率为 100%。
- PrimeQA train/dev exact normalized question overlap 为 0。
- 但 MSQA 仍然不能作为最终 held-out test set 使用。
- 默认 runtime 不变。
- Stage 51 candidate 继续保持 frozen non-default candidate。

### 我学到的

- 外部数据集即使是公开 benchmark，也要把 README 摘要和本地文件解析结果分开记录；两者不一致时只能如实报告，不能替上游“修正”。
- `test_id.txt` 不是天然等于本项目 test split；必须结合上游 split 代码和本项目目标重新冻结 split。
- 行级 source URL 覆盖率和答案文本中的 documentation-link 覆盖率不是同一个概念；前者可支持来源归因，后者才更接近文档证据覆盖。
- exact overlap 为 0 是好信号，但它只覆盖最窄的 leakage 类型，不能替代 near-duplicate/semantic leakage 审计。

### 下一步

- 做 Stage 57：MSQA adapter contract、near-duplicate leakage audit 与 project-owned evaluation split freeze。
- Stage 57 仍然不直接跑最终指标，先完成：
  1. 明确 MSQA row -> 本项目 evaluation sample 的字段映射；
  2. 明确 answer field 使用 `AnswerText`、`ProcessedAnswerText` 还是其他字段；
  3. 明确 citation/source field 使用 row-level `Url` 的方式；
  4. 运行 near-duplicate leakage audit；
  5. 冻结本项目自己的 MSQA eval split。

## 2026-07-14 Stage 57：MSQA adapter contract、near-duplicate leakage audit 与 project-owned split freeze

### 目标

在 Stage 56 已完成 MSQA 本地 schema/source-link probe 的基础上，继续推进到可评估前的最后准备步骤：

- 明确 MSQA row 到本项目 evaluation sample 的字段映射；
- 明确 answer field，不做任何字段兜底；
- 明确 source/citation 边界；
- 对 MSQA 和 PrimeQA train/dev 做 near-duplicate leakage audit；
- 冻结本项目自己的 MSQA evaluation split；
- 仍然不运行 RAG answer-quality 指标；
- 仍然不比较 top-k 和 Stage 51；
- 仍然不改变默认 runtime。

### 新增与修改

新增 Stage 57 模块：

```text
src/ts_rag_agent/application/msqa_evaluation_split.py
```

新增 CLI：

```text
scripts/freeze_msqa_evaluation_split.py
```

新增测试：

```text
tests/test_msqa_evaluation_split.py
```

新增文档：

```text
docs/msqa_evaluation_split.md
```

更新文档：

```text
docs/data_strategy.md
docs/evaluation_strategy.md
docs/external_eval_datasets.md
docs/msqa_schema_probe.md
```

### Adapter contract

Contract version：

```text
msqa_eval_adapter_v1
```

字段映射：

```text
sample id: QuestionId
answer id: AnswerId
question: QuestionText
gold answer: ProcessedAnswerText
source URL: Url
source split: Split
metadata: Tags, IsAzure, IsM365, IsOther, isShort, isLong
```

明确禁止兜底：

```text
Do not fall back to AnswerText or DoubleProcessedAnswerText.
```

原因：

- Stage 56 已确认 `ProcessedAnswerText` 在 32,236 行中 0 缺失；
- `DoubleProcessedAnswerText` 有 76 行缺失；
- 使用多个 answer field 自动切换会让评估语义不稳定；
- 本项目 AGENTS 规则要求不能擅自加入兜底策略。

Citation/source 边界：

- 使用 row-level `Url` 作为 source URL；
- 不宣称 answer text 中的文档链接等于完整 documentation-span citation coverage；
- 不支持 native unanswerable/refusal evaluation；
- 不支持 document-span exact citation evaluation。

### Near-duplicate leakage audit

第一次尝试：

```text
使用现有通用 heldout leakage 函数对 32,236 * 910 做多阈值 near-duplicate 预跑。
```

实际情况：

```text
124 秒超时，没有形成有效 artifact。
```

处理：

- 没有把超时输出当作结论；
- 新增 MSQA 专用审计实现；
- 仍然使用相同的归一化规则与 token Jaccard 定义；
- 使用 development token inverted index；
- 使用 Jaccard 0.9 的长度上下界剪枝；
- 真实 artifact 由新实现生成。

最终运行命令：

```powershell
python scripts\freeze_msqa_evaluation_split.py --msqa-csv data\raw\msqa_repo\data\msqa-32k.csv --primeqa-train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json --primeqa-dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json --output artifacts\msqa_evaluation_split_stage57.json --split-output artifacts\msqa_evaluation_split_stage57.jsonl --visualization-dir artifacts\msqa_evaluation_split_stage57_visuals --near-duplicate-threshold 0.9
```

运行耗时：

```text
24.5 seconds
```

Leakage audit 结果：

```text
MSQA questions: 32236
PrimeQA train/dev questions: 910
PrimeQA train: 600
PrimeQA dev: 310
exact_overlap_count: 0
exact_overlap_pair_count: 0
near_duplicate_overlap_count: 0
near_duplicate_overlap_pair_count: 0
unhandled_overlap_count: 0
MSQA questions without detected overlap: 32236
```

### Project-owned split freeze

Split name：

```text
msqa_stage57_project_eval_v1
```

Protocol version：

```text
msqa_project_eval_split_v1
```

Selection rule：

```text
Use MSQA rows whose CSV Split is 'test', then exclude rows with invalid row-level Microsoft Learn Q&A URLs, internal normalized-question duplicates, or detected PrimeQA exact/near-duplicate leakage.
```

过滤结果：

```text
loaded_contract_rows: 32236
source_split_candidates: 3301
excluded_invalid_source_url: 0
excluded_internal_normalized_duplicates: 0
excluded_primeqa_leakage: 0
selected_question_count: 3301
```

Selected domain counts：

```text
IsAzure=True: 3301
IsM365=True: 490
IsOther=True: 0
isShort=True: 285
isLong=True: 0
```

Selected question IDs checksum：

```text
26cab0b636845cd321a48c12e8bcbeb5b563e5eb234e63383bbc9d0a9d8cb93b
```

Frozen split JSONL checksum：

```text
b2beb8f20351999ee38c8679e37619da2a005d635d116ff0dedabf14f9600e54
```

JSONL 行数：

```text
3301
```

### 可视化结果

```text
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_leakage_counts.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_split_filter_counts.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_selected_domain_flags.svg
artifacts/msqa_evaluation_split_stage57_visuals/stage57_msqa_adapter_field_coverage.svg
```

完整 artifacts：

```text
artifacts/msqa_evaluation_split_stage57.json
artifacts/msqa_evaluation_split_stage57.jsonl
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：朴素 near-duplicate 审计超时。
  - 原因：32,236 个 MSQA rows 与 910 个 PrimeQA rows 做全量 token-set 比较，且多阈值预跑导致成本过高。
  - 处理：不记录无效超时结果；实现等价定义下的 inverted-index + length-bound pruning。
- 问题 2：split freeze 不能直接使用 `test_id.txt`。
  - 原因：Stage 56 已发现 `test_id.txt` 有 1 个 ID 不在 CSV，且上游 split 流程有额外过滤假设。
  - 处理：本项目冻结自己的 split：以 CSV `Split == test` 为候选池，再应用本项目字段、URL、重复和泄漏过滤。
- 问题 3：是否排除 MSQA 全局内部重复需要定义清楚。
  - 原因：评估 split 只使用 `Split == test`，本阶段不使用 MSQA train 做训练。
  - 处理：内部 normalized duplicate 过滤限定在 evaluation candidate pool 内部，而不是全 MSQA。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_evaluation_split.py scripts\freeze_msqa_evaluation_split.py tests\test_msqa_evaluation_split.py
pytest -q tests\test_msqa_evaluation_split.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

artifact 校验：

```powershell
python -m json.tool artifacts\msqa_evaluation_split_stage57.json > $null
(Get-Content artifacts\msqa_evaluation_split_stage57.jsonl | Measure-Object -Line).Lines
Get-FileHash -Algorithm SHA256 artifacts\msqa_evaluation_split_stage57.jsonl
```

结果：

```text
json: passed
jsonl lines: 3301
jsonl sha256: b2beb8f20351999ee38c8679e37619da2a005d635d116ff0dedabf14f9600e54
```

### 结论

- Stage 57 完成 MSQA adapter contract。
- Stage 57 完成 PrimeQA train/dev near-duplicate leakage audit。
- Stage 57 冻结项目自有 MSQA evaluation split：`msqa_stage57_project_eval_v1`。
- 近重复泄漏审计结果为 0 exact overlap、0 near-duplicate overlap。
- 冻结 split 有 3,301 行。
- 下一步可以在该 frozen split 上跑 top-k baseline。
- 仍然不能运行 Stage 51 comparison。
- 仍然不能 defaultize runtime。

### 我学到的

- “跑完整审计”不等于用最朴素实现硬跑；只要定义不变，工程化索引和剪枝是必要的，否则超时本身会污染实验流程。
- Split freeze 必须写清楚候选池、过滤项和 checksum，否则后续跑指标时很容易滑回“临时抽样”。
- 适配器字段选择是一种评估协议决定，不能在 evaluator 内部偷偷兜底。
- 外部测试集进入指标前至少要经历 source fingerprint、adapter contract、leakage audit、split freeze 四道门。

### 下一步

- 做 Stage 58：在 `msqa_stage57_project_eval_v1` 上运行 top-k baseline。
- Stage 58 只跑 baseline，不跑 Stage 51 comparison，不改默认 runtime。
- Stage 58 需要记录：
  1. baseline 使用的 frozen split checksum；
  2. top-k baseline 的 answer/citation 可评估字段边界；
  3. baseline 指标；
  4. baseline failure modes；
  5. 是否允许下一阶段对同一 frozen split 运行 Stage 51 candidate。

## 2026-07-14 Stage 58：MSQA frozen-split top-k answer-source baseline

### 目标

在 Stage 57 冻结的 `msqa_stage57_project_eval_v1` 上运行第一个 MSQA top-k baseline，并记录 baseline quality 与 failure modes。

本阶段边界：

- 只运行 MSQA frozen split baseline；
- 不运行 Stage 51 candidate；
- 不做默认 runtime 修改；
- 不把 MSQA answer-source 指标伪装成 PrimeQA-style document-grounded verified RAG 指标；
- 如实记录 artifact 修复和超时尝试。

### 新增与修改

新增 Stage 58 baseline 模块：

```text
src/ts_rag_agent/application/msqa_baseline_evaluation.py
```

新增 CLI：

```text
scripts/evaluate_msqa_topk_baseline.py
```

新增测试：

```text
tests/test_msqa_baseline_evaluation.py
```

更新 Stage 57 split writer：

```text
src/ts_rag_agent/application/msqa_evaluation_split.py
```

更新文档：

```text
docs/msqa_topk_baseline.md
docs/msqa_evaluation_split.md
docs/data_strategy.md
docs/evaluation_strategy.md
docs/external_eval_datasets.md
```

### Stage 57 JSONL artifact 修复

运行 Stage 58 时首先发现：

```text
artifacts/msqa_evaluation_split_stage57.jsonl
```

不能被逐物理行稳定解析。

原因：

- 少数 MSQA 字符串包含 Unicode line separator `U+2028`；
- JSON 对象本身可解析，但 Python `splitlines()` 会把 `U+2028` 当作物理换行；
- Stage 57 只校验了 JSON report 和 JSONL 行数，没有逐 JSON object 校验。

处理：

- 修改 Stage 57 JSONL writer，把 `U+2028` 和 `U+2029` 显式转义；
- 修改 Stage 58 loader，用 `\n` 做物理行切分；
- 重跑 Stage 57 artifact；
- selected question ID checksum 没有变化；
- JSONL artifact checksum 发生变化，这是真实的 Stage 58 过程修正。

新的 JSONL checksum：

```text
a60db5be5b1a6bfbf24d32ffc99c5482f57ad3462c39cd7b8510cc3c8d569bb3
```

逐对象校验：

```text
jsonl objects: 3301
physical splitlines: 3301
```

### Baseline 定义

MSQA 没有 PrimeQA technotes 那种独立文档语料，只有 Q&A rows 和 row-level URL。

因此 Stage 58 运行的是 answer-source retrieval baseline：

```text
query: frozen MSQA question
gold source: same frozen MSQA Q&A row
gold answer: ProcessedAnswerText
```

指标边界：

- hit@k / MRR：gold MSQA Q&A source row 是否进入 top-k；
- top1 token F1：top1 source row answer 与 gold `ProcessedAnswerText` 的 token F1；
- oracle@k token F1：top-k 中最佳 answer 与 gold answer 的 token F1；
- 这些不是 document-span citation metrics。

Baseline variants：

```text
primary: answer_only
diagnostic: question_answer_page_text
```

`answer_only` 只索引 `ProcessedAnswerText`，不索引 question text。

`question_answer_page_text` 索引 `QuestionText + ProcessedAnswerText`，模拟 Q&A 页面文本；这个版本预期更容易，只作为诊断。

### 真实运行过程

第一次真实运行尝试：

```text
corpus_scope: all_contract_rows
corpus rows: 32236
eval rows: 3301
```

结果：

```text
304 seconds timeout
no valid Stage 58 artifact produced
```

处理：

- 不把这次超时当指标；
- 改为 `corpus_scope=frozen_split_only`，与 Stage 57 “在 frozen split 上跑 baseline” 的边界一致；
- 记录全量 corpus scope 作为未来单独实验，不混入 Stage 58。

正式运行命令：

```powershell
python scripts\evaluate_msqa_topk_baseline.py --msqa-csv data\raw\msqa_repo\data\msqa-32k.csv --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl --output artifacts\msqa_topk_baseline_stage58.json --visualization-dir artifacts\msqa_topk_baseline_stage58_visuals --top-k 1,3,5,10 --corpus-modes answer_only,question_answer_page_text --corpus-scope frozen_split_only
```

### Stage 58 结果

输入：

```text
corpus_scope: frozen_split_only
corpus rows: 3301
frozen split samples: 3301
```

Primary baseline：`answer_only`

```text
evaluated_questions: 3301
hit@1: 0.4147
hit@3: 0.5159
hit@5: 0.5604
hit@10: 0.6128
MRR: 0.4762
median_gold_rank_when_found: 1
gold_source_missing_at_10: 1278
top1_wrong_source: 1932
average_top1_token_f1: 0.5138
oracle@1 token_f1: 0.5138
oracle@3 token_f1: 0.6137
oracle@5 token_f1: 0.6541
oracle@10 token_f1: 0.7024
top1_token_f1_below_0.3: 1758
```

Diagnostic baseline：`question_answer_page_text`

```text
evaluated_questions: 3301
hit@1: 1.0
hit@3: 1.0
hit@5: 1.0
hit@10: 1.0
MRR: 1.0
average_top1_token_f1: 1.0
```

解释：

- `question_answer_page_text` 达到 1.0，说明只要把原问题文本放入索引，source-row retrieval 会变得过于容易；
- 因此 primary evidence 应看 `answer_only`；
- `answer_only` hit@10 只有 0.6128，说明仅靠答案文本从问题检索 accepted-answer source row 仍有明显困难；
- Stage 51 comparison 不能自动运行，因为 Stage 51 是围绕 PrimeQA document-grounded verified RAG 设计的，而 Stage 58 是 MSQA answer-source retrieval task。

### 可视化结果

```text
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_hit_at_1.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_hit_at_10.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_mrr.svg
artifacts/msqa_topk_baseline_stage58_visuals/stage58_msqa_top1_answer_f1.svg
```

完整 artifact：

```text
artifacts/msqa_topk_baseline_stage58.json
```

artifact checksum：

```text
d114f5f3ad1a4e9680a6296c59d66150770d94605c652fea4e2e8e9039897234
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：Stage 57 JSONL 起初不能逐物理行稳定解析。
  - 原因：`U+2028` / `U+2029` line separator 没有额外转义；
  - 处理：更新 writer 并重跑 ignored artifact。
- 问题 2：全量 32,236 source-row corpus baseline 超时。
  - 原因：当前 Python BM25 对 3,301 queries 与 32k source rows 的 answer-source 检索成本过高；
  - 处理：Stage 58 改为 frozen-split-only corpus，符合“在 frozen split 上跑 baseline”的阶段边界；全量 corpus 作为未来单独性能/检索实验。
- 问题 3：answer-only 和 page-text 的指标差异极大。
  - 原因：page-text 索引包含原始 question，source row 自匹配太强；
  - 处理：只把 answer-only 作为 primary baseline，page-text 标记为 diagnostic。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_baseline_evaluation.py src\ts_rag_agent\application\msqa_evaluation_split.py scripts\evaluate_msqa_topk_baseline.py tests\test_msqa_baseline_evaluation.py tests\test_msqa_evaluation_split.py
pytest -q tests\test_msqa_baseline_evaluation.py tests\test_msqa_evaluation_split.py
```

结果：

```text
ruff: passed
pytest: 5 passed
```

artifact 校验：

```powershell
python -m json.tool artifacts\msqa_topk_baseline_stage58.json > $null
python -m json.tool artifacts\msqa_evaluation_split_stage57.json > $null
```

结果：

```text
json: passed
```

JSONL 校验：

```text
jsonl objects: 3301
physical splitlines: 3301
jsonl sha256: a60db5be5b1a6bfbf24d32ffc99c5482f57ad3462c39cd7b8510cc3c8d569bb3
```

### 结论

- Stage 58 完成 MSQA frozen-split top-k answer-source baseline。
- Primary answer-only baseline：hit@10 0.6128，MRR 0.4762，average top1 token F1 0.5138。
- Diagnostic question+answer page-text baseline 达到 1.0，但这是因为索引包含原问题文本，不能作为主判断依据。
- Stage 51 candidate 仍然不能直接运行。
- 默认 runtime 不变。

### 我学到的

- JSONL 不只是“看起来一行一个对象”；还要考虑 `U+2028/U+2029` 这类 line separator 对 line-oriented tools 的影响。
- 外部 QA 数据集如果没有独立文档语料，baseline 语义会从 document-grounded RAG 变成 answer-source retrieval，必须在指标名前写清楚。
- 把 question text 放入 source-row index 会造成强自匹配，能作为诊断上限，但不能作为主 baseline。
- 在跑 candidate 之前先看 baseline failure modes 是必要的，否则很容易拿不可比任务强行做模型选择。

### 下一步

- 做 Stage 59：MSQA baseline failure mode review 与 Stage 51 compatibility gate。
- Stage 59 先判断：
  1. Stage 51 的 evidence-selection / candidate-score 机制是否能公平迁移到 MSQA answer-source task；
  2. 是否需要先设计 MSQA document/corpus construction，而不是直接跑 Stage 51；
  3. 如果可以比较，应该比较 answer-only corpus 还是 page-text corpus；
  4. 是否需要先优化 BM25 performance，再考虑全量 32k corpus baseline。

## 2026-07-14 Stage 59：MSQA failure mode review 与 Stage 51 compatibility gate

### 本阶段目标

Stage 58 已经得到 MSQA frozen-split answer-source top-k baseline。本阶段目标不是运行 Stage 51 candidate，而是先做兼容性门禁：

1. 复查 Stage 58 answer-only baseline 的失败模式；
2. 明确 Stage 51 PrimeQA document-grounded composition policy 是否能直接用于 MSQA answer-source task；
3. 判断 `question_answer_page_text` diagnostic variant 能否作为候选比较目标；
4. 若不能直接比较，记录阻塞原因、可视化结果和下一阶段协议设计方向；
5. 继续保持默认 runtime 不变。

### Stage 59 预检中的真实 artifact 修正

本阶段开始时发现一个必须如实记录的问题：

```text
artifacts/msqa_topk_baseline_stage58.json
```

本地 ignored artifact 一度不是 Stage 58 文档中记录的 frozen-split-only 结果，而是旧的 all-contract-row 尝试残留：

```text
corpus_contract_rows: 32236
hit@1: 0.3196
hit@10: 0.5068
missing corpus_scope
LastWriteTime: 2026-07-14 13:37:14
```

这与 Stage 58 已记录结论不一致，因此我没有继续基于该旧 artifact 做 Stage 59，而是在 Stage 59 预检中按 frozen-split-only 命令真实重跑 Stage 58：

```powershell
python scripts\evaluate_msqa_topk_baseline.py `
  --msqa-csv data\raw\msqa_repo\data\msqa-32k.csv `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --output artifacts\msqa_topk_baseline_stage58.json `
  --visualization-dir artifacts\msqa_topk_baseline_stage58_visuals `
  --top-k 1,3,5,10 `
  --corpus-modes answer_only,question_answer_page_text `
  --corpus-scope frozen_split_only
```

重跑后 artifact 恢复为当前一致版本：

```text
corpus_scope: frozen_split_only
frozen_split_samples: 3301
answer_only hit@10: 0.6128
answer_only MRR: 0.4762
answer_only average_top1_token_f1: 0.5138
question_answer_page_text hit@1/hit@10/MRR: 1.0
current Stage 58 report checksum: f34f1d749d94ff08e2a62f3a22b58ec9804cddea4535d971c4618666b65a4dd8
```

说明：这是 Stage 59 期间发生的预检修正，不伪装成 Stage 58 原本就已发生。由于 report 中包含 timing 字段，重跑后 checksum 与 Stage 58 文档之前记录的 checksum 不同。

### 新增与修改

新增 Stage 59 compatibility review 模块：

```text
src/ts_rag_agent/application/msqa_stage51_compatibility_review.py
```

新增能力：

- 读取 Stage 58 baseline report；
- 验证 report stage、primary variant、diagnostic variant、frozen split sample 数；
- 提取 `answer_only` 失败模式；
- 计算 primary 与 diagnostic 的指标差距；
- 输出 7 项 compatibility gate checks；
- 将 gate 结果转换为是否允许运行 Stage 51 candidate 的明确 decision；
- 生成 Stage 59 SVG 可视化。

新增 CLI：

```text
scripts/review_msqa_stage51_compatibility.py
```

新增测试：

```text
tests/test_msqa_stage51_compatibility_review.py
```

新增文档：

```text
docs/msqa_stage51_compatibility.md
```

同步更新：

```text
docs/msqa_topk_baseline.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/external_eval_datasets.md
docs/learning_journal.md
```

### Stage 59 运行命令

```powershell
python scripts\review_msqa_stage51_compatibility.py `
  --stage58-report artifacts\msqa_topk_baseline_stage58.json `
  --output artifacts\msqa_stage51_compatibility_stage59.json `
  --visualization-dir artifacts\msqa_stage51_compatibility_stage59_visuals
```

### Stage 59 结果

输入摘要：

```text
split_name: msqa_stage57_project_eval_v1
adapter_contract_version: msqa_eval_adapter_v1
corpus_scope: frozen_split_only
evaluated_questions: 3301
primary_variant: answer_only
diagnostic_variant: question_answer_page_text
max_k: 10
```

answer-only failure mode：

```text
gold_source_missing_at_10: 1278 / 3301 = 0.3872
top1_wrong_source: 1932 / 3301 = 0.5853
top1_token_f1_below_0_3: 1758 / 3301 = 0.5326
```

primary vs diagnostic gap：

```text
hit@1 gap: 0.5853
hit@10 gap: 0.3872
MRR gap: 0.5238
average_top1_token_f1 gap: 0.4862
```

compatibility gate：

```text
total_checks: 7
pass: 2
blocked: 5
blocker_count: 5
```

blocked checks：

```text
stage51_task_semantics_match_msqa
citation_identity_contract_match
candidate_feature_contract_available
diagnostic_variant_usable_for_comparison
failure_modes_are_policy_test_ready
```

decision：

```text
status: stage51_msqa_compatibility_blocked
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
rejected_comparison_variant: question_answer_page_text
recommended_next_stage: Stage 60: design the MSQA source/citation adapter and comparison protocol before any Stage 51 candidate run
```

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_stage51_gate_checks.svg
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_answer_only_failure_modes.svg
artifacts/msqa_stage51_compatibility_stage59_visuals/stage59_msqa_variant_metric_comparison.svg
```

Stage 59 report：

```text
artifacts/msqa_stage51_compatibility_stage59.json
```

Stage 59 report checksum：

```text
9f72f74262ee4c2da0613e2482043366049051aae3a6ea647b617f2bfd6d79b2
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：Stage 59 预检时发现 Stage 58 本地 ignored artifact 与已记录 frozen-split 指标不一致。
  - 原因：本地 ignored artifact 残留了旧的 all-contract-row 尝试结果；
  - 处理：在 Stage 59 期间真实重跑 Stage 58 frozen-split-only 命令，并在文档中记录这是 Stage 59 预检修正。
- 问题 2：Stage 51 不能直接跑在当前 MSQA task 上。
  - 原因：Stage 51 的输入/策略围绕 PrimeQA `SentenceEvidenceCandidate`、文档 ID、citation rank 和 candidate score；当前 MSQA Stage 58 是 Q&A row answer-source retrieval，没有独立 document-span citation contract。
- 问题 3：`question_answer_page_text` 指标不能作为 candidate comparison。
  - 原因：它把 question text 放入 index，gold row 自匹配过强，hit@1/MRR/F1 全部达到 1.0；
  - 处理：Stage 59 明确将其标记为 rejected comparison variant。
- 问题 4：answer-only baseline failure modes 主要是 retrieval/source-row 层面问题。
  - 证据：1278 个 gold source missing@10，1932 个 top1 wrong source，1758 个 top1 low-F1；
  - 处理：先设计 MSQA source/citation adapter 与 candidate construction protocol，再考虑 candidate policy。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_stage51_compatibility_review.py scripts\review_msqa_stage51_compatibility.py tests\test_msqa_stage51_compatibility_review.py
pytest -q tests\test_msqa_stage51_compatibility_review.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

Stage 59 真实 artifact 生成：

```text
python scripts\review_msqa_stage51_compatibility.py --stage58-report artifacts\msqa_topk_baseline_stage58.json --output artifacts\msqa_stage51_compatibility_stage59.json --visualization-dir artifacts\msqa_stage51_compatibility_stage59_visuals
```

结果：

```text
json report: generated
svg visualizations: 3 generated
```

### 结论

- Stage 59 完成 MSQA failure mode review 与 Stage 51 compatibility gate。
- Stage 51 不能直接在当前 MSQA Stage 58 answer-source task 上运行。
- `question_answer_page_text` diagnostic variant 不能作为 candidate comparison target。
- 默认 runtime 不变。
- 当前还没有得到可以支持 defaultization 的 held-out test evidence。
- 下一步不是跑 Stage 51，而是设计 MSQA-compatible source/citation adapter 和 comparison protocol。

### 我学到的

- ignored artifact 不能因为“上一步已提交”就默认可信；跨阶段开始前必须抽查真实 artifact 内容，尤其是 artifact 不进 git 时。
- 外部数据集的 schema 兼容不等于任务兼容。MSQA 有 question/answer/source URL，但这还不是 PrimeQA-style document-grounded citation task。
- diagnostic variant 可以帮助理解上限，但如果它把 query 本身写进 index，就不能拿来做候选策略评测。
- composition policy 的比较前提是候选对象一致。没有 MSQA evidence candidate contract 时，直接跑 Stage 51 会把 retrieval task、citation task 和 reranker task 混在一起。

### 下一步

- 做 Stage 60：MSQA source/citation adapter 与 comparison protocol 设计。
- Stage 60 目标：
  1. 定义 MSQA 上的 source/citation identity；
  2. 设计如何从 MSQA Q&A row 构造 evidence candidates；
  3. 明确是否使用 answer sentence、processed answer chunk、source URL 或 Microsoft Learn link 作为 citation unit；
  4. 冻结 baseline/candidate 共享的 comparison protocol；
  5. 继续不运行 Stage 51 candidate，直到 Stage 60 protocol 明确通过。

## 2026-07-14 Stage 60：MSQA source/citation adapter 与 comparison protocol 设计

### 本阶段目标

Stage 59 已经明确：不能直接把 Stage 51 PrimeQA document-grounded evidence composition policy 跑到当前 MSQA answer-source task 上。本阶段目标是先做协议设计，而不是实现 adapter 或运行 Stage 51：

1. 复用 Stage 56 的真实 source-link coverage；
2. 复用 Stage 57 的 adapter contract 与 frozen split；
3. 复用 Stage 59 的 compatibility gate 结果；
4. 比较 source/citation identity 选项；
5. 比较 candidate construction 选项；
6. 产出推荐协议，但由于这是设计选择，按项目规则标记为需要用户确认；
7. 继续不运行 Stage 51 candidate，不改变默认 runtime。

### 新增与修改

新增 Stage 60 protocol design 模块：

```text
src/ts_rag_agent/application/msqa_stage51_protocol_design.py
```

新增能力：

- 读取并校验 Stage 56 / Stage 57 / Stage 59 artifacts；
- 从真实 Stage 56 report 中读取覆盖率：
  - `rows_with_row_url`
  - `rows_with_processed_answer_link`
  - `rows_with_processed_answer_learn_link`
  - `rows_with_processed_answer_azure_docish_link`
- 从 Stage 57 report 中读取 frozen split 与 adapter contract；
- 从 Stage 59 report 中确认 direct Stage 51 comparison 仍被 blocked；
- 用 generated protocol-fit rubric 比较 source identity options；
- 用同一 rubric 比较 candidate construction options；
- 输出 recommended protocol；
- 明确 `requires_user_confirmation: true`；
- 生成 SVG 可视化。

新增 CLI：

```text
scripts/design_msqa_stage51_protocol.py
```

新增测试：

```text
tests/test_msqa_stage51_protocol_design.py
```

新增文档：

```text
docs/msqa_stage51_protocol.md
```

同步更新：

```text
docs/msqa_stage51_compatibility.md
docs/msqa_topk_baseline.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/external_eval_datasets.md
docs/learning_journal.md
```

### Stage 60 运行命令

```powershell
python scripts\design_msqa_stage51_protocol.py `
  --schema-probe artifacts\msqa_schema_probe_stage56.json `
  --evaluation-split artifacts\msqa_evaluation_split_stage57.json `
  --compatibility-review artifacts\msqa_stage51_compatibility_stage59.json `
  --output artifacts\msqa_stage51_protocol_stage60.json `
  --visualization-dir artifacts\msqa_stage51_protocol_stage60_visuals
```

### Stage 60 结果

当前约束：

```text
frozen_split: msqa_stage57_project_eval_v1
adapter_contract_version: msqa_eval_adapter_v1
selected_question_count: 3301
approved_answer_field: ProcessedAnswerText
approved_source_url_field: Url
no_answer_field_fallback: true
question_text_index_rejected: true
stage59_blocker_count: 5
default_runtime_policy: unchanged
```

source/citation identity 选项：

```text
msqa_row_source_url:
  status: recommended_for_user_confirmation
  coverage_percent: 100.0
  total_score: 9

processed_answer_links:
  status: blocked
  coverage_percent: 61.807
  total_score: 8

processed_answer_learn_links:
  status: blocked
  coverage_percent: 33.903
  total_score: 7

processed_answer_azure_docish_links:
  status: blocked
  coverage_percent: 13.473
  total_score: 7

question_answer_page_text:
  status: rejected
  coverage_percent: 100.0
  total_score: 4
```

candidate construction 选项：

```text
processed_answer_sentence_candidates:
  status: recommended_for_user_confirmation
  coverage_percent: 100.0
  total_score: 9

processed_answer_chunk_candidates:
  status: secondary_option
  coverage_percent: 100.0
  total_score: 8

source_row_single_candidate:
  status: secondary_option
  coverage_percent: 100.0
  total_score: 8

linked_learn_document_candidates:
  status: blocked
  coverage_percent: 33.903
  total_score: 6

question_answer_text_candidates:
  status: rejected
  coverage_percent: 100.0
  total_score: 5
```

推荐协议：

```text
protocol_status: draft_requires_user_confirmation
source_citation_identity: msqa_row_source_url
candidate_construction: processed_answer_sentence_candidates
retrieval_corpus_scope: frozen_split_only
retrieval_index_text: ProcessedAnswerText only
excluded_index_text: QuestionText
gold_source_identity: QuestionId + AnswerId + Url
candidate_identity: QuestionId::processed_answer_sentence::<one_based_sentence_index>
```

下一阶段 adapter 必须具备字段：

```text
question_id
answer_id
source_url
candidate_id
candidate_sentence
retrieval_rank
retrieval_score
candidate_score
source_row_id
```

明确排除：

```text
Do not use AnswerText fallback.
Do not use DoubleProcessedAnswerText fallback.
Do not index QuestionText for candidate comparison.
Do not require processed-answer links as citation ground truth.
Do not fetch external pages in this protocol.
Do not change the default runtime.
```

decision：

```text
status: msqa_stage51_protocol_ready_for_user_confirmation
requires_user_confirmation: true
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
recommended_source_citation_identity: msqa_row_source_url
recommended_candidate_construction: processed_answer_sentence_candidates
```

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_source_identity_scores.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_candidate_construction_scores.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_source_coverage.svg
artifacts/msqa_stage51_protocol_stage60_visuals/stage60_decision_flags.svg
```

Stage 60 report：

```text
artifacts/msqa_stage51_protocol_stage60.json
```

Stage 60 report checksum：

```text
2267f1ac14e0866eb4c4835f40a06124d00833503a0c756bc06f0e891983db25
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：Stage 60 涉及真实设计选择，不能擅自冻结成最终实施方案。
  - 原因：用户规则要求对模糊或特殊落地方案列出选项并确认；
  - 处理：本阶段产出推荐协议，但 decision 中明确 `requires_user_confirmation: true`。
- 问题 2：processed-answer links 更接近 document citation，但不能作为当前 required citation identity。
  - 原因：真实覆盖率只有 `61.807%`，learn links 只有 `33.903%`，azure doc-like links 只有 `13.473%`；
  - 处理：全部标记为 blocked，不引入 fallback。
- 问题 3：`question_answer_page_text` 虽然覆盖率 100%，但不能进入比较协议。
  - 原因：Stage 58/59 已显示 question text 自匹配导致 diagnostic variant 达到 1.0；
  - 处理：标记为 rejected，推荐协议显式排除 `QuestionText`。
- 问题 4：`source_row_single_candidate` 便于 smoke test，但不适合作为真实 Stage 51 reranker comparison。
  - 原因：单行单候选缺少 candidate competition，基本退化为 Stage 58 source-row retrieval；
  - 处理：保留为 secondary option，不作为推荐 candidate construction。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_stage51_protocol_design.py scripts\design_msqa_stage51_protocol.py tests\test_msqa_stage51_protocol_design.py
pytest -q tests\test_msqa_stage51_protocol_design.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

Stage 60 真实 artifact 生成：

```text
python scripts\design_msqa_stage51_protocol.py --schema-probe artifacts\msqa_schema_probe_stage56.json --evaluation-split artifacts\msqa_evaluation_split_stage57.json --compatibility-review artifacts\msqa_stage51_compatibility_stage59.json --output artifacts\msqa_stage51_protocol_stage60.json --visualization-dir artifacts\msqa_stage51_protocol_stage60_visuals
```

结果：

```text
json report: generated
svg visualizations: 4 generated
```

### 结论

- Stage 60 完成 MSQA source/citation adapter 与 comparison protocol 设计。
- 推荐协议是：

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

- 但该协议仍处于：

```text
draft_requires_user_confirmation
```

- 当前不能实现 adapter、不能运行 Stage 51 candidate、不能 defaultize。
- 默认 runtime 不变。

### 我学到的

- “看起来更像 document citation”的 link-based 方案，如果覆盖率不足，就不能在 no-fallback 规则下作为 required citation identity。
- 对 MSQA 当前数据，最诚实的 source identity 是 row-level `QuestionId + AnswerId + Url`，它不是 document-span citation，必须把这个边界写清楚。
- Stage 51 comparison 的前提不是“能构造一个输入对象”，而是 baseline 和 candidate 在同一个 source/citation identity、同一个 candidate unit、同一个 metric boundary 下比较。
- 设计阶段也要产出 artifact 和可视化，方便后续确认时知道每个方案为什么被推荐、阻塞或拒绝。

### 下一步

- 请用户确认是否按推荐协议进入 Stage 61：

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

- 若确认，Stage 61 做 MSQA row-source answer-sentence candidate adapter 与 dry-run contract tests。
- Stage 61 仍然不跑最终 Stage 51 comparison；它只验证 adapter contract、candidate fields、样本稳定性和 no-fallback/no-question-text 约束。

## 2026-07-14 Stage 61：MSQA row-source answer-sentence candidate adapter dry-run

### 本阶段目标

用户已确认 Stage 60 推荐方案 A：

```text
msqa_row_source_url + processed_answer_sentence_candidates
```

本阶段目标是实现 adapter dry-run，而不是运行 Stage 51 candidate：

1. 强制要求显式 `--confirmed-protocol`；
2. 从 Stage 57 frozen split JSONL 读取 MSQA rows；
3. 仅使用 `ProcessedAnswerText` 构建 answer-only BM25 source-row index；
4. 对每个 query 检索 top10 source rows；
5. 将 retrieved source row 的 `ProcessedAnswerText` 拆成 answer sentence candidates；
6. 生成 Stage 60 要求的 candidate fields；
7. 验证 no fallback、no question-text、no external fetch；
8. 输出 dry-run report、candidate JSONL 和 SVG 可视化；
9. 继续不运行 Stage 51，不改变默认 runtime。

### 新增与修改

新增 Stage 61 adapter 模块：

```text
src/ts_rag_agent/application/msqa_stage51_candidate_adapter.py
```

新增能力：

- `MsqaStage51AdapterSample`：读取 Stage 57 JSONL 中的 row-source sample；
- `MsqaStage51CandidateRow`：表示 row-source answer-sentence candidate；
- 强制校验 Stage 60 protocol report；
- 强制 `confirmed_protocol=True`；
- 使用 answer-only BM25 source-row retrieval；
- 不把 `QuestionText` 写入 candidate rows；
- 不使用 `AnswerText` / `DoubleProcessedAnswerText` fallback；
- 不抓取外部页面；
- 生成 dry-run report；
- 输出 full candidate JSONL；
- 生成 SVG 可视化。

新增 CLI：

```text
scripts/dry_run_msqa_stage51_candidate_adapter.py
```

新增测试：

```text
tests/test_msqa_stage51_candidate_adapter.py
```

新增文档：

```text
docs/msqa_stage51_candidate_adapter.md
```

同步更新：

```text
docs/msqa_stage51_protocol.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/external_eval_datasets.md
docs/learning_journal.md
```

### Stage 61 运行命令

```powershell
python scripts\dry_run_msqa_stage51_candidate_adapter.py `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --protocol-report artifacts\msqa_stage51_protocol_stage60.json `
  --output artifacts\msqa_stage51_candidate_adapter_stage61.json `
  --candidate-output artifacts\msqa_stage51_candidate_adapter_stage61_candidates.jsonl `
  --visualization-dir artifacts\msqa_stage51_candidate_adapter_stage61_visuals `
  --confirmed-protocol `
  --top-k 10 `
  --min-sentence-chars 1 `
  --sample-limit 20
```

### Stage 61 结果

用户确认记录：

```text
confirmed_protocol_option: A
confirmed_source_citation_identity: msqa_row_source_url
confirmed_candidate_construction: processed_answer_sentence_candidates
confirmation_source: current Codex conversation on 2026-07-14
```

adapter contract：

```text
source_citation_identity: msqa_row_source_url
candidate_construction: processed_answer_sentence_candidates
retrieval_index_text: ProcessedAnswerText only
excluded_index_text: QuestionText
top_k: 10
min_sentence_chars: 1
no_answer_field_fallback: true
external_fetch_used: false
```

dry-run summary：

```text
evaluation_samples: 3301
candidate_rows: 266647
samples_with_candidates: 3301
samples_without_candidates: 0
samples_with_gold_source_candidate: 2023
average_candidates_per_sample: 80.7776
median_candidates_per_sample: 79.0
unique_source_rows_in_candidates: 2879
```

source retrieval summary：

```text
hit@1: 0.4147
hit@10: 0.6128
MRR: 0.4762
gold_source_missing_at_10: 1278
```

contract checks：

```text
user_confirmed_stage60_protocol: pass
protocol_matches_stage60_recommendation: pass
no_question_text_indexed_or_written_to_candidates: pass
no_answer_field_fallback_used: pass
no_external_fetch_used: pass
all_candidates_have_required_fields: pass
all_samples_have_candidate_rows: pass
```

candidate JSONL 校验：

```text
jsonl rows: 266647
rows_with_question_key: 0
```

decision：

```text
status: msqa_stage51_candidate_adapter_dry_run_passed
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
recommended_next_stage: Stage 62: review MSQA adapter candidate distribution and decide whether a single Stage 51 adapter comparison is fair
```

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_candidate_counts.svg
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_source_hit_rates.svg
artifacts/msqa_stage51_candidate_adapter_stage61_visuals/stage61_adapter_contract_checks.svg
```

Stage 61 report：

```text
artifacts/msqa_stage51_candidate_adapter_stage61.json
```

Stage 61 full candidate JSONL：

```text
artifacts/msqa_stage51_candidate_adapter_stage61_candidates.jsonl
```

Stage 61 report checksum：

```text
c43d8dd6a38b1539bde8d1681c23878b60f663916f01c877f93ff5d38400d783
```

Stage 61 candidate JSONL checksum：

```text
e505895730f1bf4451dc3c7e0130798692d0c2f298c1b284f19083cb99b96980
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。candidate JSONL 约 228 MB，不提交。

### 问题与原因

- 问题 1：Stage 61 必须区分“用户确认协议”和“可以跑 Stage 51”。
  - 原因：用户确认的是 adapter protocol，不是默认化，也不是最终 candidate comparison；
  - 处理：CLI 强制 `--confirmed-protocol`，但 report 仍写 `can_run_stage51_candidate_now: false`。
- 问题 2：candidate rows 很多。
  - 现象：3301 个样本生成 266647 条 candidate rows，平均每题 80.7776 条；
  - 原因：top10 source rows 的 processed answers 通常会拆出多句；
  - 处理：本阶段只做 dry-run 和分布记录，下一阶段先分析 candidate distribution，再决定是否能公平跑一次 Stage 51 comparison。
- 问题 3：只有 2023 / 3301 样本有 gold-source candidate。
  - 原因：这继承了 answer-only source retrieval 的 hit@10 上限：gold_source_missing_at_10 为 1278；
  - 处理：Stage 61 不把 adapter dry-run 伪装成质量提升；它只是把 Stage 58 的 source-row retrieval 结果转换成 candidate rows。
- 问题 4：candidate score 容易被误解为 Stage 51 model score。
  - 原因：adapter 必须提供 `candidate_score` 字段，但本阶段没有训练或运行 Stage 51；
  - 处理：文档明确 `candidate_score` 是 dry-run adapter score，不是 tuned Stage 51 model score。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_stage51_candidate_adapter.py scripts\dry_run_msqa_stage51_candidate_adapter.py tests\test_msqa_stage51_candidate_adapter.py
pytest -q tests\test_msqa_stage51_candidate_adapter.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

Stage 61 真实 artifact 生成：

```text
python scripts\dry_run_msqa_stage51_candidate_adapter.py --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl --protocol-report artifacts\msqa_stage51_protocol_stage60.json --output artifacts\msqa_stage51_candidate_adapter_stage61.json --candidate-output artifacts\msqa_stage51_candidate_adapter_stage61_candidates.jsonl --visualization-dir artifacts\msqa_stage51_candidate_adapter_stage61_visuals --confirmed-protocol --top-k 10 --min-sentence-chars 1 --sample-limit 20
```

结果：

```text
json report: generated
candidate jsonl rows: 266647
svg visualizations: 3 generated
```

### 结论

- Stage 61 完成用户确认后的 MSQA row-source answer-sentence candidate adapter dry-run。
- Adapter contract 检查全部通过。
- 所有 3301 个样本都有 candidate rows。
- Candidate JSONL 没有写入 question text。
- Stage 61 没有运行 Stage 51 candidate。
- Stage 61 没有改变默认 runtime。
- 当前仍不能 defaultize。

### 我学到的

- “adapter 能生成候选”不等于“候选比较已经公平”。候选分布本身可能影响 Stage 51 reranker 行为，必须先看分布。
- row-source protocol 的上限仍受 answer-only retrieval 限制：如果 gold source 不在 top10，就不会有 gold-source candidate。
- 把 `candidate_score` 写成 dry-run score 很重要，否则后续容易误把它当成 Stage 51 模型输出。
- 候选 JSONL 不包含 question text，是验证 Stage 60 no-question-text 约束的一个直接证据。

### 下一步

- 做 Stage 62：MSQA adapter candidate distribution review。
- Stage 62 目标：
  1. 分析每题 candidate 数量分布；
  2. 分析 gold-source candidate 覆盖与 source retrieval hit@10 的关系；
  3. 检查 candidate_score / retrieval_rank 分布是否异常；
  4. 判断是否允许一次 Stage 51 adapter comparison；
  5. 即使允许，也仍不 defaultize。

## 2026-07-14 Stage 62：MSQA adapter candidate distribution review

### 本阶段目标

Stage 61 证明 adapter contract 可以生成完整 candidate rows，但还不能说明直接跑 Stage 51 是公平的。本阶段目标是分析 candidate distribution，并与 Stage 31 的 Stage 51 训练候选池契约对照：

1. 读取 Stage 61 adapter report；
2. 流式分析 Stage 61 candidate JSONL；
3. 读取 Stage 31 candidate-reranker summary；
4. 对比每题 candidate 数量、gold-source candidate 覆盖、retrieval rank 分布；
5. 判断是否允许一次 direct Stage 51 adapter comparison；
6. 继续不运行 Stage 51，不改默认 runtime。

### 新增与修改

新增 Stage 62 distribution review 模块：

```text
src/ts_rag_agent/application/msqa_stage51_candidate_distribution_review.py
```

新增能力：

- 读取 Stage 61 report；
- 读取 Stage 61 full candidate JSONL；
- 读取 Stage 31 summary；
- 计算 Stage 61 candidate count per query 分布；
- 计算 Stage 61 candidate score 分布；
- 计算 retrieval rank candidate row count；
- 统计 gold-source candidate coverage；
- 与 Stage 31 candidate pool 做对照；
- 输出 fairness checks；
- 明确 decision 是否允许 Stage 51 adapter comparison；
- 生成 SVG 可视化。

新增 CLI：

```text
scripts/review_msqa_stage51_candidate_distribution.py
```

新增测试：

```text
tests/test_msqa_stage51_candidate_distribution_review.py
```

新增文档：

```text
docs/msqa_stage51_candidate_distribution.md
```

同步更新：

```text
docs/msqa_stage51_candidate_adapter.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/external_eval_datasets.md
docs/learning_journal.md
```

### Stage 62 运行命令

```powershell
python scripts\review_msqa_stage51_candidate_distribution.py `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage61.json `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage61_candidates.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_candidate_distribution_stage62.json `
  --visualization-dir artifacts\msqa_stage51_candidate_distribution_stage62_visuals
```

### Stage 62 结果

Stage 61 candidate count per query：

```text
count: 3301
min: 14
p10: 51
p25: 64
median: 79
p75: 95
p90: 113
p95: 125
p99: 151
max: 189
average: 80.7776
```

Stage 31 candidate count per question：

```text
count: 610
min: 2
p10: 5
p25: 15
median: 15
p75: 15
p90: 15
p95: 15
p99: 15
max: 15
average: 13.2131
```

Stage 31 training candidate contract：

```text
retrieval_top_k: 5
max_candidates_per_document: 3
candidate_limit: 25
effective_max_candidates: 15
evidence_selector: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
```

candidate-pool comparison：

```text
average_candidate_count_ratio_stage61_vs_stage31: 6.1134
median_candidate_count_ratio_stage61_vs_stage31: 5.2667
stage61_median_exceeds_stage31_max: true
stage61_p10_exceeds_stage31_max: true
gold_candidate_rate_delta_stage61_minus_stage31: -0.0069
```

gold-source coverage：

```text
Stage61 gold-source candidate rate: 0.6128
Stage31 gold-document candidate rate: 0.6197
delta: -0.0069
```

fairness checks：

```text
stage61_adapter_contract_passed: pass
candidate_jsonl_has_no_question_text_field: pass
all_stage61_samples_have_candidates: pass
gold_source_candidate_rate_matches_training_pool: pass
candidate_pool_size_aligned_with_stage31: blocked
stage61_candidate_volume_within_training_limit: blocked
direct_stage51_adapter_comparison_fair_now: blocked
```

decision：

```text
status: msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch
can_run_stage51_candidate_now: false
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
```

阻塞项：

```text
candidate_pool_size_aligned_with_stage31
stage61_candidate_volume_within_training_limit
direct_stage51_adapter_comparison_fair_now
```

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_candidate_count_percentiles.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_stage31_vs_stage61_candidate_pool.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_candidate_rows_by_retrieval_rank.svg
artifacts/msqa_stage51_candidate_distribution_stage62_visuals/stage62_fairness_checks.svg
```

Stage 62 report：

```text
artifacts/msqa_stage51_candidate_distribution_stage62.json
```

Stage 62 report checksum：

```text
1948ddf4101a35c5229fe6c79e50a21d956d8dabaa6dfeed029b83afe4629c79
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题与原因

- 问题 1：Stage 61 contract passed，但直接跑 Stage 51 仍不公平。
  - 原因：Stage 61 candidate pool 是 uncapped top10 source-row answer sentences；Stage 31 训练候选池是 top5 retrieval、每 document 最多 3 个 candidates；
  - 证据：Stage61 median 79 candidates/query，而 Stage31 max 15 candidates/question。
- 问题 2：candidate pool mismatch 不是少数 outlier。
  - 证据：Stage61 p10 已经是 51 candidates/query，仍高于 Stage31 max 15；
  - 处理：阻塞 direct Stage 51 adapter comparison。
- 问题 3：gold-source coverage 接近 Stage31，但这不足以放行。
  - 证据：Stage61 gold-source candidate rate 0.6128，Stage31 gold-document candidate rate 0.6197；
  - 原因：coverage 接近只能说明 source availability 相近，不能抵消候选池规模和候选排序空间的巨大差异。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_stage51_candidate_distribution_review.py scripts\review_msqa_stage51_candidate_distribution.py tests\test_msqa_stage51_candidate_distribution_review.py
pytest -q tests\test_msqa_stage51_candidate_distribution_review.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

Stage 62 真实 artifact 生成：

```text
python scripts\review_msqa_stage51_candidate_distribution.py --adapter-report artifacts\msqa_stage51_candidate_adapter_stage61.json --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage61_candidates.jsonl --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json --output artifacts\msqa_stage51_candidate_distribution_stage62.json --visualization-dir artifacts\msqa_stage51_candidate_distribution_stage62_visuals
```

结果：

```text
json report: generated
svg visualizations: 4 generated
```

### 结论

- Stage 62 完成 MSQA adapter candidate distribution review。
- Stage 61 adapter contract 仍然有效。
- Direct Stage 51 adapter comparison 被阻塞。
- 阻塞原因不是 source coverage，而是 candidate pool size 与 Stage31 训练候选池严重不一致。
- 当前不能运行 Stage 51 candidate。
- 当前不能 defaultize。
- 默认 runtime 不变。

### 我学到的

- adapter contract passed 只是必要条件，不是 sufficient condition。
- candidate-pool shape 是 reranker fairness 的一部分；训练时最大 15 个候选，评估时中位数 79 个候选，会把 protocol shift 和 policy effect 混在一起。
- gold-source candidate rate 接近并不能保证 reranker comparison 公平，因为候选池竞争空间已经完全不同。
- 只有先把 MSQA candidate pool 对齐 Stage31 训练契约，后续的一次 Stage51 comparison 才更像策略比较，而不是数据管线比较。

### 下一步

- 做 Stage 63：Stage31-aligned MSQA candidate-pool cap 设计与 adapter dry-run。
- Stage 63 目标：
  1. 按 Stage31 训练契约设计 MSQA cap；
  2. 优先候选：top5 source rows、每 source row 最多 3 个 answer-sentence candidates；
  3. 保持 no question text、no fallback、no external fetch；
  4. rerun adapter dry-run；
  5. 再判断是否允许一次 Stage51 adapter comparison。

## 2026-07-14 - Stage 63：Stage31-aligned MSQA candidate-pool cap dry-run

### 背景

Stage 62 证明 Stage 61 adapter contract 已通过，但 uncapped top10 answer-sentence
candidate pool 与 Stage31 训练候选池严重不一致：

```text
Stage61 median candidates/query: 79
Stage31 max candidates/question: 15
Stage61 p10 candidates/query: 51
```

所以 Stage 63 的目标不是跑 Stage51，而是把 MSQA adapter 候选池先对齐
Stage31 训练候选池形状，再判断是否可以进入一次 capped comparison。

### 用户确认的特殊处理

Stage31 只给出了“每 document 最多 3 个 candidates”的训练契约，没有直接规定
MSQA answer sentences 应取哪 3 个。因此本阶段先向用户列出选项并确认：

```text
A. 按现有 dry-run candidate_score 取每个 source row 的 top3
B. 按答案句原始顺序取前 3 句
C. 先找 Stage31 evidence selector 逻辑并复用它取 top3
```

用户确认选择 A。

本阶段真实采用的 cap rule：

```text
top_k: 5
max_candidates_per_source_row: 3
effective_candidate_pool_cap: 15
candidate selection: generate all answer-sentence candidates, rank by dry-run
candidate_score descending, retrieval_rank ascending, candidate_id ascending,
then keep top3 per source row
```

注意：`candidate_score` 仍然只是 adapter dry-run score，不是 tuned Stage51 model score。

### 新增能力

- `build_msqa_stage51_candidate_adapter_dry_run` 新增：
  - `max_candidates_per_source_row`
  - `stage_name`
  - `candidate_pool_cap_rule`
  - `effective_candidate_pool_cap`
- CLI `scripts/dry_run_msqa_stage51_candidate_adapter.py` 新增：
  - `--max-candidates-per-source-row`
  - `--stage-name`
- `review_msqa_stage51_candidate_distribution` 现在同时接受 Stage61 和 Stage63
  adapter report。
- distribution review 新增通用字段：
  - `adapter_summary`
  - `adapter_candidate_distribution`
  - `average_candidate_count_ratio_adapter_vs_stage31`
  - `median_candidate_count_ratio_adapter_vs_stage31`
  - `adapter_median_exceeds_stage31_max`
  - `adapter_p10_exceeds_stage31_max`
  - `gold_candidate_rate_delta_adapter_minus_stage31`
- Stage63 可视化会使用 `stage63_*` 文件名。

### Stage 63 运行命令

Adapter dry-run：

```powershell
python scripts\dry_run_msqa_stage51_candidate_adapter.py `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --protocol-report artifacts\msqa_stage51_protocol_stage60.json `
  --output artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --candidate-output artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --visualization-dir artifacts\msqa_stage51_candidate_adapter_stage63_capped_visuals `
  --confirmed-protocol `
  --top-k 5 `
  --max-candidates-per-source-row 3 `
  --min-sentence-chars 1 `
  --sample-limit 20 `
  --stage-name "Stage 63"
```

Distribution review：

```powershell
python scripts\review_msqa_stage51_candidate_distribution.py `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --visualization-dir artifacts\msqa_stage51_candidate_distribution_stage63_capped_visuals `
  --stage-name "Stage 63"
```

### Stage 63 adapter dry-run 结果

```text
evaluation_samples: 3301
candidate_rows: 47342
samples_with_candidates: 3301
samples_without_candidates: 0
samples_with_gold_source_candidate: 1850
average_candidates_per_sample: 14.3417
median_candidates_per_sample: 15.0
max_candidates_per_sample_contract: 15
unique_source_rows_in_candidates: 2624
```

Source retrieval：

```text
hit@1: 0.4147
hit@5: 0.5604
mrr: 0.4692
gold_source_missing_at_5: 1451
```

Contract checks：

```text
passed: 7 / 7
rows_with_question_key: 0
no_answer_field_fallback_used: true
no_external_fetch_used: true
stage51_candidate_run_performed: false
default_runtime_policy: unchanged
```

### Stage 63 distribution review 结果

Stage63 candidate count per query：

```text
count: 3301
min: 7.0
p10: 13.0
p25: 14.0
median: 15.0
p75: 15.0
p90: 15.0
p95: 15.0
p99: 15.0
max: 15.0
average: 14.3417
```

Stage31 candidate count per question：

```text
count: 610
min: 2.0
p10: 5.0
p25: 15.0
median: 15.0
p75: 15.0
p90: 15.0
p95: 15.0
p99: 15.0
max: 15.0
average: 13.2131
```

Comparison：

```text
average_candidate_count_ratio_adapter_vs_stage31: 1.0854
median_candidate_count_ratio_adapter_vs_stage31: 1.0
adapter_median_exceeds_stage31_max: false
adapter_p10_exceeds_stage31_max: false
gold_candidate_rate_delta_adapter_minus_stage31: -0.0593
```

Fairness checks：

```text
adapter_contract_passed: pass
candidate_jsonl_has_no_question_text_field: pass
all_adapter_samples_have_candidates: pass
gold_source_candidate_rate_matches_training_pool: warn
candidate_pool_size_aligned_with_stage31: pass
adapter_candidate_volume_within_training_limit: pass
direct_stage51_adapter_comparison_fair_now: pass
```

Decision：

```text
status: msqa_stage51_adapter_comparison_ready_for_user_confirmation
can_run_stage51_candidate_now: false
can_run_stage51_candidate_next_with_user_confirmation: true
can_defaultize_runtime_now: false
default_runtime_policy: unchanged
stage51_candidate_run_performed: false
blocker_checks: []
```

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_candidate_counts.svg
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_source_hit_rates.svg
artifacts/msqa_stage51_candidate_adapter_stage63_capped_visuals/stage63_adapter_contract_checks.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_candidate_count_percentiles.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_stage31_vs_adapter_candidate_pool.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_candidate_rows_by_retrieval_rank.svg
artifacts/msqa_stage51_candidate_distribution_stage63_capped_visuals/stage63_fairness_checks.svg
```

Stage 63 artifacts：

```text
artifacts/msqa_stage51_candidate_adapter_stage63_capped.json
artifacts/msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl
artifacts/msqa_stage51_candidate_distribution_stage63_capped.json
```

Checksums：

```text
adapter report: 56bef0e0f78365dc18c8ee1f2c63d25a2c17015123c6a78d0014e05669e0934a
candidate JSONL: 317c4502edb7a34eba97b5b5045fed63f05228a82e271bee0804af74467248d2
distribution report: c036e67c48f3dbd800fc8fbb81e174d830f9785080146ddd99cc62f8b5df69cf
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题、原因与修正

- 问题 1：Stage62 阻塞的 candidate-pool mismatch 需要被工程化成显式参数，而不是临时脚本逻辑。
  - 原因：如果 cap 不进入 adapter contract，后续 Stage64 很容易误用 uncapped candidates。
  - 修正：新增 `max_candidates_per_source_row`、`effective_candidate_pool_cap` 和
    `candidate_pool_cap_rule`，并写入 report。
- 问题 2：Stage63 top5 会降低 gold-source candidate rate。
  - 证据：Stage63 0.5604 vs Stage31 0.6197，delta -0.0593。
  - 原因：Stage63 为了对齐 Stage31 top5 训练边界，从 Stage61 top10 缩到 top5；
    source availability 因此下降。
  - 处理：该项记录为 `warn`，不是 candidate-size blocker；Stage64 解释结果时必须保留这个 warning。
- 问题 3：首次生成 distribution review 后，gold-source warning 的 `decision_effect`
  文案仍写成接近 Stage31。
  - 真实处理：本轮发现后立即修改为 source-retrieval availability tradeoff，
    并重跑 distribution review；adapter JSONL 没有重生成。

### 测试

局部验证：

```powershell
ruff check src/ts_rag_agent/application/msqa_stage51_candidate_adapter.py src/ts_rag_agent/application/msqa_stage51_candidate_distribution_review.py scripts/dry_run_msqa_stage51_candidate_adapter.py scripts/review_msqa_stage51_candidate_distribution.py tests/test_msqa_stage51_candidate_adapter.py tests/test_msqa_stage51_candidate_distribution_review.py
pytest -q tests/test_msqa_stage51_candidate_adapter.py tests/test_msqa_stage51_candidate_distribution_review.py
```

结果：

```text
ruff: passed
pytest: 10 passed
```

Stage 63 真实 artifact 生成：

```text
adapter dry-run: generated
candidate JSONL rows: 47342
adapter visualizations: 3 generated
distribution review: generated
distribution visualizations: 4 generated
```

### 结论

- Stage63 完成 Stage31-aligned MSQA candidate-pool cap dry-run。
- Candidate pool size 已对齐 Stage31 训练边界：
  - median 15.0 vs Stage31 max 15.0；
  - p10 13.0 vs Stage31 max 15.0；
  - max 15.0。
- Stage63 没有运行 Stage51。
- Stage63 没有 defaultize。
- 默认 runtime 不变。
- Source availability 存在 warning：gold-source candidate rate 比 Stage31 低 0.0593。
- 现在可以在用户确认后进入一次 capped Stage51 adapter comparison，但只能使用同一个
  Stage63 capped candidate pool。

### 我学到的

- 候选池对齐不能只看平均值；p10、median、max 都要看，否则可能把 outlier 和整体结构混在一起。
- top5/max3 能对齐 candidate-size boundary，但会带来 source availability tradeoff；
  这两个事实必须同时记录。
- 将 cap rule 写进 adapter contract 比只写在命令行里更可靠；后续 Stage64 才能审计自己是否使用同一个候选池。
- 报告文案也是实验事实的一部分；发现 warning 文案不准确时，应如实标记并重跑报告，而不是只在口头解释里修正。

### 下一步

- 做 Stage64：在用户确认后，使用同一个 Stage63 capped candidate pool 跑一次
  Stage51 adapter comparison。
- Stage64 必须：
  1. 不重建或改写 capped candidate pool；
  2. 不 defaultize；
  3. 对比同一 capped candidate pool 下的 baseline 与 Stage51 adapter；
  4. 单独记录 top5 source availability warning；
  5. 输出可视化和完整学习记录。

## 2026-07-14 - Stage 64：MSQA capped Stage51 adapter comparison

### 背景

Stage 63 已经把 MSQA candidate pool 对齐到 Stage31 训练候选池形状：

```text
top_k: 5
max_candidates_per_source_row: 3
effective_candidate_pool_cap: 15
candidate_rows: 47342
median candidates/query: 15.0
max candidates/query: 15.0
```

Stage 64 的目标是在**同一个 Stage63 capped candidate pool** 上运行一次 Stage51
adapter comparison，而不是重建候选池、调参或 defaultize。

### 用户确认的特殊处理

Stage51 scorer 需要 `runtime_features`，其中包含 route、query overlap、token count
等基于 question text 的运行时特征；但 Stage63 candidate JSONL 按协议不包含
`question` 字段。

本阶段先向用户列出选项：

```text
A. 从 Stage57 frozen split 读取 question text，只用于计算 runtime_features
B. 只使用 Stage63 candidate JSONL 里已有字段打分
C. 先做 Stage64a：只生成 MSQA runtime feature adapter 并审计，不跑 comparison
```

用户确认选择 A。

本阶段真实边界：

```text
question text source: Stage57 frozen split only
question text written to candidate JSONL: false
question text used for retrieval indexing: false
candidate pool rebuilt: false
```

### 新增能力

新增模块：

```text
src/ts_rag_agent/application/msqa_stage51_adapter_comparison.py
```

新增 CLI：

```text
scripts/compare_msqa_stage51_capped_adapter.py
```

新增测试：

```text
tests/test_msqa_stage51_adapter_comparison.py
```

新增文档：

```text
docs/msqa_stage51_adapter_comparison.md
```

同步更新：

```text
docs/evaluation_strategy.md
docs/data_strategy.md
docs/msqa_stage51_candidate_pool_cap.md
docs/learning_journal.md
```

### Stage51 adapter contract

本阶段复用真实 Stage51 runtime policy，而不是重新发明一个“像 Stage51”的策略：

```text
model: logistic_best_candidate
train_split: train
selector_name_runtime_feature: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
composition_policy: candidate_score_gte_60_rank_contained_preserve_baseline_out_of_rank_guarded_reranker
runtime_guard: candidate_score_gte_60_all_selected_citations_rank_lte_max_citation_rank_preserve_baseline_out_of_rank_docs
max_answer_candidates: 3
rank_contained_max_retrieval_rank: 3
candidate_pool_rebuilt: false
candidate_pool_rows: 47342
```

Stage64 在每个 query 内只对**已存在的 capped candidates** 赋予 Stage31-like
global candidate rank：

```text
candidate_score descending
retrieval_rank ascending
candidate_id ascending
```

注意：MSQA `candidate_score` 是 Stage63 dry-run adapter score，不是训练好的 Stage51
model score；本阶段保留 Stage51 的 `score >= 60` guard 不变，因此 score calibration
差异属于 adapter-risk 证据的一部分。

### Stage 64 运行命令

```powershell
python scripts\compare_msqa_stage51_capped_adapter.py `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --distribution-report artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_adapter_comparison_stage64.json `
  --visualization-dir artifacts\msqa_stage51_adapter_comparison_stage64_visuals `
  --model logistic_best_candidate `
  --train-split train `
  --max-answer-candidates 3 `
  --max-citation-rank 3 `
  --sample-limit 20
```

### Stage 64 结果

Answer-source proxy F1：

```text
baseline_top1_average_token_f1: 0.2881
stage51_top1_average_token_f1: 0.2863
top1_average_delta_vs_baseline: -0.0018

baseline_top3_average_answer_token_f1: 0.4199
stage51_top3_average_answer_token_f1: 0.4171
top3_average_delta_vs_baseline: -0.0028

oracle_best_single_average_token_f1: 0.4156
```

Top3 outcomes：

```text
question_count: 3301
top3_improved_count: 20
top3_regressed_count: 57
top3_tied_count: 3224
changed_answer_count: 719
changed_answer_rate: 0.2178
replacement_count: 719
replacement_rate: 0.2178
```

Gold-source citation：

```text
baseline_gold_source_citation_count: 1394
stage51_gold_source_citation_count: 1397
baseline_gold_source_citation_rate: 0.4223
stage51_gold_source_citation_rate: 0.4232
gold_source_citation_delta: +3
citation_lost_count: 0
citation_gained_count: 3
```

Decision reasons：

```text
candidate_score_gte_60_accepted: 719
candidate_score_gte_60_blocked: 479
model_selected_top_candidate: 1050
score_margin_below_min: 555
selected_rank_exceeds_limit: 498
```

Selected rank distribution：

```text
rank_1: 1050
rank_2: 717
rank_3: 585
rank_4_5: 451
rank_6_10: 415
rank_11_15: 83
```

Route metrics 摘要：

```text
other: questions 2402, top3 delta -0.0017, changed 488, improved 13, regressed 28, citation delta +3
error_or_log: questions 452, top3 delta -0.0050, changed 136, improved 4, regressed 12, citation delta 0
install_upgrade_config: questions 351, top3 delta -0.0078, changed 77, improved 3, regressed 16, citation delta 0
limitation_or_restriction: questions 95, top3 delta -0.0007, changed 18, improved 0, regressed 1, citation delta 0
```

### Stage63 warning 保留

Stage64 保留 Stage63 的 source availability warning：

```text
Stage63 gold-source candidate rate: 0.5604
Stage31 gold-document candidate rate: 0.6197
delta: -0.0593
```

这意味着 Stage64 只评价 top5 source-row boundary 下的 Stage51 adapter 行为；
missing gold source rows 不能由 Stage51 scorer 恢复。

### 可视化结果

本阶段生成：

```text
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_answer_f1.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_answer_f1_delta.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_gold_source_citation.svg
artifacts/msqa_stage51_adapter_comparison_stage64_visuals/stage64_msqa_decision_reasons.svg
```

Stage64 report：

```text
artifacts/msqa_stage51_adapter_comparison_stage64.json
```

Stage64 report checksum：

```text
2dd3427631028248c552c1ec68983ff3e271d7332940bb7768a58b63c0aaa8b3
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题、原因与修正

- 问题 1：Stage51 scorer 需要 question-derived runtime features，但 Stage63 candidate JSONL
  不能包含 question text。
  - 原因：Stage60/61 明确 no question text in candidate rows，防止把 MSQA 变成 question-text diagnostic index。
  - 处理：从 Stage57 frozen split 读取 question text，只用于 runtime feature computation；
    report 中明确记录 `question text written to candidate JSONL: false`。
- 问题 2：MSQA adapter score 与 PrimeQA Stage31 candidate score 不是同一个校准分布。
  - 原因：MSQA `candidate_score` 是 Stage63 dry-run adapter score；Stage51 guard 的
    `score >= 60` 来自 PrimeQA candidate-reranker workflow。
  - 处理：不调阈值、不重新搜索策略，把 calibration difference 作为 adapter-risk 证据记录。
- 问题 3：Stage51 在 MSQA capped pool 上 citation 有收益但 answer F1 回退。
  - 证据：gold-source citation delta `+3`，citation lost `0`；但 top3 F1 delta `-0.0028`。
  - 处理：decision status 标记为 `msqa_stage51_capped_adapter_comparison_f1_regressed`；
    不 defaultize，下一步做 changed-case/tradeoff review。

### 测试

局部验证：

```powershell
ruff check src/ts_rag_agent/application/msqa_stage51_adapter_comparison.py scripts/compare_msqa_stage51_capped_adapter.py tests/test_msqa_stage51_adapter_comparison.py
pytest -q tests/test_msqa_stage51_adapter_comparison.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

真实 artifact 生成：

```text
comparison report: generated
visualizations: 4 generated
candidate pool rebuilt: false
candidate_pool_rows: 47342
```

### 结论

- Stage64 完成一次 capped MSQA Stage51 adapter comparison。
- Stage64 没有重建或改写 Stage63 capped candidate pool。
- Stage64 没有 defaultize。
- 默认 runtime 不变。
- 结果是混合信号：
  - gold-source citation `+3`，且 citation loss `0`；
  - top3 answer-source proxy F1 `-0.0028`；
  - changed answers `719 / 3301`，其中 improved 20、regressed 57。
- 当前不能把 Stage51 推向默认化；应先审查 changed cases 与 source-citation tradeoff。

### 我学到的

- 外部 adapter comparison 必须把“candidate-pool fairness”和“score calibration”分开看；pool 对齐了，不代表 Stage51 guard 的阈值天然迁移。
- MSQA 上 Stage51 的收益更像 source correction，而不是 answer text improvement。
- 不丢 citation 不等于整体安全；719 个 changed answers 和 57 个 top3 F1 regressions 说明需要 case-level 风险复盘。
- Stage64 的结果加强了一个边界：MSQA 是外部风险证据，不是 defaultization test。

### 下一步

- 做 Stage65：review Stage64 changed cases and source-citation tradeoffs。
- Stage65 目标：
  1. 分析 57 个 top3 regression 和 20 个 improvement；
  2. 单独看 3 个 citation gained case；
  3. 判断 regression 是否集中在 route、selected rank、score margin 或 cross-source replacement；
  4. 决定是否需要另一个外部数据集，或冻结最终评估协议；
  5. 继续不 defaultize。

## Stage 65：MSQA Stage51 changed-case 与 source-citation tradeoff review

### 本阶段目标

接着 Stage64 的 capped MSQA Stage51 adapter comparison，做 case-level 复盘：

1. 重建 Stage64 全量 case view，并确认聚合指标与 Stage64 artifact 一致；
2. 分析 719 个 changed answers 中的 top3 regression、top3 improvement、citation gain/loss；
3. 生成可视化，判断风险是否集中在 route、selected rank、score margin 或 source transition；
4. 明确是否可以 defaultize Stage51。

本阶段没有重建 candidate pool，没有重跑新的 Stage51 comparison，没有调参，也没有改变默认 runtime。

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/msqa_stage51_changed_case_review.py`
- 新增脚本：
  - `scripts/review_msqa_stage51_changed_cases.py`
- 新增测试：
  - `tests/test_msqa_stage51_changed_case_review.py`
- 更新 Stage64 comparison 模块：
  - `src/ts_rag_agent/application/msqa_stage51_adapter_comparison.py`
  - 将 Stage64 case rebuild 抽成可复用 helper；
  - 暴露 full case rebuild、metrics summary、case-to-dict helper，供 Stage65 复用；
  - 保留 runtime policy identity，避免把 policy 名称写死成字符串。
- 新增文档：
  - `docs/msqa_stage51_changed_case_review.md`
- 更新文档索引与策略：
  - `docs/msqa_stage51_adapter_comparison.md`
  - `docs/data_strategy.md`
  - `docs/evaluation_strategy.md`

### Stage65 运行命令

```powershell
python scripts\review_msqa_stage51_changed_cases.py `
  --stage64-report artifacts\msqa_stage51_adapter_comparison_stage64.json `
  --split-jsonl artifacts\msqa_evaluation_split_stage57.jsonl `
  --candidate-jsonl artifacts\msqa_stage51_candidate_adapter_stage63_capped_candidates.jsonl `
  --adapter-report artifacts\msqa_stage51_candidate_adapter_stage63_capped.json `
  --distribution-report artifacts\msqa_stage51_candidate_distribution_stage63_capped.json `
  --candidate-reranker-dataset artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl `
  --stage31-summary artifacts\candidate_reranker_dataset_stage31_dev_train_hybrid_summary.json `
  --output artifacts\msqa_stage51_changed_case_review_stage65.json `
  --visualization-dir artifacts\msqa_stage51_changed_case_review_stage65_visuals
```

### 真实运行结果

Rebuild contract：

```text
candidate_pool_rebuilt: false
case_count: 3301
model_name: logistic_best_candidate
train_split: train
max_answer_candidates: 3
max_citation_rank: 3
consistency_checks_passed: true
```

Changed-case summary：

```text
question_count: 3301
changed_answer_count: 719
changed_answer_rate: 0.2178

top3_regression_count: 57
top3_improvement_count: 20
regression_to_improvement_count_ratio: 2.85

regression_loss_sum: -10.5636
improvement_gain_sum: 1.3434
net_top3_delta_sum: -9.2202

citation_gained_count: 3
citation_lost_count: 0
citation_delta: +3
regressions_with_citation_gain: 0
improvements_with_citation_gain: 3
changed_without_f1_or_citation_gain: 642
```

Route concentration：

```text
changed cases:
other: 488
error_or_log: 136
install_upgrade_config: 77
limitation_or_restriction: 18

top3 regressions:
other: 28
install_upgrade_config: 16
error_or_log: 12
limitation_or_restriction: 1
```

关键集中性发现：

```text
regression_selected_rank_share:
rank_4_5: 57 / 57

regression_source_transition_share:
leading_source_changed: 57 / 57

citation_gain_source_transition_share:
gold_source_added: 3 / 3
```

### 可视化结果

本阶段生成 4 个 SVG：

```text
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_changed_outcomes.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_regressions_by_route.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_changed_by_selected_rank.svg
artifacts/msqa_stage51_changed_case_review_stage65_visuals/stage65_msqa_source_transitions.svg
```

Stage65 report：

```text
artifacts/msqa_stage51_changed_case_review_stage65.json
```

Stage65 report checksum：

```text
089003101663b7b2880ad359b5eb2f6065778a6ece3983e154f3ba59b8e69def
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题、原因与修正

- 问题 1：Stage64 case rebuild 原来是 `compare_msqa_stage51_capped_adapter` 内部逻辑，Stage65 不能可靠复用全量 case。
  - 原因：Stage64 最初只需要输出聚合 report 和 sample cases，没有为后续 changed-case review 暴露 full case API。
  - 修正：新增 `build_msqa_stage51_capped_adapter_cases`、`summarize_msqa_stage51_comparison_cases`、`msqa_stage51_comparison_case_to_dict`，让 Stage65 从同一套逻辑重建 full case view。
- 问题 2：helper 抽取后 Stage64 report 中 `composition_policy` 一度引用旧局部变量 `policy`。
  - 原因：policy 对象被移动到 case rebuild helper 内部，但 compare report 仍需要记录真实 policy name。
  - 修正：让内部 case rebuild 返回 cases 与 `policy_name`，report 继续记录真实 runtime policy identity，不写死。
- 问题 3：Stage65 不能只看 aggregate delta，否则会误读 citation gain。
  - 原因：Stage64 aggregate 显示 citation +3、F1 -0.0028，但不知道风险集中在哪里。
  - 修正：按 route、selected rank、score margin、source transition 拆解 changed cases；结果显示 57 个 top3 regression 全部来自 rank 4-5 的 leading-source change。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\msqa_stage51_adapter_comparison.py src\ts_rag_agent\application\msqa_stage51_changed_case_review.py scripts\review_msqa_stage51_changed_cases.py tests\test_msqa_stage51_changed_case_review.py tests\test_msqa_stage51_adapter_comparison.py
pytest -q tests\test_msqa_stage51_adapter_comparison.py tests\test_msqa_stage51_changed_case_review.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

真实 artifact 生成：

```text
Stage65 report: generated
visualizations: 4 generated
consistency_checks_passed: true
candidate_pool_rebuilt: false
```

### 结论

- Stage65 完成了 Stage64 changed-case 与 source-citation tradeoff review。
- Stage65 重建的全量 case view 与 Stage64 aggregate 指标一致。
- Citation gain 是真实存在的 MSQA proxy 信号：`+3` gained、`0` lost。
- 但 answer 风险更强：top3 regression `57`，top3 improvement `20`，regression 数量是 improvement 的 `2.85x`。
- 57 个 top3 regression 全部集中在 rank 4-5 的 leading-source change。
- 当前不能 defaultize Stage51。
- 默认 runtime 继续保持不变。

### 我学到的

- 外部 adapter 的收益不能只看 citation gain；必须同时看 changed-case answer risk。
- Stage51 在 MSQA 上的少量 citation 修正主要来自 gold-source-added cases，但这种收益规模太小，不能抵消 rank 4-5 leading-source change 带来的 answer regression。
- “同源句子改写”数量很大，但真正的回归风险集中在跨 leading source 的深 rank 替换，这比单纯看 changed-answer count 更有诊断价值。
- 后续如果还想在 MSQA 上实验，必须冻结一个明确的新协议；不能围绕 Stage64/65 结果反复调参直到指标好看。

### 下一步

做 Stage66：显式选择下一条 evaluation route。

候选路线：

1. 找另一个外部数据集，并重新做 schema/license/leakage qualification；
2. 设计一个 MSQA-specific 的 rank 4-5 leading-source risk guard，并只跑一次新的 frozen MSQA experiment；
3. 冻结 Stage51 为 non-default research evidence，继续保持 top-k 为默认 runtime。

在 Stage66 路线明确前，继续不 defaultize Stage51。

## Stage 66：第二轮 external evaluation dataset rediscovery

### 本阶段目标

接着 Stage65，用户选择“再找一个数据集”。本阶段目标是做第二轮外部数据集发现：

1. 在线核查新的或此前未作为主候选的数据集；
2. 优先寻找技术支持、企业 IT、操作系统排障、软件支持相关 QA 数据集；
3. 检查 public source、schema、license、citation/context 适配可能性；
4. 推荐一个下一步 schema probe 候选；
5. 不下载数据、不跑指标、不修改默认 runtime。

### 在线来源

本阶段核查了这些公开页面：

```text
https://data.mendeley.com/datasets/p85z3v45xk/1
https://www.sciencedirect.com/science/article/pii/S2352340923003645
https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa
https://ciir.cs.umass.edu/downloads/msdialog/
https://archive.org/download/stackexchange
https://stackoverflow.com/help/data-dumps
```

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/external_eval_dataset_rediscovery.py`
- 新增脚本：
  - `scripts/rediscover_external_eval_datasets.py`
- 新增测试：
  - `tests/test_external_eval_dataset_rediscovery.py`
- 新增文档：
  - `docs/external_eval_dataset_rediscovery.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/evaluation_strategy.md`
  - `docs/external_eval_datasets.md`
  - `docs/learning_journal.md`

### Stage66 运行命令

```powershell
python scripts\rediscover_external_eval_datasets.py `
  --output artifacts\external_eval_dataset_rediscovery_stage66.json `
  --visualization-dir artifacts\external_eval_dataset_rediscovery_stage66_visuals
```

### 真实运行结果

推荐候选：

```text
recommended_candidate: hqa_data_ubuntu_dialogue
recommended_candidate_name: HQA-Data from Ubuntu Dialogue Corpus
recommended_next_stage: Stage 67: HQA-Data local schema probe, file checksum capture, context-span coverage audit, and PrimeQA/MSQA leakage protocol
can_run_final_metrics_now: false
can_download_without_user_confirmation: false
default_runtime_policy: unchanged
```

候选排名：

```text
hqa_data_ubuntu_dialogue:
  status: recommended_for_stage67_schema_probe
  fit_score: 15
  domain_fit_score: 2
  citation_fit_score: 3
  license_fit_score: 2
  adapter_effort_score: 2

multidoc2dial:
  status: strong_document_grounding_reference_not_new_primary
  fit_score: 15
  domain_fit_score: 1
  citation_fit_score: 3
  license_fit_score: 3
  adapter_effort_score: 3

msdialog:
  status: blocked_until_access_and_license_boundary_confirmation
  fit_score: 14
  domain_fit_score: 3
  citation_fit_score: 1
  license_fit_score: 0
  adapter_effort_score: 4

askubuntu_stackexchange_dump:
  status: derivation_candidate_blocked_by_size_access_and_attribution_plan
  fit_score: 11
  domain_fit_score: 2
  citation_fit_score: 1
  license_fit_score: 1
  adapter_effort_score: 4

hf_ubuntu_dialogue_qa:
  status: blocked_by_license_metadata_mismatch
  fit_score: 10
  domain_fit_score: 2
  citation_fit_score: 1
  license_fit_score: 0
  adapter_effort_score: 2
```

### HQA-Data source-backed facts

Mendeley 页面记录：

```text
published date: 2022-12-15
DOI: 10.17632/p85z3v45xk.1
source: Ubuntu Dialogue Corpus conversations by dialogueID
formats: CSV and JSON
train QA pairs: 29,150
test QA pairs: 7,288
total contexts: 9,364
total QA pairs: 36,438
license: CC BY 4.0
```

ScienceDirect 文章补充说明：

```text
questions and answers are contained within the context
original data source: Mendeley Data
keywords include Ubuntu dialogue corpus and question answering generation
```

### 可视化结果

本阶段生成 4 个 SVG：

```text
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_fit_score.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_domain_fit.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_citation_fit.svg
artifacts/external_eval_dataset_rediscovery_stage66_visuals/stage66_candidate_effort_score.svg
```

Stage66 report：

```text
artifacts/external_eval_dataset_rediscovery_stage66.json
```

Stage66 report checksum：

```text
a357e2b466d102c1b30a374e87aca3b895f906ef8d5de6b7ea386741f5f6ace3
```

以上 artifacts 位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### 问题、原因与修正

- 问题 1：MSDialog 的领域匹配非常强，但不能作为下一步直接使用。
  - 原因：CIIR 页面要求联系获取访问权限，并写明 internal research only、不能分享数据集。
  - 处理：标记为 `blocked_until_access_and_license_boundary_confirmation`，除非用户明确批准访问与非再分发边界，否则不下载、不做 artifact。
- 问题 2：Ask Ubuntu StackExchange dump 领域相关，但不是现成 evaluation set。
  - 原因：需要从 StackExchange XML 中派生 QA 样本，并处理 CC BY-SA attribution/share-alike；当前最新 dump 访问还涉及账户设置和 non-LLM-training affirmation。
  - 处理：标记为 `derivation_candidate_blocked_by_size_access_and_attribution_plan`，暂不推荐下一步。
- 问题 3：Hugging Face `sedthh/ubuntu_dialogue_qa` 页面许可信息不一致。
  - 原因：页面 metadata 显示 MIT，但 dataset card 文本写 Apache License 2.0。
  - 处理：标记为 `blocked_by_license_metadata_mismatch`，在许可不清楚前不用它。
- 问题 4：HQA-Data 虽然最适合下一步 probe，但不是最终测试集。
  - 原因：问题和答案是从 Ubuntu dialogue context 生成的，不是自然用户问题配人工 accepted answer。
  - 处理：只推荐为 Stage67 schema probe 候选，不下载、不跑指标、不 defaultize。

### 测试

局部验证：

```powershell
ruff check src\ts_rag_agent\application\external_eval_dataset_rediscovery.py scripts\rediscover_external_eval_datasets.py tests\test_external_eval_dataset_rediscovery.py
pytest -q tests\test_external_eval_dataset_rediscovery.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

真实 artifact 生成：

```text
Stage66 report: generated
visualizations: 4 generated
recommended_candidate: hqa_data_ubuntu_dialogue
can_run_final_metrics_now: false
can_download_without_user_confirmation: false
```

### 结论

- Stage66 完成第二轮外部数据集发现。
- HQA-Data 是当前最适合进入下一步 schema probe 的第二外部数据集候选。
- HQA-Data 的优势是：Ubuntu 技术支持对话来源、CSV/JSON、train/test、context/span、CC BY 4.0。
- HQA-Data 的核心风险是：QA 是生成的，不是自然用户问题与人工 accepted answer。
- 当前不能跑最终指标，不能下载前置确认之外的数据，不能 defaultize。

### 我学到的

- “再找一个数据集”不等于“马上下载和跑指标”；数据源、许可、schema、citation/context 和 leakage 都要先过关。
- 技术支持领域的数据集常常在 domain fit 与 license/artifact boundary 之间取舍：MSDialog domain 最强，但访问和再分发限制太硬；HQA domain 稍弱但更适合公开 schema probe。
- 镜像数据集不能只看名字像；Hugging Face metadata 与 card 文字冲突时，必须把 license mismatch 当成 blocker。
- StackExchange 类数据源看起来丰富，但其实是派生数据集工程，不是轻量外部评估集。

### 下一步

做 Stage67：HQA-Data local schema probe。

Stage67 只能在用户确认后下载 HQA-Data，并需要记录：

1. 文件 URL、大小、checksum；
2. CSV/JSON schema；
3. context 与 answer span coverage；
4. 与 PrimeQA/MSQA 的 exact/near-duplicate leakage audit；
5. 是否能冻结项目自有 HQA evaluation split。

在 Stage67 通过前，不能运行 HQA baseline 或 Stage51 comparison。

## Stage 67: PrimeQA/TechQA hybrid split protocol dry run

### 阶段目标

本阶段从原先 Stage66 的 HQA 外部数据集 probe 路线，转向用户确认的 PrimeQA/TechQA 文档型 RAG 重切分路线。

用户选择的路线是：

```text
1A: 挑 10% answer document 做完全 document-disjoint 隔离
2A: 其余 grouped rows 按 70/15/15 切成 train/dev/random-test
3A: 把 PrimeQA validation_reference 的 20 行也放进 planning pool
```

本阶段只做 dry-run split plan，不冻结最终 split 文件，不重建索引，不跑模型指标，不改变默认 runtime。

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/primeqa_hybrid_split_plan.py`
- 新增脚本：
  - `scripts/plan_primeqa_hybrid_split.py`
- 新增测试：
  - `tests/test_primeqa_hybrid_split_plan.py`
- 新增文档：
  - `docs/primeqa_hybrid_split.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/dataset_snapshot.md`
  - `docs/evaluation_strategy.md`
  - `docs/external_eval_dataset_rediscovery.md`
  - `docs/learning_journal.md`

### Stage67 运行命令

```powershell
python scripts\plan_primeqa_hybrid_split.py `
  --train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json `
  --dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json `
  --validation-reference data\raw\primeqa_techqa\TechQA\validation\validation_reference.json `
  --output artifacts\primeqa_hybrid_split_stage67.json `
  --assignments-output artifacts\primeqa_hybrid_split_stage67_assignments.jsonl `
  --visualization-dir artifacts\primeqa_hybrid_split_stage67_visuals `
  --document-disjoint-answer-doc-ratio 0.10 `
  --remainder-train-ratio 0.70 `
  --remainder-dev-ratio 0.15 `
  --remainder-test-ratio 0.15 `
  --seed 20260714
```

### 真实输入摘要

```text
total rows: 930
groups: 889
duplicate groups: 40
duplicate rows: 81
answerable rows: 621
unanswerable rows: 309
answerable rate: 0.6677
unique answer docs: 496
unique candidate docs: 28461

source rows:
  primeqa_train: 600
  primeqa_dev: 310
  primeqa_validation: 20
```

### 真实 dry-run split 结果

Document-disjoint 部分：

```text
selected answer docs: 50
selected answer docs sha256: aedb4bef21d64aa58d6a99ccc2099ad4c0b03f07790d6d6c5194a386bd0dbdf9
document_disjoint groups: 121
document_disjoint rows: 126
answer_doc_selected groups: 67
candidate_doc_intersection_only groups: 54
answerable rows: 103
unanswerable rows: 23
```

最终 dry-run 分布：

```text
train:
  rows: 562
  groups: 534
  answerable: 370
  unanswerable: 192
  answerable_rate: 0.6584

dev:
  rows: 121
  groups: 117
  answerable: 76
  unanswerable: 45
  answerable_rate: 0.6281

test:
  rows: 247
  groups: 238
  answerable: 175
  unanswerable: 72
  answerable_rate: 0.7085
  document_disjoint rows: 126
  group_random_test rows: 121
```

### 泄漏检查

```text
normalized_question_answer_doc_groups_do_not_cross_splits: passed
selected_document_answer_docs_only_in_document_disjoint_test: passed
selected_document_candidate_doc_ids_only_in_document_disjoint_test: passed
```

第三项是本阶段补强的检查：不仅确认抽中的 answer doc 不作为非隔离答案文档出现，也确认这些文档不会出现在 train/dev/random-test 的 candidate `DOC_IDS` 中。

### 可视化结果

本阶段生成 4 个本地 SVG artifact：

```text
artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_split_rows.svg
artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_answerable_rows.svg
artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_test_subtypes.svg
artifacts/primeqa_hybrid_split_stage67_visuals/stage67_primeqa_source_rows.svg
```

这些可视化位于本地 `artifacts/`，按 `.gitignore` 不纳入 git。

### Artifact checksum

```text
artifacts/primeqa_hybrid_split_stage67.json
sha256: c59ad5269861d866364468031f08fe577a05aeb84970fd89f48b0226a08718e2

artifacts/primeqa_hybrid_split_stage67_assignments.jsonl
sha256: 713bd017ab52e27a6524733499bf240a79a4eb46503ab5b8b82a526e0940599e
```

`assignments.jsonl` 不包含原始 question text 或 answer text，只保存 source split、question id、assigned split/subtype、group hash、answerability、answer document id、candidate doc count 和 candidate doc hash。

### 问题、原因与修正

- 问题 1：Stage66 文档中仍把 HQA schema probe 写成下一步。
  - 原因：Stage66 之后用户明确最终目标仍是文档型 RAG，并要求从之前数据集重新划分随机测试集。
  - 修正：保留 HQA 发现结果为历史外部数据集 snapshot，同时把当前主线改成 PrimeQA/TechQA hybrid split。
- 问题 2：初版 leakage check 只证明 selected answer docs 不越界，没有在报告里直接证明 candidate `DOC_IDS` 级别也不越界。
  - 原因：assignment artifact 为了避免泄露原始文本和过多候选列表，只保存了 candidate doc hash。
  - 修正：在 planner 内部用原始 group rows 增加 `selected_document_candidate_doc_ids_only_in_document_disjoint_test` 检查，但仍保持 assignment artifact 不写 raw question/answer text。
- 问题 3：旧的 NVIDIA dataset snapshot 仍写着 reserved for evaluation。
  - 原因：这是 Stage53 leakage audit 前的早期快照表述。
  - 修正：更新为“原计划 evaluation，当前因 910/910 exact overlap 不能作为独立 held-out”。

### 验证

```text
ruff check .: passed
pytest -q: 182 passed
git diff --check: passed
git check-ignore artifacts/primeqa_hybrid_split_stage67*: passed
```

### 结论

- Stage67 完成了 PrimeQA/TechQA hybrid split dry-run protocol。
- 当前有可复验的 split plan、row-level assignment artifact、4 个 SVG 可视化和独立文档。
- Stage67 没有冻结 split，没有运行 final metrics，没有使用测试集做调参，也没有改变默认 runtime。
- 旧 Stage31-66 的模型选择证据不能直接迁移成新 split 的 final-test evidence。

### 我学到的

- 当最终目标是“文档型 RAG”，外部 QA 数据集即使更独立，也可能因为不是 technote/document-grounded citation 任务而不适合作为主线。
- PrimeQA validation_reference 可以进入 planning pool 做去重和分布规划，但不能被说成独立测试集。
- 真正的 document-disjoint 不只是 answer doc 不重复，还要检查候选文档列表中也不能出现被隔离文档。
- 新 split 一旦冻结，之前在旧 train/dev 上做出的 tuning 和 comparison 只能作为历史参考，不能作为最终默认化证据。

### 下一步

做 Stage68：review and freeze PrimeQA/TechQA hybrid split。

Stage68 需要先确认是否接受 Stage67 dry-run 分布；如果接受，再把 dry-run plan materialize 成正式 train/dev/test artifacts。冻结前仍不能跑 final metrics，冻结后才可以从新 train/dev/test 边界重新开始训练、开发和最终测试流程。

## Stage 68: Freeze PrimeQA/TechQA hybrid split

### 阶段目标

本阶段承接 Stage67 的 dry-run split plan，把它冻结成项目自己的 PrimeQA/TechQA train/dev/test 边界：

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
```

本阶段只冻结 split artifacts，不重建检索索引，不训练或调参，不运行 final metrics，不改变默认 runtime。

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/primeqa_hybrid_split_freeze.py`
- 新增脚本：
  - `scripts/freeze_primeqa_hybrid_split.py`
- 新增测试：
  - `tests/test_primeqa_hybrid_split_freeze.py`
- 新增文档：
  - `docs/primeqa_hybrid_split_freeze.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/dataset_snapshot.md`
  - `docs/evaluation_strategy.md`
  - `docs/primeqa_hybrid_split.md`
  - `docs/learning_journal.md`

### Stage68 运行命令

```powershell
python scripts\freeze_primeqa_hybrid_split.py `
  --train-questions data\raw\primeqa_techqa\TechQA\training_and_dev\training_Q_A.json `
  --dev-questions data\raw\primeqa_techqa\TechQA\training_and_dev\dev_Q_A.json `
  --validation-reference data\raw\primeqa_techqa\TechQA\validation\validation_reference.json `
  --output artifacts\primeqa_hybrid_split_stage68_freeze.json `
  --split-output-dir artifacts\primeqa_hybrid_split_stage68_splits `
  --visualization-dir artifacts\primeqa_hybrid_split_stage68_visuals `
  --document-disjoint-answer-doc-ratio 0.10 `
  --remainder-train-ratio 0.70 `
  --remainder-dev-ratio 0.15 `
  --remainder-test-ratio 0.15 `
  --seed 20260714
```

### 真实冻结结果

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
total rows: 930
sample_id_sha256: 4adca87e4c2537b38d6fed1b36bebdd77a7fa89ba9ff5577b7421acb6ec67bff

train:
  rows: 562
  groups: 534
  answerable: 370
  unanswerable: 192
  answerable_rate: 0.6584
  sample_id_sha256: 8d632a9f7bb88240f6136259ce039155bd9ffbc42727d91a9e75d676707ff0e5

dev:
  rows: 121
  groups: 117
  answerable: 76
  unanswerable: 45
  answerable_rate: 0.6281
  sample_id_sha256: 3bafd55fc4a604c1dc08c63f67da99a4d5f8872fa7683e9fdb3272c636329e63

test:
  rows: 247
  groups: 238
  answerable: 175
  unanswerable: 72
  answerable_rate: 0.7085
  sample_id_sha256: 8eb92ba16e4066903bf6255850665ce24ac9a525abeed892179ff2fdc48a029f
  document_disjoint rows: 126
  group_random_test rows: 121
```

### 泄漏检查与冻结检查

继承 Stage67 的 leakage checks：

```text
normalized_question_answer_doc_groups_do_not_cross_splits: passed
selected_document_answer_docs_only_in_document_disjoint_test: passed
selected_document_candidate_doc_ids_only_in_document_disjoint_test: passed
```

Stage68 freeze checks：

```text
all_stage67_leakage_checks_passed: passed
all_raw_rows_have_assignments: passed
frozen_row_count_matches_input: passed
train_dev_test_are_nonempty: passed
document_disjoint_test_rows_materialized: passed
```

### 生成的本地 artifacts

Report：

```text
artifacts/primeqa_hybrid_split_stage68_freeze.json
sha256: 9d6873ccbf9ae5d23882c68b64d264b8e957725f5e1ad03c7a3534158df59525
```

Frozen split JSONL：

```text
artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_train.jsonl
rows: 562
sha256: cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_dev.jsonl
rows: 121
sha256: 071c54f80657592bda7f8e4095afc8800a2be112362c3a275191a0fc8e28bd5f

artifacts/primeqa_hybrid_split_stage68_splits/primeqa_hybrid_split_stage68_test.jsonl
rows: 247
sha256: f2479cf636bd40f6d066e1c9be03431f9b37460a8b489a37222523f2d902b1c1
```

这 3 个 JSONL 包含真实 question/answer/candidate docs，原因是后续本地 rebuild loader 和 derived candidate artifacts 需要这些字段；它们位于 ignored `artifacts/`，不会进入 git。

可视化：

```text
artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_split_rows.svg
artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_answerable_rows.svg
artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_test_subtypes.svg
artifacts/primeqa_hybrid_split_stage68_visuals/stage68_primeqa_frozen_source_rows.svg
```

### 问题、原因与修正

- 问题 1：Stage67 只输出 assignment artifact，没有正式 train/dev/test JSONL。
  - 原因：Stage67 设计为 dry-run，只验证分布与泄漏边界。
  - 修正：Stage68 新增 freeze module，把同一 split 边界 materialize 成 3 个本地 JSONL。
- 问题 2：冻结 report 不应包含 raw question/answer text。
  - 原因：公开提交需要保持 public-safe；raw 数据应该只留在 ignored artifacts。
  - 修正：`freeze_primeqa_hybrid_split` 返回 report + samples bundle；report 只保存统计、检查和 checksum，writer 单独把 raw samples 写入 ignored JSONL。
- 问题 3：JSONL 需要保持 line-oriented 稳定。
  - 原因：raw answer/text 里可能出现 Unicode line separator。
  - 修正：JSONL writer 对 `U+2028` 和 `U+2029` 做转义，测试覆盖该行为。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_split_freeze.py scripts\freeze_primeqa_hybrid_split.py tests\test_primeqa_hybrid_split_freeze.py
pytest -q tests\test_primeqa_hybrid_split_freeze.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 185 passed
```

### 结论

- Stage68 已冻结 `primeqa_hybrid_stage68_v1`。
- 冻结边界有 report、train/dev/test JSONL、4 个 SVG 可视化、文档和测试。
- 当前可以进入 rebuild 阶段，但仍不能运行 final metrics。
- 默认 runtime 仍然 unchanged。

### 我学到的

- “冻结 split” 与“跑指标”必须分开；冻结只是定义边界，不代表模型质量已经验证。
- report 和 raw split artifacts 应该分层：report 负责可公开审计，raw JSONL 负责本地可执行重建。
- 只要 split 边界变了，后续训练/验证必须从新 train/dev/test 边界重新构建，不能沿用旧 Stage31-66 的调参产物当最终证据。

### 下一步

做 Stage69：rebuild PrimeQA train/dev/test loaders and derived candidate artifacts。

Stage69 应该让后续训练和开发流程读取 `primeqa_hybrid_stage68_v1`，并明确阻止 test split 参与调参。只有 rebuild 完成后，才可以重新跑 train/dev 上的基线、候选 reranker 或 verified RAG 流程。

## Stage 69: Rebuild PrimeQA hybrid loaders and train/dev candidate artifacts

### 阶段目标

本阶段基于 Stage68 冻结的 `primeqa_hybrid_stage68_v1` 重建后续开发所需的本地 artifacts：

1. 读取 frozen train/dev/test JSONL；
2. 导出 PrimeQA-compatible question JSON；
3. 只用 train/dev 构建 candidate-reranker derived artifacts；
4. 显式阻止 test split 进入训练或调参。

本阶段不运行 final metrics，不训练或选择最终模型，不改变默认 runtime。

### 新增与修改

- 新增 loader：
  - `src/ts_rag_agent/infrastructure/primeqa_hybrid_split_loader.py`
- 新增应用模块：
  - `src/ts_rag_agent/application/primeqa_hybrid_split_rebuild.py`
- 新增脚本：
  - `scripts/rebuild_primeqa_hybrid_artifacts.py`
- 新增测试：
  - `tests/test_primeqa_hybrid_split_loader.py`
  - `tests/test_primeqa_hybrid_split_rebuild.py`
- 新增文档：
  - `docs/primeqa_hybrid_rebuild.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/evaluation_strategy.md`
  - `docs/primeqa_hybrid_split_freeze.md`
  - `docs/learning_journal.md`

### Stage69 运行命令

```powershell
python scripts\rebuild_primeqa_hybrid_artifacts.py `
  --output artifacts\primeqa_hybrid_rebuild_stage69.json `
  --question-output-dir artifacts\primeqa_hybrid_rebuild_stage69_questions `
  --candidate-output artifacts\primeqa_hybrid_rebuild_stage69_candidates.jsonl `
  --candidate-summary-output artifacts\primeqa_hybrid_rebuild_stage69_candidates.summary.json `
  --visualization-dir artifacts\primeqa_hybrid_rebuild_stage69_visuals `
  --candidate-splits train,dev `
  --retrieval-top-k 5 `
  --evidence-selector hybrid-routing `
  --max-candidates-per-document 3 `
  --candidate-limit 25 `
  --min-candidate-score 2.0
```

### Loader contract

```text
split_name: primeqa_hybrid_stage68_v1
protocol_version: primeqa_hybrid_split_v1
PrimeQAQuestion.id: source_split:QUESTION_ID
candidate_artifact_splits: train, dev
forbidden_tuning_splits: test
```

使用 `source_split:QUESTION_ID` 是为了避免 validation rows 与 dev rows 复用原始 `QUESTION_ID` 后造成 candidate id 冲突。

### 真实 loaded split 摘要

```text
train:
  rows: 562
  answerable: 370
  unanswerable: 192
  unique_answer_doc_ids: 309
  unique_candidate_doc_ids: 19602

dev:
  rows: 121
  answerable: 76
  unanswerable: 45
  unique_answer_doc_ids: 71
  unique_candidate_doc_ids: 5373

test:
  rows: 247
  answerable: 175
  unanswerable: 72
  unique_answer_doc_ids: 145
  unique_candidate_doc_ids: 8954
```

test split 只被加载和导出 question artifact，用来验证 loader contract；没有进入 candidate training artifact。

### 真实 candidate artifact 摘要

```text
selector: hybrid_routing_answer_aware_mcpd3_section_span_mcpd1
candidate_splits: train, dev
total answerable questions used: 446
total candidate rows: 5993
train questions: 370
dev questions: 76
train candidate rows: 5006
dev candidate rows: 987
average rows per question: 13.4372
average top candidate token F1: 0.2343
average best candidate token F1: 0.4263
average oracle gain vs top candidate: 0.1920
questions with gold-document candidate: 274
gold-document candidate rows: 716
```

Rows by route：

```text
other: 2658
error_or_log: 1300
install_upgrade_config: 1087
how_to_or_lookup: 592
security_bulletin_vulnerability_detail: 236
limitation_or_restriction: 60
security_bulletin_post_fix_behavior: 30
security_bulletin_remediation: 30
```

### Guard checks

```text
test_split_not_used_for_candidate_training_artifact: passed
candidate_build_splits_match_allowed_train_dev: passed
all_loaded_splits_have_rows: passed
candidate_rows_have_no_test_split: passed
```

### 生成的本地 artifacts

Report：

```text
artifacts/primeqa_hybrid_rebuild_stage69.json
sha256: 7e92de033c58fe5623e0d4bff422f7e5f3acd1bb5673f7f919af9315ea5ec633
```

Question artifacts：

```text
artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_train_Q_A.json
rows: 562
sha256: 1db9f81f5ad0aec1f79eb42d90d201117e84f1f29331f31e63dc31a8a6c0105c

artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_dev_Q_A.json
rows: 121
sha256: 7b84c09f8dd1e30aacf10e075a8c1da8f732cf84643fa789fefda42ca4b43081

artifacts/primeqa_hybrid_rebuild_stage69_questions/primeqa_hybrid_stage69_test_Q_A.json
rows: 247
sha256: 1ad3c643131c8fddbed36aefb11141cf616022374f2774a248577d62bd04a021
```

Candidate artifacts：

```text
artifacts/primeqa_hybrid_rebuild_stage69_candidates.jsonl
rows: 5993
sha256: d379d59f5172394a40bcd1852aa8188f2dec18d4abcae20d08acd992a802da4d

artifacts/primeqa_hybrid_rebuild_stage69_candidates.summary.json
sha256: a753848fe2f6c111e2a376c53522ce5ca67536d0203d5addd135f86beaa6332d
```

可视化：

```text
artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_loaded_split_rows.svg
artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_loaded_answerable_rows.svg
artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_candidate_rows_by_split.svg
artifacts/primeqa_hybrid_rebuild_stage69_visuals/stage69_primeqa_candidate_questions_by_split.svg
```

### 问题、原因与修正

- 问题 1：旧 candidate-reranker build 默认读取原始 PrimeQA train/dev 文件。
  - 原因：Stage68 之前没有项目自有 frozen split loader。
  - 修正：新增 frozen split loader，把 Stage68 JSONL 转成 `PrimeQAQuestion`，并导出 PrimeQA-compatible question JSON。
- 问题 2：validation rows 可能复用 dev 的原始 `QUESTION_ID`。
  - 原因：PrimeQA validation_reference 与 dev 存在重复样本。
  - 修正：loader 将 `PrimeQAQuestion.id` 设为 `source_split:QUESTION_ID`，避免 candidate id 冲突。
- 问题 3：test split 必须锁住，不能进入调参 candidate artifact。
  - 原因：Stage68 已经把 test 冻结为未来最终评估边界。
  - 修正：Stage69 rebuild module 和 CLI 对 `candidate_splits` 做 guard；`test` 会被拒绝，真实 artifact 只包含 train/dev。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\infrastructure\primeqa_hybrid_split_loader.py src\ts_rag_agent\application\primeqa_hybrid_split_rebuild.py scripts\rebuild_primeqa_hybrid_artifacts.py tests\test_primeqa_hybrid_split_loader.py tests\test_primeqa_hybrid_split_rebuild.py
pytest -q tests\test_primeqa_hybrid_split_loader.py tests\test_primeqa_hybrid_split_rebuild.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 189 passed
git diff --check: passed
git check-ignore artifacts/primeqa_hybrid_rebuild_stage69*: passed
```

真实运行耗时：

```text
load_splits: 0.018s
load_documents: 0.887s
bm25_index: 4.150s
candidate_build: 106.450s
total: 111.505s
```

### 结论

- Stage69 已完成 frozen split loader 和 train/dev candidate artifact rebuild。
- 后续开发可以读取 `primeqa_hybrid_stage68_v1` 的 train/dev artifacts 重新跑基线和开发检查。
- test split 仍然 locked，不能用于调参或多轮模型选择。
- 当前没有运行 final metrics，也没有改变 runtime。

### 我学到的

- 重新划分 split 之后，不只是换几个文件路径；所有依赖 question id、candidate id、训练边界的代码都要重新对齐。
- validation rows 混入 planning pool 后，必须重新定义稳定样本 ID，否则会在 candidate row 层发生隐性冲突。
- train/dev derived artifacts 可以计算 gold labels，但这些 labels 只能服务离线开发，不能泄漏进 runtime features。

### 下一步

做 Stage70：rerun PrimeQA train/dev baselines and candidate-reranker development checks。

Stage70 应该只在 `primeqa_hybrid_stage68_v1` 的 train/dev 上重跑 baseline 与候选 reranker 开发检查，继续保持 test split locked，不运行 final test metrics。

## Stage 70: PrimeQA hybrid train/dev development checks

### 阶段目标

本阶段基于 Stage68 冻结的 `primeqa_hybrid_stage68_v1` 和 Stage69 重建的
train/dev candidate artifact，做开发阶段检查：

1. 只在 train/dev 上重新跑 BM25 baseline；
2. 审计 Stage69 candidate JSONL 是否只包含 train/dev；
3. 检查 runtime features 是否混入 offline gold label keys；
4. 继续锁住 frozen test split，不运行 final test metrics；
5. 不改变默认 runtime policy。

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/primeqa_hybrid_development_checks.py`
- 新增脚本：
  - `scripts/run_primeqa_hybrid_development_checks.py`
- 新增测试：
  - `tests/test_primeqa_hybrid_development_checks.py`
- 新增文档：
  - `docs/primeqa_hybrid_development_checks.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/evaluation_strategy.md`
  - `docs/primeqa_hybrid_rebuild.md`
  - `docs/learning_journal.md`

### Stage70 运行命令

```powershell
python scripts\run_primeqa_hybrid_development_checks.py `
  --output artifacts\primeqa_hybrid_development_checks_stage70.json `
  --visualization-dir artifacts\primeqa_hybrid_development_checks_stage70_visuals `
  --top-k 1,5,10 `
  --bm25-k1 1.5 `
  --bm25-b 0.75
```

### 真实 BM25 train/dev baseline

```text
train:
  total_questions: 562
  evaluated_questions: 370
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  mrr: 0.5023

dev:
  total_questions: 121
  evaluated_questions: 76
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  mrr: 0.5331
```

这些只是开发集 baseline，不是最终 held-out test 结果。

### Candidate artifact 检查结果

```text
row_count: 5993
train rows: 5006
dev rows: 987
train questions: 370
dev questions: 76
rows_with_runtime_features: 5993
rows_with_gold_labels: 5993
rows_with_test_split: 0
rows_with_forbidden_runtime_gold_keys: 0
```

### Guard checks

```text
development_baseline_splits_are_train_dev_only: passed
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
candidate_artifact_checks_passed: passed
final_test_metrics_not_run: passed
```

### 生成的本地 artifacts

这些 artifacts 位于 `artifacts/`，被 git ignore，不随提交进入仓库。

Report：

```text
artifacts/primeqa_hybrid_development_checks_stage70.json
sha256: 3d85033ac4b831fac4d2978af255d7e28c301e638fb02e0b87f97e4d2ea3e92d
```

可视化：

```text
artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_bm25_hit_at_k.svg
sha256: a2122ccc07f44621832af61d6900b17a0beee3651d5f84a06f1637c04effcd29

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_bm25_mrr.svg
sha256: a8439d71ebc95fdab64f4a7a96b3924701ea0804d6a0598fcc24b76f52f7b44b

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_candidate_questions_by_split.svg
sha256: 679c550f4c44254f267766f000fc2aae0678d80d398def731fab4fd172b60f67

artifacts/primeqa_hybrid_development_checks_stage70_visuals/stage70_primeqa_candidate_rows_by_split.svg
sha256: 8b4665863e52697a6d2cb80ee1a91392dd6aeb51c9ea95767b0452d5916f8bea
```

### 问题、原因与修正

- 问题 1：Stage69 candidate artifact 含有 offline gold labels，需要防止后续误当成 runtime features。
  - 原因：train/dev candidate development 需要 gold labels 计算候选质量，但 runtime 不能使用这些标签。
  - 修正：Stage70 scan 同时统计 `runtime_features` 和 `gold_labels`，并显式检查 runtime feature keys 不包含 gold label keys。
- 问题 2：开发检查容易被误解为 final evaluation。
  - 原因：当前已经有 frozen test split，但本阶段只应做 train/dev 开发检查。
  - 修正：report 增加 `final_test_metrics_not_run` guard，decision 明确 `can_run_final_test_metrics_now: false` 和 `can_use_test_for_tuning: false`。
- 问题 3：需要可视化结果但不能提交大数据产物。
  - 原因：`artifacts/` 按仓库策略被 ignore，真实产物只保留在本地。
  - 修正：生成 SVG 可视化并记录路径与 checksum，提交代码和文档，不提交 artifact 文件。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_development_checks.py scripts\run_primeqa_hybrid_development_checks.py tests\test_primeqa_hybrid_development_checks.py
pytest -q tests\test_primeqa_hybrid_development_checks.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

真实运行耗时：

```text
load_splits: 0.015s
load_documents: 0.898s
bm25_index: 4.278s
bm25_evaluate: 107.806s
candidate_checks: 0.050s
total: 113.047s
```

### 结论

- Stage70 已完成 train/dev-only development checks。
- Stage70 没有加载 test split 跑 BM25 baseline。
- Stage70 没有运行 final test metrics。
- Stage70 没有改变默认 runtime policy。
- 当前可以进入 Stage71，但 Stage71 仍只能在 train/dev 上做 candidate-reranker policy development。

### 我学到的

- 冻结 test split 之后，每一个开发脚本都应该把 split contract 写进 report，而不是只靠命令约定。
- candidate artifact 可以同时包含 runtime features 和 offline labels，但必须用结构化检查保证两者边界清楚。
- 可视化不只是展示结果，也能帮助快速看出 train/dev 分布是否异常；但本地 artifact 不能代替文档里的可追溯 checksum。

### 下一步

做 Stage71：run train/dev candidate-reranker policy development on
`primeqa_hybrid_stage68_v1`。

Stage71 只能使用 train/dev candidate artifact 做开发检查，继续保持 frozen test split locked，不运行 final test metrics，不改变默认 runtime。

## Stage 71: PrimeQA hybrid candidate-reranker development

### 阶段目标

本阶段基于 Stage68 冻结的 `primeqa_hybrid_stage68_v1` 和 Stage69 生成的
train/dev candidate artifact，做候选 reranker 开发实验：

1. 只在 train split 上做 grouped cross-validation；
2. 同时比较 `logistic_best_candidate` 和 `ridge_candidate_token_f1`；
3. 对两个模型都做 train-to-dev guarded policy validation；
4. 使用 frozen train/dev JSONL 读取 gold answers，按 `split::sample_id` 对齐 Stage69 candidate `question_id`；
5. 不读取 test split，不运行 final metrics，不改变默认 runtime。

### 新增与修改

- 新增应用模块：
  - `src/ts_rag_agent/application/primeqa_hybrid_candidate_reranker_development.py`
- 新增脚本：
  - `scripts/run_primeqa_hybrid_candidate_reranker_development.py`
- 新增测试：
  - `tests/test_primeqa_hybrid_candidate_reranker_development.py`
- 新增文档：
  - `docs/primeqa_hybrid_candidate_reranker_development.md`
- 更新文档：
  - `docs/data_strategy.md`
  - `docs/evaluation_strategy.md`
  - `docs/primeqa_hybrid_development_checks.md`
  - `docs/learning_journal.md`

### Stage71 运行命令

```powershell
python scripts\run_primeqa_hybrid_candidate_reranker_development.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_development_stage71.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_development_stage71_visuals `
  --fold-count 5 `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 3
```

### 真实 train-only grouped CV 结果

```text
logistic_best_candidate:
  baseline_average_token_f1: 0.2269
  selected_average_token_f1: 0.2523
  average_delta_vs_top_candidate: +0.0254
  oracle_gap_closed_rate: 0.1344
  improved: 120
  regressed: 88
  tied: 162

ridge_candidate_token_f1:
  baseline_average_token_f1: 0.2269
  selected_average_token_f1: 0.2652
  average_delta_vs_top_candidate: +0.0383
  oracle_gap_closed_rate: 0.2025
  improved: 120
  regressed: 87
  tied: 163
```

train-only CV 中 `ridge_candidate_token_f1` 更强，但这仍然只是 train split 内的开发证据。

### 真实 train-to-dev guarded policy validation

Dev holdout top3 proxy：

```text
logistic_best_candidate:
  best top3 policy: candidate_score_gte_60
  best top3 delta: +0.0004
  best top3 regressions: 0
  best top3 gold citation delta: +0

ridge_candidate_token_f1:
  best top3 policy: stage36_main
  best top3 delta: +0.0003
  best top3 regressions: 1
  best top3 gold citation delta: +0
```

Dev holdout single-candidate proxy：

```text
logistic stage36_main:
  delta: +0.0436
  replacements: 21
  regressions: 4
  gold citation delta: +7

logistic candidate_score_gte_60:
  delta: +0.0337
  replacements: 17
  regressions: 4
  gold citation delta: +6

ridge stage36_main:
  delta: +0.0144
  replacements: 8
  regressions: 0
  gold citation delta: +3

ridge candidate_score_gte_60:
  delta: +0.0027
  replacements: 3
  regressions: 0
  gold citation delta: +2
```

结论要谨慎：single-candidate proxy 的增益明显，但 top3 rewrite proxy 几乎不动，所以还不能进入 final test 或 runtime 默认化。

### Guard checks

```text
candidate_artifact_splits_are_train_dev_only: passed
candidate_summary_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
gold_answer_splits_are_train_dev_only: passed
train_cv_uses_train_only: passed
split_validations_are_train_to_dev: passed
candidate_artifact_checks_passed: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

### 生成的本地 artifacts

这些 artifacts 位于 `artifacts/`，被 git ignore，不随提交进入仓库。

Report：

```text
artifacts/primeqa_hybrid_candidate_reranker_development_stage71.json
sha256: bb5c665295a7cd0768e8c69d805c0dd60c5fdfb5839aba4cd77f7161c35a4573
```

可视化目录：

```text
artifacts/primeqa_hybrid_candidate_reranker_development_stage71_visuals/
svg files: 20
```

重点可视化：

```text
candidate_reranker_model_delta.svg
sha256: ef1bdf5408b008360ba1beeeb9015dbad1d289a62cdd62998b21d9887ed0e19f

candidate_reranker_model_gap_closed.svg
sha256: 1ea1fca9730233a86aecd321840b11c788332a41b88039fb55c063769f6bce97

stage71_logistic_best_candidate_dev_holdout_top3_policy_delta.svg
sha256: 56b1fb622bf6ca7ad82e0071442241c3b193bae44d11d6fa2075f11e7eedb689

stage71_ridge_candidate_token_f1_dev_holdout_top3_policy_delta.svg
sha256: 79bb7896fe2d7df4ebe01e77d25baeb1c7a7a12f5b1d6c97bb9bdfd7aa8f23a8
```

### 问题、原因与修正

- 问题 1：旧的 split-validation 脚本从原始 PrimeQA train/dev 读取 gold answers，不能直接适配 Stage68 的 `source_split:QUESTION_ID` 新 ID。
  - 原因：Stage69 candidate row 的 `question_id` 是 `sample_id`，例如 `primeqa_dev:DEV_Q000`，而旧脚本的 gold answer key 只使用原始 dev/train question ID。
  - 修正：新增 Stage71 专用封装，从 frozen train/dev JSONL 读取 answer，并用 `split::sample_id` 构建 gold answer key。
- 问题 2：第一次真实运行只对 logistic 做 policy validation，但 train-only CV 显示 ridge 更强。
  - 原因：沿用了旧 Stage39/40 的 logistic policy lineage，覆盖不够完整。
  - 修正：扩展 Stage71，让 policy validation 默认跟随 `--models`，对 logistic 和 ridge 都做 train-to-dev validation。
- 问题 3：top3 proxy 和 single-candidate proxy 给出的信号强度不一致。
  - 原因：single-candidate proxy 只看 leading candidate，而 top3 rewrite proxy 会保留多个候选句子，更接近组合答案风险。
  - 修正：结论中明确只允许进入 changed-case review，不允许直接 runtime/defaultization。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_candidate_reranker_development.py scripts\run_primeqa_hybrid_candidate_reranker_development.py tests\test_primeqa_hybrid_candidate_reranker_development.py
pytest -q tests\test_primeqa_hybrid_candidate_reranker_development.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

真实运行耗时：

```text
load_candidates: 0.052s
load_train_dev_splits: 0.038s
train_only_cv: 0.635s
train_to_dev_policy_validation: 1.473s
candidate_checks: 0.002s
total: 2.200s
```

### 结论

- Stage71 已完成 train/dev-only candidate-reranker development。
- Stage71 没有读取 frozen test split。
- Stage71 没有运行 final test metrics。
- Stage71 没有改变默认 runtime policy。
- ridge 在 train-only CV 上优于 logistic，但 dev top3 proxy 的收益非常小。
- 当前只能进入 Stage72 changed-case review，不能直接进入 final test gate。

### 我学到的

- 当 split ID contract 变了，gold answer loader 也必须一起变；否则评估脚本看似能跑，实际 key 对不齐。
- CV 最优模型和 holdout policy 最优信号可能不一致，必须把 model-level 和 policy-level 证据分开记录。
- single-candidate proxy 容易显得乐观，top-k rewrite proxy 更能暴露组合答案层面的收益变小问题。

### 下一步

做 Stage72：review Stage71 train/dev candidate-reranker changed cases。

Stage72 应该检查 logistic/ridge 在 dev 上改变了哪些问题、哪些是 top3 proxy 微弱收益或潜在风险，再决定是否存在值得进入 final-test evaluation gate 的候选策略。test split 继续 locked。

## Stage72：PrimeQA hybrid candidate-reranker changed-case review

时间：2026-07-14

### 目标

继续上一阶段的推荐路线，只复盘 Stage71 在 train/dev 范围内的 candidate-reranker changed cases，并补上可视化结果。

这一阶段的边界是：

- 不读取 frozen test split；
- 不运行 final test metrics；
- 不使用 test 做 tuning；
- 不改变默认 runtime policy；
- 只判断 dev changed-case 信号是否足够支持进入下一步 gate 决策。

### 本次新增

- 新增 `src/ts_rag_agent/application/primeqa_hybrid_candidate_reranker_changed_case_review.py`。
- 新增 `scripts/review_primeqa_hybrid_candidate_reranker_changed_cases.py`。
- 新增 `tests/test_primeqa_hybrid_candidate_reranker_changed_case_review.py`。
- 新增文档 `docs/primeqa_hybrid_candidate_reranker_changed_case_review.md`。
- 更新 `docs/evaluation_strategy.md`、`docs/data_strategy.md`、`docs/primeqa_hybrid_candidate_reranker_development.md`。

### 真实运行命令

```powershell
python scripts\review_primeqa_hybrid_candidate_reranker_changed_cases.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_changed_case_review_stage72.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 3 `
  --sample-limit 20
```

### 真实结果

Dev top3 changed-case summary：

```text
logistic_best_candidate / stage36_main:
  delta: +0.0001
  regressions: 1
  changed cases: 21
  gold citation delta: +0

logistic_best_candidate / candidate_score_gte_60:
  delta: +0.0004
  regressions: 0
  changed cases: 17
  gold citation delta: +0

ridge_candidate_token_f1 / stage36_main:
  delta: +0.0003
  regressions: 1
  changed cases: 8
  gold citation delta: +0

ridge_candidate_token_f1 / candidate_score_gte_60:
  delta: +0.0000
  regressions: 0
  changed cases: 3
  gold citation delta: +0
```

Policy-vs-main changed cases：

```text
logistic_best_candidate:
  candidate_score_gte_60 differs from stage36_main on 4 / 76 dev cases
  better / tied / worse: 1 / 2 / 1
  average candidate delta vs main on changed cases: +0.0059

ridge_candidate_token_f1:
  candidate_score_gte_60 differs from stage36_main on 5 / 76 dev cases
  better / tied / worse: 1 / 2 / 2
  average candidate delta vs main on changed cases: -0.0039
```

当前 train/dev-only 最好的 dev top3 proxy 仍然是：

```text
model: logistic_best_candidate
policy: candidate_score_gte_60
dev top3 delta: +0.0004
regressions: 0
gold citation delta: +0
```

### 可视化结果

本次生成了 4 个 SVG：

```text
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_dev_top3_delta_by_policy.svg
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_changed_cases_by_policy.svg
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_candidate_score_vs_main_outcomes.svg
artifacts/primeqa_hybrid_candidate_reranker_changed_case_review_stage72_visuals/stage72_residual_regressions_by_policy.svg
```

报告和 SVG 都在 `artifacts/` 下，按项目 git policy 被 ignore，不提交进仓库。

### Guard checks

```text
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
gold_answer_splits_are_train_dev_only: passed
stage71_final_test_metrics_not_run: passed
stage72_review_uses_dev_holdout_only: passed
stage72_report_is_public_safe_no_raw_answer_text: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

额外检查：

```text
git check-ignore: Stage72 report and SVG are ignored by artifacts/*
raw text search: no raw candidate sentence or answer text found; only the guard name contains raw_answer_text wording
```

### 问题、原因与修正

- 问题 1：Stage71 的 top3 proxy 收益太小，不能只看单一平均 delta。
  - 原因：candidate-reranker 改动的是候选句顺序，top3 组合答案会稀释 single-candidate proxy 的收益。
  - 修正：Stage72 按 policy、model、changed-case、policy-vs-main 维度拆开复盘，并输出 residual regression 图。
- 问题 2：changed-case review 需要可公开记录，但原始候选句和答案可能包含数据集文本。
  - 原因：项目当前不打算公开数据集，记录不能泄露 raw answer/candidate text。
  - 修正：报告只写 case ID、route、rank、score、document ID 和 metric delta，不写原文。
- 问题 3：Stage72 结果仍然不足以自动打开 final test gate。
  - 原因：最佳 dev top3 delta 只有 +0.0004，虽然 regressions 为 0，但收益非常微弱。
  - 修正：结论中明确 `can_open_final_test_gate_now: false`，下一步只做 gate path 决策。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_candidate_reranker_changed_case_review.py scripts\review_primeqa_hybrid_candidate_reranker_changed_cases.py tests\test_primeqa_hybrid_candidate_reranker_changed_case_review.py
pytest -q tests\test_primeqa_hybrid_candidate_reranker_changed_case_review.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

真实报告运行耗时：

```text
load_inputs: 0.057s
load_train_dev_splits: 0.047s
review_changed_cases: 0.234s
guard_checks: 0.000s
total: 0.338s
```

### 结论

- Stage72 已完成 train/dev-only changed-case review。
- Stage72 生成了可视化结果。
- Stage72 没有读取 test split。
- Stage72 没有运行 final test metrics。
- Stage72 没有改变默认 runtime policy。
- `logistic_best_candidate / candidate_score_gte_60` 是当前 dev top3 proxy 最稳的候选，但收益仍然非常小。
- 当前不能自动进入 final test metrics，只能进入 Stage73 gate path decision。

### 我学到的

- changed-case review 不能只看“改了多少”，还要看改动相对原 policy 是 improvement、tie 还是 regression。
- 对不公开的数据集，实验报告必须从设计上避免写入原始文本，而不是事后再清理。
- 当 dev proxy 收益很小，即使 regression 为 0，也应该把“是否打开 final test gate”作为显式决策，而不是自动推进。

### 下一步

做 Stage73：决定 gate path。

Stage73 只能在两个方向中做明确选择：

- 继续 refine train/dev candidate-reranker policy gates；
- 或显式批准一次 one-time final-test evaluation gate，只评估一次选定候选，不用 test 做 tuning。

## Stage73：PrimeQA hybrid candidate-reranker top10 diagnostic

时间：2026-07-15

### 目标

回应“top10 跑一下看看”的请求，只在 train/dev 上跑 reranker answer proxy 的 top10 diagnostic。

本阶段继续遵守新的硬边界：

- 训练阶段永远不能在测试集上做评估；
- 不读取 frozen test split；
- 不运行 final test metrics；
- 不使用 test 做 tuning；
- 不改变默认 runtime policy。

### 本次新增

- 新增 `src/ts_rag_agent/application/primeqa_hybrid_candidate_reranker_topk_diagnostic.py`。
- 新增 `scripts/run_primeqa_hybrid_candidate_reranker_top10_diagnostic.py`。
- 新增 `tests/test_primeqa_hybrid_candidate_reranker_topk_diagnostic.py`。
- 新增 `docs/primeqa_hybrid_candidate_reranker_top10_diagnostic.md`。
- 更新 `docs/evaluation_strategy.md`、`docs/data_strategy.md`、`docs/primeqa_hybrid_candidate_reranker_changed_case_review.md`。

### 真实运行命令

```powershell
python scripts\run_primeqa_hybrid_candidate_reranker_top10_diagnostic.py `
  --output artifacts\primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73.json `
  --visualization-dir artifacts\primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals `
  --models logistic_best_candidate,ridge_candidate_token_f1 `
  --max-answer-candidates 10
```

### 真实结果

Train-only CV top10 proxy：

```text
logistic_best_candidate / stage36_main:
  delta: +0.0000
  policy F1: 0.1845
  baseline F1: 0.1845
  oracle F1: 0.1893
  replacements: 105
  improved / regressed / tied: 0 / 0 / 370
  gold citation delta: +0

logistic_best_candidate / candidate_score_gte_60:
  delta: +0.0000
  policy F1: 0.1845
  baseline F1: 0.1845
  oracle F1: 0.1893
  replacements: 77
  improved / regressed / tied: 0 / 0 / 370
  gold citation delta: +0

ridge_candidate_token_f1 / stage36_main:
  delta: +0.0000
  policy F1: 0.1845
  baseline F1: 0.1845
  oracle F1: 0.1893
  replacements: 40
  improved / regressed / tied: 0 / 0 / 370
  gold citation delta: +0

ridge_candidate_token_f1 / candidate_score_gte_60:
  delta: +0.0000
  policy F1: 0.1845
  baseline F1: 0.1845
  oracle F1: 0.1893
  replacements: 25
  improved / regressed / tied: 0 / 0 / 370
  gold citation delta: +0
```

Dev holdout top10 proxy：

```text
logistic_best_candidate / stage36_main:
  delta: +0.0000
  policy F1: 0.1933
  baseline F1: 0.1933
  oracle F1: 0.1947
  replacements: 21
  improved / regressed / tied: 0 / 0 / 76
  gold citation delta: +0

logistic_best_candidate / candidate_score_gte_60:
  delta: +0.0000
  policy F1: 0.1933
  baseline F1: 0.1933
  oracle F1: 0.1947
  replacements: 17
  improved / regressed / tied: 0 / 0 / 76
  gold citation delta: +0

ridge_candidate_token_f1 / stage36_main:
  delta: +0.0000
  policy F1: 0.1933
  baseline F1: 0.1933
  oracle F1: 0.1947
  replacements: 8
  improved / regressed / tied: 0 / 0 / 76
  gold citation delta: +0

ridge_candidate_token_f1 / candidate_score_gte_60:
  delta: +0.0000
  policy F1: 0.1933
  baseline F1: 0.1933
  oracle F1: 0.1947
  replacements: 3
  improved / regressed / tied: 0 / 0 / 76
  gold citation delta: +0
```

### top3 与 top10 对比

Stage72 top3 还有一个非常小的 dev signal：

```text
logistic_best_candidate / candidate_score_gte_60:
  top3 dev delta: +0.0004
  regressions: 0
  gold citation delta: +0
```

Stage73 top10 后，这个 signal 消失：

```text
logistic_best_candidate / candidate_score_gte_60:
  top10 dev delta: +0.0000
  regressions: 0
  gold citation delta: +0
```

### 可视化结果

本次生成 20 个 SVG，位于：

```text
artifacts/primeqa_hybrid_candidate_reranker_top10_diagnostic_stage73_visuals/
```

重点图：

```text
stage73_logistic_best_candidate_dev_holdout_top10_policy_delta.svg
stage73_logistic_best_candidate_dev_holdout_top10_policy_regressions.svg
stage73_ridge_candidate_token_f1_dev_holdout_top10_policy_delta.svg
stage73_ridge_candidate_token_f1_dev_holdout_top10_policy_regressions.svg
```

报告和 SVG 都在 `artifacts/` 下，按项目 git policy 被 ignore，不提交进仓库。

### Guard checks

```text
candidate_artifact_splits_are_train_dev_only: passed
candidate_summary_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
gold_answer_splits_are_train_dev_only: passed
train_cv_uses_train_only: passed
split_validations_are_train_to_dev: passed
candidate_artifact_checks_passed: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
stage73_topk_window_matches_requested_value: passed
stage73_split_validations_are_train_to_dev: passed
stage73_final_test_metrics_not_run: passed
stage73_default_runtime_policy_unchanged: passed
```

### 问题、原因与修正

- 问题 1：直接复用 Stage71 runner 会把 top10 结果写成 Stage71。
  - 原因：原 runner 的 stage metadata 固定为 Stage71。
  - 修正：新增 Stage73 top-k diagnostic 封装，复用 train/dev 评估逻辑，但单独写 Stage73 metadata、decision 和 visualization prefix。
- 问题 2：第一版 Stage73 摘要只把 dev holdout top10 提到顶层。
  - 原因：我最初按 gate decision 的视角只汇总 dev。
  - 修正：把 `train_cv_policy_rows` 和 `dev_holdout_policy_rows` 都提升到 `topk_diagnostic` 顶层，保证“训练/验证集”都有结构化结果。
- 问题 3：top10 结果没有带来 F1 gain。
  - 原因：当 answer proxy 包含 10 个候选时，替换 leading candidate 不足以改变整体 composed answer token F1。
  - 修正：结论中明确 top10 削弱继续推进当前 reranker policy 的理由，不进入 final test gate。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_candidate_reranker_topk_diagnostic.py scripts\run_primeqa_hybrid_candidate_reranker_top10_diagnostic.py tests\test_primeqa_hybrid_candidate_reranker_topk_diagnostic.py
pytest -q tests\test_primeqa_hybrid_candidate_reranker_topk_diagnostic.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

真实报告运行耗时：

```text
load_candidates: 0.056s
load_train_dev_splits: 0.052s
train_only_cv: 0.692s
train_to_dev_policy_validation: 2.087s
candidate_checks: 0.002s
total: 2.890s
```

### 结论

- Stage73 已完成 train/dev-only top10 diagnostic。
- Stage73 没有读取 test split。
- Stage73 没有运行 final test metrics。
- Stage73 没有改变默认 runtime policy。
- top10 proxy 在 train-only CV 和 dev holdout 上都是 `+0.0000`。
- 这说明当前 candidate-reranker policy 的收益只存在于非常窄的 top3/leading-candidate 表面；扩到 top10 后不再体现。
- 当前不能进入 final test metrics，也不应该把当前 reranker policy defaultize。

### 我学到的

- top-k answer proxy 的窗口大小会显著改变结论：top3 的微小信号在 top10 下可能完全消失。
- 如果更宽的组合答案窗口没有收益，那么 reranker 的排序微调不一定能转化为答案层面的真实改进。
- 对训练/验证阶段，报告应该同时写 train-CV 和 dev-holdout，而不是只写 dev。

### 下一步

做 Stage74：决定是否停止当前 reranker-policy 路线，或只在 train/dev 上重新设计更有实质影响的 reranker gates。

Stage74 不能使用 test 做评估或 tuning。

## Stage74：停止当前 candidate-reranker policy 路线

时间：2026-07-15

### 目标

根据 Stage71-Stage73 的 train/dev 证据，正式停止当前 candidate-reranker policy 路线。

这一步是 decision checkpoint，不是新的实验：

- 不读取 frozen test split；
- 不运行 final test metrics；
- 不使用 test 做 tuning；
- 不改变默认 runtime policy；
- 不继续在同一 reranker-policy 路线上调参。

### 使用的真实证据

Stage71 的 train-only CV 显示 reranker model 本身在 candidate-level 有提升：

```text
ridge_candidate_token_f1:
  train-CV selected F1: 0.2652
  train-CV delta: +0.0383

logistic_best_candidate:
  train-CV selected F1: 0.2523
  train-CV delta: +0.0254
```

但是 Stage71/72 的 dev top3 answer proxy 只有极小收益：

```text
logistic_best_candidate / candidate_score_gte_60:
  dev top3 delta: +0.0004
  regressions: 0
  gold citation delta: +0
```

Stage73 的 top10 diagnostic 进一步削弱了这条路线：

```text
train-only CV top10:
  selected policy deltas: +0.0000
  regressions: 0

dev holdout top10:
  selected policy deltas: +0.0000
  regressions: 0
```

### 本次新增

- 新增 `docs/primeqa_hybrid_candidate_reranker_stop_decision.md`。
- 更新 `docs/evaluation_strategy.md`。
- 更新 `docs/data_strategy.md`。
- 更新 `docs/primeqa_hybrid_candidate_reranker_top10_diagnostic.md`。
- 更新 `docs/learning_journal.md`。

### 决策

```text
status: candidate_reranker_policy_route_stopped_as_non_actionable
default_runtime_policy: unchanged
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
current_reranker_policy_defaultization: blocked
```

### 问题、原因与修正

- 问题 1：candidate-level CV 有提升，但 answer-level proxy 几乎没有收益。
  - 原因：当前 reranker policy 主要改变候选排序，不能稳定转化为组合答案 F1 改进。
  - 修正：停止当前 reranker-policy 路线，不继续在同一路径上做小幅调参。
- 问题 2：Stage72 top3 的 `+0.0004` 容易被误读为可以进入 final gate。
  - 原因：只看 top3 会放大 leading-candidate 表面的微弱信号。
  - 修正：把 Stage73 top10 的 `+0.0000` 纳入 stop decision，明确 blocked defaultization。
- 问题 3：如果不记录停止决策，后续容易再次回到这条路线。
  - 原因：Stage71-73 都留下了可继续 train/dev 的技术入口。
  - 修正：新增 Stage74 stop-decision 文档，并在总策略页把当前 reranker-policy route 标记为 stopped。

### 验证

本阶段是 docs-only decision checkpoint。验证在提交前执行：

```text
ruff check .
pytest -q
git diff --check
```

### 结论

- Stage74 已停止当前 candidate-reranker policy 路线。
- 当前 reranker policy 不 defaultize。
- 当前 reranker policy 不进入 final-test gate。
- 测试集继续完全 locked，不能用于训练阶段评估或 tuning。
- 默认 runtime policy 保持 unchanged。

### 我学到的

- candidate-level reranker 指标不能直接等价于 answer-level RAG 改进。
- 当 top3 的微弱收益在 top10 消失时，应优先停止路线，而不是继续局部调参。
- “停止”也需要作为正式阶段记录，否则后续很容易把已否定路线重新当成待推进事项。

### 下一步

Stage75 只有在用户明确确认新的方向后再开始。

当前不自动开启新的实验路线；如果继续，必须先确认新的非 reranker-policy train/dev 方向，并继续遵守：训练阶段永远不能在测试集上评估。

## Stage75：BM25 top10 没召回分析

时间：2026-07-15

### 目标

根据用户确认的方向，优先解决“没有召回”的问题，而不是继续当前 reranker-policy 路线。

本阶段只分析 `primeqa_hybrid_stage68_v1` 的 train/dev：

- 不读取 frozen test split；
- 不运行 final test metrics；
- 不用 test 做 tuning；
- 不改变默认 runtime policy；
- 不写入原始 question、answer、document title 或 document body text。

### 本次新增

- 新增 `src/ts_rag_agent/application/primeqa_hybrid_bm25_miss_analysis.py`。
- 新增 `scripts/analyze_primeqa_hybrid_bm25_top10_misses.py`。
- 新增 `tests/test_primeqa_hybrid_bm25_miss_analysis.py`。
- 新增 `docs/primeqa_hybrid_bm25_top10_miss_analysis.md`。
- 更新 `docs/evaluation_strategy.md`。
- 更新 `docs/data_strategy.md`。
- 更新 `docs/primeqa_hybrid_candidate_reranker_stop_decision.md`。
- 更新 `docs/learning_journal.md`。

### 真实运行命令

```text
python scripts\analyze_primeqa_hybrid_bm25_top10_misses.py `
  --output artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json `
  --visualization-dir artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals `
  --top-k 10 `
  --search-depth 50
```

真实运行耗时：

```text
load_splits: 0.012s
load_documents: 0.865s
scan_candidate_routes: 0.053s
bm25_index: 4.624s
miss_analysis: 114.647s
total: 120.201s
```

### 真实结果

```text
train:
  evaluated_questions: 370
  hit@10: 0.6622
  miss_count: 125
  miss_rate: 0.3378

dev:
  evaluated_questions: 76
  hit@10: 0.6974
  miss_count: 23
  miss_rate: 0.3026

cross split:
  evaluated_questions: 446
  hit_count: 298
  miss_count: 148
  hit@10: 0.6682
  miss_rate: 0.3318
```

miss driver：

```text
gold_doc_not_found_within_top50: 110
gold_doc_rank_21_to_50: 24
gold_doc_rank_11_to_20: 14
top1_query_overlap_exceeds_gold: 103
gold_doc_query_overlap_ratio_lt_0_25: 20
gold_doc_low_query_overlap_lte_1: 1
top10_contains_source_candidate_doc: 148
```

dev miss：

```text
dev miss_count: 23
dev not_found_top50: 17
dev rank_21_to_50: 4
dev rank_11_to_20: 2
dev top1_query_overlap_exceeds_gold: 14
dev gold_doc_query_overlap_ratio_lt_0_25: 3
```

### 可视化产物

```text
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_count_by_split.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_rate_by_split.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_reason_tags.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_rank_buckets.svg
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_dev_miss_routes.svg
```

Stage75 JSON SHA256：

```text
45272301A177A275C3489B176ABB5F307CD3E809151E31760C36F6168E38A116
```

### Guard checks

```text
analysis_splits_are_train_dev_only: passed
candidate_artifact_splits_are_train_dev_only: passed
candidate_rows_have_no_test_split: passed
search_depth_covers_top_k: passed
stage75_report_is_public_safe_no_raw_text: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

额外 raw-text 检查：

```text
Select-String over Stage75 JSON for question_title, question_text, gold_answer,
candidate_sentence, document_title, document_text, and known raw-text snippets:
no matches
```

artifact 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75_visuals\stage75_bm25_miss_reason_tags.svg
```

### 问题、原因与修正

- 问题 1：继续 reranker 不能解决未召回问题。
  - 原因：110 / 148 个 train/dev miss 的 gold document 不在 BM25 top50 内；gold document 没进候选窗口时，reranker 没有可重排对象。
  - 修正：Stage75 转向 BM25 top10 miss analysis，并把 Stage76 定义为 retrieval-recall 改进设计。
- 问题 2：旧的 BM25 error analyzer 不适合直接用于当前记录。
  - 原因：旧脚本会输出 raw question/answer 文本，不符合当前 public-safe 记录边界。
  - 修正：新增 Stage75 专用 analyzer，只输出 ID、rank bucket、route、overlap features、reason tags 和统计汇总。
- 问题 3：第一版实现有小的代码质量问题。
  - 原因：`Sequence` import 缺失，且 ruff 要求从 `collections.abc` 引入。
  - 修正：补充并调整 import；同时修正 guard check 中 train/dev 顺序比较，统一用排序后的 split 列表。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_bm25_miss_analysis.py scripts\analyze_primeqa_hybrid_bm25_top10_misses.py tests\test_primeqa_hybrid_bm25_miss_analysis.py
pytest -q tests\test_primeqa_hybrid_bm25_miss_analysis.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

提交前完整验证：

```text
ruff check .
pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 199 passed
git diff --check: passed
```

### 结论

- Stage75 已完成 BM25 top10 miss analysis。
- 训练阶段没有使用测试集。
- final test metrics 没有运行。
- 默认 runtime policy 保持 unchanged。
- 当前最关键问题是 retrieval recall：dev 仍有 `23 / 76` answerable questions 未在 BM25 top10 召回 gold document，其中 `17 / 23` 连 top50 都没有找到。
- 因为大多数 miss 不是 shallow rerank 问题，下一步应先设计 train/dev-only 的召回改进候选，而不是继续 reranker-policy 路线。

### 我学到的

- 对 RAG 来说，召回失败是比 reranker 排序更靠前的瓶颈；gold document 不进候选窗口，后续 composition 和 reranker 都无法弥补。
- top10 miss 里要区分 `rank_11_to_50` 和 `not_found_top50`：前者可以先看重排或轻量 retrieval feature，后者更像 query/document matching 问题。
- 记录分析类实验时，public-safe 边界要在脚本层面保证，不能只靠文档里承诺不展示 raw text。

### 下一步

Stage76：基于 Stage75 miss drivers，设计 train/dev-only 的 retrieval-recall improvement candidates。

Stage76 仍然不能使用 test 做评估或 tuning，也不能运行 final test metrics。

## Stage76：设计 retrieval-recall 改进候选

时间：2026-07-15

### 目标

承接 Stage75 的结论：当前关键问题不是 reranker，而是 BM25 top10 没召回。

本阶段把 Stage75 的 miss drivers 转成可执行的 retrieval-recall candidate design，给下一阶段直接跑 train/dev 实验。

边界：

- 只读取 Stage75 的 public-safe JSON 报告；
- 不读取 frozen test split；
- 不运行 final test metrics；
- 不用 test 做 tuning；
- 不改变默认 runtime policy；
- 不把 source `DOC_IDS` 作为可用 runtime retrieval evidence。

### 本次新增

- 新增 `src/ts_rag_agent/application/primeqa_hybrid_retrieval_recall_candidate_design.py`。
- 新增 `scripts/design_primeqa_hybrid_retrieval_recall_candidates.py`。
- 新增 `tests/test_primeqa_hybrid_retrieval_recall_candidate_design.py`。
- 新增 `docs/primeqa_hybrid_retrieval_recall_candidate_design.md`。
- 更新 `docs/evaluation_strategy.md`。
- 更新 `docs/data_strategy.md`。
- 更新 `docs/primeqa_hybrid_bm25_top10_miss_analysis.md`。
- 更新 `docs/learning_journal.md`。

### 真实运行命令

```text
python scripts\design_primeqa_hybrid_retrieval_recall_candidates.py `
  --output artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76.json `
  --visualization-dir artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals
```

### 输入

```text
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
sha256: 45272301a177a275c3489b176abb5f307cd3e809151e31760c36f6168e38a116
```

### 真实结果

Stage76 推荐的执行顺序：

```text
1. query_view_ablation_full_title_dedup
2. fielded_title_text_bm25_score_fusion
3. section_bm25_doc_rollup_train_dev_probe
4. dense_sparse_rrf_train_dev_probe
5. bm25_k1_b_grid_train_to_dev
```

候选覆盖情况：

```text
query_view_ablation_full_title_dedup:
  priority_score: 196
  target_miss_count: 143
  dev_targets: 22

fielded_title_text_bm25_score_fusion:
  priority_score: 195
  target_miss_count: 143
  dev_targets: 23

section_bm25_doc_rollup_train_dev_probe:
  priority_score: 163
  target_miss_count: 119
  dev_targets: 17

dense_sparse_rrf_train_dev_probe:
  priority_score: 134
  target_miss_count: 111
  dev_targets: 17

bm25_k1_b_grid_train_to_dev:
  priority_score: 69
  target_miss_count: 38
  dev_targets: 6

source_doc_ids_oracle_union_blocked:
  priority_score: 0
  target_miss_count: 148
  dev_targets: 23
  status: blocked_from_train_dev_experiment
```

### 可视化产物

```text
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_priority_scores.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_target_misses.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_dev_targets.svg
artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_allowed_vs_blocked_candidates.svg
```

Stage76 JSON SHA256：

```text
DDEBD276B3C70B33A0B8445622BBD0B5BA9C46C1B79DE08068B8AC7326692BFF
```

### Guard checks

```text
source_report_is_stage75: passed
source_development_splits_are_train_dev_only: passed
source_forbidden_final_splits_include_test: passed
source_candidate_rows_have_no_test_split: passed
source_final_test_metrics_not_run: passed
source_default_runtime_policy_unchanged: passed
stage76_design_only_no_runtime_default_change: passed
stage76_uses_public_safe_stage75_report_only: passed
```

额外 raw-text 检查：

```text
Select-String over Stage76 JSON for question_title, question_text, gold_answer,
candidate_sentence, document_title, document_text, and known raw-text snippets:
no matches
```

artifact 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_retrieval_recall_candidate_design_stage76_visuals\stage76_candidate_priority_scores.svg
```

### 问题、原因与修正

- 问题 1：`source DOC_IDS` 看起来可以覆盖全部 miss，但不能作为 retrieval 方案。
  - 原因：这些 ID 是数据集源 metadata，不是实际用户查询时可见的 runtime signal。
  - 修正：Stage76 把 `source_doc_ids_oracle_union_blocked` 明确标成 blocked diagnostic，不允许进入 train/dev experiment 或 runtime defaultization。
- 问题 2：最初测试断言把首位推荐候选写死。
  - 原因：小 fixture 中低风险 BM25 grid 会因为优先级公式排到第一，测试不应依赖这个局部排序。
  - 修正：改成验证关键候选存在、blocked 约束存在、目标计数正确。
- 问题 3：现有 `SectionBM25Retriever` 可用，但不是最高优先级。
  - 原因：Stage76 的 priority score 结合了 target miss count、dev target count、readiness 和 risk。query view / fielded BM25 覆盖了更多 Stage75 miss case。
  - 修正：把 section BM25 保留为第三优先候选，并在文档中说明下一步先跑最高优先 allowed candidate。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_recall_candidate_design.py scripts\design_primeqa_hybrid_retrieval_recall_candidates.py tests\test_primeqa_hybrid_retrieval_recall_candidate_design.py
pytest -q tests\test_primeqa_hybrid_retrieval_recall_candidate_design.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

提交前完整验证：

```text
ruff check .
pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 201 passed
git diff --check: passed
```

### 结论

- Stage76 已完成 retrieval-recall candidate design。
- Stage76 没有读取测试集。
- Stage76 没有运行 final test metrics。
- 默认 runtime policy 保持 unchanged。
- `source DOC_IDS` oracle union 已明确 blocked。
- 下一步不应该回到 reranker-policy 路线，而应该先跑 query view ablation 的 train/dev 实验。

### 我学到的

- “全部 miss 都在 source DOC_IDS 里”不等于可以用它补召回；如果 runtime 不可见，就只能作为诊断证据。
- 召回改进候选需要同时看覆盖面、dev 覆盖、实现就绪度和风险；不是只按某一个指标贪心排序。
- Stage75 的 public-safe 报告足够支撑 Stage76 设计，不需要再次读取原始问题、答案或文档正文。

### 下一步

Stage77：运行 `query_view_ablation_full_title_dedup` 的 train/dev-only retrieval-recall 实验。

Stage77 仍然不能使用 test 做评估或 tuning，也不能运行 final test metrics。

## Stage77：query view ablation 召回实验

时间：2026-07-15

### 目标

运行 Stage76 推荐优先级最高的 retrieval-recall candidate：
`query_view_ablation_full_title_dedup`。

实验设计：

- 同一个 BM25 document index；
- 同一组 frozen Stage68 train/dev split；
- 对比不同 query view；
- train 上选择 challenger，dev 只做验证；
- 不读取 test；
- 不运行 final test metrics；
- 不用 source `DOC_IDS` 作为 runtime retrieval evidence；
- 不改变默认 runtime policy。

### 本次新增

- 新增 `src/ts_rag_agent/application/primeqa_hybrid_query_view_ablation.py`。
- 新增 `scripts/run_primeqa_hybrid_query_view_ablation.py`。
- 新增 `tests/test_primeqa_hybrid_query_view_ablation.py`。
- 新增 `docs/primeqa_hybrid_query_view_ablation.md`。
- 更新 `docs/evaluation_strategy.md`。
- 更新 `docs/data_strategy.md`。
- 更新 `docs/primeqa_hybrid_retrieval_recall_candidate_design.md`。
- 更新 `docs/learning_journal.md`。

### 真实运行命令

```text
python scripts\run_primeqa_hybrid_query_view_ablation.py `
  --output artifacts\primeqa_hybrid_query_view_ablation_stage77.json `
  --visualization-dir artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals
```

真实运行耗时：

```text
load_splits: 0.025s
load_documents_and_stage75: 0.930s
bm25_index: 4.520s
query_view_evaluate: 212.609s
total: 218.083s
```

### Query views

```text
full_question_baseline:
  current retrieval query: question title plus question text

title_only:
  question title only

full_question_dedup_terms:
  tokenized full question with duplicate lexical terms removed while preserving
  first occurrence order
```

### 真实结果

Train：

```text
full_question_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR: 0.5023

title_only:
  hit@1: 0.3865
  hit@5: 0.5486
  hit@10: 0.6054
  MRR: 0.4589
  hit@10 delta vs baseline: -0.0568

full_question_dedup_terms:
  hit@1: 0.4135
  hit@5: 0.5892
  hit@10: 0.6432
  MRR: 0.4908
  hit@10 delta vs baseline: -0.0190
```

Dev：

```text
full_question_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR: 0.5331

title_only:
  hit@1: 0.3947
  hit@5: 0.5789
  hit@10: 0.6184
  MRR: 0.4638
  hit@10 delta vs baseline: -0.0790
  top10 improvements/regressions: 1 / 7

full_question_dedup_terms:
  hit@1: 0.4342
  hit@5: 0.6053
  hit@10: 0.6579
  MRR: 0.5116
  hit@10 delta vs baseline: -0.0395
  top10 improvements/regressions: 1 / 4
```

Train selected challenger：

```text
selected_view_id: full_question_dedup_terms
selected_dev_hit@10_delta: -0.0395
selected_dev_top10_improvements: 1
selected_dev_top10_regressions: 4
```

### 可视化产物

```text
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_train_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_top10_changes.svg
```

Stage77 JSON SHA256：

```text
CD0DE3D3B864C06261D6147D2C34947A93CA671E2604B0ED10CD25CE7E9F03DC
```

### Guard checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
stage75_source_report_is_stage75: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

额外 raw-text 检查：

```text
Select-String over Stage77 JSON for question_title, question_text, gold_answer,
candidate_sentence, document_title, document_text, and known raw-text snippets:
no matches
```

artifact 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_query_view_ablation_stage77.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_query_view_ablation_stage77_visuals\stage77_query_view_dev_hit_at_10.svg
```

### 问题、原因与修正

- 问题 1：query-view challenger 没有超过 baseline。
  - 原因：无论 title-only 还是 full-question dedup，都会丢失一部分对 BM25 有帮助的词频或上下文信号。
  - 修正：不推进 query-view route，不进入 final-test gate，下一步转到 Stage76 第二候选 fielded title/text BM25 score fusion。
- 问题 2：train-selected challenger 在 dev 上也退化。
  - 原因：即便 `full_question_dedup_terms` 是两个 challenger 中 train 表现最好的，它仍低于 full-question baseline；dev hit@10 delta 为 `-0.0395`。
  - 修正：报告明确区分 baseline 与 challenger；选择 challenger 只用于验证候选，不意味着可 defaultize。
- 问题 3：query view 实验需要读取 train/dev 原始问题字段来构造查询。
  - 原因：检索本身必须使用问题文本作为 runtime query。
  - 修正：脚本允许读取 train/dev split 中的 query 字段，但报告和文档只输出 public-safe 的指标、样本 ID、rank 和聚合结果，不输出 raw question/answer/document text。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_query_view_ablation.py scripts\run_primeqa_hybrid_query_view_ablation.py tests\test_primeqa_hybrid_query_view_ablation.py
pytest -q tests\test_primeqa_hybrid_query_view_ablation.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

提交前完整验证：

```text
ruff check .
pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 203 passed
git diff --check: passed
```

### 结论

- Stage77 已完成 query-view ablation。
- Stage77 没有使用测试集。
- Stage77 没有运行 final test metrics。
- 默认 runtime policy 保持 unchanged。
- query-view route 不推进：两个 challenger 都低于 full-question baseline。
- 当前最合适下一步是进入 Stage76 第二候选 `fielded_title_text_bm25_score_fusion`。

### 我学到的

- 简单缩短 query 或去重 query terms 不一定能改善 BM25 召回；BM25 可能正在利用问题正文中的上下文词和词频信号。
- 对召回实验，必须同时看 improvements 和 regressions；`full_question_dedup_terms` 在 dev 上只新增 1 个 top10 hit，却丢掉 4 个原本命中的样本。
- train/dev-only 的候选验证也需要明确“不推进”的结论，否则容易把一个退化候选误认为只是还没调好。

### 下一步

Stage78：运行 `fielded_title_text_bm25_score_fusion` 的 train/dev-only retrieval-recall 实验。

Stage78 仍然不能使用 test 做评估或 tuning，也不能运行 final test metrics。

## Stage78：fielded title/text BM25 score fusion 召回实验

时间：2026-07-15

### 目标

继续解决 Stage75 暴露出来的 BM25 top10 召回不足问题。本阶段按照 Stage76 的第二个候选方向，评估 `fielded_title_text_bm25_score_fusion`：

- 仍然只使用 Stage68 冻结后的 train/dev split；
- 不读取 test split 做评估或调参；
- 不运行 final test metrics；
- 不使用 source `DOC_IDS` 作为 runtime 检索证据；
- 不改变默认 runtime policy；
- 输出可视化结果，方便对比每个 fielded BM25 配置。

### 我做了什么

新增 Stage78 train/dev-only 实验实现：

```text
src\ts_rag_agent\application\primeqa_hybrid_fielded_bm25_fusion.py
scripts\run_primeqa_hybrid_fielded_bm25_fusion.py
tests\test_primeqa_hybrid_fielded_bm25_fusion.py
docs\primeqa_hybrid_fielded_bm25_fusion.md
```

实验同时建立三个 BM25 index：

```text
full_document_bm25_baseline:
  title + text

title_retriever:
  title-only field

text_retriever:
  text-only field
```

fielded fusion 的固定候选网格：

```text
fielded_title_0_25_text_1_00
fielded_title_0_50_text_1_00
fielded_title_1_00_text_1_00
fielded_title_1_50_text_1_00
fielded_title_2_00_text_1_00
```

打分方式：

```text
normalized_title_score * title_weight + normalized_text_score * text_weight
```

选择规则：

```text
只在 train 上按 hit@10、hit@5、hit@1、MRR、config_id 选择 challenger；
dev 只用于验证 train-selected challenger；
test 不参与。
```

真实运行命令：

```text
python scripts\run_primeqa_hybrid_fielded_bm25_fusion.py `
  --output artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78.json `
  --visualization-dir artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals `
  --candidate-depth 100
```

运行耗时：

```text
load_splits: 0.026s
load_documents_and_reports: 0.854s
bm25_indexes: 9.690s
fielded_fusion_evaluate: 233.145s
guard_checks: 0.001s
total: 243.716s
```

### 真实结果

Train baseline：

```text
hit@1: 0.4243
hit@5: 0.6054
hit@10: 0.6622
MRR: 0.5023
miss_count_at_10: 125
```

Train challengers：

```text
fielded_title_0_25_text_1_00:
  hit@10: 0.6378
  hit@10 delta: -0.0244
  top10 improvements/regressions: 4 / 13

fielded_title_0_50_text_1_00:
  hit@10: 0.6135
  hit@10 delta: -0.0487
  top10 improvements/regressions: 9 / 27

fielded_title_1_00_text_1_00:
  hit@10: 0.5730
  hit@10 delta: -0.0892

fielded_title_1_50_text_1_00:
  hit@10: 0.5432
  hit@10 delta: -0.1190

fielded_title_2_00_text_1_00:
  hit@10: 0.5405
  hit@10 delta: -0.1217
```

Dev baseline：

```text
hit@1: 0.4342
hit@5: 0.6579
hit@10: 0.6974
MRR: 0.5331
miss_count_at_10: 23
```

Dev challengers：

```text
fielded_title_0_25_text_1_00:
  hit@1: 0.4737
  hit@5: 0.6579
  hit@10: 0.6974
  MRR: 0.5550
  hit@10 delta: +0.0000
  hit@1 delta: +0.0395
  MRR delta: +0.0219
  top10 improvements/regressions: 1 / 1

fielded_title_0_50_text_1_00:
  hit@10: 0.6842
  hit@10 delta: -0.0132
  top10 improvements/regressions: 2 / 3

fielded_title_1_00_text_1_00:
  hit@10: 0.6316
  hit@10 delta: -0.0658
  top10 improvements/regressions: 2 / 7

fielded_title_1_50_text_1_00:
  hit@10: 0.6053
  hit@10 delta: -0.0921
  top10 improvements/regressions: 2 / 9

fielded_title_2_00_text_1_00:
  hit@10: 0.6053
  hit@10 delta: -0.0921
  top10 improvements/regressions: 3 / 10
```

Train-selected challenger：

```text
selected_config_id: fielded_title_0_25_text_1_00
selected_dev_hit@10_delta: +0.0000
selected_dev_top10_improvements: 1
selected_dev_top10_regressions: 1
```

### 可视化产物

```text
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_top10_changes.svg
```

Stage78 JSON SHA256：

```text
6EE3DAE47AA3927AA4CF613A0DDC1C0CD5CDA6D4C62FE21C615AC38883CF6FAD
```

### Guard checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
candidate_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage77_source_report_is_stage77: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
stage77_did_not_open_final_test_gate: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

额外 raw-text 检查：

```text
Select-String over Stage78 JSON for question_title, question_text, gold_answer,
candidate_sentence, document_title, document_text, and known raw-text snippets:
no matches
```

artifact 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_fielded_bm25_fusion_stage78_visuals\stage78_fielded_bm25_dev_hit_at_10.svg
```

### 问题、原因与修正

- 问题 1：fielded BM25 fusion 没有提升 top10 召回。
  - 原因：轻 title 权重在 dev 上提升了 top1/MRR，但没有增加 hit@10；更高 title 权重反而明显拉低 train/dev hit@10。
  - 修正：不推进 fielded BM25 fusion route，不进入 final-test gate，不改变 runtime 默认策略。
- 问题 2：train-selected challenger 虽然是 challenger 中最优，但低于 baseline。
  - 原因：选择规则只在 challenger 之间做排序，`fielded_title_0_25_text_1_00` 只是 challenger 里损失最小，不代表超过 baseline。
  - 修正：报告和文档同时记录 baseline 对比 delta，防止把“候选最优”误读成“优于当前系统”。
- 问题 3：title/text field 拆分容易强化标题短词信号。
  - 原因：标题字段短且高密度，归一化后轻微权重就会改变排序；当标题词不足以定位答案文档时，会带来 top10 regression。
  - 修正：下一步不继续加大 title 权重，而转向 section-level BM25 doc rollup，测试更细粒度文本片段能否让答案文档进入候选窗口。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_fielded_bm25_fusion.py scripts\run_primeqa_hybrid_fielded_bm25_fusion.py tests\test_primeqa_hybrid_fielded_bm25_fusion.py
pytest -q tests\test_primeqa_hybrid_fielded_bm25_fusion.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

提交前完整验证：

```text
ruff check .
pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 205 passed
git diff --check: passed
```

### 结论

- Stage78 已完成 fielded title/text BM25 score fusion 实验。
- Stage78 没有使用测试集。
- Stage78 没有运行 final test metrics。
- Stage78 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- 默认 runtime policy 保持 unchanged。
- fielded BM25 fusion route 不推进：train-selected challenger 在 dev 上 hit@10 delta 为 `+0.0000`，top10 improvements/regressions 为 `1 / 1`。
- 当前最合适下一步是进入 Stage76 第三候选 `section_bm25_doc_rollup_train_dev_probe`。

### 我学到的

- 对召回问题，top1/MRR 的提升不能替代 hit@10 的提升；当前瓶颈是 gold document 是否进入候选窗口。
- fielded score fusion 必须与 baseline 做显式 delta 对比；否则 challenger 内部的“最优配置”容易被误解。
- title 信号有价值，但直接加权 title-only BM25 并不稳定；更合理的下一步是看 section-level 检索能不能把长文档里的相关片段召回出来。

### 下一步

Stage79：运行 `section_bm25_doc_rollup_train_dev_probe` 的 train/dev-only retrieval-recall 实验。

Stage79 仍然不能使用 test 做评估或 tuning，不能运行 final test metrics，也不能使用 source `DOC_IDS` 作为 runtime 检索证据。

## Stage79：section BM25 doc rollup 召回实验

时间：2026-07-15

### 目标

继续解决 Stage75 暴露的 BM25 top10 召回不足问题。本阶段按照 Stage76 的第三个候选方向，评估 `section_bm25_doc_rollup_train_dev_probe`：

- 仍然只使用 Stage68 冻结后的 train/dev split；
- 不读取 test split 做评估或调参；
- 不运行 final test metrics；
- 不使用 source `DOC_IDS` 作为 runtime 检索证据；
- 不下载或静默选择新的 dense retrieval 依赖；
- 不改变默认 runtime policy；
- 输出可视化结果，比较 full-document BM25 baseline 与 section BM25 max-section rollup。

### 我做了什么

新增 Stage79 train/dev-only 实验实现：

```text
src\ts_rag_agent\application\primeqa_hybrid_section_bm25_doc_rollup.py
scripts\run_primeqa_hybrid_section_bm25_doc_rollup.py
tests\test_primeqa_hybrid_section_bm25_doc_rollup.py
docs\primeqa_hybrid_section_bm25_doc_rollup.md
```

本阶段使用项目已有的 `SectionBM25Retriever` 语义：

```text
section_bm25_max_section_rollup:
  先对 section 建 BM25 index；
  对每个 query 检索 section；
  父文档分数取该文档最高 section score；
  返回 parent document ranking。
```

对比对象：

```text
full_document_bm25_baseline:
  title + full document text 的 BM25 baseline
```

真实运行命令：

```text
python scripts\run_primeqa_hybrid_section_bm25_doc_rollup.py `
  --output artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79.json `
  --visualization-dir artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals `
  --search-depth 50
```

运行耗时：

```text
load_splits: 0.011s
load_documents_sections_and_reports: 2.353s
bm25_indexes: 9.768s
section_rollup_evaluate: 446.859s
guard_checks: 0.004s
total: 458.995s
```

加载的 section index：

```text
document_count: 28482
section_count: 216648
documents_with_sections: 28482
documents_without_sections: 0
average_sections_per_document: 7.6065
```

### 真实结果

Train baseline：

```text
hit@1: 0.4243
hit@5: 0.6054
hit@10: 0.6622
MRR@10: 0.5023
not_found@50: 93
```

Train section BM25 max-section rollup：

```text
hit@1: 0.4054
hit@5: 0.5324
hit@10: 0.5919
MRR@10: 0.4631
not_found@50: 94
```

Train delta：

```text
hit@1 delta: -0.0189
hit@5 delta: -0.0730
hit@10 delta: -0.0703
MRR@10 delta: -0.0392
not_found@50 delta: +1
top10 improvements/regressions: 8 / 34
search-depth improvements/regressions: 17 / 18
```

Dev baseline：

```text
hit@1: 0.4342
hit@5: 0.6579
hit@10: 0.6974
MRR@10: 0.5331
not_found@50: 17
```

Dev section BM25 max-section rollup：

```text
hit@1: 0.4342
hit@5: 0.6184
hit@10: 0.6447
MRR@10: 0.4974
not_found@50: 18
```

Dev delta：

```text
hit@1 delta: +0.0000
hit@5 delta: -0.0395
hit@10 delta: -0.0527
MRR@10 delta: -0.0357
not_found@50 delta: +1
top10 improvements/regressions: 1 / 5
search-depth improvements/regressions: 2 / 3
```

### 可视化产物

```text
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_not_found_at_50.svg
artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_top10_changes.svg
```

Stage79 JSON SHA256：

```text
32A6B0CF36E041E9C67AE430569AD4B4587EC07210C93D9654C8A4305406E1E7
```

### Guard checks

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
stage75_source_report_is_stage75: passed
stage78_source_report_is_stage78: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
stage78_did_not_open_final_test_gate: passed
section_index_has_nonempty_sections: passed
source_doc_ids_not_used_as_runtime_evidence: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

额外 raw-text 检查：

```text
Select-String over Stage79 JSON for question_title, question_text, gold_answer,
candidate_sentence, document_title, document_text, and known raw-text snippets:
no matches
```

artifact 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_section_bm25_doc_rollup_stage79_visuals\stage79_section_bm25_dev_hit_at_10.svg
```

### 问题、原因与修正

- 问题 1：section BM25 max-section rollup 没有提升 top10 召回，反而回退。
  - 原因：max-section rollup 会强化局部片段匹配，但也会丢失 full-document BM25 中对父文档整体上下文有帮助的信号。
  - 修正：不推进 section BM25 doc-rollup route，不进入 final-test gate，不改变 runtime 默认策略。
- 问题 2：not_found@50 没有下降。
  - 原因：section 级索引确实找回了一些 baseline top50 外的样本，但也丢掉了一些 baseline 已在 top50 内的样本；dev search-depth improvements/regressions 为 `2 / 3`。
  - 修正：报告同时记录 top10 changes 与 search-depth changes，避免只看单个改善案例。
- 问题 3：下一个 Stage76 候选 `dense_sparse_rrf_train_dev_probe` 涉及模型/cache/依赖选择。
  - 原因：Stage76 已把 dense+sparse 标为 high risk，并要求确认本地模型/cache 可用性。
  - 修正：下一步先做 feasibility check，记录本地模型/cache identity；不静默下载模型，不静默选择外部 dense retrieval 依赖。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_bm25_doc_rollup.py scripts\run_primeqa_hybrid_section_bm25_doc_rollup.py tests\test_primeqa_hybrid_section_bm25_doc_rollup.py
pytest -q tests\test_primeqa_hybrid_section_bm25_doc_rollup.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

提交前完整验证：

```text
ruff check .
pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 207 passed
git diff --check: passed
```

### 结论

- Stage79 已完成 section BM25 max-section document rollup 实验。
- Stage79 没有使用测试集。
- Stage79 没有运行 final test metrics。
- Stage79 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage79 没有下载或静默选择 dense retrieval 依赖。
- 默认 runtime policy 保持 unchanged。
- section BM25 doc-rollup route 不推进：dev hit@10 delta 为 `-0.0527`，top10 improvements/regressions 为 `1 / 5`，not_found@50 delta 为 `+1`。
- 当前最合适下一步是 Stage80：先检查 `dense_sparse_rrf_train_dev_probe` 的本地模型/cache 可行性。

### 我学到的

- 更细粒度的 section 检索不必然提升文档级召回；max-section 聚合可能牺牲父文档整体上下文。
- 对召回实验，除了 top10 improvements/regressions，还必须看 search-depth improvements/regressions；否则会误判“找回了几个 top50 外样本”这一局部信号。
- 涉及 dense retrieval 的下一步必须先确认模型和缓存边界，不能为了推进实验而静默下载或选型。

### 下一步

Stage80：检查 `dense_sparse_rrf_train_dev_probe` 的本地模型/cache 可行性，并记录可用模型、缓存路径、是否需要用户确认。

Stage80 仍然不能使用 test 做评估或 tuning，不能运行 final test metrics，不能使用 source `DOC_IDS` 作为 runtime 检索证据，也不能静默下载或选择新的 dense retrieval 依赖。

## Stage80：dense+sparse RRF 本地可行性检查

时间：2026-07-15

### 目标

进入 Stage76 第四个候选 `dense_sparse_rrf_train_dev_probe` 前，先确认本地条件是否允许无下载、无静默选型地运行 dense+sparse RRF。

本阶段只做 feasibility check：

- 不运行 train/dev retrieval metrics；
- 不读取 test split；
- 不运行 final test metrics；
- 不使用 source `DOC_IDS` 作为 runtime 检索证据；
- 不下载模型；
- 不静默选择 dense retrieval 模型或依赖；
- 记录本地 package、dense cache、Hugging Face model cache 和下一步需要确认的选项。

### 我做了什么

新增 Stage80 feasibility 实现：

```text
src\ts_rag_agent\application\primeqa_hybrid_dense_sparse_rrf_feasibility.py
scripts\check_primeqa_hybrid_dense_sparse_rrf_feasibility.py
tests\test_primeqa_hybrid_dense_sparse_rrf_feasibility.py
docs\primeqa_hybrid_dense_sparse_rrf_feasibility.md
```

真实运行命令：

```text
python scripts\check_primeqa_hybrid_dense_sparse_rrf_feasibility.py `
  --output artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json `
  --visualization-dir artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals
```

运行耗时：

```text
load_reports: 0.001s
load_documents: 0.875s
inspect_local_environment: 0.338s
guard_checks: 0.000s
total: 1.214s
```

### 真实发现

本地依赖：

```text
numpy: available, 2.2.6
sentence-transformers: available, 5.6.0
transformers: available, 5.13.1
torch: available, 2.13.0
scikit-learn: available, 1.7.2
scipy: available, 1.15.3
huggingface-hub: available, 1.23.0
faiss-cpu: not available, but not required for the existing NumPy RRF path
```

已有代码路径：

```text
src\ts_rag_agent\infrastructure\dense_retriever.py
src\ts_rag_agent\infrastructure\dense_embedding_cache.py
src\ts_rag_agent\infrastructure\hybrid_retriever.py
scripts\evaluate_hybrid.py
```

兼容当前 corpus 的本地 dense cache：

```text
model: intfloat/e5-small-v2
cache: data\indexes\dense\intfloat__e5-small-v2_512_passage.npz
cache sha256: 91edc7db00363769d6ae4e6a4b1b5a9c23e8682f2d8fefff42f98d0015fb9d63
document_text_max_chars: 512
document_prefix: "passage: "
embedding_shape: 28482 x 384
document_ids_match_current_corpus: true
can_run_without_reencoding_documents: true
can_run_without_model_download: true
huggingface snapshot: ffb93f3bd4047442299a41ebb6fa998a38507c52
```

```text
model: sentence-transformers/all-MiniLM-L6-v2
cache: data\indexes\dense\sentence-transformers__all-MiniLM-L6-v2_1600.npz
cache sha256: 5f2d2ad64ecd5902e6859f0932f69d05c54309dd7216598b708d1fab52975008
document_text_max_chars: 1600
document_prefix: ""
embedding_shape: 28482 x 384
document_ids_match_current_corpus: true
can_run_without_reencoding_documents: true
can_run_without_model_download: true
huggingface snapshot: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

历史 dense/hybrid metric artifact：

```text
artifacts\dense_dev_metrics.json
artifacts\hybrid_dev_metrics.json
artifacts\dense_e5_small_v2_512_dev_metrics.json
artifacts\hybrid_e5_small_v2_512_dev_metrics.json
```

边界说明：这些旧 metric 只能证明本地 cache 和脚本曾经能运行，不能作为当前 Stage68 frozen split 的 train/dev 结论，更不能当 final-test evidence。

### Candidate options

Stage80 找到三个可选下一步。因为这是 dense 模型/cache 协议选择，必须先确认，不能静默进入 train/dev run。

推荐选项：

```text
compare_existing_cached_dense_models
```

含义：

```text
在 Stage81 用两个已存在的本地 cache 都跑 train/dev-only dense+sparse RRF；
只在 train 上选择；
dev 只做验证；
不下载模型；
不碰 test。
```

其他可选项：

```text
single_cached_model::intfloat/e5-small-v2
single_cached_model::sentence-transformers/all-MiniLM-L6-v2
```

### 可视化产物

```text
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_dense_cache_readiness.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_dependency_availability.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_candidate_options.svg
```

Stage80 JSON SHA256：

```text
2441BB1CB1E7888299D3F57962B18CD59DF84E2086AC281105ABCACFC144880F
```

### Guard checks

```text
stage76_source_report_is_stage76: passed
stage76_dense_sparse_candidate_is_allowed: passed
stage79_source_report_is_stage79: passed
stage79_did_not_open_final_test_gate: passed
required_cached_rrf_packages_available: passed
existing_dense_hybrid_code_available: passed
compatible_local_dense_cache_available: passed
no_model_download_attempted: passed
train_dev_metrics_not_run: passed
final_test_metrics_not_run: passed
source_doc_ids_not_used_as_runtime_evidence: passed
default_runtime_policy_unchanged: passed
```

额外 raw-text 检查：

```text
Exact raw-field Select-String over Stage80 JSON for question_title,
question_text, gold_answer, candidate_sentence, document_title, document_text,
and known raw text snippets:
no matches

Broader substring check matched document_text_max_chars. This is a config field,
not raw document text.
```

artifact 和 cache 忽略状态：

```text
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json
.gitignore:19:artifacts/* artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80_visuals\stage80_dense_cache_readiness.svg
.gitignore:18:data/indexes/* data\indexes\dense\intfloat__e5-small-v2_512_passage.npz
```

### 问题、原因与修正

- 问题 1：本地有两个可运行 dense cache，不能静默选一个。
  - 原因：模型/cache 选择会影响 Stage81 结论；Stage76 已把 dense+sparse route 标为 high risk。
  - 修正：Stage80 只报告可行性和选项，不运行 train/dev metrics；下一步需要确认模型/cache protocol。
- 问题 2：旧 dense/hybrid dev metric 容易被误认为当前证据。
  - 原因：旧 metric 来自 Stage68 split 之前的 dev 文件，不是当前 frozen split 的 train/dev 证据。
  - 修正：报告将这些 metric 标记为 `historical_pre_stage68_reference_only_not_current_split_evidence`。
- 问题 3：raw-text 检查中出现 `document_text_max_chars` 假阳性。
  - 原因：宽泛字符串 `document_text` 会匹配配置字段名。
  - 修正：使用精确 raw-field 正则复查，确认没有 raw `document_text` 字段或已知原文片段。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_dense_sparse_rrf_feasibility.py scripts\check_primeqa_hybrid_dense_sparse_rrf_feasibility.py tests\test_primeqa_hybrid_dense_sparse_rrf_feasibility.py
pytest -q tests\test_primeqa_hybrid_dense_sparse_rrf_feasibility.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

### 结论

- Stage80 已完成 dense+sparse RRF 本地可行性检查。
- Stage80 没有运行 train/dev retrieval metrics。
- Stage80 没有使用测试集。
- Stage80 没有运行 final test metrics。
- Stage80 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage80 没有下载模型。
- 默认 runtime policy 保持 unchanged。
- 本地可无下载运行 dense+sparse RRF：两个 dense cache 都与当前 corpus document IDs 匹配。
- 进入 Stage81 前需要确认 dense model/cache protocol。

### 我学到的

- feasibility check 和 retrieval metric run 必须分开；前者确认条件，后者才产出指标。
- 已有缓存不等于可以静默默认化模型；模型/cache 协议本身就是实验设计的一部分。
- 历史 metric 可以帮助判断成本和可运行性，但 split boundary 变化后不能当当前效果证据。

### 下一步

Stage81 需要先确认 dense model/cache protocol。

推荐选项：`compare_existing_cached_dense_models`。

该路线会在 Stage81 用两个已存在本地 cache 跑 train/dev-only dense+sparse RRF，对比后按 train 选择、dev 验证；仍然不碰 test、不跑 final metrics、不使用 source `DOC_IDS`、不下载模型。

## Stage81：dense+sparse RRF train/dev 对比
时间：2026-07-15

### 目标

按用户确认的 Stage80 A 路线运行：`compare_existing_cached_dense_models`。

本阶段只比较 Stage80 已确认的两个本地 dense cache：
- `intfloat/e5-small-v2`
- `sentence-transformers/all-MiniLM-L6-v2`

边界：
- 只跑 train/dev；
- 不读取 test split；
- 不运行 final test metrics；
- 不使用 source `DOC_IDS` 作为 runtime 检索证据；
- 不下载模型；
- 不改变 runtime 默认策略。

### 我做了什么

新增实现：

```text
src\ts_rag_agent\application\primeqa_hybrid_dense_sparse_rrf_comparison.py
scripts\run_primeqa_hybrid_dense_sparse_rrf_comparison.py
tests\test_primeqa_hybrid_dense_sparse_rrf_comparison.py
docs\primeqa_hybrid_dense_sparse_rrf_comparison.md
```

真实运行命令：

```text
$env:HF_HUB_OFFLINE='1'
python scripts\run_primeqa_hybrid_dense_sparse_rrf_comparison.py `
  --output artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json `
  --visualization-dir artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals
```

运行耗时：

```text
total: 385.456s
```

### 真实结果

train：

```text
baseline hit@10: 0.6622
e5-small-v2 RRF hit@10: 0.6703, delta +0.0081
all-MiniLM-L6-v2 RRF hit@10: 0.6973, delta +0.0351
```

dev：

```text
baseline hit@10: 0.6974
e5-small-v2 RRF hit@10: 0.6711, delta -0.0263
all-MiniLM-L6-v2 RRF hit@10: 0.6842, delta -0.0132
```

按 train-only 选择规则，Stage81 选中：

```text
dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

但该 train-selected challenger 在 dev 上没有通过：

```text
selected dev hit@10 delta: -0.0132
selected dev top10 improvements/regressions: 3 / 4
selected dev not_found@50 delta: -6
```

补充观察：
- dense+sparse RRF 明显降低 top50 not-found 数量；
- e5-small-v2 在 dev hit@1 和 MRR@10 上变好；
- 但主指标 hit@10 在 dev 上回退，所以不能推进 runtime/default 策略。

### Guard checks

全部通过：

```text
analysis_splits_are_train_dev_only
top_k_values_include_primary_top10
search_depth_covers_primary_top10
stage75_source_report_is_stage75
stage80_source_report_is_stage80
stage80_can_run_dense_sparse_rrf_without_download
stage80_requires_user_confirmation_before_train_dev_run
user_confirmed_protocol_matches_stage80_option
selected_cache_count_matches_compare_protocol
dense_cache_files_exist
dense_cache_metadata_matches_stage80
dense_cache_document_ids_match_current_corpus
dense_cache_embedding_rows_match_current_corpus
query_prefix_protocol_resolved_from_stage80
local_model_snapshots_exist
baseline_train_hit10_matches_stage75
baseline_dev_hit10_matches_stage75
no_model_download_attempted
source_doc_ids_not_used_as_runtime_evidence
final_test_metrics_not_run
default_runtime_policy_unchanged
```

### 可视化产物

```text
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_train_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_not_found_at_50.svg
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81_visuals\stage81_dense_sparse_rrf_dev_top10_changes.svg
```

Stage81 JSON SHA256：

```text
56AFBE70545C05780631750EE39B9DBC2B41B26BFD310D5997F60448FB3B4C03
```

### 问题、原因与修正

- 问题 1：不能静默决定 dense query prefix。
  - 原因：E5 模型通常需要 `query: ` 前缀，all-MiniLM 不需要；如果随意选择，会污染实验结论。
  - 修正：Stage81 从 Stage80 legacy dense metric 中解析 query prefix；解析不到就阻断，不兜底。
- 问题 2：dense+sparse RRF 在 train 上更好，但 dev 主指标回退。
  - 原因：dense 信号能补回一些 top50 召回，但 RRF 同时会把部分 BM25 已命中的 top10 文档推出 top10。
  - 修正：保留 train/dev 分离判断，不因 train 提升或 top50 not-found 下降而推进 runtime/default。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_dense_sparse_rrf_comparison.py scripts\run_primeqa_hybrid_dense_sparse_rrf_comparison.py tests\test_primeqa_hybrid_dense_sparse_rrf_comparison.py
pytest -q tests\test_primeqa_hybrid_dense_sparse_rrf_comparison.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

### 结论

- Stage81 已完成 dense+sparse RRF train/dev 对比。
- Stage81 没有使用 test。
- Stage81 没有运行 final test metrics。
- Stage81 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage81 没有下载模型。
- 默认 runtime policy 保持 unchanged。
- dense+sparse RRF route 不能推进：train-selected challenger 的 dev hit@10 delta 为 `-0.0132`。

### 我学到了

- top50 not-found 下降不等于 top10 主指标变好；召回更深的位置有改善，但排序融合仍可能损失可交付的 top10 命中。
- train-only selection 必须严格执行；即使 train hit@10 提升明显，只要 dev 未验证，就不能进入 runtime/default 策略。
- query-prefix 协议本身就是实验条件，必须记录来源，不能作为隐形实现细节。

### 下一步

Stage82：进入 Stage76 剩余候选 `bm25_k1_b_grid_train_to_dev`。

Stage82 仍然只能使用 train/dev，不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage82：BM25 k1/b 小网格 train/dev 实验
时间：2026-07-15

### 目标

运行 Stage76 剩余候选 `bm25_k1_b_grid_train_to_dev`。

用户确认选择 A 路线：小网格。

固定网格：

```text
k1: 1.2, 1.5, 1.8
b: 0.55, 0.75, 0.95
```

边界：
- 只跑 train/dev；
- 不读取 test split；
- 不运行 final test metrics；
- 不使用 source `DOC_IDS` 作为 runtime 检索证据；
- 不能用 dev 反选参数；
- 不改变 runtime 默认策略。

### 我做了什么

新增实现：

```text
src\ts_rag_agent\application\primeqa_hybrid_bm25_k1_b_grid.py
scripts\run_primeqa_hybrid_bm25_k1_b_grid.py
tests\test_primeqa_hybrid_bm25_k1_b_grid.py
docs\primeqa_hybrid_bm25_k1_b_grid.md
```

真实运行命令：

```text
python scripts\run_primeqa_hybrid_bm25_k1_b_grid.py `
  --output artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82.json `
  --visualization-dir artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals
```

最终成功运行耗时：

```text
total: 93.445s
```

### 真实结果

train-only selection 选中：

```text
full_document_bm25_baseline
```

train 关键指标：

```text
baseline hit@10: 0.6622
bm25_grid__k1_1_20__b_0_75 hit@10: 0.6622
```

`bm25_grid__k1_1_20__b_0_75` 与 baseline 的 train hit@10 持平，但 hit@5 低于 baseline：

```text
baseline train hit@5: 0.6054
bm25_grid__k1_1_20__b_0_75 train hit@5: 0.6027
```

所以按预设 train-only tie-breaker，baseline 胜出。

dev 上的事后观察：

```text
bm25_grid__k1_1_20__b_0_95 dev hit@10: 0.7105
bm25_grid__k1_1_50__b_0_95 dev hit@10: 0.7105
baseline dev hit@10: 0.6974
```

但这两个 `b=0.95` 配置没有在 train 上被选中，不能用 dev 反选，因此不能推进为 runtime/default 策略。

Stage82 决策：

```text
status: primeqa_hybrid_bm25_k1_b_grid_completed
selected_config_id: full_document_bm25_baseline
selected_dev_hit10_delta: 0.0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_rank_11_to_50_count_delta: 0
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Guard checks

全部通过：

```text
analysis_splits_are_train_dev_only
top_k_values_include_primary_top10
search_depth_covers_primary_top10
stage75_source_report_is_stage75
stage76_source_report_is_stage76
stage76_bm25_grid_candidate_is_allowed
stage76_requires_fixed_grid_values
stage81_source_report_is_stage81
stage81_did_not_open_final_test_gate
stage81_recommends_bm25_grid_next
user_confirmed_small_grid_protocol
grid_values_fixed_before_run
grid_includes_stage75_baseline
baseline_train_hit10_matches_stage75
baseline_dev_hit10_matches_stage75
source_doc_ids_not_used_as_runtime_evidence
final_test_metrics_not_run
default_runtime_policy_unchanged
```

### 可视化产物

```text
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_train_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_near_miss_11_to_50.svg
artifacts\primeqa_hybrid_bm25_k1_b_grid_stage82_visuals\stage82_bm25_grid_dev_top10_changes.svg
```

Stage82 JSON SHA256：

```text
EB9E0D5EC66401418E6254381A9638316EDE0A040156B0EF49C95CF6BDD786CA
```

### 问题、原因与修正

- 问题 1：第一版真实运行超时。
  - 原因：实现对 train/dev 和每个 grid 配置重复建 BM25 索引，真实 corpus 下成本过高。
  - 修正：改为 Stage82 专用共享 BM25 grid index；每个 split 只建一次倒排表，并在同一 query 上计算 9 个配置的 rank。
- 问题 2：第二次运行生成了 blocked 报告。
  - 原因：guard 只接受 Stage81 next-step 中的内部 candidate ID，但 Stage81 文档写的是人类可读的 `BM25 k1/b grid candidate`。
  - 修正：guard 同时接受内部 ID 和人类可读名称，然后覆盖重跑最终报告。
- 问题 3：dev 上存在更高 hit@10 的配置，容易被误认为可用。
  - 原因：`b=0.95` 的两个配置 dev hit@10 达到 `0.7105`，但不是 train-selected。
  - 修正：记录为事后观察，明确禁止用 dev 反选。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_bm25_k1_b_grid.py scripts\run_primeqa_hybrid_bm25_k1_b_grid.py tests\test_primeqa_hybrid_bm25_k1_b_grid.py
pytest -q tests\test_primeqa_hybrid_bm25_k1_b_grid.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

### 结论

- Stage82 已完成 BM25 k1/b 小网格 train/dev 实验。
- Stage82 没有使用 test。
- Stage82 没有运行 final test metrics。
- Stage82 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- 默认 runtime policy 保持 unchanged。
- BM25 参数网格 route 不能推进：train-only selection 选择了现有 baseline。

### 我学到了

- dev 上更好的参数不是可部署证据；只要它不是 train 选出来的，就只能作为后续研究线索。
- 小网格虽然低风险，但对当前 frozen train/dev 来说没有给出可验证的 runtime 参数替代。
- 实验 guard 既要严格，也要匹配前序记录的真实表达；否则会产生“实际可解释但报告被格式误伤”的 blocked artifact。

### 下一步

Stage83：汇总 Stage76 的 retrieval-recall 候选全部结果，判断下一条 train/dev-only 改进路线。

Stage83 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage83：retrieval-recall 候选耗尽汇总

### 目标

Stage83 的目标不是再跑一个新检索实验，而是把 Stage76 设计出的全部 retrieval-recall 候选做一次收口：

- 确认 Stage77 到 Stage82 是否已经覆盖全部 allowed candidates。
- 确认有没有任何候选能进入 runtime/default 策略。
- 明确 source `DOC_IDS` oracle union 仍然 blocked，不能作为 runtime 检索证据。
- 给出 Stage84 的 train/dev-only 下一路线选项。
- 继续锁住 frozen test split，不运行 final metrics。

### 本次改动

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_recall_exhaustion_summary.py
scripts/summarize_primeqa_hybrid_retrieval_recall_exhaustion.py
tests/test_primeqa_hybrid_retrieval_recall_exhaustion_summary.py
docs/primeqa_hybrid_retrieval_recall_exhaustion_summary.md
```

更新：

```text
docs/primeqa_hybrid_bm25_k1_b_grid.md
docs/primeqa_hybrid_retrieval_recall_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\summarize_primeqa_hybrid_retrieval_recall_exhaustion.py --output artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json --visualization-dir artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals
```

运行耗时：

```text
total: 0.002s
```

### 真实结果

Stage76 的 5 个 allowed retrieval-recall candidates 均已覆盖：

```text
query_view_ablation_full_title_dedup: Stage77, dev hit@10 delta -0.0395, not advanced
fielded_title_text_bm25_score_fusion: Stage78, dev hit@10 delta 0.0000, not advanced
section_bm25_doc_rollup_train_dev_probe: Stage79, dev hit@10 delta -0.0527, not advanced
dense_sparse_rrf_train_dev_probe: Stage81, dev hit@10 delta -0.0132, not advanced
bm25_k1_b_grid_train_to_dev: Stage82, dev hit@10 delta 0.0000, not advanced
```

汇总：

```text
allowed_stage76_candidate_count: 5
allowed_candidate_outcome_count: 5
allowed_candidates_completed: true
blocked_candidate_count: 1
runtime_advancing_candidate_count: 0
best_selected_dev_hit10_delta: 0.0
stage76_retrieval_recall_set_exhausted: true
```

Stage82 有两个 dev-only observation：

```text
bm25_grid__k1_1_20__b_0_95 dev hit@10: 0.7105, baseline: 0.6974, delta: +0.0131
bm25_grid__k1_1_50__b_0_95 dev hit@10: 0.7105, baseline: 0.6974, delta: +0.0131
```

但它们没有被 train-only rule 选中，因此不能用 dev 反选，不能推进 runtime。

### Guard checks

全部通过：

```text
source_reports_are_expected_stages: passed
stage76_allowed_candidates_all_accounted_for: passed
source_doc_ids_candidate_remains_blocked: passed
no_allowed_candidate_advanced_to_runtime: passed
all_source_decisions_keep_final_test_locked: passed
all_source_decisions_forbid_test_tuning: passed
all_source_decisions_keep_runtime_defaults_unchanged: passed
stage83_runs_summary_only_no_new_retrieval_metrics: passed
stage83_final_test_metrics_not_run: passed
stage83_default_runtime_policy_unchanged: passed
```

### 可视化产物

```text
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_dev_hit10_deltas.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_top10_net_changes.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_candidate_advancement_status.svg
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83_visuals\stage83_next_route_options.svg
```

Stage83 JSON SHA256：

```text
8A775C08F867C0E6C6DFB2E62606CF04BD53AC449EA0C85B75A674450FA8C7B4
```

### 问题、原因与修正

- 问题 1：Stage83 artifact 已成功生成后，我用 Bash 风格的 `python - <<'PY'` 在 PowerShell 中读取摘要，命令失败。
  - 原因：PowerShell 不支持 Bash here-doc 重定向写法。
  - 修正：改用 PowerShell here-string 管道到 `python -`，重新读取 Stage83 JSON 摘要成功。
  - 影响：该问题只影响人工校验摘要读取命令，不影响 Stage83 artifact、指标、guard checks 或文档结论。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_recall_exhaustion_summary.py scripts\summarize_primeqa_hybrid_retrieval_recall_exhaustion.py tests\test_primeqa_hybrid_retrieval_recall_exhaustion_summary.py
pytest -q tests\test_primeqa_hybrid_retrieval_recall_exhaustion_summary.py
```

结果：

```text
ruff: passed
pytest: 2 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage83 JSON: no matches
git check-ignore Stage83 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 215 passed
git diff --check: passed
```

### 结论

- Stage83 已完成 Stage76 retrieval-recall candidate set 收口。
- 5 个 allowed candidates 全部已经验证或可行性检查完成。
- 没有任何 allowed candidate 可以推进到 runtime/default policy。
- source `DOC_IDS` oracle union candidate 仍然 blocked。
- Stage83 没有使用 test。
- Stage83 没有运行 final test metrics。
- Stage83 没有运行新的 retrieval metrics。
- 默认 runtime policy 保持 unchanged。

### 我学到了

- 当第一波 retrieval-recall candidate 全部没有通过 train/dev gate 时，下一步不应该直接打开 test，也不应该把 dev-only observation 当成选择依据。
- stage summary 本身也需要 guard checks，否则很容易把“已经试过很多路线”误写成“可以放宽测试集边界”。
- dev 上更好的 BM25 参数仍然只能是后续设计线索，不能成为当前策略选择。

### 下一步

Stage84：先确认下一条 train/dev-only 路线，再运行任何新指标。

推荐路线是 `second_wave_retrieval_candidate_design`：汇总 Stage75 和 Stage77-82 changed-case evidence，设计第二波 retrieval candidate set。

Stage84 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage84：第二波 retrieval candidate design

### 目标

Stage84 的目标是在 Stage83 已确认第一波 Stage76 retrieval-recall 候选全部耗尽之后，设计第二波 retrieval candidate set。

本阶段只做设计和排序：

- 汇总 Stage75 miss drivers。
- 汇总 Stage77-82 的 changed-case / metric lessons。
- 确认 source `DOC_IDS` oracle union 仍然 blocked。
- 产出下一轮候选和推荐顺序。
- 不运行新的 retrieval metrics。
- 不触碰 test，不跑 final metrics，不改变 runtime 默认策略。

### 路线确认

Stage83 要求 Stage84 路线必须先确认。本轮用户说“好下一步”，我按上一轮推荐路线解释为确认：

```text
route_id: second_wave_retrieval_candidate_design
confirmed: true
confirmation_note: user confirmed recommended Stage84 route in current turn
```

### 本次改动

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_second_wave_retrieval_candidate_design.py
scripts/design_primeqa_hybrid_second_wave_retrieval_candidates.py
tests/test_primeqa_hybrid_second_wave_retrieval_candidate_design.py
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
```

更新：

```text
docs/primeqa_hybrid_retrieval_recall_exhaustion_summary.md
docs/primeqa_hybrid_retrieval_recall_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\design_primeqa_hybrid_second_wave_retrieval_candidates.py --user-confirmed-route --confirmation-note "user confirmed recommended Stage84 route in current turn" --output artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json --visualization-dir artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals
```

运行耗时：

```text
total: 0.006s
```

### Stage75 miss summary

```text
evaluated_questions: 446
hit_at_top_k: 0.6682
miss_count: 148
train misses: 125
dev misses: 23
not_found_top50: 110
rank_21_to_50: 24
rank_11_to_20: 14
unique_terms_16_plus: 125
top1_query_overlap_exceeds_gold: 103
source_doc_ids_oracle_presence_count: 148
```

`source_doc_ids_oracle_presence_count: 148` 仍然只是诊断事实，不能转成 runtime 检索证据。

### 第一波路线 lesson

```text
Stage77 query-view ablation:
  selected_dev_hit10_delta: -0.0395
  selected_dev_top10_net: -3

Stage78 fielded title/text BM25:
  selected_dev_hit10_delta: 0.0000
  selected_dev_top10_net: 0
  selected_dev_mrr_delta: +0.0219

Stage79 section BM25 rollup:
  selected_dev_hit10_delta: -0.0527
  selected_dev_top10_net: -4
  dev_search_depth_net: -1

Stage81 dense+sparse RRF:
  selected_dev_hit10_delta: -0.0132
  selected_dev_top10_net: -1
  selected_dev_not_found_delta: -6
  selected_dev_search_depth_net: +6

Stage82 BM25 k1/b grid:
  selected_dev_hit10_delta: 0.0000
  best_dev_non_selected_config_id: bm25_grid__k1_1_20__b_0_95
  best_dev_non_selected_hit10_delta: +0.0131
```

Stage82 的 `b=0.95` 仍然不能被 dev 反选，只能作为后续设计线索。

### 第二波候选

```text
lexical_cluster_diversity_rerank_design:
  priority: 210
  target_miss_count: 143
  dev_targets: 22
  prior_signal_score: 0.50

structured_query_keyphrase_compaction_design:
  priority: 207
  target_miss_count: 143
  dev_targets: 22
  prior_signal_score: 0.35

section_signal_guarded_expansion_design:
  priority: 174
  target_miss_count: 119
  dev_targets: 17
  prior_signal_score: 0.45

score_margin_bm25_normalization_gate_design:
  priority: 171
  target_miss_count: 111
  dev_targets: 18
  prior_signal_score: 0.48

selective_dense_sparse_low_overlap_gate_design:
  priority: 159
  target_miss_count: 111
  dev_targets: 17
  prior_signal_score: 0.68

source_doc_ids_oracle_union_blocked:
  status: blocked_from_train_dev_experiment
  target_miss_count: 148
```

推荐顺序：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

### Guard checks

全部通过：

```text
source_reports_are_expected_stages: passed
user_confirmed_stage84_recommended_route: passed
stage83_recommended_route_matches_stage84: passed
stage83_required_confirmation_was_respected: passed
stage76_candidates_are_exhausted: passed
stage83_has_no_runtime_advancing_candidate: passed
source_doc_ids_candidate_not_reintroduced: passed
source_doc_ids_candidate_remains_blocked: passed
all_source_decisions_keep_final_test_locked: passed
all_source_decisions_forbid_test_tuning: passed
all_source_decisions_keep_runtime_defaults_unchanged: passed
stage84_design_only_no_new_retrieval_metrics: passed
stage84_final_test_metrics_not_run: passed
stage84_default_runtime_policy_unchanged: passed
```

### 可视化产物

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_priority_scores.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_target_misses.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_dev_targets.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_candidate_prior_signal_scores.svg
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84_visuals\stage84_second_wave_allowed_vs_blocked_candidates.svg
```

Stage84 JSON SHA256：

```text
AB09F566ADF085A05DDBB28BF7065B4FD9320C8148D7E2F723D73D73F50EB8EF
```

### 问题、原因与修正

- 问题 1：探索现有源码时，一个带 Windows 不兼容 glob 的 `rg` 命令失败。
  - 原因：PowerShell/Windows 对该路径 glob 表达式解析为非法文件名。
  - 修正：改用明确文件路径和 Python 摘要脚本读取报告结构。
  - 影响：只影响探索命令，不影响 Stage84 artifact。
- 问题 2：Stage84 新模块首次 `ruff` 报两处字符串行过长。
  - 原因：`runtime_evidence_policy` 中两条说明超过项目行宽。
  - 修正：拆分字符串，逻辑未变化。
  - 影响：只影响格式检查，不影响报告结果。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_second_wave_retrieval_candidate_design.py scripts\design_primeqa_hybrid_second_wave_retrieval_candidates.py tests\test_primeqa_hybrid_second_wave_retrieval_candidate_design.py
pytest -q tests\test_primeqa_hybrid_second_wave_retrieval_candidate_design.py
```

结果：

```text
ruff: passed after line-length fix
pytest: 2 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage84 JSON: no matches
git check-ignore Stage84 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 217 passed
git diff --check: passed
```

### 结论

- Stage84 已完成第二波 retrieval candidate design。
- Stage84 没有运行新的 retrieval metrics。
- Stage84 没有使用 test。
- Stage84 没有运行 final test metrics。
- Stage84 没有把 source `DOC_IDS` 重新引入候选。
- 默认 runtime policy 保持 unchanged。
- 推荐下一候选为 `lexical_cluster_diversity_rerank_design`。

### 我学到了

- 第一波路线失败后，不能简单“换一个检索器继续跑”；应该把失败 lesson 转成更细的第二波候选约束。
- dense route 虽然有 not-found 改善信号，但 top10 regression 仍然说明它不能无条件推进。
- dev-only signal 可以帮助设计下一轮 hypothesis，但不能作为选择策略的证据。
- 第二波候选开始涉及 gate / diversity / selective expansion，必须先冻结 protocol，再考虑 train/dev metric run，不能直接改 runtime。

### 下一步

Stage85：先确认并冻结 `lexical_cluster_diversity_rerank_design` 的 train/dev-only protocol，再考虑运行 train/dev 指标。

Stage85 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage85：lexical cluster diversity rerank protocol freeze

### 目标

Stage85 的目标是冻结 Stage84 推荐候选 `lexical_cluster_diversity_rerank_design` 的 train/dev-only 运行协议。

本阶段只做 protocol freeze：

- 固定 protocol id、baseline、候选配置网格、feature contract 和 train selection rule。
- 不运行 train/dev retrieval metrics。
- 不触碰 test，不跑 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不改变 runtime 默认策略。

### 候选确认

本轮用户说“好下一步”，我按上一轮 Stage84 推荐理解为确认进入推荐候选：

```text
confirmed: true
confirmed_candidate_id: lexical_cluster_diversity_rerank_design
confirmation_note: user confirmed recommended Stage85 candidate in current turn
```

### 本次改动

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_lexical_cluster_diversity_protocol.py
scripts/freeze_primeqa_hybrid_lexical_cluster_diversity_protocol.py
tests/test_primeqa_hybrid_lexical_cluster_diversity_protocol.py
docs/primeqa_hybrid_lexical_cluster_diversity_protocol.md
```

更新：

```text
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\freeze_primeqa_hybrid_lexical_cluster_diversity_protocol.py --user-confirmed-candidate --confirmed-candidate-id lexical_cluster_diversity_rerank_design --confirmation-note "user confirmed recommended Stage85 candidate in current turn" --output artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85.json --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals
```

运行耗时：

```text
total: 0.000s
```

### 冻结协议

```text
protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
candidate_id: lexical_cluster_diversity_rerank_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
```

baseline：

```text
config_id: full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

候选配置网格：

```text
lcdr_penalty_0_03_title_query_cluster: duplicate_penalty_weight 0.03
lcdr_penalty_0_06_title_query_cluster: duplicate_penalty_weight 0.06
lcdr_penalty_0_09_title_query_cluster: duplicate_penalty_weight 0.09
lcdr_penalty_0_12_title_query_cluster: duplicate_penalty_weight 0.12
```

rerank formula：

```text
adjusted_score =
  baseline_bm25_score
  - duplicate_penalty_weight * top1_bm25_score * cluster_duplicate_index
```

train selection rule：

```text
Select the candidate config on train by hit@10, then hit@5, hit@1, MRR@10,
fewer top10 regressions, fewer rank-down within top10, then config_id.
Dev is validation only.
```

### Runtime evidence boundary

允许 runtime 特征：

```text
query_token_count
query_unique_token_count
baseline_bm25_rank
baseline_bm25_score
score_margin_to_top1
score_margin_to_previous
query_overlap_count
title_query_overlap_count
document_token_count
title_query_overlap_hash
lexical_cluster_hash
cluster_duplicate_index
cluster_size_in_candidate_depth
```

禁止 runtime 特征：

```text
source_DOC_IDS
answer_doc_id
gold_document_rank
gold_label
frozen_test_split_membership
```

changed-case 只允许 public-safe 字段：

```text
sample_id
split
baseline_rank
challenger_rank
baseline_cluster_duplicate_index
challenger_cluster_duplicate_index
config_id
```

### Guard checks

全部通过：

```text
source_report_is_stage84: passed
user_confirmed_recommended_candidate: passed
confirmed_candidate_matches_stage84_recommendation: passed
candidate_is_recommended_for_protocol_design: passed
stage84_requires_confirmation_before_train_dev_run: passed
stage84_final_test_metrics_locked: passed
stage84_forbids_test_tuning: passed
stage84_runtime_default_unchanged: passed
protocol_id_is_fixed: passed
candidate_config_grid_is_predeclared: passed
source_doc_ids_forbidden_in_runtime_features: passed
report_fields_are_public_safe: passed
stage85_freezes_protocol_without_metrics: passed
stage85_final_test_metrics_not_run: passed
stage85_default_runtime_policy_unchanged: passed
```

### 可视化产物

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_candidate_config_penalties.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_feature_group_counts.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_protocol_decision_flags.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_protocol_stage85_visuals\stage85_lcdr_guard_check_status.svg
```

Stage85 JSON SHA256：

```text
A8A1AF1F38937EDDD87F146DE320C2DF45E798FF96ACBDE45A2E0F386303D814
```

### 问题、原因与修正

- 问题 1：首次 artifact 安全检查命中了 `prohibited_report_fields` 中的 raw-field key 名。
  - 原因：这些字符串是禁止字段清单，不是原文数据，但会让 raw-field pattern 检查产生噪声。
  - 修正：把禁止字段清单改成自然语言描述，例如 `question text`、`document body text`，然后重新生成 Stage85 artifact。
  - 影响：只影响报告描述字段名；protocol 逻辑、guard checks 和结论不变。重跑后安全检查无命中。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_protocol.py scripts\freeze_primeqa_hybrid_lexical_cluster_diversity_protocol.py tests\test_primeqa_hybrid_lexical_cluster_diversity_protocol.py
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_protocol.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage85 JSON: no matches after prohibited-field wording fix
git check-ignore Stage85 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 220 passed
git diff --check: passed
```

### 结论

- Stage85 已冻结 `lexical_cluster_diversity_rerank_train_dev_v1`。
- Stage85 没有运行 train/dev retrieval metrics。
- Stage85 没有使用 test。
- Stage85 没有运行 final test metrics。
- Stage85 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- 默认 runtime policy 保持 unchanged。
- 下一步经确认后可以运行 train/dev-only 指标。

### 我学到了

- protocol freeze artifact 自身也要避免出现 raw-field key 名，否则安全扫描会产生噪声。
- 第二波候选涉及 rerank 公式时，必须先把 feature contract 和禁止特征写清楚，再谈任何指标。
- `can_run_train_dev_metrics_after_user_confirmation: true` 只表示协议已准备好，不表示已经运行 train/dev 指标。

### 下一步

Stage86：在用户确认后，运行 frozen protocol `lexical_cluster_diversity_rerank_train_dev_v1` 的 train/dev-only comparison。

Stage86 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage86：lexical cluster diversity rerank train/dev comparison

### 目标

Stage86 的目标是在用户当前回合确认后，运行 Stage85 已冻结的
`lexical_cluster_diversity_rerank_train_dev_v1` train/dev-only comparison。

本阶段只允许：

- 读取冻结的 Stage68 train/dev split。
- 读取 Stage75 BM25 baseline report。
- 读取 Stage85 frozen protocol。
- 用 runtime 可见的 query/title token、BM25 rank/score、cluster hash 和 duplicate index 做 rerank。
- 在 train 上选择 config，在 dev 上验证。

本阶段明确禁止：

- 读取 test split。
- 跑 final test metrics。
- 用 source `DOC_IDS` 作为 runtime 检索证据。
- 输出 raw question、answer、document title/body text 或 raw token。
- 修改 runtime 默认策略。

### 用户确认

这次确认发生在当前回合，记录如下：

```text
confirmed: true
confirmed_protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
confirmation_note: user confirmed Stage86 train/dev metric run in current turn
```

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_lexical_cluster_diversity_comparison.py
scripts/run_primeqa_hybrid_lexical_cluster_diversity_comparison.py
tests/test_primeqa_hybrid_lexical_cluster_diversity_comparison.py
docs/primeqa_hybrid_lexical_cluster_diversity_comparison.md
```

### 更新文件

```text
docs/primeqa_hybrid_lexical_cluster_diversity_protocol.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\run_primeqa_hybrid_lexical_cluster_diversity_comparison.py --user-confirmed-protocol --confirmed-protocol-id lexical_cluster_diversity_rerank_train_dev_v1 --confirmation-note "user confirmed Stage86 train/dev metric run in current turn" --output artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86.json --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals
```

运行耗时：

```text
total: 29.330s
```

### 实现内容

Stage86 新增了一个独立的 train/dev comparison 模块：

- `_BM25CandidateIndex`：一次建立 full-document BM25 index，复用 Stage75 baseline 参数 `k1=1.5, b=0.75`。
- `LexicalClusterDiversityConfig`：承载 Stage85 冻结的四个 duplicate penalty config。
- `cluster hash`：使用 sorted normalized title tokens 与 query tokens 的交集做 SHA256，截断为 16 位；报告只写 hash 和聚合计数，不写 raw token。
- `cluster_duplicate_index`：沿 baseline BM25 rank 顺序统计同一 cluster 中已经出现过的候选数量。
- rerank 公式：

```text
adjusted_score =
  baseline_bm25_score
  - duplicate_penalty_weight * top1_bm25_score * cluster_duplicate_index
```

tie breakers：

```text
higher baseline_bm25_score
lower baseline_bm25_rank
lower stable document id sort key
```

未确认时的行为：

- 如果 `--user-confirmed-protocol` 未提供，或 confirmed protocol id 不匹配，模块会生成 blocked report。
- blocked report 不计算 train/dev retrieval metrics。
- 这保证“未确认不跑指标”是代码行为，不只是文档约束。

### Train 结果

Train-selected config：

```text
lcdr_penalty_0_06_title_query_cluster
```

Train baseline：

```text
hit@1: 0.4243
hit@5: 0.6054
hit@10: 0.6622
MRR@10: 0.5023
not_found@50: 93
rank_11_to_50: 32
```

Train selected config：

```text
hit@1: 0.4243
hit@5: 0.6054
hit@10: 0.6676
MRR@10: 0.5031
not_found@50: 93
rank_11_to_50: 30
```

Train delta：

```text
hit@10_delta: +0.0054
top10_improvement_count: 4
top10_regression_count: 2
rank_up_within_top10_count: 2
rank_down_within_top10_count: 0
rank_11_to_50_count_delta: -2
```

### Dev 结果

Dev baseline：

```text
hit@1: 0.4342
hit@5: 0.6579
hit@10: 0.6974
MRR@10: 0.5331
not_found@50: 17
rank_11_to_50: 6
```

Dev selected config：

```text
hit@1: 0.4342
hit@5: 0.6579
hit@10: 0.6974
MRR@10: 0.5331
not_found@50: 17
rank_11_to_50: 6
```

Dev delta：

```text
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
rank_up_within_top10_count: 0
rank_down_within_top10_count: 0
not_found_count_at_50_delta: 0
rank_11_to_50_count_delta: 0
```

### Guard checks

全部通过：

```text
analysis_splits_are_train_dev_only: passed
top_k_values_include_primary_top10: passed
search_depth_covers_primary_top10: passed
source_stage85_report_is_stage85: passed
stage85_protocol_id_matches: passed
stage85_candidate_id_matches: passed
user_confirmed_frozen_protocol: passed
confirmed_protocol_id_matches: passed
stage85_allows_train_dev_metrics_after_confirmation: passed
stage85_final_test_metrics_locked: passed
stage85_default_runtime_policy_unchanged: passed
candidate_config_grid_matches_frozen_protocol: passed
source_stage75_report_is_stage75: passed
baseline_train_hit10_matches_stage75: passed
baseline_dev_hit10_matches_stage75: passed
source_doc_ids_not_used_as_runtime_evidence: passed
changed_case_fields_public_safe: passed
final_test_metrics_not_run: passed
default_runtime_policy_unchanged: passed
```

### 可视化产物

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_train_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_top10_changes.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86_visuals\stage86_lcdr_dev_answer_duplicate_buckets.svg
```

Stage86 JSON SHA256：

```text
2F00764F52AA1279A7F526E5ACF0735E6FA90C6D27E63026FE011630D7EB4195
```

Visualization SHA256：

```text
stage86_lcdr_dev_answer_duplicate_buckets.svg: C106089E411BC1977C3C181150D5929F4A71801533432AB10BD157B95FD932D0
stage86_lcdr_dev_delta_hit_at_10.svg: 6F5E95E5EDCDDB9B510E091B26B477D9B482F5EC03F167A2816F2623A9DDB0B5
stage86_lcdr_dev_hit_at_10.svg: 001C0AF383446EBD801A6B43A1AAB1994C049F276A9FD2B6401E59E30949D640
stage86_lcdr_dev_top10_changes.svg: 66682BE713A073DC0CDFD0E7322A298A0511BEE36907765CEABF0822AC568445
stage86_lcdr_train_hit_at_10.svg: 89C9C170D88643E6EE90DF1B65A6731143D964C4CC60C25E2D762A01B90F1E09
```

### 问题、原因与修正

- 问题 1：新增测试夹具第一次没有产生 rank-up。
  - 原因：夹具中 answer 的 BM25 分数约 `0.99`，重复 cluster 候选分数约 `1.44`；即使用 12% penalty，第二个重复候选仍高于 answer。
  - 修正：调整小型单测夹具，让 answer 分数真实地处于“略低于第二个重复候选、但高于惩罚后重复候选”的区间；这只是测试夹具，不影响真实 Stage86 指标。
  - 影响：测试从“希望发生 rank-up”改成“在可解释分数条件下真实验证 rank-up”。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_comparison.py scripts\run_primeqa_hybrid_lexical_cluster_diversity_comparison.py tests\test_primeqa_hybrid_lexical_cluster_diversity_comparison.py
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_comparison.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage86 JSON: no matches
git check-ignore Stage86 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 223 passed
git diff --check: passed
```

### 结论

- Stage86 已完成 frozen protocol 的 train/dev-only comparison。
- Train 选中了 `lcdr_penalty_0_06_title_query_cluster`。
- Train hit@10 有小幅提升：`+0.0054`。
- Dev hit@10 没有提升：`+0.0000`。
- Dev top10 improvements/regressions 都是 `0`。
- Stage84 的 target metric contract 要求 dev hit@10 必须提升，所以 lexical cluster diversity 不能进入 runtime。
- Stage86 没有使用 test。
- Stage86 没有跑 final metrics。
- Stage86 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- 默认 runtime policy 保持 unchanged。

### 我学到了

- train-only selection 可以产生看起来不错的小幅增益，但如果 dev 没有提升，就不能把它当成泛化成功。
- diversity rerank 的收益上限受 baseline top50 召回限制：它能重排候选，但不能找回 baseline top50 之外的 answer document。
- 对 cluster 方案来说，公开报告必须特别克制：hash、计数、rank 变化可以记录，raw title/query/answer/token 不能写出。
- 未确认不跑指标应该固化在代码门禁里，而不是只靠流程说明。

### 下一步

Stage87：停止 lexical cluster diversity 作为 retrieval-recall route，除非用户明确确认一个新的 train/dev-only protocol；推荐转向下一个 second-wave candidate protocol。

Stage87 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage87：lexical cluster diversity stop decision

### 目标

Stage87 的目标是把 Stage86 的结论固化为 stop decision：

- 停止 `lexical_cluster_diversity_rerank_design` 这条 retrieval-recall route。
- 阻止这条 route defaultization。
- 不打开 final-test gate。
- 确认 Stage84 队列中的下一个候选是 `structured_query_keyphrase_compaction_design`。

本阶段是 decision checkpoint，不运行新的 retrieval metrics。

### 用户确认

这次确认发生在当前回合，记录如下：

```text
route_id: lexical_cluster_diversity_stop_decision
confirmed: true
confirmation_note: user confirmed Stage87 stop decision in current turn
```

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_lexical_cluster_diversity_stop_decision.py
scripts/decide_primeqa_hybrid_lexical_cluster_diversity_stop.py
tests/test_primeqa_hybrid_lexical_cluster_diversity_stop_decision.py
docs/primeqa_hybrid_lexical_cluster_diversity_stop_decision.md
```

### 更新文件

```text
docs/primeqa_hybrid_lexical_cluster_diversity_comparison.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\decide_primeqa_hybrid_lexical_cluster_diversity_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage87 stop decision in current turn" --output artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json --visualization-dir artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals
```

运行耗时：

```text
total: 0.001s
```

### 输入证据

Stage87 只读取 public-safe artifacts：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_lexical_cluster_diversity_comparison_stage86.json
```

没有读取 train/dev/test split，没有读取原始文档库，没有运行新的检索指标。

### Stage86 证据

Stage86 train-selected config：

```text
lcdr_penalty_0_06_title_query_cluster
```

Train 证据：

```text
train hit@10 delta: +0.0054
train top10 improvements: 4
train top10 regressions: 2
```

Dev 证据：

```text
dev hit@10 delta: +0.0000
dev top10 improvements: 0
dev top10 regressions: 0
dev rank-up within top10: 0
dev rank-down within top10: 0
dev not-found@50 delta: 0
dev rank 11-50 delta: 0
```

Stage84 对 LCDR 的 target metric contract：

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: top1-overlap decoy misses should decrease
guard: no title/body text should be written to reports
```

### Decision

```text
status: primeqa_hybrid_lexical_cluster_diversity_route_stopped
stopped_candidate_id: lexical_cluster_diversity_rerank_design
stopped_protocol_id: lexical_cluster_diversity_rerank_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: structured_query_keyphrase_compaction_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Candidate queue

Stage84 原始顺序：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Stage87 停止 LCDR 后的剩余顺序：

```text
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

下一个候选：

```text
structured_query_keyphrase_compaction_design
```

### Guard checks

全部通过：

```text
source_stage84_report_is_stage84: passed
source_stage86_report_is_stage86: passed
user_confirmed_stage87_stop_decision: passed
stage86_comparison_completed: passed
stage86_candidate_matches_lcdr: passed
stage86_protocol_matches_lcdr: passed
stage84_candidate_metric_contract_requires_dev_hit10_gain: passed
stage86_train_selected_config_has_no_dev_hit10_gain: passed
stage86_dev_top10_net_not_positive: passed
stage86_final_test_metrics_locked: passed
stage86_final_test_gate_closed: passed
stage86_forbids_test_tuning: passed
stage86_default_runtime_policy_unchanged: passed
stage84_execution_order_contains_stopped_candidate: passed
stage84_next_candidate_available_after_lcdr_stop: passed
source_doc_ids_not_selected_as_next_candidate: passed
stage87_no_new_retrieval_metrics_run: passed
stage87_final_test_metrics_not_run: passed
stage87_default_runtime_policy_unchanged: passed
```

### 可视化产物

```text
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_dev_change_counts.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_stop_decision_flags.svg
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87_visuals\stage87_lcdr_stop_guard_check_status.svg
```

Stage87 JSON SHA256：

```text
F5C9BC614F8210B20DD6B099C36126D1078FE1262169BD874AE32136A7F9FD79
```

Visualization SHA256：

```text
stage87_lcdr_dev_change_counts.svg: C00034DD885257891031C95D2DFEC9FE6F7E10459CDB076377D56A7B8C6042F1
stage87_lcdr_stop_decision_flags.svg: 4283DAA01223C72FBA86049B299BB89405781F6BFBEA798364D2A25BECFD9762
stage87_lcdr_stop_guard_check_status.svg: B21CFCE72854F0AEF7EF46EF361CEBAB96C90FF25B4AC429ED2E1C09C1F5B1B0
stage87_lcdr_train_dev_hit10_delta.svg: 46427B4BAE399D2084202B54DDB13BD1C91F7F82AF03AC17971B61F275E6187F
stage87_second_wave_remaining_candidate_priority.svg: E0D20D4DB853F04AB4CCADC75D63BA3365079689F9825851BDCDCFB3F449CD78
```

### 问题、原因与修正

- 问题 1：最初我按猜测去找 Stage74 的 Python stop-decision 文件，路径不存在。
  - 原因：Stage74 实际是文档型 stop decision，没有独立 Python 模块。
  - 修正：读取真实的 Stage74 文档作为模式参考，并为 Stage87 新增工程化 JSON + SVG decision report。
  - 影响：没有修改文件；只是定位路径失败后改用真实文件。

- 问题 2：Stage87 模块首次 lint 命中一行超过 100 字符。
  - 原因：`recommended_execution_order` 的列表推导式写在一行。
  - 修正：拆成多行后重新运行局部 lint。
  - 影响：仅格式修正。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_lexical_cluster_diversity_stop_decision.py scripts\decide_primeqa_hybrid_lexical_cluster_diversity_stop.py tests\test_primeqa_hybrid_lexical_cluster_diversity_stop_decision.py
pytest -q tests\test_primeqa_hybrid_lexical_cluster_diversity_stop_decision.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage87 JSON: no matches
git check-ignore Stage87 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 226 passed
git diff --check: passed
```

### 结论

- Stage87 已停止 lexical cluster diversity route。
- 当前 LCDR route defaultization 是 blocked。
- Stage87 没有运行新的 retrieval metrics。
- Stage87 没有使用 test。
- Stage87 没有跑 final metrics。
- Stage87 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- 默认 runtime policy 保持 unchanged。
- 下一个候选是 `structured_query_keyphrase_compaction_design`。

### 我学到了

- stop decision 也值得结构化成 artifact：这样“为什么停”和“下一候选是谁”可以被 guard checks 和可视化共同验证。
- train 上的小增益不能抵消 dev primary metric 没提升这一事实；停止路线比继续调同一路线更诚实。
- decision checkpoint 不应顺手启动下一候选指标；下一候选仍需要先冻结 protocol。

### 下一步

Stage88：确认并冻结 `structured_query_keyphrase_compaction_design` 的 train/dev-only protocol。

Stage88 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage88：structured query keyphrase compaction protocol freeze

### 目标

Stage88 的目标是确认并冻结 `structured_query_keyphrase_compaction_design` 的 train/dev-only protocol。

本阶段只做 protocol freeze：

- 不运行 retrieval metrics。
- 不读取 train/dev/test split 文件。
- 不触碰 frozen test。
- 不运行 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不改变 runtime 默认策略。

### 用户确认

本轮用户说“好确认下一步”，我按 Stage87 已记录的推荐下一步解释为确认：

```text
confirmed: true
confirmed_candidate_id: structured_query_keyphrase_compaction_design
confirmation_note: user confirmed Stage88 structured query protocol freeze in current turn
```

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_structured_query_protocol.py
scripts/freeze_primeqa_hybrid_structured_query_protocol.py
tests/test_primeqa_hybrid_structured_query_protocol.py
docs/primeqa_hybrid_structured_query_protocol.md
```

### 更新文件

```text
docs/primeqa_hybrid_lexical_cluster_diversity_stop_decision.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\freeze_primeqa_hybrid_structured_query_protocol.py --user-confirmed-candidate --confirmed-candidate-id structured_query_keyphrase_compaction_design --confirmation-note "user confirmed Stage88 structured query protocol freeze in current turn" --output artifacts\primeqa_hybrid_structured_query_protocol_stage88.json --visualization-dir artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals
```

最终记录运行耗时：

```text
total: 0.000s
```

### 输入证据

Stage88 只读取 public-safe artifacts：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json
```

Stage87 的真实 decision：

```text
status: primeqa_hybrid_lexical_cluster_diversity_route_stopped
stopped_candidate_id: lexical_cluster_diversity_rerank_design
next_candidate_id: structured_query_keyphrase_compaction_design
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Frozen protocol

```text
protocol_id: structured_query_keyphrase_compaction_train_dev_v1
candidate_id: structured_query_keyphrase_compaction_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
```

冻结的 config grid：

```text
sqkc_action_error_product_v1: action_error_product_version_terms, max_terms=18
sqkc_title_guarded_action_error_v1: title_guarded_action_error_product_terms, max_terms=16
sqkc_error_first_compact_v1: error_identifier_first_terms, max_terms=14
sqkc_noun_phrase_compact_v1: deterministic_noun_phrase_like_terms, max_terms=20
```

训练选择规则：

```text
selection_split: train
validation_split: dev
rule: train 上按 hit@10、hit@5、hit@1、MRR@10、top10 regression、rank-down within top10、平均 compacted token count、config_id 选择。
dev_selection_forbidden: true
test_selection_forbidden: true
```

### Guard checks

全部通过：

```text
23 / 23 passed
```

关键 guard：

- Stage84 report 是 Stage84。
- Stage87 report 是 Stage87。
- Stage87 已停止 LCDR route。
- 用户确认的 candidate 与 Stage87 next candidate 一致。
- Stage84 和 Stage87 都保持 final-test locked。
- Stage84 和 Stage87 都禁止 test tuning。
- Stage84 和 Stage87 都保持 runtime default unchanged。
- protocol id 固定为 `structured_query_keyphrase_compaction_train_dev_v1`。
- config grid 是预声明 4 项。
- source `DOC_IDS` 被禁止作为 runtime feature。
- report 字段只允许 public-safe id、rank、count。
- Stage88 不运行指标、不跑 final metrics、不改 runtime 默认策略。

### 可视化产物

```text
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_config_token_limits.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_feature_group_counts.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_protocol_decision_flags.svg
artifacts\primeqa_hybrid_structured_query_protocol_stage88_visuals\stage88_structured_query_guard_check_status.svg
```

Stage88 JSON SHA256：

```text
12A9045D7B65208EFF840E931E296FE56ED4D02A5852DA3C299011BA14767A7C
```

Visualization SHA256：

```text
stage88_structured_query_config_token_limits.svg: 2E3043E29343BEB0FBA12DE53131D8D5A4803AB1771870F08023BDA052F20326
stage88_structured_query_feature_group_counts.svg: 90D455A2D32F7DD2647C8532E0E9F2CBADE6D612FC171C5350924EC3B0F1B2C1
stage88_structured_query_guard_check_status.svg: 11F2670D9C3A9AD8058F91F383E6D0F13BBEAA072B887363FFB165A7AE0D3CA7
stage88_structured_query_protocol_decision_flags.svg: 67FD7B7B6651A8209912306BD4030068F93EDDE389B38EF97DF46E8A2AF5B14B
```

### 问题、原因与修正

- 问题 1：第一版测试夹具中故意塞入了 `example_raw_question`，暴露出如果整段写回 Stage84 candidate，未来可能把 raw 字段带入 Stage88 report。
  - 原因：初版 `_selected_candidate` 直接返回 candidate dict。
  - 修正：改成 explicit public candidate summary 白名单，只写出候选 id、状态、优先级、target contract 等 public-safe 字段。
  - 影响：Stage88 report 更安全，真实 Stage84 artifact 没有 raw 字段，但代码现在能防止上游结构变化误带出 raw 字段。

- 问题 2：第一版 safety scan 命中 `"answer_doc_id"`。
  - 原因：它只出现在 prohibited runtime features 中，不是实际 report 字段或 runtime evidence，但字段名形式容易造成误判。
  - 修正：改成自然语言禁止项 `answer document IDs`，保留禁止语义，同时让 raw/field safety scan 无命中。
  - 影响：协议含义不变，产物更清晰。

- 问题 3：第一次 lint 命中 `stage84_decision` 已读取但未使用。
  - 原因：Stage88 初版只校验 Stage87 的锁定状态，没有把 Stage84 decision 的 final/test/default guard 加入。
  - 修正：新增 Stage84 final-test locked、forbid test tuning、runtime default unchanged 三个 guard checks。
  - 影响：guard checks 从 20 项扩展到 23 项，协议边界更完整。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_protocol.py scripts\freeze_primeqa_hybrid_structured_query_protocol.py tests\test_primeqa_hybrid_structured_query_protocol.py
pytest -q tests\test_primeqa_hybrid_structured_query_protocol.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet field patterns over Stage88 JSON: no matches
git check-ignore Stage88 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 229 passed
git diff --check: passed
```

### 结论

- Stage88 已冻结 `structured_query_keyphrase_compaction_train_dev_v1`。
- Stage88 没有运行 retrieval metrics。
- Stage88 没有读取 train/dev/test split。
- Stage88 没有使用 test。
- Stage88 没有跑 final metrics。
- Stage88 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage88 没有改变 runtime 默认策略。
- 现在只能在用户确认后进入 Stage89 train/dev-only metric run。

### 我学到了

- Protocol freeze 不能只写“下一候选是谁”，还要把来源链路写进 guard checks：Stage84 负责候选 contract，Stage87 负责确认上一条 route 已停止。
- 对 public-safe report 来说，字段白名单比“相信输入 artifact 很干净”更可靠；尤其是后续上游结构可能扩展时。
- 禁止项本身也要避免使用容易被误判的 raw 字段名；自然语言禁止项更适合写入 public-safe artifact。
- Stage88 的价值不是证明 structured query 有效果，而是把下一次 train/dev metric run 的边界锁死，防止 dev/test 泄漏。

### 下一步

Stage89：在用户确认后，运行 frozen train/dev-only structured query keyphrase compaction comparison。

Stage89 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage89：structured query keyphrase compaction train/dev comparison

### 目标

Stage89 的目标是运行 Stage88 冻结的 `structured_query_keyphrase_compaction_train_dev_v1` train/dev-only comparison。

本阶段只做 train/dev 指标对比：

- 只读取 Stage68 frozen train/dev split。
- 不读取 frozen test split。
- 不运行 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不把 raw question、compacted query、answer、document title/body 写入 report。
- 不改变 runtime 默认策略。

### 用户确认

本轮用户说“好下一步”，我按 Stage88 已记录的下一步解释为确认运行 Stage89：

```text
confirmed: true
confirmed_protocol_id: structured_query_keyphrase_compaction_train_dev_v1
confirmation_note: user confirmed Stage89 train/dev metric run in current turn
```

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_structured_query_comparison.py
scripts/run_primeqa_hybrid_structured_query_comparison.py
tests/test_primeqa_hybrid_structured_query_comparison.py
docs/primeqa_hybrid_structured_query_comparison.md
```

### 更新文件

```text
docs/primeqa_hybrid_structured_query_protocol.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\run_primeqa_hybrid_structured_query_comparison.py --user-confirmed-protocol --confirmed-protocol-id structured_query_keyphrase_compaction_train_dev_v1 --confirmation-note "user confirmed Stage89 train/dev metric run in current turn" --output artifacts\primeqa_hybrid_structured_query_comparison_stage89.json --visualization-dir artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals
```

运行耗时：

```text
total: 44.736s
```

### 输入证据

```text
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
artifacts\primeqa_hybrid_structured_query_protocol_stage88.json
```

实际加载数据：

```text
document_count: 28482
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

### Frozen grid

```text
sqkc_action_error_product_v1: action_error_product_version_terms, max_terms=18
sqkc_title_guarded_action_error_v1: title_guarded_action_error_product_terms, max_terms=16
sqkc_error_first_compact_v1: error_identifier_first_terms, max_terms=14
sqkc_noun_phrase_compact_v1: deterministic_noun_phrase_like_terms, max_terms=20
```

### Train metrics

```text
full_document_bm25_baseline:
  hit@10: 0.6622
  mrr@10: 0.5023
  not_found@50: 93
  avg_compacted_terms: 55.1703

sqkc_action_error_product_v1:
  hit@10: 0.6351
  mrr@10: 0.4921
  not_found@50: 86
  avg_compacted_terms: 14.6811

sqkc_title_guarded_action_error_v1:
  hit@10: 0.6595
  mrr@10: 0.4940
  not_found@50: 80
  avg_compacted_terms: 13.5514

sqkc_error_first_compact_v1:
  hit@10: 0.6081
  mrr@10: 0.4653
  not_found@50: 98
  avg_compacted_terms: 12.2649

sqkc_noun_phrase_compact_v1:
  hit@10: 0.6351
  mrr@10: 0.4858
  not_found@50: 92
  avg_compacted_terms: 15.4243
```

Train-selected config：

```text
sqkc_title_guarded_action_error_v1
```

Train-selected comparison：

```text
hit@10_delta: -0.0027
top10_improvement_count: 14
top10_regression_count: 15
top10_net_improvement_count: -1
not_found_count_at_50_delta: -13
rank_11_to_50_count_delta: +14
average_compacted_query_token_count_delta: -41.6189
```

### Dev metrics

```text
full_document_bm25_baseline:
  hit@10: 0.6974
  mrr@10: 0.5331
  not_found@50: 17
  avg_compacted_terms: 49.6316

sqkc_action_error_product_v1:
  hit@10: 0.6316
  mrr@10: 0.4991
  not_found@50: 17
  avg_compacted_terms: 14.8026

sqkc_title_guarded_action_error_v1:
  hit@10: 0.6447
  mrr@10: 0.5124
  not_found@50: 16
  avg_compacted_terms: 13.5395

sqkc_error_first_compact_v1:
  hit@10: 0.6579
  mrr@10: 0.5039
  not_found@50: 16
  avg_compacted_terms: 12.2632

sqkc_noun_phrase_compact_v1:
  hit@10: 0.6711
  mrr@10: 0.5314
  not_found@50: 18
  avg_compacted_terms: 15.3947
```

Dev comparison for train-selected config：

```text
hit@10_delta: -0.0527
top10_improvement_count: 1
top10_regression_count: 5
top10_net_improvement_count: -4
rank_up_within_top10_count: 5
rank_down_within_top10_count: 5
not_found_count_at_50_delta: -1
rank_11_to_50_count_delta: +5
average_compacted_query_token_count_delta: -36.0921
```

### Guard checks

全部通过：

```text
21 / 21 passed
```

关键 guard：

- analysis splits 只有 train/dev。
- Stage88 protocol id 和 candidate id 匹配。
- 用户确认了 frozen protocol。
- Stage88 允许确认后运行 train/dev metrics。
- Stage88 final-test locked。
- Stage88 forbid test tuning。
- Stage88 runtime default unchanged。
- candidate config grid 匹配 frozen protocol。
- Stage75 baseline train/dev hit@10 与 Stage89 baseline 一致。
- source `DOC_IDS` 没有作为 runtime evidence 使用。
- changed-case fields public-safe。
- raw/compacted query text 不写入 report。
- final metrics not run。
- runtime default unchanged。

### 可视化产物

```text
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_train_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_dev_top10_changes.svg
artifacts\primeqa_hybrid_structured_query_comparison_stage89_visuals\stage89_structured_query_average_compacted_terms.svg
```

Stage89 JSON SHA256：

```text
592EF83368BB8DB268114677E1BF9FDCAC1C4E37E8BC10099917CA7956BDF4A4
```

Visualization SHA256：

```text
stage89_structured_query_average_compacted_terms.svg: 3BA48FD64FCCA65D2D30CCC0600726A108392B5A1D417F97C9590A5779D50CDF
stage89_structured_query_dev_delta_hit_at_10.svg: 2520DBEC41AE2263845D67DB1C4AACB3ADD29686655AA9EA9D080FE8616C6D94
stage89_structured_query_dev_hit_at_10.svg: 1F27A1005A1074C5F5FE9C7F5E8846FF591E4D0712BF137268E3B68468E38327
stage89_structured_query_dev_top10_changes.svg: 92150CEC7848DBDB319AD0C1A39D4DA98EAB5F44309F4310CE3D7BFA08A50A08
stage89_structured_query_train_hit_at_10.svg: 62807A6A5CD10C34C6E85E88A644DED5008B67413D3A27764E7F8ACBC2A11B8A
```

### 问题、原因与修正

- 问题 1：第一版单测试图构造一个 top10 rescue case，但真实 BM25 rank 里答案文档已经是 baseline rank1。
  - 原因：答案文档匹配稀有关键项，噪声 stopword 文档不足以压过它。
  - 修正：不硬造假 rescue，改成验证 noisy full question 能被 structured query 明显压缩，同时保持 public-safe 和边界 guard。
  - 影响：测试更诚实；不再暗示 structured query 一定能 rescue。

- 问题 2：第一版测试 Stage75 fixture 写了 baseline hit@10 = 0.0，但真实 baseline 是 1.0。
  - 原因：夹具预期没有先用真实 rank 校验。
  - 修正：把 fixture Stage75 baseline 改为真实的 1.0。
  - 影响：guard check 的 baseline consistency 继续有效。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_comparison.py scripts\run_primeqa_hybrid_structured_query_comparison.py tests\test_primeqa_hybrid_structured_query_comparison.py
pytest -q tests\test_primeqa_hybrid_structured_query_comparison.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term field patterns over Stage89 JSON: no matches
git check-ignore Stage89 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 232 passed
git diff --check: passed
```

### 结论

- Stage89 已运行 frozen structured query train/dev-only comparison。
- Stage89 没有读取 test。
- Stage89 没有跑 final metrics。
- Stage89 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage89 没有写出 raw question / compacted query / answer / document text。
- Stage89 没有改变 runtime 默认策略。
- structured query compaction 明显减少 query terms。
- 但 train-selected config 在 dev 上 hit@10 下降 `-0.0527`。
- dev top10 regression `5` 多于 improvement `1`。
- primary contract 和 secondary contract 都没有通过。
- 该路线不能进入 runtime，也不能打开 final-test gate。

### 我学到了

- 大幅压缩 query 不等于召回质量提升；当前 BM25 baseline 可能依赖长问题中的弱词累积来保住 gold doc。
- train-only selection 非常重要：dev 上最接近 baseline 的 `sqkc_noun_phrase_compact_v1` 仍然不能被 dev-only 选择，而且它也没有超过 baseline。
- 对 retrieval-recall route 来说，减少 not-found@50 不能抵消 hit@10 明显下降；最终目标仍是把答案文档放进 top10。
- 测试夹具必须服从真实 rank，不应为了展示某种能力而写不真实的 Stage75 baseline。

### 下一步

Stage90：停止 structured query keyphrase compaction 作为 retrieval-recall route，除非用户明确确认一个新的 train/dev-only protocol；推荐转向下一个 second-wave candidate protocol。

Stage90 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## 2026-07-15 - Stage90 structured query stop decision

### 目标

本阶段不是训练、不是调参、也不是 final-test 评估，而是一个明确的路线决策检查点：

- 读取 Stage84 second-wave candidate design 和 Stage89 structured query train/dev comparison 的 public-safe 报告。
- 在用户本轮确认后，停止 `structured_query_keyphrase_compaction_design` 作为当前 retrieval-recall 路线。
- 从 Stage84 队列中剔除已经停止的 `lexical_cluster_diversity_rerank_design` 和本阶段停止的 structured query 路线。
- 确认下一候选路线为 `section_signal_guarded_expansion_design`。
- 继续锁定 test split，不跑 final metrics，不使用 source `DOC_IDS`，不改变 runtime 默认策略。

### 新增和更新文件

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_structured_query_stop_decision.py
scripts/decide_primeqa_hybrid_structured_query_stop.py
tests/test_primeqa_hybrid_structured_query_stop_decision.py
docs/primeqa_hybrid_structured_query_stop_decision.md
```

更新：

```text
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\decide_primeqa_hybrid_structured_query_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage90 stop decision in current turn" --output artifacts\primeqa_hybrid_structured_query_stop_decision_stage90.json --visualization-dir artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals
```

运行耗时：

```text
total: 0.002s
```

### 输入证据

Stage90 只读取 public-safe 报告：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_structured_query_comparison_stage89.json
```

本阶段没有加载 train/dev/test split 文件，没有跑 retrieval metrics，也没有重新读取原始问题、答案或文档正文。

### Stage89 依据

Stage89 train-selected config：

```text
sqkc_title_guarded_action_error_v1
```

Stage89 selected query view：

```text
title_guarded_action_error_product_terms
```

Train 结果：

```text
hit@10_delta: -0.0027
top10_improvement_count: 14
top10_regression_count: 15
```

Dev 结果：

```text
hit@10_delta: -0.0527
top10_improvement_count: 1
top10_regression_count: 5
rank_up_within_top10_count: 5
rank_down_within_top10_count: 5
not_found_count_at_50_delta: -1
rank_11_to_50_count_delta: +5
average_compacted_query_token_count_delta: -36.0921
```

Stage84 / Stage88 对这条路线的 metric contract：

```text
primary: train-selected dev hit@10 must improve over BM25 baseline
secondary: top10 regression count must be lower than improvement count
guard: no query view may be selected by dev-only performance
```

Stage89 实际结果：

```text
primary_contract_passed: false
secondary_contract_passed: false
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Stage90 决策

```text
status: primeqa_hybrid_structured_query_route_stopped
stopped_candidate_id: structured_query_keyphrase_compaction_design
stopped_protocol_id: structured_query_keyphrase_compaction_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: section_signal_guarded_expansion_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

停止原因：

- train-selected structured-query config 明显压缩了 query。
- 但它在 dev 上 `hit@10_delta = -0.0527`。
- dev top10 regression `5` 多于 improvement `1`。
- 这同时违反 primary contract 和 secondary contract。
- 因此 structured query compaction 不能 runtime defaultization，也不能打开 final-test gate。

### Stage84 队列状态

原始 Stage84 执行顺序：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Stage87 已停止：

```text
lexical_cluster_diversity_rerank_design
```

Stage90 已停止：

```text
structured_query_keyphrase_compaction_design
```

Stage90 后剩余队列：

```text
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

下一候选：

```text
section_signal_guarded_expansion_design
```

### Guard checks

全部通过：

```text
22 / 22 passed
```

关键 guard：

- Stage84 报告确认为 Stage84。
- Stage89 报告确认为 Stage89。
- 用户已确认本轮 Stage90 stop decision。
- Stage89 candidate id 和 protocol id 都匹配 structured query route。
- Stage84 metric contract 确实要求 train-selected dev hit@10 improvement。
- Stage89 primary contract failed。
- Stage89 secondary contract failed。
- Stage89 train-selected config 在 dev 上 hit@10 下降。
- Stage89 dev top10 net 为负。
- Stage89 final-test locked。
- Stage89 forbid test tuning。
- Stage89 runtime default unchanged。
- Stage84 队列包含当前停止路线。
- Stage90 后下一候选存在。
- 已停止的 LCDR 没有重新进入 remaining queue。
- `source_doc_ids_oracle_union_blocked` 没有被选为下一候选。
- Stage90 没有跑新 retrieval metrics。
- Stage90 没有跑 final-test metrics。
- Stage90 没有改变 runtime default。

### 可视化产物

```text
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_dev_change_counts.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_stop_decision_flags.svg
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90_visuals\stage90_structured_query_stop_guard_check_status.svg
```

Stage90 JSON SHA256：

```text
E7A5B120B709E0802B352773ECE3561442943CAF6357C2F2075101A1765997B6
```

Visualization SHA256：

```text
stage90_second_wave_remaining_candidate_priority.svg: 246DC08BC22076751014FEC62C5A14DC1EBE3241C85C998CE0EC5B1CCF13AC35
stage90_structured_query_dev_change_counts.svg: B45AF0F1E99202563CDCF085217A592AE55A93222B7F04AC3EF6706546F851B6
stage90_structured_query_stop_decision_flags.svg: 15F5CC5415B8327F8B814441B9040BA13A0440D9C46744E7079E3B2BF6D18E22
stage90_structured_query_stop_guard_check_status.svg: 7B39D1C0E041374C99F8467170CE0AC21BA45A042C2BD69DA4D6CBA31F9111BA
stage90_structured_query_train_dev_hit10_delta.svg: D6A2093E2353AF37EEE5B3B4F228375136CBA7AAD5AB604E22BD6F2C1C98D012
```

### 问题、原因与修正

- 问题：这一步容易被误解成又做了一次实验。
  - 原因：Stage90 也会生成 JSON 和 SVG，但它们来自 Stage84 / Stage89 public-safe summary，不是新 retrieval run。
  - 修正：在代码、文档和 guard check 中都明确 `stage90_no_new_retrieval_metrics_run = not_run`，并且命令只读取 Stage84 / Stage89 报告。
  - 影响：路线决策可复现，但不会污染 train/dev/test 评估边界。

- 问题：停止 structured query 后，需要避免把 Stage87 已停止的 LCDR 又放回队列。
  - 原因：Stage84 原始队列仍包含 LCDR，如果只删除当前路线，会得到错误的 remaining queue。
  - 修正：Stage90 显式维护 prior stopped candidate set，remaining queue 同时排除 LCDR 和 structured query。
  - 影响：下一候选稳定为 `section_signal_guarded_expansion_design`。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_structured_query_stop_decision.py scripts\decide_primeqa_hybrid_structured_query_stop.py tests\test_primeqa_hybrid_structured_query_stop_decision.py
pytest -q tests\test_primeqa_hybrid_structured_query_stop_decision.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term field patterns over Stage90 JSON: no matches
git check-ignore Stage90 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 235 passed
git diff --check: passed
```

### 结论

- Stage90 已停止 structured query keyphrase compaction route。
- Stage90 没有加载 train/dev/test split。
- Stage90 没有跑新 retrieval metrics。
- Stage90 没有跑 final metrics。
- Stage90 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage90 没有写出 raw question / raw answer / document text / query terms。
- Stage90 没有改变 runtime 默认策略。
- 下一候选是 `section_signal_guarded_expansion_design`。

### 我学到了

- 对失败路线的停止也应该是可复现工件，而不是口头结论；否则队列推进和 gate 状态容易在后续阶段丢失。
- stop-decision 阶段不应补跑任何 retrieval 指标，只能使用已完成阶段的 public-safe summary。
- 当候选队列跨多个阶段推进时，需要显式记录 prior stopped candidates，否则可能把已停止路线重新纳入下一步。

### 下一步

Stage91：确认并冻结 `section_signal_guarded_expansion_design` 的 train/dev-only protocol。

Stage91 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## 2026-07-15 - Stage91 section signal guarded expansion protocol freeze

### 目标

本阶段继续按 Stage90 的推荐路线推进，但仍然不是训练或评估阶段：

- 确认并冻结 `section_signal_guarded_expansion_design` 的 train/dev-only protocol。
- 只读取 Stage84 second-wave candidate design 和 Stage90 structured query stop decision 的 public-safe 报告。
- 将 Stage79 的经验转成 guarded section-signal promotion protocol，而不是重复 ungated section BM25 replacement。
- 不加载 train/dev/test split 文件。
- 不运行 retrieval metrics。
- 不运行 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不改变 runtime 默认策略。

### 新增和更新文件

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_section_signal_protocol.py
scripts/freeze_primeqa_hybrid_section_signal_protocol.py
tests/test_primeqa_hybrid_section_signal_protocol.py
docs/primeqa_hybrid_section_signal_protocol.md
```

更新：

```text
docs/primeqa_hybrid_structured_query_stop_decision.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\freeze_primeqa_hybrid_section_signal_protocol.py --user-confirmed-candidate --confirmed-candidate-id section_signal_guarded_expansion_design --confirmation-note "user confirmed Stage91 section-signal protocol freeze in current turn" --output artifacts\primeqa_hybrid_section_signal_protocol_stage91.json --visualization-dir artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals
```

运行耗时：

```text
total: 0.001s
```

### 输入证据

Stage91 只读取 public-safe 报告：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90.json
```

本阶段没有加载 train/dev/test split 文件，没有运行 retrieval metrics，也没有读取原始问题、答案或文档正文。

### 冻结协议

```text
candidate_id: section_signal_guarded_expansion_design
protocol_id: section_signal_guarded_expansion_train_dev_v1
status: primeqa_hybrid_section_signal_protocol_frozen
```

Stage84 target metric contract：

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: search-depth improvements must exceed regressions
guard: section signal must not demote existing BM25 top10 hits by default
```

预声明 config grid：

```text
ssgx_shadow_no_top10_demotion_v1
ssgx_rank11_20_margin_guard_v1
ssgx_rank21_50_high_confidence_v1
ssgx_section_top50_injection_guard_v1
```

协议选择规则：

```text
Select the candidate config on train by hit@10, then search-depth net improvements,
fewer top10 regressions, hit@5, hit@1, MRR@10, lower top10 demotion budget,
then config_id. Dev is validation only.
```

允许的 runtime evidence：

```text
document BM25 rank/score/margins
best section BM25 rank/score
section-to-document score ratio
query/document/section overlap counts
gate state features
```

明确禁止：

```text
source DOC_IDS
answer document IDs
gold document rank
gold labels
test split membership
raw question / answer / document / section text
dev-only config selection
runtime default change
```

### Stage91 决策

```text
status: primeqa_hybrid_section_signal_protocol_frozen
protocol_id: section_signal_guarded_expansion_train_dev_v1
candidate_id: section_signal_guarded_expansion_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Guard checks

全部通过：

```text
26 / 26 passed
```

关键 guard：

- Stage84 报告确认为 Stage84。
- Stage90 报告确认为 Stage90。
- 用户已确认本轮 Stage91 protocol freeze。
- Stage90 已停止 structured query route。
- confirmed candidate 匹配 Stage90 next candidate。
- Stage90 仍要求下一 protocol 前用户确认。
- Stage90 final-test locked。
- Stage90 forbid test tuning。
- Stage90 runtime default unchanged。
- Stage84 candidate 仍是 recommended_for_train_dev_protocol_design。
- Stage84 metric contract 要求 dev hit@10 improvement。
- Stage84 metric contract 要求 search-depth improvements exceed regressions。
- Stage84 guard 要求不默认 demote existing BM25 top10 hits。
- protocol id 固定。
- config grid 预声明且包含 4 个 config。
- section signal contract 只使用 runtime section/document scores。
- promotion configs 都有 top10 promotion budget 和 protected top ranks。
- source `DOC_IDS` 被禁止作为 runtime feature。
- report fields 只包含 public-safe ids、rank、bucket 和 count。
- Stage91 没有运行 retrieval metrics。
- Stage91 没有运行 final-test metrics。
- Stage91 没有改变 runtime default。

### 可视化产物

```text
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_config_promotion_budgets.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_config_ratio_thresholds.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_feature_group_counts.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_protocol_decision_flags.svg
artifacts\primeqa_hybrid_section_signal_protocol_stage91_visuals\stage91_section_signal_guard_check_status.svg
```

Stage91 JSON SHA256：

```text
41FC88ACD104042BBA8B9230477856D06359BC8A8F00DBC617564EAFA9707EA9
```

Visualization SHA256：

```text
stage91_section_signal_config_promotion_budgets.svg: 1BEDB21A2F540E26A1DDB84641AC4A0AC9C87BBC0861CEB46662C73110C1F8BC
stage91_section_signal_config_ratio_thresholds.svg: 9BB1471F36CFBBBA901D92B82714CC2F3BEB9F78C9844E80C8315F3027F9E1DC
stage91_section_signal_feature_group_counts.svg: 63EAB34291809D593A6B6AB7AB85A1073504E0C70F4B4C1E88264C55EC636D4F
stage91_section_signal_guard_check_status.svg: C5FB2D262B1330B937F2D6A1AF8973A0A996DDEFD8A10754C7E888CE86BD3234
stage91_section_signal_protocol_decision_flags.svg: 03AF6BFD684954FF00D2421E90E188AD57ED39A3AC95C62638AF3F37B55C6D5C
```

### 问题、原因与修正

- 问题：第一版安全扫描命中了 `answer_doc_id`。
  - 原因：命中位置是 `prohibited_runtime_features` 中的禁用字段名，不是实际样本字段或泄漏数据。
  - 修正：把禁用项从字段式 `answer_doc_id` 改成描述式 `answer document IDs`，重新生成 Stage91 JSON 和 SVG，并重新计算哈希。
  - 影响：安全扫描不再误报；禁用语义保持不变。

- 问题：section signal 路线如果直接复用 Stage79 ungated section BM25，会重复已失败路线。
  - 原因：Stage79 的结论是 ungated max-section rollup 整体回退，只能作为少量 rescue 信号来源。
  - 修正：Stage91 冻结的是 guarded expansion protocol：baseline 仍是 full-document BM25，只允许预声明 gate 做有限 promotion，并记录 top10 protection。
  - 影响：Stage92 可以检验 section signal 是否在不大幅破坏 BM25 top10 的前提下改善召回。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_protocol.py scripts\freeze_primeqa_hybrid_section_signal_protocol.py tests\test_primeqa_hybrid_section_signal_protocol.py
pytest -q tests\test_primeqa_hybrid_section_signal_protocol.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage91 JSON: no matches
git check-ignore Stage91 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 238 passed
git diff --check: passed
```

### 结论

- Stage91 已冻结 `section_signal_guarded_expansion_train_dev_v1`。
- Stage91 没有加载 train/dev/test split。
- Stage91 没有运行 retrieval metrics。
- Stage91 没有运行 final metrics。
- Stage91 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage91 没有写出 raw question / raw answer / document text / section text。
- Stage91 没有改变 runtime 默认策略。
- Stage92 可以在用户确认后运行 frozen train/dev-only comparison。

### 我学到了

- section BM25 不能因为曾经 rescue 少量样本就直接进入 replacement；已知整体回退的路线只能作为 gated signal 来源重新设计。
- protocol freeze 阶段也需要安全扫描，因为“禁用字段名”本身可能造成误报，最好用描述性禁用项避免和实际泄漏字段混淆。
- 对 top10 召回问题，promotion budget 和 protected top ranks 必须写进协议，否则后续实验容易为了 rescue 深排样本而破坏已有 top10 命中。

### 下一步

Stage92：在用户确认后，运行 frozen train/dev-only section signal guarded expansion comparison。

Stage92 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## 2026-07-15 - Stage92 section signal guarded expansion train/dev comparison

### 目标

本阶段在用户确认后，运行 Stage91 冻结的 train/dev-only comparison：

- 使用 `section_signal_guarded_expansion_train_dev_v1`。
- 只读取 Stage68 train/dev split 和 PrimeQA training/dev document-section 文件。
- 只用 runtime 可见的 full-document BM25 分数、section BM25 分数和 gate state。
- train-only selection，dev 只做 validation。
- 不读取 test。
- 不跑 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不写出 raw question / raw answer / document text / section text。
- 不改变 runtime 默认策略。

### 新增和更新文件

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_section_signal_comparison.py
scripts/run_primeqa_hybrid_section_signal_comparison.py
tests/test_primeqa_hybrid_section_signal_comparison.py
docs/primeqa_hybrid_section_signal_comparison.md
```

更新：

```text
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_structured_query_stop_decision.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/learning_journal.md
```

### 真实运行命令

```text
python scripts\run_primeqa_hybrid_section_signal_comparison.py --user-confirmed-protocol --confirmed-protocol-id section_signal_guarded_expansion_train_dev_v1 --confirmation-note "user confirmed Stage92 train/dev metric run in current turn" --output artifacts\primeqa_hybrid_section_signal_comparison_stage92.json --visualization-dir artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals
```

运行耗时：

```text
total: 69.522s
load_splits: 0.011s
load_documents_sections_and_reports: 2.528s
bm25_section_signal_evaluate: 66.978s
guard_checks: 0.006s
```

### 输入数据

```text
document_count: 28482
section_count: 216648
documents_with_sections: 28482
documents_without_sections: 0
average_sections_per_document: 7.6065
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

### Frozen config grid

```text
ssgx_shadow_no_top10_demotion_v1
ssgx_rank11_20_margin_guard_v1
ssgx_rank21_50_high_confidence_v1
ssgx_section_top50_injection_guard_v1
```

Train selection rule：

```text
Select the section signal config on train only by hit@10, then search-depth net improvements,
then fewer top10 regressions, then hit@5, hit@1, MRR@10, lower top10 demotion budget,
then config_id; dev is validation only.
```

### Train metrics

Baseline：

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
  rank_11_to_50_count: 32
```

Train-selected config：

```text
ssgx_section_top50_injection_guard_v1:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5063
  not_found@50: 92
  rank_11_to_50_count: 33
  section_signal_promotion_count: 92
  protected_top10_demotion_count: 92
```

Train-selected comparison：

```text
hit@10_delta: +0.0000
MRR@50_delta: +0.0002
top10_improvement_count: 1
top10_regression_count: 1
top10_net_improvement_count: 0
search_depth_improvement_count: 1
search_depth_regression_count: 0
search_depth_net_improvement_count: +1
not_found_count_at_50_delta: -1
rank_11_to_50_count_delta: +1
```

### Dev metrics

Baseline：

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR@10: 0.5331
  MRR@50: 0.5365
  not_found@50: 17
  rank_11_to_50_count: 6
```

All frozen configs on dev：

```text
ssgx_shadow_no_top10_demotion_v1: hit@10 0.6974, delta +0.0000
ssgx_rank11_20_margin_guard_v1: hit@10 0.6974, delta +0.0000
ssgx_rank21_50_high_confidence_v1: hit@10 0.6974, delta +0.0000
ssgx_section_top50_injection_guard_v1: hit@10 0.6974, delta +0.0000
```

Train-selected dev comparison：

```text
selected_config_id: ssgx_section_top50_injection_guard_v1
hit@10_delta: +0.0000
top10_improvement_count: 0
top10_regression_count: 0
top10_net_improvement_count: 0
search_depth_improvement_count: 0
search_depth_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
rank_11_to_50_count_delta: 0
section_signal_promotion_count: 22
protected_top10_demotion_count: 22
```

### Stage92 决策

```text
status: primeqa_hybrid_section_signal_comparison_completed
selected_config_id: ssgx_section_top50_injection_guard_v1
selected_dev_hit10_delta: 0.0000
selected_dev_search_depth_net_improvement_count: 0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_not_found_at_search_depth_delta: 0
selected_dev_section_signal_promotion_count: 22
selected_dev_protected_top10_demotion_count: 22
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Guard checks

全部通过：

```text
20 / 20 passed
```

关键 guard：

- analysis splits 只有 train/dev。
- top_k values 包含 primary top10。
- search_depth 覆盖 primary top10。
- Stage91 report 确认为 Stage91。
- Stage91 protocol id 和 candidate id 匹配。
- 用户已确认 frozen protocol。
- confirmed protocol id 匹配。
- Stage91 允许用户确认后运行 train/dev metrics。
- Stage91 final-test locked。
- Stage91 forbid test tuning。
- Stage91 runtime default unchanged。
- candidate config grid 匹配 frozen protocol。
- section index 非空。
- baseline train/dev hit@10 与 Stage75 一致。
- source `DOC_IDS` 没有作为 runtime evidence 使用。
- changed-case fields public-safe。
- final metrics not run。
- runtime default unchanged。

### 可视化产物

```text
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_train_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_search_depth_net.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_dev_top10_changes.svg
artifacts\primeqa_hybrid_section_signal_comparison_stage92_visuals\stage92_section_signal_guard_check_status.svg
```

Stage92 JSON SHA256：

```text
9D2DCA0C4FA2A34320DC3700CB24CAB06088E2670107F8D6A8C932817640E338
```

Visualization SHA256：

```text
stage92_section_signal_dev_delta_hit_at_10.svg: B3AE28DC10167253C7E132DB2FF48764408277FB00CA5C78A053EDEB46885E54
stage92_section_signal_dev_hit_at_10.svg: C1CDAE269291D1FC2BC1760B3A06BD5E34A7E56A2C2D1ECE495E673CE39E5F5A
stage92_section_signal_dev_search_depth_net.svg: FE45CF43AED44CFF296A94369C3D7DE2CF7D8EA3DD07E04627BAFF0D76F22D7E
stage92_section_signal_dev_top10_changes.svg: 106CE640DF79DA1C93AECA461A1CFE9939C1608B90C9A7C84ED120DE4CB53E2E
stage92_section_signal_guard_check_status.svg: 08B2F17B19C731B8E4E9B062C5DE1A0724631DF18677492B4F7BB5D856C65E7C
stage92_section_signal_train_hit_at_10.svg: 800805855133EFAFD2625FF94F3ABC1B4D858C4D000E7D2D152C70935FA47127
```

### 问题、原因与修正

- 问题 1：第一版 Stage92 report 的 `protected_top10_demotion_count` 是按答案文档的 signal evidence 计数。
  - 原因：记录 rank 时只把 answer doc 的 signal evidence 传入 accumulator，无法准确表示 query-level gate action。
  - 修正：`_apply_section_signal_gate` 改为返回 query-level `promotion_applied` 和 `top10_demotion_applied` summary；accumulator 按 query action 计数；随后重新运行 Stage92 并重新计算哈希。
  - 影响：`section_signal_promotion_count` 和 `protected_top10_demotion_count` 现在表示该 config 在多少个 query 上实际执行了对应 gate action，而不是只看 answer doc。

- 问题 2：Stage91 文档中 `ruff check .` 一行残留 `pending final Stage91 validation`。
  - 原因：Stage91 结束前 learning journal 已回填真实结果，但独立 Stage91 protocol doc 的这一行漏改。
  - 修正：本阶段将该行更正为 `ruff check .: passed`。
  - 影响：这是 Stage92 期间发生的记录修正，不伪装成 Stage91 当时已经写对。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_comparison.py scripts\run_primeqa_hybrid_section_signal_comparison.py tests\test_primeqa_hybrid_section_signal_comparison.py
pytest -q tests\test_primeqa_hybrid_section_signal_comparison.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage92 JSON: no matches
git check-ignore Stage92 JSON, console, and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 241 passed
git diff --check: passed
```

### 结论

- Stage92 已运行 frozen section signal train/dev-only comparison。
- Stage92 没有读取 test。
- Stage92 没有运行 final metrics。
- Stage92 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage92 没有写出 raw question / answer / document / section text。
- Stage92 没有改变 runtime 默认策略。
- section signal injection 在 train 上有极小 search-depth 正信号，但同时有 top10 regression。
- train-selected config 在 dev 上没有任何 hit@10 或 search-depth 改善。
- primary contract 和 secondary contract 都没有通过。
- 该路线不能进入 runtime，也不能打开 final-test gate。

### 我学到了

- 大量 gated promotion 不等于有效召回：dev 上执行了 22 次 rank10 promotion，但对 gold doc 的 top10 和 search-depth 都没有改善。
- query-level action count 和 answer-doc-level changed case 是两种不同口径，必须分开记录，否则会误导对 gate 风险的判断。
- Stage79 的 section BM25 小信号在 guarded promotion 后仍然没有泛化到 dev，说明当前 top10 blocker 可能不只是“section 分数被 full-document 分数淹没”的问题。

### 下一步

Stage93：停止 section signal guarded expansion 作为 retrieval-recall route，除非用户明确确认一个新的 train/dev-only protocol；推荐转向下一个 second-wave candidate protocol。

Stage93 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage93：停止 section signal guarded expansion 路线，并确认下一候选

### 目标

完成 Stage92 之后的路线停止决策：

- 不再继续把 `section_signal_guarded_expansion_design` 作为 retrieval-recall runtime 候选推进。
- 明确该路线不能 defaultize，不能打开 final-test gate。
- 从 Stage84 second-wave 队列中移除已经停止的路线。
- 确认下一条可推进候选为 `score_margin_bm25_normalization_gate_design`。
- 继续保持 test split 锁定，不运行 final metrics。
- 不使用 source `DOC_IDS` 作为 runtime 检索证据。
- 不改变 runtime 默认策略。
- 为停止决策生成可视化和完整记录。

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_section_signal_stop_decision.py
scripts/decide_primeqa_hybrid_section_signal_stop.py
tests/test_primeqa_hybrid_section_signal_stop_decision.py
docs/primeqa_hybrid_section_signal_stop_decision.md
```

### 更新文件

```text
docs/learning_journal.md
docs/primeqa_hybrid_section_signal_comparison.md
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_structured_query_stop_decision.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

### 执行命令

```text
python scripts\decide_primeqa_hybrid_section_signal_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage93 stop decision in current turn" --output artifacts\primeqa_hybrid_section_signal_stop_decision_stage93.json --visualization-dir artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals
```

结果：

```text
stage: Stage 93
route_id: section_signal_stop_decision
confirmed: true
timing total: 0.007s
```

### 输入证据

Stage93 只读取公开安全报告：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_section_signal_comparison_stage92.json
```

本阶段没有读取 frozen test split，没有读取 train/dev split 明细，没有运行新 retrieval metrics，也没有运行 final metrics。

### Stage92 依据

Stage92 train-selected config：

```text
ssgx_section_top50_injection_guard_v1
```

Train evidence：

```text
train hit@10 delta: +0.0000
train search-depth net improvement: +1
train top10 improvements: 1
train top10 regressions: 1
train section-signal promotions: 92
```

Dev evidence：

```text
dev hit@10 delta: +0.0000
dev search-depth net improvement: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev not-found@50 delta: 0
dev rank 11-50 delta: 0
dev section-signal promotions: 22
dev protected top10 demotion actions: 22
```

Stage84/Stage91 目标契约：

```text
primary: dev hit@10 must improve over BM25 baseline
secondary: search-depth improvements must exceed regressions
guard: section signal must not demote existing BM25 top10 hits by default
```

### 决策

```text
status: primeqa_hybrid_section_signal_route_stopped
stopped_candidate_id: section_signal_guarded_expansion_design
stopped_protocol_id: section_signal_guarded_expansion_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: score_margin_bm25_normalization_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

停止原因：

- section-signal route 在 train 上确实有 92 次 promotion action，但 train hit@10 没有提升。
- train search-depth net 只有 `+1`，同时有 `1` 个 top10 improvement 和 `1` 个 top10 regression。
- dev 上虽然有 22 次 promotion action，但 hit@10、search-depth、not-found@50、rank 11-50 都没有改善。
- primary contract 和 secondary contract 均未通过。
- guard contract 通过只说明风险边界可控，不说明召回效果足够推进。

### 候选队列

Stage84 原始执行顺序：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

已停止路线：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
```

Stage93 后剩余队列：

```text
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

下一候选：

```text
score_margin_bm25_normalization_gate_design
```

注意：这只是 Stage94 的 protocol-freeze 候选，不是 runtime 策略，也不是 final-test gate。

### Guard checks

全部通过：

```text
26 / 26 passed
```

关键 guard：

- source Stage84 report 确认为 Stage84。
- source Stage92 report 确认为 Stage92。
- 用户已确认 Stage93 stop decision。
- Stage92 comparison 已完成。
- Stage92 candidate/protocol 与 section signal route 匹配。
- Stage84 target metric contract 要求 dev hit@10 gain。
- Stage84 target metric contract 要求 search-depth gain。
- Stage92 primary contract failed。
- Stage92 secondary contract failed。
- Stage92 guard contract passed。
- Stage92 train-selected config 没有 dev hit@10 gain。
- Stage92 dev search-depth net 不为正。
- Stage92 dev top10 net 不为正。
- Stage92 final-test metrics locked。
- Stage92 final-test gate closed。
- Stage92 forbids test tuning。
- Stage92 runtime default unchanged。
- Stage84 execution order 包含 stopped candidate。
- Stage84 next candidate 在停止 section-signal 后可用。
- 已停止的 LCDR 和 structured-query route 已从 remaining queue 中移除。
- `source_doc_ids_oracle_union_blocked` 没有被选为 next candidate。
- Stage93 没有运行新 retrieval metrics。
- Stage93 没有运行 final-test metrics。
- Stage93 runtime default unchanged。

### 可视化产物

```text
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_dev_change_counts.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_stop_decision_flags.svg
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93_visuals\stage93_section_signal_stop_guard_check_status.svg
```

Stage93 JSON SHA256：

```text
F409A99EA7DCF0823C140EFA6C69AED512A9B7BAE23FE63442602C21B59BF90A
```

Visualization SHA256：

```text
stage93_second_wave_remaining_candidate_priority.svg: E29F4679863809EEC1969A0D079BCABF0169A16F44AC1B6F93BDCE655C0A3332
stage93_section_signal_dev_change_counts.svg: 883D57F772EDFA60FFFE1E4C1471ACA946D6490C695EE2DCCD47DDD67EBBA481
stage93_section_signal_stop_decision_flags.svg: 4CE969467572C3EC9A63C8F607C8E25D736E93F6AE1DFA13447BD010C1E67E23
stage93_section_signal_stop_guard_check_status.svg: 2A3FCCEDE15EB338F6C0289451644A07D02C397ADAE8B64141DF08A0B9F338F3
stage93_section_signal_train_dev_hit10_delta.svg: DA5152C754A0246D8C0316440574A97EFFACEC17FF3C7AD0B38050AA7C6FA120
```

### 问题、原因与修正

- 问题 1：第一版补丁输出过长，终端显示被截断。
  - 原因：新增模块、脚本、测试一次性 patch 内容较多。
  - 修正：先用 `Test-Path` 和定向 ruff/pytest 确认三个新增文件完整落盘，再继续运行 Stage93。
  - 影响：没有发现文件缺失或语法问题。

- 问题 2：读取文档局部区间时，PowerShell 的 `Select-Object -Index 540..570` 写法报错。
  - 原因：`-Index` 需要整数数组，范围表达式要加括号。
  - 修正：改为 `Select-Object -Index (540..570)`。
  - 影响：只影响读取上下文，不影响代码或数据产物。

- 问题 3：`docs/primeqa_hybrid_structured_query_comparison.md` 仍残留 “Stage93 是下一步”。
  - 原因：这是更早阶段文档的历史 next-step 指针。
  - 修正：同步更新为 Stage93 已停止 section signal，当前下一步为 Stage94。
  - 影响：跨阶段文档指针保持一致。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_section_signal_stop_decision.py scripts\decide_primeqa_hybrid_section_signal_stop.py tests\test_primeqa_hybrid_section_signal_stop_decision.py
pytest -q tests\test_primeqa_hybrid_section_signal_stop_decision.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage93 JSON: no matches
git check-ignore Stage93 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 244 passed
git diff --check: passed
```

### 结论

- Stage93 已停止 `section_signal_guarded_expansion_design`。
- Stage93 没有读取 test。
- Stage93 没有运行 retrieval metrics。
- Stage93 没有运行 final metrics。
- Stage93 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage93 没有写出 raw question / answer / document / section text。
- Stage93 没有改变 runtime 默认策略。
- 当前可推进候选是 `score_margin_bm25_normalization_gate_design`。

### 我学到了

- gated promotion action 很多不代表能改善 gold-doc recall；Stage92 dev 上 22 次 action 没有换来任何 hit@10 或 search-depth 改善。
- 一个 route 的 guard 通过只能说明“没有越界推进”，不能替代 primary/secondary metric contract。
- second-wave 队列推进需要显式 stop decision，否则旧路线会反复占据下一步。

### 下一步

Stage94：确认并冻结 `score_margin_bm25_normalization_gate_design` 的 train/dev-only protocol。

Stage94 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能用 dev-only 观察选择 runtime threshold，不能改变 runtime 默认策略。

## Stage94：冻结 score-margin BM25 normalization gate protocol

### 目标

完成 Stage93 之后的下一候选 protocol-freeze：

- 确认 `score_margin_bm25_normalization_gate_design` 是当前 Stage84 队列的下一候选。
- 冻结一个 train/dev-only protocol，用于后续 Stage95 指标比较。
- 明确 Stage82 的 dev-only `b=0.95` 观察只能作为设计动机，不能选择 runtime rule。
- 明确 allowed runtime evidence 只能来自 BM25 score、rank、document length、gate state 等 runtime 特征。
- 禁止 source `DOC_IDS`、answer document IDs、gold rank、dev-selected config、test membership 进入 runtime features。
- 不加载 test，不运行 retrieval metrics，不运行 final metrics，不改变 runtime 默认策略。
- 生成可视化并完整记录。

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_score_margin_bm25_protocol.py
scripts/freeze_primeqa_hybrid_score_margin_bm25_protocol.py
tests/test_primeqa_hybrid_score_margin_bm25_protocol.py
docs/primeqa_hybrid_score_margin_bm25_protocol.md
```

### 更新文件

```text
docs/learning_journal.md
docs/primeqa_hybrid_section_signal_stop_decision.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/primeqa_hybrid_section_signal_comparison.md
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_structured_query_stop_decision.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

### 执行命令

```text
python scripts\freeze_primeqa_hybrid_score_margin_bm25_protocol.py --user-confirmed-candidate --confirmed-candidate-id score_margin_bm25_normalization_gate_design --confirmation-note "user confirmed Stage94 score-margin BM25 protocol freeze in current turn" --output artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94.json --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals
```

结果：

```text
stage: Stage 94
confirmed_candidate_id: score_margin_bm25_normalization_gate_design
protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
timing total: 0.000s
```

### 输入证据

Stage94 只读取公开安全报告：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93.json
```

本阶段没有读取 frozen train/dev/test split 明细，没有运行 retrieval metrics，也没有运行 final metrics。

### Stage84 候选契约

```text
candidate_id: score_margin_bm25_normalization_gate_design
category: lexical_parameter_gate
risk_level: medium
implementation_readiness: 0.70
priority_score: 171
target_miss_count: 111
target_miss_count_by_split:
  train: 93
  dev: 18
```

目标 rank bucket：

```text
not_found_top50: 73
rank_11_to_20: 14
rank_21_to_50: 24
```

Stage84 target metric contract：

```text
primary: train-selected rule must improve dev hit@10
secondary: rank 11-50 near misses should decrease
guard: dev-only b=0.95 observations cannot select a runtime rule
```

### Frozen protocol

```text
status: primeqa_hybrid_score_margin_bm25_protocol_frozen
protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
candidate_id: score_margin_bm25_normalization_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Candidate config grid

```text
smbn_rank11_20_long_doc_b095_margin_v1:
  challenger BM25: k1=1.5, b=0.95
  eligible baseline rank: 11-20
  challenger rank max: 10
  max score margin to rank10: 0.05
  length gate: long_document_only, min length ratio 1.20
  max top10 promotions per query: 1

smbn_rank21_50_long_doc_b095_high_confidence_v1:
  challenger BM25: k1=1.5, b=0.95
  eligible baseline rank: 21-50
  challenger rank max: 15
  max score margin to rank10: 0.03
  length gate: long_document_only, min length ratio 1.50
  max top10 promotions per query: 1

smbn_rank11_20_short_doc_b055_margin_v1:
  challenger BM25: k1=1.5, b=0.55
  eligible baseline rank: 11-20
  challenger rank max: 10
  max score margin to rank10: 0.04
  length gate: short_document_only, max length ratio 0.85
  max top10 promotions per query: 1

smbn_rank11_50_dual_length_band_margin_v1:
  challenger BM25: k1=1.5, b=0.55 for short docs and b=0.95 for long docs
  eligible baseline rank: 11-50
  challenger rank max: 12
  max score margin to rank10: 0.02
  length gate: outside_length_band_short_or_long
  branch rule: use b=0.55 when length ratio <= 0.75; use b=0.95 when length ratio >= 1.35
  max top10 promotions per query: 1
```

Train selection rule：

```text
Select the candidate config on train only by hit@10, then fewer rank 11-50 near misses,
fewer top10 regressions, hit@5, hit@1, MRR@10, lower top10 promotion budget,
then config_id. Dev is validation only.
```

### Runtime evidence contract

允许特征：

- baseline BM25 rank / score / score margin。
- challenger BM25 rank / score / score margin。
- document token count / average document token count / length ratio / length bucket。
- gate eligibility bucket / challenger rank bucket / promotion budget / reason code。

禁止特征：

- source `DOC_IDS`。
- answer document IDs。
- gold rank / gold label。
- dev-selected config。
- Stage82 dev-selected b value。
- frozen test split membership。
- raw question / answer / document text。
- raw document title。
- query terms。
- matched token strings。

### Guard checks

全部通过：

```text
31 / 31 passed
```

关键 guard：

- Stage84 report 确认为 Stage84。
- Stage93 report 确认为 Stage93。
- 用户已确认 score-margin protocol。
- Stage93 已停止 section-signal route。
- confirmed candidate 匹配 Stage93 next candidate。
- Stage93 要求 next protocol 前用户确认。
- Stage93 final-test locked / gate closed / forbid test tuning。
- Stage84 final-test locked / forbid test tuning / runtime unchanged。
- Stage84 candidate status 为 recommended_for_train_dev_protocol_design。
- Stage84 contract 要求 train-selected dev hit@10 gain。
- Stage84 contract 要求 rank 11-50 near misses decrease。
- Stage84 guard 明确禁止 dev-only b=0.95 选择 runtime rule。
- protocol id 固定。
- candidate config grid 固定且包含 4 个 predeclared configs。
- 每个 config 都包含 rank、score-margin、length、promotion gate。
- train selection rule 禁止 dev/test selection。
- Stage82 historical signal 只作为 motivation。
- score-margin feature contract 只允许 runtime score/rank/length features。
- source `DOC_IDS` 和 answer document IDs 禁止进入 runtime features。
- changed-case fields public-safe。
- `source_doc_ids_oracle_union_blocked` 没有被选择。
- Stage94 不运行 metrics。
- Stage94 不运行 final-test metrics。
- Stage94 runtime default unchanged。

### 可视化产物

```text
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_config_b_values.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_margin_thresholds.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_length_thresholds.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_feature_group_counts.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_protocol_decision_flags.svg
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94_visuals\stage94_score_margin_bm25_guard_check_status.svg
```

Stage94 JSON SHA256：

```text
FB6E8B0E8EDBA4BBB8C9DD6E684F4D807AE26CD91A658D3B0AA1F694B202F2E8
```

Visualization SHA256：

```text
stage94_score_margin_bm25_config_b_values.svg: 26849CE3714FD3FF04401D0127546E9D476244E80D8FBD1C537045724A0570FD
stage94_score_margin_bm25_feature_group_counts.svg: B655D1A429489F430647550168428BB8B2D27551B72613C9F1AB2DE13F1B5187
stage94_score_margin_bm25_guard_check_status.svg: 724AF26B2A06F6AB26E93C1CAA839CD058CE5CAB962528028355EB80CB56E6E5
stage94_score_margin_bm25_length_thresholds.svg: 8F32B109D1BC082A314AE2EA76AB58B6139CB94F0BC9B77D0A160E743B1DC09C
stage94_score_margin_bm25_margin_thresholds.svg: 660DEFEF47832B9768CC37212C4E5950E9FA95384CCCED601F7BFE4CADE18A2F
stage94_score_margin_bm25_protocol_decision_flags.svg: 2F6DE073A1F41B240BF4890EFDD5713A103A011088F581920B4385BBE3093B72
```

### 问题、原因与修正

- 问题 1：初版代码有 3 行 guard 表达式超过 ruff 行长限制。
  - 原因：`any(...)` 表达式直接写在一行。
  - 修正：拆成多行生成器表达式。
  - 影响：逻辑不变，定向 ruff 随后通过。

- 问题 2：dual-length config 的短文档/长文档分支一开始只写了 min/max length ratio，容易被误读成两个条件必须同时满足。
  - 原因：字段缺少 `length_gate_mode` 和 branch rule。
  - 修正：为所有 configs 增加 `length_gate_mode`；为 dual config 增加 `normalization_branch_rule`，明确 `<= 0.75` 用 `b=0.55`，`>= 1.35` 用 `b=0.95`。
  - 影响：重新运行 Stage94，重新生成 JSON 和 SVG hashes。

- 问题 3：dual config 的 b-value 可视化初版用 `0.75` 作为字符串型 b value 的数值条长度。
  - 原因：为了能渲染 mixed label 做了临时 numeric value。
  - 修正：如果字符串中包含 `0.95`，可视化条长使用 `0.95`，label 仍保留完整 branch rule。
  - 影响：重新运行 Stage94 后，可视化语义更清楚。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_protocol.py scripts\freeze_primeqa_hybrid_score_margin_bm25_protocol.py tests\test_primeqa_hybrid_score_margin_bm25_protocol.py
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_protocol.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage94 JSON: no matches
git check-ignore Stage94 JSON and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 247 passed
git diff --check: passed
```

### 结论

- Stage94 已冻结 `score_margin_bm25_normalization_gate_train_dev_v1`。
- Stage94 没有读取 test。
- Stage94 没有运行 retrieval metrics。
- Stage94 没有运行 final metrics。
- Stage94 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage94 没有使用 Stage82 dev-only `b=0.95` observation 选择 runtime rule。
- Stage94 没有写出 raw question / answer / document / query terms。
- Stage94 没有改变 runtime 默认策略。

### 我学到了

- Stage82 的 dev-only 信号不能直接转成参数选择，但可以转成 train-only protocol 的设计动机。
- score-margin BM25 gate 必须把 rank gate、score-margin gate、length gate 和 promotion budget 一起冻结，否则下一阶段容易出现隐性调参。
- mixed length-band rule 必须明确是 short/long 分支的 OR，而不是 min/max threshold 的 AND。

### 下一步

Stage95：在用户确认后，运行 frozen train/dev-only `score_margin_bm25_normalization_gate_train_dev_v1` comparison。

Stage95 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能用 dev-only 观察选择 runtime rule，不能改变 runtime 默认策略。

## Stage95：运行 score-margin BM25 normalization gate train/dev comparison

### 目标

完成 Stage94 frozen protocol 的 train/dev-only 指标比较：

- 使用 `score_margin_bm25_normalization_gate_train_dev_v1`。
- 在 train 上按 frozen selection rule 选择 config。
- 在 dev 上只做 validation，不用 dev 选择 config。
- 验证 train-selected config 是否提升 dev hit@10。
- 验证 train-selected config 是否减少 dev rank 11-50 near misses。
- 确认 Stage82 dev-only `b=0.95` 观察没有被用作 runtime rule selection。
- 不读取 test，不运行 final metrics，不使用 source `DOC_IDS`，不改变 runtime 默认策略。
- 生成可视化并完整记录。

### 新增文件

```text
src/ts_rag_agent/application/primeqa_hybrid_score_margin_bm25_comparison.py
scripts/run_primeqa_hybrid_score_margin_bm25_comparison.py
tests/test_primeqa_hybrid_score_margin_bm25_comparison.py
docs/primeqa_hybrid_score_margin_bm25_comparison.md
```

### 更新文件

```text
docs/learning_journal.md
docs/primeqa_hybrid_score_margin_bm25_protocol.md
docs/primeqa_hybrid_section_signal_stop_decision.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/primeqa_hybrid_section_signal_comparison.md
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_structured_query_stop_decision.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

### 执行命令

```text
python scripts\run_primeqa_hybrid_score_margin_bm25_comparison.py --user-confirmed-protocol --confirmed-protocol-id score_margin_bm25_normalization_gate_train_dev_v1 --confirmation-note "user confirmed Stage95 train/dev metric run in current turn" --output artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95.json --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals *> artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95.console.txt
```

结果：

```text
stage: Stage 95
confirmed_protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
shell timing: 87.7s
```

### 输入数据

```text
document_count: 28482
average_document_token_count: 475.7771
train rows: 562
train answerable rows: 370
dev rows: 121
dev answerable rows: 76
test_split_loaded: false
```

Stage95 只读取：

```text
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
data\raw\primeqa\TechQA\training_and_dev\training_dev_technotes.sections.json
artifacts\primeqa_hybrid_bm25_top10_miss_analysis_stage75.json
artifacts\primeqa_hybrid_score_margin_bm25_protocol_stage94.json
```

没有读取 frozen test split。

### Frozen configs

```text
smbn_rank11_20_long_doc_b095_margin_v1
smbn_rank21_50_long_doc_b095_high_confidence_v1
smbn_rank11_20_short_doc_b055_margin_v1
smbn_rank11_50_dual_length_band_margin_v1
```

实现说明：

- Numeric configs 先计算全 corpus challenger BM25 view，再按 frozen length gate 和 score-margin gate 判断 promotion eligibility。
- Dual length-band config 使用 frozen branch rule：`<= 0.75` 用 `b=0.55`，`>= 1.35` 用 `b=0.95`。
- 中间 length band 不作为 promotion eligible branch；未满足 frozen gate 的候选保持 baseline BM25 顺序。
- 每个 config 每个 query 最多 promotion 1 个 rank 11-50 候选到 rank10。

### Train metrics

Baseline：

```text
full_document_bm25_baseline:
  hit@1: 0.4243
  hit@5: 0.6054
  hit@10: 0.6622
  MRR@10: 0.5023
  MRR@50: 0.5061
  not_found@50: 93
  rank_11_to_50_count: 32
```

所有 frozen configs 的 train hit@10 和 rank 11-50 count 都与 baseline 相同：

```text
smbn_rank11_20_long_doc_b095_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 1

smbn_rank21_50_long_doc_b095_high_confidence_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 0

smbn_rank11_20_short_doc_b055_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 4

smbn_rank11_50_dual_length_band_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 4
```

Train-selected config：

```text
smbn_rank11_20_long_doc_b095_margin_v1
```

Train-selected comparison：

```text
hit@10_delta: +0.0000
rank_11_to_50_count_delta: 0
top10_improvement_count: 0
top10_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
score_margin_gate_promotion_count: 1
```

### Dev metrics

Baseline：

```text
full_document_bm25_baseline:
  hit@1: 0.4342
  hit@5: 0.6579
  hit@10: 0.6974
  MRR@10: 0.5331
  MRR@50: 0.5365
  not_found@50: 17
  rank_11_to_50_count: 6
```

所有 frozen configs 的 dev hit@10 和 rank 11-50 count 都与 baseline 相同：

```text
smbn_rank11_20_long_doc_b095_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 0

smbn_rank21_50_long_doc_b095_high_confidence_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 0

smbn_rank11_20_short_doc_b055_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 2

smbn_rank11_50_dual_length_band_margin_v1:
  hit@10_delta: +0.0000
  rank_11_to_50_count_delta: 0
  top10 improvements/regressions: 0 / 0
  gate actions: 1
```

Train-selected dev comparison：

```text
selected_config_id: smbn_rank11_20_long_doc_b095_margin_v1
hit@10_delta: +0.0000
rank_11_to_50_count_delta: 0
top10_improvement_count: 0
top10_regression_count: 0
search_depth_net_improvement_count: 0
not_found_count_at_50_delta: 0
score_margin_gate_promotion_count: 0
length_band_gate_count: 0
```

### Stage95 决策

```text
status: primeqa_hybrid_score_margin_bm25_comparison_completed
selected_config_id: smbn_rank11_20_long_doc_b095_margin_v1
selected_dev_hit10_delta: 0.0000
selected_dev_rank_11_to_50_count_delta: 0
selected_dev_top10_improvements: 0
selected_dev_top10_regressions: 0
selected_dev_score_margin_gate_promotion_count: 0
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Guard checks

全部通过：

```text
23 / 23 passed
```

关键 guard：

- analysis splits 只有 train/dev。
- top_k values 包含 primary top10。
- search_depth 覆盖 primary top10。
- Stage94 report 确认为 Stage94。
- Stage94 protocol id 和 candidate id 匹配。
- 用户已确认 frozen protocol。
- confirmed protocol id 匹配。
- Stage94 允许用户确认后运行 train/dev metrics。
- Stage94 final-test metrics locked。
- Stage94 forbid test tuning。
- Stage94 runtime default unchanged。
- candidate config grid 匹配 frozen protocol。
- Stage94 train selection rule 禁止 dev selection。
- Stage94 train selection rule 禁止 Stage82 dev-only observation selection。
- Stage82 historical signal 仅作为 motivation。
- baseline train/dev hit@10 与 Stage75 一致。
- source `DOC_IDS` 没有作为 runtime evidence 使用。
- answer doc IDs 只作为 evaluation label 使用，不作为 runtime evidence。
- changed-case fields public-safe。
- final metrics not run。
- runtime default unchanged。

### 可视化产物

```text
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_train_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_delta_hit_at_10.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_top10_changes.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_dev_gate_actions.svg
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95_visuals\stage95_score_margin_bm25_guard_check_status.svg
```

Stage95 JSON SHA256：

```text
B712AFC1197DECD5EB50BB3471A58F0D0902AE35641B3C720AED855BC0CB3237
```

Visualization SHA256：

```text
stage95_score_margin_bm25_dev_delta_hit_at_10.svg: 78522CED909FF4E61C3A44847E204D061F8F218C6A3E586168C859FBBA4F43AF
stage95_score_margin_bm25_dev_gate_actions.svg: D728CA8A8C70D571CD8BE14198B9347D4D1BCCEC1C1B18259116BE096566CC9F
stage95_score_margin_bm25_dev_hit_at_10.svg: F98C235BF5040A7885EE9BDC276745E88064AC20D0E0670215B9B358583C68FD
stage95_score_margin_bm25_dev_rank_11_to_50_delta.svg: 62EB74B362A245EC44E151F95602AA470CA1CB9A048BDC8AEEA0E44C51B471E4
stage95_score_margin_bm25_dev_top10_changes.svg: 6FF10AD6FD8A0C30756DF8164F61AF450D7F84F65BED5FA7CD87ABF1F0C59A56
stage95_score_margin_bm25_guard_check_status.svg: A6010A3B3B3FA954ECB94EC561AE583000C6CF4AD131B24666CCF4A86DEF5DD3
stage95_score_margin_bm25_train_hit_at_10.svg: 646BEF677C7E8C3D023CC876DEC76A84E27CB43BE625D45A56C31AC506263F0C
```

### 问题、原因与修正

- 问题 1：Stage95 需要实现 dual length-band challenger view。
  - 原因：Stage94 frozen protocol 中 dual config 同时包含 short-doc 和 long-doc branch。
  - 修正：实现 `mixed_length_band_view`，短文档分支使用 `b=0.55`，长文档分支使用 `b=0.95`，中间段候选不满足 frozen length gate，保持 baseline order。
  - 影响：没有引入新的 threshold 或 dev-only 选择。

- 问题 2：正式运行 console 输出较大。
  - 原因：CLI 会输出 metrics、comparisons、guards 和 decision。
  - 修正：将 console 输出保存到 `.console.txt` artifact，并用结构化 JSON 读取关键结果。
  - 影响：artifact 被 `.gitignore` 忽略，不进入 git。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_comparison.py scripts\run_primeqa_hybrid_score_margin_bm25_comparison.py tests\test_primeqa_hybrid_score_margin_bm25_comparison.py
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_comparison.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw question / answer / document / snippet / query-term / section-text field patterns over Stage95 JSON: no matches
git check-ignore Stage95 JSON, console, and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 250 passed
git diff --check: passed
```

### 结论

- Stage95 已运行 frozen score-margin BM25 normalization gate train/dev comparison。
- Stage95 没有读取 test。
- Stage95 没有运行 final metrics。
- Stage95 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage95 没有使用 Stage82 dev-only `b=0.95` observation 选择 runtime rule。
- Stage95 没有写出 raw question / answer / document / query terms。
- Stage95 没有改变 runtime 默认策略。
- train-selected config 在 dev 上没有任何 hit@10 或 rank 11-50 改善。
- primary contract 和 secondary contract 都没有通过。
- 该路线不能进入 runtime，也不能打开 final-test gate。

### 我学到了

- Stage82 的 length-normalization 信号在严格 train-selected gate 下几乎没有转化成可见收益。
- 少量 promotion action 如果没有改变 answer doc 的 top10/search-depth rank，就不能视为有效召回改进。
- 当前 second-wave 中多条“精细 gate”都无法突破 PrimeQA top10 blocker，下一步应该显式停止该路线，而不是继续加阈值。

### 下一步

Stage96：停止 score-margin BM25 normalization 作为 retrieval-recall route，除非用户明确确认一个新的 train/dev-only protocol；推荐转向 Stage84 队列中剩余的 `selective_dense_sparse_low_overlap_gate_design`。

Stage96 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage96：停止 score-margin BM25 normalization route

### 目标

Stage96：把 Stage95 的 train/dev-only comparison 结果正式收束为 stop decision。

这一步只读取 public-safe Stage84 / Stage95 reports，不读取 train/dev/test split 文件，不运行新的 retrieval metrics，不运行 final metrics，不使用 source `DOC_IDS` 作为 runtime 检索证据，不改变 runtime 默认策略。

### 本次修改

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_score_margin_bm25_stop_decision.py
scripts/decide_primeqa_hybrid_score_margin_bm25_stop.py
tests/test_primeqa_hybrid_score_margin_bm25_stop_decision.py
docs/primeqa_hybrid_score_margin_bm25_stop_decision.md
```

更新：

```text
docs/learning_journal.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/primeqa_hybrid_score_margin_bm25_comparison.md
docs/primeqa_hybrid_score_margin_bm25_protocol.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/primeqa_hybrid_section_signal_stop_decision.md
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_section_signal_comparison.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_structured_query_stop_decision.md
```

### 执行命令

```text
python scripts\decide_primeqa_hybrid_score_margin_bm25_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage96 score-margin BM25 stop decision in current turn" --output artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json --visualization-dir artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals *> artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.console.txt
```

### Stage96 使用的数据

Stage96 只读取：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_score_margin_bm25_comparison_stage95.json
```

Stage96 没有读取：

```text
data/processed/primeqa_hybrid_stage68_v1/test.jsonl
```

Stage96 没有运行新的 train/dev retrieval comparison，也没有运行 final-test metrics。

### Stage95 证据

Stage95 train-selected config：

```text
smbn_rank11_20_long_doc_b095_margin_v1
```

Train evidence：

```text
train hit@10 delta: +0.0000
train rank 11-50 delta: 0
train top10 improvements: 0
train top10 regressions: 0
train score-margin gate promotions: 1
train length-band gate count: 1
```

Dev evidence：

```text
dev hit@10 delta: +0.0000
dev rank 11-50 delta: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev rank-up within top10: 0
dev rank-down within top10: 0
dev not-found@50 delta: 0
dev score-margin gate promotions: 0
dev length-band gate count: 0
```

Stage84 / Stage94 contract：

```text
primary: train-selected rule must improve dev hit@10
secondary: rank 11-50 near misses should decrease
guard: dev-only b=0.95 observations cannot select a runtime rule
```

### Stage96 决策

```text
status: primeqa_hybrid_score_margin_bm25_route_stopped
stopped_candidate_id: score_margin_bm25_normalization_gate_design
stopped_protocol_id: score_margin_bm25_normalization_gate_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: selective_dense_sparse_low_overlap_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

原因：

- train-selected config 在 dev 上没有 hit@10 提升。
- train-selected config 在 dev 上没有减少 rank 11-50 near misses。
- train-selected config 在 dev 上没有 score-margin promotion。
- primary contract failed。
- secondary contract failed。
- guard contract passed，但 guard 通过只说明边界安全，不说明该 route 有收益。

### Candidate Queue

Stage84 原始顺序：

```text
lexical_cluster_diversity_rerank_design
structured_query_keyphrase_compaction_design
section_signal_guarded_expansion_design
score_margin_bm25_normalization_gate_design
selective_dense_sparse_low_overlap_gate_design
```

Stage96 停止后剩余：

```text
selective_dense_sparse_low_overlap_gate_design
```

下一候选：

```text
selective_dense_sparse_low_overlap_gate_design
```

Stage96 不冻结该候选的 protocol，也不运行该候选的 metrics；它只是把下一候选明确记录出来。

### Guard Checks

Stage96 guard：

```text
30 / 30 passed
```

关键 guard：

```text
source_stage84_report_is_stage84: passed
source_stage95_report_is_stage95: passed
user_confirmed_stage96_stop_decision: passed
stage95_comparison_completed: passed
stage95_candidate_matches_score_margin_bm25: passed
stage95_protocol_matches_score_margin_bm25: passed
stage84_candidate_metric_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_metric_contract_requires_rank_11_to_50_decrease: passed
stage84_candidate_guard_blocks_dev_only_b95_runtime_selection: passed
stage95_primary_contract_failed: passed
stage95_secondary_contract_failed: passed
stage95_guard_contract_passed: passed
stage95_train_selected_config_has_no_dev_hit10_gain: passed
stage95_dev_rank_11_to_50_not_reduced: passed
stage95_selected_config_has_no_dev_score_margin_promotions: passed
stage96_no_new_retrieval_metrics_run: passed
stage96_final_test_metrics_not_run: passed
stage96_default_runtime_policy_unchanged: passed
```

### 可视化

```text
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_dev_change_counts.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_second_wave_remaining_candidate_priority.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_stop_decision_flags.svg
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96_visuals\stage96_score_margin_bm25_stop_guard_check_status.svg
```

Stage96 JSON SHA256：

```text
1EF1180676209A5A311E2AB22A4745E07E12FBCBCAE594D1774CD9CD58634C87
```

Visualization SHA256：

```text
stage96_score_margin_bm25_dev_change_counts.svg: A8F65CFF117762A7B65F29A61C0177E0785ABEEB677309205A3413E54258C9A1
stage96_score_margin_bm25_stop_decision_flags.svg: 437EEBD0F4508236D974B469BD8CC6ADE37F59EB4AC767230EDFC6248AB505DB
stage96_score_margin_bm25_stop_guard_check_status.svg: D9A2EC65EAE7216D499BB5C7E0343E180BABE1C099354E5D6363BE84DF8C5BCC
stage96_score_margin_bm25_train_dev_hit10_delta.svg: 072D4F52F9DE8F77FF69F8F7E98D5C4CE050A9E6DC339643E4EC81538128CF75
stage96_second_wave_remaining_candidate_priority.svg: 6606310CDC836D6C433542C63C7B94F0F5A73C5F39BA96D8E390218A206B71D5
```

### 问题、原因与修正

- 问题 1：Stage96 是 stop decision，不应继续调 threshold。
  - 原因：Stage95 已经显示 train-selected config 在 dev 上没有任何 hit@10 / rank 11-50 改善。
  - 修正：只做 stop-decision report，明确 `current_route_defaultization: blocked`，不新增候选阈值。
  - 影响：避免在失败 route 上继续局部贪心调参。

- 问题 2：宽泛 `Select-String` 中的 `document` 会命中 policy 文本。
  - 原因：Stage96 JSON 合法包含 `answer document IDs` 这类安全策略描述。
  - 修正：改用更具体的 raw/private field patterns 扫描，例如 `question_text`、`answer_text`、`document_text`、`snippet_text`、`query_terms`、`private_example_strings`。
  - 影响：确认 Stage96 report 没有写出原始题目、答案、文档正文或样例文本，同时避免把安全说明误判为泄漏。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_score_margin_bm25_stop_decision.py scripts\decide_primeqa_hybrid_score_margin_bm25_stop.py tests\test_primeqa_hybrid_score_margin_bm25_stop_decision.py
pytest -q tests\test_primeqa_hybrid_score_margin_bm25_stop_decision.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

产物安全检查：

```text
Select-String raw/private field patterns over Stage96 JSON: no matches
git check-ignore Stage96 JSON, console, and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 253 passed
git diff --check: passed
```

### 结论

- Stage96 已停止 score-margin BM25 normalization route。
- Stage96 没有读取 test。
- Stage96 没有运行 final metrics。
- Stage96 没有运行新的 retrieval metrics。
- Stage96 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage96 没有改变 runtime 默认策略。
- Stage96 没有把 artifacts 纳入 git。
- Stage84 second-wave retrieval queue 只剩 `selective_dense_sparse_low_overlap_gate_design`。

### 我学到了

- score-margin / length-normalization gate 的实际动作太少，且无法移动 answer doc 的 top10/search-depth rank，因此不能继续作为有效召回路线。
- stop decision 需要和 comparison 分离：comparison 负责给出度量，stop decision 负责冻结路线状态和候选队列状态。
- 对 public-safe JSON 做安全扫描时，应该区分“安全策略里的 document ID 描述”和“原始 document text 泄漏”。

### 下一步

Stage97：确认并冻结 `selective_dense_sparse_low_overlap_gate_design` 的 train/dev-only protocol。

Stage97 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能改变 runtime 默认策略。

## Stage97：冻结 selective dense+sparse low-overlap gate protocol

### 目标

Stage97：确认并冻结 Stage84 队列最后一个候选 `selective_dense_sparse_low_overlap_gate_design` 的 train/dev-only protocol。

这一步只读取 public-safe Stage84 / Stage96 / Stage80 / Stage81 reports，不读取 train/dev/test split 文件，不运行 retrieval metrics，不运行 final metrics，不下载模型，不刷新 dense cache，不使用 source `DOC_IDS` 作为 runtime 检索证据，不改变 runtime 默认策略。

### 本次修改

新增：

```text
src/ts_rag_agent/application/primeqa_hybrid_selective_dense_sparse_protocol.py
scripts/freeze_primeqa_hybrid_selective_dense_sparse_protocol.py
tests/test_primeqa_hybrid_selective_dense_sparse_protocol.py
docs/primeqa_hybrid_selective_dense_sparse_protocol.md
```

更新：

```text
docs/learning_journal.md
docs/evaluation_strategy.md
docs/data_strategy.md
docs/primeqa_hybrid_score_margin_bm25_comparison.md
docs/primeqa_hybrid_score_margin_bm25_protocol.md
docs/primeqa_hybrid_score_margin_bm25_stop_decision.md
docs/primeqa_hybrid_second_wave_retrieval_candidate_design.md
docs/primeqa_hybrid_section_signal_stop_decision.md
docs/primeqa_hybrid_section_signal_protocol.md
docs/primeqa_hybrid_section_signal_comparison.md
docs/primeqa_hybrid_structured_query_comparison.md
docs/primeqa_hybrid_structured_query_stop_decision.md
```

### 执行命令

```text
python scripts\freeze_primeqa_hybrid_selective_dense_sparse_protocol.py --user-confirmed-candidate --confirmed-candidate-id selective_dense_sparse_low_overlap_gate_design --confirmation-note "user confirmed Stage97 selective dense sparse protocol freeze in current turn" --output artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97.json --visualization-dir artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals *> artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97.console.txt
```

### Stage97 使用的数据

Stage97 只读取：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json
artifacts\primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json
artifacts\primeqa_hybrid_dense_sparse_rrf_comparison_stage81.json
```

Stage97 没有读取：

```text
data/processed/primeqa_hybrid_stage68_v1/train.jsonl
data/processed/primeqa_hybrid_stage68_v1/dev.jsonl
data/processed/primeqa_hybrid_stage68_v1/test.jsonl
```

Stage97 没有运行新的 train/dev retrieval comparison，也没有运行 final-test metrics。

### 冻结协议

```text
protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
candidate_id: selective_dense_sparse_low_overlap_gate_design
protocol_status: frozen_requires_user_confirmation_before_metric_run
source_stages: Stage84, Stage96, Stage80, Stage81
development_splits: train, dev
forbidden_final_splits: test
```

Baseline：

```text
full_document_bm25_baseline
bm25_k1: 1.5
bm25_b: 0.75
candidate_depth: 50
primary_top_k: 10
```

Dense cache contract：

```text
allowed_cache_source: stage80_compatible_local_dense_caches_only
download_required: false
document_reencoding_allowed: false
query_encoding_mode: local_snapshot_path_with_local_files_only
model_selection_mode: predeclared_grid_then_train_selection_only
```

允许的 dense configs：

```text
dense_sparse_rrf__intfloat_e5_small_v2__512_passage
dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

### Candidate Policy Grid

```text
sdsl_e5_low_overlap_balanced_v1:
  dense_config_id: dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  max BM25 top1 query-overlap ratio: 0.25
  max BM25 top10 mean query-overlap ratio: 0.22
  dense_weight: 1.00
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_minilm_low_overlap_balanced_v1:
  dense_config_id: dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
  max BM25 top1 query-overlap ratio: 0.25
  max BM25 top10 mean query-overlap ratio: 0.22
  dense_weight: 1.00
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_e5_low_overlap_dense_bias_v1:
  dense_config_id: dense_sparse_rrf__intfloat_e5_small_v2__512_passage
  max BM25 top1 query-overlap ratio: 0.20
  max BM25 top10 mean query-overlap ratio: 0.18
  dense_weight: 1.25
  max dense top10 promotions per query: 2
  protected BM25 top ranks: 5

sdsl_minilm_low_overlap_conservative_v1:
  dense_config_id: dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
  max BM25 top1 query-overlap ratio: 0.30
  max BM25 top10 mean query-overlap ratio: 0.25
  dense_weight: 0.85
  max dense top10 promotions per query: 1
  protected BM25 top ranks: 7
```

所有 policy 都要求 dense promotion candidate 必须在 BM25 top10 之外，避免无条件替换 BM25 已有强结果。

### Feature Contract

允许 runtime 使用的特征组：

```text
query_aggregate_features
bm25_lexical_features
sparse_rank_score_features
dense_rank_score_features
rrf_gate_features
action_budget_features
```

明确禁止 runtime 使用：

```text
source_DOC_IDS
answer document IDs
gold_document_rank
gold_label
dev_selected_gate_threshold
dev_selected_dense_model
stage81_dev_selected_config
frozen_test_split_membership
raw_question_text
raw_answer_text
raw_document_text
raw_document_title
query_terms
matched_token_strings
```

### Train Selection Rule

```text
Select the gated dense+sparse policy on train only by hit@10, then larger not-found@50 reduction, fewer top10 regressions, hit@1 non-collapse, MRR@10, lower dense promotion budget, then policy_id. Dev is validation only.
```

Stage98 才能在用户确认后运行 comparison；Stage97 只冻结 protocol。

### Stage97 决策

```text
status: primeqa_hybrid_selective_dense_sparse_protocol_frozen
protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
candidate_id: selective_dense_sparse_low_overlap_gate_design
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_run_train_dev_metrics_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

### Guard Checks

Stage97 guard：

```text
46 / 46 passed
```

关键 guard：

```text
source_stage84_report_is_stage84: passed
source_stage96_report_is_stage96: passed
source_stage80_report_is_stage80: passed
source_stage81_report_is_stage81: passed
user_confirmed_selective_dense_sparse_protocol: passed
stage96_stopped_score_margin_bm25_route: passed
confirmed_candidate_matches_stage96_next_candidate: passed
stage80_can_run_without_download: passed
stage80_compatible_cache_count_at_least_two: passed
stage81_dense_sparse_comparison_completed: passed
stage84_candidate_contract_requires_train_selected_dev_hit10_gain: passed
stage84_candidate_contract_requires_not_found_decrease: passed
stage84_candidate_guard_blocks_downloads_and_dev_thresholds: passed
dense_cache_contract_uses_stage80_and_stage81_only: passed
stage80_stage81_dense_cache_identities_match: passed
candidate_policy_grid_is_predeclared: passed
candidate_policy_grid_reuses_existing_dense_configs: passed
candidate_policy_grid_has_low_overlap_gates: passed
train_selection_rule_forbids_dev_and_test_selection: passed
source_doc_ids_forbidden_in_runtime_features: passed
answer_doc_ids_forbidden_in_runtime_features: passed
downloads_and_cache_refresh_forbidden: passed
stage97_freezes_protocol_without_metrics: passed
stage97_final_test_metrics_not_run: passed
stage97_default_runtime_policy_unchanged: passed
```

### 可视化

```text
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_cache_readiness.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_gate_thresholds.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_rrf_weights.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_feature_group_counts.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_protocol_decision_flags.svg
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97_visuals\stage97_selective_dense_sparse_guard_check_status.svg
```

Stage97 JSON SHA256：

```text
61041D5C3CC2E862F71041D40106845BC3F468E9C2BC0C6E8EF6B7BB659640E4
```

Visualization SHA256：

```text
stage97_selective_dense_sparse_cache_readiness.svg: 189E1C73FB9C8F8DAC88EE9A2650D15E849B0F815B6C56AE4E39A2290D0417E9
stage97_selective_dense_sparse_feature_group_counts.svg: A760D77B693B4C6970A07FA1C157143EBC144D7FC5DF12FBB323B5D4EB4A5A54
stage97_selective_dense_sparse_gate_thresholds.svg: 501DECAA5DF00D9C341ECF6D0C0E316EEA8F8E1F5F9D610F0128906C46055AB2
stage97_selective_dense_sparse_guard_check_status.svg: 7F9F286DDC7B3EB9E1A19D850932EF3BBCE929E7B8199CA5AD14B6A15D1E920F
stage97_selective_dense_sparse_protocol_decision_flags.svg: 1E71BB23C2F8BB2BC167793EDFFD691CB2A2922617E5A33839A98A7BB8B6A253
stage97_selective_dense_sparse_rrf_weights.svg: 2AB4F308DF793121490FE969D1368EB30CFFDB4464E636626E717A2F0E9F1D78
```

### 问题、原因与修正

- 问题 1：Stage97 不能简单复用 Stage81 dense+sparse 全量 RRF。
  - 原因：Stage81 已经证明无 gate 的 dense+sparse 在 dev hit@10 上回退；Stage84 候选要求的是 low-overlap gate。
  - 修正：冻结 selective low-overlap gate policy grid，所有 dense promotion 都必须满足 BM25 低 lexical-overlap 条件，且 dense candidate 必须在 BM25 top10 之外。
  - 影响：Stage98 比较的是一个新 protocol，不是重复 Stage81。

- 问题 2：dense cache / model 不能静默选择。
  - 原因：dense route 涉及本地模型、cache、query prefix、snapshot path；如果不冻结，后续指标无法复现。
  - 修正：Stage97 将 Stage80 compatible cache 和 Stage81 dense config identity 写入 protocol，并禁止 download / cache refresh。
  - 影响：Stage98 只能使用已确认的两个本地 dense configs。

- 问题 3：宽泛 raw-field scan 会命中禁止特征列表。
  - 原因：Stage97 JSON 合法包含 `raw_question_text`、`query_terms` 等作为 prohibited runtime features。
  - 修正：补充实际私有样例字符串扫描和 raw text value field 扫描。
  - 影响：确认 report 没有写出真实题目、答案、文档正文或样例文本。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_protocol.py scripts\freeze_primeqa_hybrid_selective_dense_sparse_protocol.py tests\test_primeqa_hybrid_selective_dense_sparse_protocol.py
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_protocol.py
```

结果：

```text
ruff: passed
pytest: 4 passed
```

产物安全检查：

```text
Select-String actual private snippets over Stage97 JSON: no matches
Select-String raw text value fields over Stage97 JSON: no matches
git check-ignore Stage97 JSON, console, and SVG artifacts: ignored by .gitignore
```

全量验证：

```text
ruff check .: passed
pytest -q: 257 passed
git diff --check: passed
```

### 结论

- Stage97 已冻结 `selective_dense_sparse_low_overlap_gate_train_dev_v1`。
- Stage97 没有读取 train/dev/test split 文件。
- Stage97 没有运行 retrieval metrics。
- Stage97 没有运行 final metrics。
- Stage97 没有下载模型。
- Stage97 没有刷新 dense cache。
- Stage97 没有使用 source `DOC_IDS` 作为 runtime 检索证据。
- Stage97 没有改变 runtime 默认策略。
- Stage98 可以在用户确认后运行 frozen train/dev-only comparison。

### 我学到了

- dense+sparse route 如果不加 gate，会同时救回 not-found@50 和制造 top10 regressions；新的 protocol 必须把 gate 写进协议，而不是只调 RRF 权重。
- 对 dense route 来说，模型、cache、query prefix、document prefix 和 snapshot path 都属于协议的一部分，必须冻结。
- 低 lexical-overlap gate 只能使用 runtime 可观测 aggregate/bucket 特征；不能写出 query terms 或 raw document text。

### 下一步

Stage98：在用户确认后运行 frozen train/dev-only `selective_dense_sparse_low_overlap_gate_train_dev_v1` comparison。

Stage98 仍然不能触碰 test，不能跑 final metrics，不能使用 source `DOC_IDS`，不能下载模型或刷新 dense cache，不能改变 runtime 默认策略。
## Stage98：运行 selective dense+sparse train/dev-only comparison

### 学习时间

2026-07-15

### 本阶段目标

在用户确认 Stage97 frozen protocol 后，运行
`selective_dense_sparse_low_overlap_gate_train_dev_v1` 的 train/dev-only
comparison。边界保持不变：不读取 test，不运行 final metrics，不使用 source
`DOC_IDS` 作为 runtime 检索证据，不下载模型，不刷新 dense cache，不改变 runtime
默认策略。

### 本阶段做了什么

新增并验证了 Stage98 的独立比较链路：

```text
src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_comparison.py
scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py
tests\test_primeqa_hybrid_selective_dense_sparse_comparison.py
docs\primeqa_hybrid_selective_dense_sparse_comparison.md
```

同步更新：

```text
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

Stage98 实现内容：

- 读取 Stage68 train/dev split、PrimeQA documents、Stage75 baseline report、
  Stage97 frozen protocol。
- 只使用 Stage97 冻结的两个本地 dense cache 和本地 snapshot。
- 按 Stage97 policy grid 运行 4 个 low-overlap selective dense+sparse policy。
- 只用 train 选择 policy，dev 仅做 validation。
- 报告只写公开安全字段、聚合指标和 bucket，不写 raw question、raw answer、
  raw document text/title、query terms、matched token strings。
- 生成 6 个 SVG 可视化结果。

### 真实运行结果

Stage98 真实命令重跑后 exit code 为 `0`，stderr 为空。

```text
selected_policy_id: sdsl_minilm_low_overlap_conservative_v1

train baseline hit@10: 0.6622
train selected hit@10: 0.6622
train hit@10 delta: +0.0000
train not_found@50 delta: 0

dev baseline hit@10: 0.6974
dev selected hit@10: 0.6974
dev hit@10 delta: +0.0000
dev hit@1 delta: +0.0000
dev not_found@50 delta: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev gate activations: 0
dev promotions: 0
```

决策：

```text
status: primeqa_hybrid_selective_dense_sparse_comparison_completed
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

结论：Stage98 的 comparison 本身完成且 guard 通过，但 selected policy 没有在
dev 上带来 hit@10 提升，也没有减少 not-found@50；因此 selective dense+sparse
route 不应进入 final-test gate，也不应改 runtime 默认策略。

### 可视化结果

```text
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_train_hit_at_10.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_hit_at_10.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_hit10_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_not_found_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_dev_promotions.svg
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98_visuals\stage98_selective_dense_sparse_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：单元测试 fixture 最初把 `alpha` 放进了 toy answer document title，
  导致 BM25 baseline 也能命中，破坏了“dense gate 救回 BM25 未召回样本”的
  测试目标。
  - 原因：测试 query 和 toy document 之间意外存在 lexical overlap。
  - 修正：去掉 toy answer document 里的 query token，只保留 dense embedding
    可检索信号。
  - 影响：修正后 targeted pytest 通过，测试目的和实现逻辑一致。
- 问题 2：第一次真实运行虽然完整写出了 Stage98 JSON 和 SVG，但 shell 返回
  exit code `1`。
  - 原因：模型加载进度条写入 stderr，在 PowerShell 全流重定向下被包装成
    `NativeCommandError`。
  - 修正：设置 `HF_HUB_DISABLE_PROGRESS_BARS=1` 和 `TQDM_DISABLE=1`，并将
    stdout/stderr 分开保存后重跑。
  - 影响：第二次真实运行 exit code 为 `0`，stderr 为空，报告语义结果未改变。
- 问题 3：selected policy 在 dev 上没有任何 gate activation。
  - 原因：Stage97 冻结的 low-overlap 条件在 dev 的 76 个 answerable 样本上
    全部未触发，主要被 BM25 top1 overlap gate 阻断。
  - 修正：没有擅自放宽 threshold，因为 Stage97 明确禁止 dev-selected
    threshold tuning。
  - 影响：route 必须按协议停止，不能打开 final-test gate。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_comparison.py scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py tests\test_primeqa_hybrid_selective_dense_sparse_comparison.py
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_comparison.py
```

结果：

```text
ruff: passed
pytest: 3 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 260 passed
git diff --check: passed
```

真实 Stage98 run：

```text
python scripts\run_primeqa_hybrid_selective_dense_sparse_comparison.py ...: exit code 0
stderr: empty
guard checks: 23 / 23 passed
```

产物安全检查：

```text
Select-String Stage98 JSON for question_text/question_title/answer_doc_id/DOC_IDS/query_terms/matched_token_strings: no matches
artifact_safety.raw_question_text_written: false
artifact_safety.raw_answer_text_written: false
artifact_safety.raw_document_text_written: false
artifact_safety.source_doc_ids_used_as_runtime_evidence: false
```

### 我学到了

- dense+sparse 如果加了太保守的 low-overlap gate，可能完全避免 regression，但
  也会让 dev 上没有任何有效 action；这类 route 不能靠“看起来更安全”推进，
  必须看 train-selected policy 的 dev 合同是否真实通过。
- Stage98 这类 train/dev comparison 必须把“运行成功”和“指标合同通过”分开：
  本阶段运行成功，guard 通过，但 metric contract 没通过。
- 对会输出模型加载进度的命令，最好从一开始就关闭 progress bar 并拆分
  stdout/stderr，避免 PowerShell 把 stderr 进度条包装成失败状态。

### 下一步

Stage99：停止 selective dense+sparse route，并记录 stop decision。Stage99 仍然
不能读取 test，不能运行 final metrics，不能调 dev threshold，不能改 runtime
默认策略。它的目标是把 Stage97-Stage98 的证据总结为 route stop，而不是继续
微调这个 gate。
## Stage99：停止 selective dense+sparse route

### 学习时间

2026-07-15

### 本阶段目标

根据 Stage98 的真实 train/dev-only comparison 结果，停止
`selective_dense_sparse_low_overlap_gate_design` route，并记录 stop decision。
本阶段是决策检查点，不是新实验：不读取 train/dev/test split 文件，不运行新的
retrieval metrics，不运行 final metrics，不调 dev threshold，不改变 runtime 默认
策略。

### 本阶段做了什么

新增并验证了 Stage99 stop-decision 链路：

```text
src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_stop_decision.py
scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py
tests\test_primeqa_hybrid_selective_dense_sparse_stop_decision.py
docs\primeqa_hybrid_selective_dense_sparse_stop_decision.md
```

同步更新：

```text
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

Stage99 只读取 public-safe 报告：

```text
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_selective_dense_sparse_protocol_stage97.json
artifacts\primeqa_hybrid_selective_dense_sparse_comparison_stage98.json
```

### 真实运行结果

Stage99 真实命令 exit code 为 `0`：

```text
status: primeqa_hybrid_selective_dense_sparse_route_stopped
stopped_candidate_id: selective_dense_sparse_low_overlap_gate_design
stopped_protocol_id: selective_dense_sparse_low_overlap_gate_train_dev_v1
current_route_defaultization: blocked
next_candidate_id: null
remaining_actionable_candidate_count: 0
route_family_exhausted: true
can_continue_train_dev_development: false
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

停止依据：

```text
selected_policy_id: sdsl_minilm_low_overlap_conservative_v1
dev hit@10 delta: +0.0000
dev hit@1 delta: +0.0000
dev not-found@50 delta: 0
dev top10 improvements: 0
dev top10 regressions: 0
dev gate activations: 0
dev promotions: 0
primary_contract_passed: false
secondary_contract_passed: false
guard_contract_passed: true
```

结论：Stage99 停止 selective dense+sparse route。Stage84 second-wave
retrieval actionable candidates 已经耗尽；`source_doc_ids_oracle_union_blocked`
仍然只是 blocked diagnostic，不能作为 train/dev experiment 或 runtime default。

### 可视化结果

```text
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_train_dev_hit10_delta.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_dev_contract_deltas.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_gate_actions.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_second_wave_route_status.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_stop_decision_flags.svg
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99_visuals\stage99_selective_dense_sparse_stop_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage99 不能简单说“Stage98 没提升，所以停止”，必须验证停止动作本身
  没有越过评估边界。
  - 原因：stop decision 也是项目事实的一部分，如果没有 guard，后面容易误以为
    route 可以继续调 dev threshold 或打开 final-test gate。
  - 修正：新增 37 个 guard，确认 Stage97 protocol、Stage98 comparison、合同失败、
    test/final/defaultization 锁定、artifact safety、队列耗尽都成立。
  - 影响：停止结论可复现，不依赖口头判断。
- 问题 2：宽泛 raw-field 扫描会命中 `raw_question_text_written` 这类安全 flag 和
  `source DOC_IDS` policy 文本。
  - 原因：Stage99 合法记录了禁止字段名称和安全 flag。
  - 修正：安全记录区分“禁止字段名作为 policy/flag 出现”和“真实 raw/private
    内容写入”；并扫描具体 private fixture snippet 与 PrimeQA 原始行字段。
  - 影响：避免把合规的安全声明误判为数据泄漏。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_selective_dense_sparse_stop_decision.py scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py tests\test_primeqa_hybrid_selective_dense_sparse_stop_decision.py
pytest -q tests\test_primeqa_hybrid_selective_dense_sparse_stop_decision.py
python scripts\decide_primeqa_hybrid_selective_dense_sparse_stop.py ...: exit code 0
```

结果：

```text
ruff: passed
pytest: 3 passed
Stage99 run: passed
guard checks: 37 / 37 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 263 passed
git diff --check: passed
```

### 我学到了

- 停止路线也需要工程化产物：代码、测试、报告、guard 和可视化，而不是只在聊天里
  总结一句“这条不行”。
- 当所有 second-wave retrieval candidates 耗尽后，下一步不应该继续局部微调某个
  已失败 gate；更合理的是做 route-exhaustion summary，重新从现有 train/dev 证据
  选择下一条研究方向。
- 对 public-safe report 来说，禁止字段名可以作为 policy/flag 出现，但真实问题、
  答案、文档正文、query terms 仍然不能写入报告。

### 下一步

Stage100：总结 second-wave retrieval route exhaustion，并基于现有 train/dev 证据
决定下一条研究方向。Stage100 仍然不能读取 test，不能运行 final metrics，不能把
blocked source `DOC_IDS` diagnostic 变成 runtime/default policy。
## Stage100：总结 second-wave retrieval route exhaustion

### 学习时间

2026-07-15

### 本阶段目标

基于既有 public-safe train/dev 证据，总结 Stage84 second-wave retrieval
routes 是否已经耗尽，并决定下一条研究方向。本阶段不读取 train/dev/test split
文件，不运行新的 retrieval metrics，不运行 final metrics，不调 dev threshold，
不改变 runtime 默认策略。

### 本阶段做了什么

新增并验证了 Stage100 route-exhaustion summary 链路：

```text
src\ts_rag_agent\application\primeqa_hybrid_second_wave_route_exhaustion_summary.py
scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py
tests\test_primeqa_hybrid_second_wave_route_exhaustion_summary.py
docs\primeqa_hybrid_second_wave_route_exhaustion_summary.md
```

同步更新：

```text
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

Stage100 只读取 public-safe 报告：

```text
artifacts\primeqa_hybrid_retrieval_recall_exhaustion_summary_stage83.json
artifacts\primeqa_hybrid_second_wave_retrieval_candidate_design_stage84.json
artifacts\primeqa_hybrid_lexical_cluster_diversity_stop_decision_stage87.json
artifacts\primeqa_hybrid_structured_query_stop_decision_stage90.json
artifacts\primeqa_hybrid_section_signal_stop_decision_stage93.json
artifacts\primeqa_hybrid_score_margin_bm25_stop_decision_stage96.json
artifacts\primeqa_hybrid_selective_dense_sparse_stop_decision_stage99.json
```

### 真实运行结果

Stage100 真实命令 exit code 为 `0`：

```text
status: primeqa_hybrid_second_wave_route_exhaustion_summary_completed
first_wave_retrieval_candidates_exhausted: true
second_wave_retrieval_route_family_exhausted: true
runtime_advancing_second_wave_candidate_count: 0
remaining_actionable_candidate_count: 0
recommended_next_direction: answer_pipeline_error_decomposition
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
default_runtime_policy: unchanged
```

二次 retrieval route 结果：

```text
lexical_cluster_diversity_rerank_design: dev hit@10 delta +0.0000, stopped
structured_query_keyphrase_compaction_design: dev hit@10 delta -0.0527, stopped
section_signal_guarded_expansion_design: dev hit@10 delta +0.0000, stopped
score_margin_bm25_normalization_gate_design: dev hit@10 delta +0.0000, stopped
selective_dense_sparse_low_overlap_gate_design: dev hit@10 delta +0.0000, stopped
```

结论：Stage76 第一轮 retrieval candidates 和 Stage84 第二轮 retrieval routes 均已
耗尽，没有任何 runtime-advancing retrieval candidate。下一条研究方向应转向
`answer_pipeline_error_decomposition`，先把剩余 train/dev 失败拆成 retrieval、
evidence selection、citation、answer composition 等 bucket，再决定下一种干预。

### 可视化结果

```text
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_dev_hit10_deltas.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_top10_net_changes.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_second_wave_route_outcomes.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_next_direction_readiness.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_route_exhaustion_decision_flags.svg
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100_visuals\stage100_route_exhaustion_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage100 需要“决定下一方向”，但不能直接启动新实验。
  - 原因：Stage99 只说明 retrieval route family exhausted；下一步方向需要基于证据
    记录，但 Stage100 自身仍是 summary/decision checkpoint。
  - 修正：输出 `next_direction_options`，推荐
    `answer_pipeline_error_decomposition`，但不运行 Stage101 的指标或 protocol。
  - 影响：下一步方向清楚，同时保持 test/final/defaultization 锁定。
- 问题 2：再次设计 retrieval route 的诱惑很强，但已有两轮 retrieval 候选都没有
  train-selected dev 合同通过。
  - 原因：Stage76 与 Stage84 已覆盖 query view、fielded BM25、section rollup、
    dense+sparse、BM25 参数、lexical diversity、structured query、section signal、
    score-margin 和 selective dense+sparse。
  - 修正：把 `third_wave_retrieval_design` 标为不推荐，除非未来先有新的 deployable
    diagnostic signal。
  - 影响：避免继续局部贪心地围着 retrieval gate 微调。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_second_wave_route_exhaustion_summary.py scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py tests\test_primeqa_hybrid_second_wave_route_exhaustion_summary.py
pytest -q tests\test_primeqa_hybrid_second_wave_route_exhaustion_summary.py
python scripts\summarize_primeqa_hybrid_second_wave_route_exhaustion.py ...: exit code 0
```

结果：

```text
ruff: passed
pytest: 3 passed
Stage100 run: passed
guard checks: 17 / 17 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 266 passed
git diff --check: passed
```

产物安全检查：

```text
Select-String Stage100 JSON for private fixture snippets and PrimeQA raw row fields: no matches
```

### 我学到了

- 当 retrieval 候选连续两轮都没有 train-selected dev 合同通过时，继续扩大
  retrieval grid 可能只是制造更多局部搜索；更好的下一步是先做 pipeline-level
  error decomposition。
- “下一方向推荐”也需要 guard：必须证明来源报告阶段正确、候选都已停止、final
  metrics 仍锁定、runtime 默认策略未变、source `DOC_IDS` diagnostic 仍 blocked。
- route exhaustion summary 的价值是防止后续重复回到已失败候选上微调阈值。

### 下一步

Stage101：设计 train/dev-only `answer_pipeline_error_decomposition` protocol。
Stage101 仍然不能读取 test，不能运行 final metrics，不能改 runtime 默认策略；它
应该只定义如何从现有 public-safe artifacts 里拆分 retrieval、evidence selection、
citation 和 answer composition 错误。

## Stage101：冻结 answer-pipeline error decomposition protocol

### 学习时间

2026-07-15

### 本阶段目标

按照 Stage100 的推荐方向，设计并冻结 train/dev-only
`answer_pipeline_error_decomposition` protocol。这个阶段只定义下一步如何做错误
分解，不运行新的 retrieval 指标、不运行 answer 指标、不读取 test、不运行 final
metrics、不改变 runtime 默认策略，也不新增 fallback strategy。

### 本阶段做了什么

新增并验证了 Stage101 protocol freeze 链路：

```text
src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
docs\primeqa_hybrid_answer_pipeline_error_decomposition_protocol.md
```

同步更新：

```text
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

Stage101 只读取：

```text
artifacts\primeqa_hybrid_second_wave_route_exhaustion_summary_stage100.json
```

它没有读取 train/dev/test split 文件，也没有读取旧 answer-gap case artifacts。原因是旧
answer-gap artifacts 可能包含 question/title/answer/doc id 级别样本；Stage101 只负责
冻结 public-safe 输出契约，真正的 train/dev-only 分解留给 Stage102 在确认后执行。

### 真实运行结果

Stage101 真实命令 exit code 为 `0`：

```text
python scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage101 answer-pipeline error decomposition protocol in current turn" --output artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.json --visualization-dir artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals
```

Stage101 decision：

```text
status: primeqa_hybrid_answer_pipeline_error_decomposition_protocol_frozen
protocol_id: answer_pipeline_error_decomposition_train_dev_v1
recommended_direction: answer_pipeline_error_decomposition
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_analysis: true
can_run_train_dev_error_decomposition_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

冻结的错误分解 bucket：

```text
answerability_false_answer: answerability, precedence 1, priority 1.55
retrieval_context_miss: retrieval, precedence 2, priority 1.35
evidence_selection_miss: evidence_selection, precedence 3, priority 1.70
verification_over_refusal: verification, precedence 4, priority 1.20
gold_span_beats_selected_answer: answer_composition, precedence 5, priority 1.45
low_overlap_gold_cited_answer: answer_composition, precedence 6, priority 1.10
answer_supported_and_cited: non_error_reference, precedence 7, priority 0.25
```

public-safe case sample 字段数量为 `18`，只允许写 bucket、split、rank bucket、
status bucket、policy id/name 等安全字段。gold answer、answer document id、
retrieved/cited document id 只能在预测之后用于计算指标 bucket，不能写入公开报告，
也不能作为 runtime evidence。

### 可视化结果

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_error_bucket_priority_weights.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_pipeline_stage_order.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_public_case_field_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_output_artifact_contract.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_protocol_decision_flags.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101_visuals\stage101_guard_check_status.svg
```

Stage101 JSON SHA256：

```text
952821C48CF882A01ADC1EFA5DD8C6F041B05ED7B009B1DEA2C29CE406538CFA
```

Visualization SHA256：

```text
stage101_error_bucket_priority_weights.svg: C7FAF98716C938263B3E4214BBCD3B8AC5E00E0080675A6A99CAE562773B4E57
stage101_guard_check_status.svg: 009900216392943CD2EB7C319FC6BBCBE4B3C75B33E9DEFBDE22966348675FA4
stage101_output_artifact_contract.svg: F0EA26EBDA00B560F63FC7E41904F476803A3FCE0D92318A7199AB871A217043
stage101_pipeline_stage_order.svg: 007E551FD1B9F92C0688ACC181848D6A7F484741CB316147E6C2FEE057586723
stage101_protocol_decision_flags.svg: 30114C5F6C2B168C4DB738B8CC7037A595DE842E97E1A23513BC5D58AAFBE462
stage101_public_case_field_counts.svg: 1C56955DDABA9226711C7B65A2A8B65339452CF65E97E53D53C5A759B2D43E73
```

### 问题、原因与修正

- 问题 1：初次目标 ruff 检查发现 1 行超过 100 字符。
  - 原因：public-safe label 说明文本过长。
  - 修正：把长字符串拆成多行。
  - 影响：代码风格恢复通过。
- 问题 2：初次目标测试中，成功路径被
  `metric_labels_allowed_only_after_prediction` guard block。
  - 原因：protocol 中两条 private label 使用说明只写了 “used only”，没有明确
    “not written”，guard 比文本更严格。
  - 修正：把三条 private label 说明统一改成 “only ... and not written”。
  - 影响：guard 语义更清楚，成功路径 26/26 guard 全部通过。
- 问题 3：旧 answer-gap artifacts 里存在 case 级字段的历史输出可能不满足新的
  public-safe contract。
  - 原因：早期阶段为了学习和人工分析保存过更多样本细节；Stage101 现在要面向
    后续可持续评估，不能继承这些原始 case 输出习惯。
  - 修正：Stage101 只读取 Stage100 public-safe summary，并把 Stage102 输出限定为
    aggregate + sanitized case fields。
  - 影响：下一步需要按 Stage101 contract 重新生成或净化 train/dev-only 分解报告。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
pytest -q tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py
python scripts\freeze_primeqa_hybrid_answer_pipeline_error_decomposition_protocol.py ...: exit code 0
```

结果：

```text
ruff: passed
pytest: 4 passed
Stage101 run: passed
guard checks: 26 / 26 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 270 passed
git diff --check: passed
```

产物安全检查：

```text
Select-String Stage101 JSON for private fixture snippets and raw case field patterns: no matches
```

### 我学到了

- retrieval route exhausted 之后，下一步不应该马上继续构造 retrieval 候选，而是先
  把 answer pipeline 的损失分解清楚：到底是 answerability、retrieval context、
  evidence selection、verification，还是 answer composition。
- protocol freeze 的价值是提前固定边界：哪些 split 可用、哪些 label 只能用于评分、
  哪些字段绝不能写入报告、哪些 guard 必须通过。这样 Stage102 跑分析时不会临时改口径。
- 对历史 artifacts 不能只因为“已经存在”就直接复用；如果它们包含 raw case 信息，
  新阶段必须重新定义 public-safe 输出契约。
- fallback strategy 也必须显式受控；本阶段没有加入 fallback，缺少必要安全输入时，
  下一阶段应 block，而不是自动切换到别的路径。

### 下一步

Stage102：在用户确认后，运行 Stage101 冻结的 train/dev-only
`answer_pipeline_error_decomposition` 分析。Stage102 仍然不能读取 test，不能运行
final metrics，不能改变 runtime 默认策略；输出必须是 public-safe aggregate 和
sanitized case fields。

## Stage102：运行 train/dev-only answer-pipeline error decomposition

### 学习时间

2026-07-15

### 本阶段目标

按 Stage101 冻结的 protocol，运行真实 train/dev-only answer-pipeline error
decomposition。这个阶段可以比之前更大：一次性完成分析实现、CLI、测试、真实运行、
可视化、文档、学习记录和提交准备。但边界不变：不读取 test，不运行 final metrics，
不改变 runtime 默认策略，不新增 fallback strategy，不把 raw question / answer /
document id / token 字段写入报告。

### 本阶段做了什么

新增并验证了 Stage102 分析链路：

```text
src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py
tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
docs\primeqa_hybrid_answer_pipeline_error_decomposition.md
```

同步更新：

```text
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

Stage102 读取：

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_protocol_stage101.json
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
```

Stage102 没有读取：

```text
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_test.jsonl
```

### 真实运行过程

第一次真实运行超过 120 秒工具超时；检查进程后发现 Python 子进程仍在继续运行，并最终写出了
JSON，但 console 文件为空。因此这次不作为最终记录结果。

随后用更长 timeout 重跑同一命令，真实返回：

```text
stage102_exit_code=0
```

最终命令：

```text
python scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py --user-confirmed-analysis --confirmation-note "user confirmed Stage102 train-dev answer-pipeline error decomposition in current turn" --output artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json --visualization-dir artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals
```

运行耗时：

```text
total_seconds: 197.39
```

### 分析配置

```text
diagnostic_profile: stage102_bm25_top10_bm25_sentence_mcpd3_topk_verified_evidence7
retriever: BM25
bm25_k1: 1.5
bm25_b: 0.75
retrieval_top_k: 10
answer_generator: extractive_sentence_baseline
evidence_selector: bm25_sentence
max_candidates_per_document: 3
composition_policy: top_k
max_sentences: 3
min_sentence_score: 2.0
answer_verifier: citation_and_evidence_gate
min_evidence_score: 7.0
max_citation_rank: 3
min_citations: 1
max_gold_window_sentences: 3
gold_span_gap_margin: 0.05
low_answer_f1_threshold: 0.2
sample_limit_per_bucket: 5
```

### 数据规模

```text
documents: 28482
train rows: 562
train answerable: 370
train unanswerable: 192
dev rows: 121
dev answerable: 76
dev unanswerable: 45
```

### 真实分解结果

Bucket counts：

```text
train:
answerability_false_answer: 180 / 562 = 0.3203
retrieval_context_miss: 125 / 562 = 0.2224
evidence_selection_miss: 67 / 562 = 0.1192
verification_over_refusal: 3 / 562 = 0.0053
gold_span_beats_selected_answer: 174 / 562 = 0.3096
low_overlap_gold_cited_answer: 0 / 562 = 0.0000
answer_supported_and_cited: 13 / 562 = 0.0231

dev:
answerability_false_answer: 41 / 121 = 0.3388
retrieval_context_miss: 23 / 121 = 0.1901
evidence_selection_miss: 12 / 121 = 0.0992
verification_over_refusal: 0 / 121 = 0.0000
gold_span_beats_selected_answer: 41 / 121 = 0.3388
low_overlap_gold_cited_answer: 0 / 121 = 0.0000
answer_supported_and_cited: 4 / 121 = 0.0331
```

Verified metrics：

```text
train:
gold_doc_citation_rate: 0.4958
answerable_refusal_rate: 0.0459
unanswerable_refusal_rate: 0.0625
average_token_f1: 0.2017

dev:
gold_doc_citation_rate: 0.6029
answerable_refusal_rate: 0.1053
unanswerable_refusal_rate: 0.0889
average_token_f1: 0.2040
```

Answerable gold context：

```text
train: 245 / 370 = 0.6622
dev: 53 / 76 = 0.6974
```

Stage102 decision：

```text
status: primeqa_hybrid_answer_pipeline_error_decomposition_completed
analysis_id: answer_pipeline_error_decomposition_train_dev_analysis_v1
train_top_bucket: answerability_false_answer
dev_top_bucket: answerability_false_answer
train_evidence_selection_miss: 67
dev_evidence_selection_miss: 12
train_answerability_false_answer: 180
dev_answerability_false_answer: 41
train_verified_average_token_f1: 0.2017
dev_verified_average_token_f1: 0.2040
recommended_next_direction: evidence_selection_and_answerability_candidate_design
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

### 可视化结果

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_bucket_counts_by_split.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_pipeline_stage_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_answerability_bucket_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_verified_metric_rates.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_public_case_sample_counts.svg
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102_visuals\stage102_guard_check_status.svg
```

Stage102 JSON SHA256：

```text
08F2CA0F31F1B79AD78619AA33F8C53B3BFA7C6CB68DAEC3013987D43549B7C6
```

Visualization SHA256：

```text
stage102_answerability_bucket_counts.svg: 7D9651F9F91C528A4F18545968DC706529EFC76774412AE70FA20BC7FAAE4297
stage102_bucket_counts_by_split.svg: 5507B0FF0612656D517550C7B7E0E168E0191B63E66A06304DDDBEDF3A0FCAFB
stage102_guard_check_status.svg: 26F1B6DFBD4B648FFAFBF3DF316D4DF5E85C0816BBC950B8CF6EE5D1F7176AEF
stage102_pipeline_stage_counts.svg: E444FB7F5E95191C3611E60FC9947028FFB44B75F2228D18ACD28795E025EB40
stage102_public_case_sample_counts.svg: B17BA9B59D125D96310AE9AF9D46D88EFD4F559F0CF407B9E029EF1C7A03135E
stage102_verified_metric_rates.svg: 0CE34354E426705B12166226EA619894769FBC1070B6CFD38C506C816DE7A3F6
```

### 问题、原因与修正

- 问题 1：第一次真实运行被 120 秒工具 timeout 中断父 shell。
  - 原因：Stage102 是真实 train/dev answer-pipeline 分析，需要对 683 条 train/dev
    样本跑 BM25 检索、句子选择、验证和 gold span bucket 计算，耗时约 197 秒。
  - 修正：检查 Python 进程，确认没有重复启动；等原进程完成后，因 console 为空，使用更长
    timeout 重跑同一命令并拿到 `stage102_exit_code=0`。
  - 影响：最终记录的是第二次明确 exit code 0 的运行结果。
- 问题 2：Stage102 需要输出 case samples，但不能泄露 raw text 或 document id。
  - 原因：错误分解内部必须用 gold answer 和 answer document id 评分，但这些 label 不能作为
    runtime evidence，也不能写入 public-safe 报告。
  - 修正：只写 Stage101 允许的 18 个 sanitized 字段，例如 bucket、rank bucket、status
    bucket、route、selector、composition policy。
  - 影响：可以做 case-level 抽样定位，同时保持报告 public-safe。
- 问题 3：`answerability_false_answer` 和 `gold_span_beats_selected_answer` 同时在 train/dev
  很高。
  - 原因：当前 verifier 对 unanswerable 的拒答率仍低，且 extractive answer composition 经常
    选不到更接近 gold span 的文本。
  - 修正：Stage102 不直接改策略，只把下一阶段方向定为
    `evidence_selection_and_answerability_candidate_design`。
  - 影响：Stage103 应先设计候选 intervention，不应直接改 runtime 或跑 test。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
pytest -q tests\test_primeqa_hybrid_answer_pipeline_error_decomposition_analysis.py
python scripts\run_primeqa_hybrid_answer_pipeline_error_decomposition.py ...: stage102_exit_code=0
```

结果：

```text
ruff: passed
pytest: 4 passed
Stage102 run: passed
guard checks: 20 / 20 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 274 passed
git diff --check: passed
```

产物安全检查：

```text
Select-String Stage102 JSON for raw question / answer / document id / token field patterns: no matches
```

### 我学到了

- “大一点的 stage”可以一次性完成实现、真实运行、可视化和记录，但必须把中途异常如 timeout
  如实记录，不能把第一次不完整 console 伪装成成功运行。
- 当前瓶颈已经不是单纯 retrieval：train/dev 同时显示 answerability false answer 和 gold-span
  composition gap 很大。
- answerability 问题不能靠简单提高 verifier threshold 解决，因为 dev answerable refusal rate
  已经达到 0.1053；下一步需要同时考虑 unanswerable false answer 和 answerable over-refusal。
- sanitized samples 仍然能支持工程判断，只要把 doc id 和 raw text 全部替换成 bucket/status。

### 下一步

Stage103：基于 Stage102 的共同瓶颈，设计 train/dev-only candidate intervention protocol。
优先方向是 `evidence_selection_and_answerability_candidate_design`，但仍不能读取 test、不能运行
final metrics、不能改 runtime 默认策略、不能加入 fallback strategy。

## 2026-07-15 Stage103：冻结 evidence/answerability candidate protocol

### 本阶段目标

Stage103 的目标不是跑新指标，而是把 Stage102 暴露出来的共同瓶颈转成下一步可执行的
train/dev-only candidate protocol。输入只允许使用 Stage102 已保存的 public-safe aggregate JSON，
不读取 train/dev/test split，不读取 corpus documents，不运行 retrieval/answer/final metrics，不改 runtime 默认策略，
不加入 fallback strategy。

### 新增内容

代码与脚本：

```text
src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_candidate_protocol.py
scripts\design_primeqa_hybrid_evidence_answerability_candidates.py
tests\test_primeqa_hybrid_evidence_answerability_candidate_protocol.py
```

文档：

```text
docs\primeqa_hybrid_evidence_answerability_candidate_protocol.md
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

产物：

```text
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.console.txt
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_shared_bottleneck_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_shared_bottleneck_rates.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_priority_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_target_case_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_candidate_feature_group_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_protocol_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103_visuals\stage103_guard_check_status.svg
```

### 真实运行结果

运行命令：

```text
python scripts\design_primeqa_hybrid_evidence_answerability_candidates.py --user-confirmed-protocol --confirmation-note "Stage103 user-confirmed design freeze after Stage102 decomposition"
```

真实返回：

```text
stage103_exit_code=0
```

Stage103 decision：

```text
status: primeqa_hybrid_evidence_answerability_candidate_protocol_frozen
design_id: evidence_selection_and_answerability_candidate_design_v1
recommended_direction: evidence_answerability_train_dev_candidate_comparison
requires_user_confirmation_before_train_dev_run: true
can_continue_train_dev_development: true
can_run_train_dev_candidate_comparison_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage103 只读取：

```text
artifacts\primeqa_hybrid_answer_pipeline_error_decomposition_stage102.json
```

Stage103 没有读取：

```text
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_test.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
```

### Stage102 共同瓶颈

```text
answerability_false_answer:
train: 180 / 562 = 0.3203
dev: 41 / 121 = 0.3388
combined: 221

gold_span_beats_selected_answer:
train: 174 / 562 = 0.3096
dev: 41 / 121 = 0.3388
combined: 215

retrieval_context_miss:
train: 125 / 562 = 0.2224
dev: 23 / 121 = 0.1901
combined: 148

evidence_selection_miss:
train: 67 / 562 = 0.1192
dev: 12 / 121 = 0.0992
combined: 79
```

结论：Stage103 把前两个作为 primary candidate target，把 `evidence_selection_miss` 作为 evidence
secondary context，把 `retrieval_context_miss` 作为 secondary context only，不重新开启新的 retrieval-route family。

### 冻结的候选协议

候选策略：

```text
answerability_margin_gate_candidate_v1
- target: answerability_false_answer
- target_combined_case_count: 221
- priority_score: 271.2996

evidence_window_reselector_candidate_v1
- target: gold_span_beats_selected_answer + evidence_selection_miss
- target_combined_case_count: 294
- priority_score: 313.1271

joint_gate_then_window_candidate_v1
- target: answerability_false_answer + gold_span_beats_selected_answer + evidence_selection_miss
- target_combined_case_count: 515
- priority_score: 366.6990
```

推荐 Stage104 执行顺序：

```text
1. joint_gate_then_window_candidate_v1
2. evidence_window_reselector_candidate_v1
3. answerability_margin_gate_candidate_v1
```

这个顺序只是 Stage104 的候选比较顺序，不是 runtime 默认策略，也不是最终上线结论。

### 问题、原因与修正

- 问题 1：Stage102 的最大瓶颈不是单一模块，`answerability_false_answer` 和
  `gold_span_beats_selected_answer` 在 train/dev 都很高。
  - 原因：只调 retrieval route 已经进入收益耗尽区；当前错误更像 answerability gate 与 evidence/window
    selection 的耦合问题。
  - 修正：Stage103 不继续开 retrieval route，而是冻结 answerability gate、evidence window reselector、
    joint gate-then-window 三个候选协议。

- 问题 2：下一步如果直接调阈值，容易把 dev 变成 tuning 数据。
  - 原因：候选策略会涉及 threshold/grid，若允许 dev 参与选择，就会破坏 train/dev/test 边界。
  - 修正：Stage103 明确 Stage104 的 `candidate_thresholds_selected_on: train_only`、
    `dev_threshold_tuning_allowed: false`、`test_access_allowed: false`。

- 问题 3：候选协议需要说明哪些特征能作为 runtime signal，哪些必须禁止。
  - 原因：Stage102 内部可以用 gold labels 评估 bucket，但这些不能变成 runtime evidence 或 public output。
  - 修正：Stage103 明确允许 question/retrieval/evidence/composition observable features；
    禁止 gold answers、gold spans、answer document identifiers、source DOC_IDS、test labels、
    dev-selected thresholds 和 raw private strings。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_candidate_protocol.py scripts\design_primeqa_hybrid_evidence_answerability_candidates.py tests\test_primeqa_hybrid_evidence_answerability_candidate_protocol.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_candidate_protocol.py
python scripts\design_primeqa_hybrid_evidence_answerability_candidates.py ...: stage103_exit_code=0
```

结果：

```text
ruff: passed
pytest: 5 passed
Stage103 run: passed
guard checks: 31 / 31 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 279 passed
git diff --check: passed
```

产物安全检查：

```text
Allowed output fields intersect forbidden fields: []
Stage103 source_files 只记录 Stage102 JSON，不记录 train/dev/test split 或 corpus document 路径。
字段名扫描会在 forbidden_fields / explicit_exclusions 清单里命中 question_text、answer_doc_id 等禁止字段名；
这些是预期的禁止清单，不是 allowed output fields，也不是 raw private values。
```

### 我学到了

- 当 Stage102 已经表明主要错误跨越 answerability 和 evidence/composition 时，下一步不能再用单点 retrieval
  思维硬试；应该先冻结候选干预协议，再跑 train/dev-only 比较。
- 设计阶段也要有 guard checks 和可视化，否则后续实验容易忘记边界，尤其是 train-only selection 与 dev validation
  的区别。
- 候选策略里的 runtime features 必须写清楚允许和禁止项；否则很容易把评估标签、oracle doc id 或 dev tuning
  偷偷混进方案。
- 可视化不一定只服务最终指标，这一阶段用 bottleneck counts/rates、candidate priority、feature groups 和
  guard status 图，可以让下一步选择更可检查。

### 下一步

Stage104：在用户确认后，运行 frozen train/dev-only evidence-answerability candidate comparison，
对比 Stage102 verified baseline。Stage104 仍然不能读取 test，不能运行 final metrics，不能用 dev 调阈值，
不能改 runtime 默认策略，不能加入 fallback strategy。

## 2026-07-15 Stage104：冻结 evidence/answerability comparison grid protocol

### 本阶段目标

Stage104 的目标是先冻结 Stage105 将要运行的 train/dev-only comparison grid，而不是直接跑指标。
上一阶段 Stage103 已经确定候选方向，但还没有把 candidate threshold grid 写死；如果直接跑比较，
就会把设计和实验混在一起。Stage104 因此只读取 Stage103 public-safe protocol JSON，冻结 9 个候选配置、
train-only 选择规则、dev-only 验证规则、guard metrics 和 public-safe 输出契约。

本阶段不读取 train/dev/test split，不读取 corpus documents，不运行 retrieval/answer/final metrics，
不改 runtime 默认策略，不加入 fallback strategy。

### 新增内容

代码与脚本：

```text
src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_comparison_protocol.py
scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py
tests\test_primeqa_hybrid_evidence_answerability_comparison_protocol.py
```

文档：

```text
docs\primeqa_hybrid_evidence_answerability_comparison_protocol.md
docs\evaluation_strategy.md
docs\data_strategy.md
docs\learning_journal.md
```

产物：

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.json
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104.console.txt
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_counts_by_candidate.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_min_evidence_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_config_max_citation_ranks.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_selector_mix.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_train_selection_guard_thresholds.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_protocol_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_protocol_stage104_visuals\stage104_guard_check_status.svg
```

### 真实运行结果

运行命令：

```text
python scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py --user-confirmed-protocol --confirmation-note "Stage104 user-confirmed comparison-grid freeze after Stage103 protocol"
```

真实返回：

```text
stage104_exit_code=0
```

Stage104 decision：

```text
status: primeqa_hybrid_evidence_answerability_comparison_protocol_frozen
protocol_id: evidence_answerability_candidate_train_dev_comparison_v1
recommended_direction: run_evidence_answerability_train_dev_candidate_comparison
requires_user_confirmation_before_train_dev_metric_run: true
can_continue_train_dev_development: true
can_run_train_dev_candidate_comparison_after_user_confirmation: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

Stage104 只读取：

```text
artifacts\primeqa_hybrid_evidence_answerability_candidate_protocol_stage103.json
```

Stage104 没有读取：

```text
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_train.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_dev.jsonl
artifacts\primeqa_hybrid_split_stage68_splits\primeqa_hybrid_split_stage68_test.jsonl
data\raw\primeqa_techqa\TechQA\training_and_dev\training_dev_technotes.sections.json
```

### 冻结的网格

这些阈值是 Stage104 预声明的工程设计参数，不是从 train/dev/test label 或 Stage104 metric run 拟合出来的。
Stage104 没有运行任何指标。

```text
answerability_margin_gate_candidate_v1:
- amg_bm25_evidence8_rank3_v1
- amg_bm25_evidence9_rank3_v1
- amg_bm25_evidence8_rank2_v1

evidence_window_reselector_candidate_v1:
- ewr_answer_window_mcpd3_evidence7_rank3_v1
- ewr_hybrid_window_mcpd3_evidence7_rank3_v1
- ewr_answer_window_mcpd5_evidence7_rank3_v1

joint_gate_then_window_candidate_v1:
- jgw_answer_window_mcpd3_evidence8_rank3_v1
- jgw_hybrid_window_mcpd3_evidence8_rank3_v1
- jgw_answer_window_mcpd5_evidence8_rank2_v1
```

总计：

```text
9 configs
3 configs per Stage103 candidate family
selectors: bm25_sentence, answer_window, hybrid_window
composition_policy_name: top_k
```

### Stage105 选择规则

Stage105 必须只在 train 上选择 config：

```text
weighted target score =
1.55 * answerability_false_answer
+ 1.45 * gold_span_beats_selected_answer
+ 1.70 * evidence_selection_miss
```

Train selectability guards：

```text
max_train_answerable_refusal_rate_delta: 0.05
max_train_average_token_f1_drop: 0.01
max_train_gold_doc_citation_rate_drop: 0.03
```

Dev 只能做一次验证：

```text
dev_retuning_allowed: false
dev_selection_allowed: false
test_access_allowed: false
```

### 问题、原因与修正

- 问题 1：Stage103 虽然允许进入 comparison，但还没有冻结具体 threshold grid。
  - 原因：Stage103 是 candidate design，不是 metric protocol；如果直接跑实验，就会在实验实现里隐式选择阈值。
  - 修正：Stage104 单独冻结 9 个 config，并把 grid_derivation_policy 写清楚：没有用 train/dev/test labels
    来选择阈值，本阶段也没有跑 metrics。

- 问题 2：answerability gate 可能减少 unanswerable false answers，但也可能过度拒答 answerable 问题。
  - 原因：Stage102 的 dev answerable refusal rate 已经是 0.1053，不能只看 false answer 下降。
  - 修正：Stage104 冻结 train selectability guards，要求 Stage105 选择 config 时约束 answerable refusal、
    average token F1 和 gold citation rate 的 train 退化。

- 问题 3：evidence/window selector 与 gate 的组合会让比较空间变大。
  - 原因：如果把 selector、mcpd、evidence threshold 和 citation rank 全部展开，会让 Stage105 运行和解释成本过高。
  - 修正：Stage104 固定为每个 candidate family 3 个 config，总计 9 个，覆盖核心方向但避免无边界网格扩张。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_comparison_protocol.py scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py tests\test_primeqa_hybrid_evidence_answerability_comparison_protocol.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_comparison_protocol.py
python scripts\freeze_primeqa_hybrid_evidence_answerability_comparison_protocol.py ...: stage104_exit_code=0
```

结果：

```text
ruff: passed
pytest: 5 passed
Stage104 run: passed
guard checks: 31 / 31 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 284 passed
git diff --check: passed
```

产物安全检查：

```text
Allowed output fields intersect forbidden fields: []
Stage104 source_files 只记录 Stage103 JSON，不记录 train/dev/test split 或 corpus document 路径。
Stage104 JSON 扫描没有命中 private fixture、split path 或 corpus path patterns。
```

### 我学到了

- “下一步跑比较”之前必须先看协议是否已经冻结到可复现参数粒度；否则后续指标即使跑出来，也会夹带未记录的设计选择。
- answerability gate 的网格不能只追求拒答更多 unanswerable；必须同步冻结 answerable refusal、F1 和 citation guard。
- 把 Stage104 作为 protocol freeze，而把 Stage105 作为 metric run，可以保持 test lock、dev validation 和 runtime default
  边界更清楚。
- 可视化不只展示结果，也可以展示 protocol 结构：config family count、min evidence score、citation rank、selector mix、
  train guards 和 guard status。

### 下一步

Stage105：在用户确认后，运行 frozen train/dev-only evidence-answerability candidate comparison，对比 Stage102
verified baseline。Stage105 必须只用 train 选择 config，只用 dev 做一次验证；仍不能读取 test、不能运行 final metrics、
不能改 runtime 默认策略、不能加入 fallback strategy。

## Stage105：运行 evidence-answerability train/dev 候选比较

### 目标

本阶段执行 Stage104 冻结的 9 个 evidence/answerability candidate config，对比 Stage102 verified BM25 top10
answer pipeline baseline。Stage105 的边界仍然是：

```text
只读 train/dev split
不读取 test split
不运行 final test metrics
只用 train 做 config selection
dev 只做一次 validation
不改 runtime default
不加入 fallback strategy
不把 source DOC_IDS 当作 runtime evidence
```

### 新增和修改

代码：

```text
src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_comparison.py
scripts\run_primeqa_hybrid_evidence_answerability_comparison.py
tests\test_primeqa_hybrid_evidence_answerability_comparison.py
```

文档：

```text
docs\primeqa_hybrid_evidence_answerability_comparison.md
docs\learning_journal.md
```

本地产物：

```text
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105.json
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105.console.txt
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_weighted_target_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_dev_weighted_target_scores.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_target_score_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_dev_target_score_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_train_selectability_guards.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_changed_answer_counts.svg
artifacts\primeqa_hybrid_evidence_answerability_comparison_stage105_visuals\stage105_guard_check_status.svg
```

### 真实运行命令

```text
python scripts\run_primeqa_hybrid_evidence_answerability_comparison.py --user-confirmed-comparison --confirmation-note "user confirmed Stage105 train/dev candidate comparison on 2026-07-15; test locked; runtime defaults unchanged; no fallback strategies"
```

真实运行结果：

```text
exit_code: 0
timing_seconds.total: 368.557
guard_checks: 25 / 25 passed
```

### 数据和 baseline

```text
documents: 28482
train rows: 562
train answerable rows: 370
train unanswerable rows: 192
dev rows: 121
dev answerable rows: 76
dev unanswerable rows: 45
test_split_loaded: false
```

Baseline：

```text
baseline_config_id: stage102_verified_bm25_top10_answer_pipeline
train weighted target score: 645.2
dev weighted target score: 143.4
```

Stage105 重新计算的 baseline bucket counts 和 verified metrics 与 Stage102 saved report 完全匹配，说明本阶段比较仍然接在
Stage102 verified baseline 上，而不是换了隐式 baseline。

### Train selection 结果

Train objective：

```text
1.55 * answerability_false_answer
+ 1.45 * gold_span_beats_selected_answer
+ 1.70 * evidence_selection_miss
```

Train-selected config：

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_candidate_id: answerability_margin_gate_candidate_v1
selected_train_weighted_target_delta: 0.0
selectable_config_count: 2 / 9
```

Train ranking 摘要：

```text
amg_bm25_evidence8_rank3_v1: train delta 0.00, selectable true, changed 1
amg_bm25_evidence9_rank3_v1: train delta 0.00, selectable true, changed 1
jgw_answer_window_mcpd5_evidence8_rank2_v1: train delta -133.90, selectable false, changed 542
ewr_answer_window_mcpd3_evidence7_rank3_v1: train delta -89.25, selectable false, changed 546
ewr_answer_window_mcpd5_evidence7_rank3_v1: train delta -89.25, selectable false, changed 546
jgw_answer_window_mcpd3_evidence8_rank3_v1: train delta -89.25, selectable false, changed 546
amg_bm25_evidence8_rank2_v1: train delta -29.25, selectable false, changed 37
ewr_hybrid_window_mcpd3_evidence7_rank3_v1: train delta -16.60, selectable false, changed 550
jgw_hybrid_window_mcpd3_evidence8_rank3_v1: train delta -16.60, selectable false, changed 550
```

### Dev validation 结果

Dev 只验证 train-selected config，没有用于选择或调参。

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
dev_validation_passed: false
```

Dev target bucket deltas：

```text
answerability_false_answer: 0
gold_span_beats_selected_answer: 0
evidence_selection_miss: 0
```

Dev metric deltas：

```text
answerable_refusal_rate: 0.0
unanswerable_refusal_rate: 0.0
gold_doc_citation_rate: 0.0
average_token_f1: 0.0
```

### 决策

```text
status: primeqa_hybrid_evidence_answerability_comparison_completed_dev_guard_failed
selected_config_id: amg_bm25_evidence8_rank3_v1
selectable_config_count: 2
dev_validation_passed: false
can_continue_train_dev_development: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_direction: evidence_answerability_stop_decision
```

结论：Stage105 不支持把 Stage103/104 这组 evidence-answerability policy 进入 runtime，也不能打开 test gate。虽然若干
answer-window/joint configs 在 train/dev target score 上有更大的负 delta，但它们没有通过 train selectability guards，不能从 dev
反选。

### 问题、原因与修正

- 问题 1：实现时最初的 `dev_validation_not_used_for_selection` guard 只是自反判断，没有证明 dev 没进入 selection。
  - 原因：第一次实现只验证了 train selection 结果存在，没有审计 selection ranking 字段边界。
  - 修正：改为检查 train selection ranking 只能包含固定 train 字段：`rank`、`config_id`、`candidate_id`、
    `train_weighted_target_score`、`train_weighted_target_delta`、`train_selectable`、`train_changed_answer_count`。
- 问题 2：单元测试最初用字符串 `"answer_doc_id"` 做粗粒度扫描，误伤了合法统计字段 `unique_answer_doc_ids`。
  - 原因：统计字段不是原始 answer doc id，但字符串包含相同子串。
  - 修正：测试改为扫描精确 JSON key：`"answer_doc_id":` 和 `"question_text":`。
- 问题 3：窗口类候选出现显著 target score 改善，但不能入选。
  - 原因：改善主要来自更激进拒答或证据重选，同时触发 answerable refusal 或 citation guard。
  - 修正：按 Stage104 冻结协议如实阻断这些 config，不从 dev 反选，不新增 fallback，也不改 runtime default。

### 验证

局部验证：

```text
ruff check --fix src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_comparison.py scripts\run_primeqa_hybrid_evidence_answerability_comparison.py tests\test_primeqa_hybrid_evidence_answerability_comparison.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_comparison.py
python scripts\run_primeqa_hybrid_evidence_answerability_comparison.py --user-confirmed-comparison ...
```

结果：

```text
targeted pytest: 2 passed
Stage105 run: passed
guard checks: 25 / 25 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 286 passed
git diff --check: passed
```

### 我学到了

- target bucket objective 必须和 train selectability guard 一起看；只看 weighted target score 会把大量 answerable refusal/citation
  风险误判成改进。
- Dev 上出现更好的非入选候选不能作为选择依据。Stage105 的正确行为是记录 dev 现象，但仍保持 train-only selection。
- answerability margin gate 这次几乎是 no-op：train 只改变 1 个答案，dev 改变 0 个答案，所以 dev validation 失败是合理结果。
- 可视化对这类阶段很有用：train/dev score、delta、selectability、changed answers、guard status 可以一起说明“为什么不能推进”。

### 下一步

Stage106：记录 evidence-answerability candidate family 的 stop/redesign decision。下一步需要解释：

```text
为什么 train-selected config 没有通过 dev validation；
为什么 dev 上看起来更好的非 selectable configs 不能被反选；
是否停止这条 Stage103/104 candidate family；
如果 redesign，新的方案必须继续保持 train/dev-only、test locked、runtime defaults unchanged、no fallback strategies。
```

## Stage106：记录 evidence-answerability family stop decision

### 目标

本阶段不再跑新指标，而是把 Stage105 的失败结果正式转成可审计的 stop/redesign decision。Stage106 的边界：

```text
只读 Stage104 protocol report 和 Stage105 comparison report
不读取 train/dev/test split
不读取 corpus documents
不运行 retrieval metrics
不运行 answer metrics
不运行 final test metrics
不从 dev-only observations 反选 config
不改 runtime default
不加入 fallback strategy
```

### 新增和修改

代码：

```text
src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_stop_decision.py
scripts\decide_primeqa_hybrid_evidence_answerability_stop.py
tests\test_primeqa_hybrid_evidence_answerability_stop_decision.py
```

文档：

```text
docs\primeqa_hybrid_evidence_answerability_stop_decision.md
docs\primeqa_hybrid_evidence_answerability_comparison.md
docs\learning_journal.md
```

本地产物：

```text
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106.json
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106.console.txt
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_evidence_answerability_target_deltas.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_train_selectability_by_family.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_train_guard_failure_reasons.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_stop_decision_flags.svg
artifacts\primeqa_hybrid_evidence_answerability_stop_decision_stage106_visuals\stage106_stop_guard_check_status.svg
```

### 事实修正

Stage106 期间重新读取了 Stage105 JSON，确认 Stage105 的真实 guard check 数量是：

```text
25 / 25 passed
```

我之前在 Stage105 人工总结和文档里写成了 `29 / 29 passed`。这不是数据变化，而是人工汇总错误。本阶段已经把
`docs\primeqa_hybrid_evidence_answerability_comparison.md` 和本 learning journal 中对应位置修正为 `25 / 25 passed`。

### 真实运行命令

```text
python scripts\decide_primeqa_hybrid_evidence_answerability_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage106 evidence-answerability stop decision on 2026-07-15 after Stage105 dev validation failed; test locked; runtime defaults unchanged; no fallback strategies"
```

真实运行结果：

```text
exit_code: 0
guard_checks: 30 / 30 passed
```

### Stage105 证据

Train-selected config：

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_candidate_id: answerability_margin_gate_candidate_v1
selectable_config_count: 2 / 9
selected_train_weighted_target_delta: 0.0
```

Dev validation：

```text
dev_validation_passed: false
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
answerability_false_answer delta: 0
gold_span_beats_selected_answer delta: 0
evidence_selection_miss delta: 0
answerable_refusal_rate delta: 0.0
gold_doc_citation_rate delta: 0.0
average_token_f1 delta: 0.0
```

Dev 上更好的 non-selectable configs：

```text
jgw_answer_window_mcpd5_evidence8_rank2_v1: dev delta -29.15, train delta -133.90, failed answerable refusal + gold citation guards
ewr_answer_window_mcpd3_evidence7_rank3_v1: dev delta -17.05, train delta -89.25, failed answerable refusal + gold citation guards
ewr_answer_window_mcpd5_evidence7_rank3_v1: dev delta -17.05, train delta -89.25, failed answerable refusal + gold citation guards
jgw_answer_window_mcpd3_evidence8_rank3_v1: dev delta -17.05, train delta -89.25, failed answerable refusal + gold citation guards
ewr_hybrid_window_mcpd3_evidence7_rank3_v1: dev delta -3.90, train delta -16.60, failed answerable refusal guard
jgw_hybrid_window_mcpd3_evidence8_rank3_v1: dev delta -3.90, train delta -16.60, failed answerable refusal guard
amg_bm25_evidence8_rank2_v1: dev delta -3.10, train delta -29.25, failed answerable refusal guard
```

### 决策

```text
status: primeqa_hybrid_evidence_answerability_candidate_family_stopped
stopped_family_id: evidence_answerability_candidate_family
stopped_protocol_id: evidence_answerability_candidate_train_dev_comparison_v1
current_route_defaultization: blocked
redesign_required_before_any_runtime_or_test_gate: true
recommended_next_direction: evidence_answerability_redesign_decision
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

结论：当前 Stage103/104 evidence-answerability candidate family 停止，不能进入 runtime，不能打开 test gate。下一步如果继续，只能在用户确认后做新的 train/dev-only redesign protocol；不能从 dev 反选现有 config。

### 问题、原因与修正

- 问题 1：Stage105 的人工总结 guard 数写错。
  - 原因：根据 console 人工整理时把 guard 数误写成 29，真实 Stage105 JSON 是 25。
  - 修正：Stage106 重新读取 JSON 后纠正文档与 journal，并在 Stage106 文档中显式记录这个事实修正。
- 问题 2：Dev 上有明显更好的 config，但不能推进。
  - 原因：Stage104 协议明确禁止 dev selection，且这些 config 全部没有过 train selectability guards。
  - 修正：Stage106 把这些 config 单独列为 `dev_better_nonselectable_configs`，说明它们是观察证据，不是选择依据。
- 问题 3：停止当前 family 后是否马上 redesign 存在选择空间。
  - 原因：redesign 会引入新的实验协议和设计取舍，不能在 stop decision 里悄悄决定。
  - 修正：Stage106 只记录 family stopped，并把 Stage107 定义为“只有用户确认后才冻结新的 train/dev-only redesign protocol”。

### 验证

局部验证：

```text
ruff check --fix src\ts_rag_agent\application\primeqa_hybrid_evidence_answerability_stop_decision.py scripts\decide_primeqa_hybrid_evidence_answerability_stop.py tests\test_primeqa_hybrid_evidence_answerability_stop_decision.py
pytest -q tests\test_primeqa_hybrid_evidence_answerability_stop_decision.py
python scripts\decide_primeqa_hybrid_evidence_answerability_stop.py --user-confirmed-stop ...
```

结果：

```text
targeted pytest: 3 passed
Stage106 run: passed
guard checks: 30 / 30 passed
```

全量验证：

```text
ruff check .: passed
pytest -q: 289 passed
git diff --check: passed
```

### 我学到了

- Stop decision 不只是“失败记录”，还要把不能推进的原因拆开：selected config 没提升、dev-better config 不可选、runtime/test gate 仍锁定。
- 对人工总结的数字必须回查 JSON；Stage105 guard 数从报告看是 25，不是 29。
- “redesign”本身是新协议，不应该被 stop decision 偷偷决定；只能作为下一阶段、用户确认后的动作。
- 可视化在 stop 阶段同样有价值：target deltas、family selectability、guard failure reasons 和 decision flags 能直接解释为什么停。

### 下一步

Stage107：只有在用户确认后，冻结新的 train/dev-only evidence-answerability redesign protocol，或者明确转向其他研究方向。Stage107 必须继续保持：

```text
test locked
no final metrics
no dev-only selection
runtime defaults unchanged
no fallback strategies
```

## Stage112：运行 retrieval_context_miss 根因审计

### 记录顺序说明

Stage111 已经记录在本文件中，但当前文件末尾仍保留着后写入的 Stage110 段落。
这次 Stage112 记录追加在文件末尾，保持本次真实发生的记录可追溯；没有改写历史
段落内容。

### 目标

按照 Stage111 冻结的协议，运行 train/dev-only `retrieval_context_miss`
根因审计。这个阶段只回答一个问题：Stage102 中 gold context 没进 BM25 top10
上下文的 148 个 answerable case，主要是哪些根因。

本阶段不是候选策略实验，不选择 config，不调阈值，不改变 runtime 默认策略。

### 新增内容

新增审计实现：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_context_miss_root_cause_audit.py
```

新增运行脚本：

```text
scripts/run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py
```

新增测试：

```text
tests/test_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py
```

新增正式记录：

```text
docs/primeqa_hybrid_retrieval_context_miss_root_cause_audit.md
```

同时更新索引：

```text
docs/data_strategy.md
docs/evaluation_strategy.md
```

### 真实运行结果

命令：

```text
python scripts\run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py --user-confirmed-audit --confirmation-note "user confirmed Stage112 train-dev retrieval-context-miss root-cause audit on 2026-07-16; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

输出：

```text
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112.json
```

结果：

```text
stage: Stage 112
analysis_id: primeqa_hybrid_retrieval_context_miss_root_cause_audit_v1
status: primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed
guard checks: 19 / 19 passed
timing total: 235.469s
```

本阶段只加载：

```text
Stage111 frozen protocol
Stage68 train split
Stage68 dev split
PrimeQA training/dev corpus sections
```

没有加载 test split，没有运行 final metrics，没有做 dev-only selection，没有 threshold
tuning，没有修改 runtime defaults，没有加入 fallback strategies。

### 数据规模

真实加载：

```text
document_count: 28482
section_count: 216648
train_rows: 562
dev_rows: 121
train_answerable_rows: 370
dev_answerable_rows: 76
test_split_loaded: false
```

审计集合与 Stage102 `retrieval_context_miss` 完全对齐：

```text
train: 125
dev: 23
total: 148
```

answerable 条件下比例：

```text
train: 125 / 370 = 0.3378
dev: 23 / 76 = 0.3026
total: 148 / 446 = 0.3318
```

### 根因分布

train/dev 合计 primary root cause：

```text
title_heading_mismatch: 74
query_expression_gap: 65
long_document_score_dilution: 4
entity_version_error_code_mismatch: 3
bm25_field_weighting_or_index_structure: 2
```

分 split：

```text
train title_heading_mismatch: 63
train query_expression_gap: 55
train long_document_score_dilution: 4
train bm25_field_weighting_or_index_structure: 2
train entity_version_error_code_mismatch: 1

dev title_heading_mismatch: 11
dev query_expression_gap: 10
dev entity_version_error_code_mismatch: 2
```

共同出现在 train/dev 的 primary root cause：

```text
entity_version_error_code_mismatch
query_expression_gap
title_heading_mismatch
```

### 诊断信号

148 个 audit case 中的 high-signal dimension：

```text
title_heading_mismatch: 137
bm25_field_weighting_or_index_structure: 121
entity_version_error_code_mismatch: 80
query_expression_gap: 65
long_document_score_dilution: 37
section_boundary_or_span_locality: 10
```

gold document diagnostic rank bucket：

```text
not_found_top50: 110
rank_21_to_50: 24
rank_11_to_20: 14
```

question route 分布：

```text
other: 57
error_or_log: 40
install_upgrade_config: 32
how_to_or_lookup: 12
security_bulletin_vulnerability_detail: 4
limitation_or_restriction: 3
```

confidence band：

```text
high_multi_signal: 103
medium_two_signal: 39
low_single_signal: 6
```

### 解释

主要问题不是 answer composition，而是检索入口层面的 title/heading mismatch 和
query expression gap。下一步如果继续推进，应该先冻结一个 retrieval/index
redesign protocol，重点围绕：

```text
title/heading weighting
section-level indexing
entity/version/error-code handling
```

但这只是 Stage112 的审计解释，不是候选策略选择。当前仍不能 defaultize，不能跑
test，也不能用 dev 反选 config。

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_audit_case_counts_by_split.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_primary_root_cause_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_dimension_high_signal_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_gold_rank_bucket_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_question_route_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_root_cause_audit_stage112_visuals\stage112_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：实现时 answerable 计数一开始写成了
  `sum(sample.answerable and sample.answer_doc_id ...)`。
  - 原因：Python `and` 会返回 doc id 字符串，导致 `sum` 出现 int 和 str 相加。
  - 修正：改为显式 `sum(1 for sample in samples if ...)`。
- 问题 2：单测中 split guard 一开始按 `train, dev` 顺序比较，但观测值排序后是
  `dev, train`。
  - 原因：guard 的目标是集合一致，不应该依赖顺序。
  - 修正：期望值也排序后比较。
- 问题 3：Stage112 需要 `question_route`，但 Stage111 合同没有允许读 Stage69
  candidate artifact。
  - 原因：额外读取 Stage69 会扩大 Stage111 冻结的输入边界。
  - 修正：使用已有 `classify_question_route` 从 runtime-visible question title/text
    直接计算，不读 Stage69。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_context_miss_root_cause_audit.py scripts\run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py tests\test_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py
python -m pytest tests\test_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py -q
python scripts\run_primeqa_hybrid_retrieval_context_miss_root_cause_audit.py --user-confirmed-audit ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage112 run: passed
guard checks: 19 / 19 passed
```

全仓验证：

```text
python -m ruff check .
python -m pytest -q
```

结果：

```text
ruff: passed
pytest: 306 passed
```

### 我学到了

- `retrieval_context_miss` 的最大根因不是单纯 top10 rank near miss；110 / 148
  个 case 在 top50 里都找不到 gold doc。
- train 和 dev 的 dominant primary root cause 都集中在
  `title_heading_mismatch` 与 `query_expression_gap`，说明下一步要做 retrieval/index
  protocol，而不是继续 answer-pipeline 阈值。
- `bm25_field_weighting_or_index_structure` 虽然 primary count 只有 2，但 high-signal
  count 有 121，说明它更像二级结构信号，适合纳入下一步协议设计。
- 当前仍然没有使用 test；test 继续锁住。

### 下一步

Stage113：只有用户确认后，冻结 train/dev-only retrieval/index redesign protocol。
建议围绕：

```text
title/heading weighting
section-level indexing
entity/version/error-code handling
```

继续保持：

```text
test locked
no final metrics
no dev-only selection
no threshold tuning
runtime defaults unchanged
no fallback strategies
```

## Stage111：冻结 retrieval_context_miss 根因审计协议

### 目标

用户选择路线 A 后，本阶段只做协议冻结，不运行审计实验。目标是把
`retrieval_context_miss` 的下一轮 train/dev-only 根因分析边界先写死，避免
Stage112 在读取 corpus 和 gold doc id 之后，把离线诊断信号误用成 runtime
策略。

### 新增内容

新增协议实现：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_context_miss_audit_protocol.py
```

新增运行脚本：

```text
scripts/freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py
```

新增测试：

```text
tests/test_primeqa_hybrid_retrieval_context_miss_audit_protocol.py
```

新增正式记录：

```text
docs/primeqa_hybrid_retrieval_context_miss_audit_protocol.md
```

同时更新索引：

```text
docs/data_strategy.md
docs/evaluation_strategy.md
```

### 真实运行结果

命令：

```text
python scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py --user-confirmed-protocol --confirmation-note "user selected route A and confirmed Stage111 retrieval-context-miss root-cause audit protocol on 2026-07-16; protocol freeze only; test locked; runtime defaults unchanged; no fallback strategies"
```

输出：

```text
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111.json
```

结果：

```text
stage: Stage 111
protocol_id: primeqa_hybrid_retrieval_context_miss_audit_protocol_v1
route_id: retrieval_context_miss_root_cause_audit_protocol
status: primeqa_hybrid_retrieval_context_miss_audit_protocol_frozen
recommended_next_direction: run_retrieval_context_miss_root_cause_audit_train_dev
guard checks: 24 / 24 passed
timing total: 0.002s
```

本阶段只读取 Stage102、Stage107、Stage110 的 public-safe 报告，没有读取
train/dev/test split 文件，没有读取 corpus 文档，没有运行 retrieval 或 answer
metrics，也没有打开最终 test gate。

### 冻结的审计维度

Stage112 允许审计但不能 runtime 使用的维度：

```text
query_expression_gap
title_heading_mismatch
section_boundary_or_span_locality
long_document_score_dilution
entity_version_error_code_mismatch
bm25_field_weighting_or_index_structure
```

这些维度都是离线诊断用，不是候选策略，也不是上线特征。

### 使用的证据

Stage102 说明 retrieval_context_miss 仍是明显失败桶：

```text
train_retrieval_context_miss_count: 125
dev_retrieval_context_miss_count: 23
train_answerability_false_answer_count: 180
dev_answerability_false_answer_count: 41
train_verified_average_token_f1: 0.2017
dev_verified_average_token_f1: 0.204
```

Stage107 说明 dev answerable failure 中存在 gold context absent：

```text
answerable_gold_context_absent_count: 23
answerable_gold_context_absent_rate: 0.3026
answerable_gold_context_present_count: 53
context_present_failure_count: 53
```

Stage110 说明上一条 failure-pattern redesign family 已停止：

```text
decision_status: primeqa_hybrid_failure_pattern_redesign_family_stopped
stage109_selectable_config_count: 0
stage109_config_count: 7
requires_user_confirmation_before_next_protocol: true
```

### Stage112 合同

Stage112 只有在用户确认后才允许执行。确认后可读取：

```text
Stage111 frozen protocol
Stage68 train split
Stage68 dev split
PrimeQA training/dev corpus sections
Stage102 public-safe report for baseline bucket targets
```

继续禁止：

```text
test split
final-test labels
runtime oracle document identifiers
raw question, answer, or document text in public outputs
```

关键边界：

```text
reported_splits: train, dev
retrieval_depth_for_diagnostic_only: 50
stage112_may_use_gold_doc_id_for_offline_labeling: true
gold_doc_id_allowed_as_runtime_feature: false
selection_or_threshold_tuning_allowed: false
candidate_defaultization_allowed: false
final_test_metrics_allowed: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

### 可视化结果

本阶段生成了 5 个 SVG 诊断图：

```text
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_retrieval_context_miss_counts.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_audit_dimension_priorities.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_stage112_data_access_contract.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_protocol_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_context_miss_audit_protocol_stage111_visuals\stage111_guard_check_status.svg
```

### 问题、原因和处理

问题：Stage110 之后不能直接设计新 retrieval 策略，否则会跳过根因验证。

原因：Stage109 已证明 answer-pipeline 方向可以降低部分失败桶，但代价是过多拒答
answerable 样本；现在更关键的问题是 gold context 为什么没进检索上下文。

处理：先冻结 Stage111 协议，把 Stage112 的可读数据、禁止数据、输出字段、
gold doc id 使用边界和 visualizations 全部写清楚。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_context_miss_audit_protocol.py scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py tests\test_primeqa_hybrid_retrieval_context_miss_audit_protocol.py
python -m pytest tests\test_primeqa_hybrid_retrieval_context_miss_audit_protocol.py -q
python scripts\freeze_primeqa_hybrid_retrieval_context_miss_audit_protocol.py --user-confirmed-protocol ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage111 run: passed
guard checks: 24 / 24 passed
```

全仓验证：

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 303 passed
git diff --check: passed
```

### 我学到了

- 现在不能急着做新 retrieval 策略，必须先知道 `retrieval_context_miss` 是查询表达、
  标题/heading、section locality、长文档稀释、实体版本码，还是 index 结构问题。
- gold doc id 可以用于 Stage112 离线标注，但必须明确禁止进入 runtime 特征。
- 协议冻结阶段也要生成可视化，这样下一步执行审计时能直接对照边界。
- 当前仍然没有使用 test；test 继续锁住。

### 下一步

Stage112：在用户确认后，运行 frozen train/dev-only
`retrieval_context_miss` 根因审计。要求：

```text
use train/dev only
report train/dev separately
gold doc id offline labeling only
no test
no final metrics
no dev-only selection
no threshold tuning
runtime defaults unchanged
no fallback strategies
```

## Stage107：冻结并运行 validation failure pattern analysis

### 目标

按用户要求先“冻结一下”，再查找当前验证集失败规律。本阶段把 Stage107 定义为 public-safe diagnostic：

```text
只读 Stage102 / Stage105 / Stage106 已保存 JSON 报告
不读取 train/dev/test split 文件
不读取 corpus documents
不重新运行 retrieval metrics
不重新运行 answer metrics
不运行 final test metrics
不从 dev-only observations 反选 config
不改 runtime default
不加入 fallback strategy
```

### 新增内容

代码：

```text
src\ts_rag_agent\application\primeqa_hybrid_validation_failure_pattern_analysis.py
scripts\analyze_primeqa_hybrid_validation_failure_patterns.py
tests\test_primeqa_hybrid_validation_failure_pattern_analysis.py
```

文档：

```text
docs\primeqa_hybrid_validation_failure_pattern_analysis.md
docs\learning_journal.md
```

本地产物：

```text
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107.json
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107.console.txt
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_failure_bucket_counts.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_train_dev_bucket_rate_drift.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_route_failure_counts.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_dev_answerable_failure_flow.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_stage105_candidate_behavior.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_decision_flags.svg
artifacts\primeqa_hybrid_validation_failure_pattern_analysis_stage107_visuals\stage107_guard_check_status.svg
```

### 冻结协议

```text
protocol_id: primeqa_hybrid_validation_failure_pattern_analysis_v1
validation_split: dev
source reports only: Stage102, Stage105, Stage106
case-level rows written: false
aggregate counts only: true
select config from dev: false
retune thresholds on dev: false
open runtime gate: false
open final test gate: false
```

### 真实运行命令

```text
python scripts\analyze_primeqa_hybrid_validation_failure_patterns.py --user-confirmed-analysis --confirmation-note "user confirmed Stage107 frozen validation-failure pattern analysis on 2026-07-16; read saved Stage102/105/106 public-safe reports only; test locked; runtime defaults unchanged; no fallback strategies"
```

真实运行结果：

```text
exit_code: 0
guard_checks: 19 / 19 passed
```

### 验证集失败规律

整体：

```text
dev rows: 121
dev failure rows: 117 / 121 = 0.9669
answerable dev failure rows: 76 / 76 = 1.0000
unanswerable false-answer rows: 41 / 45 = 0.9111
```

主要 failure buckets：

```text
answerability_false_answer: 41, overall 0.3388, unanswerable conditional 0.9111
gold_span_beats_selected_answer: 41, overall 0.3388, answerable conditional 0.5395
retrieval_context_miss: 23, overall 0.1901, answerable conditional 0.3026
evidence_selection_miss: 12, overall 0.0992, answerable conditional 0.1579
```

answerable dev flow：

```text
answerable rows: 76
gold context absent from top10: 23 / 76 = 0.3026
gold context present in top10: 53 / 76 = 0.6974
context-present evidence selection miss: 12 / 53 = 0.2264
context-present gold span beats selected answer: 41 / 53 = 0.7736
answerable supported-and-cited rows: 0
```

这说明召回仍然是问题，但即使 gold context 已经进入 top10，当前 answer pipeline 也没有把它稳定转成正确 supported answer。

Train/dev 相似性：

```text
answerability_false_answer: train 0.3203, dev 0.3388, dev-train +1.85 pp
gold_span_beats_selected_answer: train 0.3096, dev 0.3388, dev-train +2.92 pp
retrieval_context_miss: train 0.2224, dev 0.1901, dev-train -3.23 pp
evidence_selection_miss: train 0.1192, dev 0.0992, dev-train -2.00 pp
```

这说明当前 dev 失败结构不是明显的 dev-only anomaly。

Route 集中度：

```text
other: 45 / 46 failures, dominant gold_span_beats_selected_answer
error_or_log: 25 / 25 failures, dominant answerability_false_answer
install_upgrade_config: 16 / 17 failures, dominant answerability_false_answer
how_to_or_lookup: 14 / 15 failures, dominant gold_span_beats_selected_answer
security_bulletin_vulnerability_detail: 13 / 14 failures, dominant gold_span_beats_selected_answer
```

### Stage105 candidate 失败规律

Stage105 的 train-selected config：

```text
selected_config_id: amg_bm25_evidence8_rank3_v1
selected_train_weighted_target_delta: 0.0
dev_weighted_target_delta: 0.0
dev_changed_answer_count: 0
```

候选族分成三类：

```text
train_selectable_noop_or_near_noop: 2 configs, dev changed answers 0, best dev delta 0.00
train_nonselectable_low_change: 1 config, dev changed answers 4, best dev delta -3.10
train_nonselectable_high_change: 6 configs, dev changed answers 695 total, best dev delta -29.15
```

所有 7 个 dev-better non-selectable configs 都失败于 train answerable-refusal guard，其中 4 个还失败于 train gold-citation guard。

### 决策

```text
status: primeqa_hybrid_validation_failure_pattern_analysis_completed
protocol_id: primeqa_hybrid_validation_failure_pattern_analysis_v1
validation_split: dev
dev_failure_count: 117
dev_failure_rate: 0.9669
answerable_failure_rate: 1.0
unanswerable_false_answer_rate: 0.9111
stage105_selected_config_was_dev_noop: true
recommended_next_direction: failure_pattern_driven_train_dev_redesign_protocol
can_continue_train_dev_development: true
requires_user_confirmation_before_next_protocol: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

结论：Stage107 只是诊断阶段，不支持 runtime defaultization，也不能打开 final-test gate。

### 问题、原因与修正

- 问题 1：用户要“冻结”和“找规律”，这可能被误解成先做 protocol freeze、下阶段再分析。
  - 原因：前面很多阶段把 freeze 和 metric run 分成两个 stage。
  - 修正：本阶段冻结的是 diagnostic protocol，并立即对已保存 public-safe reports 做只读聚合分析；没有新增 candidate comparison，也没有运行新指标。
- 问题 2：如果输出 case-level rows，可能泄露验证集样本内容或诱导人工反选。
  - 原因：Stage102 有 public-safe case samples，但 Stage107 的目标是看整体规律，不需要逐题样本。
  - 修正：Stage107 协议明确 `case-level rows written: false`，只输出 aggregate counts。
- 问题 3：dev 上 high-change configs 看起来有收益，容易再次诱导 dev selection。
  - 原因：Stage105 的 aggressive window/joint configs 在 dev target 上更好，但 train guards 已经证明风险不可接受。
  - 修正：Stage107 只把它们归类为失败规律证据，仍不允许反选，也不改变 runtime default。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_validation_failure_pattern_analysis.py scripts\analyze_primeqa_hybrid_validation_failure_patterns.py tests\test_primeqa_hybrid_validation_failure_pattern_analysis.py
pytest -q tests\test_primeqa_hybrid_validation_failure_pattern_analysis.py
python scripts\analyze_primeqa_hybrid_validation_failure_patterns.py --user-confirmed-analysis ...
```

结果：

```text
ruff: passed
pytest: 2 passed
Stage107 run: passed
guard checks: 19 / 19 passed
```

### 我学到了

- 当前验证集失败不是小修小补能解释的：answerable dev 侧 76/76 都进了 failure bucket，unanswerable 侧 41/45 被误答。
- 召回没完全解决，但 answer pipeline 本身也很关键：53 个 gold context 已在 top10 的 answerable dev 样本，仍全部在 evidence selection 或 answer composition 阶段失败。
- Stage105 的失败不是“没有任何方向”，而是“可改变 dev 的候选在 train guard 下不可选”；下一步应该先围绕失败机制设计新协议，而不是调现有阈值。
- 可视化能帮助把问题从“某个 config 没过”转成“失败结构是什么”：bucket counts、train-dev drift、route concentration、answerable flow、candidate behavior 都要一起看。

### 下一步

Stage108：在用户确认后，冻结 train/dev-only failure-pattern-driven redesign protocol。它应围绕 Stage107 暴露的失败机制设计，而不是从 dev 反选现有 config；继续保持：

```text
test locked
no final metrics
no dev-only selection
runtime defaults unchanged
no fallback strategies
```

## Stage108：冻结 failure-pattern-driven redesign protocol

### 目标

按 Stage107 的结论继续下一步，但本阶段只做 protocol freeze，不跑指标、不实现 runtime、不打开 test gate。Stage108 的边界：

```text
只读 Stage107 public-safe report
不读取 train/dev/test split 文件
不读取 corpus documents
不重新运行 retrieval metrics
不重新运行 answer metrics
不运行 final test metrics
不从 dev-only observations 反选 config
不改 runtime default
不加入 fallback strategy
```

### 新增内容

代码：

```text
src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_protocol.py
scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py
tests\test_primeqa_hybrid_failure_pattern_redesign_protocol.py
```

文档：

```text
docs\primeqa_hybrid_failure_pattern_redesign_protocol.md
docs\learning_journal.md
```

本地产物：

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108.json
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108.console.txt
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_candidate_family_priorities.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_candidate_config_counts.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_target_bucket_weights.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_train_cv_guard_thresholds.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_protocol_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_protocol_stage108_visuals\stage108_guard_check_status.svg
```

### 真实运行命令

```text
python scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage108 failure-pattern redesign protocol freeze on 2026-07-16 after Stage107 validation-failure analysis; test locked; runtime defaults unchanged; no fallback strategies"
```

真实运行结果：

```text
exit_code: 0
guard_checks: 23 / 23 passed
```

### Stage107 依据

Stage108 只使用 Stage107 的 public-safe aggregate findings：

```text
dev_failure_count: 117
dev_failure_rate: 0.9669
answerable_failure_rate: 1.0
unanswerable_false_answer_rate: 0.9111
answerable_gold_context_absent_rate: 0.3026
context_present_gold_span_beats_selected_rate: 0.7736
context_present_evidence_selection_miss_rate: 0.2264
stage105_selected_config_was_dev_noop: true
stage105_dev_better_nonselectable_config_count: 7
```

核心设计判断：

```text
retrieval miss 仍存在，但 53 个 answerable dev rows 的 gold context 已经在 top10 中，
仍全部在 evidence selection 或 answer composition 阶段失败。
所以 Stage109 先比较 answer-pipeline redesign，同时把 retrieval_context_miss 作为监控边界。
```

### 冻结候选族

Stage108 冻结 3 个 candidate families，合计 7 个 configs：

```text
support_aware_answerability_gate_candidate_v1: 2 configs
context_present_span_composer_candidate_v1: 3 configs
joint_support_gate_span_composer_candidate_v1: 2 configs
```

具体 config：

```text
saag_support2_evidence7_rank3_v1
saag_support2_evidence6_rank5_v1
cpsc_anchor_top2_mcpd3_rank3_v1
cpsc_anchor_top3_mcpd3_rank3_v1
cpsc_title_query_anchor_top2_mcpd3_rank3_v1
jsgc_support2_evidence7_anchor_top2_v1
jsgc_support2_evidence6_title_anchor_top2_v1
```

这些 config 不复用已经停止的 Stage104 config IDs（`amg_*`, `ewr_*`, `jgw_*`），也不是 runtime default，只是 Stage109 的 comparison grid。

### Train selection 规则

Stage109 必须使用 train grouped CV 选择：

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
train_cv_fold_count: 5
dev_config_selection_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
```

Train-CV objective：

```text
1.75 * answerability_false_answer
1.80 * gold_span_beats_selected_answer
1.50 * evidence_selection_miss
```

No-op candidate 不可选：

```text
requires_negative_train_cv_weighted_delta: true
no_op_candidate_selectable: false
```

### Guard

Stage109 train-CV selectability guards：

```text
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_retrieval_context_miss_delta: 0
```

Dev validation 规则：

```text
validation_split: dev
validated_item: single train-CV-selected config
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
```

### 决策

```text
status: primeqa_hybrid_failure_pattern_redesign_protocol_frozen
protocol_id: primeqa_hybrid_failure_pattern_redesign_protocol_v1
recommended_next_direction: run_failure_pattern_redesign_train_cv_dev_validation
candidate_family_count: 3
candidate_config_count: 7
train_selection_mode: train_grouped_cross_validation_then_full_train_refit
can_run_train_dev_comparison_after_user_confirmation: true
can_continue_train_dev_development: true
requires_user_confirmation_before_train_dev_run: true
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

结论：Stage108 是协议冻结，不支持 runtime defaultization，也不能打开 final-test gate。

### 问题、原因与修正

- 问题 1：Stage107 显示 dev 失败很强，但不能直接把 dev 观察转成 config 选择。
  - 原因：dev 仍然只能做 validation；直接按 dev 现象挑阈值会变成 dev selection。
  - 修正：Stage108 把 Stage109 选择规则改为 train grouped CV，只允许 dev 做一次 validation。
- 问题 2：Stage105 的 train-selectable config 是 no-op。
  - 原因：原协议允许“通过 guard 但 target delta 为 0”的 config 进入 dev validation。
  - 修正：Stage108 明确 `requires_negative_train_cv_weighted_delta: true` 和 `no_op_candidate_selectable: false`。
- 问题 3：Stage105 aggressive configs 在 dev 上更好，但 train guard 风险明显。
  - 原因：它们主要违反 answerable-refusal guard，部分还违反 gold-citation guard。
  - 修正：Stage108 把 train-CV answerable refusal 和 citation guard 收紧，并要求 Stage109 报告这些风险。
- 问题 4：retrieval_context_miss 仍存在，但本阶段不是重启 retrieval 路线。
  - 原因：Stage84 第二波 retrieval candidate queue 已经耗尽；Stage107 同时说明 context-present answer pipeline failure 更突出。
  - 修正：Stage108 把 retrieval_context_miss 定为 monitored boundary，不把它作为当前 answer-pipeline redesign 的直接 target。

### 验证

局部验证：

```text
ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_protocol.py scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py tests\test_primeqa_hybrid_failure_pattern_redesign_protocol.py
pytest -q tests\test_primeqa_hybrid_failure_pattern_redesign_protocol.py
python scripts\freeze_primeqa_hybrid_failure_pattern_redesign_protocol.py --user-confirmed-protocol ...
```

结果：

```text
ruff: passed
pytest: 4 passed
Stage108 run: passed
guard checks: 23 / 23 passed
```

### 我学到了

- “找出失败规律”之后不能马上实现一个看起来合理的策略；必须先冻结选择口径，尤其要防止 dev 反选。
- 这一次必须把 train grouped CV 写进协议，因为 Stage105 已经证明单次 train selection 容易选到 no-op。
- No-op 不应再被视为可推进候选：没有负向 train-CV target delta 的 config 不能进入 dev validation。
- 当前设计应优先处理 context-present 的 answer pipeline failure，同时继续监控 retrieval_context_miss，避免把 retrieval 和 answer pipeline 两条问题线混在一起。

### 下一步

Stage109：在用户确认后，实现 Stage108 冻结的 candidate components，并运行 train grouped-CV + dev validation comparison。Stage109 必须继续保持：

```text
test locked
no final metrics
train-CV selection only
dev validation only
runtime defaults unchanged
no fallback strategies
```

## Stage109：运行 failure-pattern redesign train-CV/dev comparison

### 目标

按 Stage108 冻结的协议继续下一步，真正实现 3 个 candidate families / 7 个 configs，并运行：

```text
train grouped-CV selection only
dev single validation only
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

本阶段不是 runtime defaultization，也不是 final-test gate。

### 本阶段新增内容

新增：

```text
src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_comparison.py
scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py
tests\test_primeqa_hybrid_failure_pattern_redesign_comparison.py
docs\primeqa_hybrid_failure_pattern_redesign_comparison.md
```

真实运行输出：

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109.json
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\
```

注意：`artifacts/` 目录按仓库规则被 `.gitignore` 忽略；提交中记录的是代码、脚本、测试和文档摘要，不把 artifact JSON/SVG 作为 tracked 文件。

### 实现边界

Stage109 只读取：

```text
Stage108 frozen protocol
Stage102 public-safe decomposition report
Stage68 train split
Stage68 dev split
PrimeQA training/dev corpus sections
```

没有读取 test split，没有运行 final metrics，没有把 dev 用于 config selection 或 threshold tuning。

Train grouped-CV 的 grouping 内部使用：

```text
normalized question + answer document / unanswerable marker
```

报告只写 fold counts，不写 raw group values。

### 真实数据

```text
documents: 28482
train rows: 562
train answerable rows: 370
train unanswerable rows: 192
dev rows: 121
dev answerable rows: 76
dev unanswerable rows: 45
```

Train grouped-CV：

```text
fold_count: 5
fold_1: 113 rows / 107 groups
fold_2: 113 rows / 107 groups
fold_3: 112 rows / 106 groups
fold_4: 112 rows / 106 groups
fold_5: 112 rows / 107 groups
raw_group_values_written: false
```

### Baseline

Stage109 复现的 Stage102 verified BM25 top10 answer-pipeline baseline：

```text
train-CV weighted target score: 728.70
train-full weighted target score: 728.70
dev weighted target score: 163.55
```

Train-CV verified：

```text
answerable_refusal_rate: 0.0459
unanswerable_refusal_rate: 0.0625
gold_doc_citation_rate: 0.4958
average_token_f1: 0.2017
```

Dev verified：

```text
answerable_refusal_rate: 0.1053
unanswerable_refusal_rate: 0.0889
gold_doc_citation_rate: 0.6029
average_token_f1: 0.2040
```

### Train-CV 结果

7 个 frozen configs 的 train-CV ranking：

| Rank | Config | Train-CV delta | Selectable | Changed answers |
| ---: | --- | ---: | --- | ---: |
| 1 | jsgc_support2_evidence7_anchor_top2_v1 | -199.65 | false | 534 |
| 2 | cpsc_title_query_anchor_top2_mcpd3_rank3_v1 | -123.05 | false | 543 |
| 3 | jsgc_support2_evidence6_title_anchor_top2_v1 | -115.90 | false | 551 |
| 4 | saag_support2_evidence7_rank3_v1 | -111.15 | false | 117 |
| 5 | saag_support2_evidence6_rank5_v1 | -31.65 | false | 49 |
| 6 | cpsc_anchor_top2_mcpd3_rank3_v1 | -28.50 | false | 533 |
| 7 | cpsc_anchor_top3_mcpd3_rank3_v1 | 0.00 | false | 0 |

结论：

```text
selected_config_id: null
selectable_config_count: 0 / 7
```

虽然多数 config 都让 weighted target score 下降，但全部被 frozen train-CV selectability guard 挡住。

### 失败规律

主要失败原因是 answerable refusal 增加太多。

Frozen guard：

```text
max_train_cv_answerable_refusal_rate_delta: 0.02
```

实际：

```text
jsgc_support2_evidence7_anchor_top2_v1: +0.3784
cpsc_title_query_anchor_top2_mcpd3_rank3_v1: +0.3027
jsgc_support2_evidence6_title_anchor_top2_v1: +0.2352
saag_support2_evidence7_rank3_v1: +0.1919
saag_support2_evidence6_rank5_v1: +0.0568
cpsc_anchor_top2_mcpd3_rank3_v1: +0.0541
cpsc_anchor_top3_mcpd3_rank3_v1: +0.0000
```

`cpsc_anchor_top3_mcpd3_rank3_v1` 是 no-op：

```text
train-CV delta: 0.00
changed answers: 0
failed guard: train_cv_weighted_target_delta_negative
```

因此它也不能被选中。

### Dev validation

Dev 没有被用于选择或调参。因为 train-CV 没有选出任何 config，所以没有 selected config 可以进入 dev validation：

```text
selected_config_id: null
status: no_train_cv_selectable_config
dev_validation_passed: false
```

部分 config 在 dev 上也有 weighted target delta 改善，但它们已经在 train-CV guard 上失败，不能从 dev 反选。

### 决策

```text
status: primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config
selected_config_id: null
selectable_config_count: 0
dev_validation_passed: false
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
recommended_next_direction: record_failure_pattern_redesign_stop_decision
```

Stage109 不支持 runtime defaultization，也不能打开 final-test gate。

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_train_cv_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_dev_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_train_cv_selectability.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_changed_answer_counts.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_dev_metric_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_comparison_stage109_visuals\stage109_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：所有 candidate 都不能进入 dev selected validation。
  - 原因：没有任何 config 同时满足负向 train-CV target delta 和 frozen selectability guards。
  - 修正：如实记录为 no selectable config，不从 dev 反选。
- 问题 2：多数 target-score 改善来自更强的拒答。
  - 原因：support gate 和 joint gate 降低了 false-answer / span-miss buckets，但显著增加 answerable refusal。
  - 修正：保留 guard 阻断，不放宽阈值，不加入兜底策略。
- 问题 3：一个 config 是 no-op。
  - 原因：`cpsc_anchor_top3_mcpd3_rank3_v1` 与 baseline 结果等价。
  - 修正：Stage108 的 no-op block 生效，不能把它作为可推进 config。

### 验证

局部验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_comparison.py scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py tests\test_primeqa_hybrid_failure_pattern_redesign_comparison.py
python -m pytest tests\test_primeqa_hybrid_failure_pattern_redesign_comparison.py -q
python scripts\run_primeqa_hybrid_failure_pattern_redesign_comparison.py --user-confirmed-comparison ...
```

结果：

```text
ruff: passed
pytest: 2 passed
Stage109 run: passed
guard checks: 20 / 20 passed
runtime defaults: unchanged
fallback strategies: disabled
test/final metrics: locked
```

全仓验证：

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 297 passed
git diff --check: passed
```

### 我学到了

- train-CV 的价值在这里很明确：dev 上看起来有改善的 config，只要 train-CV guard 不过，就不能推进。
- 当前这一批 redesign 仍然把“减少 false answer”变成了“更多 answerable refusal”，说明简单 support gate / anchor composer 不是可默认化路线。
- no-op 不再能作为“安全但没收益”的候选进入下一关，这是 Stage108 协议里最关键的修正之一。
- 下一步不是调阈值救这个 family，也不是看 test，而是先记录 stop decision，明确为什么这条 frozen redesign family 停止。

### 下一步

Stage110：记录 frozen Stage108 failure-pattern redesign family 的 stop decision。必须说明：

```text
all 7 configs failed train-CV selectability
dev cannot be used to rescue nonselectable configs
test remains locked
runtime defaults remain unchanged
fallback strategies remain disabled
```

## Stage110：记录 failure-pattern redesign family stop decision

### 目标

Stage110 不再跑新指标，而是把 Stage109 的真实结果正式转成 stop decision：

```text
all 7 configs failed train-CV selectability
dev cannot rescue train-CV nonselectable configs
test remains locked
runtime defaults remain unchanged
fallback strategies remain disabled
```

本阶段只读 Stage109 public-safe report，不读取 split 文件，不读取 corpus documents，不运行 retrieval / answer metrics。

### 本阶段新增内容

新增：

```text
src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_stop_decision.py
scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py
tests\test_primeqa_hybrid_failure_pattern_redesign_stop_decision.py
docs\primeqa_hybrid_failure_pattern_redesign_stop_decision.md
```

更新：

```text
docs\data_strategy.md
docs\evaluation_strategy.md
docs\learning_journal.md
```

真实运行输出：

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110.json
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\
```

### 真实 stop 证据

Stage109 已确认：

```text
status: primeqa_hybrid_failure_pattern_redesign_completed_no_train_cv_selectable_config
selected_config_id: null
selectable_config_count: 0 / 7
dev_validation_status: no_train_cv_selectable_config
dev_validation_passed: false
guard checks: 20 / 20 passed
```

Stage110 重新读取 Stage109 JSON 后记录：

```text
status: primeqa_hybrid_failure_pattern_redesign_family_stopped
stopped_family_id: failure_pattern_redesign_candidate_family
stopped_protocol_id: primeqa_hybrid_failure_pattern_redesign_protocol_v1
stopped_analysis_id: primeqa_hybrid_failure_pattern_redesign_train_cv_dev_validation_v1
current_route_defaultization: blocked
redesign_required_before_any_runtime_or_test_gate: true
recommended_next_direction: user_confirmed_next_research_direction_required
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

### Family summary

```text
joint_support_gate_span_composer_candidate_v1:
  configs: 2
  train-CV selectable: 0
  best train-CV delta: -199.65
  best dev delta: -48.25
  max answerable refusal delta: +0.3784

context_present_span_composer_candidate_v1:
  configs: 3
  train-CV selectable: 0
  best train-CV delta: -123.05
  best dev delta: -47.15
  max answerable refusal delta: +0.3027

support_aware_answerability_gate_candidate_v1:
  configs: 2
  train-CV selectable: 0
  best train-CV delta: -111.15
  best dev delta: -24.75
  max answerable refusal delta: +0.1919
```

解释：target-score improvement 是真实存在的，但它主要来自更强拒答；在 frozen guard 下不能推进。

### Dev-improved but nonselectable

以下 config 在 dev 上有 weighted target delta 改善，但都 train-CV nonselectable，不能从 dev 反选：

| Config | Dev delta | Train-CV delta | Failed train-CV guards |
| --- | ---: | ---: | --- |
| jsgc_support2_evidence7_anchor_top2_v1 | -48.25 | -199.65 | answerable_refusal_rate_delta |
| cpsc_title_query_anchor_top2_mcpd3_rank3_v1 | -47.15 | -123.05 | answerable_refusal_rate_delta |
| jsgc_support2_evidence6_title_anchor_top2_v1 | -34.95 | -115.90 | answerable_refusal_rate_delta |
| saag_support2_evidence7_rank3_v1 | -24.75 | -111.15 | answerable_refusal_rate_delta |
| saag_support2_evidence6_rank5_v1 | -1.75 | -31.65 | answerable_refusal_rate_delta |
| cpsc_anchor_top2_mcpd3_rank3_v1 | -0.30 | -28.50 | answerable_refusal_rate_delta, average_token_f1_drop, gold_doc_citation_rate_drop |

No-op block：

```text
config_id: cpsc_anchor_top3_mcpd3_rank3_v1
train_cv_weighted_target_delta: 0.0
train_cv_changed_answer_count: 0
failed_train_cv_guard: train_cv_weighted_target_delta_negative
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_train_cv_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_dev_weighted_target_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_answerable_refusal_deltas.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_selectability_by_family.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_stop_decision_flags.svg
artifacts\primeqa_hybrid_failure_pattern_redesign_stop_decision_stage110_visuals\stage110_stop_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：dev 上看起来仍有强改善 config。
  - 原因：这些 config 全部 train-CV nonselectable，主要违反 answerable-refusal guard。
  - 修正：Stage110 明确把它们记录为 `dev_improved_train_cv_nonselectable_configs`，只能作为失败证据，不能作为选择依据。
- 问题 2：最强 train-CV target delta 容易被误读为“可推进”。
  - 原因：`jsgc_support2_evidence7_anchor_top2_v1` 的 `-199.65` target delta 同时带来 `+0.3784` answerable refusal delta。
  - 修正：Stage110 把 answerable refusal deltas 单独可视化，并在 stop reason 中明确该 family 停止。
- 问题 3：stop decision 容易偷偷决定下一条实验路线。
  - 原因：停止一个 family 后自然会想马上切到另一个策略。
  - 修正：Stage110 只记录 `user_confirmed_next_research_direction_required`，不擅自选择 Stage111 路线。

### 验证

局部验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_failure_pattern_redesign_stop_decision.py scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py tests\test_primeqa_hybrid_failure_pattern_redesign_stop_decision.py
python -m pytest tests\test_primeqa_hybrid_failure_pattern_redesign_stop_decision.py -q
python scripts\decide_primeqa_hybrid_failure_pattern_redesign_stop.py --user-confirmed-stop ...
git diff --check
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage110 run: passed
guard checks: 27 / 27 passed
git diff --check: passed
```

全仓验证：

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 300 passed
git diff --check: passed
```

### 我学到了

- Stop decision 的职责不是“找一个还能救的 config”，而是把不能推进的事实边界固定下来。
- dev-improved config 必须和 train-CV selectable config 分开看；否则很容易违反 train-first 规则。
- 这轮失败不是没有 signal，而是 signal 的代价太高：它把 answerable 问题拒答得太多。
- 下一步必须先让用户确认研究方向；Stage110 不能偷偷选择新路线。

### 下一步

Stage111：只有用户确认后，选择下一条 train/dev-only research direction。继续保持：

```text
test locked
no final metrics
no dev-only selection
runtime defaults unchanged
no fallback strategies
```

## 当前阶段尾部索引：Stage110 到 Stage112

说明：Stage110、Stage111、Stage112 都已经在本文件中记录；由于历史追加锚点重复，
这三段正文在文件中的物理顺序不是时间顺序。以当前真实提交顺序为准：

```text
Stage110: failure-pattern redesign family stop decision
Stage111: retrieval_context_miss audit protocol freeze
Stage112: retrieval_context_miss root-cause audit
```

Stage112 当前状态：

```text
status: primeqa_hybrid_retrieval_context_miss_root_cause_audit_completed
audit cases: train 125, dev 23, total 148
guard checks: 19 / 19 passed
full ruff: passed
full pytest: 306 passed
git diff --check: passed
test locked
runtime defaults unchanged
no fallback strategies
```

当前下一步：

```text
Stage113: 用户确认后，冻结 train/dev-only retrieval/index redesign protocol。
```

## Stage113：冻结 retrieval/index redesign protocol

### 目标

根据 Stage112 的 `retrieval_context_miss` 根因审计结果，冻结下一轮
train/dev-only retrieval/index redesign protocol。本阶段只定义候选族、候选
config、train-CV 选择规则和 dev validation 边界，不运行候选实验。

### 新增内容

新增协议实现：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_index_redesign_protocol.py
```

新增运行脚本：

```text
scripts/freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py
```

新增测试：

```text
tests/test_primeqa_hybrid_retrieval_index_redesign_protocol.py
```

新增正式记录：

```text
docs/primeqa_hybrid_retrieval_index_redesign_protocol.md
```

同时更新索引：

```text
docs/data_strategy.md
docs/evaluation_strategy.md
```

### 真实运行结果

命令：

```text
python scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage113 train-dev retrieval/index redesign protocol freeze on 2026-07-16 after Stage112 root-cause audit; protocol freeze only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

输出：

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113.json
```

结果：

```text
stage: Stage 113
protocol_id: primeqa_hybrid_retrieval_index_redesign_protocol_v1
status: primeqa_hybrid_retrieval_index_redesign_protocol_frozen
recommended_next_direction: run_retrieval_index_redesign_train_cv_dev_validation
candidate families: 3
candidate configs: 8
guard checks: 20 / 20 passed
timing total: 0.002s
```

本阶段只读取 Stage112 public-safe report，没有读取 train/dev/test split，没有读取 corpus，
没有运行 retrieval 或 answer metrics，也没有打开 final-test gate。

### Stage112 依据

Stage112 审计集合：

```text
train: 125
dev: 23
total: 148
```

Primary root cause：

```text
title_heading_mismatch: 74
query_expression_gap: 65
long_document_score_dilution: 4
entity_version_error_code_mismatch: 3
bm25_field_weighting_or_index_structure: 2
```

High-signal dimension：

```text
title_heading_mismatch: 137
bm25_field_weighting_or_index_structure: 121
entity_version_error_code_mismatch: 80
query_expression_gap: 65
long_document_score_dilution: 37
section_boundary_or_span_locality: 10
```

### 冻结的候选族

```text
title_heading_weighted_bm25_candidate_v1: priority 0.95, configs 3
section_level_index_rollup_candidate_v1: priority 0.90, configs 3
entity_version_error_code_handling_candidate_v1: priority 0.80, configs 2
```

候选 config：

```text
thw_title2_heading2_body1_doc_bm25_v1
thw_title3_heading2_body1_doc_bm25_v1
thw_title_heading_query_view_rrf_v1
slr_section_top1_doc_rollup_v1
slr_section_top3_rrf_doc_rollup_v1
slr_heading_section_title_rollup_v1
evc_special_token_exact_boost_v1
evc_special_token_title_heading_boost_v1
```

这些都是 Stage114 的候选协议项，本阶段没有实现或评估它们。

### 选择规则

Stage114 如果执行，必须使用：

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
train_group_key: normalized_question_plus_answer_document_or_technote
minimum_train_folds: 5
validation_split: dev
dev_validation_mode: single_pass_no_retuning
```

主要目标：

```text
reduce_retrieval_context_miss
required_train_cv_delta: negative
weight: 2.0
```

关键 guard：

```text
max_train_cv_average_token_f1_drop: 0.005
max_train_cv_gold_doc_citation_rate_drop: 0.015
max_train_cv_answerable_refusal_rate_delta: 0.02
max_train_cv_answerability_false_answer_delta: 0
max_train_cv_evidence_selection_miss_delta: 0
max_train_cv_gold_span_beats_selected_delta: 0
max_train_cv_changed_answer_rate: 0.25
```

继续禁止：

```text
dev_selection_allowed: false
dev_retuning_allowed: false
dev_threshold_tuning_allowed: false
test_access_allowed: false
final_test_metrics_allowed: false
test_tuning_allowed: false
default_runtime_policy: unchanged
fallback_strategies_enabled: false
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_stage112_primary_root_causes.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_stage112_high_signal_dimensions.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_candidate_family_priorities.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_candidate_config_counts.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_selection_guard_thresholds.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_protocol_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_protocol_stage113_visuals\stage113_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：一开始有几处候选 config 描述字符串超过 ruff 行宽。
  - 原因：协议文本写得太直接。
  - 修正：只做字符串折行，不改变协议内容。
- 问题 2：Stage113 容易被误解成已经开始 retrieval/index 实验。
  - 原因：本阶段已经列出了 8 个 config。
  - 修正：文档和 decision 中明确它们只是 frozen protocol candidates，真正运行要等 Stage114 用户确认。
- 问题 3：Stage113 不能直接使用 Stage112 的 gold doc id 诊断信号进入 runtime。
  - 原因：gold doc id 只允许离线标注，不允许做 runtime feature。
  - 修正：候选族的 forbidden runtime features 明确包含 gold document identifier、gold answer span、test labels。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_index_redesign_protocol.py scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py tests\test_primeqa_hybrid_retrieval_index_redesign_protocol.py
python -m pytest tests\test_primeqa_hybrid_retrieval_index_redesign_protocol.py -q
python scripts\freeze_primeqa_hybrid_retrieval_index_redesign_protocol.py --user-confirmed-protocol ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage113 run: passed
guard checks: 20 / 20 passed
```

全仓验证：

```text
python -m ruff check .
python -m pytest -q
git diff --check
```

结果：

```text
ruff: passed
pytest: 309 passed
git diff --check: passed
```

### 我学到了

- Stage112 的强信号可以转化为候选族，但不能直接转化为默认策略。
- 这次 protocol 的核心是 train grouped-CV 先筛选，dev 只做 single-pass validation。
- `title_heading_mismatch` 和 `query_expression_gap` 是主目标，`bm25_field_weighting_or_index_structure`
  是重要结构信号，适合和 section rollup 放在同一协议里比较。
- 当前仍然没有使用 test；test 继续锁住。

### 下一步

Stage114：只有用户确认后，运行 frozen train grouped-CV retrieval/index redesign
comparison，再做一次 dev validation。继续保持：

```text
test locked
no final metrics
no dev-only selection
no threshold tuning
runtime defaults unchanged
no fallback strategies
```

## 2026-07-16 - Stage 114 retrieval/index redesign train-CV/dev comparison

### 本阶段目标

在 Stage113 已冻结的 retrieval/index redesign protocol 下，真正运行 8 个候选
config：先用 train grouped-CV 做选择，再对 dev 做一次 report-only validation。

继续保持：

```text
test locked
no final metrics
no dev-only selection
no threshold tuning
runtime defaults unchanged
no fallback strategies
```

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_index_redesign_comparison.py
scripts/run_primeqa_hybrid_retrieval_index_redesign_comparison.py
tests/test_primeqa_hybrid_retrieval_index_redesign_comparison.py
```

文档：

```text
docs/primeqa_hybrid_retrieval_index_redesign_comparison.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_retrieval_index_redesign_comparison_stage114.json
artifacts/primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals/
```

### 实现口径

Stage114 实现了 Stage113 冻结的 8 个 retrieval/index 候选：

```text
thw_title2_heading2_body1_doc_bm25_v1
thw_title3_heading2_body1_doc_bm25_v1
thw_title_heading_query_view_rrf_v1
slr_section_top1_doc_rollup_v1
slr_section_top3_rrf_doc_rollup_v1
slr_heading_section_title_rollup_v1
evc_special_token_exact_boost_v1
evc_special_token_title_heading_boost_v1
```

其中：

- title/heading weighted BM25 用加权索引文本实现，返回原始文档用于回答。
- section top1 用 section BM25 后按父文档最大 section score rollup。
- section top3 RRF 先按 section 排名聚合到文档，再与文档 BM25 做 RRF。
- special-token 候选只使用运行时可见的问题文本、标题、section heading、正文中的特殊 token。
- 所有候选都不使用 gold document id、gold answer span、test label。

### 真实运行结果

命令：

```text
python scripts\run_primeqa_hybrid_retrieval_index_redesign_comparison.py --user-confirmed-comparison --confirmation-note "user confirmed Stage114 train grouped-CV retrieval/index redesign comparison on 2026-07-16; train selection only; one dev report; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

数据：

```text
documents: 28482
sections: 216648
train rows: 562
train answerable: 370
dev rows: 121
dev answerable: 76
train grouped-CV folds: 5
```

baseline：

```text
train-CV gold doc hit: 245 / 370
train-CV gold doc miss: 125 / 370
train-CV gold doc recall@10: 0.6622
train-CV verified F1: 0.2017
train-CV gold citation rate: 0.4958

dev gold doc hit: 53 / 76
dev gold doc miss: 23 / 76
dev gold doc recall@10: 0.6974
dev verified F1: 0.2040
dev gold citation rate: 0.6029
```

Stage114 selection：

```text
selectable configs: 0 / 8
decision status: primeqa_hybrid_retrieval_index_redesign_completed_no_train_cv_selectable_config
```

最佳 train-CV retrieval movement：

```text
evc_special_token_title_heading_boost_v1:
  retrieval_context_miss delta: -4
  gold doc recall@10 delta: +0.0108
  changed answer rate: 0.7278
  failed guards: evidence_selection_miss, gold_span_beats_selected_answer, changed_answer_rate

evc_special_token_exact_boost_v1:
  retrieval_context_miss delta: -4
  gold doc recall@10 delta: +0.0108
  changed answer rate: 0.1833
  failed guards: answerability_false_answer, evidence_selection_miss
```

Section-level candidates 在本实现下没有通过：

```text
slr_section_top3_rrf_doc_rollup_v1:
  retrieval_context_miss delta: +9
  recall@10 delta: -0.0244

slr_section_top1_doc_rollup_v1:
  retrieval_context_miss delta: +26
  recall@10 delta: -0.0703
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_retrieval_context_miss_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_gold_doc_recall_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_average_token_f1_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_train_cv_selectability.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_dev_retrieval_context_miss_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_dev_gold_doc_recall_delta.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_comparison_stage114_visuals\stage114_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：测试里的 synthetic grouped-CV 样本最初都落到同一个 fold。
  - 原因：测试样本使用了完全相同的 normalized question + answer document 分组键。
  - 修正：测试样本 question text 加入 scenario 序号，让 5 个 train 样本形成 5 个组。
- 问题 2：Stage114 guard 需要确认 raw group values 没有写入，但 public result 最初没有显式字段。
  - 原因：只有 data_summary 有 fold public-safe 摘要，baseline public result 里没有 mirror 字段。
  - 修正：在 public result 中加入 `train_cv_group_values_written: false`。
- 问题 3：dev 没有 Stage113 冻结的 pass/fail 阈值。
  - 原因：Stage113 只冻结了 dev report-only、no selection、no retuning。
  - 修正：Stage114 报告中把 `dev_gate_status` 写成 `report_only_no_frozen_pass_threshold`，
    不伪造 dev passed/failed。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_index_redesign_comparison.py scripts\run_primeqa_hybrid_retrieval_index_redesign_comparison.py tests\test_primeqa_hybrid_retrieval_index_redesign_comparison.py
python -m pytest tests\test_primeqa_hybrid_retrieval_index_redesign_comparison.py -q
python scripts\run_primeqa_hybrid_retrieval_index_redesign_comparison.py --user-confirmed-comparison ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 2 passed
Stage114 run: passed
guard checks: 16 / 16 passed
```

真实 Stage114 run 用时：

```text
total: 3199.058 seconds
```

### 我学到了

- 这批 retrieval/index 候选确实能在少量 train-CV 样本上移动召回，但收益很小。
- 只提升 retrieval_context_miss 不够；召回变化会传导到 evidence selection、answerability false answer 和 changed-answer risk。
- `evc_special_token_exact_boost_v1` 是这一批里风险最低的线索，但仍然违反 false-answer 和 evidence-miss guard。
- section-level rollup 在当前实现和当前答案流水线下不是可推进方向，召回和 changed-answer 行为都变差。
- dev 这一步只能作为 report-only 观察，不能反向挑选候选。

### 下一步

Stage115：记录 frozen Stage113 retrieval/index redesign family 的 stop decision。

继续保持：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

## 2026-07-16 - Stage 115 retrieval/index redesign stop decision

### 本阶段目标

把 Stage114 的真实结论正式固化为 stop decision：冻结的 Stage113
retrieval/index redesign family 没有 train-CV-selectable config，不能进入 runtime、
不能打开 final-test gate，也不能用 dev report-only 结果反向挑选候选。

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_retrieval_index_redesign_stop_decision.py
scripts/decide_primeqa_hybrid_retrieval_index_redesign_stop.py
tests/test_primeqa_hybrid_retrieval_index_redesign_stop_decision.py
```

文档：

```text
docs/primeqa_hybrid_retrieval_index_redesign_stop_decision.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115.json
artifacts/primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals/
```

### 真实运行

命令：

```text
python scripts\decide_primeqa_hybrid_retrieval_index_redesign_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage115 retrieval/index redesign stop decision on 2026-07-16 after Stage114 selected 0 of 8 configs; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

结果：

```text
status: primeqa_hybrid_retrieval_index_redesign_family_stopped
stopped_family_id: retrieval_index_redesign_candidate_family
stopped_protocol_id: primeqa_hybrid_retrieval_index_redesign_protocol_v1
stopped_analysis_id: primeqa_hybrid_retrieval_index_redesign_train_cv_dev_validation_v1
guard checks: 25 / 25 passed
recommended_next_direction: user_confirmed_next_research_direction_required
```

继续禁止：

```text
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

### 停止原因

Stage114 的 8 个候选没有任何一个通过 train-CV selectability。

最佳 retrieval 改善只有：

```text
retrieval_context_miss delta: -4
gold doc recall@10 delta: +0.0108
```

但这些候选被 downstream guard 拦住：

```text
evc_special_token_exact_boost_v1:
  changed answer rate: 0.1833
  failed guards:
    train_cv_answerability_false_answer_delta_within_guard
    train_cv_evidence_selection_miss_delta_within_guard

evc_special_token_title_heading_boost_v1:
  changed answer rate: 0.7278
  failed guards:
    train_cv_evidence_selection_miss_delta_within_guard
    train_cv_gold_span_beats_selected_delta_within_guard
    train_cv_changed_answer_rate_within_guard
```

title/heading 加权候选也有小幅 retrieval 改善，但 changed-answer rate 太高：

```text
thw_title2_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: -3
  recall@10 delta: +0.0081
  changed answer rate: 0.7171

thw_title3_heading2_body1_doc_bm25_v1:
  retrieval_context_miss delta: -3
  recall@10 delta: +0.0081
  changed answer rate: 0.8327
```

section-level family 没有形成 retrieval 改善：

```text
best section objective delta: +31.5
best section retrieval_context_miss delta: +9
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_retrieval_context_miss_deltas.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_gold_doc_recall_deltas.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_changed_answer_rates.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_selectability_by_family.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_stop_decision_flags.svg
artifacts\primeqa_hybrid_retrieval_index_redesign_stop_decision_stage115_visuals\stage115_stop_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage115 很容易被误解成可以根据 dev 里 F1 小涨继续挑选候选。
  - 原因：Stage114 中部分 config 在 dev report-only 上有正向 F1。
  - 修正：Stage115 明确 `dev_selection_used: false`、`dev_retuning_used: false`，
    并把下一步设置为 `user_confirmed_next_research_direction_required`。
- 问题 2：Stage115 不应再次运行 retrieval/answer metrics。
  - 原因：stop decision 只负责冻结路线状态，不负责产生新指标。
  - 修正：实现只读取 Stage114 public-safe JSON；guard 中显式记录 split files、
    corpus documents、new metrics、final metrics 均未运行。
- 问题 3：后续路线不能由 stop decision 偷偷决定。
  - 原因：新研究方向会引入新的假设和设计取舍。
  - 修正：decision 只要求用户确认下一研究方向，不自动启动 Stage116 协议。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_retrieval_index_redesign_stop_decision.py scripts\decide_primeqa_hybrid_retrieval_index_redesign_stop.py tests\test_primeqa_hybrid_retrieval_index_redesign_stop_decision.py
python -m pytest tests\test_primeqa_hybrid_retrieval_index_redesign_stop_decision.py -q
python scripts\decide_primeqa_hybrid_retrieval_index_redesign_stop.py --user-confirmed-stop ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage115 run: passed
guard checks: 25 / 25 passed
```

### 我学到了

- stop decision 不能只写“失败了”，还要保存为什么失败、哪个 guard 阻断、dev 为什么不能救。
- retrieval recall 的小幅改善如果带来 evidence selection 和 changed-answer 风险，就不能进入 runtime。
- Stage115 之后已经没有当前 retrieval/index redesign route 可以继续默认推进；下一步必须先确认新研究方向。

### 下一步

Stage116：只有用户确认新研究方向后，才能冻结新的 train/dev-only protocol。

继续保持：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

## 2026-07-16 - Stage 116 high-recall multi-route union candidate pool

### 本阶段目标

按用户确认的新方向，先做第一阶段高召回候选池：用多路检索路线取并集，先把金文档尽量召回到
top100/top200 候选池里；第二阶段再考虑精排或精确查找。

本阶段只做 train/dev 离线 recall-only 实验：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
no source DOC_IDS oracle route
```

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_high_recall_union_comparison.py
scripts/run_primeqa_hybrid_high_recall_union_comparison.py
tests/test_primeqa_hybrid_high_recall_union_comparison.py
```

文档：

```text
docs/primeqa_hybrid_high_recall_union_comparison.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_high_recall_union_stage116.json
artifacts/primeqa_hybrid_high_recall_union_stage116_visuals/
```

### 真实运行

命令：

```text
python scripts\run_primeqa_hybrid_high_recall_union_comparison.py --user-confirmed-direction --confirmation-note "user confirmed Stage116 first-stage high-recall multi-route union candidate-pool experiment on 2026-07-16; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

实际耗时：

```text
total: 835.505 seconds
```

使用路线：

```text
full_document_bm25
section_bm25_max_section_rollup
title_heading_weighted_bm25
title_heading_only_bm25
special_token_boosted_bm25
dense_cache__intfloat_e5_small_v2__512_passage
dense_cache__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix
```

dense 路线只使用已有本地 Stage80 cache：

```text
dense_channel_preflight.status: dense_channels_ready
can_run_without_download: true
no_model_download_attempted: true
```

核心结果：

```text
train answerable: 370
dev answerable: 76

train union hit@100: 332 / 370 = 0.8973
train union hit@200: 345 / 370 = 0.9324
train uncapped union: 358 / 370 = 0.9676

dev union hit@100: 66 / 76 = 0.8684
dev union hit@200: 69 / 76 = 0.9079
dev uncapped union: 72 / 76 = 0.9474
```

相对 full-document BM25：

```text
train hit@100 delta: +20 hits, +0.0541
train hit@200 delta: +14 hits, +0.0378

dev hit@100 delta: +3 hits, +0.0395
dev hit@200 delta: +3 hits, +0.0395
```

train 5-fold 稳定性：

```text
hit@100 average: 0.8965
hit@100 min/max/spread: 0.8592 / 0.9200 / 0.0608

hit@200 average: 0.9326
hit@200 min/max/spread: 0.9067 / 0.9565 / 0.0498
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_channel_hit_at_100.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_union_recall_by_pool_depth.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_union_delta_vs_baseline.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_dev_marginal_hits_by_channel.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_train_fold_union_hit_at_100.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_candidate_pool_size_summary.svg
artifacts\primeqa_hybrid_high_recall_union_stage116_visuals\stage116_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage116 不能再被理解为 top10 替代排序实验。
  - 原因：Stage81、Stage97/98 已经验证过 dense+sparse 作为 top10/runtime 替代没有稳定收益。
  - 修正：本阶段明确改成 first-stage candidate pool，指标看 recall@50/100/200 和 uncapped union，不跑回答生成。
- 问题 2：未截断并集平均候选过大。
  - 真实数据：dev average uncapped union size = 662.2632，p95 = 806。
  - 修正：文档明确 practical boundary 是 RRF 排序后的 top200 pool；下一步不能把未截断并集直接喂给回答链。
- 问题 3：dense cache 不能触发下载，也不能在失败时静默降级。
  - 原因：这会把实验变成不透明的兜底策略。
  - 修正：dense enabled 时必须通过 Stage80 本地 cache preflight；本次通过，未尝试下载。
- 问题 4：小 topK 单测不能假设真实 top100/top200 一定存在。
  - 原因：单测使用小型 fixture。
  - 修正：decision 逻辑改成读取本次配置中可用的 primary/max pool depth。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_high_recall_union_comparison.py scripts\run_primeqa_hybrid_high_recall_union_comparison.py tests\test_primeqa_hybrid_high_recall_union_comparison.py
python -m pytest tests\test_primeqa_hybrid_high_recall_union_comparison.py -q
python scripts\run_primeqa_hybrid_high_recall_union_comparison.py --user-confirmed-direction ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage116 run: passed
guard checks: 11 / 11 passed
public_safe_contract.forbidden_keys_found: []
```

### 我学到了

- 这条路线终于把“先召回、后精排”的两阶段边界拆清楚了：第一阶段只负责让金文档进入候选池，不负责直接回答。
- 多路并集确实能补 BM25 的尾部召回，dev top200 从 66/76 提到 69/76；虽然提升不大，但足够进入第二阶段精排设计。
- dense route 单独替代 top10 不稳定，但作为 union 的尾部补充仍有价值；这和前面 dense+sparse 失败并不矛盾。
- 候选池规模是下一阶段最关键约束：top200 可以作为工程边界，uncapped union 太大。

### 下一步

Stage117：设计 second-stage precision/reranking protocol over fixed Stage116 top200 candidate pool。

继续保持：

```text
test locked
train/dev only
no final metrics
runtime defaults unchanged
no fallback strategies
```

## 2026-07-16 - Stage 117 second-stage reranking protocol freeze

### 本阶段目标

把 Stage116 的第一阶段 top200 高召回候选池固定下来，设计第二阶段 precision/reranking 的
train-CV/dev 验证协议。

本阶段只冻结协议，不构建候选 rows，不跑 reranking 指标，不跑 answer metrics：

```text
test locked
train/dev protocol only
no final metrics
runtime defaults unchanged
no fallback strategies
```

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_second_stage_reranking_protocol.py
scripts/freeze_primeqa_hybrid_second_stage_reranking_protocol.py
tests/test_primeqa_hybrid_second_stage_reranking_protocol.py
```

文档：

```text
docs/primeqa_hybrid_second_stage_reranking_protocol.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_second_stage_reranking_protocol_stage117.json
artifacts/primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals/
```

### 真实运行

命令：

```text
python scripts\freeze_primeqa_hybrid_second_stage_reranking_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage117 second-stage precision/reranking protocol over fixed Stage116 top200 candidate pool on 2026-07-16; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

结果：

```text
status: primeqa_hybrid_second_stage_reranking_protocol_frozen
protocol_id: primeqa_hybrid_second_stage_reranking_protocol_v1
recommended_next_direction: run_second_stage_reranking_train_cv_dev_validation
guard checks: 19 / 19 passed
public_safe_contract.forbidden_keys_found: []
```

继续禁止：

```text
can_open_final_test_gate_now: false
can_run_final_test_metrics_now: false
can_use_test_for_tuning: false
fallback_strategies_enabled: false
default_runtime_policy: unchanged
```

### 冻结的协议

固定候选池：

```text
source_pool_id: stage116_multi_route_union_candidate_pool
candidate_pool_depth: 200
reranker_may_reorder_pool: true
reranker_may_add_documents: false
reranker_may_drop_documents_before_top200_metric: false
uncapped_union_is_not_runtime_input: true
```

候选 family：

```text
channel_rank_feature_reranker_family_v1: 2 configs
lexical_document_feature_reranker_family_v1: 3 configs
supervised_lightweight_reranker_family_v1: 3 configs
```

冻结的 8 个 config：

```text
crf_route_agreement_best_rank_v1
crf_lexical_routes_first_v1
ldf_title_heading_overlap_v1
ldf_title_heading_body_coverage_v1
ldf_special_token_title_heading_v1
slr_logistic_balanced_v1
slr_logistic_hard_negative_v1
slr_ridge_rank_proxy_v1
```

选择规则：

```text
selection_split: train
selection_mode: train_grouped_cross_validation_then_full_train_refit
minimum_train_folds: 5
validation_split: dev
dev_validation_mode: single_pass_report_only_no_retuning
```

主要指标：

```text
mrr_at_20_delta_vs_stage116_order
hit_at_10_delta_vs_stage116_order
hit_at_20_delta_vs_stage116_order
```

关键 guard：

```text
maximum_train_cv_hit_at_200_loss_count: 0
maximum_train_cv_bm25_top10_gold_demotions_to_below_50: 0
maximum_train_cv_hit_at_20_regression_rate: 0.02
maximum_train_cv_top10_regression_count: 3
minimum_train_cv_mrr_at_20_delta: 0.0
```

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_stage116_candidate_pool_recall.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_candidate_family_priorities.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_candidate_config_counts.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_objective_weights.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_guard_thresholds.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_protocol_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_protocol_stage117_visuals\stage117_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：第二阶段很容易被误解成可以直接用 Stage116 的 uncapped union。
  - 原因：uncapped union 的 recall 更高，但平均候选数达到数百篇。
  - 修正：Stage117 固定 top200 pool，并显式 blocked `uncapped_union_as_answer_input_blocked`。
- 问题 2：监督 reranker 需要 gold label，但不能把 gold label 变成 runtime feature。
  - 原因：训练时可用 train label，运行时不能依赖答案文档或测试信息。
  - 修正：feature contract 区分 `training_only_label_sources` 和 `forbidden_runtime_feature_sources`。
- 问题 3：Stage117 不能偷跑候选 rows 或指标。
  - 原因：它只是 protocol freeze，下一阶段才运行 train-CV/dev validation。
  - 修正：guard 记录 `candidate_rows_not_built_in_stage117: true` 和
    `protocol_freeze_only_no_candidate_rows_no_metrics`。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_second_stage_reranking_protocol.py scripts\freeze_primeqa_hybrid_second_stage_reranking_protocol.py tests\test_primeqa_hybrid_second_stage_reranking_protocol.py
python -m pytest tests\test_primeqa_hybrid_second_stage_reranking_protocol.py -q
python scripts\freeze_primeqa_hybrid_second_stage_reranking_protocol.py --user-confirmed-protocol ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage117 run: passed
guard checks: 19 / 19 passed
```

### 我学到了

- 第二阶段应该解决“金文档已经在池里，但排名不够靠前”的问题，而不是再扩大召回池。
- top200 是这条路线目前最合适的工程边界：允许精排有空间，同时避免 uncapped union 变成不可控输入。
- 监督 reranker 可以进入候选，但必须把训练标签和 runtime 特征分清楚，否则会形成泄漏。

### 下一步

Stage118：按 Stage117 冻结协议，运行 second-stage reranking train-CV/dev validation。

继续保持：

```text
test locked
train/dev only
no final metrics
dev report-only, no retuning
runtime defaults unchanged
no fallback strategies
```
## 2026-07-16 - Stage 118 second-stage reranking train-CV/dev validation

### 本阶段目标

按照 Stage117 冻结协议，在固定的 Stage116 top200 candidate pool 上运行第二阶段 reranking train-CV/dev 验证。

本阶段继续保持：

```text
test locked
train/dev only
no final metrics
dev report-only, no retuning
runtime defaults unchanged
no fallback strategies
```

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_second_stage_reranking_validation.py
scripts/run_primeqa_hybrid_second_stage_reranking_validation.py
tests/test_primeqa_hybrid_second_stage_reranking_validation.py
```

文档：

```text
docs/primeqa_hybrid_second_stage_reranking_validation.md
docs/primeqa_hybrid_second_stage_reranking_protocol.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_second_stage_reranking_validation_stage118.json
artifacts/primeqa_hybrid_second_stage_reranking_validation_stage118_visuals/
```

### 真实运行

命令：

```text
python scripts\run_primeqa_hybrid_second_stage_reranking_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage118 train-CV/dev second-stage reranking validation in current turn; test locked; dev report-only; runtime defaults unchanged; no fallback strategies"
```

结果：

```text
status: primeqa_hybrid_second_stage_reranking_completed_no_train_cv_selectable_config
recommended_next_direction: record_second_stage_reranking_stop_decision
selectable_config_count: 0 / 8
guard checks: 16 / 16 passed
public_safe_contract.forbidden_keys_found: []
```

Stage116 top200 pool 复现结果：

```text
train answerable: 370
train top200 gold present: 345 / 370 = 0.9324

dev answerable: 76
dev top200 gold present: 69 / 76 = 0.9079
```

baseline order：

```text
train hit@10: 0.6892
train hit@20: 0.7541
train hit@200: 0.9324
train mrr@20: 0.5062

dev hit@10: 0.7237
dev hit@20: 0.8026
dev hit@200: 0.9079
dev mrr@20: 0.5588
```

### 主要实验结论

8 个冻结 reranking config 都没有通过 train-CV selection guard，因此不能进入 runtime/default，也不能打开 final test gate。

最接近的 config：

```text
crf_lexical_routes_first_v1:
  objective: -0.0229
  train mrr@20 delta: +0.0102
  train hit@10 delta: -0.0162
  train hit@20 delta: -0.0190
  train hit@200 delta: +0.0000
  failed guards:
    hit@20 regression rate: 0.0324 > 0.02
    top10 regression count: 15 > 3
```

另一个看起来有 top20 收益但仍失败的 config：

```text
crf_route_agreement_best_rank_v1:
  objective: -5.9883
  train mrr@20 delta: +0.0018
  train hit@10 delta: +0.0000
  train hit@20 delta: +0.0081
  train hit@200 delta: +0.0000
  failed guards:
    BM25 top10 gold demotions to below 50: 3 > 0
    hit@20 regression rate: 0.0243 > 0.02
    top10 regression count: 21 > 3
```

关键结论：本阶段没有丢 top200 recall，问题是 reranking 把太多原本 top10/top20 的好案例向后移动。

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_objective_scores.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_mrr_at_20_delta.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_hit_at_10_delta.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_train_cv_guard_pass_counts.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_dev_selected_config_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_selection_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_validation_stage118_visuals\stage118_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage117 的一个特征名是 `bm25_top10_non_gold_indicator`，名字容易被误解为使用 gold label。
  - 原因：协议命名不够严谨，但 runtime 不能使用 gold label。
  - 修正：Stage118 显式记录别名：
    `bm25_top10_non_gold_indicator -> bm25_top10_indicator`。
    该特征只表示候选是否出现在 full-document BM25 top10，不读取 answer document label。

- 问题 2：监督模型必须按 config 声明的 feature set 训练，不能把所有 feature 都喂给每个 supervised config。
  - 原因：否则多个 supervised config 会变成同一个模型，协议失真。
  - 修正：新增 feature projection，并保留特征别名映射。

- 问题 3：小样本单测中某些 train-CV fold 可能只有单类 label，logistic 无法训练。
  - 原因：fixture 太小，而真实数据中候选池有大量负例。
  - 修正：配置训练失败时记录为该 config 的 `training_status: failed` 和不可入选，不把它兜底成其他 reranker，也不伪造成功。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_second_stage_reranking_validation.py scripts\run_primeqa_hybrid_second_stage_reranking_validation.py tests\test_primeqa_hybrid_second_stage_reranking_validation.py
pytest -q tests\test_primeqa_hybrid_second_stage_reranking_validation.py
python scripts\run_primeqa_hybrid_second_stage_reranking_validation.py --user-confirmed-validation ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 2 passed
Stage118 real run: passed
guard checks: 16 / 16 passed
```

### 我学到了

- 两阶段设计的第一阶段 top200 recall 已经足够稳定复现，但第二阶段不能只追求平均 MRR，小范围 reranking 很容易牺牲已在 top10/top20 的好案例。
- 当前 reranking 家族的主要失败模式不是召回缺失，而是排序回退风险过大；尤其 top10 regression count 是硬伤。
- 第二阶段 precision/reranking 需要更保守的 rerank window 或只重排低置信候选，不能全量重排 top200。

### 下一步

Stage119：记录 second-stage reranking stop decision。

继续保持：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```


## 2026-07-16 - Stage 119 second-stage reranking stop decision

### 本阶段目标

把 Stage118 的 `0 / 8` 可选配置结果正式记录为 second-stage reranking stop decision。

本阶段只读 Stage118 public-safe report，不重新加载 split 文件、不加载 corpus、不重建 candidate rows、不运行 retrieval/reranking/answer/final metrics。

继续保持：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

### 新增内容

代码：

```text
src/ts_rag_agent/application/primeqa_hybrid_second_stage_reranking_stop_decision.py
scripts/decide_primeqa_hybrid_second_stage_reranking_stop.py
tests/test_primeqa_hybrid_second_stage_reranking_stop_decision.py
```

文档：

```text
docs/primeqa_hybrid_second_stage_reranking_stop_decision.md
docs/primeqa_hybrid_second_stage_reranking_validation.md
docs/evaluation_strategy.md
docs/data_strategy.md
```

产物：

```text
artifacts/primeqa_hybrid_second_stage_reranking_stop_decision_stage119.json
artifacts/primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals/
```

### 真实运行

命令：

```text
python scripts\decide_primeqa_hybrid_second_stage_reranking_stop.py --user-confirmed-stop --confirmation-note "user confirmed Stage119 second-stage reranking stop decision in current turn after Stage118 selected 0 of 8 configs; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

结果：

```text
status: primeqa_hybrid_second_stage_reranking_family_stopped
stopped_family_id: second_stage_reranking_candidate_family
recommended_next_direction: user_confirmed_next_research_direction_required
guard checks: 27 / 27 passed
public_safe_contract.forbidden_keys_found: []
```

停止依据：

```text
Stage118 selected configs: 0 / 8
train top200 gold present: 0.9324
dev top200 gold present: 0.9079
hit@200 loss across configs: 0
```

这说明问题不是 top200 recall 丢失，而是 reranking 造成 top10/top20 排名回退。

### 主要停止证据

两个有正向信号但被 guard 阻止的 config：

```text
crf_lexical_routes_first_v1:
  train mrr@20 delta: +0.0102
  train hit@10 delta: -0.0162
  train hit@20 delta: -0.0190
  train hit@200 delta: +0.0000
  failed guards:
    train_cv_hit_at_20_regression_rate_within_guard
    train_cv_top10_regression_count_within_guard

crf_route_agreement_best_rank_v1:
  train mrr@20 delta: +0.0018
  train hit@10 delta: +0.0000
  train hit@20 delta: +0.0081
  train hit@200 delta: +0.0000
  failed guards:
    train_cv_bm25_top10_gold_demotions_to_below_50_within_guard
    train_cv_hit_at_20_regression_rate_within_guard
    train_cv_top10_regression_count_within_guard
```

family 结果：

```text
channel_rank_feature_reranker_family_v1: 0 / 2 selectable
lexical_document_feature_reranker_family_v1: 0 / 3 selectable
supervised_lightweight_reranker_family_v1: 0 / 3 selectable
```

dev 观察：

```text
dev_used_for_selection: false
dev_used_for_retuning: false
dev_observations_are_non_adoptable: true
best dev mrr@20 delta: +0.0085 from crf_lexical_routes_first_v1
```

dev 观察不能推翻 train-CV stop decision。

### 可视化结果

已生成：

```text
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_objective_scores.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_mrr_at_20_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_hit_at_10_deltas.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_train_cv_guard_failure_reasons.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_selectability_by_family.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_stop_decision_flags.svg
artifacts\primeqa_hybrid_second_stage_reranking_stop_decision_stage119_visuals\stage119_stop_guard_check_status.svg
```

### 问题、原因与修正

- 问题 1：Stage118 在 no-selectable 情况下没有在 `train_cv_selection` 内重复写 `selection_mode`。
  - 原因：Stage118 的 selection mode 已在 split contract 中，no-selectable branch 没有重复保存。
  - 修正：Stage119 summary 从 split contract 补全 `selection_mode`，报告中显示为 `train_grouped_cross_validation_then_full_train_refit`。

- 问题 2：dev 中仍有 `crf_lexical_routes_first_v1` 的小幅 `mrr@20 +0.0085`。
  - 原因：Stage118 对所有 config 都算了 dev report-only 指标。
  - 修正：Stage119 明确记录 `dev_observations_are_non_adoptable: true`，不允许用 dev 反向选择或救回 train-CV 失败配置。

### 验证

目标验证：

```text
python -m ruff check src\ts_rag_agent\application\primeqa_hybrid_second_stage_reranking_stop_decision.py scripts\decide_primeqa_hybrid_second_stage_reranking_stop.py tests\test_primeqa_hybrid_second_stage_reranking_stop_decision.py
pytest -q tests\test_primeqa_hybrid_second_stage_reranking_stop_decision.py
python scripts\decide_primeqa_hybrid_second_stage_reranking_stop.py --user-confirmed-stop ...
```

结果：

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage119 real run: passed
guard checks: 27 / 27 passed
```

### 我学到了

- second-stage reranking 这条路线现在应该停止：它没有解决召回缺失，也无法安全改善排序；主要问题是把已有 top10/top20 正例往后推。
- 如果后续继续做 precision，只能重新开一个更保守的新研究方向，例如限定 rerank window 或只处理低置信候选；但这需要新的用户确认，不能从当前 stop decision 自动跳过去。
- dev 中的小幅正向信号只能作为观察，不能作为选择依据。

### 下一步

Stage120：需要用户确认后，选择新的 train/dev-only 研究方向。

继续保持：

```text
test locked
no final metrics
runtime defaults unchanged
no fallback strategies
```

## 2026-07-16 - Stage 120 fast-filter screening protocol freeze

Goal: replace the stopped full-top200 second-stage reranking direction with a
train/dev-only protocol that first performs a cheap fast filter, then applies a
more constrained alternate screening selector.

What changed:

- Added `src/ts_rag_agent/application/primeqa_hybrid_fast_filter_screening_protocol.py`.
- Added `scripts/freeze_primeqa_hybrid_fast_filter_screening_protocol.py`.
- Added `tests/test_primeqa_hybrid_fast_filter_screening_protocol.py`.
- Added `docs/primeqa_hybrid_fast_filter_screening_protocol.md`.
- Updated evaluation/data strategy docs and linked Stage119 to the Stage120
  follow-up.

Real run:

```text
python scripts\freeze_primeqa_hybrid_fast_filter_screening_protocol.py --user-confirmed-protocol --confirmation-note "user confirmed Stage120 fast-filter plus alternate screening protocol after Stage119 stopped full top200 reranking; train/dev only; test locked; no final metrics; runtime defaults unchanged; no fallback strategies"
```

Result:

```text
status: primeqa_hybrid_fast_filter_screening_protocol_frozen
recommended_next_direction: run_fast_filter_screening_train_cv_dev_validation
guard checks: 16 / 16 passed
candidate families: 3
candidate configs: 6
public_safe_contract.forbidden_keys_found: []
```

Frozen design:

```text
source pool: Stage116 top200
full_top200_rerank_allowed: false
screening_may_add_documents: false
screening_may_use_uncapped_union: false
minimum protected_prefix_depth: 5
maximum promotion_budget_top10: 1
```

Important boundaries:

```text
train grouped-CV selection only
dev report-only
test locked
no final metrics
runtime defaults unchanged
fallback_strategies_enabled: false
```

Visualizations generated:

```text
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_stage119_stop_summary.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_candidate_family_counts.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_fast_filter_window_sizes.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_promotion_budgets.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_guard_thresholds.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_protocol_decision_flags.svg
artifacts\primeqa_hybrid_fast_filter_screening_protocol_stage120_visuals\stage120_guard_check_status.svg
```

Verification:

```text
targeted ruff: passed
targeted pytest: 3 passed
Stage120 real run: passed
full ruff: passed
full pytest: 328 passed
```

What I learned:

- The next reasonable route is not to rerank the whole top200. Stage119 showed
  that this preserves hit@200 but damages top10/top20 ordering.
- The new protocol must protect the already-good prefix and only allow a small
  number of high-confidence tail candidates to compete for top positions.
- Stage120 is only a protocol freeze. It does not prove better recall or
  ranking quality yet.

Next step: Stage121 should run the frozen fast-filter plus alternate-screening
protocol on train grouped cross-validation and dev report-only validation.

## 2026-07-16 - Stage 121 fast-filter screening train-CV/dev validation

Goal: run the Stage120 fast-filter plus alternate-screening protocol on train
grouped cross-validation and dev report-only validation, then compare it with
public-source high-recall RAG/agent retrieval patterns.

What changed:

- Added `src/ts_rag_agent/application/primeqa_hybrid_fast_filter_screening_validation.py`.
- Added `scripts/run_primeqa_hybrid_fast_filter_screening_validation.py`.
- Added `tests/test_primeqa_hybrid_fast_filter_screening_validation.py`.
- Added `docs/primeqa_hybrid_fast_filter_screening_validation.md`.
- Added `docs/rag_high_recall_agent_research_notes.md`.
- Updated evaluation/data strategy docs and Stage120 protocol follow-up notes.

Final real run:

```text
python scripts\run_primeqa_hybrid_fast_filter_screening_validation.py --user-confirmed-validation --confirmation-note "user confirmed Stage121 train-CV/dev fast-filter plus alternate-screening validation after Stage120 protocol freeze; rerun after protected-prefix duplicate segment fix; test locked; dev report-only; no final metrics; runtime defaults unchanged; no fallback strategies"
```

Baseline reproduced:

```text
train hit@10: 0.6892
train hit@20: 0.7541
train hit@200: 0.9324
train mrr@20: 0.5062

dev hit@10: 0.7237
dev hit@20: 0.8026
dev hit@200: 0.9079
dev mrr@20: 0.5588
```

Result:

```text
status: primeqa_hybrid_fast_filter_screening_completed_train_cv_selected_dev_reported
selected_config_id: special_token_exact_window40_rule_selector_v1
selected_family_id: evidence_density_fast_filter_family_v1
guard checks: 14 / 14 passed
guard-passed configs: 2 / 6
selectable positive configs: 1 / 6
```

Selected config train-CV result:

```text
hit@10 delta: +0.0000
hit@20 delta: +0.0027
hit@20 count delta: +1
hit@200 delta: +0.0000
mrr@20 delta: +0.0000
average promoted tail docs into top10: 0.0
```

Selected config dev report-only result:

```text
hit@10 delta: +0.0000
hit@20 delta: +0.0000
hit@200 delta: +0.0000
mrr@20 delta: -0.0008
dev_used_for_selection: false
dev_used_for_retuning: false
```

Important failed-but-interesting signal:

```text
top10_locked_route_vote_window50_pairwise_logistic_v1:
  train hit@20 delta: +0.0108
  dev hit@20 delta: +0.0132
  blocked by train_cv_hit_at_20_regression_rate: 0.0189 > 0.01
```

Bug found and fixed before final run:

```text
when protected_prefix_depth > 10, ranks 11-20 were appended again
```

This duplicated already-protected records and created an invalid hit@200 loss
for one config. I fixed the segment assembly and reran the real Stage121
validation. The final artifact and docs use the rerun result.

Public high-recall RAG/agent research notes:

- LlamaIndex and Elastic document RRF/fusion patterns over multiple result
  lists.
- Haystack documents hybrid BM25 plus embedding retrieval, followed by ranking.
- ColBERT, SPLADE, HyDE, RAPTOR, and Self-RAG show broader first-stage,
  learned sparse/late-interaction, query expansion, hierarchical retrieval, and
  adaptive retrieval directions.

What I learned:

- Stage121 is safer than full top200 reranking, but the selected improvement is
  weak: train-CV gains one hit@20 case and dev hit@10/hit@20/hit@200 are flat.
- The fixed Stage116 top200 pool still caps recall at 0.9324 train and 0.9079
  dev; second-stage screening cannot recover missing gold documents outside
  top200.
- The strongest public pattern for genuinely higher recall is to improve the
  first-stage candidate generator through multi-query/hybrid/learned-sparse or
  late-interaction retrieval, then keep a conservative second-stage screener.

Verification:

```text
targeted ruff: passed
targeted pytest: 2 passed
Stage121 final real rerun: passed
full ruff: passed
full pytest: 330 passed
```

Next step: Stage122 should review changed cases for the selected safe config
and the stronger-but-blocked logistic config before designing any new
first-stage recall expansion protocol.
