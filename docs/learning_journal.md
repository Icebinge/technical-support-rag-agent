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
