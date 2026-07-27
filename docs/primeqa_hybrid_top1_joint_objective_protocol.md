# Stage 202：分组 Top-1 联合约束目标协议

## 1. 阶段目标

Stage 202 根据 Stage 201 的真实失败归因，冻结 Stage 203 的 train-only 嵌套交叉验证协议。它不是效果实验：本阶段只读取 SHA-256 精确匹配的 Stage 199 和 Stage 201 公开聚合报告，模型拟合、逐题预测和策略评估均为 0，development/test 继续关闭。

用户确认路线 A：不再继续调普通 pairwise、ListNet、LambdaRank label gain 或人工 winner rule，而是训练一个直接面向最终 Top-1 winner 的 question-grouped 联合目标。LightGBM 4.7 的 `LGBMRanker` 自定义 objective 可以接收 `group`，因此能够在现有依赖内实现逐问题损失；这是技术可行性依据，不代表效果已经得到证明。[LightGBM LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)

## 2. 为什么研究目标函数

Stage 201 的 140 个 policy cell 全部不合格。主失败项为：

| 约束 | 失败 cell |
|---|---:|
| conditional strict capture | 135 / 140 |
| strict-success precision | 125 / 140 |
| unsafe selection rate | 116 / 140 |
| minimum capture fold count | 89 / 140 |
| minimum unsafe fold count | 51 / 140 |

第一阶段池 recall 与 citation/F1 aggregate delta 均为 `0/140` 失败。41,440 个 question-cell 中，`winner_selection_miss` 为 15,294，明显高于 safety-pool exclusion 476 与 risk-frontier exclusion 2,954。冻结诊断得分为 objective research 292、model research 224、representation research 13。因此 Stage 202 针对 winner objective，而不重开候选池或特征研究。

## 3. 固定候选路径

每个 outer context 沿用 Stage 199 保存的 Stage 196 source spec：

- citation-loss 与 F1-loss safety head、pool representation 和 estimator 固定；
- pool cap 固定为 16，并在 cap 后并入唯一 baseline；
- custom objective 使用 source gain representation 与 source gain tree profile；
- 旧 risk signal 与 winner-rule 因子网格不重开；
- custom objective 对完整固定 pool 打分，不再施加旧 risk frontier；
- gold label 不进入 runtime，最终只按模型标量分数和 canonical action order 破同分。

不施加旧 frontier 是路线 A 的候选域定义，不是失败后的替代路径或 fallback。第一阶段 pool 本身保持冻结，Stage 203 也不得在运行中扩大 pool。

## 4. 目标编码

训练 partition 内每个 action 归入一个互斥类别：

```text
unsafe        = 0  citation_delta < 0 或 F1 delta < -1e-12
safe_zero     = 1  citation_delta == 0 且 abs(F1 delta) <= 1e-12
baseline      = 2  action family == baseline
strict_success= 3  非 baseline 且 citation/F1 均不回归
```

对问题组内的 action 构造三个固定概率分布：

1. `q_capture`：存在 strict action 时，在全部 strict action 上均匀分布；不存在时对 baseline one-hot。
2. `q_safety`：在全部 non-unsafe action 上均匀分布。
3. `q_precision`：在 strict action 与 baseline 的并集上均匀分布，排除会降低 changed-answer precision 的 safe-zero 和 unsafe action。

给定 safety 权重 `lambda_s` 与 precision 权重 `lambda_p`：

```text
q = (q_capture + lambda_s * q_safety + lambda_p * q_precision)
    / (1 + lambda_s + lambda_p)
```

模型对每个问题组的标量 score 做数值稳定 softmax 得到 `p`，优化交叉熵 `CE(q, p)`：

```text
gradient_i = p_i - q_i
hessian_i  = max(p_i * (1 - p_i), 1e-6)
```

完整 softmax Hessian 使用正对角近似，确保 LightGBM 的二阶接口得到有限且严格为正的 Hessian。Stage 203 必须用单元测试验证每组 target 和为 1、group size 与 row 数一致、每题恰有一个 baseline、gradient/Hessian 有限、Hessian 为正，并做确定性重复检查。

## 5. 冻结消融

```text
lambda_s ∈ {0.0, 0.5, 1.0, 2.0}
lambda_p ∈ {0.0, 0.5, 1.0, 2.0}
```

完整 4×4 网格产生 16 个 custom objective：1 个 strict-only、3 个 safety-only、3 个 precision-only、9 个 full-joint。另保留 1 个 Stage 196 精确控制，共 17 个 candidate config。必须同时报告相对 exact control 和 strict-only 的 paired delta，并报告随权重变化的方向性响应；方向性响应是诊断，不替代冻结 eligibility gate。

## 6. 嵌套交叉验证

- 5 个 outer fold、每个 outer context 4 个 inner fold；同一问题的全部 action 不拆分。
- 20 个 inner partition 每个拟合 4 个 source model 与 16 个 custom objective，共 20 fits。
- 只有通过原 13 项 inner eligibility constraint 的 config 才能进入 outer refit。
- outer refit 只拟合已选 config；若没有 eligible config，记录失败并停止该 outer context，不用较弱候选替代。
- 模型拟合上限 425；LightGBM tree 上限 112,500。这些是 Stage 203 预算，不是 Stage 202 已发生消耗。
- 原 13 项 inner constraint 和 17 项 advancement gate 数值全部保持不变。

## 7. 资源与执行

Stage 203 使用 CPU、8 个物理线程、固定 300 trees、无 early stopping，并逐个释放 custom model。系统可用内存预检保持 4 GiB。正式进程必须由同一条 PowerShell 命令只调用一次 `Wait-Process`，挂起等待同一 PID 自然结束；不轮询、不设置实验 timeout、不 retry、不缩减网格、不启用 fallback。

## 8. 正式冻结结果

Stage 202 正式报告：

```text
status:        stage202_top1_joint_objective_protocol_frozen
guards:        81 / 81 passed
model fits:    0
private rows:  0
dev/test:      closed / closed
visuals:       10 SVG + 10 PNG
report hash:   0818f0ae7186eea19d4137023b87963010e76e9cbf20e8f1ef61576a60569bdb
manifest hash: 812187c175f474ee562f86d8cef7df6762710b88387692069f621b469353ece2
```

过程更新曾在读取最终 guard 数组前口头写成 82/82；重新读取落盘 JSON 后立即更正为真实的 81/81。首次冻结后，最终设计复核还发现 outer refit 字段把包含 exact control 的候选统称为 `objective`，随后更正为 `config` 并重新生成报告；首次报告哈希 `907a8b812afcdf4dd71e28e7f3a845590c19c1591810469006a8b1b0640a2f54` 已被上方正式哈希取代，预算和行为没有改变。

Stage 202 只授权 Stage 203 train-only experiment。full-train selection、replacement、runtime E2E、development/test、Stage 178B 和默认启用均未授权。
