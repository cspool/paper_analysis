## Hyper-Parallel Scaling (超并行扩展)

术语解释
一种新的推理时扩展范式，通过在每 token 层面增加模型内部计算量和计算路径多样性，直接提升模型的内在 next-token 预测质量，与传统的 sequential scaling（如 CoT，生成更长的推理步骤）和 parallel scaling（如 Self-Consistency，生成多条完整序列后投票）正交。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hyper-Parallel Scaling 是论文 Introducing 的第三类推理时扩展范式，其核心定义与其他两类范式的区别如下：

| 范式 | 操作对象 | 聚合粒度 | 代表性方法 |
|------|---------|---------|-----------|
| Sequential Scaling | 输出生成过程（生成更长步骤） | 序列级（更长的输出链） | Chain-of-Thought, Tree-of-Thoughts |
| Parallel Scaling | 输出生成过程（多次采样） | 序列级（多条完整序列投票） | Self-Consistency, Beam Search |
| **Hyper-Parallel Scaling** | **模型内部计算路径** | **Token 级（每次 next-token 预测内）** | **RoE (本文)** |

关键区别：
- Sequential/Parallel Scaling 把模型当作黑盒，在外层操作输出序列
- Hyper-Parallel Scaling 进入模型内部，在每 token 预测层面多样化计算路径并聚合
- Hyper-Parallel 是 Sequential/Parallel 的**正交补充**，可与两者同时使用

在 MoE 模型中，Hyper-Parallel Scaling 利用"每 token 仅激活 k 个 expert"的稀疏性——E−k 个 inactive expert 闲置——通过随机激活不同的 expert 子集来释放模型的全部潜力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hyper-Parallel Scaling 的执行流程（以 RoE 实现为例）：

```
# 单 token 生成中的 Hyper-Parallel Scaling
输入: prefix tokens, MoE model M
输出: 下一个预测 token

# 传统 greedy decoding（单路径）
h = M.forward(prefix)           # 单次确定性 forward
logits = M.lm_head(h)           # 单组 logits
next_token = argmax(logits)     # 确定性预测

# Hyper-Parallel Scaling with RoE（多路径）
candidate_logits = []
for i in range(n):              # n = sample count (e.g., 32)
    h_i = M.forward(prefix, routing_mode="gumbel_top_k", tau=τ)
    logits_i = M.lm_head(h_i)
    candidate_logits.append(logits_i)

# Token 级聚合（非序列级！）
final_probs = mean(softmax(candidate_logits), dim=0)  # probability averaging
next_token = argmax(final_probs)
```

关键点：聚合发生在 **logits/probability 层面**，而非序列层面。这使 Hyper-Parallel Scaling 适用于开放生成任务（如代码生成），而 Self-Consistency 的 majority voting 只在可验证答案的任务上有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 Hyper-Parallel Scaling 需要三个组件：
1. **内部路径多样化机制**：论文在 MoE 中使用 Gumbel-Top-K routing；对于 dense 模型，论文提出可用 dropout-based variation（Shelmanov et al., 2021）或 recurrent re-computation（Lin et al., 2022）。
2. **高效执行**：通过 batched inference 将多次 forward 合并，利用 GPU sub-linear batch scaling。
3. **缓存优化**：Clean Cache 策略避免维护 N 份 KV-cache。

论文指出 Hyper-Parallel Scaling 是 domain-agnostic 的，可扩展到 vision、audio、video 等模态。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
