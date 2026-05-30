## Speculative Decoding (SD / 投机解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Speculative Decoding（投机解码）是一种无损的 LLM 推理加速技术，最初由 Leviathan et al. (ICML 2023) 和 Chen et al. (2023) 独立提出。核心思想：用一个小型 draft model（通常比 target model 小 10-100×）快速自回归生成 γ 个候选 tokens，然后用大 target model 并行验证这些候选 tokens 的正确性（一次 forward pass 处理所有 γ 个 tokens），通过 rejection sampling 丢弃 draft model 预测错误的 tokens。加速原理：target model 验证 γ 个 tokens 的计算时间 ≈ 单 token 解码时间（memory-bound 时），但可接受 σ×(γ+1) 个 tokens，因此 speedup ≈ σ×(γ+1)（理想情况）。MoESD 将此分析扩展到 MoE：指出验证时间 T_T(B,γ) 的额外开销来自 (1) compute-bound 导致的逐 token 计算增加和 (2) 验证多 token 时额外激活 expert 导致的参数加载增加。

从算法pipeline角度拆解术语：
```
# SD 一轮（per decoding round）
# 输入: prefix tokens P, target model M_T, draft model M_D, draft length γ

# Step 1: Draft（自回归）
draft_tokens = []
for i in 1..γ:
    logits_D = M_D.forward(prefix + draft_tokens)     # T_D(B, 1)
    next_token = sample(logits_D[-1])
    draft_tokens.append(next_token)

# Step 2: Verify（并行）
logits_T = M_T.forward(prefix + draft_tokens)          # T_T(B, γ)
# 一次 forward 同时处理所有 draft tokens

# Step 3: Rejection Sampling
accepted = []
for i in 1..γ:
    p_D = softmax(logits_D[i])
    p_T = softmax(logits_T[i])
    if random() < min(1, p_T[draft_tokens[i]] / p_D[draft_tokens[i]]):
        accepted.append(draft_tokens[i])
    else:
        accepted.append(sample_from_residual(p_T - p_D))
        break  # 后续 tokens 全部丢弃

# 本轮产出: len(accepted) 个新 tokens
# Speedup = (total_accepted_tokens) / (R × (γ×T_D + T_T(B,γ) + T_reject))
```

变体：(a) **Eagle**：用集成在 target model 内的 trained speculation head 替代独立 draft model，利用 feature-level uncertainty 提升 acceptance rate；(b) **Tree-structured SD**（SpecInfer, Medusa, Eagle-2/3）：一次生成多分支 draft token tree 而非单链，扩大候选空间；(c) **Self-speculative SD**：target model 自身早期层作为 draft（无需额外模型）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 框架支持：vLLM（batched SD + cudagraph）、SGLang、TensorRT-LLM（Medusa）、HuggingFace Transformers（assisted generation API）。
- Acceptance rate α 是核心算法指标——α 越高，每轮产出 tokens 越多。MoESD 补充指出 α 无法解释系统瓶颈（如 MoE 的 expert 激活开销或 batch size 效应），需结合 Target Efficiency。
- MoESD 的关键新发现：传统观点认为 SD 对大 batch 和 MoE 无效（T_T(B,γ) 显著增长），但中等 batch size 下 MoE 所有 expert 已激活 → 验证不增加参数加载 → SD 反而对 MoE 加速效果优于 dense 模型（尤其在稀疏度高的 MoE 上）。最长 speedup 2.29×（Qwen2-57B-A14B, γ=4, humaneval, temperature=0, 2xGPU-B）。

涉及论文标题：
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE
