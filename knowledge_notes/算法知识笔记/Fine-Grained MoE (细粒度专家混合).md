## Fine-Grained MoE (细粒度专家混合)

术语解释
Fine-Grained MoE 是 DeepSeekMoE (Dai et al. 2024) 提出的 MoE 架构变体：在总参数量不变的前提下，将传统 MoE 的少量大 expert 拆分为大量小 expert（更多 expert 数量、更小的每个 expert 参数），并增大 top-k 值以激活更多 expert。核心动机是提升 expert specialization——更细粒度的 expert 可以更精准地学习不同知识子域。

术语是什么？
传统 MoE：每层 8 个 expert，每个 expert FFN hidden dim ~4h，top-2 激活 → 粗粒度，expert 难以充分 specialization。
Fine-Grained MoE：每层 64 个 expert，每个 expert FFN hidden dim ~h/4，top-8 激活 → 细粒度，每个 expert 专注更窄的知识域。

关键数量关系：总参数量 ≈ N_experts × d_expert_size。Fine-Grained 通过增大 N、减小 d 保持总参数量不变，但 expert 的专业化能力因更细的分工而增强。已被 DeepSeek-V2、Qwen2-57B-A14B 等模型采用。

然而，Fine-Grained MoE 面临严重的 All-to-All 通信瓶颈——需要激活更多 expert（top_k 更大），导致 All-to-All 通信量与 top_k 线性增长。BigMac 论文 Table 1 显示：top_k=8 时 All-to-All 占训练时间 91.8%、推理时间 90.6%。

从算法pipeline角度拆解术语：
给定同一个 MoE 层，fine-grained vs vanilla 的对比如下：

```
# Vanilla MoE (粗粒度，如 Mixtral)
E = 8           # 8 个 expert
top_k = 2       # 每 token 激活 2 个
d_ff = 5632     # 每个 expert 的 FFN intermediate dim

# Fine-Grained MoE (细粒度，如 DeepSeekMoE/BigMac)
E = 64          # 64 个 expert
top_k = 8       # 每 token 激活 8 个
d_ff = 704      # 每个 expert 的 FFN intermediate dim (约 1/8)

# Fine-Grained MoE forward:
x = input_token               # [h]
logits = x @ W_gate           # [64]
probs = TopK(SoftMax(logits), k=8)
output = sum(probs[i] * Expert_i(x) for i in selected_experts)
# 8 个 expert 各执行 FFN(x) = W_down · σ(W_gate · x) ⊙ (W_up · x)
```

术语一般如何实现？如何使用？
- DeepSeekMoE: 2 shared experts（always active）+ 64 routed experts（fine-grained），top-6 routed
- DeepSeek-V2: fine-grained 扩展到 160 experts，routed top-6 + shared top-2
- Qwen2-57B-A14B: 继承 fine-grained MoE 设计
- BigMac: 在 fine-grained MoE 基础上叠加 DCCA 通信优化

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs（Ling-Lite 16.8B/2.75B active 和 Ling-Plus 290B/28.8B active 均采用 Fine-Grained Experts 策略：扩展 expert 数量同时等比缩小每个 expert 的 intermediate size，搭配 Shared Expert 解决单个 expert 在有限容量下难以同时发展通用能力和专业能力的问题。Ling 引入 Stochastic Routing Warmup 防止 fine-grained 下的训练初期 router 崩溃）
- Continual Pre-training of MoEs How robust is your router（Granular MoE CPT 研究：E=31 routed + 1 shared, K=3 active per token, FFN intermediate=704（dense 的 1/4）。Granular MoE 在 CPT 中显著优于 Switch MoE (E=8, K=1)：更低 validation loss、更好 benchmark、更低的 MRI。早期层（0-6）MRI 在 Granular MoE 中远低于 Switch MoE，表明细粒度架构有利于 CPT 场景下的负载均衡稳定性）

---
