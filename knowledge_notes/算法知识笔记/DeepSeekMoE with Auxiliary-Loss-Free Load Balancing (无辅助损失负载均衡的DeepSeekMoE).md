## DeepSeekMoE with Auxiliary-Loss-Free Load Balancing (无辅助损失负载均衡的DeepSeekMoE)

术语解释
DeepSeekMoE 是 DeepSeek 系列模型采用的 Mixture-of-Experts 架构（Dai et al. 2024），使用细粒度专家分割（fine-grained expert segmentation）和共享专家隔离（shared expert isolation）。DeepSeek-V3 在此基础上引入 Auxiliary-Loss-Free Load Balancing（Wang et al. 2024a），通过每个专家的可学习 bias 项 b_i 动态调整路由决策，替代传统 auxiliary loss，在保证负载均衡的同时消除 auxiliary loss 对模型性能的负面影响。

术语是什么？
DeepSeekMoE 的核心设计：(1) **细粒度专家分割**：使用大量小规模 routed experts（256 个，每个 intermediate dim=2048）而非粗粒度大专家，提升专家特化程度；(2) **共享专家隔离**：设置 1 个 shared expert 处理通用知识，routed experts 专注特定领域，减少知识冗余；(3) **Auxiliary-Loss-Free 路由**：为每个 expert 引入 bias b_i，仅在 Top-K 路由选择时加到 affinity score s_{i,t} 上（s_{i,t}+b_i 决定 Top-K），但 gating value 仍使用原始 s_{i,t}（Sigmoid 归一化）。训练每步结束时动态调整 b_i：过载 expert 的 b_i -= γ(0.001)，欠载 expert 的 b_i += γ(0.001)。配合极小的 complementary sequence-wise balance loss（α=0.0001）防止单序列极端不均衡。(4) **Node-Limited Routing**：每 token 最多路由到 M=4 个节点的 expert，平均每节点 3.2 个 expert，实际选择 K_r=8。

从算法pipeline角度拆解术语：
```
=== DeepSeekMoE Forward Pass (per token, per MoE layer) ===

Input: u_t ∈ R^d  (FFN input after attention, d=7168)

// 1. Gate Computation
for i in 1..256:
    s_{i,t} = Sigmoid(u_t^T · e_i)       // token-to-expert affinity

// 2. Aux-Loss-Free Routing (bias adjustment at step boundary)
selected = TopK({s_{j,t} + b_j | j=1..256}, K_r=8)  // bias only for routing!
g_{i,t}' = s_{i,t} if i in selected else 0          // gating from raw affinity
g_{i,t} = g_{i,t}' / sum_j(g_{j,t}')                // normalize

// 3. Node-Limited Constraint
// Ensure selected experts are on at most M=4 nodes

// 4. Expert Computation
h_t' = u_t
     + sum_{i=1}^{1} FFN_i^{(s)}(u_t)               // shared expert (always active)
     + sum_{i in selected} g_{i,t} · FFN_i^{(r)}(u_t) // 8 routed experts

// 5. Post-Step Bias Update (at end of each training step)
for i in 1..256:
    load_i = tokens_routed_to_expert_i / expected_tokens
    b_i += γ * (1 - load_i)  // γ=0.001, drives toward balanced load
```

Auxiliary-loss-free vs batch-wise auxiliary loss 对比：实验表明 batch-wise balancing 比 sequence-wise balancing 更灵活，允许专家在不同 domain 上特化。1B MoE 模型验证：sequence-wise loss=2.258, aux-loss-free=2.253, batch-wise loss=2.253。Pile-test 上 aux-loss-free 模型展现更强的 domain-specific expert specialization patterns。

术语一般如何实现？如何使用？
DeepSeek-V3 使用 61 层 Transformer，前 3 层为 dense FFN，后 58 层为 MoE。每个 MoE 层：1 shared expert + 256 routed experts（intermediate dim=2048）。K_r=8，M=4（node-limited）。Sigmoid gating with top-K affinity normalization。训练时：γ=0.001（first 14.3T tokens）→ 0.0（last 500B tokens），α=0.0001。推理时：bias 固定（不再更新），shared expert 在 decoding 阶段视为 always-selected routed expert，实际激活 9 experts/token。无 token dropping（训练和推理均不丢 token）。

涉及论文标题：
- DeepSeek-V3 Technical Report
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
