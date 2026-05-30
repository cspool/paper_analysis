## Clean Cache (清洁缓存 / 共享 KV-Cache for Batched Inference)

术语解释
RoE 提出的一种 KV-cache 优化策略，在 batched multi-sample forward 中，仅第一个样本使用确定性路由（τ=0）产生"清洁"KV-cache，其余所有样本共享该缓存，从而将多路径推理的 KV-cache 内存开销压缩到与单样本相同。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clean Cache 的核心洞察：在 RoE 的 batched inference 中，n 个样本（每个样本有独立的路由随机性）的 attention hidden state 虽然因 MoE 层的不同 expert 选择而发散，但**仅当前 token 的 expert 选择需要随机性即可产生输出多样性**。所有样本可以共享同一份"清洁"的 KV-cache 历史（由确定性路由产生），从而避免维护 n 份 KV-cache。

KV-cache 在 autoregressive decoding 中的标准内存公式为：
$$M_{\text{kv}} = 2 \times L \times H \times S \times \text{precision}$$

其中 L=层数, H=hidden_size, S=序列长度。在 naive RoE 实现中需乘 n（n 个样本各维护一份），Clean Cache 将此因子消除为 1。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Clean Cache 在 RoE 推理系统架构中的运转流程：

```
Token Step t:
┌─────────────────────────────────────────────────────┐
│  Batch Input: [h₀, h₁, h₂, ..., h_{n-1}]  (n×d)   │
└────────────────────┬────────────────────────────────┘
                     │
  ┌──────────────────▼────────────────────────────────────┐
  │  Layer l: Attention                                   │
  │                                                       │
  │  Sample 0 (Clean Path):                               │
  │    Q₀, K₀, V₀ = proj(h₀)                             │
  │    attn₀ = Softmax(Q₀·K_past^T/√d)·V_past  ← 共享    │
  │    K_past ← concat(K_past, K₀)  ← 仅更新一次!         │
  │    V_past ← concat(V_past, V₀)                        │
  │                                                       │
  │  Samples 1..n-1 (Diverse Paths):                      │
  │    Q_i, K_i, V_i = proj(h_i)                          │
  │    attn_i = Softmax(Q_i·K_past^T/√d)·V_past  ← 复用! │
  │    # K_i, V_i 不追加到 KV-cache (减少内存)             │
  └──────────────────┬────────────────────────────────────┘
                     │
  ┌──────────────────▼────────────────────────────────────┐
  │  Layer l: MoE FFN                                    │
  │                                                       │
  │  Sample 0: τ=0 (deterministic Top-K)                  │
  │  Samples 1..n-1: τ=τ_l (Gumbel-Top-K stochastic)      │
  │  → 不同 expert 组合 → hidden state 开始发散           │
  └──────────────────────────────────────────────────────┘
```

内存对比：
- Naive: n × (KV-cache per sample) → O(n·L·H·S)
- Clean Cache: 1 × (KV-cache) + n × (current hidden states) → O(L·H·S + n·d)
- 节省：>90% KV-cache 内存（n 较大时）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **Sample 0 角色**：batch 中固定第一个位置为 clean path，使用 τ=0 确定性路由。Sample 0 的 attention K/V 追加到共享 KV-cache。
2. **其余 sample**：共用 sample 0 维护的 KV-cache，仅对当前 token 做 attention lookup，不更新 KV-cache。
3. **残留发散容忍**：虽然 sample 1..n-1 的历史 hidden state 与 sample 0 有差异（因 expert 选择不同），论文的实证结果表明仅当前 token 的多样化路由已能产生足够的输出多样性，因为早期层的路由随机性在后续层会持续传播。
4. **与标准 cache 切换**：论文中 OLMoE 使用 Clean Cache，其余模型（Mixtral、GPT-OSS）使用标准 cache（每 sample 独立 KV-cache）。选择取决于模型对 KV-cache 偏离的敏感度。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
