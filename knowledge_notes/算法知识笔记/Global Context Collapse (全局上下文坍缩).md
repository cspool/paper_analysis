## Global Context Collapse (全局上下文坍缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Global Context Collapse 是 Zhang et al. (ICLR 2026) 在对 linear attention 进行系统分析时识别出的核心失败模式。其本质是：standard linear attention 将所有 token 压缩进一个固定的 d×d 全局 KV summary（G = Σ_j φ(K_j)^T V_j ∈ R^(d×d)），所有 query 共享这一 summary，导致当序列长度 N 增长时，信息量超出 d×d 矩阵的承载能力，模型的表示多样性和注意力选择性发生坍缩。

该现象可通过两个互补指标量化：
1. **Rank 限制**：A_lin = Q̃ K̃^T 的 rank ≤ min(rank(Q̃), rank(K̃)) ≤ d。当 N >> d 时，attention 矩阵严重秩亏，模型无法捕获多样的 query-conditioned 注意力模式。实测显示 linear-attention 模型 attention score 的 rank 始终被 head dimension（d_h ≤ 72）限制。
2. **稀疏性丧失/熵升高**：随 N 增大，每个 token 对全局 summary 的贡献趋于无穷小，注意力分布趋向均匀分布。实测 linear attention 的注意力熵显著高于 softmax attention，表明模型无法聚焦于少量信息量高的 token。

MHLA 论文通过 Imagenet DeiT 和 Wan2.1 视频生成（N=31500 tokens）的实验验证了该现象：视频生成中 vanilla linear attention 几乎无法训练（loss 平台高），而 MHLA 恢复了正常收敛。

从算法pipeline角度拆解术语。

**Global Context Collapse 在 attention pipeline 中的表现**：

```
// Standard Linear Attention — 共享全局 summary
G = zeros(d, d)
z = zeros(d)
for j in 1..N:
    G += phi(k_j)^T @ v_j       // 所有 token 信息混合进 [d, d]
    z += phi(k_j)
for i in 1..N:
    o_i = (phi(q_i)^T @ G) / (phi(q_i)^T @ z)  // 所有 query 共用同一 G

// 问题：当 N >> d 时，G 的信息容量饱和（rank ≤ d）
//   → 不同 query 获得的 context 几乎没有差异
//   → 注意力分布趋于 uniform（高熵）
//   → 模型失去聚焦能力
```

术语一般如何实现？如何使用？

该概念用于诊断 linear attention 的性能瓶颈，而非直接实现。缓解策略包括：(a) MHLA——将单个全局 summary 拆为 M 个局部 summary + 可学习混合；(b) Focused Linear Attention——添加 DW-Conv 注入局部信息；(c) GLA——门控机制；(d) Mamba——state space 模型。理解该概念有助于在长序列场景下选择或设计合适的 attention 变体。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
