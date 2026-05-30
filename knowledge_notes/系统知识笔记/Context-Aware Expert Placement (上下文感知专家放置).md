## Context-Aware Expert Placement (上下文感知专家放置)

术语解释
Context-Aware Expert Placement 是一种基于 prefill 阶段激活统计的 per-sequence 动态 expert 放置策略，用于 GPU-NDP 异构 MoE 推理系统。与传统的 context-agnostic 静态/on-demand 放置不同，它利用 prefill 统计作为 decoding 行为的早期指标，在 prefill 后一次性地将 hot experts 迁移至 GPU（FP16）、cold experts 保留在 NDP（量化），decoding 期间零迁移。

术语是什么？
Context-Aware Expert Placement 解决的核心问题：MoE expert 激活是高度 context-dependent 的——同一 expert 在不同输入序列、不同 decoding step 上的"热度"不同。静态放置（基于全局频率）无法捕捉这种变化，on-demand 放置（每次需要时迁移）又引入频繁的 PCIe 带宽争用。

关键机制：
1. Prefill 统计收集：在 prefill 阶段累加每 expert 的激活计数 P_{l,e} 和路由评分 W_{l,e}
2. 重要性计算：S_{l,e} = α·P̃_{l,e} + (1-α)·W̃_{l,e}
3. Hot/Cold 分类：per-layer 按 S 排序，top-K → GPU (FP16)，其余 → NDP
4. Once-per-sequence migration：prefill 后一次性完成 expert 迁移，decoding 期间不变

从系统架构角度拆解术语：
Context-Aware Placement 在 GPU-NDP 系统推理全流程中的调度：

```
Sequence 到达:
  ┌─────────────────────────────────────────────────────┐
  │ Prefill Stage (GPU only)                            │
  │  tokens → Attention → Router → Expert FFN           │
  │  └→ 收集 (P_{l,e}, W_{l,e})   # 轻量计数器累加     │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │ Placement Decision (prefill 后, 一次)                │
  │  S_{l,e} = αP̃_{l,e} + (1-α)W̃_{l,e}                │
  │  H_l = TopK(S_l, K)       # GPU: FP16 hot experts   │
  │  C_l = rest               # NDP: quantized cold     │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │ Single Expert Migration (一次性)                     │
  │  H_l 中不在 GPU 的 experts → PCIe → GPU HBM (FP16) │
  │  C_l 中不在 NDP 的 experts → PCIe → NDP DDR (量化)  │
  └─────────────────────────────────────────────────────┘
                          ↓
  ┌─────────────────────────────────────────────────────┐
  │ Decoding Stage (固定 placement, GPU-NDP 重叠)       │
  │  for each token:                                    │
  │    Router → top-2 experts                           │
  │    if both in GPU: GPU FFN (FP16)                   │
  │    if GPU+NDP: GPU FFN || NDP FFN (quantized)       │
  │    if both in NDP: NDP FFN × 2 (sequential)         │
  │  → 零 expert migration during decoding               │
  └─────────────────────────────────────────────────────┘
```

对比 baseline (MoNDE context-agnostic):
```
MoNDE (on-demand):
  Decoding step t: Router → need expert e
    → if e in GPU: OK
    → if e in NDP: [expert weight ~170MB NDP→GPU via PCIe] OR
                   [activation ~8KB GPU→NDP, FP16 NDP compute]
    → 频繁迁移 + NDP FP16 瓶颈

本论文 (context-aware):
  Prefill → Placement once → Zero migration during decoding
  → per-token: activation ~8KB if NDP expert selected
  → 无 weight migration, NDP 量化执行
```

术语一般如何实现？如何使用？
- 统计收集：per-layer per-expert 计数器（2 metrics × E experts × L layers，可忽略的内存开销）
- K 选择：由 GPU HBM 容量决定——Mixtral-8×7B: K=4 (80GB HBM 可容纳)，Mixtral-8×22B: K=2
- α 参数：控制 frequency vs routing score 的权重，论文使用 α=0.5
- 核心假设：prefill-decode activation similarity ~0.89 (cosine sim) —— prefill 统计可靠预测 decoding 行为
- 适用条件：prefill 阶段的 expert 激活分布与 decoding 阶段相似 → MoE 推理（非 training）
- 局限性：对极短 prompt（prefill tokens 少）统计可能不稳定；仅适用于 encoder-decoder 的 decoder 阶段或 decoder-only 模型

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
