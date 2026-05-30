## Shared FFN (Cross-Layer Feed-Forward Network Sharing)

术语解释
在 Transformer 架构中，让所有（或部分）Transformer block 共享同一套 FFN（Feed-Forward Network）参数，而非每层拥有独立 FFN。FFN 通常包含 gate projection（W_gate）、up projection（W_up）和 down projection（W_down）三组权重，占 Transformer 总参数的约 65%。共享 FFN 的核心思想是利用 FFN 层间的参数冗余，将 22 份独立 FFN 缩减为 1 份共享 FFN。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在标准 Transformer decoder（如 Llama 架构）中，每层包含 MHA（Multi-Head Attention）+ MLP/FFN 两个子模块，每个子模块有独立的参数。FFN 通常采用 SwiGLU 结构：`FFN(x) = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))`。这三个权重矩阵（W_gate, W_up, W_down）在每层都有独立副本。在 large-base（1.2B 参数）中，FFN 占总参数的 65%（attention 30%、heads 5%）。

MobiLlama 的 Shared FFN 设计：**所有 22 层 Transformer block 共用同一套 W_gate、W_up、W_down**，而 attention 层的 Q/K/V/O projection 每层独立。共享 FFN 将总参数从 1.2B（large-base）降至 0.5B（减少约 60%），同时保持 22 层的深度和 hidden 2048 的宽度。关键设计理念是"从大开始再缩小"：先设计高容量架构（large-base: 22 层 + hidden 2048），再通过参数共享机制降低参数量，而非从一开始就在层数或宽度上妥协。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准独立 FFN vs 共享 FFN 的计算对比：

```
# === 标准每层独立 FFN (如 Llama) ===
for layer in 1..L:
    h = h + MHA[layer](RMSNorm(h))          # 每层独立的 attention 参数
    h = h + FFN[layer](RMSNorm(h))          # 每层独立的 FFN 参数
    # FFN[layer] 含 W_gate^l, W_up^l, W_down^l，每层独立
    # 总 FFN 参数 = L × 3 × d_model × d_intermediate

# === 共享 FFN (MobiLlama) ===
shared_FFN = SwiGLU_FFN(W_gate, W_up, W_down)   # 仅 1 份 FFN 参数
for layer in 1..L:
    h = h + MHA[layer](RMSNorm(h))          # 每层独立的 attention 参数（保留）
    h = h + shared_FFN(RMSNorm(h))          # 所有层共享同一 FFN
    # 总 FFN 参数 = 1 × 3 × d_model × d_intermediate
    # 参数节省: (L-1) × 3 × d_model × d_intermediate

# === SwiGLU FFN 内部计算 ===
def SwiGLU_FFN(x, W_gate, W_up, W_down):
    gate = SiLU(x @ W_gate.T)       # SiLU(x) = x * sigmoid(x)
    up = x @ W_up.T
    return (gate * up) @ W_down.T   # element-wise product then down-project
```

在 MobiLlama 0.5B 中，每层 attention 仍有独立参数（32 heads, Q/K/V/O 各投影），仅 FFN 共享。0.8B 版本通过 widening 共享 FFN（增加 hidden dim 和 intermediate size）在不增加层数的前提下扩大容量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **代码层面**：在模型定义中仅实例化一个 SwiGLU FFN 模块，所有 decoder layer 引用同一模块对象。PyTorch 中为 `self.shared_ffn = SwiGLU_FFN(config)`，每层 forward 中调用 `self.shared_ffn(x)`。
2. **训练**：共享参数的梯度从所有层汇聚到同一份参数上，AdamW 优化。学习率与独立 FFN 训练相同。
3. **适用场景**：资源受限设备上的 SLM（0.5B-1B 级别），需在参数预算内最大化模型容量。
4. **与 Basis Sharing 的区别**：Basis Sharing（SVD-based）通过低秩分解共享权重的"基向量"，每层有不同组合系数。Shared FFN 更激进——完全共用相同参数，无层间差异。
5. **局限性**：所有层共享同一 FFN 意味着 FFN 无法学习层特定的特征变换，某些需要层特异性 FFN 的任务可能退化。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices
