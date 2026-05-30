## Chain-of-Experts (CoE, Expert 链式处理架构)

术语解释
Chain-of-Experts (CoE) 是 Wang et al. (2025) 提出的一种新型 MoE 架构，将传统 MoE 层内 expert 的并行独立激活改为 C 步迭代顺序处理，引入 intra-layer expert communication。每步使用独立 Router 基于前一步的中间表示重新选择 expert，配合 inner residual connection 稳定多步训练。CoE 在相同 expert compute budget 下实现更高的 expert 组合多样性和更低的 validation loss。

术语是什么？
传统 MoE 层中，Router 一次性为 token x 计算 gating scores，TopK 选择 K 个 expert，所有 expert 并行独立处理同一输入 x，输出加权求和：$y = \sum g_i \cdot E_i(x)$。Expert 之间无交互。

CoE 修改为 C 步迭代：初始 $x^{(0)} = x$，每一步 $t = 1,...,C$ 执行：
$$x^{(t)} = \sum_{i=1}^{N} g_{t,i} \cdot E_i(x^{(t-1)}) + \mathbb{I}_r \cdot x^{(t-1)}$$

其中 $g_{t,i}$ 由第 t 步的独立 Router 计算（TopK 选择 K/C 个 expert），$\mathbb{I}_r = 1$ 为 inner residual connection。最终输出 $y = x^{(C)}$。

关键设计要素：
1. **Iteration-based Independent Routing**：每步有独立的 Router 参数（而非所有步骤共享同一 Router），使模型基于精炼后的中间表示动态调整路由
2. **Inner Residual Connection**：每一步将前一步的输出直接加到当前步输出，稳定多步训练（消融：inner=1.12 vs outer=1.21 vs init=1.18 loss）
3. **Sparsity Preservation**：每步仅选 K/C 个 expert，总计算量 = C×(K/C) = K，与标准 MoE 相同

从算法pipeline角度拆解术语：
```
# CoE Layer Forward (pseudocode)
def coe_forward(x, experts[1..N], routers[1..C], K, C):
    x_cur = x
    for t in range(1, C+1):
        logits_t = x_cur @ W_router[t]         # [d], N per step
        topk_scores, topk_idx = TopK(Softmax(logits_t), K/C)
        expert_out = sum(topk_scores[i] * experts[i](x_cur) for i in topk_idx)
        x_cur = expert_out + x_cur              # inner residual
    return x_cur
```

与标准 MoE 的关键差异：标准 MoE 为单步并行（所有 expert 看到同一 x），CoE 为多步迭代（后续 expert 的输入是 predecessors 精炼后的中间表示）。总 FLOPs 相同（K experts × 1 pass vs K/C experts × C passes）。组合空间从 C(N, 2K) 扩展到 C(N, K)^C（N=64, K=4, C=2 时 823×）。

术语一般如何实现？如何使用？
- 实现：PyTorch + veRL FSDP Trainer (https://github.com/volcengine/verl)，扩展 multi-round expert execution
- 模型配置（论文）：DeepSeek-V2-Lite 缩小版，544M params，4 layers，63 routed + 1 shared expert/layer，C=2, K=4/step
- 训练：AdamW，lr=3e-4，10% warmup，H100 单 GPU，<1 GPU hour/run
- 性能：val loss 1.20→1.12 (MetaMathQA)；C=2,L=4 ≈ MoE L=12 (-42% memory)；N=48,C=2 ≈ MoE N=64 (-17.6% memory)
- 局限：C>2 diminishing returns；单设备；sequential 减少单步 GEMM 并行度
- 开源：https://github.com/ZihanWang314/coe

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---
