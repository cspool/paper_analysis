## Mixture-of-Memories (MoM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mixture-of-Memories (MoM) 是一种在 Linear Attention 框架中引入多个独立 memory state 的架构，受生物神经元 theta-gamma 振荡机制和 Mixture-of-Experts 思想的启发。核心设计：用 M 个独立的 memory state M^m ∈ R^{d×d} 替代传统线性模型中单一固定大小的 memory state，通过 Router 网络将每个输入 token 路由到 top-k 个 memory state 进行选择性更新，最后加权混合得到输出。

MoM 与 MoE 的关键区别：(1) Purpose: MoE 是为了在不显著增加计算量的情况下扩大参数量，MoM 是为了扩大线性模型的 memory capacity 并消除 memory interference；(2) Structure: MoE 的 experts 是 FFN 内的独立网络（channel mixing），MoM 的每个 memory 是 RNN state 配独立 K/V projection weights（token mixing）。

MoM 与 Gating Mechanism 的区别：Gating（如 forget gate a_t）是通过数据依赖的衰减系数选择性"遗忘"旧信息来减少干扰；MoM 是通过将不同信息写入不同 memory state 来实现"分离存储"，从根本上避免不同信息之间的覆盖（而非衰减后再覆盖）。

从算法pipeline角度拆解术语。

**MoM 层前向计算流程**：

```
输入: X ∈ R^{T×d}, 参数: W_g ∈ R^{d×M}, W_k^m, W_v^m ∈ R^{d×d} for m=1..M

Step 1 - Router (token assignment):
  for each token t:
    scores_t = TopK(softmax(x_t @ W_g), k)   # 选 top-k 个 memory
    g_t = scores_t / sum(scores_t)             # 归一化 importance weights

Step 2 - Memory-specific K/V projection:
  for each activated memory m:
    k_t^m = x_t @ W_k^m    # memory-specific key
    v_t^m = x_t @ W_v^m    # memory-specific value

Step 3 - Memory update (仅对激活的 memory):
  M_t^m = UpdateRule(M_{t-1}^m, k_t^m, v_t^m)
  # 非激活 memory: M_t^m = M_{t-1}^m (保持不变)

Step 4 - Shared memory update (始终激活):
  M_t^shared = UpdateRule(M_{t-1}^shared, k_t^shared, v_t^shared)

Step 5 - Memory mixing:
  M̃_t = Σ_{m in activated} g_t^{(m)} · M_t^m + M_t^shared

Step 6 - Output:
  o_t = q_t @ M̃_t
  o_t = activation(norm(o_t)) @ W_o
```

默认配置：M=4 memories + 1 shared memory，top-k=2 激活（activation ratio=0.5）。

复杂度：training 保持 O(n)（每个 memory 处理对应 subsequence，总计算量仍与总 token 数成线性），inference O(1)（每个 memory 维护固定大小的 d×d state，与序列长度无关）。

术语一般如何实现？如何使用？

MoM 的硬件高效实现通过 Triton varlen kernel 实现：① 将 tokens 按 routing 结果分组到各自 memory bucket；② 同 bucket tokens concat 为 varlen 序列；③ Triton kernel 对每个 segment 独立并行计算（chunk-wise parallel scan）；④ 输出加权混合。开源代码：https://github.com/OpenSparseLLMs/MoM 和 https://github.com/OpenSparseLLMs/Linear-MoE。MoM 使用 Gated DeltaNet 作为默认 memory update 方法，替换 K/V projection 为 memory-specific 版本，并施加 auxiliary loss（参考 Switch Transformer 的 load balancing loss）确保 memory 路由均衡。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---
