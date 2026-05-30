## SwiGLU FFN in MoE (MoE中的SwiGLU前馈网络)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwiGLU (Swish-Gated Linear Unit) 是现代 LLM（包括 MoE 模型）中最常用的 FFN 激活函数。其计算定义为 FFN(X) = (SiLU(X·W_gate) ⊙ (X·W_up))·W_down，其中 SiLU(x) = x·σ(x)（σ 为 sigmoid）。W_gate, W_up ∈ R^{d_model × d_intermediate} 是两个上投影矩阵，W_down ∈ R^{d_intermediate × d_model} 是下投影矩阵，⊙ 是逐元素乘法。与传统 ReLU-FFN（两层 MLP + ReLU）相比，SwiGLU 用门控机制（gate 通道 + up 通道的逐元素乘）替代简单非线性，提供更好的训练稳定性和模型质量。

从算法pipeline角度拆解术语：
MoE-Prism 利用 SwiGLU 的一个关键数学性质——列独立性——来实现 expert 分解：
```
# SwiGLU FFN 前向计算
X = input  # [B, d_model]
A_gate = X @ W_gate  # [B, C], C = intermediate_size
A_up = X @ W_up      # [B, C]
A = SiLU(A_gate) * A_up  # [B, C], element-wise, 每列独立计算
output = A @ W_down  # [B, d_model]

# 列独立性: output_j 仅依赖于 A[:, j] 和 W_down[j, :]
# 按列分组 = 按neuron分组, 将FFN分解为子专家
for sub_expert S_n containing neurons [j1, j2, ...]:
    A_n = A[:, [j1, j2, ...]]           # 仅相关列
    output_n = A_n @ W_down[[j1, j2, ...], :]  # 仅相关行
    # expert总输出 = sum(output_n for all sub_experts)
```
这个列独立性意味着：每个 neuron（W_gate 的一列 + W_up 的一列 + W_down 的一行）可以独立计算其对最终输出的贡献，因此 SwiGLU FFN 天然可分解为 neuron 组（子 expert），且总输出是各组输出的精确求和。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- SwiGLU 由 Shazeer (2020) 在 "GLU Variants Improve Transformer" 中推广，现已是 LLaMA、Mixtral、DeepSeek、Qwen 等主流 LLM 的默认 FFN 激活。标准实现中 intermediate_size 通常为 d_model 的 8/3 到 4 倍。
- MoE-Prism 中使用的模型 intermediate_size：OLMoE-1B-7B: 1024→256 (per sub-expert)、DeepSeek-V2-Lite: 1408→352、Qwen3-30B-A3B: 6144→1536。
- 列独立性是 MoE-Prism Sub-Expert Decomposition 在数学上保持输出恒等性（不改变 FFN 输出值）的根本保证，无需微调即可重构 expert。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 从训练内存角度分析了 SwiGLU 的内存瓶颈——SwiGLU 需要两次投影（a = x·W1, b = x·W2）→ 逐元素 SiLU(a)·b → W3 下投影，传统 kernel 需在 HBM 中保存 a, b, σ(a), SiLU(a), y_swi 等 5 个中间张量用于反向传播，单个 MoE 层的中间激活可达约 98GB（DeepSeek 规模：L≈2M, h=24576, bf16）。MoEBlaze 提出 fused SwiGLU training kernel：将 W1/W2 两个 GEMM 融合为单 kernel，同时在 register/shared memory 中计算 SiLU 和 element-wise multiply；反向传播时采用 activation checkpoint——不保存 SiLU(a)，仅保存 a, b, y_swi，backward 时 recompute SiLU(a)（element-wise 操作，memory bandwidth bound，recompute 开销极低）。此方法在 SwiGLU 下实现最高 4× 激活内存减少（conf3: 40GB→10GB），训练速度提升 2×–6.2×。

---
