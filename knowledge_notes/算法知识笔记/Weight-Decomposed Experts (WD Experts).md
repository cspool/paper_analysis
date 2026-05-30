## Weight-Decomposed Experts (WD Experts)

术语解释
Weight-Decomposed Experts 是将 MoE 中每个 expert 的 FFN 权重矩阵替换为低秩分解（Low-Rank Decomposition），以在保持模型质量的同时减少总参数量的一种技术。由 CoSMoEs (Huber et al., 2025) 提出，应用于端侧设备（on-device）的 MoE 预训练。

术语是什么？
Weight-Decomposed Experts 的核心思想：每个 expert 旨在"专门化"处理约 1/E 的 token 子集（E = 总 expert 数），因此不需要 full-rank 的 FFN 权重矩阵来捕获其专业知识。将每个 expert 的权重矩阵 M ∈ R^{n×m} 替换为两个低秩矩阵的乘积：

$$M_{n \times m} \approx L_{n \times r} \times R_{r \times m}$$

其中 r ≪ n 且 r ≪ m。CoSMoEs 论文采用 r = n/2（half hidden dimension）作为最优 trade-off。每个 expert 的三个 FFN 子矩阵（gate_proj, up_proj, down_proj in SwiGLU）均被低秩分解替代。

WD 与 LoRA (Hu et al., 2021) 相似但应用于预训练阶段：LoRA 在冻结的预训练权重上添加低秩适配器（W + ΔW = W + BA），WD 从零开始以低秩形式训练 expert 的权重。

从算法pipeline角度拆解术语。
WD Expert 在 MoE layer 中的前向计算（以 SwiGLU FFN 为例）：

```
# 标准 Expert FFN:
# Expert_i(x) = W_down_i @ (SiLU(W_gate_i @ x) * W_up_i @ x)

# WD Expert FFN (三个权重矩阵各分解为 L×R):
def wd_expert_forward(x, expert_idx):           # x ∈ R^{seq × d_model}
    # Gate: W_gate ∈ R^{d_ff × d_model} → L_gate ∈ R^{d_ff × r}, R_gate ∈ R^{r × d_model}
    h_gate = SiLU(x @ R_gate[expert_idx].T @ L_gate[expert_idx].T)
    # Up: W_up ∈ R^{d_ff × d_model} → L_up ∈ R^{d_ff × r}, R_up ∈ R^{r × d_model}
    h_up = x @ R_up[expert_idx].T @ L_up[expert_idx].T
    # Gated activation
    h = h_gate * h_up                              # [seq, d_ff]
    # Down: W_down ∈ R^{d_model × d_ff} → L_down ∈ R^{d_model × r}, R_down ∈ R^{r × d_ff}
    out = h @ R_down[expert_idx].T @ L_down[expert_idx].T  # [seq, d_model]
    return out
```

参数量对比（Phone-sized WD MoE, d_model=1600, d_ff≈6400, r=800）：
- 标准 expert: 3 × 6400 × 1600 = 30.7M params per expert
- WD expert: 3 × (6400×800 + 800×1600) = 3 × 6.4M = 19.2M params per expert
- 节省约 37% per expert

不同 r 的 trade-off（CoSMoEs 初步实验）：
- r = n (full rank): baseline 性能，无参数节省
- r = n/2: 最佳 trade-off，参数显著减少 + 性能提升（WD > standard MoE by 1.1%）
- r = n/4: 参数更少但性能下降，表达能力不足
- r = n/8: 参数极少但性能大幅退化

术语一般如何实现？如何使用？
- 实现：在 HuggingFace Transformers 中，修改 expert 的 `nn.Linear` 层为两个串联的 `nn.Linear(d_model, r)` + `nn.Linear(r, d_ff)`
- 训练：WD Experts 从零开始预训练，需要端到端训练 L 和 R 矩阵
- 适用场景：端侧设备部署（内存受限），减少每个 expert 的存储和加载开销；expert 数量较多时（E≥4）积累的参数节省显著
- 与 BlES loss 正交组合：WD 减少每个 expert 大小（Memory），BlES 减少 expert 切换频率（Latency）
- 局限：低秩分解增加矩阵乘法次数（3→6），但 batch=1 时小矩阵乘法更 cache-friendly；r 的选择需要针对模型规模调优

涉及论文标题：
- CoSMoEs Compact Sparse Mixture of Experts

---
