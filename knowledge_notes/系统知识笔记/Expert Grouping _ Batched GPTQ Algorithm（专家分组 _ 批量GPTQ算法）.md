## Expert Grouping / Batched GPTQ Algorithm（专家分组 / 批量GPTQ算法）

术语是什么？
Expert Grouping 是 QMoE 为应对 MoE 模型 "1000× 更多小层导致 GPU 利用率极差" 问题而提出的优化策略：将每层多个 expert（默认 16 个）的 GPTQ 量化批处理为联合操作。核心操作：(1) 对 group E 内的每个 expert，提取其校准输入 X_E 并计算 Hessian H_E = X_E · X_E^T（因 MoE expert 矩阵小，直接 matmul 计算比 per-sample 累加快）；(2) 将所有权重矩阵 W_E 和 Hessian H_E 堆叠为 3D tensors；(3) 修改 GPTQ 逐列量化算法使其在 3D batch 维度上同时操作——压缩 group 内所有 expert 同步进行，减少 kernel launch overhead 并提高 GPU SM 利用率。

从系统架构角度拆解术语：
```
# 标准 GPTQ (per-expert, 串行):
for each expert E in layer:
    加载 W_E 到 GPU
    for each column j:
        量化 W_E[:,j]
        用 H_E 逆校正误差并传播
    → GPU 利用率低（小矩阵无法填满 SM）

# Expert Grouping (QMoE batched GPTQ):
for each expert group E (|E|=16):
    同时加载 W_E1, W_E2, ..., W_E16 到 GPU
    提取 X_E1, X_E2, ..., X_E16（各可能不同大小，zero-pad）
    计算 H_Ei = X_Ei @ X_Ei^T（直接 matmul，无 per-sample 累加）
    # 堆叠为 3D tensors:
    W_stack = [W_E1; W_E2; ...; W_E16]  # [|E|, d_row, d_col]
    H_stack = [H_E1; H_E2; ...; H_E16]  # [|E|, d_col, d_col]
    
    # Batched GPTQ（3D tensor 操作）
    for each column j:
        Q_stack[:,:,j] = quantize(W_stack[:,:,j])
        error = (W_stack[:,:,j] - Q_stack[:,:,j]) / H_stack[:,j,j]
        W_stack[:,:, j+1:] -= error * H_stack[:,j, j+1:]  # broadcast 校正
```

效果（switch-base-128 sparse layer, 10k samples）：

| |E| | Time | Speedup |
|-----|------|--------|
| 1   | 174.1s | 1× |
| 4   | 54.4s | 3.2× |
| 16  | 28.8s | 6.0× |

|E|=16 选择的 trade-off：更大 group 需要更多 GPU memory 同时 hold 更多 expert weights + activations——16 为 A6000 48GB 下的最优 trade-off。

术语一般如何实现？如何使用？
- 实现：修改 GPTQ 的 column-wise 循环，将 2D (d_row, d_col) 操作扩展为 3D (|E|, d_row, d_col)
- 直接 matmul 计算 Hessian（vs 原 GPTQ 的 per-sample accumulation）：可行因 MoE expert 矩阵小（通常 d_col ≤ 2048）
- 适用：任何需要批量处理大量小矩阵的 quantization 场景（不仅 GPTQ，也适用 ZeroQuant 等）
- 限制：|E| 受 GPU memory 约束（需同时 hold |E|×d_row×d_col weights + |E|×max_tokens×d_col activations）

涉及论文标题：
- QMoE Sub-1-Bit Compression of Trillion-Parameter Models
