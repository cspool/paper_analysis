## Kernel Fusion for Manifold-Constrained Hyper-Connections (mHC Fused Kernels)

术语是什么？

mHC 的 Kernel Fusion 是为缓解 n-stream 残差设计引入的显存带宽瓶颈而设计的专用融合 GPU kernel 集合。由于 HC/mHC 将残差流宽度扩展 n 倍，标准实现下每个 token 的显存 I/O 增加约 $(8n+2)C$ 个元素读/写（n=4 时约 33C vs 标准残差连接的 3C）。mHC 通过 5 个融合 kernel 将显存带宽利用优化到 n=4 时仅 6.7% 额外时间开销。

五个融合 kernel 分别处理不同的计算阶段：
1. **融合线性投影+Norm kernel**（Eq.14-15）：将两次对 $\vec{\mathbf{x}}_l$ 的扫描（RMSNorm r 的计算 + 线性投影 $\vec{\mathbf{x}}_l \varphi_l$）融合为单一 kernel，利用 MMA 单元最大化显存带宽。反向两个矩阵乘法同样融合为单 kernel。
2. **融合后处理 kernel**（Eq.16-18）：将小系数上的轻量操作（RMSNorm 归一化 + gating factor 乘法 + bias 加法 + Sigmoid 激活）融合为单一 kernel，减少 kernel launch 开销。
3. **Sinkhorn-Knopp kernel**（Eq.19）：20 次交替行列归一化在单 kernel 内完成。反向实现自定义 kernel，片上重计算中间结果。
4. **Pre 映射应用 kernel**：计算 $\mathcal{H}_l^{\text{pre}} \mathbf{x}_l$ 聚合 n-stream → 1-stream。
5. **Post+Res 融合应用 kernel**：将 $\mathcal{H}_l^{\text{post}}$ 和 $\mathcal{H}_l^{\text{res}}$ 的应用与 residual merge 融合——读取元素从 $(3n+1)C$ 降至 $(n+1)C$，写入从 $3nC$ 降至 $nC$。

从kernel调度角度拆解：

mHC kernel 执行流程（前向）：
```
// Kernel 1: Fused Linear Projection + Norm
// Input: x_l (n, C) bf16, phi (nC, n^2+2n) tf32
// Output: H_tilde (1, n^2+2n) f32, r f32
// Grid: (1,), Block: single wave
// Pipeline: load bf16 → cast f32 → MMA tf32 → store f32
x_flat = flatten_to_1d(x_l)         // in-register
r = norm(x_flat) / sqrt(n*C)        // fused into same kernel
H_tilde = x_flat @ phi               // MMA on tensor cores

// Kernel 2: Fused Post-processing
// Input: H_tilde, alpha scalars, bias vector
// Output: H_pre, H_post (1,n), H_res_raw (n,n)
H_scaled = (1/r) * [alpha_pre*H_pre, alpha_post*H_post, alpha_res*H_res] + bias
H_pre = sigmoid(H_pre_part); H_post = 2*sigmoid(H_post_part)

// Kernel 3: Sinkhorn-Knopp (single kernel)
// Input: H_res_raw (n,n); Output: H_res (n,n) ~doubly stochastic
M = exp(H_res_raw)
for t=1..20: M = col_norm(row_norm(M))

// Kernel 4: Pre Mapping Application
// Input: H_pre (1,n), x_l (n,C); Output: layer_in (C,)
layer_in = H_pre @ x_l  // reduction: n streams → 1

// [Standard layer computation: F(layer_in, W_l)]

// Kernel 5: Post+Res Fused Application with Residual Merge
// Input: H_res (n,n), x_l (n,C), H_post (1,n), layer_out (C,)
// Output: x_next (n,C)
// Fused: eliminates separate read/write of intermediate results
x_next = H_res @ x_l + H_post.T * layer_out
// I/O: reads (n+1)C, writes nC (vs separate: reads (3n+1)C, writes 3nC)
```

术语一般如何实现？如何使用？

大部分 kernel（除 Kernel 1 的 MMA 融合核外）使用 TileLang 框架实现，TileLang 简化了复杂计算过程 kernel 的实现。混合精度策略：输入 bfloat16、权重 tfloat32、计算 float32。精细的 load→cast→compute→store 流水线处理混合精度。反向 pass 中 mHC kernel 被选择性重计算（recompute）而非保存所有中间激活。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
