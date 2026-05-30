## Fused RoPE Kernel with Symmetric Sorting (融合RoPE Kernel与对称排序)

术语是什么？
Fused RoPE Kernel with Symmetric Sorting 是 UniQL 为支持结构化剪枝后的 LLM 推理而设计的融合 CUDA kernel。其核心挑战：结构化权重排序破坏了 Rotary Position Embedding (RoPE) 的原始位置嵌入索引顺序——因为 Q 和 K 投影矩阵的列被排序矩阵 $\mathbf{S}_{qk}$ 重排，使得第 j 列的 RoPE 嵌入不再对应原始的第 j 个维度分量。UniQL 通过对称排序 + 融合索引 gather 在单个 kernel 中解决此问题。

**对称排序策略**：RoPE 按维度对 $(2d, 2d+1)$ 应用旋转：$\operatorname{RoPE}(\mathbf{x}; \theta) = [\cos\theta_d \cdot \mathbf{x}_{2d} - \sin\theta_d \cdot \mathbf{x}_{2d+1}, \sin\theta_d \cdot \mathbf{x}_{2d} + \cos\theta_d \cdot \mathbf{x}_{2d+1}]_{(2d,2d+1)}$。为在排序后保持 RoPE 的正确语义，UniQL 将 norm score 向量 $s \in \mathbb{R}^{D_{hd}}$ 对半分：$[s_1, s_2] = s$，然后对 $s_1 + s_2$ 排序（而非独立排序 $s_1$ 和 $s_2$），得到对称索引向量 $idx_{sym} = [\operatorname{argsort}(s_1 + s_2), D_{hd}/2 + \operatorname{argsort}(s_1 + s_2)]$。这保证每个 RoPE 维度对的相对顺序不变——对于原维度对的 $(d, d + D_{hd}/2)$，排序后变为对应的新位置对 $(d', d' + D_{hd}/2)$。

从kernel调度角度拆解：
```
# Kernel 输入
# - X_q/X_k: [T, D'_hd] 或 [T, D_hd], 已排序的 Q/K 激活
# - idx_sym: [D'_hd], 对称排序索引 (前半 = 后半 + D_hd/2)
# - cos_table, sin_table: [T, D_hd], 原始 RoPE 嵌入表

# 传统两阶段实现 (无融合):
# Stage 1: gather cos/sin 索引对应的值
cos_k = cos_table[:, idx_sym]                   # [T, D_hd], global mem read
sin_k = sin_table[:, idx_sym]                   # [T, D_hd], global mem read
# Stage 2: 应用 RoPE 旋转
for d in range(0, D_hd, 2):
    x0, x1 = X_k[:, d], X_k[:, d+1]
    X_k_rope[:, d]   = cos_k[:, d] * x0 - sin_k[:, d] * x1
    X_k_rope[:, d+1] = cos_k[:, d+1] * x1 + sin_k[:, d+1] * x0

# UniQL 融合 Kernel (单个 CUDA kernel):
__global__ void fused_rope_kernel(
    half* X, half* out,
    const half* cos, const half* sin,
    const int* idx_sym,
    int T, int D_hd
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_D = D_hd / 2;
    
    // 每个线程处理一对 RoPE 维度
    int d = tid * 2;
    if (d >= D_hd) return;
    
    int idx0 = idx_sym[d];         // 索引对的前半
    int idx1 = idx_sym[d + 1];     // 索引对的后半 (自动 = idx0 + half_D)
    
    for (int t = 0; t < T; t++) {
        half cos0 = cos[t * D_hd + idx0];
        half sin0 = sin[t * D_hd + idx0];
        half cos1 = cos[t * D_hd + idx1];
        half sin1 = sin[t * D_hd + idx1];
        
        half x0 = X[t * D_hd + d];
        half x1 = X[t * D_hd + d + 1];
        
        // 直接使用 gather 后的 cos/sin 计算 RoPE
        out[t * D_hd + d]     = cos0 * x0 - sin0 * x1;
        out[t * D_hd + d + 1] = cos1 * x1 + sin1 * x0;
    }
}
```

术语一般如何实现？如何使用？
Kernel 基础实现改编自 Liger-Kernel (Hsu et al., 2025) 的 RoPE kernel 和 Marlin (Frantar et al., 2024) 的 4-bit kernel。关键优化：①对称排序将索引存储减半（仅需前半 $D_{hd}/2$ 个索引，后半由 $+D_{hd}/2$ 隐式推导），节省寄存器/共享内存；②融合 gather + slice + RoPE 旋转在单个 kernel 中，避免多次 global memory 往返。在 A6000 上 Profile 结果（Table 9）：4-bit Llama-3.1-8B at 0% 剪枝率下 TPOT 从 9.9ms（无融合）降至 9.0ms（融合），10% latency reduction；at 25% 剪枝率从 8.6ms 降至 7.7ms（1.1× speedup）。Qwen-2.5-7B 同样显示 9.1ms → 8.3ms 和 7.9ms → 7.1ms 的改善。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
