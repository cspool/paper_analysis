## Sparse GEMV Kernel for MoE Expert (Triton-based, Column-Major, Selective Loading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse GEMV Kernel 是 FloE 提出的针对 MoE expert FFN 的自定义稀疏矩阵-向量乘法 kernel，基于 Triton 语言实现（参考 CATS kernel 修改）。核心设计：(1) 将 W_down 转置为列主序存储（W_down^T），使其列与 W_gate 的列对齐——同一 intermediate neuron 对应的 gate 列和 down 列在内存中连续，共享相同的稀疏掩码；(2) 根据 up projection 输出的幅值掩码（mask = |x @ W_up| ≥ t），选择性仅加载 W_gate 和 W_down^T 中被掩码选中的列，跳过其余列的读取；(3) 将 SiLU 激活和 element-wise multiply（Hadamard 积）融合到每个 Triton block 中执行，避免中间结果 x' 的多次 global memory 读写，减少 kernel launch 次数。在 RTX 3090 上，90% 稀疏度时单 expert 计算延迟从 0.524ms 降至 0.263ms（1.99× 加速）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FloE Sparse GEMV Kernel 伪代码 (Triton-based, Algorithm 1)
// 输入: hidden state x [1, d_hidden=4096]
//        sparse threshold t_ij (per-expert, offline determined)
//        expert weights E_ij = {W_gate, W_down^T, W_up}
//        W_gate:  [d_hidden, d_intermediate], row-major, 仅 mask 选中列驻留 GPU
//        W_down^T: [d_hidden, d_intermediate], column-major (转置后), 仅 mask 选中列
//        W_up:    [d_hidden, d_intermediate], INT2 quantized → dequant on CPU

// GPU Kernel 入口 (单个 expert, batch=1):
function sparse_expert_gemv(x, t_ij, E_ij):
    // Step 1: up projection (全精度, 密集 GEMV)
    v = x @ W_up_deq                    // [1, d_intermediate], W_up 已解量化
    
    // Step 2: 生成稀疏掩码 (element-wise)
    mask = (|v| >= t_ij)                // bool[d_intermediate], True≈10% at 90% sparsity
    
    // Step 3: 融合 sparse gate GEMV + SiLU + Hadamard (Triton fused)
    //         每个 Triton block 处理若干选中的列
    x_prime = fused_sparse_gate(v, mask, x, W_gate_cols)
    // fused_sparse_gate 内部:
    //   for each col j where mask[j] is True:
    //       gate_j = SiLU(dot(x, W_gate[:, j]))
    //       x_prime[j] = gate_j * v[j]
    
    // Step 4: sparse down GEMV
    //         W_down^T[:, mask] 列主序, 列宽 d_hidden
    y = sparse_down_gemv(x_prime, mask, W_down_T_cols)
    // sparse_down_gemv 内部:
    //   对每个输出维度 k:
    //       y[k] = sum_{j where mask[j]} x_prime[j] * W_down[j, k]
    //              = sum_{j where mask[j]} x_prime[j] * W_down^T[k, j]
    return y

// Triton kernel 关键优化:
// 1. W_down^T 列主序: 对每个选中的列 j, W_down^T[:, j] 在内存中连续
//    配合 W_gate[:, j] 连续, 两列可一次合并读取
// 2. 列选择性加载: 通过 mask 索引数组 indptr 定位选中列
//    load(W_gate_base + indptr[j] * d_hidden) 而非遍历所有 d_intermediate 列
// 3. 融合操作: SiLU + multiply 在寄存器中完成, 无需写回 global memory
```

FloE Table 1 单 expert 执行延迟 (ms) 对比：
| GPU | Dense (0%) | 50% sparse | 70% sparse | 90% sparse |
|-----|-----------|-----------|-----------|-----------|
| RTX 3090 | 0.524 | 0.379 (1.43×) | 0.305 (1.72×) | 0.263 (1.99×) |
| A6000 | 0.524 | 0.365 (1.44×) | 0.305 (1.72×) | 0.277 (1.89×) |
| A100 | 0.253 | 0.195 (1.30×) | 0.176 (1.44×) | 0.155 (1.63×) |
| H100 | 0.253 | 0.134 (1.26×) | 0.176 (1.44×) | 0.155 (1.63×) |

注：RTX 3090 和 A6000 在 90% 稀疏度下达 ~2× 加速；H100/A100 受 kernel launch overhead 限制，高稀疏度下加速递减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 Triton (Tillet et al., 2019, https://github.com/triton-lang/triton) 实现，参考 CATS (Lee et al. 2024a) 的 sparse kernel 设计
- W_down 的转置+列主序存储是性能关键——传统 row-major 下，稀疏化使每行仅部分列有效，导致非连续访问；转置后按列加载，每列连续，内存合并效率高
- 配合 compact weights layout (gate 列 + down 列 co-locate in DRAM)，一次 PCIe 传输即可获得两列数据
- 在 consumer GPU (RTX 3090) 上稀疏加速比显著，在数据中心 GPU (H100) 上因计算吞吐极高，稀疏化的相对收益递减
- 该 kernel 的输入 x 为单个 token (batch=1)，是典型的 latency-sensitive 场景；若增加 batch size，稀疏化的不规则访存可能成为瓶颈

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
