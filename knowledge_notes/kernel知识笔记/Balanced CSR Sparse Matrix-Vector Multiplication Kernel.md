## Balanced CSR Sparse Matrix-Vector Multiplication Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Balanced CSR SpMV kernel 是处理高度不均衡稀疏模式的矩阵-向量乘法 CUDA kernel。标准 CSR kernel 将每行分配给一个线程处理，但当各行的 nonzero 数量严重不均衡时（如某些通道含远超平均的 outliers），会导致严重的线程间负载不均衡（部分线程空闲等待，部分线程执行大量计算）。Balanced kernel 改为按固定 nonzeros/thread（如 10 nz/thread）分配工作——一行可由多个线程合作处理，每线程处理固定数量的连续 nonzero 元素。代价是需要 atomicAdd 合并同一行的多线程部分结果，但在高度不均衡的稀疏模式下总体收益远大于同步开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Balanced CSR SpMV kernel (per-thread固定nonzeros, 非per-row)
// 处理 Dense-and-Sparse decomposition 的 sparse matrix S

__global__ void balanced_csr_matvec_kernel(
    const int* row_ptrs,        // [out_rows + 1]
    const short* col_indices,   // [total_nnz]
    const half* values,         // [total_nnz]
    const half* activation,     // [in_features]
    half* output                // [out_rows], 需初始化为0
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nz_per_thread = 10;
    int nz_start = tid * nz_per_thread;
    int nz_end = min(nz_start + nz_per_thread, total_nnz);

    half local_acc = 0.0;
    int current_row = -1;
    half row_acc = 0.0;

    for (int nz = nz_start; nz < nz_end; nz++) {
        int col = col_indices[nz];
        half val = values[nz];

        // 确定该 nonzero 所属的行
        int row = binary_search_row(row_ptrs, nz);
        if (row != current_row) {
            if (current_row >= 0)
                atomicAdd(&output[current_row], row_acc);
            current_row = row;
            row_acc = 0.0;
        }
        row_acc += val * activation[col];
    }
    if (current_row >= 0)
        atomicAdd(&output[current_row], row_acc);
}
```

性能对比（A6000, LLaMA-7B 3-bit + 0.45% sparsity, 128 tokens）：
| Kernel Type | Latency (s) |
|------------|-------------|
| Dense-only (0% sparse) | 1.5 |
| Standard CSR (0.45%) | 3.9 (+160%) |
| Balanced CSR (0.45%) | 1.7 (+13%) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源实现：https://github.com/SqueezeAILab/SqueezeLLM。基于 Flegar & Quintana-Ortí (Euro-Par 2017) 的 balanced CSR 方法。关键参数：10 nonzeros/thread 是论文实验的最佳设置。适用场景：(1) 稀疏矩阵中 per-row nonzero 分布高度 skewed（少数行含大量 nonzero，多数行几乎为空），常见于 Dense-and-Sparse decomposition 中 outliers 集中在特定 attention head/output channel；(2) batch_size=1 的 memory-bound 推理场景。与 cuSPARSE 通用 SpMV 的对比：(1) cuSPARSE 在低 batch size + 低 sparsity 下 overhead 大（format conversion, 间接内存访问），可能比 FP16 dense 推理更慢；(2) Balanced kernel 专为该场景设计，与 dense LUT kernel 融合在单次 launch 中。局限性：atomicAdd 在极多线程竞争同一行时可能成为瓶颈（虽然通常不会，因为 skewed 分布意味着大部分行只有少量 nonzero）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
