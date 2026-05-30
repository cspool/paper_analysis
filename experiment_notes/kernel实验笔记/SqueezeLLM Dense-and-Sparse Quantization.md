## SqueezeLLM Dense-and-Sparse Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  自研CUDA kernel实现两类操作：(1) **LUT-based非均匀量化矩阵-向量乘法**：对3/4-bit量化的dense权重矩阵，加载压缩bit indices → LUT查表获得FP16 centroid值 → FP16向量内积。每个output channel有独立LUT（如8个FP16 centroid对应3-bit），kernel按piece-by-piece方式dequantize以最小化内存带宽占用。(2) **Balanced CSR稀疏矩阵-向量乘法**：处理Dense-and-Sparse decomposition中的稀疏分量（CSR格式存储outliers+sensitive values）。由于sparsity pattern在各输出channel间高度不均衡（部分channel含大量nonzero），标准thread-per-row策略效率低。采用balanced hybrid kernel：按固定nonzeros/thread (10 nz/thread)分配工作，线程间额外同步但负载均衡。Dense和sparse kernel在单次launch中融合执行，避免中间结果叠加开销。

  实验比较：在A6000 GPU上对比FP16 baseline、GPTQ（non-grouped和grouped g128 with activation ordering）的延迟(s)和峰值内存(GB)，生成128和1024 tokens。A100上额外对比kernel-only matrix-vector runtime。关键对比项：dense-only (0% sparsity) vs 0.45% sparsity balanced kernel vs standard CSR kernel vs 0.45% sparsity。

- 后端平台是什么，配置是什么。
  NVIDIA A6000 GPU (48GB, primary latency benchmark)、NVIDIA A100 GPU (80GB, kernel-only matrix-vector runtime benchmark)。CUDA kernel实现，LUT-based dequant + balanced CSR SpMV。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Torch CUDA profiler测量延迟和峰值内存。自研kernel代码开源在 https://github.com/SqueezeAILab/SqueezeLLM。修改/新增内容：
  - 新增3/4-bit LUT-based非均匀dequantization+矩阵向量乘CUDA kernel
  - 新增balanced CSR稀疏矩阵-向量乘kernel (10 nz/thread)
  - Dense+Sparse kernel融合launch

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/SqueezeAILab/SqueezeLLM (MIT license)。

  **Kernel推理全流程（以LLaMA-7B 3-bit + 0.45% sparsity, A6000 GPU, batch_size=1为例）**：

  **1. 权重数据加载到GPU内存**：
  ```
  对每个Linear层:
    Dense分量:
      - indices_3bit: uint8 packed array [in_features // packed_elements]
        将连续的3-bit indices pack进更大的整数类型以高效memory access
      - LUTs: FP16 array [out_features × 8]
        每个output channel的8个centroid值 (对应3-bit的8个量化级别)
    Sparse分量 (CSR格式):
      - row_ptrs: int32 [out_features + 1]    行边界偏移量
      - col_indices: int16 [nnz]               列索引
      - values: FP16 [nnz]                     稀疏权重值 (≈0.45% × 总参数)
  ```

  **2. LUT-based Dense Matrix-Vector Kernel**：
  ```
  // grid: (out_features / BLOCK_SIZE) blocks
  // block: BLOCK_SIZE threads
  
  __global__ void lut_dequant_matvec_kernel(
      const uint32_t* packed_indices,  // 3-bit indices packed
      const half* LUTs,                 // [out_c × 8] FP16 centroids
      const half* activation,           // [in_features] FP16
      half* output                      // [out_features]
  ) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row >= out_features) return;
  
      half* lut_row = LUTs + row * 8;  // 当前channel的8-entry LUT
      half acc = 0.0;
  
      // 逐块加载packed indices, LUT查表, 乘accumulate
      for (int chunk = 0; chunk < num_chunks; chunk++) {
          uint32_t packed = packed_indices[row * num_chunks + chunk];
          for (int j = 0; j < indices_per_chunk; j++) {
              uint8_t idx = extract_bits(packed, j * 3, 3);  // 提取3-bit index
              half w_deq = lut_row[idx];                     // LUT查表→FP16
              acc += w_deq * activation[global_col];         // FP16乘累加
          }
      }
      output[row] = acc;
  }
  ```
  关键设计：weight按块(而非一次性)dequantize以减少寄存器压力和内存带宽；所有算术在FP16完成。

  **3. Balanced CSR Sparse Matrix-Vector Kernel**：
  ```
  // 问题：标准CSR kernel (每线程处理一行)在行间nonzeros严重不均衡时效率低下
  // 解决：Balanced kernel (每线程固定10个nonzeros, 一行可由多线程合作处理)
  
  __global__ void balanced_csr_matvec_kernel(
      const int32_t* row_ptrs,
      const int16_t* col_indices,
      const half* values,
      const half* activation,
      half* output
  ) {
      // 按nonzeros总数分配线程: num_threads = nnz / 10
      int nz_start = threadIdx.x + blockIdx.x * blockDim.x * 10;
      int nz_end = nz_start + 10;
  
      half local_acc = 0.0;
      for (int nz = nz_start; nz < min(nz_end, total_nnz); nz++) {
          int col = col_indices[nz];
          half val = values[nz];
          local_acc += val * activation[col];
      }
  
      // 确定该nonzeros范围所属的行
      int row = binary_search_row(row_ptrs, nz_start);
  
      // Atomic add到output (同一行可能被多个线程更新)
      atomicAdd(&output[row], local_acc);
  }
  ```
  性能对比：Standard CSR kernel 0.45% sparsity → 3.9s (7B); Balanced kernel 0.45% sparsity → 1.7s (7B) (>2x faster)。

  **4. Fused Kernel Launch (Single Call)**：
  ```
  // Dense和Sparse kernel在单个CUDA stream中顺序launch
  // 但output buffer复用, 无需额外中间结果sum kernel
  cudaMemset(output, 0, ...);
  lut_dequant_matvec_kernel<<<grid, block>>>(...); // Y = D @ X (写入output)
  balanced_csr_matvec_kernel<<<grid, block>>>(...); // Y += S @ X (累加到output)
  ```

  **5. 评估原理和端到端性能 (A6000, 128 tokens, LLaMA-7B 3-bit)**：
  | Kernel配置 | Latency (s) | Mem (GB) | PPL (C4) |
  |-----------|-------------|----------|----------|
  | FP16 Baseline | 3.2 | 12.7 | 7.08 |
  | GPTQ 3-bit (no group) | 1.4 | 2.9 | 9.55 |
  | SqueezeLLM 0% sparse | 1.5 | 2.9 | 7.75 |
  | GPTQ 3-bit g128 (w/ reorder) | 13.7 | 3.0 | 7.89 |
  | SqueezeLLM 0.45% (standard CSR) | 3.9 | 3.2 | 7.56 |
  | **SqueezeLLM 0.45% (balanced)** | **1.7** | **3.1** | **7.56** |

  Key takeaway：
  - LUT-based dequantization overhead vs uniform quant: ~7% latency increase (1.4→1.5s)换来perplexity从9.55→7.75
  - Balanced sparse kernel将CSR overhead从>2x (3.9s)降至~13% (1.7s vs 1.5s)
  - GPTQ grouped kernel因activation ordering引发的scattered memory access导致严重降速(13.7s)
  - A100上的kernel-only benchmark: SqueezeLLM 3-bit达到1.5-2.5x speedup vs FP16 matvec kernel
