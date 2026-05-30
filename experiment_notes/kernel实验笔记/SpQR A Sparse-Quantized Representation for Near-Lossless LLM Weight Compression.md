## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  设计了专门的GPU稀疏矩阵乘法kernel来处理SpQR格式中的unstructured outlier weights。核心实现：(1) 基于CSR（Compressed Sparse Row）格式存储outlier weights：将outlier按row-first, column-second排序，每个outlier存储16-bit value + 16-bit column index，每行一个32-bit cumulative row pointer；(2) 在推理时，warp-level load balancing：将权重矩阵划分为等大小blocks，每个thread block加载outlier slice到shared memory (SRAM)，每个GPU core判断outlier是否在其segment内，从DRAM加载对应权重值，执行有效行上的稀疏矩阵乘法；(3) 与dense-quantized matmul相结合：先用bilevel quantized weights做dense dequantize+matmul，再用CSR sparse kernel处理1% outliers的contribution。实验比较SpQR optimized kernel vs PyTorch sparse（cuSPARSE）vs FP16 baseline的token generation latency（tokens/s），在LLaMA-7B/13B/30B/65B上，batch size=1，分别测from scratch（100 tokens）和prefix 1024（扩展到1124 tokens）两种场景。

- 后端平台是什么，配置是什么。
  单张NVIDIA A100 GPU（80GB）。CUDA kernel为自研实现。PyTorch版本≥2.0.0 with CUDA support。

- 评估性能的软件/脚本是什么。修改了什么。
  评估脚本：`inference_demo.py`（SpQR源码仓库中的端到端推理脚本）。自研的SpQR CUDA kernel替换PyTorch默认的cuSPARSE实现。Kernel原理及修改：
  
  1. **Weight Layout变换**：quantized weights和量化统计量按block（β₁×β₂ = 256 weights）连续存储于DRAM，每个block包含256个packed 3-bit codes + 16 packed 3-bit scales/zeros + 4个FP16 second-level statistics。
  
  2. **Dense DequantMatmul Kernel**：Thread block加载当前block的统计量到SRAM → 第二层反量化(3-bit→FP16)→第一层反量化(3-bit→FP16) → 加载block内的packed 3-bit weights到SRAM → 逐weight dequantize到FP16 → 与SRAM中的activation vector执行点积 → 累加到output。
  
  3. **CSR Sparse Kernel（outlier处理）**：
     ```
     步骤1: 将矩阵划分为等大小blocks (tile)
     步骤2: 每个thread block加载一段outlier slice到shared memory (SRAM)
     步骤3: 每个GPU core遍历其tile内的rows:
             if tile包含该行的outlier:
                 从row pointer确定该行outlier range
                 加载列索引和对应FP16值
     步骤4: 执行sparse dot product: output[row] += Σ col_value[outlier] × activation[col_idx]
     ```
     通过步骤1-3实现load balancing，步骤4因outlier的row-wise pattern获得连续内存访问。

  4. **最终merge**：dense_matmul_result + sparse_outlier_result = final output。

  评估原理：在单张A100上，batch_size=1逐token生成，测量Scratch（从零生成100 token）和Prefix 1024（在1024-token prompt后追加100 token）两种场景下的tokens per second。结果显示SpQR optimized kernel相比FP16 baseline获得20-30%加速，比PyTorch稀疏+量化组合快约2倍。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Vahe1994/SpQR

  推理全流程（以LLaMA-65B 4-bit SpQR在A100上逐token生成）：
  
  1. **模型加载**：SpQR量化后的模型包含：
     - Dense部分：packed quantized weights (3-bit/4-bit) + bilevel quantized scales/zeros
     - Sparse部分：CSR格式outliers (row_pointers[N+1], col_indices[num_outliers], values[num_outliers])
  
  2. **逐层推理**（每层Linear层）：
     ```
     Thread Block分配: 每block负责一段连续的output rows (tile)
     
     // Dense MatMul部分
     for each weight block in tile:
         加载 quantization statistics 到 SRAM
         反量化统计量 (second→first level)
         for each group in block:
             加载packed weights到SRAM
             反量化到FP16
             dot_product(weights, activation_segment) → partial_dense[tile]
     
     // Sparse MatMul部分
     加载outlier slice (row_pointers tile范围) 到SRAM
     for each row in tile:
         if row有outliers:
             遍历该行outliers:
                 partial_sparse[row] += value[k] × activation[col_idx[k]]
     
     output[tile] = partial_dense[tile] + partial_sparse[tile]
     ```

  3. **性能关键**：SpQR的token generation是memory-bound操作，高压缩率（3.4x+ memory reduction）降低了DRAM读取量，即使增加sparse compute开销，整体wall-clock time仍比16-bit推理少20-30%。
