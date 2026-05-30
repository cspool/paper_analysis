## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本文为综述论文，无原创kernel实验。§5.3.1 "Inference Accelerating" 系统梳理了LLM推理kernel优化技术：
  (i) **FlashAttention [76]**：IO-aware fused attention kernel，利用tiling和recomputation避免具体化完整N×N attention矩阵，将HBM读写从O(N²)降至O(N)，实现prefill阶段加速。FlashAttention-2 [75]通过改进work partitioning减少非MatMul FLOPs并增加并行度。
  (ii) **Flash-Decoding [78]**：针对decoding阶段batch size大、seqlen短的特点，在seqlen维度上额外并行化，设计专门CUDA kernel加速decode。
  (iii) **FlashDecoding++ [146]**：在Flash-Decoding基础上进一步优化softmax操作和flat GEMM，并增加AMD GPU支持。
  (iv) **DeepSpeed-Inference [21]**：针对小batch size场景（FM serving常见但FM training罕见）的GPU kernel优化。
  (v) **ByteTransformer [468]**、Google PaLM serving system [314]也提供了GPU/TPU的小batch优化kernel。
  (vi) 论文表4给出了多种attention变体的时间复杂度与空间复杂度对比（Transformer O(T²d)、Reformer O(T log T d)、Linear Transformers O(T d²)、RetNet O(T d)、RWKV O(d)等）。

- 后端平台是什么，配置是什么。
  被引述kernel的硬件平台：NVIDIA A100/H100 GPU、AMD GPU、Google TPU v4。综述未进行统一实验。

- 评估性能的软件/脚本是什么。修改了什么。
  论文使用flops-profiler（https://pypi.org/project/flops-profiler/）对GPT-2及Stable Diffusion 2.1进行FLOPs分析（§2.1.3、§2.3.3），而非性能benchmark。该工具为现有工具，综述未修改其实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  综述全部材料开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey。以下以FlashAttention为例说明kernel输入到性能输出的全过程：

  传统attention vs FlashAttention的kernel计算流程：
  ```
  // 传统Attention (Standard)
  // 瓶颈：N×N attention matrix具体化到HBM
  Load Q, K from HBM           // [N, d] each
  S = Q @ K^T                  // [N, N], 写入HBM
  P = softmax(S)               // [N, N], 读写HBM
  O = P @ V                    // [N, d], 读写HBM
  // HBM访问量: O(N²) >> N
    
  // FlashAttention (IO-Aware Fused Kernel)
  // 分块计算，避免具体化完整N×N矩阵
  for j in 0..T_c-1:           // K,V blocks loaded once per outer loop
      Load K_j, V_j from HBM to SRAM   // [B_c, d] each
      for i in 0..T_r-1:       // Q, O blocks
          Load Q_i, O_i, l_i, m_i from HBM to SRAM
          S_ij = Q_i @ K_j^T            // [B_r, B_c] on-chip
          m_ij = rowmax(S_ij)           // local softmax rescaling
          P_ij = exp(S_ij - m_ij)       // safe softmax numerator
          l_ij = rowsum(P_ij)           // local denominator
          // 在线更新running statistics
          m_new = max(m_i, m_ij)
          l_new = exp(m_i - m_new)*l_i + exp(m_ij - m_new)*l_ij
          O_i = diag(exp(m_i - m_new)) * O_i + exp(m_ij - m_new) * P_ij @ V_j
          m_i = m_new; l_i = l_new
          Store O_i, l_i, m_i to HBM
  // SRAM: ~20TB/s vs HBM: ~1.5-2TB/s (A100)
  // 结果：2-4× 加速，10-20× 内存节省
  ```
