## Trainable_Dynamic_Mask_Sparse_Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现一个专用 CUDA kernel（Flash Dynamic Mask Attention），融合 FlashAttention 风格的 tiling 与 DMA 的可训练稀疏性。核心机制：(1) **Block-level mask skip**：outer loop 中加载 K/V block 前先加载对应 mask block M_j，调用 Judge(M_j) 判断该 block 是否全零——若 active=0，则 advance stream pointers 跳过该 K/V block，直接进入下一 block，避免矩阵乘法和内存访问。(2) **Forward pass**（Algorithm 1）：Q/K/V/M 分块加载到 SRAM，对 K block 未被屏蔽的位置计算 S_ij = Q_i K_j^T / sqrt(d_h) + M_j，使用 online softmax（m/l/O 递推）保证数值稳定性，只在不全零的 block 上执行计算。(3) **Backward pass**（Algorithm 2）：与 Forward 共享统一 skip logic，只在必要时 fetch K/V tiles。backward 中 dM=dS，kernel 只需局部重算 S 而无需额外存储中间 mask 梯度张量。梯度链包含 fused bias gradients。整个 pipeline 完全可微，支持端到端训练。使用 shared memory aliasing、pipelined prefetching、coalesced memory accesses 优化带宽和 occupancy。实验比较 MHA (FlashAttention)、SWA、MLA (FlashMLA)、NSA 的 forward/backward/decoding kernel 性能。

- 后端平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB GPU。Benchmark 配置：32 heads、8 KV heads、d_h=128、bf16 精度。各变体统一参数对比，forward pass 在 token 长度 8192/16384/32768 下，decode phase 在 key 长度 65536/131072/262144/524288 下，backward pass 在 8192/16384/32768 下测试。3 次 warmup + 1000 次 run 取平均。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 PyTorch + CUDA 自定义 kernel。相对于标准 FlashAttention 的修改：增加了 block-level mask skip 逻辑——在 K/V block 加载前检查 mask block，跳过全 zero block；增加了 mask 和 bias 的 batch/head/query broadcasting 支持；forward/backward 共享 skip logic；backward 中融合 bias 梯度和 mask 梯度（dM=dS，直接从 dS 推导而不额外存储）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/flash-algo/flash-sparse-attention。kernel 评估原理基于 tiled FlashAttention 架构。**输入到输出全过程**：
  (1) HBM→SRAM：Q/K/V/M 矩阵分块（T_r×T_c blocks，block size B）从 HBM 加载到 SRAM。每次 outer loop 加载一个 M block 到 SRAM。
  (2) Judge 判断：计算 active = Judge(M_j)。若 M block 所有元素为 −∞（全 zero），active=0，则 advance stream pointers 跳过该 K_j/V_j block，不执行任何 M×M（matrix multiply）操作。
  (3) 非跳过 block 计算：加载 K_j/V_j 到 SRAM，遍历所有 Q blocks。计算 S_ij = Q_i K_j^T × d_h^{−0.5} + M_j，使用 online softmax 递推公式合并到累计输出 O_i（m_new = max(m, rowmax(S_ij))，l_new = exp(m−m_new)·l + exp(S_ij−m_new)·rowsum）。
  (4) SRAM→HBM：将更新后的 O_i、l_i、m_i 写回 HBM。最终输出 O 为所有非跳过 block 的加权累积结果。有效复杂度 O(n·w·d_h)，内存 O(n·d_h)（无需物化完整 attention matrix）。
  Forward 速度提升：相对 MHA 在 8192/16384/32768 token 长度分别约 26.1×/10.2×/21.5×。Decode 提速：在 65536/131072/262144 key 长度分别约 49.6×/92.7×/171.1×。Backward 提速：在 8192/16384/32768 分别约 2.5×/4.4×/7.9×。
