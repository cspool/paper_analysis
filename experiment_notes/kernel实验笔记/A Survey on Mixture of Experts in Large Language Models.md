## A Survey on Mixture of Experts in Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文为综述，不提供原始实验。它在系统计算的 Section 5.1 中综述了面向 MoE 动态稀疏性的 GPU kernel 与编译器优化：
  - **MegaBlocks [137]**：将 MoE 计算重新表述为 block-sparse 操作，开发专用 block-sparse GPU kernel，在不丢弃 token 的前提下高效处理动态负载
  - **PIT [139]**：面向 MoE 动态稀疏性的深度学习编译器，利用 Permutation Invariant Transformation（数学可证性质）将多个稀疏微 tile 变换为 GPU 高效密集 tile，再执行 dense GEMM，不改变计算结果
  - **ScatterMoE [138]**：通过 ParallelLinear 模块执行分散组的并行线性运算，避免 scatter-to-group 数据拷贝，减少内存占用，且易扩展到 FFN 以外的模块（如 Attention experts）
  - **定制 GPU kernel**：DeepSpeed-MoE [64]、FastMoE [129]、HetuMoE [134]、Tutel [130] 均针对 MoE 特有的 gate routing/input encode/output decode 操作开发定制 kernel，消除冗余计算和内存搬运

- 后端平台是什么，配置是什么。
  GPU 后端：NVIDIA GPU（MegaBlocks, ScatterMoE, PIT, DeepSpeed-MoE, FastMoE, Tutel 均针对 GPU 平台）。

- 评估性能的软件/脚本是什么。修改了什么。
  - **MegaBlocks**：修改了 MoE 前向/反向 kernel（scatter → block-sparse GEMM），基于 block-sparse 矩阵乘法实现
  - **PIT**：修改了深度学习编译器（tiling 机制），插入 PIT 变换规则在 operator 级别生成优化 kernel
  - **ScatterMoE**：修改了 MoE 的 scatter-to-group 流程，用 ParallelLinear 的 grouped GEMM 直接操作分散的 token 组
  - **DeepSpeed-MoE/FastMoE/Tutel**：修改了 gate routing/encode/decode 的 GPU kernel 实现

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  **开源链接**（Table 4）：
  - MegaBlocks: https://github.com/stanford-futuredata/megablocks
  - ScatterMoE: https://github.com/shawntan/scattermoe
  - PIT：论文未明确给出独立开源链接
  - DeepSpeed-MoE: https://github.com/microsoft/DeepSpeed
  - FastMoE: https://github.com/laekov/fastmoe
  - Tutel: https://github.com/microsoft/tutel

  **MegaBlocks Block-Sparse Kernel 执行流程**：
  1. **输入**：token 序列经 Router 得到 (token → expert) 映射（sparse matrix S ∈ {0,1}^{T×N}）
  2. **Block 化**：将 token-expert 映射矩阵 S 划分为固定大小的 block（如 128×128）
  3. **稀疏性编码**：只保留非空 block，形成 Block-Sparse 表示（CSR/CSC 变体）
  4. **Block-Sparse GEMM**：对非空 block 执行 batched dense GEMM（每 block 内部 dense 计算）
  5. **结果组装**：将各 block 的 GEMM 输出按原始 token 顺序组装为最终输出
  6. **输出**：每个 token 的 expert 计算输出（不丢弃任何 token）

  **PIT Compiler Kernel 执行流程**：
  1. **输入**：包含 MoE 层的模型计算图
  2. **PIT Tiling**：识别 MoE 操作中满足 Permutation Invariant Transformation 属性的 operator，将稀疏分散的 micro-tile 按固定 tile 大小重新排列为 dense tile
  3. **Kernel 生成**：对重组后的 dense tile 生成标准高效 GEMM kernel
  4. **结果逆变换**：将 dense tile 输出恢复为原始 token 顺序
  5. **输出**：与原始 MoE 计算等价的结果，但 GPU 利用率更高

  **ScatterMoE ParallelLinear 执行流程**：
  1. **输入**：token embeddings x 和 (token → expert) 映射
  2. **分组**：按 expert 将 token 分组（保持原顺序，无需 scatter-to-group 拷贝）
  3. **ParallelLinear**：对每组 token 直接执行 grouped GEMM（PyTorch-native tensor 操作）
  4. **组装**：将各 expert 输出按 token 原序拼接
  5. **输出**：中间表示保持为标准 PyTorch tensor，便于扩展至非 FFN expert
