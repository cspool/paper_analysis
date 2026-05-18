## Sparse Tensor Core (SpTC)

术语是什么？

Sparse Tensor Core (SpTC) 是 NVIDIA Ampere 架构（A100）开始引入的专用硬件单元，利用 2:4 结构化稀疏模式加速稀疏矩阵计算。2:4 sparsity 要求每连续 4 个元素中至少 2 个为零，达到固定 50% 稀疏率。SpTC 通过非零元素位置元数据（2-bit per 4-element group）指导硬件只对非零元素执行乘累加，跳过所有乘零操作，理论上实现密集 TC 2 倍的数学吞吐量（A100 FP16: 312→624 TFLOPS sparse）。SpTC 执行的是 sparse × dense = dense 的矩阵乘累加（与密集 TC 的 dense × dense = dense 不同），第一个操作数 A 必须是 2:4 稀疏压缩格式，第二个操作数 B 是密集矩阵，输出 C 是密集矩阵。

从硬件架构角度拆解术语：

SpTC 硬件执行流程（以 Drawloom Short Mapping 在 A100 上为例）：
1. **稀疏矩阵压缩**：非零元按 2:4 pattern 分组，每 4 个位置保留最多 2 个非零元，存储压缩 value（50% 存储节省）+ 2-bit 元数据编码非零位置（0/1/2/3）
2. **向量 X 重映射**：根据稀疏矩阵的列索引（remapped CID 而非 original CID），将密集向量 X 对应位置的元素加载到 SpTC 的 B 矩阵输入
3. **SpTC 执行**：硬件根据元数据选择 B 中对应的 activation，只对非零元素执行乘累加——实际计算量减半
4. **输出**：累加结果写回，与密集 TC 相同的输出格式

术语一般如何实现？如何使用？

SpTC 通过 CUDA 的 `cusparseLt` 库或 CUTLASS 的 2:4 sparse GEMM kernel 使用。cuSPARSELt 提供 prune + compress + matmul 的端到端 API。编程模式下，权重矩阵需预剪枝满足 2:4 模式（通常按 weight magnitude 保留每组最大的 2 个权重），剪枝后固定稀疏模式 fine-tune。在 Drawloom 中，SpTC 被用于 Short Mapping——nnz≤T2 的短行 strip 通过 2:4 sparsity 压缩后映射到 SpTC block，避免了 DASP 将这些短行 fallback 到 CUDA Cores 的低效方案。Drawloom 是首个同时调度 TC 和 SpTC 计算不同稀疏度矩阵区域的 SpMV 方案。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
- Uni-STC: Unified Sparse Tensor Core

