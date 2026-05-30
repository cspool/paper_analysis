## Sparse Tensor Core (SpTC)

术语解释
Sparse Tensor Core（SpTC）是 NVIDIA 从 Ampere 架构（SM80, A100）开始引入的稀疏矩阵乘法硬件单元，能对满足 2:4 结构化稀疏模式的矩阵运算提供 2× 峰值吞吐提升。

术语是什么？
Sparse Tensor Core 是 NVIDIA Tensor Core 的扩展，在标准 Tensor Core 的 FMA（Fused Multiply-Add）数据通路中增加了一个**稀疏选择器（sparsity selector）**。对于稀疏矩阵 A（满足 2:4 即每连续 4 个元素中恰好 2 个为零），A 被压缩为：(1) data 矩阵——shape 为 $m \times k/2$，仅存储非零值；(2) metadata 矩阵——每 4 元素组用 2-bit 编码非零值在组内的位置（00/01/10/11 对应 4 种可能）。SpTC 执行时，metadata 控制选择器从密集矩阵 B 中选取对应元素参与乘法，非零值直接相乘而零值被跳过。每个 SpTC 指令（如 `mma.sp.m16n8k32`）的完成延迟与等效密集指令相同（~24.7 cycles for FP16），但有效计算量为 2×——因为 M=16 个非零元素来自原始的 32 个（50% 稀疏率）。NVIDIA Ampere 的 SpTC 要求稀疏矩阵在**A 操作数**位置，B 操作数必须为密集。支持的 shape：`m16n8k32`、`m16n8k16`（FP16）、`m16n8k16`、`m16n8k8`（TF32）、`m16n8k64`、`m16n8k32`（INT8）。

从硬件架构角度拆解术语：
SpTC 在 GPU SM（Streaming Multiprocessor）内部的 Tensor Core 流水线中工作：

```
SM 内部 SpTC 计算流程（以 mma.sp.m16n8k32 FP16 为例）：
1. 数据加载阶段：
   - ldmatrix 指令从 shared memory 加载：
     - A_data: 16×16 FP16（来自压缩的 16×32 原始矩阵，仅非零值）
     - A_metadata: 16 个 2-bit 选择器（编码 16 组 4 选 2）
     - B: 16×8 FP16（密集矩阵，从 32×8 中按 metadata 选取）
   
2. SpTC 阵列执行（硬件选择器通路）：
   for each FMA lane (128 lanes in m16n8k32):
       group_4_idx = k_pos / 4           # 每 4 个 k 元素一组
       sel_bits = metadata[group_4_idx]   # 2-bit 选择码
       nonzero_a1 = A_data[group_4_idx * 2]      # 第 1 个非零值
       nonzero_a2 = A_data[group_4_idx * 2 + 1]  # 第 2 个非零值
       b1 = B[sel_pos1]  # 按选择码选取
       b2 = B[sel_pos2]
       C += nonzero_a1 * b1 + nonzero_a2 * b2
   # 结果：16×8 输出，等效于 32×8 密集运算（2× 计算量）
   
3. 硬件资源约束：
   - 每 SM 4 个 Tensor Core partition
   - 每 SM 的寄存器文件 65536 × 32-bit
   - 每 SM shared memory 最大 164 KB (A100) / 228 KB (H100)
   - SpTC 指令不增加额外延迟——选择器已集成在标准 FMA 通路中
```

SpTC 的关键硬件属性：(1) 选择器在每周期选择 2 个非零 B 元素参与 FMA，4 周期完成一组 4 元素处理；(2) 由于选择器的存在，即使执行密集 mma 指令也可能经过选择器（但此时 metadata 为全 0，不跳过任何元素），因此 SpTC 的延迟与密集 TC 相同但吞吐翻倍；(3) A100 上 `m16n8k16` 变体吞吐不如 `m16n8k32`（选择器开销在较短 k 维度上摊销不足）；此问题在 RTX 3070 Ti 等消费级 Ampere GPU 上不出现。

术语一般如何实现？如何使用？
- **PTX ISA**：通过 `mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32` 指令直接编程。A 矩阵需 `layout.aligned` 保证 16-byte 对齐。
- **CUDA 库**：cuSPARSELt（NVIDIA 官方）、CUTLASS SpMM。在 PyTorch 中通过 `torch.sparse.semi_structured` 和 ASP（Automatic Sparsity）训练工作流使用。
- **兼容硬件**：NVIDIA A100/A30/A40/A10/RTX 3090/RTX 4070/RTX 4090/H100/H200/B200（Ampere+）；AMD MI300/CDNA3 系列有等效 sparse ALU 但缺少 `cp.async` 和 `ldmatrix` 原生支持，移植时内存效率可能下降（Samoyeds 论文 Table 1）。
- **使用约束**：(1) 仅 A 操作数可稀疏（Ampere 代），Hopper（SM90）引入 `wgmma.mma_async.sp` 扩展为异步指令；(2) 固定 50% 稀疏率（2:4）；(3) 需要离线将权重矩阵编码为 data+metadata 对。

涉及论文标题：
- Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores
