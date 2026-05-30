## Sparse Tensor Cores (稀疏张量核心)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Tensor Cores 是 NVIDIA 从 Ampere 架构（SM 8.0, A100）起引入的硬件单元，专门加速结构化稀疏矩阵乘法。它们在标准 Tensor Cores 基础上增加了对 2:4 结构化稀疏的原生支持——每 4 个连续元素中保留 2 个非零值，硬件自动跳过零值元素。Sparse Tensor Core 通过 `mma.sp.sync` PTX 指令执行稀疏 MMA（Matrix Multiply-Accumulate），单指令处理 2:4 稀疏权重与稠密激活的矩阵乘。理论峰值吞吐量为同等稠密 Tensor Core 的 2×（因为一半计算被跳过）。Hopper 架构（SM 9.0, H100）进一步增强了 Sparse Tensor Core，支持 FP8/INT8 稀疏模式。Blackwell（SM 10.0, B100/B200）支持 FP4 sparse。这是 OBR 压缩模型推理加速实现的硬件基础。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Sparse Tensor Core 在 Ampere SM 中的运作流程：

```
// SM 内的 Tensor Core 数据路径

// 1. Metadata 解码
//   从 shared memory 读取 2:4 metadata (2-bit selector per group)
//   解码：group of 4 → select 2 of 4 values

// 2. 稀疏数据选择
//   从 packed 稀疏权重中选择非零值
//   [w0, 0, w2, 0] → 选择 w0, w2（position 0, 2）
//   减少 50% 的乘法运算

// 3. 稀疏 MMA
//   A (dense activation, M×K matrix tile)
//   B (2:4 sparse weight, K×N matrix tile, 50% zero)
//   C (accumulator, M×N matrix tile)
//   → 每个 Tensor Core 执行:
//     For k = 0 to K/4:
//       group = B[k*4 : (k+1)*4]
//       (idx0, idx1) = decode_metadata(group)
//       C += A[k*4+idx0] × group[idx0] + A[k*4+idx1] × group[idx1]

// 4. 吞吐量
//   A100 Sparse Tensor Core:
//     - FP16: 312 TFLOPS (dense) → 624 TFLOPS (sparse, theoretical)
//     - INT8: 624 TOPS (dense) → 1248 TOPS (sparse)
//   H100 Sparse Tensor Core (FP8):
//     - FP8: 1979 TFLOPS (dense) → 3958 TFLOPS (sparse)
```

Sparse Tensor Core 的约束：仅支持特定 tile shape（通常 M=16, N=8, K=32 for FP16）；稀疏模式固定为 2:4（不可配置为其他比例）；metadata 需 2bit/group 额外存储（约占权重的 6.25%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编程接口：(1) CUTLASS 模板 API（推荐）；(2) cuSPARSELt 库（高抽象级别）；(3) PTX `mma.sp.sync` 指令（底层）。使用条件：2:4 结构化稀疏（NVIDIA ASP 库或剪枝算法生成）；SM 8.0+ (A100) 或更高架构；数据类型兼容性（当前支持 FP16/BF16/TF32/INT8/FP8）。PyTorch 通过 `torch.sparse.semi_structured` 和 `torch._C._cuda_spspmm` 间接访问。OBR 利用 Sparse Tensor Cores 实现 INT4 2:4 sparse GEMM 的 5.9× vs FP16 dense 加速。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
