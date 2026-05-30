## Tensor Parallelism (TP / 张量并行)

术语是什么？
Tensor Parallelism (TP) 是将单个 Transformer layer 内的权重矩阵沿特定维度切分到多个 GPU 上的模型并行技术。由 Megatron-LM (Shoeybi et al., 2019) 提出。在 FFN 模块中，第一个 GEMM 的权重沿列切分（column-wise），第二个 GEMM 沿行切分（row-wise），使得中间计算在各设备上独立执行（无需通信），仅在 Dropout 前/后各需一次 all-reduce 同步。TP 的通信为节点内 NVLink all-reduce（高带宽低延迟），因此 TP size 通常限制在单节点 GPU 数（如 8）。

从kernel调度角度拆解术语：
FFN 模块的 TP 切分与通信调度（T 个设备）：
```
输入: X [b*s, h]
权重: A [h, 4h/T] per device (column-wise cut)
      B [4h/T, h] per device (row-wise cut)

// Forward
Y_i = GeLU(X @ A_i) @ B_i    // [b*s, h], 独立计算 T 份
Y = all_reduce(Y_1, ..., Y_T) // inner-node NVLink, 2*(T-1)*bsh/B 数据量

// Backward
∂L/∂Y_i = ∂L/∂Y              // 直接使用（已 all-reduced）
∂L/∂X_i = ∂L/∂Y_i @ B_i^T @ GeLU'(...) @ A_i^T
∂L/∂X = all_reduce(∂L/∂X_1, ..., ∂L/∂X_T)
```

在 PPMoE 中，TP 不仅用于 backbone，还用于 expert parallel——experts 分布在 TP group 内，MoE 层的 all-reduce 与 TP FFN 的 all-reduce 完全一致。

术语一般如何实现？如何使用？
Megatron-LM/Core 通过 `tensor_model_parallel_size` 配置。TP 的通信开销分析（Eq. 5）：t_all-reduce/t_cal = (T-1)TF/(4Bh)，以 V100 (F=125 TFLOPS, B=300 GB/s NVLink, T=8, h=10^3) 为例约 35/6≈6，远低于 DPMoE 的 all-to-all 开销。通常与 DP/PP 组合使用（3D 并行）。Pipeline MoE 论文中 TP=8 是默认配置，保证所有 experts 在单节点内。

涉及论文标题：
- Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism
