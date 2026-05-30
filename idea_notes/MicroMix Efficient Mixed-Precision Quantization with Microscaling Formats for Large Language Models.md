## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- baseline方法是什么？
  - 现有 INT4 weight-activation 量化方法（如 Atom、QuaRot、QUIK、FlatQuant）采用固定数量的高精度通道（如 Atom 固定 128 个 INT8 通道），或使用旋转/平滑变换抑制 activation outlier。这些方法的 INT kernel 需要在 CUDA Core 上进行反量化（因为 INT8 Tensor Core 仅输出 INT32 部分和），无法利用 Blackwell 的 FP4 Tensor Core。
  - 全栈执行例子（Atom baseline）：
    - **算法层**：固定 keeper_size=128 个 INT8 通道，其余为 per-group INT4 (group_size=128)，activation sort metric="hessian"，无自适应层间精度分配
    - **系统框架层**：PyTorch + 自定义 CUDA kernel，INT8 Tensor Core 执行 GEMM，CUDA Core 执行 dequant + partial sum（INT32→FP16 转换），仅支持 Llama2-7B
    - **编译框架层**：论文未明确说明
    - **kernel调度层**：INT4×INT4 MMA 在 INT8 Tensor Core 上计算 → INT32 部分和 → CUDA Core 上 dequant（乘以 scale）→ FP16 累加。INT8 Tensor Core 限制：FP4 吞吐为 FP16 的 4×，而 INT8 Tensor Core 仅 2× FP16 吞吐；且 dequant 在慢速 CUDA Core 上执行
    - **硬件架构层**：NVIDIA RTX 4090 (Ada Lovelace) 等非 Blackwell 架构，无 FP4 Tensor Core 支持

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MicroMix 通过三个关键设计解决 baseline 缺陷：(1) **自适应混合精度分配**——每层基于激活分布动态计算 MXFP4/MXFP6/MXFP8 通道比例，替代固定 128 通道；(2) **量化阈值 T(n)**——从 INT8 量化误差上界推导 MXFP4/MXFP6 的允许值域，超阈值元素升级精度，确保 MXFP 误差不超 INT8；(3) **Blackwell 原生 FP4 kernel**——利用 Blackwell MMA 指令直接执行 FP4/FP6/FP8 GEMM，反量化融合在 Tensor Core 内完成，无需 CUDA Core dequant。
  - 全栈执行例子（MicroMix）：
    - **算法层**：每层离线计算 p4^k/p6^k/p8^k 和排列 σ^k（基于校准数据的通道均值排序 + 阈值分组），激活 online fused reorder-and-quantize，权重 offline 预量化。量化 block_size=32（比 Atom 的 group_size=128 更细粒度），使用 E8M0 scale 实现纯移位反量化
    - **系统框架层**：PyTorch + 自定义 CUTLASS GEMM kernel，支持 Llama/Qwen/Mixtral 多模型系列，模型无关的通用混合精度框架
    - **编译框架层**：基于 CUTLASS 模板实例化各精度 GEMM kernel，未修改编译器框架本身
    - **kernel调度层**：Blackwell Tensor Core 上执行：
      1. Fused reorder-and-quantize kernel（共享内存内重排 + 32 元素 block-wise MX 量化，E8M0 scale shift-only dequant）
      2. 三路 MXFP GEMM（MXFP4/MXFP6/MXFP8），MMA 指令融合 scale dequant → FP32 累加 → BF16 输出
      3. 通道恢复排列 σ^{-1}
      FP4 Tensor Core 提供 4× FP16 吞吐，dequant 零额外开销（MMA 内置），CUTLASS GEMM 高度解耦支持任意精度比例
    - **硬件架构层**：NVIDIA RTX 5070Ti/5090/PRO 6000 (Blackwell)。利用 FP4 Tensor Core (4× FP16 吞吐)，MMA 指令原生支持 MX 格式的 block-scaled 数据类型。shared memory / Tensor Memory 缓存 input tile + scales
  - 缺陷→设计映射：
    - 固定通道数忽略层间分布差异 → 自适应 p4/p6/p8（每层独立校准），p4 始终 >50% 保证效率
    - INT kernel 需要 CUDA Core dequant → Blackwell MXFP MMA 原生融合 scale dequant，反量化零额外延迟
    - 无 MXFP 异常值阈值定义 → 首次给出 MXFP4/MXFP6 的显式量化阈值 T(n)，确保误差不超过 INT8 上界
    - Atom kernel 仅支持 Llama2-7B → MicroMix 的 CUTLASS 解耦设计支持任意模型和多精度组合
    - 粗粒度 group quantization (group_size=128) → MX 标准 block_size=32 的细粒度量化，E8M0 移位反量化消除乘法开销
