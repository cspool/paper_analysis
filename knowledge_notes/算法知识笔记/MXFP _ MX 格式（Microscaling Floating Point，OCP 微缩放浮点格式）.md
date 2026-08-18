## MXFP / MX 格式（Microscaling Floating Point，OCP 微缩放浮点格式）

术语解释
MXFP（OCP Microscaling Formats，MX）是 AMD/Intel/Microsoft/NVIDIA/Qualcomm 在 Open Compute Project 下定义的块共享指数（block-wise shared exponent）缩放浮点格式：把一组 FP 值（block，如 32 元素）共用一个 8-bit 指数（E8M0）做归一化，block 内每个元素用更少的 exponent/mantissa 位表示（MXFP4 E2M1、MXFP6 E2M3/E3M2、MXFP8 E4M3/E5M2），以少量元数据换取大幅扩展的动态范围。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：极低位宽浮点（如 FP4，S1E2M1）单个张量只用一个缩放因子 S，离群值抬高 S 导致正常元素丢失精度（精度差）；→ 块共享指数把整张量切成 block，每个 block 独立 8-bit 共享指数（E_b^shared，OCP 规范中为 E8M0），先按 block 归一化再量化，等效按 block 调整动态范围、缓解离群值影响（精度提升）；→ block 结构天然匹配 Tensor Core 的块状数据（如 NVIDIA Blackwell 原生支持 MXFP4/FP4 低比特执行），元数据（8B/block）被 block 内元素分摊（硬件/内存高效）。数学形式（MXFP4，S1E2M1）：X_{b,fp4}^q = Quant(X_b^{fp16}/2^{E_b^{shared}})，E_b^{shared} = |log2(max_b(|X_b^{fp16}|))| − E_element^MAX，其中 E_element^MAX 是元素格式最大指数（E2M1 的 bias=2^(2−1)−1=1，E_element^MAX=11_2−bias=2）。Web 证据：OCP Microscaling Formats (MX) Specification v1.0（https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf）；NVIDIA NVFP4 博客（https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在量化推理 pipeline 中，MX 格式按"块内共享指数 + 固定元素配置"组织张量：例如 4-bit MXFP4，block=32：
```
对每个 block b:
  E_b^shared = |log2(max_b|X_b^{fp16}|)| - E_element^MAX   # E_element^MAX=2 (E2M1)
  for x_i in X_b: x̂_i = quant(x_i / 2^{E_b^shared})        # 4-bit E2M1
```
Tensor Core 执行时先按 block 取回 8-bit 共享指数与量化元素，点积后按指数缩放累加进高精度 accumulator。局限：MXFP 对所有 block 用单一固定配置（如 MXFP4 一律 E2M1），无法适应块间/块内值多样性（见"块间/块内值多样性"条目），正是 MXFFP 论文的切入点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：OCP 规范定义 E8M0 共享指数与 E4M3/E5M2/E3M2/E2M3/E2M1 等元素格式；参考实现与生态：OpenXLA/StableHLO 加入 MX 类型（f4E2M1FN、f6E2M3FN、f6E3M2FN、f8E8M0FNU，https://github.com/openxla/stablehlo/pull/2582）、ggml/llama.cpp 的 MXFP 实现（https://github.com/ggml-org/llama.cpp PR #20609）、NVIDIA CUTLASS。使用：权重静态离线转 MX 格式、激活运行时转换（OCP-compliant conversion rule）；NVIDIA Blackwell 硬件原生支持 FP4/MXFP 低比特执行，XLA 生态通过新增 MX primitive type 打通到硬件。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
