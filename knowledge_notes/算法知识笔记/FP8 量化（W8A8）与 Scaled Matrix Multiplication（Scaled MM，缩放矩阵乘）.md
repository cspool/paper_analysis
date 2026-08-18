## FP8 量化（W8A8）与 Scaled Matrix Multiplication（Scaled MM，缩放矩阵乘）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8 量化是 W8A8（8-bit 权重 + 8-bit 激活）低精度推理范式：权重与激活都量化到 FP8（E4M3/E5M2），推理时用 FP8 数据做矩阵乘，输出前按每 token/每 channel 的缩放因子反缩放（dequantize）。Scaled MM（Scaled Matrix Multiplication，vLLM 的 FP8 后端 kernel，源自 CUTLASS/DeepGEMM）正是 W8A8 推理的核心算子：`C = (A_fp8 * scale_A) @ (B_fp8 * scale_B)`，缩放因子随张量/块携带，避免整型量化需要的在线反量化开销——FP8 的浮点格式使 MMA 可直接在 Tensor Core 上以 FP8 精度执行、以更高精度累加，再乘回 scale。在 PIPEWEAVE 中它是 6 类被建模 kernel 之一（vLLM、CUDA C++、FP8、Tensor pipeline、HW/SW 双调度范式），是 FP8 推理场景性能预测的验证对象。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W8A8 FP8 推理的量化-计算 pipeline（以 GEMM 为例）：
```
# 离线量化（post-training）：
A_fp8 = quantize_to_fp8(A_fp32, scale_A)   # 每 token 或每 tensor 一个 scale
B_fp8 = quantize_to_fp8(B_fp32, scale_B)   # 每 channel 或每 block 一个 scale
# 在线推理（Scaled MM kernel）：
#   Tensor Core 执行 FP8 MMA：D = A_fp8 @ B_fp8 (FP32 累加)
#   epilogue 阶段：C = D * (scale_A * scale_B)   # 反缩放，fuse 进 epilogue
# PIPEWEAVE 对 Scaled MM 的建模维度：M∈[2,131072], N∈[384,8192], K∈[256,8192]
#   Tensor ops = α·tile_M·tile_N·tile_K（α=2），FP8 使每 SM Tensor 吞吐翻倍
```
FP8 在 Hopper 上每 SM 的 Tensor 吞吐是 BF16 的 2 倍，Scaled MM 因此成为 decode 阶段（带宽受限）之外 prefill/GEMM 密集阶段的主要加速手段。PIPEWEAVE 评估：在 seen GPU（H20/H800）上 Scaled MM 的 MAPE 1.9%/4.1%，unseen（H100/H200）4.2%/5.2%，比 Roofline/Linear/Habitat/Neusight 精度提升 10.8×/9.5×/5.5×/7.8×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：vLLM 的 FP8 路径用 CUTLASS scaled-GEMM 模板（scaled_mm 支持 E4M3/E5M2 与 per-tensor/per-channel scale），DeepGEMM 提供 Hopper/Ada 的高性能 FP8 GEMM（WGMMA + TMA），Transformer Engine 提供 FP8 训练推理栈。使用方式：模型权重离线量化到 FP8（可配合 activation 校准得到 scale）→ 推理框架（vLLM）在 FP8 支持的 GPU 上调用 Scaled MM kernel 替代 BF16 GEMM → PIPEWEAVE 等性能模型可据此预测 FP8 kernel 的延迟（其 Tensor pipeline demand 按 FP8 吞吐计算）。注意：FP8 精度敏感，需要 scale 校准与 outlier 处理（常与 SmoothQuant 式激活缩放结合）。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
