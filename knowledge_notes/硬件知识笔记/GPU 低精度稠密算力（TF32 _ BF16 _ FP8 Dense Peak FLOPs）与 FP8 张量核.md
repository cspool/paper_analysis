## GPU 低精度稠密算力（TF32 / BF16 / FP8 Dense Peak FLOPs）与 FP8 张量核

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPU 数据中心的稠密（dense，非稀疏）峰值算力按数值精度分层：TF32（TensorFloat-32，NVIDIA Ampere 起的 19-bit 格式，用于不需要完整 FP32 的矩阵乘）、BF16（bfloat16，8-bit 指数/7-bit 尾数，训练主力）、FP8（E4M3/E5M2，8-bit 浮点，Hopper/Blackwell 的新一代低精度主力）。低精度格式通过更高效利用芯片面积与功耗换取 FLOPs 倍增。论文 Table I（NVIDIA 官方 datasheet 数据）：
  - A100：TF32 156 / BF16 312 / FP8·INT8 624 TFLOPs
  - H100 NVL：418 / 840 / 1671
  - B200 HGX：1100 / 2250 / 4500
  即 B200 的 TF32 是 A100 的 7×，FP8 是 A100 TF32 的 29×，FP4 更达 58×。
- 含义：FP8 算力远超 TF32/BF16，但只有模型真正以 FP8 执行才兑现。LLM 已成功采用 FP8（DeepSeek-V3 等），而 LRM 因数值敏感、小 GEMM 量化开销、通信密集迟迟未用（论文 survey：top-500 Ads 模型 FP8 训练 0%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件实现：Tensor Core（张量核）按精度提供不同吞吐路径——同一组 MMA（matrix multiply-accumulate）硬件单元在低精度下每周期算更多元素（FP8 每指令算 2× BF16 元素、4× TF32 元素）；累加器仍为 FP32。B200（Blackwell）相对 A100 的 7×/29×/58× 来自新工艺、更大 chip 与更宽的张量核数据通路。
- 运转流程（一次 FP8 GEMM）：x、W 以 FP8 进入 Tensor Core → MMA 指令（如 mma.m16n8k32 的 FP8 变体）在单周期内完成更多乘加 → FP32 累加器累加 → epilogue 反量化/归一化/激活 → 写 HBM。LRM 场景的落差：27 个生产 shape 实测最大有效 TFLOPS < 峰值 20%——硬件算力被小 GEMM 的量化/布局开销与 tile 利用率限制，无法接近峰值。
- 与 SM/线程组织关系：Tensor Core 位于 SM 内（见"Tensor Core 与 SM 内管线划分"条目）；低精度加速的前提是数据以正确 layout 与缩放到达张量核，这由 kernel 调度层（量化 recipe、tile 选择、SM wave quantization 规避）保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：NVIDIA A100/H100/B200（CUDA Tensor Core）、GB200 NVL72、AMD MI300X/MI350X（ROCm 张量核，FP8 支持）；由 cuBLAS/cuDNN、Transformer Engine、DeepGEMM、FBGEMM、TorchAO 等暴露 FP8 kernel。使用：低精度训练/推理框架按算子选择 FP8 路径；LoKA 的逐算子 Dispatch 即按 shape 选最快 FP8 实现，在 H100/B200/GB200/MI300X/MI350X 五种硬件上都验证（B200 因显存更大→batch 更大，低精度收益比 H100 更高；开发期未见的 GB200/MI350X 无修改直接获得相当加速）。别名：dense peak FLOPs、低精度张量核吞吐、FP8 Tensor Core。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
