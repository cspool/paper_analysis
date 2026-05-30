## Weight-only Quantization (W4A16)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-only Quantization（仅权重量化）是一种只对模型权重进行低比特量化而保持激活值为高精度（通常 FP16/BF16）的量化策略。典型配置为 W4A16（4-bit 权重 + 16-bit 激活）。与 W8A8（权重和激活均为 8-bit）相比，W4A16 的权重存储更紧凑且不需要对激活进行量化-反量化操作，简化了推理系统的 kernel 设计。W4A16 模式下推理流程为：(1) 加载 packed 4-bit 权重；(2) dequantize 权重到 FP16；(3) 执行 FP16 × FP16 GEMM。这种方式在 memory-bound 的 LLM 解码阶段（batch=1, decode token）中效果尤为显著，因为瓶颈在于从显存中读取权重。AFPQ 论文采用 W4A16/W3A16 的 weight-only 量化策略，专注于优化权重量化格式（FP-asym、NF-asym）的精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W4A16 推理的计算流程（以 AFPQ 的 NF4-asym 为例）：
```
# 存储格式
# 权重: packed byte array, 每 2 个 4-bit NF4 索引占 1 byte
# 参数: 每 group (128 个权重) 存储 scale_pos 和 scale_neg (各 FP16)

# 推理时逐层计算
for each Linear layer:
    for each group of 128 weights:
        # Step 1: 解包
        for i in range(0, 128, 2):
            byte = packed_weights[byte_idx]
            nf4_idx_0 = byte & 0x0F  # 低 4-bit
            nf4_idx_1 = (byte >> 4) & 0x0F  # 高 4-bit
        
        # Step 2: LUT 映射 (NF4 index → FP16 value)
        for each index:
            fp16_val = NF4_LUT[index]  # 16-entry lookup table
        
        # Step 3: 非对称反量化
        for each fp16_val:
            if fp16_val > 0:
                w_deq = scale_pos * fp16_val
            elif fp16_val < 0:
                w_deq = scale_neg * fp16_val
            else:
                w_deq = 0
    
    # Step 4: FP16 GEMM
    output = FP16_GEMM(w_deq, activation_fp16)
```
与 W8A8 的区别：后者需在计算前同时 dequantize 权重和激活，且激活量化的信息损失更大（激活分布更难预测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W4A16 在 HuggingFace 中通过 `BitsAndBytesConfig(load_in_4bit=True)` 使用。GPTQ 和 AWQ 默认也采用 W4A16 策略。在自定义推理系统中（如 AFPQ 基于的 FasterTransformer），W4A16 需要实现低比特 dequantization kernel。主流推理框架对 W4A16 的支持：vLLM 支持 GPTQ/AWQ 的 W4A16 推理；TensorRT-LLM 通过 Weight-Only Quantization plugin 支持 INT4/FP4 W4A16；llama.cpp 通过 GGUF 格式支持各种 W4A16 格式。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- AffineQuant Affine Transformation Quantization for Large Language Models
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 采用 W4A16/W3A16/W2A16 的 weight-only 量化策略，仅对权重进行低比特量化而保持激活为 FP16。论文论证理由：(1) LLM 推理中激活不是显著瓶颈；(2) 通过权重量化减少内存移动即可充分加速 LLM 推理；(3) 可兼容仅支持整数运算的硬件（如 NPU）作为未来扩展方向。，通过 scaled pairwise rotation 变换在量化前抑制权重离群值。与仅 dequantize + FP16 GEMM 的标准 W4A16 不同，ParoQuant 在 GEMM 前插入 fused CUDA kernel 对激活 X 应用逆旋转变换 T^{-1}(X)，使变换在推理时在线完成（~10% 开销）而非离线合并（受限且无法覆盖所有线性层）。
