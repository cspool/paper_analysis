## QESC (Quantization with Expert-Selection Calibration)

术语解释
QESC 是 EAC-MoE 提出的 MoE-LLM 静态量化方法，核心思想是在标准 GPTQ 权重量化之外，逐层校准 MoE router 以缓解低比特量化引起的 expert-shift 问题。与 PMQ/BSP 等基于 expert 使用频率分配混合精度的策略不同，QESC 不依赖静态校准集确定 expert 重要性（避免跨任务过拟合），而是直接对齐量化前后 router 的输出，确保模型仍能为当前任务选对 expert。

术语是什么？
QESC 的量化流程：
1. **逐层处理**：从第 0 层到第 L-1 层顺序量化
2. **MHSA 量化**：每层 MHSA 量化为 4-bit（group-wise asymmetric, group_size=128, GPTQ）
3. **Router 校准**：使用 WikiText2 校准集（128 条 × 2048 tokens）前向传播，记录全精度 router 输出的 top-K expert 分布作为标签；获取通过量化 MHSA 和已量化 expert 的激活值作为输入；用 TopK-MSE Loss 更新 router 权重对齐输出
4. **Expert 量化**：将该层所有 expert 量化为 B-bit（GPTQ, group-wise, asymmetric）
5. **Router 保持精度**：Router 权重保持 FP16（仅占 <0.03% 参数）

位宽配置：MHSA 4-bit，expert 2/2.5/3-bit，最终平均位宽 2.06/2.54/3.03-bit。2.5-bit 设置下前半层 expert 分配 3-bit，后半层分配 2-bit。

从算法pipeline角度拆解术语：
```
=== QESC 逐层量化与校准 ===
For layer l in [0..L-1]:
    # Step 1: 量化该层 MHSA
    W_attn_q = GPTQ_quantize(W_attn, bits=4, groupsize=128)
    
    # Step 2: 获取校准输入
    for each calibration sequence:
        x_l = Forward(model_quantized[:l], input)   # 到当前层的 hidden state
    
    # Step 3: 对当前层的每个 MoE router 进行校准
    for each MoE_router at this layer:
        y_full = router_W @ x_l                     # 全精度参考
        x_hat_l = Forward_with_quantized_MHSA(x_l)  # 量化后激活
        
        # TopK-MSE Loss
        topK = arg_top_k(y_full, K_l)  # K_l 通过网格搜索确定
        loss = mean((y_full[i] - (router_W @ x_hat_l)[i])^2 for i in topK)
        router_W = optimizer_step(router_W, loss)
    
    # Step 4: 量化该层所有 expert
    for each expert e at layer l:
        W_expert_q[e] = GPTQ_quantize(W_expert[e], bits=B, groupsize=128)

输出: 量化后 MoE 模型（MHSA 4-bit, experts B-bit, router FP16）
```

术语一般如何实现？如何使用？
- 使用 GPTQ 作为底层量化框架，QESC 在 GPTQ 基础上增加 router 校准步骤
- 量化过程在单张 A100 40G GPU 上执行；router 校准开销仅 ~2% 总时间（如 Mixtral-8x7B: GPTQ 1.30h + Calibration 0.02h）
- 使用 BitBLAS 处理量化后权重的混合精度 BLAS 操作实现 GPU 加速
- 在 3.03-bit 下，Mixtral-8x7B 和 Deepseek-moe-16b-base 的准确率几乎无损（<0.5%），可实际部署
- QESC 理论上与其他减少量化误差的方法（如 QuaRot、SmoothQuant）正交兼容
- 相比 BSP/PMQ：QESC 不依赖静态 expert 频率分配位宽，跨任务泛化性显著更好（详见论文 Table 9 过拟合分析）

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
