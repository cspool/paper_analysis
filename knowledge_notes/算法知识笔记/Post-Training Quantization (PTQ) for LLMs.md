## Post-Training Quantization (PTQ) for LLMs

术语是什么？
Post-Training Quantization (PTQ) 是一类无需重新训练或微调即可将预训练模型量化的技术总称。与 QAT (Quantization-Aware Training，需数百 GPU 小时训练) 相比，PTQ 仅需少量校准数据（或无数据如 HQQ）做前向推理确定量化参数（scale、zero-point、bit-width 分配），计算开销极小。常见 PTQ 方法：(1) Round-to-nearest (RTN)：直接四舍五入，简单但极低位宽精度低；(2) GPTQ：Hessian 引导逐列量化 + 误差补偿；(3) AWQ：激活感知权重等效变换（per-channel scaling）；(4) Omniquant：可学习权重裁剪（LWC）+ 校准优化；(5) HQQ：半二次优化无需校准数据。LLM PTQ 的核心挑战是 activation outlier（某些 channel 激活值远大于其他），需 per-channel/group quantization、smooth quantization 或 Hadamard rotation 缓解。

从算法pipeline角度拆解术语：
```
// 阶段1: 校准 (确定量化参数)
for batch in calibration_data:
    activations = model.forward(batch)
    layer.scale = (max(W)-min(W)) / (2^B-1)
    layer.zero_point = round(-min(W) / layer.scale)

// 阶段2: 量化
W_q = clamp(round(W/scale + zero_point), 0, 2^B-1)

// 阶段3: 推理 (融合反量化)
Y = X @ (W_q * scale + zero_point)
```

术语一般如何实现？如何使用？
- 适用场景：(a) 资源受限无法做 QAT；(b) 原型验证不同位宽效果；(c) 一次性批量部署
- 局限：(a) ≤4 bit 精度损失显著（尤其推理/长上下文任务）；(b) 依赖校准数据分布；(c) 无法像 QAT 那样通过训练适应特定任务
- MC-MoE：使用 GPTQ PTQ + expert-wise 混合精度策略，验证 MoE 场景下 PTQ 可接近 QAT 效果

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
