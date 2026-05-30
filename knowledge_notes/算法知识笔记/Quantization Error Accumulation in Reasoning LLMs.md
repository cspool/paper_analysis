## Quantization Error Accumulation in Reasoning LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量化误差累积（Quantization Error Accumulation）是推理 LLM 在长链式思维（Chain-of-Thought）生成中面临的独特挑战。自回归解码中，每个 token 的生成依赖于所有历史 token。每一步解码的量化层输出误差 ε_{l,t} = ||Q(W_l)·h_{l-1,t} - W_l·h_{l-1,t}|| 会被传播到后续层和后续 token。在短生成（如常识 QA，<100 tokens）中误差可忽略；但在推理任务（MMLU-Pro、AIME 等，CoT 可达数千至数万 tokens）中，误差在每一步叠加，导致正确推理路径偏离。ParoQuant 量化 Qwen3-4B 在 MMLU-Pro 上的数据：FP16=71.0, AWQ=68.2（降 2.8%），ParoQuant=70.1（仅降 0.9%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
误差累积的数学模型：
```
# 总误差近似为所有步所有层的局部误差之和
# E_total ≈ Σ_{t=1}^{T} Σ_{l=1}^{L} (Π_{k=l+1}^{L} ||J_k||) · ε_{l,t}
# 其中 J_k 是第 k 层 FP16 的 Jacobian，ε_{l,t} 是单步单层量化误差
# 对于有残差连接的 Transformer, ||J_k|| ≈ 1 + δ (δ 很小)
# 因此 E_total ≈ Σ_{t=1}^{T} Σ_{l=1}^{L} ε_{l,t} = O(T·L·avg(ε))

# ParoQuant 的逐层优化策略:
for each layer l:
    # 使用已量化前层的输出 X' 作为校准输入
    X' = quantized_previous_layers_output(X)
    # 目标: 最小化当前层量化后输出与原始 FP16 输出的差异
    loss = ||Q(l)(X') - l(X)||
    # 后续层看到的是量化输出, 可以补偿前层误差
```
关键：ParoQuant 在逐层优化时使用 X'（已量化前层输出）而非 X（原始 FP16 输入），使每层的变换参数在优化时感知前层的量化误差，后续层学会补偿这些误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解误差累积的方法：(1) 使用长文本推理 benchmark（MMLU-Pro、AIME）评估量化质量，而非仅用 PPL（PPL 无法反映长生成中的误差累积效应）；(2) 逐层优化时使用已量化前层输出，使变换参数对累积误差有感知和补偿能力；(3) 多样化校准集（混合多个数据集）防止变换参数对单一数据分布的过拟合，提升对多种推理路径误差分布的鲁棒性。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---
