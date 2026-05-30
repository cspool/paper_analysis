## MoEQuant

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoEQuant 是 ICML 2025 发表的针对 Mixture-of-Experts (MoE) LLM 的后训练量化框架，由 Houmo AI 和东南大学联合提出。框架包含两个核心组件：EBSS（Expert-Balanced Self-Sampling）和 AGQ（Affinity-Guided Quantization），二者均为插件式设计，可与 GPTQ、AWQ 等现有 PTQ 方法无缝集成。

MoEQuant 解决的核心问题是：现有 LLM PTQ 方法（GPTQ、AWQ）在 MoE 模型上性能严重下降，原因是忽略了 MoE 架构的两个关键特性——(1) inter-expert imbalance：校准集中不同 expert 负载极不均衡；(2) intra-expert imbalance：不同 token 对同一 expert 的贡献权重不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MoEQuant Complete Pipeline
# Phase 1: EBSS — Generate balanced calibration set
calib_data = EBSS(
    model=M,           # MoE LLM (e.g., Qwen-MoE-14B)
    beam_width=4,      # w
    seq_length=512,    # n
    temperature=1.2    # τ
)
# → D*: expert-balanced calibration sequences

# Phase 2: AGQ — Affinity-guided quantization
for layer in M.moe_layers:
    X = forward_and_collect_activations(D*, layer)
    
    for expert in layer.experts:
        # Get tokens routed to this expert and their gating weights
        X_e, c_e = get_expert_inputs(X, expert)
        
        # AGQ-modified Hessian (for GPTQ)
        H = (X_e * sqrt(c_e)).T @ (X_e * sqrt(c_e))
        
        # Standard GPTQ with AGQ Hessian
        for weight_matrix in [W_gate, W_up, W_down]:
            GPTQ_columnwise_quantize(weight_matrix, H, bits=4)
            # Uses H for error compensation
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoEQuant 的实现基于 GPTQ 和 AWQ 官方仓库修改。关键配置：4-bit/3-bit per-channel 对称均匀量化，EBSS 参数 w=4 / τ=1.2，AGQ 与 GPTQ 集成时使用改进的 Hessian H = (X⊙c)X^T。硬件平台为 NVIDIA A6000 GPU。实验覆盖 Qwen-MoE-14B、DeepSeek-MoE-16B、Mixtral-8x7B 及其 instruction-tuned 变体。性能：4-bit MoEQuant++（基于 GPTQ）在三个模型上的平均分分别比 GPTQ 提升 0.59/1.00/2.16 分，3.2x 内存节省，1.2x 推理加速。代码发布于 https://anonymous.4open.science/r/MoEQuant-DDFD/README.md。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
