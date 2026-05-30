## Weight-Activation Quantization (W4A4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-Activation Quantization（权重-激活联合量化，表示为 W4A4、W8A8 等）是在 LLM 量化中同时对模型权重和各层激活值（中间特征图）执行低比特量化的策略。与 Weight-only Quantization（仅量化权重，保持激活 FP16/BF16）相比，W4A4 的优势：(1) 计算可在低位宽整数域完成（INT4×INT4 GEMM），理论上比 W4A16（FP16 GEMM）计算量更小；(2) 激活的存储和带宽需求也减少 4×（FP16→INT4）。但挑战更大：(a) 激活分布依赖于输入数据和上下文，比权重更难预测，量化误差更难控制；(b) LLM 激活中存在大量 outlier channels——某些通道的激活幅度远超其他，直接量化导致严重信息丢失；(c) 激活量化参数需实时计算或离线统计。AffineQuant 对 W4A4 的改进：在 LayerNorm 后使用对角仿射矩阵 A（仅更新对角线的 A），将 A 合并入 LN weight/bias，对激活进行等价变换以降低量化难度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AffineQuant 中 W4A4 的完整前向传播流程：
```
for each transformer_block:
    # === Attention 部分 ===
    x_norm = LayerNorm(x)
    x_norm = x_norm * diag(A_attn)   # 对角 A 等价变换（可合并入 LN）
    x_norm_q = quantize(x_norm, 4bit) # 激活 INT4 量化
    
    # QKV 投影: INT4 weight × INT4 activation
    q = Q_linear(x_norm_q)  # W_qkv 已合并 A 的权重部分: Q(AW_qkv)
    k = K_linear(x_norm_q)
    v = V_linear(x_norm_q)
    
    attn_out = attention(q, k, v)   # FP16 attention（不量化 softmax）
    attn_q = quantize(attn_out, 4bit)
    o = O_linear(attn_q)            # Out proj: Q(AW_out)
    
    # === MLP 部分 ===
    x_res = x + o
    x_norm2 = LayerNorm(x_res) * diag(A_mlp)  # 对角 A（可合并入 LN）
    x_norm2_q = quantize(x_norm2, 4bit)
    
    gate = GELU(FC1(x_norm2_q))     # fc1: Q(AW_fc1), INT4×INT4
    # ⚠ fc1-fc2 之间不做仿射变换（GELU 使 XA⁻¹ 等价变换失效）
    y = FC2(gate)                    # fc2: Q(AW_fc2), FP16（gate 不量化）
    output = x_res + y
```
W4A4 下的关键结果：LLaMA2-7B C4 PPL 15.76（OmniQuant 18.02），WikiText2 PPL 12.69（OmniQuant 14.26）；LLaMA-30B zero-shot avg 58.61%（OmniQuant 56.63%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W4A4 在 LLM 推理中的实际部署：(1) 离线 PTQ 阶段用 AffineQuant/SmoothQuant/OmniQuant 等方法优化权重和激活的量化参数；(2) 推理框架需实现 INT4×INT4 GEMM kernel——TensorRT-LLM 和 MLC-LLM 支持 W4A4 推理；(3) 激活量化需实时计算 scale/zero_point——通常从校准数据离线统计 running min/max 或使用 per-token 动态量化。当前 W4A4 的局限性：(a) 现代 GPU 的 Tensor Core 对 INT4 的支持有限（INT8 更成熟），实际加速比低于理论值；(b) 激活量化在 decode 阶段引入额外延迟（per-token quantization overhead）；(c) 在边缘设备（移动 GPU、NPU）上，INT4 MAC 操作的能耗和延迟优势更显著。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models

- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
---
