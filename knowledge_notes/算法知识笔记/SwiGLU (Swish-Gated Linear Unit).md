## SwiGLU (Swish-Gated Linear Unit)

术语解释
一种用于 Transformer FFN 层的门控激活函数，由 Swish（SiLU）激活和门控线性单元组合而成。被 Llama、MobiLlama 等现代 LLM/SLM 广泛采用，替代传统 ReLU/GELU FFN。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwiGLU FFN 计算：
```
# 输入: x ∈ R^{d_model}
gate = x @ W_gate.T           # gate projection
up   = x @ W_up.T             # up projection
gate_act = SiLU(gate)         # SiLU(x) = x * σ(x)
gated = gate_act ⊙ up         # 逐元素门控乘法
output = gated @ W_down.T     # down projection
```

对比：ReLU FFN 用 2 个权重矩阵 + 1 次非线性；SwiGLU 用 3 个权重矩阵（gate/up/down）+ 门控机制提供更强的非线性表达能力。

从算法pipeline角度拆解术语，给出具体例子。
在 MobiLlama 共享 FFN 中：d_model=2048, d_intermediate=5632。所有 22 层对相同的 W_gate/W_up/W_down 执行 SwiGLU 计算。归一化后的 hidden state 经 RMSNorm → 线性投影 → SiLU → element-wise multiply → down projection → 残差连接。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中 `LlamaMLP` 类实现：`gate_proj`, `up_proj`, `down_proj` 三个 `nn.Linear` + `nn.SiLU`。推理框架中常 fused 为单一 kernel（gate projection + SiLU + up projection + multiply + down projection）减少内存带宽。Shared FFN 场景下，fused SwiGLU kernel 可进一步获益：所有层对相同权重的访问被 L2 cache 高效服务。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices
- Scaling Law for Quantization-Aware Training

在 Scaling Law for QAT 中，SwiGLU 被识别为 W4A4 QAT 中激活量化误差的关键瓶颈来源。FC2 Proj（即 down_proj）的输入来自 SwiGLU 输出：gate_act = SiLU(x @ W_gate) ⊙ (x @ W_up)。gating + SiLU + element-wise multiply 的复合非线性变换产生系统性 outlier，导致 FC2 Proj 输入 kurtosis=89（即使 QAT 正则化后），远高于 QKV Proj、O Proj、FC1 Proj 等层（均 <10）。论文实验证明：对 FC2 Proj 输入使用 8-bit 混合精度可将 W4A4 量化误差降 20.5%（G=32）至 42.9%（G=256），激活误差对 G 的敏感度 γ_G 从 0.9812 降至 0.4471。
