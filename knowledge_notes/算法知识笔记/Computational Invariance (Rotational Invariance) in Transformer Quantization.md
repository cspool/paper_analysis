## Computational Invariance (Rotational Invariance) in Transformer Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Computational Invariance（计算不变性/旋转不变性）是 LLM 量化中通过插入正交旋转矩阵来改变激活分布而不改变模型数学输出的核心技术。原理：对线性层 Y = XW^T，插入正交矩阵 R（RR^T=I）后 Y = (XR)(R^T W^T) = XW^T，输出不变。在 Transformer block 中可插入 R1-R4 四个旋转矩阵：(1) R1 右乘 W_q/W_k/W_v/W_up/W_gate，R1^T 左乘 W_out/W_down/W_embedding，R1 右乘 W_lm_head；(2) R2 插入 W_v 和 W_o 之间的多头注意力路径（per-head）；(3) R3 为在线 Hadamard 变换用于 KV cache 量化（因 RoPE 无法融合）；(4) R4 为在线 Hadamard 变换用于 FFN down-projection（因 gating 机制）。R1/R2 可离线融合入权重实现零推理开销，R3/R4 使用快速 Hadamard kernel 在线计算。核心性质：(a) 旋转保持 L2 范数不变；(b) RMSNorm 与旋转可交换（RMSNorm(XR)=RMSNorm(X)R）。

从算法pipeline角度拆解术语，给出具体例子。
以下为插入旋转矩阵后的 Transformer 计算流程（离线融合后）：
```
# 离线：W_q'=W_q@R1, W_k'=W_k@R1, W_v'=W_v@R1@R2
#       W_o'=R2^T@W_o, W_up'=W_up@R1, W_gate'=W_gate@R1
#       W_down'=R1^T@W_down, W_embed'=R1^T@W_embed, W_lm_head'=W_lm_head@R1
# 在线（推理时）：
Q, K, V = X@W_q'^T, X@W_k'^T, X@W_v'^T
K_rope, Q_rope = RoPE(K), RoPE(Q)
scores = Q_rope @ (Hadamard(K_rope))^T   # R3 在线 Hadamard
attn_out = softmax(scores) @ Hadamard(V)  # R3 双向抵消 = Hadamard(attn_out)
O = attn_out @ W_o'^T
gate, up = X@W_gate'^T, X@W_up'^T
down_in = Hadamard(SiLU(gate)*up)         # R4 在线 Hadamard
F = down_in @ W_down'^T                   # W_down' 已含 R1^T
X_out = X_in + O + F
```

术语一般如何实现？如何使用？
实现基于 QuaRot/SpinQuant 代码框架：(1) 加载 HuggingFace 模型后自动识别 Q/K/V/O/Up/Gate/Down 权重位置；(2) 用 `torch.linalg.qr()` 或 Cayley SGD 初始化正交旋转矩阵；(3) 按 Computational Invariance 规则融合入相邻权重（矩阵乘法）；(4) 在线 R3/R4 使用 QuIP# 的快速 Hadamard CUDA kernel。DartQuant 用 Whip Loss + QR-Orth 替代端到端微调加速旋转矩阵获取。

ResQ 扩展了 Computational Invariance 的使用——它在 PCA 排序后的高低精度子空间内分别应用随机旋转。关键区别：ResQ 的 U = PR（P 为 PCA 特征向量矩阵，R 为随机正交旋转），P 负责分配通道到高低精度组，R 在每个组内独立抑制 outliers。投影矩阵 U_A 融合到 block 边界权重（零运行时开销），U_B 融合到 attention value 路径（o_proj 左乘 U_B^T），U_C 因 RoPE 需在线计算但量化为 8-bit（key/query 对称投影保持 attention dot product 不变：q_proj K_proj^T = (q U_C)(U_C^T K^T) = q K^T），U_D 用 Hadamard 矩阵实现快速在线投影。ResQ 的 invariance 设计同时支持 4/8-bit 混合精度（而 QuaRot 仅支持统一 4-bit）。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

---
