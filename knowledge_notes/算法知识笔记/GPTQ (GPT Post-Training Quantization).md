## GPTQ (GPT Post-Training Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPTQ (Frantar et al., 2022) 是一种基于 Hessian 误差补偿的后训练权重量化方法，专门针对 GPT 系列 LLM 设计。核心原理源自 Optimal Brain Quantization (OBQ)：逐列量化权重矩阵，量化每个列后计算输出误差，使用 Hessian 逆矩阵将误差按比例补偿到剩余的未量化列上，从而保证最终输出的近似精度。

GPTQ 的关键技术：（1）H = 2X^T X 作为近似 Hessian 矩阵；（2）对 Hessian 做 Cholesky 分解提高数值稳定性；（3）在量化时对权重列按 Hessian 对角线大小排序（贪心顺序）；（4）对较大矩阵分 block 量化以控制内存；（5）所有列量化完后的总误差 = Σ_j (H^{-1})_{jj} · ε_j^2，被 H^{-1} 缩小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GPTQ (simplified for one linear layer)
# W ∈ R^{o×c}, X ∈ R^{b×c}
H = 2 * X.T @ X  # [c, c], Hessian approximation
H_inv = inverse(H + lambda*eye(c))  # damped inverse for stability

W_q = zeros_like(W)
perm = argsort(diag(H_inv))  # order columns by sensitivity
W = W[:, perm]
H_inv = H_inv[perm][:, perm]

for j in range(c):
    # Quantize column j
    for i in range(o):
        s_i = max(abs(W[i,j])) / q_max
        W_q[i,j] = clamp(round(W[i,j] / s_i), q_min, q_max)
    
    # Error in weight space
    err = W_q[:,j] - W[:,j]  # [o]
    
    # Compensate remaining columns (k > j)
    for k in range(j+1, c):
        W[:,k] -= (H_inv[j,k] / H_inv[j,j]) * err

# Reorder back
inv_perm = argsort(perm)
W_q = W_q[:, inv_perm]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ 的开源实现在 https://github.com/IST-DASLab/gptq，支持 2/3/4/8-bit 量化，默认使用 128 个 2048-token 序列作为校准集，序列来自 C4 数据集。推理时配合 HuggingFace Transformers 或 vLLM 等框架的量化 kernel。在 MoEQuant 论文中，GPTQ 作为核心 baseline 使用，配合 QuaRot 的 Hadamard 变换预处理（消除权重 outlier）但不使用在线变换。GPTQ 在 MoE 模型上的问题：Hessian 未考虑 token-expert 亲和力（AGQ 的改进方向），且校准集未针对 MoE 架构做专家均衡（EBSS 的改进方向）。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
