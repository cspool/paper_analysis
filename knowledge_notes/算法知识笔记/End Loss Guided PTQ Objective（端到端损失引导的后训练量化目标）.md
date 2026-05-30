## End Loss Guided PTQ Objective（端到端损失引导的后训练量化目标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
End Loss Guided PTQ Objective 是 GuidedQuant 提出的量化目标函数：`||(∂ℓ/∂Z) ⊙ (XW - XŴ)||²_F`，即用 end loss 对每层输出的梯度 ∂ℓ/∂Z 作为权重，对 layer-wise output error 进行逐元素加权。与标准 layer-wise output-based 目标 `||XW - XŴ||²` 将每个 output feature 的量化误差同等对待不同，end loss guided 目标根据每个 feature 对最终 loss 的敏感度分配不同的重要性权重。该目标基于 end loss 的一阶 Taylor 展开：`ℓ(Ẑ) - ℓ(Z) ≈ ∂ℓ/∂Z · (Ẑ - Z)`。等价于 block-diagonal Fisher 二次近似（见 Block-Diagonal Fisher Approximation 术语），但通过 backprop 梯度直接计算避免了显式构建 Fisher 矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GuidedQuant 的 end loss guided 量化流程（Algorithm 1）：
```
# 输入：校准数据，预训练模型，分组数 g
# 超参数：g=4 (7B/13B), g=2 (70B) for weight-only; g=1 for W+A

# Step 1: 划分 output channels 为 g 组
J_k = {d_out*(k-1)/g + 1, ..., d_out*k/g}  for k in [1, g]

# Step 2: 单次 backward pass，计算并平均梯度的平方
for each layer l, group k:
    s_k = (1/|J_k|) * Σ_{j∈J_k} (∂ℓ/∂z_j)²   # n 维向量

# Step 3: 构建 guided Hessian
    H̄_k = Xᵀ @ Diag(s_k) @ X                    # d_in × d_in

# Step 4: 调用 base quantizer Q 量化该组
    Ŵ[:, J_k] = Q(H̄_k, W[:, J_k])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GuidedQuant 作为 plugin 可直接集成到任何 layer-wise output-based PTQ 方法中：(1) LNQ + GQuant：将 H=XᵀX 替换为 H̄_k，调用 LNQ（Algorithm 2）量化每组；(2) QTIP + GQuant：将 BlockLDLQ 的 Hessian 替换为 H̄_k；(3) SpinQuant + GQuant：将 GPTQ weight quantizer 的 Hessian 替换为 H̄_k。梯度计算只需一次 backward pass（Llama-2-7B: 0.3h on 1×A100），Hessian 可缓存复用多次量化。End loss guided 目标在极端低比特（2-bit）下收益最大：Llama-2-7B non-uniform scalar 2.01-bit Wiki2 perplexity 从 23.31 (LNQ) 降至 8.83 (LNQ+GQuant)。代码开源：github.com/snu-mllab/GuidedQuant。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---
