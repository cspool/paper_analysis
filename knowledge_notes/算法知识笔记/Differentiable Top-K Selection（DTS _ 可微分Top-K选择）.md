## Differentiable Top-K Selection（DTS / 可微分Top-K选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Differentiable Top-K Selection (DTS) 是 VisionSelector 提出的可微分 Top-K token 选择机制。与传统 Gumbel-Softmax STE 通过随机采样实现可微分选择不同，DTS 通过 sigmoid 连续松弛 + 隐函数微分（implicit differentiation）实现确定性、单调的端到端梯度透传。Forward：给定重要性得分向量 s ∈ R^{B×N} 和保留数量 k，二分搜索阈值 t 使 Σ σ(s_i + t) ≈ k，得到 soft mask M = σ(s + t) ∈ (0,1)^N。由于 sigmoid 的严格单调性：s_i > s_j ⇔ M_i > M_j，保证高得分 token 获得更高 soft 权重，避免 Gumbel-Softmax 随机扰动导致的不稳定性和非单调性。Backward：在约束 Σ σ(s_i + t) = k 下隐式求导，得闭合形式梯度 ∂M/∂s = diag(v) − vv^T/Σv_i（v_i = M_i(1−M_i)），进一步化简为 ∂L/∂s = v⊙g − (v^T g/Σv_i)·v（g 为上游梯度）。推理时直接使用标准 Top-K 硬选择，无二分搜索或无额外开销。与 Gumbel-Softmax 的关键区别：(a) DTS 不需要随机噪声，forward 和 backward 均为确定性，(b) DTS 保持 scores→mask 的单调性，(c) DTS 通过 curriculum annealing 而非 temperature annealing 桥接训练-推理 gap。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === DTS Forward: 二分搜索 + sigmoid 连续松弛 ===
def DiffTopK_forward(s, k):
    # s ∈ R^{N} per batch, k = N * budget
    lower = -max(s) - 10   # sigmoid(-∞)=0 时保证全部 < k
    upper = -min(s) + 10   # sigmoid(+∞)=1 时保证全部 > k
    for _ in range(64):    # 64 次二分迭代达到充分精度
        mid = (lower + upper) / 2
        count = sum(sigmoid(s + mid))
        mask = (count < k)
        lower[mask] = mid[mask]     # 增大 mid → sigmoid 值增大
        upper[~mask] = mid[~mask]   # 减小 mid → sigmoid 值减小
    t = (lower + upper) / 2
    M_soft = sigmoid(s + t)  # ∈ [0,1]^N, 近似: sum(M_soft) ≈ k
    return M_soft

# === DTS Backward: 隐函数微分(闭式解) ===
def DiffTopK_backward(grad, s, t):
    M = sigmoid(s + t)
    v = M * (1 - M)       # σ'(s+t) = σ(s+t)(1-σ(s+t))
    v_sum = sum(v)
    uv = grad * v          # g ⊙ v
    uv_sum = sum(uv)
    grad_s = uv - (uv_sum / v_sum) * v  # 见论文公式(8)
    return grad_s          # ∂L/∂s 用于更新 LIS 参数

# === 对比: Gumbel-Softmax STE ===
# gumbel_softmax: z = softmax((logits + gumbel_noise) / τ)
# STE: backward = ∂L/∂z_hard ≈ ∂L/∂z_soft  (近似)
# DTS: backward = closed-form exact gradient  (精确)
```

Annotations: sigmoid: σ(x) = 1/(1+e^{-x})。二分搜索 64 次 = O(B×N×log range) 每 batch，相对 LLM forward 开销极小。显式梯度公式(8)源于论文的数学推导：对等式约束 Σ σ(s_i+t) = k 求全微分可得 ∂t/∂s_j = -v_j/Σv_i，代入 dM = σ'(s+t)·(ds + dt) 展开得到 ∂M/∂s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中实现 DTS：forward 通过 torch.sigmoid(s + t) 和二分搜索（无 autograd），backward 通过自定义 autograd.Function 注册上述闭式梯度。训练时 DTS 输出 soft mask M_soft ∈ [0,1]^N 与 token features 逐元素乘 V_pruned = M_soft ⊙ V；推理时完全跳过 DTS，直接 TopK(s, k) 得硬 mask。训练仅需在 LIS 模块上 (~12.85M 参数，Qwen2.5-VL-7B)，DTS 本身无可训练参数。二分搜索对任意 batch size 和 k 值为确定性操作，保证 sum(M_soft) 在给定精度内等于 k。VisionSelector 开源实现：https://github.com/JulietChoo/VisionSelector。与 Gumbel-Softmax 的性能对比：DTS 训练更稳定（无随机噪声从 forward 引入方差），梯度更精确（隐函数微分精确梯度 vs STE 近似），收敛更快（论文约 40 分钟 / 8 A800 训练完成）。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs
