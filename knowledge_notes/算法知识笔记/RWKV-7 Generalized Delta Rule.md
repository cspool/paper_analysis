## RWKV-7 Generalized Delta Rule

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gated Delta Rule 是一种统一的线性RNN状态更新规则，将 Mamba2 的 gating 机制（α_t 控制全局衰减）与 DeltaNet 的 delta rule（β_t 控制精确 key-value 更新）结合为一个公式：S_t = S_{t-1} (α_t (I - β_t k_t k_t^T)) + β_t v_t k_t^T。其中 S_t ∈ R^{d×d} 是矩阵值隐藏状态，α_t ∈ (0,1) 是数据依赖的 forget gate，β_t ∈ (0,1) 是数据依赖的 writing strength，k_t, v_t ∈ R^d 是当前 token 的 key 和 value 投影。该规则统一了两种互补的记忆操作：当 α_t→0 时快速清除所有记忆（context switch 场景），当 α_t→1 时退化为纯 delta rule（精确 memorization 场景）。从在线学习视角（Liu et al., 2024），Gated Delta Rule 优化目标为 min_{S_t} ||S_t - α_t S_{t-1}||_F^2 - 2⟨S_t k_t, β_t(v_t - α_t S_{t-1} k_t)⟩，同时具备 adaptive weight decay（α_t 项）和精确回归 loss（β 项）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
推理时 per-token 更新（O(d²) per head per token）：
```
q_t = L2Norm(SiLU(ShortConv(W_q x_t)))
k_t = L2Norm(SiLU(ShortConv(W_k x_t)))
v_t = SiLU(ShortConv(W_v x_t))
α_t = sigmoid(W_α x_t + b_α)
β_t = sigmoid(W_β x_t)
S_t = α_t · S_{t-1} · (I - β_t k_t k_t^T) + β_t · v_t k_t^T
o_t = S_t q_t
output_t = W_o (RMSNorm(o_t) ⊙ SiLU(W_g x_t))
```

与 baseline 的精确区别：
- Mamba2: S_t = α_t S_{t-1} + v_t k_t^T（仅有全局衰减）
- DeltaNet: S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T（仅有精确更新）
- Gated DeltaNet: S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T（两者兼有）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVlabs/GatedDeltaNet。训练时使用基于 WY 表示的 chunkwise 并行算法，推理时退化为 RNN 式 O(d²) per-token 递归更新，无需 KV cache。α_t 使用 Mamba2 的参数化方式（sigmoid 投影 + bias），β_t 由 sigmoid 投影生成。适用于需要同时具备长序列记忆保持和自适应遗忘的线性 RNN 语言模型。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
