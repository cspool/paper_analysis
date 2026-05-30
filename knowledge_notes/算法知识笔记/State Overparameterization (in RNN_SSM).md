## State Overparameterization (in RNN/SSM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
State Overparameterization 是清华团队在 Stuffed Mamba 论文（2024）中提出的概念，指 RNN/SSM 模型的递归状态大小相对于训练上下文长度过大，导致模型无需学习有效遗忘机制即可最小化语言建模损失的现象。Mamba-2 的状态大小 N_S = HPN = 256d（N=128, P=64, H=2d/P），约等于同等 Transformer 的 KV cache 大小。在 8K 训练长度下，状态容量远大于 8K token 所包含的信息量，模型学会将所有 token 信息保留在状态中（α_t 始终接近 1），这在训练长度内表现良好，但超过训练长度后状态被"塞满"（stuffed），不同 token 的信息相互干扰，导致记忆召回失败。实证：(1) 遗忘阈值 T_forget = 5.172·N_S - 4.469 (R² > 0.999)；(2) 更多训练数据反而加剧问题——Passkey Retrieval 精度随数据量增加而下降（Figure 8）；(3) 大模型（780M, N_S=19.3M）比小模型更差，因其状态更大。本质是一种过拟合：状态分布仅在短上下文下变化不足，无法泛化到长上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
State Overparameterization 的诊断流程：
```
# 检测 State Overparameterization
# 输入: Mamba-2 模型, 训练长度 T_train
for each head in each layer:
    # 1. 计算首 token 记忆保留强度
    for t in 1..T_train:
        α_{1:t} = ∏_{j=1}^{t} α_j  # 累积衰减因子
    if α_{1:T_train} > 0.99:  # 几乎无衰减 → 过参数化

    # 2. 检测方差爆炸
    for t in 1..2*T_train:
        h_t = update(h_{t-1}, input_t)  # 用"newlines" prompt
        var_t = variance(h_t, dim=channel)
    if max(var_{T_train:}) > 10 * max(var_{:T_train}):
        # 超过训练长度后方差异常增大 → 状态崩溃
        outlier_channels = top_k(var_excess, k=5%)  # ~5% channel 驱动

# 3. 验证遗忘阈值
for different N_S (state sizes):
    train with increasing T_train
    find T_forget where LM loss < 2× max_loss_within_Ttrain at 1M tokens
    # 得到: T_forget = 5.172 * N_S - 4.469
```
诊断依据：(a) 某些 head 的首 token α_{1:t} 始终 > 0.997——累积 8K 步后几乎不衰减；(b) 状态方差在 T_train 后由少数 outlier channel 驱动爆炸；(c) 遗忘只发生在 T_train > T_forget 时。核心启示：RNN 的状态大小和训练长度必须匹配——训练长度应随状态大小线性增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用方式：(1) 训练前估算最小训练长度 T_train > 5.172·N_S，对于 370M Mamba-2 (N_S=12.9M)，T_train > 66.7K；(2) 检测首 token 保留强度 α_{1:t} 作为过参数化的早期指标；(3) 使用 Passkey Retrieval（而非 validation loss）作为验证指标——它对过参数化的敏感度远高于 loss。该概念适用于所有门控线性注意力 RNN（GLA、RWKV、RetNet），因为它们共享类似的加权和状态形式。论文中 370M Mamba-2 在 256K 训练长度下达到近乎完美的 Passkey Retrieval，验证了消除过参数化后的长度泛化能力。Albert Gu（Mamba 作者）确认了这一发现："Feed your Mamba until it's full, and it will perform at its best!"

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
