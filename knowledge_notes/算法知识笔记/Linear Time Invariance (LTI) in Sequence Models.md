## Linear Time Invariance (LTI) in Sequence Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Time Invariance (LTI，线性时不变性) 是序列模型动态参数在所有时间步上保持不变的属性。在 SSM 语境：Ā, B̄, C 对所有 t 相同 → 递归 h_t = Ā h_{t-1} + B̄ x_t 等价于全局卷积 y = x ∗ K̄，可用 FFT O(L log L) 训练。所有先前的结构化 SSM（S4, H3, Hyena, RWKV, RetNet）均为 LTI。优点：计算效率高、可并行训练。核心缺陷：无法做内容感知推理——动态对所有 token 相同，不能根据 token 内容决定选择/过滤哪些信息。在 Selective Copying（token 间距随机）和 Induction Heads（上下文关联召回）任务上暴露为致命弱点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LTI SSM (S4): Ā, B̄, C 对所有 t 相同
# 卷积模式 (训练): 所有 timestep 共享相同卷积核
K̄ = [CB̄, CĀB̄, CĀ²B̄, ..., CĀ^{L-1}B̄]  # 固定卷积核
y = x ∗ K̄                                    # FFT 加速, O(L log L)

# 循环模式 (推理): Ā 不依赖 x_t
h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t                 # 每步相同动态!
```
LTI 无法解决 Selective Copying：不同位置需记忆/忽略的 token 间距随机 → 但卷积核固定长度 → 无法适配 → 只能依赖 time-awareness 而非 content-awareness。Mamba 的选择机制打破 LTI：Ā_t = f(x_t) → 获得内容感知 → 但损失卷积可用性 → 需要硬件感知实现补偿。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LTI SSM 的实现族：S4 (DPLR + FFT 卷积), S4D (纯对角 + FFT), H3 (S4 嵌入 gated architecture), Hyena (MLP-parameterized 全局卷积替换 S4)。在 GPU 上通过 PyTorch FFT primitives 实现。Mamba 证明了 LTI 是内容感知的主要障碍，打破它并补偿效率损失可以匹配 Transformer 性能同时保持线性复杂度。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
