## Linear Time Invariance (LTI) in State Space Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Time Invariance (LTI，线性时不变性) 是结构化 SSM（S4 及其变体）的一个核心属性：模型的动力学参数 (Δ, A, B, C) 在所有时间步保持恒定，不随输入变化。这一性质使 SSM 等价于：(1) 一个线性递归（recurrent view），(2) 一个全局卷积（convolutional view）。从递归角度看，LTI 意味着每个时间步的状态转移 A_bar 和输入投影 B_bar 完全相同——模型对所有 token 的响应方式一致。从卷积角度看，LTI 允许训练时使用 FFT 进行高效并行化。Mamba 论文的核心发现是 LTI 是 SSM 在离散模态（文本、DNA）上性能不足的根本原因：因为 LTI 模型无法根据输入内容进行选择性推理（如区分需要记忆的 token 和需要忽略的噪声 token）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LTI vs Non-LTI 对比：
```
# LTI SSM (S4): 所有时间步参数相同
A_bar: (D, N) ← 离散化(Δ, A)    # 常数，time-invariant
B_bar: (D, N) ← 离散化(Δ, A, B)  # 常数
h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_t  # 恒定转移
# → 可写成卷积: y = x * K  (使用FFT)

# Non-LTI / Selective SSM (S6/Mamba):
Δ_t: (B, L, D) ← f_Δ(x_t)         # 输入依赖!
B_t: (B, L, N) ← f_B(x_t)          # 输入依赖!
C_t: (B, L, N) ← f_C(x_t)          # 输入依赖!
A_bar_t: (B, L, D, N) ← 离散化(Δ_t, A)  # 时变!
h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊗ x_t   # 时变转移
# → 不能用卷积!  失去与 FFT 的等价性
```

Mamba 论文通过 Selective Copying 和 Induction Heads 两个合成任务验证了 LTI 的根本局限：
- Selective Copying：LTI 模型只能跟踪时间间隔（固定 spacing 的 Copying 任务可解），无法根据内容选择性记忆。S4 准确率仅 18.3%，而 S6（selective）达 97.0%。
- Induction Heads：LTI 模型无法根据上下文决定何时检索和输出。Mamba（含 S6）可完美泛化到 1M 长度（4000× 训练长度），所有 LTI 模型在 >2× 训练长度后崩溃。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LTI SSM 的实现通常是基于 FFT 的全局卷积（如 S4、DSS、S4D、Hyena、H3）。Mamba 之前的几乎所有 SSM 工作都默认使用 LTI 属性以利用卷积模式的高效训练。打破 LTI 需要解决计算效率问题（不能用卷积 → 必须用递归 + scan），这正是 Mamba 通过 hardware-aware 算法解决的问题。LTI 模型在连续信号模态（音频、视频）上仍有优势，因为连续信号具有平滑性、均匀采样的归纳偏置与 LTI 匹配。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---
