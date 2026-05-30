## Mamba (Selective State Space Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba是由Gu和Dao(2023)提出的基于选择性状态空间模型（Selective State Space Model, SSM）的序列建模架构，作为Transformer中self-attention层的替代方案。其核心创新在于将传统SSM的线性时不变（LTI）系统改造为输入依赖的选择性机制：SSM的参数（特别是离散化步长Δ和输入投影B、C）不再是固定的，而是由当前输入token通过可学习的线性投影动态生成。这使得Mamba能够像attention一样"选择性"地关注或忽略输入中的特定token，同时保持SSM的线性计算复杂度（O(n)而非Transformer的O(n²)）和常量推理内存（无需KV cache，仅需维护一个固定大小的隐状态h_t ∈ R^{D×N}，其中D=hidden_dim，N=state_dim）。Mamba使用硬件感知的并行扫描算法在GPU上高效实现训练，推理时退化为RNN式的逐token递归更新。论文中使用的8B Mamba模型配置：56层，hidden dim 4096，state dim 128，GELU激活，RMSNorm归一化，无位置编码，untied embeddings，无bias，无Dropout。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba层的前向传播计算流程（单层）：
```
Input: x ∈ R^{B×L×D}  (batch, seq_len, hidden_dim)
Output: y ∈ R^{B×L×D}

// Step 1: Input projection (expand to 2*expand*D for Δ/B/C and gate)
x_proj = Linear_proj(RMSNorm(x))  // shape: (B, L, 2*expand*D + D*state_dim)
Δ, B, C, z = split(x_proj)

// Step 2: Discretization (input-dependent, this is the "selective" part)
Δ = softplus(Linear_Δ(x) + bias_Δ)  // shape: (B, L, D), per-channel step size
A_bar = exp(Δ ⊙ A)  // A ∈ R^{D×N} is a learned diagonal matrix
B_bar = Δ ⊙ B  // B ∈ R^{B×L×N}, ⊙ is element-wise broadcast

// Step 3: Selective SSM scan (can be computed via parallel scan)
h_0 = 0  // initial state ∈ R^{D×N}
for t in 0..L-1:
    h_t = A_bar[t] * h_{t-1} + B_bar[t] * x[t]'  // h_t ∈ R^{D×N}
    y[t] = C[t] @ h_t  // C ∈ R^{B×L×N}

// Step 4: Gating with SiLU
y = y * SiLU(Linear_z(z))  // element-wise gating

// Step 5: Output projection + residual
output = x + Linear_out(y)
```
训练时：Mamba的SSM递归可通过parallel scan算法并行化（associative scan），将O(L)的串行递归转化为O(log L)的并行操作。推理时：prefill阶段使用parallel scan初始化SSM状态，decode阶段仅需O(1)计算量更新h_t并生成输出token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba开源实现：https://github.com/state-spaces/mamba（官方CUDA kernel）。论文中使用NVIDIA Megatron-LM中的Mamba实现（https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba），支持tensor parallelism（每层需2次all-reduce，比Transformer的1次多）和序列并行，但不支持pipeline parallelism。Mamba训练速度约为同参数Mamba-2的3倍慢（因state dim 128导致scan开销大）。适用于：需要线性时间复杂度处理长序列的语言建模任务、对推理时内存（无需KV cache）有严格限制的场景。

在Attamba中的用法：Attamba将Mamba SSM用作Transformer attention内部的Key和Value投影替代。传统Transformer用线性投影W_K, W_V将输入X映射为K,V；Attamba用SSM_K和SSM_V block替代W_K和W_V，让SSM在chunk of P tokens上进行自回归扫描，将P个token的序列信息压缩为单个表示（推理时仅取每个chunk的最后输出K^{(p)}[-1], V^{(p)}[-1]）。Query投影W_Q保持不变。这种设计将KV-Cache从O(n)降至O(n/P)，attention FLOPs从O(n²)降至O(n²/P)。SSM的状态维度D_s=16即足够（>32对P=8无明显收益），总SSM参数开销约4M（60M模型的~6.7%）。

在LongMamba中的长上下文分析：LongMamba发现Mamba的隐藏状态通道（沿d_e维度）可基于感受野长度分为局部通道和全局通道。全局通道的感受野覆盖训练序列长度（如2k），但当输入长度S≫L时，累积衰减∏_{k=1}^S Ā_k = exp((ΣΔ_k)⊙A)（A为负矩阵）指数级趋近于零，导致全局通道无法捕获早期token的信息。LongMamba通过token filtering（跳过Δ_t<g的token更新）使有效衰减步数保持在≈L的规模，从而扩大全局通道的感受野。Mamba-1.4B在LongBench-E上通过LongMamba将accuracy从8.37%提升至17.33%（+8.96%），PG-19 60k tokens上的perplexity从>40降至<20。

在原始 Mamba 论文（Gu & Dao, 2023）中的定义：Mamba 架构由同质堆叠的 Mamba block 组成（expansion factor E=2），使用 SiLU/Swish 激活函数，可选 normalization layer。2x Mamba blocks 参数匹配 1x Transformer layer（12D²）。Block 内流程：x → Linear projection (expand 2x) → 分叉为两条路径：[Path 1: Causal Conv1d(kernel=4) → SiLU → Selective SSM scan], [Path 2: SiLU gate] → element-wise multiply → Linear output projection。SSM 参数：Δ 投影维度 R=D/16，state dimension N=16，S4D-Real 初始化（A_n = -(n+1)），实数值状态。与 H3 block 的主要区别：第一个 multiplicative gate 被 activation function 替换，第二个 multiplicative gate（SSM 后的 output gate）保留。论文验证了 Mamba 在 125M-2.8B 参数范围的 scaling laws、多项 zero-shot 下游评估、DNA/audio 等多模态建模、以及 synthetic tasks（Selective Copying, Induction Heads）上的性能。

在 SAMBA 论文中的用法：SAMBA 将 Mamba 作为混合架构的核心递归组件。Mamba 层按 d_e=2d_m（扩展因子2）、d_r=d_m/16（低秩投影维度）、d_s=16（状态维度）、kernel size=4（短卷积）配置。选择性门控 Δ 由 Softplus(U@W_r@W_q + b) 计算，Δ 初始化为 [0.001, 0.1] 范围，使用 S4D-Real 初始化（A_ij = log(j)）。在 Samba 架构中，Mamba 层负责捕获时间依赖语义并通过递归状态进行长程信息压缩——选择性门控使模型能在递归状态中"记住"重要信息，同时"遗忘"不相关信息。SAMBA 论文的分析表明：与纯 Mamba 模型相比，在混合架构中 Mamba 层的选择熵（S6 selection entropy）在中间层更高，说明有 SWA 层负责精确检索后，Mamba 层可以更专注于建模递归结构而非执行精确的输入选择（Figure 5b）。纯 Mamba 在 Passkey Retrieval 的零样本准确率与 SWA 模型相当（约 35% at step 0），表明纯 Mamba 确实存在记忆召回瓶颈。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces
- An_Empirical_Study_of_Mamba-based_Language_Models
- Attamba__Attending_To_Multi-Token_States
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement
- Rethinking_Token_Reduction_for_State_Space_Models
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## Structured State Space Model (SSM / S4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured State Space Model（S4）是由 Gu, Goel, and Ré (2022) 提出的一类深度学习序列模型，受经典状态空间模型（Kalman 1960）启发，可解释为 RNN 和 CNN 的组合。S4 通过一个隐式潜在状态 h(t) ∈ R^N 将 1D 输入序列 x(t) 映射为输出 y(t)，其连续时间动力学方程为 h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)，包含四个参数 (Δ, A, B, C)。经过离散化（将连续参数转为离散参数 A_bar, B_bar）后，模型可用两种方式计算：(1) 线性递归 h_t = A_bar·h_{t-1} + B_bar·x_t（推理时 O(1) per step）；(2) 全局卷积 y = x * K，其中 K = (CB, CAB, CA²B, ...)（训练时利用 FFT 并行化，O(L log L)）。"结构化"指的是 A 矩阵需要施加特定结构（最常用的是对角结构，如 S4D）以实现高效计算——A ∈ R^{N×N} 可表示为 N 个数而非 N²。S4 独立应用于每个输入通道（single-input single-output, SISO），总的隐藏状态维度为 D×N per input。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
S4 的核心计算流程（LTI 版本，训练时卷积模式）：
```
Input: x ∈ R^{B×L×D}
Parameters: A ∈ R^{D×N} (diagonal), B ∈ R^{D×N}, C ∈ R^{D×N}, Δ ∈ R^D (all time-invariant)

# Step 1: Discretization (ZOH)
A_bar = exp(ΔA)           # ∈ R^{D×N}
B_bar = (ΔA)^{-1}(exp(ΔA)-I)·ΔB  # ∈ R^{D×N}

# Step 2: Convolution kernel construction
K = (CB_bar, C·A_bar·B_bar, ..., C·A_bar^{L-1}·B_bar)  # ∈ R^L

# Step 3: FFT convolution (per channel)
y_d = x_d ∗ K_d  # for each channel d

# 推理时切换为 recurrent 模式:
h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_t  # O(1) per token
y_t = C^T h_t
```
关键特性：LTI 意味着参数 (Δ, A, B, C) 在所有时间步恒定。这使 S4 等价于一个线性递归和一个全局卷积，训练时卷积模式效率高，但在离散信息密集数据（文本、DNA）上表现不如 Transformer——因缺乏输入依赖的选择性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/state-spaces/s4（基于 PyTorch + CUDA extensions）。S4 有多种变体：S4-LegS（使用 HiPPO-LegS 矩阵初始化 A）、S4-FouT（傅里叶基）、S4D（对角简化版，包括 S4D-Lin 和 S4D-Real）、DSS（对角状态空间）、S5（MIMO 形式 + parallel scan）。S4 在 Long Range Arena (LRA) benchmark 上取得最优结果，在连续信号模态（音频、视频、时间序列）上表现出色。Mamba 论文指出 S4 的根本局限在于 LTI 属性——无法根据输入内容选择性关注或忽略信息。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---

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

## Zero-Order Hold (ZOH) Discretization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zero-Order Hold (ZOH) 是将连续时间 SSM 转换为离散时间 SSM 的离散化方法。连续系统 h'(t) = Ah(t) + Bx(t) 通过 ZOH 离散化后得到 h_t = A_bar·h_{t-1} + B_bar·x_t，其中转换公式为 A_bar = exp(ΔA)，B_bar = (ΔA)^{-1}(exp(ΔA) - I)·ΔB。ZOH 假设在两个采样点之间输入的 x(t) 值保持不变（即"零阶保持"），物理上等价于在采样间隔 Δ 内保持信号恒定。在 Mamba 中，Δ 不再是固定的常数，而是由输入 x_t 通过 s_Δ(x) = Broadcast_D(Linear_1(x)) 和 τ_Δ = softplus 动态生成，使离散化步长具有输入依赖性——这是 selection mechanism 的核心数学基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ZOH 离散化在 Mamba 中的具体计算：
```
# 输入: Δ_t ∈ R^{B×L×D}, A ∈ R^{D×N}, B_t ∈ R^{B×L×N}

# Step 1: 计算离散化步长（selection mechanism 的关键）
Δ_t = softplus(Linear_1(x_t) + bias_Δ)  # 输入依赖的步长

# Step 2: ZOH 离散化公式
A_bar_t = exp(Δ_t ⊙ A)        # ⊙ 广播 element-wise 乘
# A ∈ R^{D×N} 是对角矩阵，A_bar_t ∈ R^{B×L×D×N}

# B_bar 的计算:
B_bar_t = (ΔA)^{-1}(exp(ΔA) - I)·ΔB
# 由于 A 是对角的，此运算可逐元素简化
# 等价于: B_bar_t = Δ_t ⊗ B_t  （在 A→0 或简单初始化下）

# Step 3: 离散递归
h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊗ x_t
```

与 RNN gating 的联系（Theorem 1）：当 N=1, A=-1, B=1, s_Δ=Linear_1, τ_Δ=softplus 时，ZOH 离散化使 selective SSM 退化为经典 gated RNN：
```
Δ_t = softplus(Linear_1(x_t))
A_bar_t = exp(-Δ_t) = σ(-Linear_1(x_t)) = 1 - σ(Linear_1(x_t))
B_bar_t = 1 - A_bar_t = σ(Linear_1(x_t))
→ g_t = σ(Linear(x_t))
→ h_t = (1-g_t)·h_{t-1} + g_t·x_t
```
这证明了 selective SSM 中的 discretization 是 RNN gating 机制的原则性数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZOH 是最常用的离散化规则，替代方案包括双线性变换（bilinear/Tustin）和欧拉方法。在 Mamba 的 fused kernel 中，离散化和递归/scan 在 GPU SRAM 中融合完成，不将 A_bar_t 和 B_bar_t 写入 HBM，减少 IO 传输 O(N) 倍。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---

## HIPPO Theory / S4D Initialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HiPPO（High-order Polynomial Projection Operators, Gu, Dao, et al. 2020）是一种在线信号压缩理论：将输入信号 f(t) 的最优多项式近似投影到正交多项式基上，得到 recurrent 形式的系数更新方程 h'(t) = Ah(t) + Bf(t)。其中 A 矩阵的构造依赖于所选的正交基——最著名的是 HiPPO-LegS，使用指数扭曲的 Legendre 多项式，产生特定结构的 A 矩阵。S4D（Diagonal State Space Models, Gu, Gupta, et al. 2022）在此基础上将 A 矩阵限制为对角形式，并发展了多种初始化方案：S4D-Lin（A_n = -1/2 + iπn，线性间距频率，源于傅里叶基）、S4D-Real（A_n = -(n+1)，纯实数值）、S4D-Inv（A_n = -1/2 + i·N/π·(N/(2n+1)-1)，反比频率）。实部的 -1/2 或负值确保系统稳定（basis 函数被 e^{-t/2} 包络限定），虚部控制振荡频率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba 使用 S4D-Real 初始化（默认）：
```
# S4D-Real: 纯实数值对角初始化
A_n = -(n + 1)  # for n = 0, 1, ..., N-1
# A ∈ R^{D×N}, 所有 N 个元素都是负实数

# S4D-Lin (备选，用于 complex-valued SSM):
A_n = -1/2 + i·π·n  # for n = 0, 1, ..., N-1
# 实部固定 -1/2 (保证稳定性), 虚部线性增长 (振荡频率)
```

Mamba 论文中的 ablation（Table 8, 350M LM）：
- S4D-Lin (complex): perplexity 9.16
- S4D-Real (real): 8.85
- Random init (real, with S4D-Real parameterization): 8.71
- Random init (real, with original Mamba setup): 8.71

结论：在语言建模（离散模态）上，实数值 S4D-Real 和随机初始化均优于传统的复数值 S4D-Lin。这与早期 SSM 工作在音频等连续模态上需要复数值的发现互补——Mamba hypothesis：complex 适合 continuous modalities，real 适合 discrete modalities。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HiPPO 初始化和 S4D 变体在 Mamba 代码库中实现（https://github.com/state-spaces/mamba）。对于大多数 language modeling 应用，S4D-Real 或随机初始化足够。对于音频等连续信号任务，S4D-Lin 或 complex-valued 变体可能更好。S4 和 S4D 的完整理论细节见 Gu, Goel, and Ré (2022) 和 Gu, Gupta, et al. (2022)。S4D-Real 中 A 矩阵作为可学习参数，初始化后通过训练进一步优化。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---

## Mamba-2 (Structured State Space Duality / SSD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2由Dao和Gu(2024)提出，通过揭示SSM与attention之间的对偶性（Structured State Space Duality, SSD），将SSM重新表述为矩阵形式，使其可以利用类似FlashAttention的tiling策略进行高效计算。核心发现：SSM的序列变换可以写为半可分离矩阵（semiseparable matrix）M = L ◦ (C · B^T)，其中L是下三角矩阵，◦是逐元素乘法，使得SSM的前向传播等价于矩阵乘法Y = M · X。这一发现使Mamba-2能利用Tensor Core优化的矩阵乘法（而非Mamba的逐元素scan），训练速度达Mamba的8倍。SSD使用head_dim=64的多头结构（类似attention的多头），8个groups，expansion factor=2，conv window=4。论文中8B Mamba-2配置：56层，hidden dim 4096，state dim 128，8 groups，head dim 64，无位置编码。Mamba-2支持tensor parallelism仅需1次all-reduce（与Transformer持平），但需使用GroupNorm（而非LayerNorm）作为内部归一化，且group size需>256以保证统计量精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba-2 SSD层的前向传播（矩阵形式）：
```
Input: x ∈ R^{B×L×D}

// Step 1: Project to Q, K, V (analogous to attention)
Q, K, V = Linear_proj(RMSNorm(x))  // Q,K ∈ R^{B×L×H×P}, V ∈ R^{B×L×H×P}
// H = D/head_dim 个head, P = head_dim

// Step 2: Short convolution + activation
K = CausalConv1d(K, window=4)
Q = SiLU(Q)

// Step 3: SSM scan via matrix multiplication formulation
// M = L ◦ (Q · K^T) where L is lower triangular
// Equivalent computation via chunked parallel scan:
// Split L into chunks, process intra-chunk as MatMul, inter-chunk as recurrent

// Step 4: Output = M @ V (matrix multiply form)
Y = SSD_scan(Q, K, V, A)  // chunked scan using Tensor Core MatMuls

// Step 5: Gating + output
output = x + Linear_out(Y * SiLU(gate))
```
关键优势：Mamba-2的SSD scan通过chunked parallelism利用Tensor Core加速。将序列分为chunks，chunk内使用高效MatMul（Q·K^T和·V），chunk间使用recurrent state传递。论文中的Mamba-2使用head_dim=64、8 groups、state_dim=128、expansion=2、conv window=4。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba-2开源实现：https://github.com/state-spaces/mamba（包含SSD kernel）。论文中使用Megatron-LM的实现，支持tensor/sequence/pipeline parallelism。Mamba-2比Mamba训练快约3x（在8B规模），因SSD scan比Mamba的selective scan快8x。TP通信仅需1次all-reduce，与Transformer持平。适用场景：需要线性复杂度但追求比Mamba更高训练吞吐的大规模语言模型训练。

在 H-Net 中的用法：H-Net 使用 Mamba-2 作为 encoder/decoder 的构建模块。选择 Mamba-2 而非 Transformer 的原因：(1) SSM 的压缩归纳偏置——Mamba-2 将信息压缩为固定大小的 hidden state，天然适合 encoder 将多个输入 token 压缩为 richer representations 的角色；(2) 对 fine-grained 数据的处理能力——在字节级、DNA base-pair 级输入上 Mamba-2 显著优于 Transformer（消融中纯 Transformer E/D 表现最差）；(3) 效率——Mamba-2 的 O(L) 复杂度在操作未压缩长序列（L^0=8192）时至关重要。H-Net 的 E/D 使用 M4（4 层纯 Mamba-2 无 MLP），参数量约 6D²/layer（vs Transformer 的 12D²/layer）。消融还发现在 BPE token 级别输入上 Mamba E/D 也优于 Transformer E/D，说明优势不仅来自 fine-grained 输入处理，也来自 SSM 的压缩能力本身。

在 ML-Mamba 中的多模态用法：ML-Mamba 使用 Mamba-2 2.7B（在Pile数据集300B tokens上预训练）替换传统Transformer backbone（如LLaMA/Vicuna）作为MLLM的语言模型部分。关键设计优势：(1) RNN-like特性使推理时每token O(1)计算且内存恒定——即使处理729个visual tokens + 长文本生成，hidden state大小不变，无KV-Cache增长，实现171 tokens/s的生成速度（vs TinyLLaVA 38 tokens/s, MobileVLM v2 50 tokens/s）；(2) Mamba-2比Mamba-1快2-8倍，使ML-Mamba在推理性能上优于基于Mamba-1的MLLM（VL-Mamba、Cobra）。消融实验（Table 4）显示Mamba-2 2.7B在所有benchmark上全面超越780m和1.3b变体，验证了Mamba-2在MLLM中的scaling特性——更大SSM backbone类似Transformer scaling law持续提升多模态性能。Mamba-2的selective scan的input-dependent特性使模型能自适应地对visual token中的重要patch分配更高"注意力"，对不重要的background patches快速遗忘。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models
- Dynamic_Chunking_for_End-to-End_Hierarchical_Sequence_Modeling
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2
- Rethinking_Token_Reduction_for_State_Space_Models
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

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

## Memory Decay (α_t) and Forgetting Mechanism in Mamba-2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 中的 Memory Decay（记忆衰减）由 α_t = exp(-Δ_t · exp(A)) 控制，其中 Δ_t = Softplus(W_Δ u_t + b_Δ) 是输入依赖的门控标量，A 是可学习的标量参数。α_t ∈ (0,1) 决定每个时间步保留多少历史状态信息：α_t → 1 完全保留（h_t ≈ h_{t-1}），α_t → 0 完全遗忘（h_t ≈ Δ_t·B_t·x_t）。衰减通过乘法累积：第 i 个 token 在 t 时刻的记忆强度为 α_{i:t} = ∏_{j=i+1}^{t} α_j。整个状态可写为加权和 h_t = Σ_{i=1}^{t} α_{i:t} · B̄_i · x_i，这是 Sliding Window 方法和遗忘分析的关键性质。Stuffed Mamba 论文发现：在 8K 训练长度下，某些 head 的 α_t 始终接近 1（如首 token 累积 α_{1:t} > 0.997），模型未学会在必要时遗忘——即"遗忘机制失效"。这正是长度泛化失败的根本原因。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba-2 的记忆衰减和遗忘诊断：
```
# Mamba-2 head 中的遗忘机制
Δ_t = Softplus(W_Δ @ u_t + b_Δ)    # 输入依赖的门控
α_t = exp(-Δ_t * exp(A))          # 单个时间步的衰减因子

# 累积衰减（第 i 个 token 在时间 t 的保留强度）
α_{i:t} = ∏_{j=i+1}^{t} α_j       # t-i 次乘法累积

# 状态加权和形式（关键性质）
h_t = Σ_{i=1}^{t} α_{i:t} · (Δ_i · B_i) · x_i
    = Σ_{i=1}^{t} α_{i:t} · B̄_i · x_i

# Stuffed Mamba 的诊断发现
# 问题：α_t 始终 ≈ 1 → α_{1:T_train} > 0.997 → 几乎不遗忘
# 结果：超训练长度后 memory interference 导致检索失败
# 检索误差（公式 7）：
y_t = α_{s:t} · (C_t·B̄_s) · x_s + Σ_{i≠s} α_{i:t} · (C_t·B̄_i) · x_i
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                               当 token 过多且 α_{i:t} ≈ 1 时，此项急剧增大
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mamba-2 中 α_t 由 Softplus 激活的 Δ_t 经 exp(-Δ_t·exp(A)) 计算。训练时模型通过梯度下降学习 W_Δ, b_Δ, A 来控制衰减行为。Stuffed Mamba 的发现：短训练长度下模型"学会"设置 α_t ≈ 1（不遗忘）因为状态容量足够大。解决方向：(1) 训练长度 > 遗忘阈值（T_forget ∝ N_S）；(2) 推理时干预：RRI 缩放 α_t' = α_t^{0.9999} 强制轻微加速衰减，B_t' = 0.75·B_t 减弱插入强度。Sliding Window 利用加权和性质直接截断：h_t^{(w)} = h_t - α_{t-w+1:t}·h_{t-w}。该机制适用于所有可写为加权和的 RNN（GLA: G_t∈(0,1)^d 门控衰减；RWKV: e^{-w} channel-wise decay；RetNet: γ 固定衰减）。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## Sliding Window State Computation for Weighted-Sum RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding Window State Computation 是 Stuffed Mamba 论文提出的无需训练的推理时遗忘诱导方法。利用 Mamba-2 状态可写为加权和 h_t = Σ_{i=1}^{t} α_{i:t}·B̄_i·x_i 的性质，通过 h_t^{(w)} = h_t - α_{t-w+1:t}·h_{t-w} 精确计算最近 w 个 token 的状态，等价于在序列上滑动一个 w 大小的窗口。维护 h_t（正常状态）、h_{t-w}（w 步前的状态）和 Δ_sum（Δ 的累积和），每步计算 α_window = exp(-Δ_sum·exp(A)) 并通过矩阵减法得到窗口状态。该方法的优势：(1) 无需重新训练；(2) 数学上精确（非近似）；(3) 额外计算和内存开销极小（两个额外状态张量 + 一个标量乘法和矩阵减法）；(4) 适用于所有可写为加权和的 RNN（GLA、RWKV、RetNet 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sliding Window 推理算法：
```
# 推理时维护三个量
h_t      = 0  # [N, P] 正常递归状态
h_{t-w}  = 0  # [N, P] w 步前的状态（用于减法）
Δ_sum    = 0  # scalar, Δ 累积和
window_size = w

for each token at step t:
    # 1. 正常 Mamba-2 状态更新
    Δ_t = Softplus(W_Δ @ u_t + b_Δ)
    α_t = exp(-Δ_t * exp(A))
    B̄_t = Δ_t * B_t
    h_t = h_{t-1} * α_t + B̄_t * x_t   # [N, P]

    # 2. 维护 Δ 累积和（避免浮点不稳定）
    Δ_sum = Δ_sum + Δ_t

    # 3. 计算窗口衰减因子
    α_window = exp(-Δ_sum * exp(A))

    # 4. 精确窗口状态 = 完整状态 - 窗口前的状态
    h_t^{(w)} = h_t - α_window * h_{t-w}   # [N, P]

    # 5. 使用窗口状态 query
    y_t = C_t @ h_t^{(w)} + D ⊙ x_t        # [1, P]

    # 6. 更新 h_{t-w}（延迟 w 步）
    if t > w:
        h_{t-w} = update_buffer(h_{t-w})   # FIFO 或循环缓冲区
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 用 Ring Buffer 维护 h_{t-w} 的历史；(2) 维护 Δ_sum 而非直接计算 α_window 的乘积，因为 Δ ∈ R，求 exp(-sum·exp(A)) 比连乘 exp(-Δ_i·exp(A)) 更数值稳定；(3) 窗口大小 w 是超参数，对短上下文性能有影响（窗口太小则信息不足，太大则遗忘不足）。Stuffed Mamba 实验表明，Sliding Window 在 32K 上下文上将 Mamba-2 370M 的 LM loss 从 ~15 降至 ~8-10，但短上下文性能略有下降。适用场景：已有训练好的 Mamba-2 模型、需要处理超训练长度的上下文、不希望或无法重新训练时的推理时干预。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## RRI (Reduced Memory Retention and Insertion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RRI 是 Stuffed Mamba 论文提出的无需训练的遗忘诱导方法。对 Mamba-2 的记忆保留强度 α_t 和记忆插入强度 B_t 分别施加缩放因子：α_t' = α_t^{0.9999}（将衰减因子推向 0，加速遗忘），B_t' = 0.75·B_t（减弱新信息写入强度）。超参（0.9999 和 0.75）通过在 32K 上下文的预训练数据上验证选择。RRI 的核心思想是：既然模型在短训练长度下"学会"了过度保留信息（α_t ≈ 1），推理时可以通过人为干预系统性地注入遗忘来缓解。这比 LongMamba 的简单 Δ_t/2 方法更精细——RRI 分别控制保留和插入两个维度，且缩放因子经过验证选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RRI 干预流程：
```
# MoE 原始 Mamba-2 状态更新（一个 head）
Δ_t = Softplus(W_Δ @ u_t + b_Δ)
α_t = exp(-Δ_t * exp(A))
B_t = σ(Conv(W_B @ u_t))       # [N, 1]
h_t = h_{t-1} * α_t + Δ_t * B_t * x_t

# === RRI 干预 ===
# 参数: λ_α = 0.9999, λ_B = 0.75 (通过 32K 验证集选择)
α_t' = α_t ** λ_α              # α_t ∈ (0,1), 幂次缩放 → 更接近 0
B_t' = B_t * λ_B               # 减弱插入强度
h_t' = h_{t-1} * α_t' + Δ_t * B_t' * x_t

# 效果:
# - α_t: 从 ≈ 0.999 → ≈ 0.9989 (对每个 step), 累积效果显著
# - B_t: 插入强度减弱 25%, 降低新 token 对状态的冲击
# 结果: 32K context 下 LM loss 从 ~15 降至 ~8-10
# 代价: 短上下文性能下降（记忆插入更弱）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 Mamba-2 的推理代码中，在 Δ_t、α_t、B_t 计算之后、状态更新之前插入缩放操作。无需修改模型权重或重新训练。λ_α=0.9999 看起来接近 1（几乎不改变单步 α_t），但累积效应在长序列中显著（如 32K 步：0.9999^{32000} ≈ 0.04 的附加衰减）。λ_B=0.75 通过验证集网格搜索选择。与 Sliding Window 相比：RRI 更温和（渐进遗忘而非硬截断），但超参选择需要任务相关调整。与 LongMamba 相比：LongMamba 仅缩放 Δ_t（Δ_t→Δ_t/k），同时影响 α_t 和 B_t 的耦合方式，RRI 解耦了保留和插入两个维度。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## Truncated BPTT (Backpropagation Through Time)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Truncated BPTT 是将完整 BPTT（沿整个序列反向传播梯度）截断为固定长度 K 的技巧，通过在每个截断边界 detach 隐藏状态来阻止梯度继续向更早的时间步流动。Stuffed Mamba 论文使用此技术训练长上下文 Mamba-2：将 12 个序列拼接，每个序列处理后 detach 隐藏状态，下一个序列从 detach 的状态继续前向。这等价于 concatenation + 在序列边界截断梯度。目的：(1) 使状态初始值分布更多样化（非始终零初始化）；(2) 降低内存成本——只需缓存 K 步的激活值，而非完整序列。GLA 论文（Yang et al., 2024a）先提出此方法用于扩展 RNN 的上下文长度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Truncated BPTT with 12 sequences per sample
h = zeros(N, P)  # 初始状态
total_loss = 0

for seq_idx in range(12):  # 12 个拼接序列
    # 前向传播（状态连续）
    for t in range(len(seq)):
        h = update(h, seq[t])       # 使用上一序列的最终状态
        total_loss += CE(head(h), seq[t+1])

    # 梯度截断边界
    h = h.detach()  # 停止梯度向更早序列传播

    # 反向传播仅到当前序列
    total_loss.backward()  # 梯度传播范围: 当前序列
    optimizer.step()
    total_loss = 0
```
关键区别：状态 h 在推理时连续流动（保存上下文信息），但梯度在序列边界截断（节省内存）。每序列内的 BPTT 长度 = 序列长度，总共 12 个独立反向传播段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：在每个序列边界调用 `h = h.detach()`。Stuffed Mamba 使用 12 序列拼接（≈ 12 × T_train 总长），结合 WSD LR scheduler 和 0.5M tokens/batch。该技术已被 GLA、RWKV、Mamba-2 等 RNN 训练广泛采用。优势：内存 O(K) vs 完整 BPTT 的 O(T)，且状态初始化条件更丰富（非始终零初始化）。局限：无法学习跨 K 步的长距离依赖——但论文中的 12×8K=96K 已足够捕获大多数依赖。在长序列训练中，推荐与 gradient clipping（论文用 1.0）配合使用。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## WSD (Warmup-Stable-Decay) Learning Rate Scheduler

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WSD 是由 MiniCPM 论文（Hu et al., 2024）提出的学习率调度器，将训练分为三个阶段：(1) Warmup 阶段（s < W）：线性从 0 增加到峰值 η；(2) Stable 阶段（W ≤ s ≤ T）：保持恒定最大学习率 η；(3) Decay 阶段（T < s < S）：按函数 f(s-T) 衰减。核心优势：(1) 无需预定义总训练步数（与 Cosine 不同）；(2) 可从 Stable 阶段任意 checkpoint 恢复训练——恢复到相同高 LR 继续 Stable 或直接进入 Decay；(3) Decay 阶段仅需 ~10% 总 tokens 即可达到或超越 Cosine 最优性能；(4) Loss 在 Decay 阶段经历快速显著下降。Stuffed Mamba 使用 WSD 配合 10% decay steps、1000 步 linear warmup、50K 步 linear decay。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# WSD Scheduler 定义
def WSD(step, warmup_steps, stable_steps, decay_steps, peak_lr):
    if step < warmup_steps:                    # Warmup
        return (step / warmup_steps) * peak_lr
    elif step < warmup_steps + stable_steps:   # Stable
        return peak_lr
    else:                                      # Decay (linear)
        progress = (step - warmup_steps - stable_steps) / decay_steps
        return peak_lr * (1 - progress)

# Stuffed Mamba 的具体配置
warmup_steps = 1000
decay_steps = 50000
stable_steps ≈ 10 * decay_steps  # 10% decay ratio
peak_lr ∈ {1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3}  # sweep 选择最优
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WSD 特别适合持续预训练（continue pretraining）场景——Stuffed Mamba 从 Mamba-2 8K checkpoint 出发，使用 WSD 方便地在不同训练长度下恢复和继续训练。Decay 阶段损失快速下降，使得用较少的 tokens 即可完成。Stuffed Mamba 使用 linear decay（与 MiniCPM 原论文的 exponential decay 不同），1000 步 warmup + 50K 步 decay。学习率 sweep {1e-5,...,1e-3}，通过 Passkey Retrieval 验证选择最优——注意：不同 LR 的 validation loss 可能相似但 Passkey Retrieval 精度差异巨大。WSD 也被 DeepSeek 等模型采用，体现了从 Cosine 向多阶段调度的行业趋势。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## Passkey Retrieval (Long-Context Evaluation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Passkey Retrieval 是由 Mohtashami & Jaggi (2023) 提出的简单合成长上下文评估任务，也是 ∞Bench（Zhang et al., 2024a）的核心组件。任务设计：在大量无关噪声文本（如重复的 "The grass is green. The sky is blue..."）中隐藏一个 5 位数字 "passkey"，模型需要在读取全部上下文后回答 "What is the passkey?"。Stuffed Mamba 论文广泛使用此任务评估 Mamba-2 的长上下文召回能力，核心发现：(1) 8K 训练长度的模型在 ≤8K 内近乎完美但 >16K 后降至 ~0%；(2) 该任务对长度泛化的敏感度远高于 validation loss——不同 LR 的 loss 相似但 Passkey 精度差异巨大；(3) 370M Mamba-2 在 256K 训练后达到近乎完美的 Passkey Retrieval。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Stuffed Mamba 使用的 Passkey Retrieval prompt 模板：
```
There is important info hidden inside a lot of irrelevant text. Find it and memorize it.

[重复噪声文本: "The grass is green. The sky is blue..." × N]

The passkey is 34847. Remember it. 34847 is the passkey.

[重复噪声文本: "The grass is green. The sky is blue..." × M]

What is the passkey? The passkey is
```
评估指标: 模型输出是否精确包含 5 位 passkey（greedy decoding, FP32）。Needle 位置均匀分布：n 个样本的 needle 分别插入在位置 T×i/n (i=0,...,n-1)。论文使用 accuracy=N_correct/N_total，与 ∞Bench 的 passkey 设置一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用方式：作为长上下文能力的主要验证指标（而非次要指标），因其对长度泛化的敏感度远高于 perplexity。Stuffed Mamba 的实践：(1) 用 Passkey Retrieval 而非 validation loss 进行 LR 选择和 checkpoint 选择；(2) 在 T_forget 搜索中使用 Passkey 准确率 >95% 作为 T_recall 的定义；(3) 均匀 needle 位置确保评估所有深度（开头到结尾）。注意：greedy decoding 给出最佳结果（其他 decoding 参数显著降低精度）；BF16 精度下 Δ_t 和 α_t 有 ~1e-3 误差但不影响主要结论。局限性：Passkey 是简单合成任务，完美准确率不一定翻译为真实长上下文任务的表现。Stuffed Mamba 验证了继续预训练后 370M 在 256K 达到近乎完美（首个 <1B 模型在此长度达到此性能）。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---

## Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Attention 将标准 softmax(QK^T)V 的 O(N²) 降至 O(N)（Katharopoulos et al. 2020; Schmidhuber 1992）。核心：用可分离特征映射 φ(Q)φ(K)^TV 替代 softmax，利用结合律先算 φ(K)^TV（固定大小），再与 φ(Q) 乘。等价于 RNN: s_t = s_{t-1} + φ(k_t)^T v_t, o_t = φ(q_t)s_t。每 token O(1) 且内存恒定。Naive linear attention 性能不及 softmax；RWKV/RetNet/GLA/Mamba 通过 decay/门控/数据依赖缩小差距。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RWKV WKV 是 linear attention 的直接改进——learnable per-channel exponential decay 替代等权求和：
```
s_t = diag(w)·s_{t-1} + k_t^T·v_t     # diag(w): channel-wise decay
wkv = diag(u)·k_t^T·v_t + s_{t-1}      # u: boost 当前 token
o_t = r_t @ wkv                         # r: receptance query
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
双模式：(1) RNN 模式（推理）：逐 token 更新 O(1)；(2) 并行模式（训练）：associative scan 沿时间维并行。RWKV 训练用 custom CUDA kernel SRAM-resident state 沿非时间维并行。FLA 库提供多种变体高效实现。

JRT论文使用Taylor 2阶近似特征图 φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2（feature dim d̃=273 for d=16 base），在Based架构中结合gated convolution(kernel=3)+sliding window attention(window=128)+linear attention的hybrid layout。PLA将linear attention扩展为encoder-decoder：encoder区域用非因果sum预计算KV-state，decoder区域沿用causal cumsum。JRT-Prompt通过2×prefill重复context，利用linear attention的2N prefill仍快于attention的N prefill。

RWKV 原始论文（RWKV: Reinventing RNNs for the Transformer Era, EMNLP 2023）首次将 linear attention 扩展到 14B 参数规模，验证了线性注意力在大规模 LLM 中的可行性。其 WKV 算子使用通道级可学习指数衰减 w∈(R_{≥0})^d 替代等权特征图求和：`wkv_t = (Σ e^{-(t-1-i)w+k_i}⊙v_i + e^{u+k_t}⊙v_t)/(Σ e^{-(t-1-i)w+k_i} + e^{u+k_t})`，其中分母提供归一化（而非 feature map 的 Σ φ(k)），u 为当前 token bonus。数值稳定版本使用共享指数 p_t 技巧避免 exp 溢出：`q=max(p_{t-1}, u+k_t); wkv_t = (e^{p_{t-1}-q}⊙a'_{t-1}+e^{u+k_t-q}⊙v_t)/(e^{p_{t-1}-q}⊙b'_{t-1}+e^{u+k_t-q})`。RWKV 推理时将 WKV 递归化为 RNN：`a_t = e^{-w}⊙a_{t-1}+e^{k_t}⊙v_t; b_t = e^{-w}⊙b_{t-1}+e^{k_t}; wkv_t = a_t/b_t`，实现 O(d) 空间（仅需存储 a_t,b_t,p_t 三个 d 维向量）和 O(1) 时间 per token。论文证明 RWKV scaling law 与 Transformer 相同（r²=0.994），12 项 NLP benchmark 上 FLOP-matched 性能与 Pythia/OPT/BLOOM 相当。开源：https://github.com/BlinkDL/RWKV-LM，预训练模型：https://huggingface.co/RWKV。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models
- Linearizing_Large_Language_Models

SUPRA 的具体用法：SUPRA 用 MLP kernel φ(x)=ReLU(Wx+b) 作为可学习特征图，queries 和 keys 共享同一 MLP 权重。相似度函数变为 sim(q_i,k_j)=RoPE(φ(q_i))·RoPE(φ(k_j))，加入固定衰减向量 γ∈(0,1)^h。输出用 GroupNorm 而非分母除法归一化：v'_i=GroupNorm(Σ_{j=1}^{i} γ^{i-j}·sim(q_i,k_j)·v_j)。训练时用 Lightning Attention 2 的 Triton kernel 做序列并行，推理时切换为循环模式 O(1) per-token。关键区别：(1) MLP kernel 替代固定 ELU kernel；(2) GroupNorm 替代分母除法，解决大规模训练稳定性；(3) RoPE 提供相对位置编码；(4) 固定 decay 提供位置偏置。验证了线性注意力可通过 uptraining 从强预训练 Transformer 获得，不必从零训练。

---

## Prefix Linear Attention (PLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PLA（Prefix Linear Attention）是JRT-RNN论文提出的encoder-decoder线性注意力变体。将输入序列分为前缀encoder区域（前M个token，非因果处理）和decoder区域（后N-M个token，causal处理）。Encoder使用独立投影(k_e, v_e)，decoder使用独立投影(k_d, v_d)，两套投影不共享（区别于Prefix-LM的单套投影）。核心公式：y_i = φ(q_i)(Σ_{j=1}^{i}k_d[j]^T v_d[j] + Σ_{j=1}^{M}k_e[j]^T v_e[j]) / φ(q_i)(Σ_{j=1}^{i}k_d[j] + Σ_{j=1}^{M}k_e[j])。Decode阶段O(1) per token（与标准linear attention相同）——prefix的贡献在prefill时预计算为固定KV-state s_M = Σ_{j=1}^{M}(k_e[j]^T v_e[j] + k_d[j]^T v_d[j])。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: u ∈ R^{N×d}, prefix length M, feature map φ (e.g. Taylor 2nd-order)

// Encoder projections (non-causal, tokens 1..M)
k_e = φ(W_{ke} · u_{1:M}),  v_e = W_{ve} · u_{1:M}

// Decoder projections (causal, tokens 1..N)
k_d = φ(W_{kd} · u),  v_d = W_{vd} · u,  q_d = φ(W_{qd} · u)

// Prefill: compute encoder KV-state (non-causal sum)
KV_enc = Σ_{j=1}^{M} k_e[j]^T v_e[j]       // ∈ R^{d×d̃}
K_enc  = Σ_{j=1}^{M} k_e[j]                 // ∈ R^{d̃}

// Decoder prefill: cumsum from encoder-init state
KV_dec[i] = KV_enc + Σ_{j=1}^{i} k_d[j]^T v_d[j]
K_dec[i]  = K_enc  + Σ_{j=1}^{i} k_d[j]
y_i = (q_d[i] · KV_dec[i]) / (q_d[i] · K_dec[i])

// Decoding (i > M, O(1)):
s_i = s_{i-1} + k_d[i]^T v_d[i],  z_i = z_{i-1} + k_d[i]
y_i = (q_d[i] · s_i) / (q_d[i] · z_i)
```
训练时追加MLM loss: L = (w1·L_NTP + w2·L_MLM)/(w1+w2)，encoder区域随机mask比例P的token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于Based架构实现（交替gated conv+sliding window+PLA层），feature map用2阶Taylor近似。JRT-RNN CUDA kernel扩展ThunderKittens Based kernel：先fnbased(k_e,v_e)计算encoder KV-state存入寄存器，再fnbased(q_d,k_d,v_d)从该状态续算decoder。Pre-fill比FA2快19.2×（N=32768, H100）。开源：https://github.com/HazyResearch/prefix-linear-attention。适用于需要recall-intensive ICL但保持O(1)推理内存的循环LM场景。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## WKV (Weighted Key-Value) Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WKV 是 RWKV 核心注意力原语：带 channel-wise learned exponential decay 的 linear attention。每 channel 有独立 decay w∈(0,1) 和 boost u。**原始论文（RWKV-4, EMNLP 2023）**首次提出：向量 state（head size=1），带分母的 softmax-like 归一化 + Sigmoid receptance gating。公式：`wkv_t = (Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i}⊙v_i + e^{u+k_t}⊙v_t) / (Σ_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t})`，其中 w 为非负通道级时间衰减（`e^{-(t-i)w}≤1`，确保历史信息指数衰减），u 为当前 token 的 bonus 参数（独立于衰减路径，让当前 token 获得特殊权重）。递归形式：`a_t = e^{-w}⊙a_{t-1} + e^{k_t}⊙v_t; b_t = e^{-w}⊙b_{t-1} + e^{k_t}; wkv_t = a_t/b_t`。数值稳定实现使用共享指数技巧：维护 p_t 存储 a_t,b_t 的公共指数，避免 exp 溢出。内部状态共 5 部分（x_t, y_t, a'_t, b'_t, p_t），总大小 5DL。Eagle (RWKV-5) 升级为矩阵 state s∈R^{(D/h)×(D/h)}（head=64），LayerNorm 替代分母，SiLU gating + 线性 receptance。Finch (RWKV-6) 将静态 w 升级为 data-dependent w_t。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Eagle WKV: w=exp(-exp(ω)); s_t=diag(w)·s_{t-1}+k_t^T·v_t; wkv=diag(u)·k_t^T·v_t+s_{t-1}; o_t=LayerNorm(r_t@wkv)
Finch WKV: d_t=lora_d(ddlerp_d(x_t,x_{t-1})); w_t=exp(-exp(d_t)); s_t=diag(w_t)·s_{t-1}+k_t^T·v_t
GoldFinch (Finch-C2) WKV改进：k_t = ddlerp_k(x_t,x_{t-1})W^K·(1-w_t)，key乘以(1-decay)以保持kv-state行归一化；移除Gate（减参数）；LayerNorm across all heads替代GroupNorm；第二Value u'_t = u_t W^V + tanh(u_t W^{UD})W^{UU}替代Finch的静态u(bonus)项。Finch-C2参数更少但性能优于原Finch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理 RNN 模式 O(1) per token。训练 custom CUDA kernel SRAM-resident state。16k 序列 Finch kernel 比 Flash Attn v2 快 4.2×（A100）。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## Token Shift (lerp/ddlerp)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Shift 是 RWKV 轻量时序信息混合机制，类似 kernel size=2 causal conv 但复用参数。Eagle 用静态 lerp(a,b)=a+(b-a)⊙μ（learnable per-channel 混合比）。Finch 用 ddlerp(a,b)=a+(b-a)⊙lora(a+(b-a)⊙μ_x)，lora(x)=λ+tanh(xA)B（A∈R^{D×32},B∈R^{32×D}），使混合比依赖输入内容。允许单层形成 induction heads，替代显式位置编码。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Eagle lerp: r_t = (x_t+(x_{t-1}-x_t)⊙μ_r)@W_r
# Finch ddlerp: lora_r(x)=λ_r+tanh(x@A_r)@B_r
#   r_t = (x_t+(x_{t-1}-x_t)⊙lora_r(x_t+(x_{t-1}-x_t)⊙μ_x))@W_r
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
同时用于 Time Mixing(r,k,v,g) 和 Channel Mixing(r',k')。前一 token 存 state（每层 2D float）。替代显式位置编码使 RWKV 可处理任意长度。GoldFinch进一步引入DDLoRAdapt: loradapt_□(x)=x+tanh(xC_□)D_□，在ddlerp基础上再叠加data-dependent additive LoRA偏移，用于GOLD Attention中的key和value生成。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## Matrix-Valued State in RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将 RNN hidden state 从向量 s∈R^D 扩展为矩阵 s∈R^{(D/h)×(D/h)} per head。每 head 维护 K^TV 矩阵记忆库：K 各行作 input gate，V 分配到 state 各行，每行独立 decay。RWKV-4 state 为向量（head=1）；Eagle/Finch 为矩阵（head=64），总 state 从 5DL→66DL（~13×）。矩阵 state 编码 key-value 间二阶交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
s_t = diag(w)·s_{t-1} + k_t^T·v_t   # s∈R^{(D/h)×(D/h)}
# s[i,j]: 第i key通道 × 第j value通道的加权和
# diag(w): row i 以 w[i] 衰减
```
vs 向量: s_t = w⊙s_{t-1} + k_t⊙v_t（无跨通道交互）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WKV 用 float32 计算。O(D²/h) FLOPs per token。消融：RWKV6-Pile avg 50.7% > RWKV4-Pile 47.7%（Table 18）。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## Data-Dependent Decay (Dynamic Recurrence)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch 引入 w_t=exp(-exp(d_t))，d_t 由 LoRA (A_ω∈R^{D×64},B_ω∈R^{64×D}) 基于 ddlerp 后输入生成。每 channel decay 每时间步动态变化，实现选择性记忆：重要 token 降低 decay 延长保留，无关 token 加速遗忘。与 Mamba 的 selective SSM（A_t）精神相似但实现不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
d_t = lora_d(ddlerp_d(x_t,x_{t-1}))  # rank=64
w_t = exp(-exp(d_t))                  # ∈(0,1), contraction
s_t = diag(w_t)·s_{t-1} + k_t^T·v_t   # 动态衰减
```
两次 exp 确保 w_t∈(0,1)，state 不发散。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch 每 Time Mixing 层有 LoRA A_ω/B_ω（rank 64）。训练时梯度更新，推理固定。MQAR 长上下文能力的关键驱动。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## Time Mixing and Channel Mixing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RWKV 每 residual block 两 Pre-LayerNorm 子层：Time Mixing（=Transformer attention，WKV 融合跨时间信息）和 Channel Mixing（=Transformer FFN，ReLU²+sigmoid gate 沿特征维变换）。原始 RWKV 论文（EMNLP 2023）首次提出这一架构：Time Mixing 使用 token shift→r/k/v 线性投影→WKV 带分母的 softmax-like 算子→Sigmoid(r)⊙wkv 输出门控；Channel Mixing 使用 token shift→r'/k'→Squared ReLU(k')=max(k',0)²→W'_v 线性投影→Sigmoid(r') 门控。两子块均输出 `o_t = W_o · (σ(r_t) ⊙ wkv_t)`（Time Mixing）和 `o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`（Channel Mixing）。借鉴 Gated MLP/MLP-Mixer 设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Time Mixing: Token Shift→r/k/v/g→WKV state→LayerNorm(r@wkv)→SiLU(g) gate→output。Channel Mixing: Token Shift→r'/k'→ReLU²(k')@W_v'→σ(r') gate→output。Eagle 缩 Channel Mixing hidden dim 至 3.5D 补偿新增 gate 参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两子层都用 residual connection。Finch 中两子层均用 ddlerp 替代 lerp。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

---

## LoRA-augmented Recurrence

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch 用 LoRA 生成 data-dependent 递归参数偏移：lora(x)=λ+tanh(xA)B，A∈R^{D×32},B∈R^{32×D}。不同于传统 W'=W+BA fine-tuning，此处 LoRA 使 token-shift μ_□ 和 decay ω 被 data-dependent offset 动态增强。rank-32 低秩矩阵仅 64D 参数（vs D²），约 0.5% per block 参数增量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
lora_r(x) = λ_r + tanh(x@A_r)@B_r   # rank=32, 65D params
ddlerp_r(x_t,x_{t-1}) = x_t + (x_{t-1}-x_t)⊙lora_r(x_t+(x_{t-1}-x_t)⊙μ_x)
# decay LoRA 加倍: A_ω∈R^{D×64}, B_ω∈R^{64×D} (rank=64)
```
初始化 A,B~U(-1e-4,1e-4)，初始 data-dependent 项≈0，从 Eagle 行为逐步学习。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
每 Time Mixing 层 5 组 LoRA（r/k/v/g rank-32, ω rank-64）。消融: 完整 LoRA loss=2.91 < 仅 decay 2.923 < 无 LoRA 2.926。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## MQAR (Multi-Query Associative Recall)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Arora et al. (2023) 提出的合成任务：上下文中多个 key-value pair，给定 query 需正确回忆对应 value。难度随序列长度和 pair 数量增加。与 in-context learning 能力相关（Elhage 2021, Olsson 2022），已成为评估新架构设计的重要基准。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: color->blue, animal->dog, color->  → Target: blue
```
模型需选择性检索 "color→blue" 而非其他 pair。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch 在 MQAR 上极高准确率，超越所有已知训练过大模型的非 Transformer 架构。GoldFinch作为hybrid架构（后1/3层为GOLD attention），在MQAR上取得完美分数（100% recall），与纯attention Transformer持平，验证了hybrid设计在保持linear attention效率的同时不牺牲长程精确检索能力。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## RWKV Architecture Family

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RWKV（Receptance Weighted Key Value）是基于 Linear Attention+RNN 的 LLM 架构家族。核心目标：Transformer 可并行训练 + RNN O(1) 推理。**原始论文"RWKV: Reinventing RNNs for the Transformer Era"（EMNLP 2023）**首次提出：Stacked residual blocks（Time Mixing + Channel Mixing Pre-LayerNorm），向量 state（head size=1），带分母归一化的 WKV 算子，Sigmoid receptance 门控，通道级可学习静态指数衰减 w。训练使用 time-parallel mode（类似 Transformer 并行矩阵乘法），推理使用 time-sequential mode（RNN 递归更新，O(d) 空间 + O(1) 时间）。训练 6 个规模（169M→14B）于 Pile 330B tokens，14B 为当时最大密集 RNN。关键设计：Small Init Embedding（U(±1e-4)+LayerNorm 加速收敛），Custom CUDA kernel 用于 WKV 串行扫描并行化，无位置编码（Token Shift 替代），无 bias 的线性层。演进：RWKV-4（vector state, 分母 WKV, Sigmoid r）→ Eagle/RWKV-5（matrix state, LayerNorm, SiLU gating）→ Finch/RWKV-6（ddlerp, data-dependent w_t, LoRA）→ RWKV-7（in-context learning params）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
RWKV Block: LayerNorm → Time Mixing(WKV: ddlerp→k^T·v decay+accum→receptance query→SiLU gate→output)→residual→LayerNorm→Channel Mixing(ddlerp→ReLU²(k')→σ(r')gate)→residual
State: 每层 2D(token shift history)+D²/h(WKV per head), 总 66DL
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
完全开源 Apache 2.0（GitHub+HuggingFace）。1.12T tokens 多语言训练。支持 NLP/多语言/code/长上下文/多模态（VisualRWKV/Music/Audio）。Finch-C2是GoldFinch论文中提出的Finch(RWKV-6)改进版：移除gate、LayerNorm across heads替代GroupNorm、key×(1-w)保持行归一化、数据依赖的第二Value替代静态bonus项。GoldFinch将Finch-C2作为前2/3层（线性pre-fill），后1/3层使用GOLD Transformer（full attention over compressed key cache）。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## Small Init Embedding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Small Init Embedding 是 RWKV 原始论文提出的一种嵌入层初始化策略：将 token embedding 矩阵初始化为极小值（U(±1e-4) 均匀分布，而非标准的 N(0, 0.02) 正态分布），并在 embedding 后立即加一个额外的 LayerNorm。论文观察到标准 Transformer 训练初期 embedding 矩阵变化缓慢，模型难以从初始噪声状态快速脱离。极小初始化使 embedding 值接近零，经过 LayerNorm 后因输入值小而梯度方向变化剧烈——一步微小的参数更新即可产生大幅方向改变，加速收敛。实验验证（Figure 9）：使用 small init emb 的训练 loss 下降速度和最终收敛均优于标准初始化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Standard (GPT/BERT): Embedding ~ N(0, 0.02), 无额外 LayerNorm
x = Embedding(token_ids)          # ~N(0,0.02), 初始值分散
x = x + PositionalEncoding        # 或 RoPE
→ Transformer blocks...

# RWKV Small Init Embedding:
x = Embedding(token_ids)          # ~U(±1e-4), 初始值接近零
x = LayerNorm(x)                  # 额外 LayerNorm：放大梯度方向变化
→ RWKV blocks...
```
关键机制：当 embedding 值极小时，LayerNorm 的输入均值和方差接近零，微小的参数梯度变化经过 LayerNorm 的除法（除以接近零的标准差）后被显著放大，导致 embedding 快速重组到有意义的表示空间。论文附注指出实验中使用的是 U(±1e-4) 而非 RWKV 实际使用的 N(0, 1e-4)，但差异可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`nn.Embedding(vocab_size, dim).weight.data.uniform_(-1e-4, 1e-4)`，后接 `nn.LayerNorm(dim)`。适用于深度 post-LN 架构的训练加速，尤其当 embedding 维度较大时效果显著。论文通过对比实验（batch size=400）验证了小初始化嵌入 + 额外 LayerNorm 相比标准正态初始化的 loss 收敛加速效果。该策略随后被 Eagle/Finch 等 RWKV 后续版本继承。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era

---

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

## Delta Rule (DeltaNet)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Delta Rule 由 Widrow et al. (1960) 提出，Schlag et al. (2021) 将其引入线性 Transformer 形成 DeltaNet。在 DeltaNet 中实现为：S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T，其中 S ∈ R^{d×d} 是记忆矩阵。计算分两步：（1）读旧值：v_t^old = S_{t-1} k_t；（2）写入增量：用 Householder 变换 (I - β_t k_t k_t^T) 擦除旧关联并写入 β_t v_t k_t^T。从快速权重编程视角，delta rule 等价于对在线回归目标 L(S) = 1/2 ||S k_t - v_t||² 执行一步 SGD：S_{t+1} = S_t - β_t ∇L(S_t)，β_t 为自适应学习率。DeltaNet 的优势在于精确 key-value 替换（优于 Mamba2 的简单叠加），局限在于缺乏全局遗忘机制（只能逐个修改 key-value 对）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
v_old = S_{t-1} @ k_t
v_new = β_t * v_t + (1 - β_t) * v_old
S_t = S_{t-1} - v_old @ k_t^T + v_new @ k_t^T
    = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Yang et al. (2024b) 提出基于 WY 表示的 chunkwise 并行训练算法，使 DeltaNet 训练从不可行变为接近 Mamba2 的速度。开源：https://github.com/NVlabs/GatedDeltaNet。适用于需要精确 key-value 联想记忆的序列建模场景。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---

## Chunkwise Parallel Training for Linear RNNs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunkwise Parallel Training 是将线性 RNN 序列计算分块并行的训练算法（Hua et al., 2022; Sun et al., 2023; Yang et al., 2024a,b）。将长度 L 的序列分为大小为 C 的 chunk，chunk 内使用 dense matmul（利用 Tensor Core），chunk 间通过 recurrent state 传递。实现 O(L) 时间复杂度和 O(C²L) 空间复杂度。Mamba2 的 SSD 分解等价于这种算法。Gated DeltaNet 扩展了 DeltaNet 的 chunkwise 算法：在 WY 表示中加入 chunk-local decay mask Γ（(Γ)_{ij} = γ^i / γ^j），通过修改 T = (I + strictLower(diag(β)(Γ ⊙ K K^T)))^{-1} diag(β) 实现。xLSTM 7B 使用该算法的 Tiled Flash Linear Attention (TFLA) 版本（基于 mlstm_kernels, Anonymous 2025）：引入第二层 tile 级并行（在 chunk 内的矩阵计算也进行 tiling），使 chunk size 可任意大而不再受限于 GPU SRAM，在 H100 上比 Flash Attention 和 Mamba kernel 更快。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 通用 Chunkwise 伪代码
For each chunk t (size C, head dim d):
  // chunk 内并行计算（dense matmul）
  Q, K, V = projections(X_chunk)
  // chunk 间通过 recurrent state S 传递
  S_chunk = f(S_prev, K, V)  // recurrent update
  O_chunk = g(Q, S_chunk, K, V)  // output computation
  
// xLSTM mLSTM 的 chunkwise 特有公式
// Eq. 2-9 的 chunk 形式：
// 1. chunk 内 gate 矩阵：用 cumsum 计算 chunk 内的 f_t 累积
// 2. 分子 C_t 更新：分 block-diagonal 和低秩两部分分别并行
// 3. 分母 n_t：向量递归，可用 parallel scan
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：Flash Linear Attention (FLA) https://github.com/fla-org/flash-linear-attention；xLSTM 专用 kernel 库 mlstm_kernels https://github.com/NX-AI/mlstm_kernels。PyTorch + Triton kernel 实现。Chunk size 通常 64-256。Gated DeltaNet 论文中该算法仅比 Mamba2 慢 2-3K tokens/sec（H100）。xLSTM 7B 训练基于 TFLA kernel，在 chunk 内对 outer product C_t 和 dot product 做 tiled 矩阵乘法以提升 arithmetic intensity 并降低 IO。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## WY Representation (for Householder Products)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WY 表示由 Bischof & Van Loan (1985) 提出，将 Householder 反射矩阵乘积 Π_i (I - β_i v_i v_i^T) 紧凑表示为 I - W Y^T。在 DeltaNet/Gated DeltaNet 中，每步 transition 为 Householder 形式 (I - β_t k_t k_t^T)，其累积乘积 P^r = Π_{i=1}^r (I - β_i k_i k_i^T) 可表示为 I - W K^T，其中 W = T K，T = (I + strictLower(diag(β) K K^T))^{-1} diag(β)。这使得 O(L·d²) 串行过程转为 O(C·d²) matmul 并行计算，是利用 Tensor Core 的关键。Gated DeltaNet 的扩展：在 T 矩阵计算中加入 Γ ⊙ K K^T 以融入 gating。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 原始：串行 Householder 链
P^r = Π_{i=1}^r (I - β_i k_i k_i^T)

// WY 表示后：
P = I - W K^T  // 其中 W = T @ K
T = solve_triangular(I + strictLower(diag(β) K K^T), diag(β))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
核心实现为 BLAS Level 3 的 triangular solve (trsm)。PyTorch 中用 torch.linalg.solve_triangular。使 DeltaNet/GatedDeltaNet 训练从不可行变为仅比 Mamba2 慢约 10%。Joffrain et al. (2006) 的 UT transform 进一步优化了表示计算。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---

## Hybrid Model (Linear RNN + Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Model 指在同一语言模型中混合使用不同 token mixer 层的架构策略。在 Gated DeltaNet 中，指将线性 RNN 层（Mamba2/GatedDeltaNet）与 Sliding Window Attention (SWA) 层混合。动机：线性 RNN 在局部模式建模和长程检索上有固有局限（state size 固定导致 memory collision），SWA 提供精确窗口内 attention 弥补局部模式缺陷；同时线性 RNN 提供 O(1) 推理效率。类似架构包括 Griffin（RG-LRU + local attention）、Samba（Mamba + SWA + SwiGLU MLP）、Hymba（hybrid-head attention+SSM）。SAMBA 论文是首个证明混合线性复杂度模型在大规模（3.8B）上能显著优于 SOTA Transformer 架构的工作。

SAMBA 中的 Hybrid 设计：层排列为 [Mamba → MLP (for Mamba) → SWA → MLP (for SWA)] 的 4 层 block 重复 N/4 次。关键设计：(1) Mamba 和 SWA 各有独立的 SwiGLU MLP，分别处理不同类型的信息——Mamba 的 MLP 处理压缩后的递归语义，SWA 的 MLP 处理精确检索信号；(2) SWA 窗口 2048，使用 RoPE（base=10,000）和 FlashAttention 2；(3) 训练序列长度 4096 = 窗口大小 × 2；(4) 注意力熵分析（Figure 5a）显示 Samba 中 SWA 层的注意力熵方差更大——中间层熵低（专注精确检索），顶层/底层熵高（整合全局信息），呈现专业化分工；(5) Mamba 的选择熵在混合架构的中间层更高（Figure 5b），说明有了 SWA 负责召回，Mamba 可以更专注于递归结构建模。消融研究（Table 5）表明：即使仅 1 层全注意力也无法外推至超训练长度（16K 时 perplexity 从 10.29 升到 13.66），而 SWA 可外推至 1M 且 perplexity 持续改善。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GatedDeltaNet-H1: [GatedDeltaNet, SWA, GatedDeltaNet, SWA, ...] 交替
GatedDeltaNet-H2: [Mamba2, GatedDeltaNet, SWA, Mamba2, ...] 三层最优顺序（消融验证）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SWA 层使用 FlashAttention-2 kernel，swa window 通常 2048。Hybrid 总训练吞吐量因 SWA 的高效 kernel 反而高于纯线性 RNN。层分配策略为均匀交替，attention 层比例通常 5-15%。适用于需要训练效率和长程性能最佳 balance 的大规模 LM。

GoldFinch采用不同的hybrid策略：前2/3层为Finch-C2 RNN层（O(1) per token），后1/3层为GOLD Transformer层（full MHA over compressed key cache）。关键创新：Finch-C2最终层输出被压缩为全局shared key cache（仅D/16 per token），所有GOLD层共享同一cache。这使得KV-cache从per-layer 2·d_model·n_layer降至(1+d_model/16)，约756-2550×压缩。Pre-fill仅需运行Finch-C2部分（O(1) per token），decoding时GOLD attention O(N)但仅在生成新token时运行（通常很短）。GoldFinch 1.45B在lambada ppl 48.2远优于Finch 81.9和Llama 71.7。

M1采用不同的hybrid策略：28层中6层保留为interleaved standard attention（~21%），22层替换为Mamba SSM层。动机：(1) 少量attention层提供关键的长程信息路由能力——完全去除attention会导致reasoning性能崩溃；(2) 22层Mamba提供O(1)推理效率和大batch吞吐量优势；(3) 通过MambaInLlama权重初始化+reverse KL蒸馏+分阶段SFT+GRPO RL实现跨架构推理能力迁移。M1-3B在数学推理benchmark上匹配DeepSeek-R1-Distill-Qwen-1.5B的性能，同时提供3x inference throughput（vLLM, H100, batch=512）。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## mLSTM (Matrix LSTM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
mLSTM (Matrix LSTM) 是 xLSTM 架构（Beck et al., 2024, NeurIPS 2024）的核心组件，将传统 LSTM 的标量 cell state c_t ∈ R 扩展为矩阵记忆状态 C_t ∈ R^{d_qk × d_hv}，通过 outer product 更新：C_t = f_t · C_{t-1} + i_t · (k_t ⊗ v_t)（⊗ 表示外积 v_t k_t^T），使记忆容量从标量提升到矩阵级别。其关键特性包括：(1) **全并行化训练**：由于递归的线性性质（C_t 更新为线性组合），可通过 chunkwise-parallel 模式训练，速度与 Flash Attention 相当甚至更快；(2) **常量推理记忆**：自回归生成时仅需 O(d_qk × d_hv) 的常量 GPU 内存，不随序列长度增长（vs Transformer 的 O(T) KV Cache）；(3) **指数门控**：标量输入门 i_t 和遗忘门 f_t 使用指数激活（i_t = exp(ĩ_t - m_t), f_t = exp(log σ(f̃_t) + m_{t-1} - m_t)），由 max state m_t 控制数值稳定性；(4) **Multi-head 结构**：类似 Transformer 的多头注意力，xLSTM 有 N_head = d/d_hv 个独立 mLSTM cell，每个 head 维护独立的 (C^(i), n^(i), m^(i))，输出拼接后投影。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// mLSTM Cell Recurrent Step (推理时)
输入: x_t ∈ R^d, 前一状态 (h_{t-1}, C_{t-1}, n_{t-1}, m_{t-1})
参数: W_{q,k,v} ∈ R^{d_{qkv}×d}, W_o ∈ R^{d_hv×d}, w_{i,f} ∈ R^d

// 1. 投影 (per head, head dim d_hv, d_qk = d_hv/2)
q_t, k_t = W_{q,k} @ x_t  // ∈ R^{d_qk}
v_t = W_v @ x_t          // ∈ R^{d_hv}

// 2. Gate pre-activations (scalars per head)
ĩ_t = w_i^T @ x_t + b_i  // input gate pre-activation
f̃_t = w_f^T @ x_t + b_f  // forget gate pre-activation
õ_t = W_o @ x_t + b_o    // output gate ∈ R^{d_hv}

// 3. Gate activations with max state stabilization
m_t = max(log(σ(f̃_t)) + m_{t-1}, ĩ_t)
f_t = exp(log(σ(f̃_t)) + m_{t-1} - m_t)
i_t = exp(ĩ_t - m_t)

// 4. Memory state update (outer product)
C_t = f_t · C_{t-1} + i_t · (v_t ⊗ k_t)  // C_t ∈ R^{d_qk × d_hv}
n_t = f_t · n_{t-1} + i_t · k_t          // n_t ∈ R^{d_qk}

// 5. Hidden state retrieval with normalization
q̃ = q_t / sqrt(d_qk)
h̃_t = C_t^T @ q̃ / max(|n_t^T @ q̃|, exp(-m_t))
h_t = σ(õ_t) ⊙ Norm(h̃_t)

// Multi-head: concat all heads, project
H = Concat(h_t^(1), ..., h_t^(N_head)) @ W_proj^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：
- 官方 PyTorch: https://github.com/NX-AI/xlstm
- 官方 JAX: https://github.com/NX-AI/xlstm-jax
- Triton kernel 库: https://github.com/NX-AI/mlstm_kernels
- 训练时使用 chunkwise-parallel kernel（Tiled Flash Linear Attention），将序列分块、块内 tiled matmul（利用 Tensor Core）、块间通过 recurrent state 传递。
- 推理时使用 recurrent mode（单个 kernel 即可完成 Eq. 2-9 全部计算）或 TensorRT-LLM 部署。
- xLSTM 7B 配置：8 heads, d_hv=512, d_qk=256, d=4096, 32 blocks, 总记忆状态 134.2 MB（float32）。

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## Post-up Projection Block

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-up Projection Block 是 xLSTM 7B 引入的 block 架构设计，将 mLSTM cell 直接放在 embedding 维度（d_model）运行，并在 mLSTM 层后接独立的 position-wise SwiGLU MLP。与 Mamba 和早期 xLSTM 的 *pre-up projection block*（先 up-project 到更高维度 -> mLSTM/mamba -> down-project 回 embedding 维度，无独立 FFN）形成对比。Post-up 设计的动机：(1) mLSTM 操作的计算量和 GPU 内存消耗随维度线性增长，因此在较低维度（embedding dim）运行可大幅减少开销；(2) 添加独立 SwiGLU MLP 增加了高度优化的线性层（矩阵乘法）FLOPs 占比（Tensor Core 利用率更高）；(3) 丢弃 channel-wise convolution 和 learnable skip connection 等小 kernel 调用，避免 GPU 利用率下降。xLSTM 7B 的 32 个 block 均采用此设计，获得 3.5× 训练速度提升（1.4B 参数规模），且不影响下游任务性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Post-up Projection Block (xLSTM 7B)
输入: x ∈ R^{T×d}  (d = 4096)

// Block 内两层：
// Layer 1: mLSTM (sequence mixing, 在 embedding dim 运行)
x_norm = RMSNorm(x)
x_mix = mLSTM(x_norm)       // multi-head mLSTM cells, 每 head 维度 d_hv=d/N_head
z = x + x_mix                // 残差连接

// Layer 2: Gated MLP (channel mixing)
z_norm = RMSNorm(z)
z_mlp = SwiGLU(z_norm)      // SwiGLU: x ⊙ σ(W_gate @ x) 经 W_up 投影
y = z + z_mlp                // 残差连接

// 对比 Pre-up Projection Block:
// x → UpProj(x) ∈ R^{factor×d} → Conv → mLSTM/SSM → DownProj → output
// (无独立 MLP 层)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现位于 https://github.com/NX-AI/xlstm
- SwiGLU MLP 使用 projection factor 2.66（常见于 Transformer），2 个 linear 层（gate + up）后接 SiLU 激活和 element-wise 乘，再过 1 个 linear down-project
- Norm 层使用 RMSNorm（pre-norm 设置）
- 32 个 block 堆叠，总参数 6.87B
- 该设计同时提高了推理吞吐：xLSTM 7B 在 H100 上推理比 Falcon-Mamba 和 Codestral-Mamba 快约 50%

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## Exponential Gating (in xLSTM/mLSTM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Exponential Gating 是 xLSTM 架构中 mLSTM cell 使用的门控机制，与传统 LSTM 的 sigmoid/tanh 门控不同。mLSTM 的输入门 i_t ∈ R 和遗忘门 f_t ∈ R 是标量（per head），使用指数函数激活而非 sigmoid：(1) **遗忘门** f_t = exp(log σ(f̃_t) + m_{t-1} - m_t)，结合 sigmoid 的对数和 max state 的差值来得到指数形式；(2) **输入门** i_t = exp(ĩ_t - m_t)，直接通过指数激活。这种设计允许门值超出 [0,1] 范围，为记忆状态更新 C_t = f_t·C_{t-1} + i_t·(v_t⊗k_t) 提供更灵活的缩放。**max state** m_t = max(log σ(f̃_t) + m_{t-1}, ĩ_t) 用于数值稳定，防止指数溢出。指数门控使 mLSTM 能在大范围值上进行记忆更新，这是其与 SSM/Mamba 的关键区别（Mamba 使用 selective scalar gating 但无独立的输入/遗忘门对）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Exponential Gating 的完整流程（per head, per timestep）
// Gate pre-activations (应用 soft-capping a=15)
ĩ_tilde = softcap_15(w_i^T @ x_t + b_i)  // scalar
f̃_tilde = softcap_15(w_f^T @ x_t + b_f)  // scalar

// Max state update (数值稳定关键)
m_t = max(log(σ(f̃_tilde)) + m_{t-1}, ĩ_tilde)

// Exponential gate activations
f_t = exp(log(σ(f̃_tilde)) + m_{t-1} - m_t)  // ∈ (0, 1] 实际
i_t = exp(ĩ_tilde - m_t)                       // ∈ (0, ∞)

// Memory update
C_t = f_t · C_{t-1} + i_t · (v_t ⊗ k_t)

// 为什么用指数？相比 sigmoid:
// - sigmoid: i_t ∈ (0,1), 值域受限, 不能"超量"写入
// - exponential: i_t 可 >1, 允许新信息以更高权重写入记忆
// - f_t 始终 ≤1 (因 m_t 定义), 实现稳定遗忘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现在 https://github.com/NX-AI/xlstm（PyTorch）和 https://github.com/NX-AI/xlstm-jax（JAX）
- 重要实现细节：
  - 输入门 bias 初始化为 -10（大的负值），使初始状态 i_t ≈ exp(-10) ≈ 0，训练初期模型依赖前一步记忆而非新输入，有效降低早期梯度尖峰
  - Gate pre-activations 使用 softcap_a(x) = a·tanh(x/a)（a=15 用于 gates, a=30 用于 logits）
  - 在 EOD token 处通过置 f_t = 0 使完整重置记忆（序列打包时防止跨文档信息泄露）
- 与 GLA (Gated Linear Attention) 的关系：两者都使用门控线性递归，但 mLSTM 使用标量指数门（per head）而非向量门

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## Gate Soft-Capping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gate Soft-Capping 是 xLSTM 7B 用于训练稳定性的技术，对 mLSTM 的输入门和遗忘门 pre-activations 应用 tanh-based 软上限：softcap_a(x) = a · tanh(x/a)。其中 cap value a=15 用于 gate pre-activations（输入/遗忘门），a=30 用于输出 logits。该函数在输入值接近 0 时近似线性（梯度≈1），在 |x| 远大于 a 时渐近饱和于 ±a（梯度≈0）。与 hard clipping（直接截断）不同，soft-capping 提供了平滑的饱和行为，不会产生零梯度区域，在抑制异常值的同时保持了可训练性。xLSTM 7B 在 160B token 消融实验中证实：无 soft-capping 的训练表现出更高的梯度 norm 方差和更差的验证 loss。类似技术也用于 Gemma-2 模型（logit soft-capping）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Soft-Capping 函数
def softcap_a(x):
    return a * tanh(x / a)

// 性质分析:
// |x| << a: tanh(x/a) ≈ x/a → softcap_a(x) ≈ x (线性区域, 梯度≈1)
// |x| >> a: tanh(x/a) ≈ sign(x) → softcap_a(x) ≈ ±a (饱和, 梯度≈0)
// x = 0:    softcap_a(0) = 0, 梯度 = 1
// x = a:    softcap_a(a) = a·tanh(1) ≈ 0.762a, 梯度 = sech²(1) ≈ 0.42

// 在 xLSTM 7B 中的应用
// 每层 mLSTM 的 gate 计算:
ĩ_sc = softcap_15(w_i^T @ RMSNorm(x) + b_i)   // input gate
f̃_sc = softcap_15(w_f^T @ RMSNorm(x) + b_f)   // forget gate

// 最终输出 logit:
logits_sc = softcap_30(W_lm_head @ h_final)  // logit soft-capping

// vs hard clipping:
// clip(x, -c, c) 在边界处梯度为 0，可能导致 dead neuron
// softcap 在边界处梯度非零但很小，允许缓慢逃离饱和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 实现：`a * torch.tanh(x / a)`
- 需要放在 gate pre-activation 计算之后、指数激活之前（对 gate）或 logit 输出之前（对 logit）
- 典型配置：gate cap=15, logit cap=30（xLSTM 7B）；Gemma-2 使用类似配置
- 适用于任何使用指数或大范围值门控的递归架构（线性 RNN、SSM、LSTM variants）
- 注意：与 LayerNorm/RMSNorm 互补——Norm 在统计层面稳定激活分布，soft-capping 在个体值层面防止极端异常值

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## RMSNorm (Root Mean Square Layer Normalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RMSNorm (Root Mean Square Layer Normalization, Zhang & Sennrich, 2019) 是一种简化版的 Layer Normalization，去除了均值中心化（mean subtraction），仅通过 root mean square 统计量进行缩放：RMSNorm(x) = x / RMS(x) ⊙ g，其中 RMS(x) = sqrt(mean(x²) + ε)，g 为可学习增益参数。相比 LayerNorm（需计算均值和方差），RMSNorm 省去了均值计算，在 GPU 上约快 7-15%。Transformer 架构中（LLaMA、Qwen、Gemma 等）和 SSM/RNN 架构中普遍采用。xLSTM 7B 的实验证实（Fig. 9, App. C.2）：使用 LayerNorm 作为 pre-norm 在 1.4B 参数规模导致极大的梯度 norm 和验证 loss 发散，而 RMSNorm 训练稳定。对于 head-wise state norm（Eq. 6），RMSNorm 和 LayerNorm 均表现稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// RMSNorm 前向计算
输入: x ∈ R^d (沿最后一维归一化)
参数: g ∈ R^d (可学习增益), ε = 1e-6 (通常)

rms = sqrt(mean(x²) + ε)   // 仅需一次平方+均值+sqrt
output = (x / rms) ⊙ g      // element-wise 缩放

// 对比 LayerNorm:
μ = mean(x)                  // 额外计算均值
σ² = mean((x - μ)²)         // 需先减均值再平方
output = (x - μ) / sqrt(σ² + ε) ⊙ g + b  // 额外 bias 参数

// xLSTM 7B 中 RMSNorm 的使用位置:
// 1. Pre-norm: 每个 block 进入 mLSTM 前: z_norm = RMSNorm(z)
// 2. Pre-norm: 每个 block 进入 SwiGLU MLP 前: z_norm2 = RMSNorm(z2)
// 3. Head-wise state norm 仍用 LayerNorm (Eq. 6 中的 Norm(h̃_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 无原生 RMSNorm，常用实现：
  - LLaMA-Factory/Torch: 自定义 `RMSNorm` 继承 `nn.Module`
  - HuggingFace: 多数 LLM 使用 `LlamaRMSNorm` 或其他等价实现
- 适合替代任何网络中的 LayerNorm，尤其大规模训练中对速度敏感的场景
- 对于递归架构（xLSTM/Mamba/RWKV），推荐在 pre-norm 位置使用 RMSNorm 以获得更好的训练稳定性
- 在 flash-attention 或 fused kernel 中可直接将 RMSNorm 融合进前一层的 output 计算

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---

## Memory Collision in Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Collision 是线性 Transformer/RNN 的核心瓶颈（Schlag et al., 2021）。线性 Transformer 状态 S ∈ R^{d×d} 通过外积 v k^T 存储 key-value 关联，最大可存储的正交 key-value 对数受维度 d 限制。当序列长度 L > d 时，新 key-value 无法与已有对正交存储，信息在有限状态空间中叠加导致"碰撞"，使精确检索不可能。缓解策略：(a) Gating/Forgetting（Mamba2 α_t、RWKV w_t）—主动遗忘不相关信息；(b) Delta Rule—精确替换而非叠加；(c) State Expansion（Eagle/Finch head size > 1）—扩大实际存储容量；(d) Hybrid—混合 attention 提供精确检索。Gated DeltaNet 展示了 (a)+(b) 的组合是最优策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
S_t = Σ_{i=1}^t v_i k_i^T ∈ R^{d×d}
检索: o_t = S_t q_t = Σ v_i (k_i^T q_t)
若 k_i 正交，最多存 d 个独立 key-value 对；t > d 时必然 collision
```

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---

## S-NIAH (Single Needle In A Haystack)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
S-NIAH 是 RULER benchmark (Hsieh et al., 2024) 的合成检索评测套件，包含三个难度递进任务：S-NIAH-1 (passkey retrieval)：合成上下文中放置 key-value "needle"，测试纯长期记忆保持（上下文不含其他有意义信息）；S-NIAH-2 (number in haystack)：真实文章上下文中放置数值 needle，测试选择性记忆+噪音过滤；S-NIAH-3 (UUID in haystack)：needle 的 value 是 UUID（复杂模式），测试复杂模式记忆。评测不同序列长度（1K-8K+）下的检索准确率。核心诊断能力：S-NIAH-1 测"记忆保持"，S-NIAH-2/3 测"选择性记忆+过滤"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
S-NIAH-1 格式: "The special magic number is 12345. [repeat filler × N] What is the magic number?" → Target: "12345"

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RULER 开源：https://github.com/hsiehjackson/RULER。Gated DeltaNet 论文利用该 benchmark 验证 gating 与 delta rule 互补性：DeltaNet S-NIAH-1 完美但 S-NIAH-2/3 崩溃（缺遗忘），Mamba2 S-NIAH-2/3 较好但 S-NIAH-1 长序列崩溃（过度遗忘），Gated DeltaNet 在所有任务上最佳平衡。适用于评测新线性 RNN 架构的记忆保持与选择性记忆能力。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---

## Short Convolution in Token Mixer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Short Convolution 是现代线性 RNN/SSM token mixer 中广泛使用的轻量级局部混合组件，典型配置为 kernel size=4 的 causal depthwise 1D 卷积。在 Gated DeltaNet 中，q/k/v 路径在核心计算前先经：Linear Proj → ShortConv(kernel=4) → SiLU → (可选 L2 norm)。其作用是为 token mixer 提供最近 4 个 token 的局部上下文感知，弥补线性 RNN 状态压缩可能丢失的细粒度局部模式。类似设计见于 Mamba/Mamba2（conv window=4）、HGRN/HGRN2 等。Gated DeltaNet 消融（Table S.1）证实移除 short conv 导致 perplexity +1.6（27.35→28.95）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x_conv[t] = Σ_{i=0}^{3} w_i · x_proj[t-i]  // causal depthwise 1D conv, kernel=4
x_act = SiLU(x_conv)
q_t = L2Norm(x_act)  // 对 query/key 额外加 L2 norm
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通常用 nn.Conv1d(groups=D) 实现 depthwise causal conv。参数量仅 kernel_size × D，相比主模型参数的 D×D 可忽略。多篇论文消融一致显示移除导致 1-2 ppl 下降，是 token mixer 中性价比最高的组件之一。

SAMBA 论文对 Short Convolution 的消融分析（Table 10）：将 SC 添加到不同线性递归模型中效果不同：(1) SC + SWA：perplexity 从 11.12→10.83 显著改善，说明 depthwise conv 的局部平滑对所有 token mixer 都有益；(2) SC + Sliding GLA：改善不显著（10.43→10.39），因为 GLA 已有 channel 级细粒度衰减，depthwise conv 未增加额外有用的归纳偏置；(3) SC + Sliding RetNet：改善明显（10.38→10.25），弥补了 RetNet 固定衰减的灵活度不足；(4) 在 hybrid 模型中同时给 SWA 和线性注意力层加 SC 反而产生负面效果。这些发现验证了 SC 的通用价值，但也揭示其效果依赖于 token mixer 的现有表达能力。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## Fast Weight Programming

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast Weight Programming (Schmidhuber, 1992; Irie et al., 2021/2022) 是一种区分 "slow weights"（传统梯度下降学习的参数）和 "fast weights"（推理时动态更新的参数）的框架。在 DeltaNet/Gated DeltaNet 中，隐藏状态 S_t 被解释为 fast weight matrix，每步通过 delta rule（等价于在线回归的 SGD 更新）修改：S_{t+1} = S_t - β_t ∇L(S_t)，L(S_t) = 1/2 ||S_t k_t - v_t||²。α_t（forget gate）等价于 adaptive weight decay，β_t 等价于 adaptive learning rate。Gated DeltaNet 的每次前向传播被理解为对 fast weight 执行一步含 weight decay 的 SGD 更新。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Loss(S_t) = 1/2 ||S_t k_t - v_t||²           // 在线回归目标
∇L = (S_t k_t - v_t) k_t^T                    // 回归梯度
S_{t+1} = α_t S_t - β_t ∇L                    // α_t: weight decay, β_t: LR
        = S_t(α_t I - α_t β_t k_t k_t^T) + β_t v_t k_t^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该视角的价值在于理论统一（将 Mamba2/DeltaNet/GatedDeltaNet/Longhorn/TTT/Titans 统一在 online learning 下）和设计指导（可通过改进优化器系统地设计新架构）。TTT 和 Titans 在此基础上探索了非线性回归和多步更新。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---

## Finch-C2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch-C2是GoldFinch论文中提出的Finch (RWKV-6)时间混合器改进版，作为GoldFinch架构前2/3层的核心组件。四项改进：(1) 移除Gate（SiLU gating），用新的数据依赖第二Value (u'_t) 补偿性能损失，减少参数量；(2) 将per-head GroupNorm替换为跨所有head的LayerNorm；(3) key乘以(1-w_t)以保持kv-state行归一化（受HGRN2启发，HGRN2设key=1-decay，Finch-C2则乘而非设等）；(4) u'_t = u_t W^V + tanh(u_t W^{UD})W^{UU}，数据依赖的独立token-shifted第二Value，复用W^V权重（intentional参数节省）。Finch-C2在减少参数的同时性能略优于原Finch（消融L12 D768: Finch-C2 loss=2.7082 < Finch loss=2.7191）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Finch-C2 Time Mixing (per-head):
d_t = lora_d(ddlerp_d(x_t, x_{t-1}))                  // data-dependent decay factor
w_t = exp(-exp(d_t))                                    // decay weight ∈ (0,1)
r_t = ddlerp_r(x_t, x_{t-1}) @ W^R                     // receptance
k_t = ddlerp_k(x_t, x_{t-1}) @ W^K · (1 - w_t)        // key × (1-decay) [创新]
v_t = ddlerp_{i,i}(x_t, x_{t-1}) @ W^V                 // first value
u_t = ddlerp_u(x_t, x_{t-1})                            // bonus raw
u'_t = u_t @ W^V + tanh(u_t @ W^{UD}) @ W^{UU}         // second value [创新]

// WKV linear attention:
wkv_t = diag(w_t) @ wkv_{t-1} + k_t^T @ v_t           // matrix state update
o_t = LayerNorm(concat(r_t @ wkv_t + u'_t)) @ W^O      // LayerNorm across heads [创新]
```
vs Finch原始: k_t无(1-w)乘积，gate applied to o_t (SiLU gating)，GroupNorm per head。Finch-C2可独立使用也可组合为GoldFinch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch-C2每Time Mixing层5组LoRA (r/k/v/u rank-32, d rank-64)。Channel Mixing与Finch完全一致（lerp token shift + ReLU² FFN + σ(r) gate）。代码开源：https://github.com/recursal/GoldFinch-paper (Apache 2.0)。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## Set Disjointness in Communication Complexity for LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Set Disjointness (SD)是通信复杂度理论的经典问题：两方各持集合A和B，需通过最少通信判断A∩B是否为空，Ω(n)通信下界为领域基础结果(Chattopadhyay & Pitassi, 2010)。JRT论文将SD与语言模型的associative recall建立等价：循环模型可视为处理A和B集合的streaming algorithm，需在固定内存中存储足够信息以判断交集。Causal模型存储需求为Ω(min(|A|,|B|))——小集合在前内存需求最小，即"正确数据顺序"的理论依据。JRT-Prompt将下界降至Ω(min(|A|,|B|)/p)。纯卷积架构(BaseConv)无法从JRT获益(Theorem G.6/G.7/G.11)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// SD合成任务(Algorithm 1):
Input: [prefix_token], A, [sep_token], B, [answer_token], [t]
  A/B: random tokens from disjoint halves of vocab |V|=2048
  t: intersecting token from A also inserted into B
Output: predict t

// 理论(Theorem 3.2):
// Based(BaseConv+MLP+LA+MLP) in JRT-prompt以O(min{|A|,|B|}·n)空间解SD
// n=token bit width, |A|,|B|∈{1..1024}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
JRT论文构建合成SD任务训练4层Based模型，sweep model dim∈{36..128}和feature dim∈{4..24}控制state size，评估不同数据顺序下准确率。代码基于Zoology合成仓库(https://github.com/HazyResearch/zoology)。训练数据：20000×12种(|A|,|B|)组合(mixture)。评估时需长度外推至训练未见的序列长度组合。该框架适用于分析新循环架构的memory-recall tradeoff和设计prompting策略。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## Associative Recall in Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Associative Recall (AR)是语言模型在上下文中识别和使用之前出现过的key-value对的能力。Arora et al.(2023, Zoology)将AR定义为与ICL质量高度相关的核心技能：若某token完成的bigram在上下文中之前出现过，且该bigram在训练中罕见(未被memorize)，则模型须依靠上下文而非参数知识预测。AR形式化为MQAR任务——上下文中多个(key,value)对，给定key预测value。Transformer通过attention的O(N²)全局匹配天然擅长AR；固定memory循环模型需在有限state中选择性存储，成为recall-intensive ICL的主要瓶颈。JRT论文通过Pile perplexity slicing (AR slice vs Other slice)量化各架构的AR能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// AR基本形式:
Context: "In 1957, Dr. Seuss wrote ... In 1982, Dr. ___"
// "Dr. Seuss" 是AR bigram——预测"Seuss"需回忆前面出现的"Dr."

// Pile AR slice定义(JRT Section 5.2):
// "AR hit": token完成一个训练中低频(<1000次)的re-occurring bigram
// 按bigram frequency和distance分组分析perplexity
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
JRT论文使用10M Pile训练文档计数bigram频率，在3200 seq/2048 token的Pile test上计算最后1024 token/seq的AR/Other slice perplexity。JRT-RNN在AR slice显著优于causal decoder-only baseline，在Other(non-recall) slice略差(因MLM训练仅见65% NTP tokens)。MQAR评测代码集成在LM-Eval Harness和RULER benchmark中。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## Taylor Feature Map for Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Taylor Feature Map是基于Taylor级数展开近似softmax-exponential的确定性特征映射：exp(x)≈1+x+x²/2!+...+x^k/k!，截断到k阶产生多项式核φ(q)^Tφ(k)=Σ_{m=0}^{k}(q^Tk)^m/m!。区别于Performer的随机傅里叶特征(RBF kernel)，Taylor map是确定性的无随机投影噪声。Based架构(Zhang et al. 2024, Hedgehog & Porcupine)使用2阶Taylor近似：φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2。二阶项产生高维展开d̃≈273(for base dim d=16)，使IO-aware kernel管理warp-register分片的KV-state矩阵关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
exp(α) ≈ 1 + α + α²/2    (2nd-order Taylor at α=0)

// Based feature map (可分离):
φ_1(q) = q / sqrt(sqrt(d))              // 一阶项
φ_2(q) = vec(q ⊗ q) / sqrt(2) / sqrt(d) // 二阶项, d(d+1)/2 dim
φ(q) = concat(1, φ_1(q), φ_2(q))       // 总dim ≈ 1+d+d(d+1)/2

// JRT/PLA使用相同feature map，encoder和decoder各自独立投影矩阵
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Based实现中feature dim d=16, 总展开dim d̃≈273。IO-aware CUDA kernel(ThunderKittens)将KV-state(R^{d×273})分片存储在warp registers中，A0/A1/A2寄存器分别存0/1/2阶项贡献。JRT论文的所有模型(JRT-Prompt, JRT-RNN)均使用Based的Taylor feature map。该映射也可被PLA、causal LA等任何线性注意力变体使用。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## JRT-Prompt (Just Read Twice Prompting)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
JRT-Prompt是将context(C)和question(Q)在prompt中重复两次后生成答案的ICL prompting策略：Ŷ=A(C,Q,C,Q)。动机源于SD通信复杂度分析——causal模型从左到右处理时小集合在前则存储需求最小。重复让模型在第二轮condition on完整prompt后决定存储什么，等价于展示所有数据顺序。无需模型修改或训练，off-the-shelf循环LM直接可用。16模型×6 ICL任务平均+11.0±1.3点。理论下界从Ω(max(|A|,|B|))降至Ω(min(|A|,|B|)/p)。N=32768/B=16/H100上11.9×于FA2的prefill吞吐量(sub-quadratic架构2N仍快于attention的N)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Standard ICL: A("Doc... Q?") → Ŷ          // Q前必须预测存什么
JRT-Prompt:   A("Doc... Q?... Doc... Q?") → Ŷ  // 第二轮有完整view
```
仅改prefill(context翻倍)，decode不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过LM-Eval Harness将prompt中(C,Q)拆出并重复拼接，不做任务特定定制。对off-the-shelf循环LM(Mamba, Based, GLA, Mamba-2)直接可用。缺点：重复可能增加Repetition errors；context翻倍增加prefill计算。但sub-quadratic架构效率优势仍显著。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---

## TokenCat (Token ConCatenation for KV-Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TokenCat是GoldFinch的KV-Cache压缩/解压机制，通过两步将Finch-C2层输出压缩至极小全局key cache。第一步（压缩）：取Finch-C2最终层输出x_t∈R^D，乘全局矩阵W^{KD}∈R^{D×(D/16)}压缩为c_t=x_t·W^{KD}∈R^{D/16}（16:1压缩），每token仅需D/16元素存储。第二步（解压）：拼接压缩key c_t与原始embedding x_t^0为concat(x_t^0,c_t)∈R^{D+D/16}，乘全局矩阵W^{KU}∈R^{(D+D/16)×D}并RMSNorm得proto-keys k_t^D供所有GOLD层共享。类似LoRA低秩分解思路。16:1 vs 1:1压缩loss差异可忽略（均为2.2762），验证几乎无损。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 压缩 (after last Finch-C2 layer):
c_t = x_t @ W^{KD}           // W^{KD} ∈ R^{D×(D/16)}, global matrix

// 解压 (shared by all GOLD layers):
x_t^0 = embedding_lookup(idx_t)
k_t^D = RMSNorm(concat(x_t^0, c_t) @ W^{KU})  // W^{KU} ∈ R^{(D+D/16)×D}, global

// KV-Cache size:
// Traditional: 2·d_model·n_layer·ctx_len → Llama 256k ctx=128GB
// GoldFinch: (1+d_model/16)·ctx_len → 256k ctx=0.068GB (D=4096)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W^{KD}和W^{KU}为全局参数（非per-layer），所有GOLD层共享同一proto-keys。C_t和idx_t常驻VRAM，keys on-the-fly解压。支持增量解压以降低VRAM峰值。开源实现：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## GOLD (GPTAlpha Over Linear transformer Decoder)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GOLD是GoldFinch后1/3层的Transformer变体，基于GPTAlpha但移除per-layer W^K和W^V权重，改为从TokenCat解压的proto-keys和原始embedding生成k/v。GPTAlpha是独立改进版Transformer：将Llama SwiGLU FFN替换为Finch Channel Mixer (RWKV FFN)，attention层添加ddlerp token shift和额外LayerNorm。GOLD = GPTAlpha - W^K - W^V + TokenCat输入 + DDLoRAdapt。由于无per-layer K/V权重，所有GOLD层共享同一压缩key cache，无需per-layer K cache和value cache。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// GOLD Attention (per layer, per head):
q_t = LayerNorm(ddlerp_q(x_t, x_{t-1}) @ W^Q)           // query: per-layer W^Q
a_t = lerp(x_t^0, x_{t-1}^0, μ_a)                        // embedding token-shift data
k_t = LayerNorm(loradapt_k(lerp(k_t^D, k_{t-1}^D, lora_k(a_t))))  // key from proto-keys
v_t = LayerNorm(loradapt_v(lerp(x_t^0, x_{t-1}^0, lora_v(a_t))))  // value from embeddings
o_t = LayerNorm(concat(attention(q_t, K_{1:t}, V_{1:t}))) @ W^O

// vs GPTAlpha standalone (has per-layer W^K, W^V):
k_t = LayerNorm(ddlerp_k(x_t, x_{t-1}) @ W^K)
v_t = LayerNorm(ddlerp_v(x_t, x_{t-1}) @ W^V)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTAlpha可独立使用（ablation中GPTAlpha+RoPE loss=2.6684，vs Llama 2.7125，L12 D768）。GOLD专为GoldFinch hybrid设计。K/V重建利用token shift的隐式位置信息（Finch-C2 RNN自动编码位置），训练context内无需显式位置编码；需extrapolation时可选RoPE。开源于https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## DDLoRAdapt

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DDLoRAdapt (Data-Dependent LoRA Adaptation) 是GoldFinch在GOLD Attention中使用的参数高效token shift增强。定义为loradapt_□(x)=x+tanh(xC_□)D_□，C_□∈R^{H×r}、D_□∈R^{r×H}为低秩矩阵。与标准ddlerp（乘性插值：a+(b-a)⊙lora(...)）不同，DDLoRAdapt是加性的：在输入上叠加低秩tanh偏移。用于GOLD层从共享proto-keys和embedding生成层特异的k/v，使所有GOLD层共享压缩cache时仍学习不同attention模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
loradapt_□(x) = x + tanh(x @ C_□) @ D_□      // rank r, additive LoRA offset

// GOLD key with DDLoRAdapt:
a_t = lerp(x_t^0, x_{t-1}^0, μ_a)                     // embedding shift data
k_t_raw = lerp(k_t^D, k_{t-1}^D, lora_k(a_t))          // token-shifted proto-keys
k_t = LayerNorm(loradapt_k(k_t_raw))                   // DDLoRAdapt: layer-specific adaptation

// vs standard ddlerp (multiplicative):
ddlerp(a,b) = a + (b-a) ⊙ lora(a+(b-a)⊙μ_x)           // elementwise product with ratio
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
每GOLD层分别应用loradapt_k和loradapt_v。C/D rank推测≤32，初始化≈0使初始行为接近无adaptation。代码：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## Compressed Global KV-Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Compressed Global KV-Cache是混合RNN-Attention模型中通过三项设计实现极致缓存压缩的技术：(1) Global共享——所有attention层共享RNN最终层输出的单份压缩key cache（消除n_layer因子）；(2) 无Value Cache——value不缓存而由原始embedding按需重建（仅存token index≈2 bytes/token）；(3) Key低秩压缩——16:1压缩比（D→D/16），通过W^{KD}+TokenCat编码-解码。总cache=(1+D/16)元素per token。GoldFinch达到756-2550× cache缩小，256K context仅0.068GB vs Llama 128GB。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Cache存储:
store: [c_t ∈ R^{D/16}, idx_t]  per token  // 总: (D/16+1) elements

// 解压重建:
K_all[t] = RMSNorm(concat(emb[idx_t], c_t) @ W^{KU})   // from compressed cache
V_all[t] = emb[idx_t]                                   // from token indices

// Size对比 (256k ctx, D=4096, 32 layers):
// LLlama:    2·4096·32·256K·2 = 128 GB
// GQA(8g):   8·128·32·256K·2   = 16.8 GB
// YOCO:      2·4096·256K·2     = 4 GB
// GoldFinch: (1+256)·256K·2    = 0.068 GB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解压和token shift可在contiguous region上增量执行降低VRAM峰值。长context fine-tuning仅更新GOLD层（冻结Finch-C2），约3× FLOPs节省。推理pre-fill O(1) per token（仅Finch-C2），decoding O(N)但通常很短。开源：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---

## SUPRA (Scalable UPtraining for Recurrent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SUPRA 是 Mercat et al. (2024, COLM) 提出的将预训练 softmax Transformer 大规模转换为线性 RNN 的方法。核心操作：(1) 用可学习 MLP kernel φ(x)=ReLU(Wx+b)（Q/K 共享 W）替换 softmax；(2) 用 GroupNorm 替换传统线性注意力的分母除法归一化，解决大规模数值不稳定性；(3) 引入 RoPE 相对位置编码；(4) 使用固定衰减向量 γ∈(0,1)^h。最终注意力形式：v'_i=GroupNorm(Σ_{j=1}^{i} γ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)。训练仅需约 5% 预训练 tokens（20B-100B），基于 OpenLM + Lightning Attention 2 Triton kernel，PyTorch FSDP + H100 集群。支持 Llama2 和 Mistral 作为 base model。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SUPRA uptraining 流程:
model = load_pretrained("Mistral-7B")
# 添加 MLP kernel: W_phi, b_phi (Q/K 共享)
for layer in model.layers:
    layer.W_phi = Linear(D, D, bias=True)

# 替换 attention + uptraining 5% tokens
def supura_attention(q, k, v):
    phi_q = RoPE(ReLU(q @ W_phi + b_phi))
    phi_k = RoPE(ReLU(k @ W_phi + b_phi))
    out = lightning_attn_ops(phi_q, phi_k, v, gamma)  # Triton kernel
    return GroupNorm(num_heads)(out)

# 推理切换为循环模式: s_i = diag(γ)·s_{i-1} + φ(k_i)·v_i^T, O(1) per token
```

关键洞察：不同于 T2R 的"近似 softmax"策略，SUPRA 直接替换 attention 机制，通过 uptraining 让模型学习新的计算范式。Appendix A 热力图证实 SUPRA 和 softmax 的 attention 矩阵差异很大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/TRI-ML/linear_open_lm (MIT)，模型：Mistral-SUPRA (https://huggingface.co/TRI-ML/mistral-supra)。7B 模型在 128 H100 GPU 上 uptraining 约 1.5 天。Mistral-SUPRA +100B tokens avg 64.0（vs Mamba-7B 1.2T tokens avg 64.7），仅 5% 训练成本。局限性：MMLU 大幅退化（34.2 vs Mistral 62.4），ICL 能力丧失（线性模型的已知瓶颈）。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## Uptraining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Uptraining 由 Ainslie et al. (2023) 首次使用，指在修改模型架构后继续语言建模训练——区别于 fine-tuning（通常在不同数据集上继续训练）。在 SUPRA 中，uptraining 具体指：在 softmax Transformer attention 层中添加 MLP kernel 参数后，在相同预训练语料（RefinedWeb）上继续训练约 5% 原始 tokens，同时更新新增参数和原有参数。关键洞察：不追求近似 softmax（T2R 策略），而是直接替换 attention 让模型适应新范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Uptraining vs Fine-tuning vs Pre-training:
- Pre-training: 随机初始化 → 完整训练 (1-8T tokens)
- Uptraining:  预训练模型 → 修改架构 → 继续训练 (5% tokens, 同数据集)
- Fine-tuning: 预训练模型 → 新数据集/任务 → 少量训练

SUPRA uptraining: Mistral-7B (8T tokens) → 添加 MLP kernel → RefinedWeb 100B tokens
LR: 3e-5→1e-5 cosine, 1000步 warmup, Adam (β1=0.9, β2=0.95), seq=2048, H100+FSDP
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Uptraining 的收益来自 base model 的质量：Mistral-SUPRA avg 64.0 vs Llama2-SUPRA avg 58.6（更高质量预训练数据带来持续优势）。Uptraining 需要低于预训练的 LR 以保持已学知识。局限性：Instruct-tuned 模型线性化效果差于 base model；继承 base model 的 biases。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## MLP Kernel for Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLP Kernel 是线性注意力中替代固定非线性特征图（如 ELU+1）的可学习特征映射。SUPRA 中定义为 φ(x)=ReLU(Wx+b)，W∈R^{D×D} 在 queries 和 keys 间共享。比传统固定 kernel 更强大——MLP 可学习适合特定任务的特征表示。T2R 也使用类似 MLP kernel，但追求近似 softmax；SUPRA 直接替换，学到与 softmax 完全不同的 attention 模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 固定 kernel (Katharopoulos 2020):
phi(x) = ELU(x) + 1, sim(q,k) = phi(q)·phi(k)

# MLP kernel (SUPRA):
phi(x) = ReLU(W @ x + b)  # W 共享于 Q/K
phi_q = RoPE(phi(q)), phi_k = RoPE(phi(k))
sim(q,k) = phi_q · phi_k
```

参数开销：每 head 的 W∈R^{d_h×d_h}，总约 D²/h per layer。7B 模型（D=4096, h=32, d_h=128）：128²×32×32≈16M 总参数（~0.2%）。W 初始化为接近小随机值，使初始 φ(x)≈ReLU(x)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 OpenLM fork (https://github.com/TRI-ML/linear_open_lm)。Square matrices with biases，保持 Q/K 特征维度不变。共享 W 保证相似度的对称性。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## GroupNorm in Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GroupNorm (Wu & He, 2018) 被 SUPRA 创新性地用于线性注意力的输出归一化，替代传统分母除法。传统线性注意力 v'_i = Σ sim·v / Σ sim 的分母可能数值发散或趋零。SUPRA 用 GroupNorm(num_groups=num_heads) 在每个 head 的 WKV 输出上做独立归一化（减均值除标准差），实现：(1) 数值稳定；(2) 无需维护额外归一化状态；(3) 保持 head 独立性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 传统线性注意力（不稳定）:
v'_i = (Σ sim·v) / (Σ sim)  # 分母→0 时 NaN

# SUPRA GroupNorm（稳定）:
wkv_i = Σ γ^{i-j}·sim(q_i,k_j)·v_j  # (B, h, seq, d_h)
v'_i = GroupNorm(h)(wkv_i)  # h=num_heads, 每head独立归一化
# mean_h = mean(wkv_i along d_h), std_h = std(wkv_i along d_h)
```

Table 3 消融证明：T2R（分母除法）在 1B uptraining 时 HellaSwag 40.6（vs SUPRA 57.0），归一化策略是大规模 uptraining 的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`nn.GroupNorm(num_groups=num_heads, num_channels=D)`。group_size = d_h = D/h，典型值 64-128，足够保证统计量精度。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## T2R (Finetuning Pretrained Transformers into RNNs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
T2R 由 Kasai et al. (EMNLP 2021) 提出，是最早将预训练 softmax Transformer 转换为 RNN 的方法。核心思路：用可学习 MLP 线性注意力 φ(x)=ReLU(Wx+b)（Q/K 共享 W）替换 softmax，追求近似原始 attention 矩阵。SUPRA 对 T2R 进行了三项关键改进：(1) 分母除法 → GroupNorm；(2) 无位置编码 → RoPE；(3) 无衰减 → 固定 γ。T2R 仅在 ~100M 模型上验证，需约 20% 预训练 tokens，而在 1B 规模 uptraining 时性能崩溃（Table 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
T2R: sim(q,k)=φ(q)·φ(k), φ(x)=ReLU(Wx+b)
v'_i = φ(q_i)^T Σ φ(k_j)v_j / φ(q_i)^T Σ φ(k_j)
# 问题: 分母不稳定 + 无位置编码 + 无衰减

SUPRA 改进:
v'_i = GroupNorm(Σ γ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
T2R 原始实现：https://github.com/jungokasai/T2R（fairseq）。其近似 softmax 的策略在理论上更优雅但实践中受限（需比较完整 attention 矩阵，计算昂贵，不可扩展）。SUPRA 验证直接替换优于近似。

涉及论文标题：
- Linearizing_Large_Language_Models

---

## Mamba Hidden State Decay (累积隐藏状态衰减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba隐藏状态衰减是指Mamba SSM中隐藏状态H_t随时间步的指数衰减机制。Mamba的隐藏状态更新公式为 H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t，其中 Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e} 是per-channel的衰减因子，A ∈ R_{<0}^{d_s×d_e} 是负的学习矩阵，保证 Ā_t 始终小于1。这意味着每个时间步的历史隐藏状态都会被衰减。在一段序列结束后，初始隐藏状态H_0对最终H_L的贡献被衰减为 ∏_{k=1}^L Ā_k = exp((Σ_{k=1}^L Δ_k) ⊙ A)。由于A<0且Δ_t>0，累加和ΣΔ_k随L增大而增大，使得指数项向零衰减。LongMamba的核心贡献之一就是识别了这一衰减效应作为Mamba长上下文性能瓶颈的根源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mamba SSM的衰减机制（Eq.4, 7, 12）
Δ_t = Softplus(X_t) ∈ R_{>0}^{d_e}          # per-channel正步长
A ∈ R_{<0}^{d_s×d_e}                          # 负的可学习矩阵
Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e}         # ⊙: 广播element-wise乘

H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t              # 由于Ā_t<1，旧H逐步缩小
∏_{k=1}^L Ā_k = exp((Σ_{k=1}^L Δ_k) ⊙ A)    # A<0 → L变大 → 趋近于0
```
LongMamba分析：全局通道累积衰减足够小（>θ）使其能在训练长度内保持全局信息；但当S≫L时，衰减累积使全局通道无法捕获早期token信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
A矩阵初始化为HiPPO或其他结构化矩阵，训练中通过梯度更新。Δ_t通过Linear_proj + Softplus动态生成，是Mamba选择性机制的关键。LongMamba通过token filtering（跳过Δ_t<g的token，设Ā'_t=1）使有效衰减步数≈L而非S。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## Per-Channel Receptive Field in Mamba SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Channel Receptive Field是LongMamba提出的Mamba隐藏状态通道的可视化分析概念。利用Mamba注意力分数α_{i,j} = C_i^T (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}（Ali et al., 2024），该分数是第j个token对第i个token输出的per-channel贡献。LongMamba在log scale下可视化α_{i,j}矩阵，用红色边框标记attention score > 10^{-3}的范围作为通道的"感受野"。分析发现Mamba的不同通道具有截然不同的感受野长度——有些仅关注~200 tokens的局部上下文，有些覆盖整个训练序列长度(~2000)。在Mamba-130M第12层48个通道的可视化中，感受野排序显示清晰的二分结构——累积衰减小者拥有全局感受野。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
α_{i,j}[c] = C_i[c]^T · (∏_{k=j+1}^i Ā_k[c]) ⊙ B̄_j[c]  # 标量, per-channel

# 通道c的感受野（序列末尾token L的视角）:
receptive_field[c] = argmin_j { j | α_{L,j}[c] > 10^{-3} }
# 短→local channel;  长（≈L）→ global channel

# LongMamba Step 1: 通过累积衰减而非可视化分类:
decay_c = ∏_{k=1}^L Ā_k[c]  # 沿d_s维度取平均
channel_c = "global" if decay_c > θ else "local"
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用Pile采样序列进行分析，θ通过LongBench-E上grid search确定（候选10^{-40}到5×10^{-1}）。不同模型的最优θ差异巨大（Mamba-1.4B: 10^{-30}, Mamba2-1.3B: 5×10^{-2}, Zamba2-1.2B: 10^{-5}）。全局通道需要token filtering来扩大感受野。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## Mamba Attention Score (α_{i,j})

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba Attention Score由Ali et al. (2024)提出，是衡量Mamba SSM中不同token间信息流动强度的per-channel指标。展开Mamba的递归计算后，输出Y_i = Σ_{j=1}^i α_{i,j} ⊙ X_j，其中α_{i,j} = C_i^T (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}。与Transformer的标量attention score不同，Mamba的α_{i,j}是per-channel向量，可捕捉不同通道对不同token pair的差异化关注模式。LongMamba利用此概念进行per-channel感受野分析和全局/局部通道分类。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 推导 (Eq. 8-11):
H_i = Σ_{j=1}^i (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ⊙ X_j
Y_i = C_i^T H_i = Σ_{j=1}^i [C_i^T · (∏_{k=j+1}^i Ā_k) ⊙ B̄_j] ⊙ X_j
α_{i,j} = C_i^T · (∏_{k=j+1}^i Ā_k) ⊙ B̄_j ∈ R^{d_e}

# 对比: Transformer score = q_i^T k_j (标量 per head)
#        Mamba score = α_{i,j} (向量 per-channel)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
仅用于离线分析/可视化，不参与训练或推理优化。DeciMamba也使用此概念分析Mamba的ERF和注意力稀疏性。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## DeciMamba Token Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DeciMamba（Ben-Kish et al., 2024, ICLR 2025）是首个探索Mamba上下文长度外推的training-free方法。核心机制：在Mamba深层（如第12层）利用Δ_t值作为token重要性度量，仅保留top-k个平均Δ_t最大的token进行后续处理。关键发现：Mamba隐含学习了与训练长度绑定的有效感受野（ERF），当序列超过训练长度时隐藏注意力矩阵变稀疏并在~10K token后崩溃。DeciMamba通过减少深层序列长度缓解这一问题。与LongMamba的per-channel区分策略不同，DeciMamba对所有通道统一prune token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
For layer l in decimating_layers (e.g., [12]):
    importance[t] = mean(Δ_t)  # 沿d_e维度
    k = min(max(decimation_min_seq_len, base), current_seq_len)
    kept_indices = topk(importance, k)
    X_l = X_l[kept_indices]    # 仅保留重要token继续

# 典型配置: decimation_beta=0.5, decimating_layers=[12],
#            decimation_min_seq_len=20, decimation_max_p_L_base=2000
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/assafbk/DeciMamba。LongMamba实验表明DeciMamba在PG-19 60k tokens上perplexity仍>30（vs LongMamba <20），因为无差别pruning在所有通道上丢弃token降低了局部上下文建模能力。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Token Filtering for SSM Context Extension

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Filtering是LongMamba的核心技术——跳过不重要token的隐藏状态更新来扩大Mamba全局通道的感受野。当S≫L时，对每个全局通道c，若Δ_t[c] < g_c(S)则设置Ā'_t[c]=1、B̄'_t[c]=0（H_t[c]=H_{t-1}[c]，不衰减也不更新）。阈值g_c(S)是per-channel查找表（1000-token间隔），通过Pile采样序列标定Δ_t分布并数值求解使筛选后∏Ā'_i≈∏_{trained}Ā_i来确定。核心insight：Δ_t可解释为token"重要性"——大Δ=重要token应保留更新，小Δ=不重要可跳过以减少衰减累积。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LongMamba Token Filtering (推理时，per Mamba layer):
For t = 1..S:
    Δ_t = Softplus(Linear_Δ(X_t))
    Ā_t = exp(Δ_t ⊙ A);  B̄_t = Δ_t ⊗ B_t
    For each channel c:
        if is_global[c] and Δ_t[c] < g_c(S):
            Ā'_t[c] = 1;  B̄'_t[c] = 0   # 跳过该token
        else:
            Ā'_t[c] = Ā_t[c];  B̄'_t[c] = B̄_t[c]
    H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t

# 对齐条件: ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
标定：5条Pile随机序列，grid search确定clamping百分位C∈{0,5,10,15,20}。查找表1000-token间隔预计算，推理S向下取整到最近间隔。延迟开销极小（A100 prefill增加≤3.8%）。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## Training-Free Context Extension for SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training-Free Context Extension for SSM指无需额外训练即可使预训练Mamba SSM处理远超训练长度输入序列的方法类别。传统Transformer上下文扩展（位置插值、Attention Sink等）因SSM缺少显式position encoding和attention机制而无法直接应用。当前方法包括：(1) DeciMamba——深层token pruning减少序列长度；(2) LongMamba——per-channel token filtering扩大全局通道感受野；(3) MambaExtend——校准离散化缩放因子。共同特征：不改变模型权重，仅修改推理时前向传播，通过离线标定少量超参数适配更长序列。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
通用范式:
阶段1 (离线标定): 使用训练集分析内部统计量 → 标定超参数
阶段2 (推理干预): 在SSM递归循环中插入条件逻辑 → 调整衰减/更新行为

LongMamba特化:
- 标定: Δ_t分布统计 + global/local通道分类 (θ search)
- 干预: global通道跳过Δ_t<g的token
- 对齐: ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongMamba可直接应用任何预训练Mamba/Mamba2/Zamba2模型，仅需~5条校准序列。局限：需访问训练集数据标定Δ_t分布、超参数需per-model搜索（如Mamba-1.4B θ=10^{-30} vs Mamba2-1.3B θ=5×10^{-2}）。适用于快速将预训练Mamba部署到长上下文场景。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## Zamba2 (Hybrid Transformer-SSM Model)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zamba2（Glorioso et al., 2024a）是Zyphra公司开发的混合架构语言模型，将Mamba SSM层与Transformer attention层结合。Zamba2-1.2B在LongMamba实验中作为hybrid代表被评测——训练长度4k tokens。vanilla Zamba2-1.2B在LongBench-E上avg 11.43%高于纯SSM（8.21-8.37%），LongMamba后提升至17.82%（+6.39%）。使用θ=10^{-5}、C=5配置。纯SSM在coding任务更优，hybrid在few-shot learning更优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Zamba2是shared-parameter hybrid架构，结合Mamba层和shared attention层。其Mamba层包含标准SSM机制（Δ_t, Ā_t, B̄_t等），因此LongMamba的通道分类和token filtering同样适用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
模型权重在HuggingFace发布。LongMamba直接加载官方预训练checkpoint，无微调。实验显示hybrid和纯SSM各有所长，LongMamba能缩小两者差距。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---

## GRPO (Group Relative Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO（Group Relative Policy Optimization）由DeepSeek-R1（Shao et al., 2024）提出，是一种用于LLM推理能力增强的强化学习算法。核心思想：对每个prompt生成多个rollout（如一组8个），用组内相对优势（而非绝对reward值）来估计advantage函数，从而消除对critic模型（value function）的依赖。GRPO是PPO（Proximal Policy Optimization）的简化变体：去除了value function，用组内均值归一化替代。优势估计为 $Â_i = (r_i - mean(r_{group})) / std(r_{group})$，即在组内做z-score归一化。M1论文对GRPO做了两项修改：(1) 移除原GRPO的KL penalty项（实验发现其使训练不稳定）；(2) 添加entropy bonus $\eta \cdot H(\pi_\theta)$ 鼓励策略多样性。最终loss: $L_{GRPO}(\theta) = \mathbb{E}[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot Â(s,a)] + \eta H(\pi_\theta)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# M1中的GRPO训练loop (集成在VeRL框架中)
For step in range(50):
  batch = sample_questions(128)  # 128个不同数学问题
  all_rollouts = []
  for question in batch:
    prompt = "Let's think step by step and output the final answer within \\boxed{}"
    for g in range(8):  # 每个问题生成8个rollout
      output = model.generate(question + prompt, max_new_tokens=32k)
      reward = verify_answer(output, ground_truth)  # 基于数学答案正确性
      all_rollouts.append((question, prompt, output, reward))
  
  # 计算组内相对优势 (per-question normalization)
  for group in group_by_question(all_rollouts):
    mean_r = mean([r for _,_,_,r in group])
    std_r = std([r for _,_,_,r in group])
    for (q, p, o, r) in group:
      advantage = (r - mean_r) / (std_r + 1e-8)  # z-score归一化

  # PPO更新: µ=2 iterations, mini_batch=64
  for iter in range(2):
    for mini_batch in split(shuffle(all_rollouts), 64):
      ratio = π_θ(token|context) / π_θold(token|context)
      loss = ratio * advantage + η * entropy(π_θ)
      loss.backward()
      optimizer.step()  # Adam, LR=1e-6

  if avg_critic_reward(current_model) > best_reward:
    save_checkpoint()
```
与标准PPO的关键区别：(1) 无value function → 无需训练critic → 节省约一半训练参数和计算；(2) Group-relative advantage → 天然消除reward scale和baseline估计问题；(3) 每个问题多个rollout → 在组内形成自然对比信号。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GRPO在VeRL框架中实现（https://github.com/volcengine/verl）。M1的关键超参数：batch_size=128, PPO batch_size=64（决定µ=2 PPO iterations）, 每问题8个rollout, max generation length=32k, Adam optimizer LR=1e-6, 训练50步后选highest critic reward checkpoint。M1论文修复了VeRL中Mamba+CUDA graph+PyTorch FSDP的兼容性问题，使CUDA graph启用后Mamba生成速度提升5x。GRPO适用于需要RL训练的推理模型场景，特别适合数学推理、代码生成等有verifiable reward（答案可自动验证）的任务。局限性：依赖verifiable rewards，对开放式任务（如创意写作）难以定义reward。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

---

## Cross-Architecture Knowledge Distillation (Transformer→SSM/Mamba)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
跨架构知识蒸馏是将知识从一种模型架构（通常为更强的Transformer teacher）迁移到另一种架构（如Mamba/SSM student）的技术。区别于传统同架构蒸馏（如DeepSeek-R1蒸馏到Qwen/Llama：仅需复制logits或hidden states），跨架构蒸馏面临额外挑战：teacher和student的token mixing机制根本不同（softmax attention vs selective SSM scan），直接复用或近似权重矩阵不可行。M1论文的解决方案是通过MambaInLlama方法：将Transformer attention层的Q/K/V/O投影权重映射为Mamba层的C/B/X/O投影，对GQA的KV heads扩展至full heads（因Mamba无KV cache），新增MLP（生成Δ_t）和A参数，然后通过reverse KL divergence蒸馏。M1发现直接跨架构蒸馏推理能力效果差（MATH500仅38%），创新性地采用分阶段策略：先蒸馏通用MATH能力，再SFT推理数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# M1跨架构蒸馏流程（完整三阶段）
# Teacher: Llama3.2-3B-Instruct (Transformer, GQA, 28 layers)
# Student: Hybrid Mamba (28 layers: 22 Mamba + 6 Attention保留)

# Stage 1: 权重初始化映射 (MambaInLlama Algorithm 1)
For each attention layer to convert to Mamba:
  W_C_student = W_Q_teacher       # Q投影→C投影
  # GQA扩展: 8 KV groups→28 full heads (因Mamba无KV cache)
  W_B_student = Linear_expand(W_K_teacher)  # head_dim*kv_head → head_dim*n_head
  W_X_student = Linear_expand(W_V_teacher)  # V投影→X投影 (同样扩展)
  W_O_student = W_O_teacher       # 输出投影直接复用
  MLP_Δ = random_init()           # 新增: 生成Δ_t的MLP
  A = random_init()               # 新增: dynamic parameter ∈ R^{N×N'}
  # MLP layers: 直接复用Transformer的MLP权重

# Stage 2: Reverse KL蒸馏 (token-level)
for input_ids, attention_mask in dataloader:
  # Chat template: mask user prompt, 仅计算assistant token loss
  p_teacher = Teacher(input_ids).logits  # [B, L, V]
  p_student = Student(input_ids).logits
  # Reverse KL: D_KL(p_student || p_teacher) = Σ p_student * log(p_student/p_teacher)
  loss = (p_student * (log(p_student) - log(p_teacher))).sum(dim=-1)
  loss = loss * assistant_mask  # 仅assistant token
  loss.backward()
# Optimizer: AdamW, LR=1e-5, cosine decay, β=(0.9,0.95), weight_decay=0.1
# Data packing: 合并多序列至max_len=8192

# Stage 3a: Math SFT (OpenMathInstruct-2, 2 epochs)
# Stage 3b: Reasoning SFT (10B tokens from R1-generated datasets, 5 epochs)
# Stage 3c: GRPO RL (50 steps, 128 batch, 8 rollouts/question)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
M1使用Axolotl框架（https://github.com/axolotl-ai-cloud/axolotl）实现蒸馏和SFT。关键设计决策：(1) 不追求近似softmax attention矩阵（与T2R策略相反），直接替换让Mamba学到自己的计算范式；(2) 先通用后专项的分阶段策略——先用OpenMathInstruct-2建立Mamba MATH基础（MATH500 45%→74%），再用10B reasoning tokens做推理SFT（74%→82%），克服了直接跨架构推理蒸馏数据不足问题；(3) 6/28=21%的attention层保留——完全去除attention会导致性能崩溃，少量attention层提供关键的长程信息路由能力。开源：https://github.com/jxiw/M1。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

---

## Reverse KL Divergence for LLM Knowledge Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reverse KL Divergence定义为 $D_{KL}(p_{student} \parallel p_{teacher}) = \sum_v p_{student}(v) \cdot \log\frac{p_{student}(v)}{p_{teacher}(v)}$。在LLM知识蒸馏中，它与forward KL $D_{KL}(p_{teacher} \parallel p_{student})$形成对比：forward KL是"mean-seeking"——student试图覆盖teacher所有模式（包括低概率token），可能导致student分散概率质量到低概率区域产生hallucination；reverse KL是"mode-seeking"——student聚焦teacher的高概率模式，允许忽略teacher的低概率尾部。M1选择reverse KL进行跨架构蒸馏，因为Mamba student的表达能力有限（相比Transformer teacher），mode-seeking特性使student集中学习teacher的主要推理模式，而非浪费容量覆盖所有低概率token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Forward KL (mean-seeking, standard in many distillation works):
# D_KL(teacher || student) = Σ_v p_t(v) * log(p_t(v) / p_s(v))
#   = -p_t(v) * log(p_s(v)) + const  → standard cross-entropy with soft targets
loss_fwd = -(p_teacher.detach() * log_softmax(student_logits)).sum(dim=-1)

# Reverse KL (mode-seeking, M1's choice):
# D_KL(student || teacher) = Σ_v p_s(v) * log(p_s(v) / p_t(v))
log_p_student = log_softmax(student_logits)
log_p_teacher = log_softmax(teacher_logits)
p_student = softmax(student_logits)
loss_rev = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)

# Mode-seeking behavior示例 (vocab size=3):
# Teacher: P=[0.7, 0.25, 0.05]
# Forward KL optimal: student ≈ [0.7, 0.25, 0.05] (完全匹配所有模式)
# Reverse KL optimal: student ≈ [0.95, 0.05, 0.0] (聚焦主要模式, 忽略尾部)
# → student更"自信"，输出分布更尖锐
```
M1在蒸馏阶段对每个token位置独立计算token-level reverse KL divergence，仅计算assistant output token的loss（mask user prompt部分）。结合data packing合并多序列至max_len=8192以加速训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在PyTorch中实现为 `F.kl_div(log_p_student, log_p_teacher, reduction='none', log_target=True)` 配合student概率作为权重。M1使用AdamW optimizer, LR=1e-5, β=(0.9,0.95), weight decay=0.1, cosine decay schedule。Reverse KL最适用于：(1) student容量显著小于teacher；(2) student与teacher架构不同（如Transformer→Mamba）；(3) teacher输出分布较flat（有大量低概率token）。在这些场景下reverse KL的mode-seeking特性产生更sharp、更集中的student输出分布。不适用场景：需要student保持与teacher完全相同输出多样性时。开源：https://github.com/jxiw/M1。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

---

## Test-Time Compute Scaling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Test-Time Compute Scaling指在推理阶段通过增加计算预算来提升模型准确率的策略族。两种主要形式：(1) Sample-based scaling——生成k个独立样本（使用非零temperature），通过majority voting（self-consistency, Wang et al. 2023）选出最终答案；(2) Length-based scaling——增加单个样本的最大生成长度（longer chain-of-thought），给模型更多"思考token"。M1论文的核心创新在于将Mamba的推理速度优势（3x faster throughput vs Transformer）转化为test-time compute scaling的准确率增益：同等wall-clock时间预算下，M1可生成更多样本或更长序列，从而获得更高的majority voting accuracy或long-CoT accuracy。评估使用pass@k指标（k个样本中至少一个正确的unbiased概率估计）和Maj@k（majority voting accuracy）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Pass@k unbiased estimation (Chen et al., 2021)
# N=64 total samples per problem, c=#correct, k=budget
pass_at_k = 1 - C(N-c, k) / C(N, k)  if N-c >= k else 1.0
# 使用numerically stable实现 (Chen et al. 2021)

# Majority Voting (Self-Consistency)
answers_per_question = []
for i in range(k):  # k in {1, 16, 32, 64}
  output = model.generate(question, temperature=0.7, max_len=8k)
  answer = extract_boxed(output)  # 从\boxed{...}提取最终答案
  answers_per_question.append(answer)
final_answer = majority_vote(answers_per_question)
accuracy = mean(final_answer == ground_truth)

# M1的速度→准确率转换 (核心创新)
# 最优吞吐量 (通过sweep batch size找到):
# M1-3B: 15169 tokens/s  → 每8k样本≈0.53s
# R1-1.5B-Qwen: 7263 tokens/s → 每8k样本≈1.10s
# 
# 16s wall-clock budget:
# M1可生成 16/0.53 ≈ 30 samples → Maj@30 accuracy
# R1可生成 16/1.10 ≈ 15 samples → Maj@15 accuracy
# M1用更多样本弥补单样本quality的微小差距

# Length-based scaling:
# 固定k=1, 变化max_len ∈ {2k,4k,8k,16k,24k}
# M1生成24k tokens需 24k/15169 ≈ 1.58s
# R1生成24k tokens需 24k/7263 ≈ 3.30s
# 同等时间下M1可生成更长CoT → 更高accuracy (Figure 4 right)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
M1使用VeRL evaluation tools进行评估：temperature=0.7, max_len=32k, pass@1 averaged over 64 runs, Maj@k repeated 100 times以减少统计方差。评估prompt统一为"Let's think step by step and output the final answer within \boxed{}"。最优吞吐量确定方法：从batch size=8开始逐步增加直到throughput decrease，记录峰值tokens/s。Test-time compute scaling适用于任何有高效推理的模型+可自动验证答案的任务（数学、代码），核心trade-off是compute budget vs accuracy gain。M1证明了对Mamba架构，将速度增益转化为准确率增益是可行的实践策略。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models
## Mamba-2 Scan Connector (MSC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 Scan Connector (MSC) 是ML-Mamba论文提出的新型多模态连接器，用于在MLLM中桥接2D非因果视觉特征与1D因果状态空间模型（SSM）的处理能力。核心组件：(1) Mamba-2 Visual Selective Scanning (MVSS) 模块——将2D视觉patch序列通过Mamba-2层的selective scan进行空间上下文建模；(2) 可选的SwiGLU模块——对扫描后的特征进行gated feature extraction。MSC有三种变体：MLP（纯三层MLP，baseline）、MSC-MLP Basic（MSC不含SwiGLU + MLP）、MSC-MLP Advanced（MSC含SwiGLU + MLP）。MSC的设计motivation在于：传统SSM处理的是具有因果关系的1D序列（如语言），而视觉编码器产生的patch序列缺乏自然因果顺序，直接展平为1D序列送入SSM会丢失2D空间关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MSC-MLP Advanced（含BSM扫描 + SwiGLU模块）的前向计算：
```
Input: V_img ∈ R^{N_v×D_v}  (N_v=729个visual patches, 从DINOv2+SigLIP双编码器concat)

// Step 1: Mamba-2 Visual Selective Scan (MVSS) - BSM
// 前向扫描：沿原始patch展开顺序
V_f = Mamba2_Block(V_img)        // 1D SSM scan through Mamba-2 layer
// 后向扫描：反转patch顺序
V_b = Mamba2_Block(flip(V_img))
// 合并前后向信息
V_scan = V_f + flip(V_b)         // ∈ R^{729×D_v}

// 每个Mamba2_Block内部:
//   x_proj, z_proj = Linear_in(x)  // expand 2×
//   x_conv = CausalConv1d(x_proj, window=4)
//   x_act = SiLU(x_conv)
//   Δ, B, C = Linear_dt(x_act)  // data-dependent params
//   A_bar, B_bar = discretize(A, B, Δ)  // ZOH
//   h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_act[t]  // recurrent update
//   y[t] = C ⊗ h_t
//   y = y ⊙ SiLU(z_proj)  // gating
//   output = Linear_out(y)

// Step 2: SwiGLU Feature Extraction
V_gate = Linear_gate(V_scan)     // gate projection
V_proj = Linear_proj(V_scan)     // value projection
V_swiglu = SiLU(V_gate) ⊙ V_proj  // gated activation ∈ R^{729×D_v}

// Step 3: MLP Projector (三层MLP)
V_final = MLP_3layer(V_swiglu)   // ∈ R^{729×D_llm}, 维度对齐至LLM embedding空间
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MSC实现为ML-Mamba代码库（https://github.com/WenjunHuang94/ML-Mamba, MIT License）中的核心模块。训练时MSC和MLP Projector首先在558K LAION-CC-SBU子集上做对齐训练（1 epoch，冻结视觉编码器和LLM），然后在665K Mixed Dataset上做全参数监督微调（1 epoch，解冻LLM）。消融实验（Table 6）证明MSC-MLP Advanced在VQAv2（75.26 vs MLP-only 73.42，+1.84）上优于纯MLP方案。MVSS模块使Mamba-2的selective mechanism（数据依赖的Δ/B/C参数）在visual token之间自适应分配注意力，弥补了纯MLP连接器无法建模空间关系的缺陷。使用场景：任何需要将视觉特征映射到Mamba/SSM-based LLM的多模态任务。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## Mamba-2 Visual Selective Scanning (MVSS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba-2 Visual Selective Scanning (MVSS) 是MSC模块的核心组件，利用Mamba-2层的selective scan机制处理2D视觉patch序列。MVSS探索两种2D扫描机制：(1) Bidirectional-Scan Mechanism (BSM)——沿前后两个方向扫描patch序列，捕获互补的上下文信息；(2) Cross-Scan Mechanism (CSM)——沿四个对角线方向扫描，捕获更丰富的2D空间关系。MVSS的核心insight是将2D视觉数据通过有结构的scan pattern转化为类似1D序列的输入，使Mamba-2的因果SSM能够有效处理非因果的视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// BS机制：前后向扫描
V_img_patches = flatten(image_patches_2d)  // ∈ R^{729×D}, 27×27 grid → 1D
V_f = Mamba2_Block(V_img_patches)          // forward scan
V_b = Mamba2_Block(reverse(V_img_patches)) // backward scan
V_out_bsm = V_f + reverse(V_b)            // merge

// CS机制：四方向对角线扫描
V_out_csm = zeros_like(V_img_patches)
for each direction d in [↘, ↖, ↙, ↗]:  // 4 diagonal directions
    V_d = scan_along_direction(V_img_patches, d)  // 沿方向d展开为1D序列
    V_d_out = Mamba2_Block(V_d)                    // SSM处理
    V_out_csm += unscan_to_grid(V_d_out, d)        // 恢复为2D grid并累加
V_out_csm = V_out_csm / 4  // 平均四个方向
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MVSS的实现基于Mamba-2官方kernel。消融实验（Table 7）显示BSM在大多数benchmark上优于CSM（VQAv2: 75.26 vs 75.14, GQA: 60.68 vs 60.13, VizWiz: 45.17 vs 44.89），但CSM在TextVQA（52.31 vs 52.20）和POPE（88.5 vs 88.3）上略优。这表明BSM的前后向扫描更通用，CSM的对角线扫描在某些需要细粒度空间推理的任务上受益。MVSS的设计借鉴了Vim（Vision Mamba）的双向扫描和VMamba的交叉扫描机制，但使用Mamba-2（而非Mamba-1）作为核心扫描模块，效率更高（Mamba-2 scan比Mamba-1快2-8倍）。适用于将Mamba-2用于视觉特征建模的任何场景。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## Bidirectional-Scan Mechanism (BSM) in Mamba-2 Multimodal

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bidirectional-Scan Mechanism (BSM) 是MVSS模块中的一种2D扫描策略。它将视觉encoder输出的patch特征序列沿前后两个方向分别送入Mamba-2层处理：前向扫描保持原始patch展开顺序（行优先扫描），后向扫描反转顺序（逆序扫描），然后将两路输出合并。BSM的设计哲学源于Vim（Vision Mamba）的核心insight：自然语言有因果方向，但图像没有——双向扫描可以让每个patch在SSM的处理中"看到"两边的邻居，从而捕获2D receptive field，而不像纯前向SSM那样只能看到"左边"的patch。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 假设 27×27 grid of visual patches
// 行优先展开为729-length 1D序列

// Mamba2_Block内部selective scan:
// For每个token位置t (0..728):
//   前向扫描: h_t = A_bar_f[t]·h_{t-1} + B_bar_f[t]·x[t]
//             y_f[t] = C_f[t]·h_t
//   后向扫描: h_t = A_bar_b[t]·h_{t-1} + B_bar_b[t]·x[728-t]
//             y_b[t] = C_b[t]·h_t
//   Mamba-2的A_bar_f/B_bar_f等参数由当前token x[t]动态生成（data-dependent）

// 合并: y_bsm[t] = y_f[t] + y_b[728-t]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BSM的实现复用同一Mamba-2 Block的参数处理前向和后向序列（不额外增加参数），仅需flip操作和一次额外的scan。BSM在ML-Mamba消融中被选为默认扫描机制（优于CSM），因为其实现简单、计算开销小（仅2x scan vs CSM的4x scan），且在VQAv2、GQA、VizWiz、VSR四个benchmark上表现最优。适用于希望通过双向上下文增强SSM视觉处理的场景。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## Cross-Scan Mechanism (CSM) in Mamba-2 Multimodal

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-Scan Mechanism (CSM) 是MVSS模块中的另一种2D扫描策略，将视觉patch特征沿四个对角线方向展开为1D序列并分别送入Mamba-2层处理。四个方向包括：左上→右下、右下→左上、右上→左下、左下→右上。CSM借鉴VMamba的Cross-Scan Module设计思想——对角线方向扫描可以捕获传统行列扫描难以建模的斜向空间关系（如物体的对角线边界、纹理等）。与BSM（2个方向）相比，CSM（4个方向）提供更密集的空间上下文覆盖，但计算量也翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 四方向对角线扫描
directions = [
    "左上→右下",  // scan from top-left to bottom-right
    "右下→左上",  // scan from bottom-right to top-left
    "右上→左下",  // scan from top-right to bottom-left
    "左下→右上",  // scan from bottom-left to top-right
]

for each dir in directions:
    V_seq = flatten_grid_along_direction(V_2d_grid, dir)  // 按dir展开为1D
    V_out_dir = Mamba2_Block(V_seq)  // SSM scan
    V_recovered = reshape_to_grid(V_out_dir, dir)  // 恢复为2D
    V_accumulated += V_recovered

V_final = V_accumulated / 4  // 平均
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CSM实现需要4次Mamba-2的forward pass，计算量是BSM的2倍。消融实验（Table 7）显示CSM在TextVQA（+0.11 over BSM）和POPE（+0.2）上略有优势——可能是因为对角线扫描有助于OCR任务中的文本线条检测和物体边界判断。但由于BSM在更广泛benchmark上总体更优且计算更高效，ML-Mamba最终选用BSM作为默认配置。适用于需要精细空间推理（特别是斜向纹理/文字检测）的视觉-语言任务。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## SwiGLU (in Multimodal Connector)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwiGLU (SiLU-Gated Linear Unit) 是一种门控激活函数，由Shazeer(2020)提出用于改进Transformer的FFN层。定义为 SwiGLU(x) = SiLU(xW_g + b_g) ⊙ (xW_v + b_v)，其中SiLU(x) = x·σ(x)。与传统激活（ReLU、GELU）相比，SwiGLU通过门控机制实现了输入依赖的激活模式——gate分支（SiLU）控制哪些信息通过，value分支提供原始信号。在ML-Mamba中，SwiGLU被用于MSC模块（而非LLM的FFN），对Mamba-2 scan后的视觉特征进行gated feature extraction，使MSC-MLP Advanced比Basic变体获得额外性能增益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// SwiGLU in ML-Mamba's MSC-MLP Advanced connector:
V_scan = MVSS(V_img)  // Mamba-2 visual scan output ∈ R^{N_v×D_v}

// SwiGLU feature extraction
V_gate = V_scan @ W_gate + b_gate  // gate projection ∈ R^{N_v×D_v}
V_proj = V_scan @ W_proj + b_proj  // value projection ∈ R^{N_v×D_v}
V_out = SiLU(V_gate) ⊙ V_proj     // element-wise gated activation

// 对比标准SwiGLU (in FFN): 通常有expand ratio
// V_proj expanded to 4×D, then projected back
// ML-Mamba中的SwiGLU保持D_v维度不变（无expand）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch实现：`nn.SiLU()` + element-wise multiply。ML-Mamba中的SwiGLU（Table 6消融）将MSC-MLP从Basic升级为Advanced后，VQAv2从75.09→75.26（+0.17），POPE从86.5→88.3（+1.8），证明SwiGLU的gated feature extraction对多模态特征处理有显著价值。用户可将其作为MSC模块的可选组件，以少量额外参数换取特征提取质量的提升。常用于现代Transformer/VLM架构的门控FFN和跨模态特征转换。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## DINOv2 (Self-Supervised Vision Encoder)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DINOv2是Meta AI提出的基于自监督学习的视觉基础模型（Oquab et al., 2024），通过knowledge distillation和contrastive learning在LVD-142M（大规模未标注图像数据集）上预训练。核心特点是生成高质量、语义丰富的视觉特征，特别擅长保留低层空间细节（如物体边界、纹理、几何结构）。与CLIP/SigLIP等语言对齐的视觉编码器不同，DINOv2不需要文本监督信号——其训练目标是最小化teacher和student网络输出之间的差异。ML-Mamba使用DINOv2 ViT-Large（304M参数）作为双编码器之一，负责提供图像的空间结构信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// DINOv2在ML-Mamba中的使用:
Input: 图片 X_v ∈ R^{3×384×384}
patches = patchify(X_v, P=14)  // 27×27 = 729 patches
V_dino = DINOv2_ViT_Large(patches)  // ∈ R^{729×D_dino}
// DINOv2内部: 24层ViT transformer blocks with self-attention
// 输出: patch-level dense features

// 与SigLIP特征拼接:
V_siglip = SigLIP_ViT(patches)  // ∈ R^{729×D_sig}
V_img = concat([V_siglip; V_dino], dim=-1)  // ∈ R^{729×(D_sig+D_dino)}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DINOv2开源：https://github.com/facebookresearch/dinov2 (Apache 2.0)。常用加载方式：`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`。在ML-Mamba消融（Table 5）中，单独使用DINOv2在VQAv2上达73.73，单独SigLIP达74.61，组合达75.26——证明DINOv2的低层空间特征与SigLIP的高层语义特征互补。DINOv2的典型应用场景：(1) 作为多模态模型的空间编码器补充语义编码器（如CLIP/SigLIP）；(2) 密集预测任务（分割、深度估计）；(3) 图像检索和匹配（利用其instance-level特征）。与SigLIP的配合使用时需注意分辨率对齐（两者通常需要统一input size和patch size）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## SigLIP (Sigmoid Loss Language-Image Pre-training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP是Google提出的视觉-语言预训练模型（Zhai et al., 2023），其核心创新是将CLIP的标准softmax对比损失替换为sigmoid loss来处理image-text pairs。与CLIP的softmax loss（需要在整个batch内计算归一化，O(batch²)）不同，SigLIP对每对(image, text)独立使用二分类sigmoid cross-entropy loss，训练更高效且允许任意large batch。此外，sigmoid loss天然支持multi-label匹配（一个image可以匹配多个text），在开放世界理解任务上表现优越。ML-Mamba使用shape-optimized SigLIP（比ViT-Large略大）作为语义编码器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// SigLIP训练loss vs CLIP loss:
// CLIP: L = -1/N Σ_i log(exp(x_i^T y_i / τ) / Σ_j exp(x_i^T y_j / τ))
// SigLIP: L = -1/N Σ_i log(σ(x_i^T y_i / τ + b) - 1/N² Σ_i Σ_{j≠i} log(σ(-x_i^T y_j / τ + b))
// 其中σ是sigmoid函数, b是可学习的bias, τ是温度参数

// SigLIP在ML-Mamba中的使用:
V_siglip = SigLIP_ViT(X_v)  // ∈ R^{729×D_sig}
// 与DINOv2拼接后送入MSC
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SigLIP开源实现：https://github.com/google-research/big_vision（JAX实现）。HuggingFace上提供PyTorch版本：`google/siglip-base-patch16-256`等。ML-Mamba消融（Table 5）显示SigLIP单独使用在VQAv2上达74.61（优于DINOv2的73.73），在POPE上达87.4（优于DINOv2的86.6），证实SigLIP的语义对齐能力在VLM任务中的核心作用。SigLIP + DINOv2组合成为当前高性能VLM的标准双编码器方案——SigLIP提供language-aligned semantics，DINOv2提供spatial/structural detail。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## Multimodal Large Language Model (MLLM) / Visual Language Model (VLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multimodal Large Language Model (MLLM) / Visual Language Model (VLM) 是将视觉理解能力与语言理解和生成能力结合的模型架构。典型架构包含三个核心组件：(1) 视觉编码器——将图片转换为特征向量（如CLIP、SigLIP、DINOv2）；(2) 多模态连接器/Projector——将视觉特征映射到LLM的输入空间（如MLP、Q-Former、MSC）；(3) 大语言模型（LLM）——处理拼接后的视觉+文本token并生成回答。代表性模型包括LLaVA、GPT-4V、Qwen-VL、BLIP-2等。MLLM的训练通常分两阶段：预对齐（visual-text alignment on image-caption pairs）和指令微调（instruction tuning on multi-turn dialog data）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 标准MLLM推理pipeline:
Input: Image X_v, Text Question Q

// Step 1: 视觉编码
V = VisionEncoder(X_v)  // ∈ R^{N_v×D_v}, N_v=patch数量

// Step 2: 特征投影
V_proj = Connector(V)  // ∈ R^{N_v×D_llm}, 映射到LLM embedding空间

// Step 3: token拼接
T = Tokenizer(Q)  // ∈ R^{L_text}, token IDs
Input_emb = concat([V_proj; Embedding(T)], dim=0)

// Step 4: LLM自回归生成
for each token position:
    Answer = LLM(Input_emb)  // 多轮自回归生成

// Transformer-based MLLM: O((V+T)²) attention计算, KV-Cache = O(V+T)
// SSM-based MLLM (ML-Mamba): O(V+T) scan计算, Hidden State = O(1) fixed size
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现框架：LLaVA (https://github.com/haotian-liu/LLaVA), LLaMA-Adapter, BLIP-2。ML-Mamba是将Mamba-2 SSM引入MLLM的早期工作之一，证明SSM-based backbone在保持线性复杂度的同时能匹敌Transformer-based MLLM（在POPE上88.3 vs LLaVA-1.5-7B的85.9）。通用benchmark包括VQAv2、GQA、TextVQA、POPE、VizWiz、MMBench等。当前趋势包括：(1) SSM/Linear Attention替代Transformer backbone（如ML-Mamba、VL-Mamba、Cobra）；(2) 更强视觉编码器（DINOv2+SigLIP/CLIP双编码器）；(3) 更高效的多模态连接器（Q-Former、MSC、C-Abstractor）；(4) 端到端训练（减少预对齐阶段）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

## Multimodal Connector / Projector in MLLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
多模态连接器（Multimodal Connector / Projector）是MLLM中位于视觉编码器和LLM之间的模块，负责将视觉特征映射到LLM的embedding空间，并可能对视觉特征进行进一步处理（如序列压缩、空间关系建模）。最简单的连接器是3层MLP（如LLaVA-1.5），更复杂的包括Q-Former（BLIP-2，使用learnable queries通过cross-attention压缩视觉token）、C-Abstractor（MobileVLM，通过depth-wise conv减少visual token数量）、MSC（ML-Mamba，使用Mamba-2 scan进行2D空间建模）。连接器设计影响视觉token数量、信息损失程度和计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 不同连接器设计对比:

// 1. MLP (LLaVA-1.5): 最简方案
V_out = Linear2(GELU(Linear1(V_img)))  // 2-3层MLP

// 2. Q-Former (BLIP-2): 固定数量可学习queries
Q = nn.Parameter(torch.randn(K, D))  // K << N_v
V_out = CrossAttention(Q, V_img, V_img)  // K个输出token

// 3. MSC-MLP Advanced (ML-Mamba): Mamba-2 scan + SwiGLU + MLP
V_scan = MVSS(V_img)            // 2D spatial context via Mamba-2 scan
V_swiglu = SwiGLU(V_scan)       // gated feature extraction
V_out = MLP_3layer(V_swiglu)    // dimension alignment

// 4. C-Abstractor (MobileVLM): depth-wise conv压缩
V_out = Conv2D(reshape(V_img))  // 降低token数量 729→144
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
连接器的选择由三个因素决定：(1) 视觉token数量——token越多质量越好但推理越慢；(2) 压缩率——高压缩率（如C-Abstractor的5x）可以加速但可能损失信息；(3) 空间建模能力——如MSC的Mamba-2 scan可以建模patch间2D关系。ML-Mamba消融（Table 6）证实：MLP (VQAv2 73.42) → MSC-MLP Basic (+1.67) → MSC-MLP Advanced (+1.84)，处理729个visual tokens仍保持171 tokens/s的生成速度（vs MobileVLM v2用144 tokens仅50 tokens/s）。连接器训练通常分两阶段：先在caption数据上对齐（train connector only），再在instruction数据上联合微调（train connector + LLM）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---

---

## Top-k Chunk Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-k Chunk Sparse Attention 是 RWKV-X 论文中提出的一种稀疏注意力机制，灵感来自 MoBA（Mixture of Block Attention, Lu et al., 2025）。核心思想是将自回归生成中的全序列注意力替换为仅对 top-k 个最相关 chunk 的稀疏注意力，从而将 O(N²) 的复杂度降至 O(kBN) ≈ O(N)。具体流程：(1) 将长度为 N 的输入序列等分为 n 个大小为 B 的 chunk；(2) 对每个 query token q，计算 q 与各 chunk 的 mean-pooled key vector 的内积作为相关性得分 s_i = q · (1/B Σ_j k_j^(i))；(3) 通过 TopK 操作选择得分最高的 k 个 chunk 索引 I = TopK({s_i}, k)；(4) 仅在被选中的 chunk 上计算标准 softmax attention: Attn(q, K_I, V_I) = softmax(qK_I^T/√d_k) V_I。由于 k 和 B 为小常数，总计算复杂度为 O(kBN) ≈ O(N)。该方法结合了 KV Cache Management（SnapKV 风格的重要性逐出）以确保解码阶段的 constant memory usage。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Top-k Chunk Sparse Attention 在 RWKV-X 中的前向流程：
```
# Input: h ∈ R^{B×L×D}, chunk_size B_c, topk k
q, k, v = W_Q(h), W_K(h), W_V(h)  # (B, L, d_head)

# Step 1: Chunk partitioning
n_chunks = L // B_c
k_chunks = reshape(k, (B, n_chunks, B_c, d_head))  # (B, n, B_c, d)
v_chunks = reshape(v, (B, n_chunks, B_c, d_head))

# Step 2: Mean-pooled chunk keys
k_mean = mean(k_chunks, dim=2)  # (B, n, d)

# Step 3: Chunk relevance scoring
scores = einsum("bld,bnd->bln", q, k_mean)  # (B, L, n)

# Step 4: Top-k chunk selection
topk_indices = topk(scores, k, dim=-1)  # (B, L, k)

# Step 5: Gather selected chunks
k_selected = gather(k_chunks, topk_indices)  # (B, L, k*B_c, d)
v_selected = gather(v_chunks, topk_indices)

# Step 6: Sparse attention
attn = softmax(q @ k_selected^T / sqrt(d_k))  # (B, L, k*B_c)
output = attn @ v_selected  # (B, L, d)

# Step 7: Output projection + residual
h_out = h + W_O(output)
```

在 RWKV-X 混合架构中，稀疏注意力层占约 25%（12 层中约 3 层），其余 75% 仍为 RWKV-7 循环层。在解码阶段，KV cache 通过 SnapKV 风格的重要性管理保持固定大小（论文中设为 64K），使 attention 计算量和内存均不随生成长度增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/howard-hou/RWKV-X。RWKV-X 的实现将稀疏注意力层与 RWKV-7 层混合堆叠，约每 4 层插入 1 个稀疏注意力层（通过消融实验确定 25% 注意力比例最优，Figure 5）。Block expansion 阶段零初始化新稀疏注意力层（output projection=0），确保初始状态下新层表现为恒等映射，仅传递残差。Alignment pretraining 阶段仅训练稀疏注意力层参数（freeze RWKV-7 参数），Long-context pretraining 阶段全参数微调。适用于需要长上下文检索能力但希望保持线性复杂度的 LLM 训练和推理。

与 MoBA 的区别：
- MoBA 使用 parameter-less gating 选择 top-k block，每个 query 独立选择 block
- Top-k Chunk Sparse Attention 同样使用 mean-pooled key 计算 chunk 得分，但额外集成了 KV Cache Management（SnapKV 风格的重要性逐出）以实现 constant decoding memory
- MoBA 在 autoregressive decoding 中 KV cache 随序列长度线性增长；RWKV-X 通过 cache 压缩保证 O(1) 解码内存

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---

## Long-context Cross-Entropy (LongCE) Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LongCE（Long-context Cross-Entropy）Loss 由 Fang et al.（2025, ICLR 2025）提出，发表于论文 "What is Wrong with Perplexity for Long-context Language Modeling?"。核心发现：标准 perplexity 对所有 token 等权平均，无法区分对长上下文理解关键（key tokens）和无关键（ordinary tokens）的 token，导致 perplexity 与长上下文 benchmark 性能相关性差（Pearson 接近 0）。LongCE 通过 long-short context contrastive 方法识别 key tokens（在长上下文中预测显著不同于短上下文预测的 token），并在 CE loss 中对 key tokens 施加更高权重（weight > 1），对 ordinary tokens 维持 weight ≈ 1。这使得模型在长上下文继续预训练时自动聚焦于对长程依赖关键的 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LongCE Loss 计算流程：
```
# 给定: 长上下文模型 P_long(token|ctx_N), 短上下文模型 P_short(token|ctx_k)
# k << N, 例如 k=4096, N=65536

For each token position i in sequence:
    # Compute prediction discrepancy between long and short context
    p_long = P_long(x_i | x_{i-N:i-1})
    p_short = P_short(x_i | x_{i-k:i-1})
    
    # Key token identification: token hard to predict without full context
    discrepancy = |log(p_long) - log(p_short)|
    
    # Dynamic weight assignment
    w_i = 1 + λ * discrepancy  # λ controls weighting strength
    
    # Weighted cross-entropy
    loss_i = -w_i * log(P_long(x_i | context))

LongCE_loss = mean(loss_i)  # average over sequence
```

在 RWKV-X 中，LongCE 被用于 long-context continual pretraining 阶段（ProLong-64K 数据集，64K context）。消融实验（Table 4）：S-NIAH-2 8K 上 w/ LongCE 99.8 vs w/o 67.0；S-NIAH-3 8K 上 w/ LongCE 95.6 vs w/o 62.6。LongCE 在深层长上下文推理任务（S-NIAH-2/3）上效果显著，在简单任务（S-NIAH-1 passkey retrieval）上无差异（两者均为 100%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongCE 作为 plug-and-play 训练策略，可直接替换标准 CE loss 用于任何 LLM 的长上下文继续预训练。长/短上下文模型的对比可通过：(1) 使用两个独立的模型（long context + short context checkpoints）；(2) 使用同一模型在不同 context window 下的预测差异。在 RWKV-X 中，关键 token 识别基于 ProLong 论文的 long-short context contrastive 方法。代码开源：https://github.com/PKU-ML/LongPPL。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---

## Block Expansion Method

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Expansion Method 是一种将预训练模型扩展为更大模型的技术，由 LLaMA Pro（Wu et al., 2024, ACL 2024）首次系统提出。核心思想：在预训练 LLM 的 transformer block 序列中以交错方式（interleaved）插入新 block 的副本，将 output projection 层（o_proj 和 down_proj）零初始化使新 block 初态表现为恒等映射，然后仅训练新 block 或分阶段训练以注入新知识。这种方法通过复用预训练权重避免从头训练，同时交错插入使新容量分布于各抽象层级（而非仅堆在顶层或底层），比 LoRA 等参数高效方法保留了更完整的模型表达力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block Expansion 在 RWKV-X 中的两阶段流程：
```
# Stage 0: Model Expansion (预训练 checkpoint → 扩展模型)
model_rwkv7 = load_checkpoint("RWKV-7")  # L layers
new_model = copy(model_rwkv7)
# 交错插入新层：将 L 层分为 N 组，每组后插入 1 个新 block
for group_idx in range(N):
    insert_pos = (group_idx + 1) * (L // N) + group_idx
    new_block = copy_block(model_rwkv7.layers[insert_pos - 1])
    # Zero-init output projections for identity mapping
    new_block.time_mixing.wkv_output.weight = 0
    new_block.channel_mixing.output.weight = 0
    # 对 Sparse Attention block: W_O 零初始化
    new_block.attn.W_O.weight = 0
    new_model.insert_layer(insert_pos, new_block)

# Stage 1: Alignment Pretraining (MiniPile, ctx=1024, 1.5B tokens)
# RWKV-7 blocks frozen, only new blocks trainable
for batch in MiniPile:
    loss = LongCE(new_model(batch))  # only new block params get gradients

# Stage 2: Long-context Continual Pretraining (ProLong-64K, ctx=64K, 1B tokens)
# All parameters unfrozen
new_model.unfreeze_all()
for batch in ProLong:
    loss = LongCE(new_model(batch))  # all params updated
```

参数配置（RWKV-X 3.6B）：L=32 original layers, 每 4 层插入 1 个 Sparse Attention block, 共 8 个新层, 总 40 层。Alignment phase: batch=1.024M tokens, ctx=4096, 4 GPU hours on H20。Long-context phase: batch=8.192M tokens, ctx=64K, 80 GPU hours on H200。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLaMA Pro 开源：https://github.com/TencentARC/LLaMA-Pro。RWKV-X 基于此方法扩展至混合架构领域（RNN + Attention hybrid）。关键实现细节：(1) 只零初始化 output projection（o_proj/down_proj/W_O），不能零初始化 RMSNorm（会导致梯度完全消失）；(2) 采用交错插入而非堆叠于顶层——消融证明交错方式显著优于仅堆叠顶层或底层；(3) alignment stage 仅训练新 block，保留原始模型 general knowledge 防止 catastrophic forgetting；(4) 新 block 初始为原 block 的副本（非随机初始化），零初始化 output projection 使其恒等映射，training 过程中逐渐学习非零 output。适用于：将预训练 LLM/RWKV 模型扩展以注入新领域知识或增强特定能力（如长上下文），同时保留原始通用能力。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---

## Hybrid Language Model (Linear RNN + Sparse Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
混合语言模型（Hybrid Language Model）是一种将不同类型的 token-mixing 机制（如 linear RNN/SSM + attention）在层级别组合的 LLM 架构。与纯 Transformer（全 attention, O(N²) 复杂度）和纯线性 RNN（全 recurrence, state capacity 有限）不同，混合模型利用不同层的互补优势：线性 RNN 层提供高效 short-range modeling（O(1) per-token），attention 层提供精确 long-range retrieval。早期混合模型（Jamba, Zamba, MiniMax）使用 full attention 层 + Mamba/SSM 层，保持了 O(N²) 瓶颈。RWKV-X 首次提出全线性复杂度混合架构——使用 Top-k Chunk Sparse Attention（O(N) training）替代 full attention 并结合 KV Cache Management（O(1) decoding），实现真正的线性复杂度混合模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RWKV-X 混合架构的层级配置与数据流：
```
# 模型配置: 40 layers (32 original RWKV-7 + 8 Sparse Attention)
# Layer pattern: RWKV-7, RWKV-7, RWKV-7, SparseAttn, RWKV-7, ...
# 即 N:1 = 4:1 ratio (25% attention layers, 验证为最优)

Input: x ∈ R^{B×L×D}
h = embedding(x)

For layer l in 1..40:
    if l % 4 == 0:  # Sparse Attention block (每第4层)
        # O(kBN) training, O(1) decoding with compressed KV cache
        h_norm = RMSNorm(h)
        h_attn = TopKChunkSparseAttention(h_norm)
        h = h + h_attn  # residual
        h_norm2 = RMSNorm(h)
        h_ffn = SwiGLU_FFN(h_norm2)
        h = h + h_ffn  # residual
    else:  # RWKV-7 block
        # O(N) training (parallel scan), O(1) decoding (recurrent)
        h_norm = RMSNorm(h)
        h_time = TimeMixing_WKV(h_norm)  # Generalized Delta Rule
        h = h + h_time  # residual
        h_norm2 = RMSNorm(h)
        h_chan = ChannelMixing_FFN(h_norm2)
        h = h + h_chan  # residual

output = LM_head(RMSNorm(h))
```

混合架构的设计要点：(1) 注意力层比例：消融实验（Figure 5）表明 25% 注意力层比例在 126M 参数模型上实现最优 validation loss（纯 RWKV-7=0% 和纯 Sparse Attention Transformer=100% 均不如混合）；(2) 交错插入：注意力层均匀分布（而非集中），使每个抽象层级都有 long-range retrieval 能力；(3) 两阶段训练：alignment（仅训练新注意力层）+ long-context pretraining（全参数微调）确保混合架构的稳定收敛；(4) 无位置编码：RWKV-7 的递归已提供隐式位置信息，消融证明 No Pos 优于 Abs Pos/ROPE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RWKV-X 开源：https://github.com/howard-hou/RWKV-X。一般混合模型实现方式：(1) 选择主干线性 RNN/SSM 模型（RWKV、Mamba 等）的 checkpoint；(2) 使用 block expansion 方法插入 attention 层；(3) 分阶段训练（先冻结主干、后全参数）。混合比例通过小规模消融实验确定（论文中 12 层 126M 模型探索 0%-100% attention 比例）。适用于：需要同时满足短上下文 competitive performance 和长上下文 strong retrieval 的通用 LLM 训练场景。

涉及论文标题：
- RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

---

## UTRC (Unified Token Reduction by token importance Classification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UTRC是Rethinking Token Reduction for SSMs论文提出的面向SSM（Mamba系列）的统一后训练token reduction方法。核心流程为6步：(1) 从Mamba block的SSM隐藏状态y计算token重要性 `S = Σ_d max(0, y_{:,d}) / D'`（使用ReLU clip保留正向激活通道）。(2) 按重要性将N个token两等分为集合M_A（低重要性N/2个）和M_B（高重要性N/2个）。(3) 为M_A中每个token a_i计算其到M_B中最相似token的连接：`f_i = argmax_{b_j∈M_B} cosine_sim(a_i, b_j)`，得到最大相似度g_i。(4) 按g_i降序排序所有连接，保留最相似的top-p%连接对。(5) 对保留连接执行UTR：q比例的连接执行pruning（删除M_A中的token），(1-q)比例的连接执行merging（`f_i = (a_i + f_i)/2`），q=0.5时效果最优。(6) 重新组装M_B和缩减后的M_A。设计空间：hidden states上使用hybrid（q=0.5），residual connections上仅使用merging以保护残差信息完整性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# UTRC per-layer pipeline:
y = SSM(A, B, C)(x)                       # hidden states ∈ R^{B×N×D'}
S_i = sum(max(0, y[i,:,:])) / D'          # importance ∈ R^{B×N×1}
sorted_idx = argsort(S, descending=True)
M_B = sorted_idx[:, :N//2]                 # 高重要性
M_A = sorted_idx[:, N//2:]                 # 低重要性
for a_i in M_A:
    sims = [cosine_sim(a_i, b_j) for b_j in M_B]
    f_i = M_B[argmax(sims)]
    g_i = max(sims)
num_keep = int(p * N/2)
keep = sort_by_g({(a_i, f_i, g_i)})[:num_keep]
mid = int(0.5 * num_keep)
for (a_i, f_i) in keep[:mid]:             # PRUNE
    M_A.remove(a_i)
for (a_i, f_i) in keep[mid:]:             # MERGE
    T[f_i] = (T[a_i] + T[f_i]) / 2
    M_A.remove(a_i)
output = reassemble(M_B, M_A_reduced)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/wuyushuwys/ToR_SSM。基于PyTorch + HuggingFace Transformers，作为hook注入Mamba block的SSM输出处（在Linear投影和残差加法之前），不修改模型权重。层次化应用：从第10~12层开始，每5层执行一次（如Mamba-2-2.7B在layers [12,17,22,27,32,37,42]），使用固定压缩率。p值由目标FLOPS reduction反推。评估适配：token数减少后PPL/Accuracy在调整后的logits上计算（取前(1-m%)个token对应标签）。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Token Importance Metric from SSM Hidden States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Importance Metric for SSMs是Rethinking Token Reduction论文提出的从Mamba SSM隐藏状态评估每个token重要性的度量方法。度量公式为 `S = Σ_{d=1}^{D'} max(0, y_{::d}) / D'`，其中y ∈ R^{B×N×D'}是SSM层输出隐藏状态，max(0,·)（ReLU clip）只保留正向激活通道值，沿特征维D'求和除以D'得平均重要性。选择SSM隐藏状态的原因：SSM拥有高维通道空间（D'），能对每个token进行细粒度的多通道关注度分析，不同于Transformer的单一attention矩阵。clip操作优于ℓ1/ℓ2 norm和unclipped版本：只关注正向激活更有信息量。消融证实：clip版本Mamba-2-2.7B达PPL 17.96、Avg Acc 58.7%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
y = SSM_forward(A, B, C, x)               # y ∈ R^{B×N×D'}
S_clipped = sum(max(0, y), dim=-1) / D'   # ∈ R^{B×N×1} (论文最优)
S_l1 = sum(abs(y), dim=-1) / D'           # ℓ1-norm 对比
S_l2 = sqrt(sum(y^2, dim=-1)) / D'        # ℓ2-norm 对比

# Mamba-2-2.7B @20% FLOPS:
# Clip: PPL 17.96, Acc 58.7%
# ℓ1:   PPL 17.96, Acc 58.6%
# ℓ2:   PPL 19.86, Acc 58.6%
# 无Clip: PPL 18.17, Acc 58.5%
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码：https://github.com/wuyushuwys/ToR_SSM。实现为hook读取Mamba block的selective_scan中间tensor，无需额外模型修改或权重存储。与DeciMamba（用Δ_t）、LongMamba（用Δ_t区分全局/局部通道）的重要差别：直接使用SSM输出的hidden states作为信号源。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Hybrid Token Reduction (Pruning + Merging)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Token Reduction是Rethinking Token Reduction论文提出的将token pruning和token merging以特定比例组合用于同一层内token缩减的策略。对保留的相似连接对，q比例执行pruning（直接删除低重要性token），(1-q)比例执行merging（将低重要性token信息平均融合到高重要性对应token：`f_i = (a_i + f_i)/2`）。q=0.5效果最优。技术原理：pruning消除纯冗余token（相似度极高，被counterpart完全代表），merging保留有独特信息的token（通过融合保留其语义贡献）。纯pruning信息损失大，纯merging在残差上会破坏前层信息——hybrid在两者间取得最优平衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
num_prune = int(q * num_keep_pairs)
num_merge = num_keep_pairs - num_prune
# Pruning (前q比例):
for (a_i, f_i) in keep[:num_prune]:
    M_A.remove(a_i)                       # 纯删除
# Merging (后1-q比例):
for (a_i, f_i) in keep[num_prune:]:
    T[f_i] = (T[a_i] + T[f_i]) / 2       # 平均融合
    M_A.remove(a_i)
# Hidden states: q=0.5 ← 消融最优
# Residual connections: q=0 (纯merge) ← 保护残差信息
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
q值需消融优选。论文消融（Table 5）Mamba-2-2.7B @30% FLOPS：hidden q=0.5 + residual M-only → PPL 40.61, Acc 54.7%（最优）；hidden P-only + residual P-only → PPL 42.65, Acc 53.9%；hidden M-only + residual M-only → PPL 42.61, Acc 54.0%。通用原则：hidden states可用hybrid，residual branches应保守（仅merge）。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Post-training Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-training Token Reduction是一种模型后处理效率优化技术——在不重新训练模型的情况下，通过减少推理时处理的token数量来降低计算量和内存占用。与训练时稀疏化（如structured pruning、distillation）不同，post-training方法直接应用于已训练好的checkpoint，无需访问训练数据或进行额外训练。Rethinking Token Reduction的方法属于此范畴：基于预训练Mamba模型，直接注入token reduction hook，零样本评估。优点：部署成本低（无需GPU训练集群）、即插即用（支持任何checkpoint）、可适应不同压缩率。缺点：极端压缩率下性能上限低于训练时方法（论文也指出微调可能进一步改善性能）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Post-training Pipeline:
# Phase 1: Inject hooks (offline, once)
for layer in pretrained_model.layers[layer_start::interval]:
    layer.register_hook('after_ssm', utrc_reduce)

# Phase 2: Inference (online)
output = pretrained_model(input_sequence)  # hooks auto-execute

# Contrast with training-based:
# Training-based: calibrate → fine-tune → validate → deploy
# Post-training: load_checkpoint → inject_hooks → deploy
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：(1) PyTorch hook注册在目标层的forward输出处；(2) 重要性计算基于激活统计量；(3) 排序+筛选保留top-k token；(4) 序列压缩重打包。适用场景：快速部署预训练LLM/SSM、边缘设备内存受限、多压缩率SaaS服务。局限：依赖高质量重要性度量，严重压缩时遭遇不可恢复信息损失。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Hierarchical Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Token Reduction是一种跨模型层的渐进式token缩减策略——不在每层都执行reduction，而是每隔固定层数（如5层）执行一次。Rethinking Token Reduction论文对Mamba-2-2.7B在[12,17,22,27,32,37,42]层执行。动机：(1) 相邻层token重要性分布相似，每层reduction冗余；(2) 浅层representation不成熟，不适合过早reduction；(3) 更早层reduction产生更大累积FLOPs节省。消融（Table 4）：[12,17,...]配置PPL 17.96/Acc 58.7%优于更深起始[20,25,...]的PPL 18.88/Acc 57.8%，证明适中提早策略最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Hierarchical Schedule:
config = {
    "Mamba-2-2.7B/2.8B": [12, 17, 22, 27, 32, 37, 42],
    "Mamba-2-1.3B/1.4B": [10, 15, 20, 25, 30, 35],
}

for layer_id in model.layers:
    x = forward_pre_ssm(layer, x)
    if layer_id in reduction_layers:
        x = reduce_tokens(x, r)            # Token数递减
    x = forward_post_ssm(layer, x)

# Per-layer vs Hierarchical:
# Per-layer:   每层reduction → 高开销、冗余计算
# Hierarchical: 每5层一次 → 低开销、累积效果好
# 起始层选择: 1/4~1/3总层数处, 间隔≈5层
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为层索引检查条件，reduction_layers通过消融grid search确定。通用原则：起始层≈总层数1/4处，间隔≈5层，最后一层不做reduction。适用于任何多层Transformer/SSM架构的token reduction场景。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Intra-layer Token Reduction (Hidden States + Residual Connections)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Intra-layer Token Reduction是指在同一模型层内对多个计算分支（hidden states和residual connections）分别应用不同token reduction策略的设计。Rethinking Token Reduction发现：(1) hidden states承载SSM处理后新信息，适合hybrid（pruning+merging）；(2) residual传递前层原始信息，只能merging不能pruning（pruning永久丢失残差信号）。关键问题：若hidden和residual以不同方式/不同步调reduction，重组时出现"index misalignment"（hidden中删除token在residual仍存在，维度不匹配）。UTRC通过统一M_A/M_B分类解决此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
y = SSM(A, B, C)(x)                    # hidden states
residual = T_{l-1}                      # residual

M_B, M_A = importance_classify(y)      # 共享分类

# Hidden: Hybrid (q=0.5)
y_reduced = utr_hybrid(y, M_B, M_A, q=0.5)
# Residual: Merge-only
res_reduced = utr_merge_only(residual, M_B, M_A)

output = proj(y_reduced) + res_reduced  # 维度一致!
# M_A在两个分支中被同步删除以保证alignment
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为Mamba block hook中的并行reduction路径，共享M_A/M_B索引保证dimension一致性。通用化原则：任何具有多计算分支的block（如Transformer的attention+MLP分支），hidden/states分支和residual分支需要解耦reduction策略以保证对齐。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## Cosine Similarity-based Token Matching for Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cosine Similarity-based Token Matching是UTRC中连接低重要性和高重要性token集合的机制。对每个低重要性token a_i ∈ M_A，计算其与所有高重要性token b_j ∈ M_B的余弦相似度，选择最相似的b_j作为匹配目标f_i，记录最大相似度g_i。按g_i排序后仅保留top-p%的最相似匹配对。设计动机：a_i与其counterpart f_i越相似，a_i的语义信息已在f_i中充分表示，可被安全删除或融合而丢失信息最少。与bipartite matching（强制一对一匹配不考虑质量）的关键差异：相似度阈值作为质量闸门，低相似度的pair被拒绝。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
A_norm = normalize(M_A, dim=-1)        # [N/2, D]
B_norm = normalize(M_B, dim=-1)        # [N/2, D]
sim_matrix = A_norm @ B_norm.T         # [N/2, N/2] pairwise cosine
g_values, f_indices = max(sim_matrix, dim=-1)  # row-wise best match
sorted_pairs = argsort(g_values, descending=True)
keep_pairs = sorted_pairs[:int(p * N/2)]       # top-p% 过滤
# g_i低: a_i无法在任何M_B中找到好匹配
# → 包含独特关键信息, 不能prune/merge
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch实现：F.normalize + matmul一次性计算所有pairwise cosine。p值由目标FLOPS reduction反推。与bipartite matching (ToMe)的核心差异：质量闸门过滤掉低质量匹配，对被拒绝的a_i不执行任何破坏性操作。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---

## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA (Low-Rank Adaptation) 由 Hu 等人 (2022, ICLR) 提出，是应用最广泛的 PEFT 方法之一。核心思想：预训练模型的权重更新 ΔW 具有内在低秩属性，即大模型在适配下游任务时实际有效自由度远低于全参数空间维度。因此无需更新全部参数，只需在冻结的原始权重 W_0 旁引入一对低秩分解矩阵：W_a ∈ R^{d×r}（降维投影）、W_b ∈ R^{r×d}（升维还原），其中 r << min(d, k)，典型 r=8。前向计算为：`y = xW_0 + xW_aW_b`。训练时仅更新 W_a 和 W_b（W_a Kaiming 随机初始化，W_b 零初始化，保证 ΔW 初始为零等价于原模型）。推理时可将 ΔW = W_aW_b 合并到 W_0 中（`W' = W_0 + W_aW_b`），零额外推理延迟。标准 LoRA 稠密插入所有线性层（attention Q/K/V + FFN + classifier），如 RoBERTa-base 产生约 1.3M 可训练参数（rank=8），仅为全微调的 ~1%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# LoRA 训练前向（以单层 attention Q 矩阵为例）
x = input_hidden_states          # [batch, seq, d]
W_0 = pretrained_query_weight    # [d, d], frozen
W_a = nn.Parameter(randn(d, r) * scale)  # [d, r], trainable
W_b = nn.Parameter(zeros(r, d))          # [r, d], trainable

x_lora = x @ W_a                 # [batch, seq, d] × [d, r] → [batch, seq, r]
delta = x_lora @ W_b             # [batch, seq, r] × [r, d] → [batch, seq, d]
y = x @ W_0 + delta * (alpha / r)  # alpha=16 为缩放因子

# 仅 W_a/W_b 接收梯度，W_0 保持冻结
# 推理时合并：W_merged = W_0 + (alpha/r) * W_a @ W_b
```
关键参数：rank r（控制低秩空间维度，典型值 8-16）、alpha（缩放因子，控制 ΔW 贡献幅度，默认 16）、dropout rate（默认 0.1）、target_modules（插入位置——标准为 attention 全部 + FFN）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：HuggingFace PEFT 库（`LoraConfig`, `get_peft_model`）。典型用法：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","out_proj","fc1","fc2"], lora_dropout=0.1)` → `model = get_peft_model(base_model, config)` → 标准 Trainer 训练 → `model.merge_and_unload()` 合并权重。主要变体：AdaLoRA（SVD 自适应分配 rank）、QLoRA（4-bit 量化 + LoRA）、DoRA（权重方向-幅度分解）、LoRA+（差异化 lr）、SoRA（稀疏 LoRA，pruning 减少参数）、SSMLoRA（SSM 连接跨层低秩矩阵）。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---

## PEFT (Parameter-Efficient Fine-Tuning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parameter-Efficient Fine-Tuning (PEFT) 是一类方法的统称，旨在仅更新预训练模型极小部分参数（通常 <1%）来适配下游任务，冻结绝大部分预训练权重。相比全参数微调（fine-tuning），PEFT 的核心优势：(1) 显著降低训练存储——仅需存储和更新少量 adapter 参数（如 LoRA 的 W_a/W_b）；(2) 减轻灾难性遗忘——冻结原权重使模型在通用能力上的退化更少；(3) 低数据场景表现更优——参数空间受限起到正则化效果；(4) 多任务部署灵活——同一 base model + 不同 adapter 可服务多种任务。主要技术类别包括：Adapter 类（在 Transformer 层间插入小型 bottleneck 模块，如 Houlsby 2019）、LoRA 类（低秩分解旁路）、BitFit（仅训 bias）、Prefix/Prompt Tuning（在输入前添加可训练虚拟 token）、IA3（缩放激活值）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# PEFT 通用流程（以 HuggingFace PEFT 库为例）
from peft import get_peft_model, LoraConfig

# 1. 配置 PEFT 方法
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])

# 2. 注入 adapter
model = get_peft_model(base_model, config)
# → 冻结 base_model 所有参数，仅 adapter 参数可训练
print(model.print_trainable_parameters())  # trainable: 1.0M / total: 125M (0.8%)

# 3. 标准训练（仅 adapter 参数更新）
trainer = Trainer(model=model, ...)
trainer.train()

# 4. 保存/加载（仅 adapter 权重，~几MB vs 全模型几GB）
model.save_pretrained("./lora_adapter")
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：HuggingFace PEFT 库（GitHub: huggingface/peft），支持 LoRA/QLoRA/DoRA/AdaLoRA/IA3/PromptTuning/PrefixTuning/BitFit 等。安装：`pip install peft`。在 SSMLoRA 论文中，PEFT 作为实验对比框架——SSMLoRA 属于 LoRA 变体，通过引入 SSM 状态转移和稀疏插入进一步降低参数。SSMLoRA 在 GLUE 上仅 1.0M 参数（vs LoRA 1.3M）实现可比/更优性能。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---

## Sparse Insertion in Low-Rank Adaptation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Insertion in Low-Rank Adaptation 是 LoRA 变体中通过选择性插入低秩矩阵来减少可训练参数的技术。核心动机：研究（如 SoRA, Ding et al. 2023）发现并非所有层的低秩适配对下游任务等量贡献——稠密全层插入导致参数浪费。稀疏插入策略通过仅在部分关键位置上激活 LoRA adapter，在维持性能前提下大幅减少参数。SSMLoRA 实现的"交替间隔稀疏插入"具体设计：(1) 仅插入 attention 的 query 和 value 矩阵，key 始终不插；(2) 相邻 encoder 层交替激活——layer l 激活 Q Time Module、layer l+1 激活 V Time Module、layer l+2 激活 Q...；(3) Q 和 V 各自维护独立时间轴（避免跨类型状态干扰）；(4) 非 attention 层（FFN/classifier）使用标准稠密 LoRA（无 SSM 状态）。参数压缩效果：RoBERTa-base 上 1.0M（vs LoRA 1.3M，~77%），LLaMA2-7B 上 15.8M（vs LoRA 20.0M，~79%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# SSMLoRA 中的交替间隔稀疏插入策略
for layer_idx in range(num_layers):
    if layer_idx % 2 == 0:   # 偶数层：激活 query Time Module
        q_output = time_module_q(x, h_q_state)   # SSM 状态增强
        v_output = W_0_v(x)                      # value 跳过（原始 forward）
    else:                     # 奇数层：激活 value Time Module
        q_output = W_0_q(x)                      # query 跳过
        v_output = time_module_v(x, h_v_state)   # SSM 状态增强
    k_output = W_0_k(x)                          # key 始终直接使用原始权重

# 同类矩阵沿独立时间轴累积状态信息
# h_q 只在偶数层的 Q Time Module 间传递
# h_v 只在奇数层的 V Time Module 间传递
```
与 SoRA 的动态 pruning 对比：SoRA 通过重要性评分自适应决定每个 rank 方向的参与度；SSMLoRA 使用固定结构规则（不引入额外计算开销判断重要性），依赖 SSM 状态传递补偿稀疏带来的信息损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两种实现范式：(1) 动态 pruning 类——如 SoRA，在训练中自适应学习 importance score 并逐步剪除低重要性 adapter；(2) 结构化稀疏调度类——如 SSMLoRA 的固定交替规则，实现极简无需额外开销。选择依据：结构化方法适合需要确定性参数预算的场景；动态方法适合需要自适应分配不同层参数容量的场景。SSMLoRA GLUE/SuperGLUE 消融实验验证 rank r=1-16 范围内，稀疏 ~50% 参数的 SSMLoRA 在多数任务上匹配或超越稠密 LoRA。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---

## Taylor Expansion for State Discretization in SSM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Taylor Expansion for State Discretization 是 SSMLoRA 提出的一种替代 S4 标准 Zero-Order Hold (ZOH) 离散化的 SSM 状态更新方法。在标准 S4 中，连续状态空间方程 `h'(t) = Ah(t) + Bx(t)` 需通过 ZOH 将连续参数 A、B 离散化：`Ā = exp(ΔA)`、`B̄ = (ΔA)^{-1}(Ā - I)ΔB`，需矩阵指数和矩阵求逆。SSMLoRA 利用 Taylor 展开对状态直接一阶离散化：`h_t = h_{t-1} + h'_{t-1}·Δt`（取 Δt=1），即 `h_t = h_{t-1} + (h_{t-1}·W_c + x_new·W_d)`。核心优势：避免矩阵指数/求逆（仅需 O(r²) 矩阵乘法）、参数无需离散化保持可训练、h_t 可 detach 节省显存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# 对比：Taylor 展开 vs ZOH 离散化，均以 r=8 为例

# === SSMLoRA Taylor 展开（O(r²)，零额外开销）===
h_prime = h_prev @ W_c + x_new @ W_d    # [b,r] @ [r,r] + [b,r] @ [r,r] = [b,r]
h_new = h_prime + h_prev                # 一阶 Taylor: h += h'·1
# W_c,W_d: r×r=64 params each，共128 params（完全可忽略）

# === S4 ZOH 离散化（O(r³)，需要 matrix_exp + inverse）===
A_bar = torch.linalg.matrix_exp(delta * W_c)  # O(r³) expm
B_bar = torch.linalg.inv(W_c) @ (A_bar - I) @ W_d  # O(r³) inverse
h_new = A_bar @ h_prev + B_bar @ x_new
```
关键设计：h_t 在 SSMLoRA 中脱离计算图（detach），使 SSM 部分不参与反向传播——仅 W_c、W_d、W_a、W_b 接收梯度。结合零初始化策略（W_c、W_d 初始为零），训练初期 h' 为零，h 保持零向量，模型退化为稀疏 LoRA，逐渐学习非零状态转移。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSMLoRA 中的具体实现：`self.W_c = nn.Linear(r, r, bias=False)`、`self.W_d = nn.Linear(r, r, bias=False)`（r=8 典型值），前向：`h_new = self.W_c(h_prev) + self.W_d(x_new) + h_prev`。ZOH 离散化（如 S4/Mamba 中使用）更适合精度敏感的长序列建模场景（严格数学推导保证数值稳定性）；Taylor 离散化更适合结合 LoRA——计算轻量、实现简单、小 rank 下精度损失可忽略。SSMLoRA 论文 Table 12 验证 LLaMA2-7B 推理开销与 LoRA 接近（4096 tokens: 2.740s vs LoRA 2.210s）。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---

## SSM-augmented Low-Rank Adaptation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSM-augmented Low-Rank Adaptation 是一类将 State Space Model (SSM) 的状态转移机制集成到 LoRA 低秩适配中的方法，SSMLoRA 是其代表性实现。核心创新：在标准 LoRA 仅关注"当前层输入→低秩空间→输出"的逐层独立适配基础上，引入沿时间轴跨层传递的状态向量 h_t，使第 l 层的低秩映射能够利用来自 l-1 层的上下文信息。关键技术组件：(1) Time Module 包含标准 LoRA 矩阵 W_a/W_b + SSM 状态矩阵 W_c/W_d + 持久化状态向量 h；(2) 状态更新采用 Taylor 展开离散化 `h_t = h_{t-1}·W_c + x_new·W_d + h_{t-1}`；(3) 归一化后的 h_t 作为低秩空间的偏置调整输出 `y = xW_0 + (x_new + h_t_norm)·W_b`。设计目标：通过跨层状态共享提升参数利用率，使稀疏插入（~50% 参数）仍能维持甚至超越稠密 LoRA 性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Time Module 完整 pipeline（SSMLoRA 核心计算单元）
class TimeModule:
    """
    d: model hidden dim, r: low-rank dim (typically r=8, r << d)
    State space 沿时间轴传递: 同类矩阵共享时间轴，异类矩阵独立时间轴
    """
    def __init__(self, d, r):
        self.W_a = nn.Linear(d, r, bias=False)   # LoRA 降维
        self.W_b = nn.Linear(r, d, bias=False)   # LoRA 升维（零初始化）
        self.W_c = nn.Linear(r, r, bias=False)   # SSM 状态矩阵（零初始化）
        self.W_d = nn.Linear(r, r, bias=False)   # SSM 控制矩阵（零初始化）
        self.h = torch.zeros(1, r)               # 状态向量（零初始化）

    def forward(self, x: [B,S,D]) -> [B,S,D]:
        x_new = self.W_a(x)                         # Step 1: [B,S,D]→[B,S,R]
        # 广播 h 匹配 batch+seq
        h_prev = self.h.expand(B, S, R)
        h_prime = h_prev @ self.W_c.weight.T + x_new @ self.W_d.weight.T  # Step 2: 状态导数
        h_new = h_prime + h_prev                     # Step 3: Taylor 展开
        self.h = h_new[:, -1:, :].detach()           # 取最后 token 状态传入下层
        # Step 4: min-max 归一化
        h_min, h_max = h_new.min(), h_new.max()
        h_norm = (h_new - h_min) / (h_max - h_min + 1e-8)
        output = (x_new + h_norm) @ self.W_b.weight.T  # Step 5: 偏置调整 + 升维
        return output

# 初始状态：W_b=W_c=W_d=h=0 → y = xW_0 + 0 → 等价原模型
# 随着训练进行：状态转移逐渐学习跨层关联
```
零初始化策略保证训练起点退化为稀疏 LoRA，避免早期引入噪声。随着训练进行，模型通过 W_c/W_d 逐渐学习跨层状态关联，实现从"稀疏独立适配"到"SSM 连接适配"的平滑过渡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yuhkalhic/SSMLoRA (NAACL 2025, Python 3.10 + PyTorch)。训练命令：`python src/main.py --dataset BoolQ`。关键设计决策：(1) Q/V 独立时间轴——防止不同语义角色的状态互相干扰；(2) min-max 归一化——`(h - min)/(max - min + ε)` 稳定数值，将 h_t 限定在合理范围；(3) 无需 FFT 优化——r 仅为 8，r×r 矩阵乘法开销可忽略。实验验证 GLUE 上 ~50% 参数达稠密 LoRA 性能；NarrativeQA 1000+ tokens 序列 ROUGE-L 超 LoRA 2.1%；RACE high-difficulty Acc 67.37 > LoRA 65.64。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---

## Sliding Window Attention (SWA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding Window Attention (SWA) 是一种稀疏注意力模式（Beltagy et al., 2020），每个 token 仅关注其前后固定窗口 w 内的 token，将注意力复杂度从 O(n²) 降至 O(n·w)。与全注意力不同，SWA 具有序列长度的平移不变性——任意长度序列的每 token 计算量恒定。在 SAMBA 中，SWA 窗口大小 w=2048，使用 FlashAttention 2 高效实现，配合 RoPE（base=10,000）编码相对位置。选择 w=2048 的关键原因：FlashAttention 2 在 seqlen=2048 时训练速度与 Mamba 的 selective parallel scan 相当（基于 Gu & Dao 2023 测量），使混合架构不会引入瓶颈层。训练序列长度设为 4096 = w×2（SAMBA 发现这是最优比，Table 9）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SWA 层前向（Samba 中使用）
Input: X ∈ R^{n × d_m}, window_size=2048, RoPE base=10000

Q = X @ W_q   # [n, n_heads × d_head]
K = X @ W_k
V = X @ W_v
Q, K = RoPE(Q, K, base=10000)   # Rotary Position Embedding

# FlashAttention 2 with causal sliding window
# 对每个位置 i, attention 仅在 [max(0, i-w+1), i] 范围内
O_swa = FlashAttention2(Q, K, V, causal=True, window_size=(w, 0))
O = O_swa @ W_o   # output projection
```
关键特性：(1) 计算复杂度 O(n·w) 而非 O(n²)；(2) 平移不变性——模型在训练长度外的序列上仍表现良好；(3) 窗口内保留精确 softmax attention，可精确召回近期记忆；(4) 无法直接访问超出窗口的历史 token，需依赖 SSM 层的递归状态压缩来传递长程信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) FlashAttention 2 原生支持 SWA 的 window_size 参数；(2) Hugging Face Transformers 中通过 `attention_mask` 或 `sliding_window` 配置参数支持。SAMBA 纯用 SWA（无 global attention tokens），依赖 Mamba 层处理超出窗口的长程依赖。SWA 适用于：(a) 需要 O(n) 复杂度的长文档处理；(b) 混合架构中作为精确检索组件；(c) 长度外推——训练长度 4K 的 SWA 模型在更长序列上 perplexity 自然下降（Table 3: Llama-2-SWA 在 16K 时 10.57 vs Llama-2 的 249.03）。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## Length Extrapolation in Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Length Extrapolation（长度外推）指语言模型在超过预训练序列长度的上下文上仍能保持或改善性能的能力。传统 Transformer（全注意力）在超训练长度时 perplexity 爆炸（Table 3: Llama-2 438M 在 16K 时 perplexity 从 11.14→249.03）。SAMBA 通过 SWA 的平移不变性 + Mamba 的递归压缩实现高效外推：仅用 4K 训练长度，零样本外推到 1M（256× 外推率）时 Proof-Pile perplexity 持续改善（Figure 2a）。Passkey Retrieval 上：Samba 1.7B 仅用 4K 长度 500 步微调即可外推到 256K（64× 外推率）且准确率完美（Figure 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
核心原理——SWA 的长度外推能力来自 RoPE 的相对位置编码 + 固定窗口的平移不变性：训练时（seqlen=4096, window=2048），对任意位置 i，SWA 只关注 [max(0,i-2048), i]，RoPE 编码的是相对距离而非绝对位置 → 天然平移不变；推理时（seqlen=1M）计算模式与训练时完全相同 → 无分布外问题。Mamba 递归状态累积全部历史信息 → 绑定短期精确 + 长期压缩。关键前提：RoPE 对 SWA 长度外推至关重要——Samba-NoPE 在 16K 时 perplexity 爆炸至 314.78（Table 3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流外推方法：(1) PI（线性缩放位置索引）——需要微调；(2) NTK-aware——修改 RoPE base frequency，零样本可用；(3) SelfExtend——用 group attention + neighbor window，零样本但增加延迟；(4) SWA 从零训练——最干净但需重新预训练。SAMBA 的方法属于 (4)：从零预训练就包含 SWA，使模型"原生"支持外推。评估方法：(a) perplexity 外推——Proof-Pile 测试集 sliding window 评估；(b) 检索外推——Passkey Retrieval 和 Phonebook；(c) 长文本任务——GovReport/SQuALITY 摘要。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---

## Gated Linear Attention (GLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gated Linear Attention (GLA) 是由 Yang et al. (2024, ICML 2024) 提出的线性注意力变体，通过数据依赖的 2D 遗忘门增强标准线性注意力的表达能力。核心公式：S_t = G_t ⊙ S_{t-1} + K_t^T V_t，其中 G_t ∈ R^{d×d} 是输入依赖的门控矩阵（在 channel 和 head 两个维度上运作）。相比标准线性注意力（等权累积），GLA 的门控允许模型选择性"遗忘"不相关信息。配套硬件高效实现为 FlashLinearAttention（FLA），使用分块策略：块内矩阵乘法（利用 Tensor Core），块间递归传递状态。

在 SAMBA 中：Sliding GLA 替换 Samba 架构中的 Mamba 层进行消融（Table 3）。438M 规模上 GLA 的 perplexity（10.43/10.00/9.92 at 4K/8K/16K）优于 Mamba（10.70/10.30/10.24）但不如 Samba（10.06/9.65/9.57）。GLA 训练速度（4.94×10^5 tokens/s）显著快于 Mamba（2.46×10^5），因为 Mamba 层数更多且 scan kernel 开销大。GLA 加入短卷积改善不明显（10.43→10.39），因已有 channel 级细粒度门控。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# GLA 核心递归（简化）
S_0 = 0  # [d, d]
for t in 1..n:
    G_t = sigmoid(Linear_g(X_t))  # 输入依赖的门控
    S_t = diag(G_t) @ S_{t-1} + K_t^T @ V_t   # [d, d]
    o_t = Q_t @ S_t                             # [d]
```
训练时使用分块并行化：chunk 内矩阵乘法并行，chunk 间递归传递 S_t。FlashLinearAttention 库提供 Triton 实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：Flash Linear Attention (FLA) 库（https://github.com/sustcsonglin/flash-linear-attention），Triton kernel。GLA 在 1.3B 规模保持良好长度外推：训练于 4K，16K 时 perplexity=7.19（优于 Mamba 7.15，弱于 Samba 6.96，Table 3）。相比 Mamba：训练速度更快（不需要复杂 scan kernel）、可与 SWA 直接组合；下游任务略弱于 Mamba-based 混合模型。适用场景：需要训练速度优先于极致下游性能的大规模 LM 预训练。

涉及论文标题：
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling


---

## Sandwich Prompt

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sandwich Prompt 是 VisualRWKV 针对 RNN-based VLM 设计的一种多模态提示策略。传统 Transformer VLM（如 LLaVA）由于 self-attention 机制可以随时访问任意历史 token，对 prompt 的 image token 位置不敏感。但 RNN 模型（如 RWKV）因其序列特性无法"回溯"已处理的信息——模型看到 token 后立即决定是否存入固定大小的 hidden state，无法直接访问原始输入。Sandwich Prompt 将 image token 插入 instruction token 中间，形成"指令前缀 → 图像 → 指令后缀"的三明治结构。前半段指令帮助模型确定从图像中提取什么信息（激活正确的检索意图），后半段指令确保问题在图像处理完成后仍被牢记。实验证明 Sandwich Prompt 显著优于 Image First（图像在前，模型处理图像时不考虑问题）和 Image Last（图像在后，模型先读问题但被图像 token 覆盖后遗忘问题）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种 prompt 方法的对比伪代码：
```
# Image First Prompt：
Input = [<image_tokens> | System | Question | "### Assistant:"]
# 问题: 模型处理图像时尚未读到 question，缺少上下文引导

# Image Last Prompt：
Input = [System | Question | <image_tokens> | "### Assistant:"]
# 问题: 模型读到问题后在处理 576 个 image tokens 期间，RNN state 逐渐遗忘问题内容

# Sandwich Prompt (最优)：
Input = [System | "### Human:" | <image_tokens> | "\nQuestion: ...\n### Assistant:"]
# "### Human:" 激活回答意图 → 读图时带着意图提取相关特征
# → "\nQuestion:" 再次提醒问题内容 → 生成答案
```
Sandwich Prompt 在减少 image tokens 时表现出更强的鲁棒性（Table 9），这是因为它建立了"两端指令夹图像"的信息冗余——即使中间的图像信息被压缩，两端的文本指令仍能保持语义锚定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 tokenized 序列构建阶段，将 vision encoder 输出的 visual tokens（576 个 for CLIP-L/14@336×336）插入到 tokenized text instruction 的指定位置。训练和推理时 Sandwich Prompt 保持一致格式。VisualRWKV 7B 上 Sandwich Prompt 比 Image First 在 ScienceQA 上提升 +5.49 点（69.71 vs 65.59？实际上 Table 3 显示 Image First 67.93 vs Sandwich 69.71）。特别适用于 RNN/SSM 架构的 VLM，但设计理念也可推广到 Transformer VLM 中优化长距离信息保留。

涉及论文标题：
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## 2D Image Scanning (Multi-directional Scanning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D Image Scanning 是 VisualRWKV 针对 RNN VLM 的视觉序列处理机制。RWKV（及其他线性 RNN）本质上为 1D 因果语言序列设计，其 Scan 操作假定序列具有因果方向性。但视觉 encoder（如 CLIP ViT）生成的 visual tokens 来自 2D patch grid，天然是双向/多向的非因果序列。若直接用单向 Scan（Forward-only），模型只能从左上到右下依次处理，丢失了大量空间上下文。2D Image Scanning 通过在相邻 RWKV layers 中交替排列不同扫描方向（Forward/Backward/Upward/Downward），使不同层的 RWKV blocks 从不同方向"看"图像，从而在不增加任何参数和计算开销的情况下获得等效的 2D 空间感知能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种扫描变体的 layer 排列：
```
# Unidirectional (UniDir) - VisualRWKV-Base baseline
Layer 0: Forward Scan    Layer 1: Forward Scan    
Layer 2: Forward Scan    Layer 3: Forward Scan    ...

# Bidirectional (BiDir) - 论文最优
Layer 0: Forward Scan    Layer 1: Backward Scan   
Layer 2: Forward Scan    Layer 3: Backward Scan   ... 交替

# Multidirectional (MultiDir) - 四向交替
Layer 0: Forward         Layer 1: Backward        
Layer 2: Upward          Layer 3: Downward        ... 循环
```
Forward/Backward：按 patch 的 row-major 顺序正向/反向扫描。Upward/Downward：将 2D patch grid 转置后正向/反向扫描（按列扫描）。每种扫描方向等价于对 visual token 序列做特定的 permutation。关键实现细节：训练和推理时的扫描方向必须保持一致（论文尝试过动态重排 layer 顺序但性能不稳定，因为特定 layer 已"专业化"于处理特定方向的视觉信息）。交替扫描在 layers 间形成互补——例如 Forward 层的输出被 Backward 层看见，使 Backward 层能融合前向上下文做反向推断。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 RWKV block 的输入处理中，根据当前 layer index 对输入的 visual token 序列做 permutation（Forward=none, Backward=flip, Upward=transpose+forward, Downward=transpose+backward）。每个 RWKV block 内部的 WKV scan 操作不变，仅仅是输入 token 顺序被重排。零参数开销，零额外 FLOPs。实验结果（Table 4）：BiDir VQA 65.62 > UniDir 51.03 (+14.59)，MultiDir 66.04 > UniDir (+15.01)。BiDir 在大多数 benchmark 上表现最好，是多向扫描中最高效的配置。该技术仅适用于 visual tokens——text instruction tokens 保持单向 Forward scan 不变（保持语言因果性）。论文确认已开源：https://github.com/howard-hou/VisualRWKV。

涉及论文标题：
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---

## Visual Instruction Tuning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Instruction Tuning 是 LLaVA（Liu et al., 2023a/b）提出的两阶段 VLM 训练范式，被 VisualRWKV 继承使用。第一阶段（Vision-Language Alignment Pretraining）：冻结 vision encoder 和 LLM，仅训练一个简单的 projector（通常为 1-2 层 MLP），使 vision features 能被 LLM 理解。数据通常为大规模 image-caption 对（如 558K LAION-CC-SBU subset）。第二阶段（Visual Instruction Tuning）：解冻 LLM（部分或全部），使用视觉指令数据（如 GPT-generated multimodal conversations + academic VQA）同时训练 projector 和 LLM，使模型学会遵循包含图像的指令。这种两阶段设计平衡了训练效率和最终性能：第一阶段廉价地将视觉特征映射到语言空间，第二阶段精调交互能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage 1: Feature Alignment (Pretraining)
Freeze: Vision Encoder (CLIP-L), LLM (RWKV)
Train:   Projector (2-layer MLP: [1024→D_llm])
Data:    558K LAION-CC-SBU image-caption pairs
LR:      1e-3 (projector only)
Goal:    Align vision features to language embedding space

# Stage 2: Visual Instruction Tuning
Freeze:  Vision Encoder (CLIP-L)
Train:   Projector + LLM (RWKV)
Data:    150K GPT-generated instruction data + ~515K academic VQA
LR:      4e-5 (7B model, cosine decay to 1e-5)
Goal:    Teach model to follow multimodal instructions

# 关键发现 (VisualRWKV):
# - 两阶段训练优于单阶段训练 (Figure 5)
# - RWKV 需要比 Transformer 更高的 learning rate (4e-5 vs 2e-5 for 7B)
# - Sample-level loss reduction 优于 batch-level reduction (Table 7)
```
VisualRWKV 与 LLaVA-1.5 使用完全相同的训练数据（558K + 665K）和训练策略，确保公平比较——性能差异仅来自架构差异（RNN vs Transformer）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLaVA-1.5 开源实现（https://github.com/haotian-liu/LLaVA）提供标准的训练脚本和数据 pipeline。VisualRWKV 在 LLaVA 的训练框架基础上，替换 LLM backbone 为 RWKV-6，增加 Sandwich Prompt 构建逻辑和 2D Image Scanning。训练使用 DeepSpeed ZeRO stage 1/2 分布式训练，NVIDIA PyTorch NGC Container (23.07-py3)。VisualRWKV 7B 两阶段训练总计约 318 GPU hours（159 × 2 epochs on 6×A100）。两阶段中 projector 学习率远高于 LLM（1e-3 vs 4e-5），体现 feature alignment 只需粗调、instruction following 需精调的设计理念。

涉及论文标题：
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models
