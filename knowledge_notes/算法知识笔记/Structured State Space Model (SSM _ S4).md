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
