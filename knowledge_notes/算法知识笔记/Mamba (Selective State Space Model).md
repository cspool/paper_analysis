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
