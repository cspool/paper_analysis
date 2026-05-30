## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- baseline方法是什么？
  - **AWQ (Lin et al., 2024b)**：广泛使用的 W4A16 权重量化方法，采用 channel-wise scaling（逐通道缩放因子）对权重进行预处理以抑制离群通道，缩放因子通过 grid search 优化，且可完全合并到前序算子中，推理零开销。缺陷：(1) 仅使用逐通道缩放，无法利用跨通道交互来进一步收窄组内动态范围——当离群值在通道内分散而非集中在特定通道时效果有限；(2) 在推理 LLM 的长链思维（CoT）生成中，每个解码步的量化误差会累积，导致在推理任务（如 MMLU-Pro）上精度显著下降（Qwen3-4B FP16 71.0 → AWQ 68.2）。
  - **QTIP (Tseng et al., 2024b)**：SOTA 向量量化方法，采用随机 Hadamard 变换 + trellis 量化算法。Hadamard 变换是全旋转矩阵（O(n log n) 复杂度），可跨通道交互消除离群值。缺陷：(1) Hadamard 变换固定或由随机向量生成，忽略各层权重分布的独特性；(2) Hadamard 变换仍有较大推理开销（比 AWQ 慢约 30%），因为变换在全局 channel 维度上有依赖关系，无法充分利用 GPU 并行性。
  - **SpinQuant (Liu et al., 2025b)**：将旋转矩阵合并到前序线性层权重中以避免推理开销，但仅适用于少数可合并层（如 output projection），decoder block 中多数线性层前有 element-wise 算子或残差连接，无法吸收矩阵乘法。
  - 全栈执行例子（以 AWQ W4A16 量化 Qwen3-4B 在 MMLU-Pro 上的推理为例）：
    - **算法层**：加载 FP16 权重 W → 校准集上逐 channel 计算激活幅值 s = mean(|X|)（activation-aware）→ grid search 优化逐通道缩放因子 α（缩放范围 [0.5, 1.5]）→ W' = diag(α)·W → INT4 均匀量化（group=128）：s_q = (max(W'_g)-min(W'_g))/15, W_q = round((W'_g-min)/s_q) → 推理时缩放因子合并到前序 LayerNorm/激活 X' = X·diag(1/α)。
    - **系统框架层**：PyTorch + Transformers + vLLM serving → 量化权重存储、GEMM 用 INT4 kernel。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：AWQ 提供高效 W4A16 GEMM kernel（Triton/CUDA），无需额外 transform kernel（channel-wise scaling 已合并）。
    - **硬件架构层**：NVIDIA GPU (RTX A6000/4090/H200)，论文未涉及硬件设计。
  - Baseline 核心缺陷：(1) **仅 scaling 不足以消除复杂离群值模式**——AWQ 的逐通道缩放只调整每个通道的整体幅值，无法处理通道内 token 级别的数值分散；(2) **全旋转过于昂贵**——QTIP/QuIP# 的 Hadamard 变换虽有跨通道交互能力但推理开销大（约 30% slowdown）；(3) **合并旋转的范围受限**——SpinQuant 的旋转合并策略只适用于少数线性层；(4) **推理 LLM 的长生成使误差累积放大**——现有方法在设计时未充分考虑 CoT 生成场景。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **ParoQuant** 通过三个核心设计系统性解决 baseline 的精度-效率权衡困境：
    1. **Scaled Pairwise Rotation（独立 Givens 旋转 + 逐通道缩放）**替代仅 Scaling 或全旋转：
       - 在逐通道缩放（拉平整体幅值）基础上，叠加 K=8 个 **independent rotations**（每个由 group_size/2=64 对互不重叠的 Givens 旋转组成），实现 **稀疏参数化的跨通道交互**——仅旋转幅值差异大的通道对（实验证明 top 10% 关键对的表达能力与全旋转几乎等价，Figure 2）。对应解决缺陷 (1)：既保留 scaling 拉平全局幅值的能力，又通过旋转收窄每对通道内 token 级别的数值分散（Figure 1 Right，数据点聚集到 x=y 线附近）。
    2. **Independent Rotation 约束（无依赖、全并行）**替代 Hadamard 全局依赖：
       - 强制每个 rotation 内的通道对互不重叠（每个通道最多参与一对），使所有 Givens 旋转完全并行化且无需同步。K 个 rotation 按顺序应用，但在一个 fused kernel 内完成（一次加载激活到 shared memory，8 次旋转均在 shared memory 上执行）。对应解决缺陷 (2)：推理开销仅约 10%（vs Hadamard 的 30%），且 channel 维度越大加速比越显著（Figure 4）。
    3. **两阶段逐层优化 + 混合校准集**替代 grid search/固定变换：
       - Stage 1：用 AdamW 优化旋转角度和缩放因子（而非 grid search），基于 2048 个多样化校准样本（WikiText2+C4+RedPajama 均匀混合），最小化量化层输出误差。
       - Stage 2：QAT-like 微调权重和量化参数，进一步消除 Stage 1 后残留的孤立离群值。
       - 逐层使用已量化前层的输出 X' 作为校准输入，使后续层能补偿前层累积的量化误差。对应解决缺陷 (3)(4)：每层独立学习最优变换参数，且考虑长生成中的误差传播。
  - 全栈执行例子（以 ParoQuant W4A16 量化 Qwen3-4B 在 MMLU-Pro 上推理为例，对比 AWQ）：
    - **算法层**：FP16 Qwen3-4B → 分组（group=128）→ 配对选择（Algorithm A1: shuffle 后贪婪选互不重叠 pair，跨 rotation 跳过已选 pair）→ Stage 1: AdamW 优化 θ∈R^{K×64} 和 α∈R^{128}，minimize ||l'(X')-l(X)|| → 量化：s=range/15, z=-round(min/s), W_q=clamp(round(T(W)/s)+z,0,15) → Stage 2: AdamW 微调 W, s, z → 推理：X → T^{-1}(X) = X·diag(1/α)·R_1^{-1}·...·R_K^{-1}（fused CUDA kernel, 3-level parallelism）→ INT4 GEMM (AWQ kernel) → Y。对比 AWQ：AWQ 仅 diag(α)·W → 量化 → 推理（X 直接做 INT4 GEMM，无 transform kernel，α 已合并到前序 op），无旋转、无 Stage 2 微调。
    - **系统框架层**：PyTorch 2.8.0 + Transformers 4.55.2（量化优化）→ PyTorch 2.6.0 + torch.compile max-autotune + CUDA Graphs（推理）→ vLLM 0.10.1 + Lighteval 0.8.1（推理任务评测）。对比 AWQ：统一在 Transformers 框架上仅替换量化层实现。
    - **编译框架层**：torch.compile max-autotune 用于推理图优化，论文未涉及自定义编译器。
    - **kernel调度层**：Fused CUDA kernel（token/group/pair 三级并行，shared memory 常驻）→ AWQ W4A16 GEMM kernel。对比 AWQ：AWQ 无额外 transform kernel（α 已合并），直接调用 GEMM kernel。ParoQuant 多一个 transform kernel（~10% 开销）但换取显著精度提升。
    - **硬件架构层**：NVIDIA H200 (训练)、RTX A6000/6000 Ada/4090 (推理)，论文未涉及硬件设计。
  - 关键结果：
    - 推理任务平均精度：ParoQuant 仅降 0.9%（FP16→W4），vs AWQ 降 3.3%、EfficientQAT 降 7.2%。
    - MMLU-Pro Qwen3-4B：ParoQuant 70.1 vs AWQ 68.2 vs QTIP 69.7。
    - 吞吐：ParoQuant 比 AWQ 慢约 10%，比 QTIP 快约 25%（Qwen3-4B: 160 vs 176 vs 117 tokens/s）。
