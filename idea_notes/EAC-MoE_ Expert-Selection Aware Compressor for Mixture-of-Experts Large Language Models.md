## EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  Baseline 是**直接对 MoE-LLM 使用 dense LLM 的标准量化和剪枝方法**，即：(1) **GPTQ 均匀位宽量化**：对所有 expert 施加相同位宽量化（2/3-bit），不考虑 MoE 路由器的 expert selection 特性，导致量化后路由器选错 expert（expert-shift 问题），模型精度严重退化。(2) **静态混合精度量化（PMQ/BSP）**：基于校准集统计 expert 选择频率分配不同位宽，但忽略了不同任务类别中 expert 重要性截然不同的规律，导致严重的跨任务过拟合。(3) **逐 token 动态剪枝（EES/ODP）**：对每个 token 剪枝贡献度最小的 expert，但仅减少部分 expert 的输入大小，加速效果有限（~5-8%），且未利用序列级 expert 选择频率的稀疏性。

  Baseline 全栈执行例子（以 Mixtral-8x7B 推理为例）：
  - **算法层**：标准 GPTQ 量化：W_fp16 → W_intB（group-wise asymmetric, 128 groupsize），使用 Hessian-based 误差补偿，但 MoE router 保持量化前权重不变。量化后每个 token 通过 router 选择 top-2 expert，router 输出因量化噪声偏离全精度模型，导致选错 expert（expert-shift）。以 3-bit GPTQ 量化为例，量化本身导致 PPL 从 3.84 升至 4.16，expert-shift 进一步恶化至 4.65。
  - **系统框架层**：HuggingFace Transformers + GPTQ 推理。所有 8 个 expert 的全精度权重加载到 GPU memory（~94GB），显存压力大。推理时每 token 计算 router logits（MatMul）→ Softmax → Top-2 选择 → 2 个 expert FFN 前向传播 → 加权求和。无论 expert 是否被频繁选择，所有 expert 权重均需常驻显存。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（标准 cuBLAS GEMM 执行 MoE expert 计算）。GPTQ 量化后使用 BitBLAS 处理 INT 权重的混合精度 BLAS 操作。
  - **硬件架构层**：NVIDIA A100 40G GPU（量化）/ RTX 3090（部署）。直接 GPTQ 量化的 Mixtral-8x7B 在 3.03-bit 下仍需 ~19GB 显存，精度损失~3.7%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **EAC-MoE = QESC + PESF**，从 MoE 模型最核心的 expert selection 机制出发，分别从"确保选对 expert"（量化前校准）和"跳过不重要 expert"（推理中剪枝）两个维度解决 Baseline 缺陷：

  **(1) QESC: Quantization with Expert-Selection Calibration（解决 expert-shift 和跨任务过拟合）**
  Baseline 缺陷：(a) GPTQ 量化只最小化重建误差 ||WX - W_qX||_2²，不感知 expert selection 偏差；(b) PMQ/BSP 用静态校准集分配位宽，跨任务泛化差。

  QESC 设计：
  - 逐层校准：量化每层 MHSA 后，用校准数据前向传播获得该层量化输入 x̂_l，然后用 TopK-MSE Loss 校准该层 router 权重，使量化后的 router 输出尽可能匹配全精度 router 在 top-K 上的输出，防止 expert-shift 逐层累积。
  - TopK-MSE Loss：仅对 top-K 最高概率的 expert 计算 MSE Loss（而非所有 N 个 expert）。图 4 证明 95.9% 的 shifted expert 仍在 top-16 概率内（64 expert 中），而对所有 expert 计算 MSE 会被低概率 expert 的噪声主导。TopK-MSE 让优化聚焦于"更可能被选中的 expert"。
  - 效果：在 3.03-bit 下，Mixtral-8x7B 准确率损失 <0.5%，Deepseek-moe-16b-base 准确率损失 <0.2%，远超 GPTQ/BSP/PMQ。

  **(2) PESF: Pruning based on Expert-Selection Frequency（解决加速效果有限）**
  Baseline 缺陷：EES/ODP 逐 token 剪枝低权重 expert，仅减少部分 expert 输入大小（并非完全跳过 expert 计算），加速比仅 1.05-1.08x。

  PESF 设计：
  - 序列级动态剪枝：收集当前序列所有 token 的 expert 选择统计，若某专家被选中次数 c_i < (l*K/N) * α，则直接跳过该 expert 的全部计算（而非仅减少输入）。
  - 基于 Section 3.3 的核心洞察：同一任务类型内 expert 选择频率高度相似（cosine similarity >0.8），跨任务类型显著不同。因此动态统计能准确反映当前任务的 expert 重要性。
  - 保守策略（α=0.3）：准确率几乎无损，加速 1.08-1.14x，显著优于 EES/ODP。
  - 激进策略（α=0.7）：加速 1.30-1.47x，准确率下降约 1.5%。
  - 限制：仅适用于 prefill 阶段（需要 l 个 token 的统计信息），不适用于逐 token 的 generate 阶段。

  论文方法全栈执行例子（EAC-MoE = QESC 3.03-bit + PESF α=0.3，以 Mixtral-8x7B 推理 512 token 序列为例，batch=4）：
  - **算法层**：
    Layer 0: x_0 → 量化 MHSA (4-bit, 量化权重 W_q^{attn}) → x_0' → Router (FP16 权重, 经 QESC 校准) → Top-2 expert 选择 (expert e_a, e_b)
    → 执行 PESF: 统计本层所有 512 个 token 的 expert 选择, c_i 统计 → 若 c_j < (512*2/8)*0.3 = 38.4, 剪枝 expert j
    → 仅对被保留 expert (如 e_a, e_b) 计算 quantized FFN (3-bit 权重) → 加权求和 → x_1
    ...逐层重复...
    Layer 31: 最终输出 token hidden states。
    Router 在每层都经过 QESC 校准维持正确的 expert 选择，PESF 动态跳过约 10-15% 的 expert 计算。
  - **系统框架层**：BitBLAS 加载量化权重并执行混合精度 BLAS。量化后 Mixtral-8x7B 权重从 93.41GB → 18.98GB（4.92x 压缩），可在单张 RTX 3090 (24GB) 上部署。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：BitBLAS 处理 group-wise INT4/INT3 权重的混合精度 GEMM 操作。PESF 剪枝的 expert 对应 kernel launch 被跳过，减少实际计算量。论文未提供自定义 kernel 实现。
  - **硬件架构层**：RTX 3090 24GB GPU。QESC 量化：显存 18.98GB，batch=4 seq=512: total 加速 1.54x。QESC+PESF: 加速 1.68x。准确率：71.68% (vs baseline 72.64%, 损失 <1%)。

- baseline方法是什么？
  Baseline 是 **固定计算预算的 Dense LLM 推理**，即每个 token 在每层都经过相同大小 FFN 处理，无论 token 复杂度如何。具体痛点：(1) **计算浪费**：简单 token（如标点、常见词）与困难 token（如专有名词、代码关键字）消耗相同计算量，导致资源利用率低；(2) **缺乏自适应性**：无法根据输入复杂度动态分配计算，限制了效率-精度 Pareto frontier；(3) **MoE router 训练次优**：传统 MoE router 使用 per-layer load balancing loss，强制每层内均匀分配 token 给各 expert，限制了跨层的灵活计算分配，导致路由模式偏离理论最优。

  Baseline 全栈执行例子（以 Dense 12-layer Llama-style 1.4B 模型推理一个 token 为例）：
  - **算法层**：标准 Dense Transformer，每层 Attention + FFN（inner_dim=10240, SwiGLU），每 token 固定经过 12 层相同规模 FFN。Router 不存在，无任何动态路由。
  - **系统框架层**：标准 PyTorch/HuggingFace Transformers 推理 pipeline。无 adaptive batching 或 dynamic compute allocation 机制。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM 执行 FFN 矩阵乘法。每层 FFN 对每 token 执行相同的 M×K×N 矩阵乘，无 sparsity 或 conditional execution kernel。每个 token 触发完整的 12 层 FFN forward。
  - **硬件架构层**：论文未明确说明（推断为 NVIDIA GPU，如 A100/H100），标准 GPU 执行 dense matmul。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Duo-LLM 框架**，通过在每层 FFN 中加入 big + small 两个模块，并研究 oracle 最优路由与 learned router 的差距，系统性地研究自适应计算。三个关键设计对应 Baseline 缺陷：

  **(1) Duo FFN 模块（解决计算浪费）**：每层 FFN 包含 big（inner_dim=10240）和 small（inner_dim=640, 16x smaller）两个模块。简单 token 可被路由到 small 模块以节省计算，困难 token 路由到 big 模块以获得更高精度。训练时以 random routing（p=0.5）确保两个模块可互换。
  
  **(2) Oracle 最优路由（揭示理论上界）**：穷举所有 2^L（或 3^L 含 skip）条路由路径，在给定计算预算下选择最小化 perplexity 的路由。发现核心洞察：
  - 仅激活 1 个 big layer 的 oracle 路由 perplexity **低于**所有 12 层都用 big module；
  - 最优 big layer 数量为 6/12（而非 12/12），因为 12C6 候选路径最多，增大了选到优质路径的概率；
  - 预算有限时（4 big layers），oracle 优先将 big 分配给**后层**；预算充足时（8 big layers），优先分配给**前层**；
  - 后层存在"容量阈值"——满足阈值后才值得给前层增加计算。

  **(3) Budget loss 替代 per-layer load balancing（解决 Router 次优）**：不同于传统 MoE 的 per-layer load balancing loss（强制每层内 expert 使用均匀），Duo-LLM 使用全局 budget loss `L_budget = (mean(P_big) - target_budget)^2`，允许 router 跨层灵活分配计算。暴露了 trained router 与 oracle 的巨大差距——router 的 perplexity 更接近 fixed pattern 而非 oracle，证明了现有 MoE router 训练的次优性。

  论文方法全栈执行例子（以 Duo-LLM 推理一个 token，预算=4 big/12 layers 为例）：
  - **算法层**：Token 输入第 1 层 → shared Attention → Router W_{r,1} 计算 P_big/P_small → 若 P_small > P_big 则走 SmallFFN (inner_dim=640) → 残差连接。逐层推进，直至 12 层中恰好使用了 4 个 big FFN。Oracle 模式下，路由决策由 exhaustive search 预先确定（前层多用 small、后层多用 big）。Learned router 模式下，路由由 softmax(W_r * x) 采样决定。最终 output logits 用于计算 token loss。
  - **系统框架层**：论文未实现端到端 serving 框架。论文提到 Megablocks 的 block-sparse matmul 可在单 GPU 上高效执行 Duo-LLM，但实际 efficient implementation "beyond the scope of this work"。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。理论上，每层根据路由决策执行不同尺寸的 GEMM（big: 2560×10240 或 small: 2560×640），可利用条件执行或 block-sparse matmul kernel 减少计算。
  - **硬件架构层**：论文未明确说明。
