## xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

- baseline方法是什么？
  Baseline 主要有两类：(1) **Transformer 架构**（Llama-2-7B、Llama-3.1-8B）：使用 multi-head self-attention（或 GQA），推理时每次自回归解码计算 QK^T 矩阵，复杂度 O(T²)，需要维护随序列长度线性增长的 KV Cache。长序列推理时，KV Cache 内存占用和 attention 计算开销急剧膨胀。(2) **Mamba/SSM 架构**（Falcon-Mamba-7B、Codestral-Mamba-7B）：使用选择性状态空间模型（S6/Mamba-1 或 Mamba-2），具有线性复杂度 O(T) 和常量记忆，但采用 *pre-up projection block*（mLSTM/SSM 在高于 embedding 维度的扩展空间中运行，没有独立 FFN MLP 层）以及额外组件（如 channel-wise convolution），这些设计导致 GPU 利用率不高、线性层 FLOPs 占比低。

  全栈执行例子（Llama-3.1-8B 在单 H100 GPU 上推理一个 128K token 的请求）：
  - 算法层：输入 128K tokens → Embedding → 32 层 Decoder，每层使用 GQA self-attention（8 KV heads，32 query heads）→ RoPE 编码 → QK^T 计算 128K×128K 矩阵，复杂度 O(128K²) → Softmax + V 加权求和 → SwiGLU FFN → 输出 logits → 自回归逐 token 生成。每生成一个 token 需在完整 KV Cache 上做 attention，KV Cache 随生成长度线性增长。
  - 系统框架层：标准的 HuggingFace transformers 推理管线，torch.compile + CUDA Graphs 优化。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention (Dao, 2024) kernel 将 attention 分块计算以减少 HBM 访问；torch.compile 进行 JIT 融合。
  - 硬件架构层：NVIDIA H100 GPU（Tensor Cores + HBM3）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：xLSTM 7B 是一种基于 mLSTM（Matrix Memory LSTM）cell 的纯递归 LLM 架构。通过四项关键设计解决 baseline 缺陷：
  1. **Post-up projection block 替代 Pre-up projection block**：mLSTM cell 直接在 embedding 维度（d=4096）运行，不再先投影到更高维空间再投影回来，并在每层 mLSTM 后添加 SwiGLU MLP（projection factor 2.66）。这解决了 Mamba 类模型"线性层 FLOPs 占比低"的问题（大多数 FLOPs 来自高度优化的矩阵乘法），同时降低了 mLSTM 操作本身的 GPU 内存消耗。
  2. **丢弃 channel-wise convolution 和 learnable skip connections**：减少小 kernel 调用，所有投影改用 dense linear layers，确保 Tensor Cores 有效利用。
  3. **Gate soft-capping（tanh-based）和负输入门 bias 初始化（-10）**：解决了大规模训练（7B 参数）时梯度 norm 尖峰和训练不稳定的问题，使 mLSTM 的标量指数门控在大规模训练中稳定收敛。
  4. **Fused Triton kernels for generation**：将 mLSTM recurrent 模式下的多个独立 GPU kernel（outer product、dot product、pointwise ops）融合为单个 kernel，中间结果保持在 GPU SM SRAM 上，减少 HBM 读写，进一步提升推理速度。

  全栈执行例子（xLSTM 7B 在单 H100 GPU 上推理一个 128K token 的请求）：
  - 算法层：输入 128K tokens → Embedding → 32 个 Post-up Projection Block，每层先过 RMSNorm → 8-head mLSTM（每个 head 独立维护矩阵记忆状态 C ∈ R^{256×512}，通过标量指数输入门和遗忘门更新：C_t = f_t·C_{t-1} + i_t·k_t·v_t^T）→ RMSNorm → SwiGLU MLP → 自回归逐 token 生成。每次 recurrent step 仅需 O(d_hv·d_qk) 计算量和恒定的记忆状态（xLSTM 7B 总记忆 134.2 MB），不随序列长度增长。
  - 系统框架层：HuggingFace transformers 模型实现 + torch.compile（JIT 编译优化）+ PyTorch CUDA Graphs（消除 kernel launch overhead）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Triton-based fused generation kernel，将 gate 计算、max state 更新、memory update（outer product）、normalizer update、hidden state retrieval 融合在单个 kernel 中，中间值保持在 SRAM 不写回 HBM。训练时使用 chunkwise-parallel kernel（基于 FlashLinearAttention 技术）。
  - 硬件架构层：NVIDIA H100 GPU（Tensor Cores + HBM3）。

  **对比总结**：xLSTM 7B 的核心优势在于 (a) 线性复杂度 + 常量记忆使长序列推理速度和内存不受序列长度影响（vs Transformer 的二次复杂度和线性增长 KV Cache），(b) Post-up projection block 使 GPU 利用率最大化（vs Mamba 的 pre-up projection block 中 mLSTM 在高维空间运行且缺乏独立 FFN），(c) 标量门控的指数衰减机制提供比 Mamba 状态空间更灵活的长期记忆控制（输入门 i_t 和遗忘门 f_t 可独立学习），(d) gate soft-capping + 负 bias 初始化解决了深层递归网络的大规模训练稳定性问题。

- baseline方法是什么？
  Baseline 主要有两类：(1) **纯 Transformer 架构**（Llama-2/Llama-3/Mistral），使用全注意力机制，每次自回归解码计算 `Attn(Q,K,V) = softmax(QK^T/√d_k)V`，复杂度 O(T²)，无法高效处理超长上下文；(2) **纯 Mamba/SSM 架构**，使用选择性状态空间 S6 进行递归推理，复杂度 O(T) 但缺乏精确记忆召回能力——递归隐藏状态 Z_t ∈ R^{d_e×d_s} 的固定容量 d_s=16 限制了其对任意历史 token 的精确检索。Baseline 核心矛盾：Transformer 能精确召回但效率差（二次复杂度），Mamba 效率高但召回能力弱（递归状态压缩丢失信息）。

  全栈执行例子（Llama-2 1.3B 在 SlimPajama 上推理一个 128K token 的请求）：
  - 算法层：输入 128K tokens → Embedding → 40 层 Decoder，每层 Self-Attention 计算 128K×128K 的 QK^T 矩阵（RoPE 编码位置），复杂度 O(128K²·d) → GQA 将 KV head 分组 → Softmax + V 加权 → FFN SwiGLU → 输出 logits → 自回归逐 token 生成。预训练长度 4K，需额外的 SelfExtend 等技术才能外推至 128K，但仍有 perplexity 爆炸问题。
  - 系统框架层：论文未明确说明（标准 HuggingFace/PyTorch serving）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention 2 kernel，将 attention 分块计算减少 HBM 访问；Mamba parallel scan kernel 并行化递归计算。
  - 硬件架构层：NVIDIA A100/H100 GPU，bfloat16。128K prompt 吞吐受限于 O(T²) attention 计算。

  纯 Mamba baseline 缺陷：在 SQuAD 等信息检索任务上表现差（Table 2: Mamba 1.8B SQuAD=67.66 vs Samba=77.64），Passkey Retrieval 零样本准确率与 SWA 模型相当（Figure 8 step=0），Phonebook 检索能力不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SAMBA 按层交替组合 Mamba（选择性 SSM）+ Sliding Window Attention（窗口 2048）+ SwiGLU MLP，具体排列为 [Mamba → MLP → SWA → MLP] 的 4 层 block 重复 N/4 次。

  解决 Baseline 缺陷的对应设计：
  (1) **Transformer 的 O(T²) 复杂度 → Mamba 层提供线性时间骨干**：Mamba 层通过递归状态 Z_t 压缩历史信息，每 token 解码仅需 O(1) 计算（状态更新 `Z_t = exp(-Δ⊙exp(A))⊙Z_{t-1} + ...`），无需重新计算所有历史 token 的注意力，实现无限长度流式生成。128K prompt 处理吞吐为 Llama-3 的 3.73×，64K 解码吞吐为 3.64×（Figure 2）。
  (2) **Mamba 的召回缺陷 → SWA 层提供窗口内的精确注意力召回**：SWA 窗口 2048 内的 token 通过 FlashAttention 2 直接计算精确的 softmax attention，弥补 Mamba 递归状态无法精确检索任意历史 token 的不足。窗口大小 2048 的选择使得 FlashAttention 2 的训练速度与 Mamba parallel scan 相当。
  (3) **层间专业化分工 → 交替排列实现功能互补**：注意力熵分析（Figure 5）显示 Samba 的注意力熵方差更大——中间层注意力熵低（专注精确检索），顶层和底层注意力熵高（整合全局信息）。同时 Mamba 的选择熵在中间层更高，表明有了 SWA 负责召回后，Mamba 层可以更专注于建模递归结构而非执行精确选择。这种专业化使 Samba 在几乎所有 benchmark 上优于纯 Transformer 和纯 Mamba。
  (4) **长度外推 → RoPE + SWA 的平移不变性**：RoPE 编码的相对位置使 SWA 天然具有序列长度的平移不变性，训练在 4K、窗口 2048 的模型可直接外推到 1M token 且 perplexity 持续改善（Figure 2）。移除 RoPE 后（Samba-NoPE）perplexity 在超训练长度后爆炸（Table 3）。
  (5) **参数效率 → 更少的注意力头**：Samba 的 KV head 仅为 1（3.8B 模型），query head 数量也比同规模 Transformer 少约 2×（Table 6），因为 Mamba 层已捕获低秩递归信息，注意力层只需专注于检索。

  全栈执行例子（Samba 3.8B 处理 128K 文档进行解码生成）：
  - 算法层：输入 128K tokens → Embedding (d_m=2816) → 64 层交替处理。每 4 层 block：(1) Mamba 层：`H = X@W_in [128K, 5632]` → DepthwiseConv(k=4) → Softplus 门控 Δ → S6 并行扫描 `Z_t = decay * Z_{t-1} + Δ_t*B_t⊗U_t` → `Y_t = Z_t@C_t + D⊙U_t` → GLU 门控 `O = (Y⊙SiLU(X@W_g))@W_out`；(2) MLP 层：SwiGLU `(SiLU(X@W_gate)⊙(X@W_up))@W_down`；(3) SWA 层：Q/K/V 投影 → RoPE（base=10000）→ FlashAttention 2 在 2048 窗口内计算精确 attention → `O_swa = FA2(Q,K,V,window=2048)@W_o`；(4) MLP 层 → 重复 16 次 block → 输出 logits。自回归解码时 Mamba 层仅需单步状态更新，SWA 层只需计算最后 2048 token 窗口，解码复杂度 O(1)。
  - 系统框架层：训练基于修改版 TinyLlama 代码库（https://github.com/jzhang38/TinyLlama），使用 Flash Linear Attention 库（https://github.com/sustcsonglin/flash-linear-attention）实现 GLA/RetNet 对比。开源训练代码：https://github.com/microsoft/Samba。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Mamba 层使用硬件感知并行扫描 kernel（Triton 实现，论文致谢提到 Triton 版 Mamba 实现）；SWA 层使用 FlashAttention 2 kernel，两者在 seqlen=2048 时训练速度相当。解码时 Mamba O(1) 状态更新 vs FlashAttention O(window) 计算。
  - 硬件架构层：训练 8×H100/64×H100/8×A100 GPU；吞吐测量单 A100 bfloat16。3.8B 模型训练 3.2T tokens 使用 Phi-3 数据 pipeline。

## SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

- baseline方法是什么？
  Baseline 是标准 LoRA (Low-Rank Adaptation)。LoRA 在预训练模型的每个目标层插入一对低秩矩阵 W_a (d×r) 和 W_b (r×d)，冻结原权重 W_0，通过训练 W_a 和 W_b 来适配下游任务。标准做法是将 LoRA 稠密地插入所有 attention 层的 query、key、value 矩阵以及 FFN/classifier 层。执行流程：输入 token 序列 x ∈ R^{batch×seq×d} → 逐层前向中，每一层计算 `y = xW_0 + xW_aW_b`（式1）→ 低秩投影 `x_new = xW_a` 将 d 维降至 r 维再映射回 d 维 → 梯度仅流向 W_a 和 W_b。Baseline 缺陷：(1) 稠密全层插入导致参数浪费——部分层的低秩适配对任务贡献很小（SoRA 观察）；(2) 各层低秩矩阵独立训练，无跨层信息共享——第 l 层的 `x_new^(l)` 无法利用 l-1 层的适配经验；(3) 长序列处理能力受限于 Transformer 架构本身的注意力瓶颈。

  全栈执行例子（LoRA fine-tuning RoBERTa-base 在 GLUE 上）：
  - 算法层：输入 token → Embedding → 24层 Encoder，每层 Self-Attention 的 Q/K/V 均插入 W_a (768×8)/W_b (8×768)，FFN 层也插入 → 分类头输出 → Cross-Entropy Loss → 仅更新 W_a/W_b 梯度。可训练参数 1.3M（rank=8）。
  - 系统框架层：论文未明确说明（使用标准 HuggingFace Transformers Trainer）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明（PyTorch 标准 CUDA kernel 执行矩阵乘法）。
  - 硬件架构层：NVIDIA RTX 3090 (24GB) / RTX A6000 (48GB)，标准 GPU 执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SSMLoRA 引入 Time Module（含 W_a/W_b/W_c/W_d + 状态向量 h），沿时间轴连接跨层的低秩矩阵，采用稀疏交替间隔插入策略。

  解决 Baseline 三个缺陷的对应设计：
  (1) **参数浪费 → 稀疏交替间隔插入**：只在 attention 的 query 和 value 矩阵插入 Time Module，每个 attention 层只激活 query 或 value 其中之一（交替），key 完全不插。非 attention 层用 standard LoRA。可训练参数降至 LoRA 的 <80%（如 RoBERTa-base: 1.0M vs 1.3M；LLaMA2-7B: 15.8M vs 20.0M）。
  (2) **无跨层信息共享 → SSM 状态方程连接**：Time Module 沿时间轴传递状态 h_t。前一层 Time Module 的输出状态 h_t 传入当前层：`h_t' = h_{t-1}×W_c + x_new×W_d`（式2，含 W_c/W_d 两个 r×r 矩阵），经 Taylor 展开 `h_t = h_t' + h_{t-1}`（式3），min-max 归一化后作为偏置调整低秩输出：`y = (x_new + h_t_norm) × W_b`。这使得第 l 层的低秩空间调整受益于第 l-1 层的状态信息。
  (3) **长序列能力受限 → SSM 架构优势**：SSM 擅长建模长序列依赖，FFT-based 并行训练克服 RNN 瓶颈。NarrativeQA long-text（>1000 tokens）ROUGE-L 相对 LoRA 提升 2.1%；RACE high-difficulty 子集 Acc: 67.37 vs LoRA 65.64。

  全栈执行例子（SSMLoRA fine-tuning RoBERTa-base 在 MRPC 上）：
  - 算法层：输入 token 序列 → Embedding → 24层 Encoder。第 0 层 Self-Attention: Q 矩阵激活 Time Module（W_a^Q/W_b^Q/W_c^Q/W_d^Q, h_0^Q 初始化零）→ 状态 h_1^Q 传递；V 矩阵跳过。第 1 层: Q 跳过，V 矩阵激活 Time Module（独立时间轴 h_0^V）→ 状态 h_1^V 传递。第 2 层: Q 激活（接收 h_1^Q）→ ...交替至第 23 层。FFN 层: standard LoRA（仅 W_a/W_b 无 SSM 状态）。最终 `y = xW_0 + (x_new + h_norm) × W_b`（式10）。1.0M 可训练参数。
  - 系统框架层：论文未明确说明（标准 HuggingFace Transformers, `python src/main.py --dataset MRPC`）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明（PyTorch 标准 CUDA kernel。W_c/W_d 为 r×r 小矩阵乘法，额外开销极小；h 脱离计算图不参与反向传播）。
  - 硬件架构层：NVIDIA RTX 3090 (24GB)。Table 12 显示 LLaMA2-7B 上 SSMLoRA 与 LoRA GPU 内存基本一致（1024 tokens: 25.80GB vs 25.82GB），推理延迟接近（2.740s vs 2.210s @4096 tokens），batch size 增大时 SSMLoRA 内存优势更明显。

- baseline方法是什么？
  Baseline是标准的Transformer架构（如GPT-3、LLaMA等），使用多头自注意力机制（Multi-Head Self-Attention）。其核心计算为 `Attn(Q,K,V) = softmax(QKᵀ/√d_k)V`，其中Q、K、V ∈ R^{T×d}，QKᵀ产生T×T的成对注意力矩阵，复杂度O(T²d)。虽然这种全对全token交互赋予了模型强大的长距离依赖建模能力且训练时可高度并行化，但推理时的自回归解码每次生成一个token都需要重新计算整个序列的注意力，导致计算和内存复杂度随序列长度二次增长。

  Baseline全栈执行例子（Transformer推理时生成一个token，T长度序列）：
  - 算法pipeline：输入token → Embedding → L层Transformer Block（每层: LayerNorm → 多头Q/K/V线性投影 → 对每个head计算QKT ∈ R^{T×T} → softmax → ×V → 拼接多头 → 线性投影 → +残差 → LayerNorm → FFN (两个线性层+激活) → +残差）→ LM Head → logits → 采样得到next token。每生成一个token需O(T²d)计算和O(Td) KV cache存储（保存所有历史token的K,V）。
  - 系统框架：PyTorch + FlashAttention（IO-aware kernel优化内存访问，降低O(T²)的常数因子但保持二次复杂度）。训练时使用DeepSpeed ZeRO分布式。
  - 编译框架：论文未明确说明。
  - kernel调度：标准GPU矩阵乘法kernel（cuBLAS）用于Q/K/V投影和attention计算。FlashAttention kernel通过tiling和recomputation优化attention的IO访问模式。
  - 硬件架构：NVIDIA A100 80 GB GPU。论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **推理时二次复杂度**：自回归解码时，每次生成token需O(T²d)计算和O(Td) KV cache存储。长序列（T>4096）推理时延和内存快速膨胀，不适合边缘设备和长上下文场景。
  2. **RNN虽线性复杂度但不能并行训练**：传统RNN（LSTM/GRU）虽在推理时O(1) per step，但因时间维度的数据依赖（h_t依赖h_{t-1}）无法在训练时并行化，且存在梯度消失问题，难以扩展到十亿参数规模。
  3. **现有x-former方案折衷**：Reformer/Performer等近似注意力仍保留隐藏的二次因子或log因子（见表1），且均未成功扩展到十亿参数级别与Transformer公平比较。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RWKV（Receptance Weighted Key Value）——一种结合RNN推理效率和Transformer训练并行性的新架构。核心创新在于使用**通道级线性注意力**替代点积token交互注意力，并利用**时间衰减机制**实现RNN形式的递归计算。

  **具体设计如何解决Baseline缺陷：**

  **解决缺陷1（推理二次复杂度）**：RWKV的WKV算子使用通道级时间衰减替代QKᵀ全对全交互：
  ```
  wkv_t = Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} ⊙ v_i + e^{u+k_t} ⊙ v_t
          ─────────────────────────────────────────────
          Σ_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}
  ```
  这里w ∈ (R_{≥0})^d是可学习的通道级时间衰减向量（每个通道独立的衰减率），u是当前token bonus。通过将WKV公式递归化为RNN单元：
  ```
  a_t = e^{-w} ⊙ a_{t-1} + e^{k_t} ⊙ v_t    (分子状态)
  b_t = e^{-w} ⊙ b_{t-1} + e^{k_t}           (分母状态)
  wkv_t = a_t / b_t
  ```
  推理时每步仅需更新大小为d的状态向量（a_t, b_t），**复杂度为O(d)空间和O(d)时间**，与序列长度T无关。因此支持无限长上下文推理，且内存恒定（不像Transformer的KV cache随T线性增长）。

  **解决缺陷2（RNN不能并行训练）**：RWKV在训练时使用time-parallel模式——对batch中所有时间步并行计算矩阵乘法W_λ·X（复杂度O(BTd²)，与Transformer相同），而对WKV时间依赖部分沿其他维度（batch, channel）并行化，或使用parallel scan将串行扫描降为O(B log T d)。因此训练时获得类似Transformer的并行加速。

  **解决缺陷3（x-former未规模化验证）**：论文训练了从169M到14B共6个规模的RWKV模型（14B是当时最大的密集RNN），在Pile 330B tokens上训练1 epoch。FLOP匹配的零样本评估显示RWKV与同计算量Transformer（Pythia/OPT/BLOOM）性能相当（Figure 1），且符合与Transformer相同的log-log线性scaling law（Figure 4, r²=0.994）。这是首个将线性注意力架构验证到十亿参数级别的实践。

  论文方法全栈执行例子（RWKV-14B推理时生成一个token）：
  - 算法pipeline：输入token → Small Init Embedding（U(±1e-4)初始化+额外LayerNorm）→ L=40层RWKV Block（每层: Time-Mixing: token shift x_{t},x_{t-1}通过可学习μ参数线性插值 → 线性投影得r_t,k_t,v_t → WKV递归计算(仅更新5d个状态值: a'_t,b'_t,p_t,x_t,y_t) → σ(r_t)⊙wkv_t → W_o输出投影 → Channel-Mixing: token shift → r'_t = W'_r(μ'_r⊙x_t+(1-μ'_r)⊙x_{t-1}) → σ(r'_t)⊙(W'_v·max(k'_t,0)²) → 输出）→ LayerNorm → 线性投影LM Head → logits → 采样next token。推理每token计算O(d²)（矩阵乘法主导，d=5120），状态大小仅5×5120=25600个float值。
  - 系统框架：PyTorch + DeepSpeed优化（ZeRO等策略加速训练）。使用Adam优化器（β=0.9,0.99），无weight decay，bfloat16精度。
  - 编译框架：论文未明确说明。
  - kernel调度：自定义CUDA kernel用于WKV的并行扫描计算，其余矩阵乘法和逐点运算使用标准PyTorch CUDA后端。GPU: NVIDIA A100 80GB。
  - 硬件架构：论文未涉及RTL/模拟器层面。训练由StabilityAI提供GPU集群。

## RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

- baseline方法是什么？
  Baseline是RWKV-7——一种基于Generalized Delta Rule的线性RNN架构。RWKV-7通过time-mixing和channel-mixing block实现O(N)训练和O(1)推理，其核心state evolution为S_t = S_{t-1}M_t + v_t^T·k̃_t，其中M_t=diag(w_t)-κ̂_t^T(a_t⊙κ̂_t)。虽然RWKV-7在短上下文任务上表现competitive，但其纯RNN结构在长上下文理解上存在根本缺陷——recurrent state的fixed capacity（仅靠state matrix S）难以无损保存长距离token间的精确关联信息，导致在passkey retrieval等需要跨越数万token精确检索的任务上性能随context length增长而快速退化（Figure 1a: RWKV-7 2.9B在28K后准确率崩塌）。另一个baseline是现有混合模型（Jamba、Zamba、MiniMax），它们通过交替插入full attention层增强long-range modeling，但保留了O(N²)复杂度的full attention，在超长序列推理时memory bottleneck严重。

  Baseline全栈执行例子（RWKV-7 2.9B推理时生成一个token，64K context）：
  - 算法pipeline：token → embedding → L层RWKV-7 block（每层: Time-Mixing: x→{r,k,v}线性投影 → w=exp(-exp(Linear_w(x)))数据依赖decay, a=Linear_a(x)学习率, κ̂=κ/||κ||₂归一化removal key → 并行scan/WKV算子计算state evolution S_t=S_{t-1}M_t+v_t^T·k̃_t, M_t=diag(w_t)-κ̂_t^T(a_t⊙κ̂_t) → 输出r⊙state_output → + Channel-Mixing: x→{k',v'}投影 → gate=k'⊙SiLU → v'⊙gate → 输出投影）→ LM head → logits → next token。O(1) per-token计算和常量memory（state S ∈ R^{D×N, N≈64}固定大小），但所有历史信息被压缩进fixed-size state S——随着context从4K增长到64K，state容量不足导致key-value pair的信息被后续token覆盖遗忘，passkey retrieval准确率下降。
  - 系统框架：PyTorch + custom WKV CUDA kernel（fused parallel scan）。DeepSpeed Stage 1分布式训练。RWKV-7 checkpoint from official repo。
  - 编译框架：论文未明确说明。
  - kernel调度：WKV fused parallel scan kernel——将Delta Rule的recurrence分解为并行scan操作，保持训练时的高并行度。推理时切换为纯循环模式（recurrent），O(1) per step。
  - 硬件架构：NVIDIA H20/H200 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **长上下文检索能力不足（State Capacity Bottleneck）**：RWKV-7的recurrent state S ∈ R^{D×N}是固定大小的矩阵，所有历史信息被持续压缩进这个有限容量state中。当context达到28K+ tokens时，state容量饱和——新信息的写入导致旧信息被覆盖遗忘，使得模型无法精确检索远距离的特定key-value pair。Figure 1a/b实验直接验证：即使将RWKV-7用128K-length数据继续预训练，长上下文passkey retrieval依然随长度增加而退化（Figure 1b仅modest improvement）。
  2. **混合模型保留O(N²)复杂度**：Jamba/Zamba等混合架构通过插入full attention层来增强long-range capability，但full attention的O(N²)计算和O(N) KV-cache使得它们在超长序列（>128K）推理时memory成为瓶颈——attention层成为整个模型的性能短板。
  3. **Sparse Attention方法在decoding阶段memory不恒定**：Native Sparse Attention (NSA)和MoBA等方法虽然训练效率高，但在自回归解码时KV cache随序列长度增长（MoBA linear space complexity），无法保证constant memory consumption，限制长序列生成的可扩展性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RWKV-X——一种linear-complexity混合架构，通过在RWKV-7 blocks间周期性插入Top-k Chunk Sparse Attention blocks并配合KV Cache Management，在不引入二次复杂度的情况下显著提升长上下文建模能力。

  论文方法全栈执行例子（RWKV-X-3.6B推理时生成一个token，1M context）：
  - 算法pipeline：token → embedding → L层RWKV-X block（75% of layers: RWKV-7 Time-Mixing+Channel-Mixing block, same as baseline; 25% of layers: Top-k Chunk Sparse Attention block with KV Cache Management）→ LM head → logits → next token。
    
    **Sparse Attention Block详细操作（每4层插入1次）**:
    Step 1: input x → Q/K/V linear projections
    Step 2: divide K,V into n chunks of size B; mean-pool K in each chunk → K_mean ∈ R^{n×d}
    Step 3: compute relevance score s_i = q · K_mean[i] for all n chunks
    Step 4: select top-k chunk indices I = TopK({s_i}, k)
    Step 5: compute sparse softmax attention only over selected chunks: Attn(q,K_I,V_I) = softmax(qK_I^T/√d_k)V_I
    Step 6: residual connection h_l = h_{l-1} + Linear_O(attn_output)
    
    **KV Cache Management in decoding**:
    Step 1: maintain compressed past KV cache of fixed size m=64K
    Step 2: split cache into past (K_past,V_past) and observation window (K_obs,V_obs)
    Step 3: compute importance score C = Σ_i softmax(Q_obs K_past^T/√d_k)[i,:] over past entries
    Step 4: select top-m entries by C, evict rest
    Step 5: compressed cache = selected_entries || observation_window → constant total size
    
    Each RWKV-7 block still operates in O(1) per-token recurrent mode (same as baseline). Each Sparse Attention block computes attention over fixed-size cache (m+L_obs entries). Therefore per-token decoding = O(1) overall, memory = O(1) constant even at 1M context.

  - 系统框架：PyTorch + DeepSpeed Stage 1。Training data pipeline: MiniPile (1.5B tokens, ctx=1024, alignment) → ProLong-64K (1B tokens, ctx=64K, continual pretraining with LongCE loss)。Flash-Attention v3 for full-attention baseline comparison。
  - 编译框架：论文未明确说明。
  - kernel调度：Sparse Attention使用chunk-based sparse computation（top-k chunk selection + local attention），RWKV-7 block使用WKV fused kernel（same as baseline RWKV-7）。当前sparse attention decoding实现比vanilla RWKV慢，论文指出需进一步工程优化（Limitations节）。
  - 硬件架构：NVIDIA H20/H200 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（State Capacity Bottleneck）→ **Top-k Chunk Sparse Attention突破fixed state容量限制**：RWKV-7的recurrent state只能隐式存储信息，容量有限。Sparse Attention block提供显式的token-level access——query可以直接attend到任何被选中的历史chunk中的具体token，无需通过state压缩。这种"直接检索"机制使得模型能在64K context内精确找回任意位置的key-value信息。Figure 1c验证：RWKV-X 64K continual pretraining → near-perfect passkey retrieval accuracy（接近100%），而RWKV-7在28K后崩塌。Table 2进一步验证：RWKV-X-3.6B在S-NIAH-2 8K上99.8 vs RWKV-7-2.9B 88.0（+11.8点），S-NIAH-3 8K上95.6 vs 79.0（+16.6点）。
  
  - 缺陷2（混合模型O(N²)缺陷）→ **Sparse Attention替代Full Attention实现O(N)全局建模**：与传统混合模型（Jamba/Zamba）插入full attention不同，RWKV-X插入的是Top-k Chunk Sparse Attention——仅attend top-k个chunk，计算量O(kBN)≈O(N)而非O(N²)。同时KV Cache Management将cache压缩至固定大小（m entries），使decoding阶段memory和compute均为O(1) constant。Table 1复杂度对比：Full Attention training O(N²)/decoding O(N)/memory O(N)；RWKV-X training O(kBN+N)≈O(N)/decoding O(1)/memory O(1)。Figure 3 prefill latency: RWKV-X 128K时比Flash-Attention v3快1.37×，且差距随context增长扩大。Figure 4: RWKV-X-3.6B decoding latency flat up to 1M tokens（constant-time proof），而Full Attention会linear growth。
  
  - 缺陷3（Sparse Attention decoding memory不恒定）→ **KV Cache Management（SnapKV-inspired）使memory恒定为O(1)**：MoBA等sparse attention方法虽训练效率高，但decoding阶段无cache管理——history KV entries随序列增长，memory linear increase。RWKV-X的解决方案：(a) past cache分为earlier+observation window两部分；(b) 基于softmax attention累积分数C评价earlier entries重要性；(c) 仅保留top-m最相关entry；(d) 拒绝-拼接产生固定大小compressed cache。这样无论生成多长的序列（up to 1M tokens），Sparse Attention block看到的永远是constant-size cache，实现真正的O(1) decoding memory。Table 8验证sparse attention在decoding latency上优于full attention（256K: 121.99ms vs 170.79ms），memory usage也保持更高效。

  设计选择的互补效应：
  - **RWKV-7 blocks（75%）提供高效local+medium-range modeling**：RNN结构天然适合捕获短程语法和局部语义，且O(1)计算。消融实验（Figure 5）显示100% sparse attention/0% RWKV反而validation loss更高——证明RWKV的recurrent归纳偏置在短程建模上优于纯稀疏attention，两者互补。
  - **Sparse Attention blocks（25%）提供精确long-range retrieval**：周期性插入（而非每层都有attention）确保模型既能长程检索（attention block功能），又不过度增加计算量。Figure 5中~25% attention ratio最优，验证了均衡设计。
  - **LongCE Loss增强长上下文token的注意力**：LongCE为关键token分配更高训练权重（weight>1），使模型在long-context pretraining阶段自动学会关注长程依赖的token。Table 4消融：S-NIAH-2 8K上w/ LongCE 99.8 vs w/o 67.0，S-NIAH-3 8K 95.6 vs 62.6——LongCE在深层推理长序列任务上效果critical。
  - **Block Expansion方法降低训练成本**：从RWKV-7 checkpoint出发（而非从头训练），零初始化新Sparse Attention block参数，alignment阶段仅训练新参数（freeze RWKV-7 blocks），long-context阶段再全参数微调——总训练token量仅1B-20B，远少于从头预训练的trillion-token级别。
  - **No Positional Encoding design**：消融显示No Pos (3.08) < Abs Pos (3.10) ≈ ROPE (3.11)，验证RWKV的RNN recurrence已提供足够隐式位置信息，显式位置编码反而可能干扰recurrent state dynamics。

  Efficiency gains总结：
  - Training: O(kBN+N) ≈ O(N) linear complexity（vs Transformer O(N²)）
  - Decoding: O(1) per-token with constant memory（fixed KV cache → stable latency up to 1M tokens）
  - Prefill: near-linear scaling, 128K时1.37× faster than Flash-Attention v3
  - Memory: constant usage regardless of context length（有别于Full Attention的O(N)增长）

- baseline方法是什么？
  Baseline是Linear Time Invariant (LTI) 结构化状态空间模型（S4, DSS, S4D, S5, H3, Hyena, RetNet, RWKV），以及标准Transformer（GPT-3 architecture, MHA + MLP blocks）。LTI SSMs的核心特征：SSM参数(Δ, A, B, C)在所有时间步保持恒定（time-invariant），因此模型等价于一个线性recurrence和一个全局convolution——可通过FFT高效计算。这些模型在continuous signal modalities（音频、视觉）上表现出色，但在discrete information-dense modalities（文本、DNA）上落后于Transformer。Transformer通过softmax attention实现dense information routing，但代价是O(n²)计算复杂度和O(n) KV cache。

  Baseline全栈执行例子（Transformer++推理时生成一个token）：
  - 算法pipeline：token → embedding lookup (1×D) → L层Transformer block（每层: RMSNorm → MHA: W_Q/W_K/W_V投影 → RoPE应用到Q/K → causal softmax(QK^T/√d) → weighted sum V → W_O → residual → RMSNorm → SwiGLU MLP → residual）→ LM head → logits → next token。每生成一个token需O(N) attention计算（N=context length），每层维护K/V cache ∈ R^{N×d_head}，总KV-Cache随序列长度线性增长。
  - 系统框架：PyTorch + standard LM training scripts。FlashAttention-2优化attention计算（fused kernel, IO-aware）。
  - 编译框架：论文未明确说明。
  - kernel调度：FlashAttention-2（fused attention kernel, SRAM-resident softmax + reduction），convolution用PyTorch FFT。
  - 硬件架构：NVIDIA A100 GPU，论文未涉及RTL/模拟器层面。

  Baseline (LTI SSMs) 缺陷：
  1. **无法进行content-based reasoning（选择性复制/归纳头任务失败）**：LTI SSMs的时不变参数意味着模型对所有输入token采用相同的recurrent transition——无法根据内容决定"记住什么、忽略什么"。在Selective Copying任务中，LTI模型无法区分需要记忆的colored token和需要忽略的white noise token（因为模型只跟踪time而非content）；在Induction Heads任务中，LTI模型无法根据context决定何时检索和输出正确答案。
  2. **离散模态性能不足**：LTI SSMs在continuous data（音频、视频）上表现好（因连续系统归纳偏置），但在discrete data（文本、DNA）上显著落后于Transformer——因为后者通过content-aware attention选择性地聚合信息。
  3. **不能有效利用长上下文**：LTI模型的global convolution视角意味着所有历史信息被等权聚合（或者卷积核以固定模式衰减），无法根据内容动态丢弃无关信息。实验表明HyenaDNA在更长context下perplexity反而变差。
  4. **Sequences blending problem**：在多序列拼接场景（如packing documents），LTI模型会在序列边界间"渗出"信息（因为recurrent state无法选择性reset），而Transformer可通过attention mask隔离。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Mamba——基于选择性SSM（S6）的linear-time序列建模架构，通过三个核心创新解决LTI SSM的缺陷：(a) Selection Mechanism（参数化Δ, B, C为输入的函数）使模型具备content-dependent选择性；(b) Hardware-Aware Algorithm（fused parallel scan + recomputation）在GPU上高效计算time-varying SSM；(c) Simplified Architecture（合并H3+MLP block为单一Mamba block）。

  论文方法全栈执行例子（Mamba推理时生成一个token，1.4B参数）：
  - 算法pipeline：token → embedding → L层Mamba block（每层: LayerNorm/RMSNorm → Linear_in投影expand 2× → 分叉: [Branch 1: causal Conv1d(kernel=4) → SiLU → SSM selective scan: Δ_t=softplus(Linear_1(x)+bias), B_t=Linear_N(x), C_t=Linear_N(x), A∈R^{ED×N} learned diagonal → discretize: A_bar_t=exp(Δ_t⊙A), B_bar_t=Δ_t⊗B_t → h_t=A_bar_t⊙h_{t-1}+B_bar_t⊗x_act (O(1) per token, fixed-size state h∈R^{ED×N}) → y_t=C_t^T h_t] + [Branch 2: SiLU gate z] → y_t⊙SiLU(z) → Linear_out → residual）→ LM head → logits → next token。O(1) per-token计算和常量memory，无KV cache增长。
  - 系统框架：PyTorch + custom CUDA kernels（fused selective scan, from mamba-ssm library）。训练与推理均使用同一套参数，推理时切换至recurrent模式（数学等价）。
  - 编译框架：论文未明确说明。
  - kernel调度：Fused selective scan kernel：Δ, A, B, C从HBM→SRAM → discretize+parallel scan in SRAM → 仅最终y写回HBM。Recomputation避免存储intermediates (h)。IO减少O(N)≈16×，实际速度up to 40× faster than standard scan, faster than FlashAttention-2 beyond seqlen 2K。
  - 硬件架构：NVIDIA A100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（无法content-based reasoning）→ Selection Mechanism将Δ, B, C参数化为输入x的函数：Δ_t=softplus(Linear_1(x_t)+bias)控制"关注当前输入vs保持历史状态"的平衡（large Δ→reset state/focus on x_t, small Δ→persist state/ignore x_t）；B_t=Linear_N(x_t)控制输入x_t是否进入hidden state h_t（content-based input filtering）；C_t=Linear_N(x_t)控制state h_t的哪些部分输出到y_t（context-based output modulation）。Δ的selection connection to RNN gating: Theorem 1证明当N=1, A=-1, B=1时, S6退化为g_t=σ(Linear(x_t)), h_t=(1-g_t)h_{t-1}+g_t x_t——即经典gated RNN，确认selection是RNN gating的泛化。实验验证：Selective Copying任务上S6准确率>97% vs S4 18.3%（Table 1），Induction Heads上Mamba完美泛化到1M长度（4000×训练长度），而所有LTI模型在>2×训练长度后崩溃（Table 2）。
  - 缺陷2（离散模态性能差）→ Selection enable content-aware information routing，使Mamba在discrete modalities上首次匹配或超越Transformer。LM Scaling Laws(Figure 4): Mamba是首个匹配Transformer++性能的attention-free模型；Zero-shot(Table 3): Mamba-2.8B avg 63.3 > Pythia-6.9B 61.7, Mamba-130M avg 44.7 > Pythia-160M 40.6（仅1.3×参数高出4.1点）。DNA pretraining(Figure 5 Left): Mamba用3-4×更少参数匹配HyenaDNA和Transformer++性能。
  - 缺陷3（不能利用长上下文）→ Selection允许模型在任何时间步reset state（Δ_t→∞使h_{t+1}≈B_bar_{t+1}⊗x_{t+1}, 丢弃所有历史），从而选择性忽略无关context。DNA context length scaling(Figure 5 Right): Mamba perplexity单调改善至1M长度，而HyenaDNA随长度增长变差。Speech generation: Mamba在更长序列上持续优于SaShiMi（Figure 7）。Filtering context的解释：global convolutions聚合所有信息（包括噪声），selective model可以"reset and restart"。
  - 缺陷4（sequences blending）→ Selective SSM可通过Δ_t→∞在序列边界reset state（Boundary Resetting, Section 3.5.2），等价于Transformer的attention mask隔离效果，但通过可学习的输入依赖机制实现而非手动mask。

  设计选择的互补效应：
  - Δ是most important selective parameter（Table 7: Δ alone 9.81 ppl, all three 8.71 ppl），因其连接RNN gating（Theorem 1）且是唯一影响A_bar=exp(ΔA)中decay的参数。
  - B和C的selectivity synergizes with Δ（Table 7: all three 8.71 > Δ alone 9.81），提供finer-grained content-based filtering（B控制信息进入state, C控制信息从state输出）。
  - SSM state dimension N的scaling影响仅当B和C也是selective时显著（Table 10: N=1→16 with constant B/C仅改善0.07 ppl, with selective B/C改善1.17 ppl），验证了selectivity unlock the benefit of larger state。
  - Real-valued SSM (S4D-Real) vs Complex (S4D-Lin): 在LM上real更好(8.71 vs 9.16 ppl, Table 8)，论文假设real更适合discrete modalities, complex更适合continuous modalities(audio)——在audio实验中complex S6确实更好（Appendix E.4, Figure 10）。

  Efficiency gains:
  - Training: linear scaling in seqlen (vs Transformer's quadratic), fused scan IO reduction by O(N)≈16×
  - Inference: 5× higher throughput than Transformers (no KV cache → much higher batch sizes), O(1) per-token
  - Memory: Mamba 4.8GB vs Transformer(w/FlashAttention-2) 4.6GB at 125M/batch=1 (Table 15)

## Linearizing_Large_Language_Models

- baseline方法是什么？
  Baseline是标准softmax Transformer（Llama2-7B、Mistral-7B），使用因果多头自注意力（MHA with softmax(QK^T/√d)）作为token mixing机制。这些模型在高质量、大规模预训练数据上训练了数万亿tokens（Mistral约8T，Llama2约2T），在下游NLU benchmark上表现最强。同时对比各种从零预训练的循环模型（RWKV-5、Mamba、RetNet），这些模型使用线性注意力或状态空间模型实现O(1)推理，但训练成本高且性能落后于同参数量的Transformer。

  Baseline全栈执行例子（Llama2-7B/Mistral-7B推理时生成一个token）：
  - 算法pipeline：token → embedding lookup (1×4096) → L层Transformer block（每层: RMSNorm → MHA: W_Q/W_K/W_V投影 → RoPE应用到Q/K → causal softmax(QK^T/√d) → weighted sum V → W_O output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。每生成一个token需要对所有历史token计算QK内积（O(N) attention计算），且每层维护K/V cache ∈ R^{N×d_head}，总KV-Cache = 2·n_layers·N·D（32层7B模型：每token约512KB cache，10K context需约5GB）。
  - 系统框架：PyTorch + HuggingFace Transformers或vLLM。推理时使用FlashAttention-2优化attention计算，PagedAttention管理KV-Cache内存。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用FlashAttention等标准GPU kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **推理成本随序列长度线性增长**：softmax attention每个生成步骤需要访问完整KV-Cache并计算QK内积，FLOPs和内存访问均为O(N)，长序列推理延迟高。而RNN可用固定大小hidden state实现O(1) per-token推理。
  2. **从零训练循环模型成本极高**：Mamba/RWKV需要从头预训练（1-6T tokens），且受限于可用数据和计算资源，难以匹配强Transformer（如Mistral在8T高质量数据上训练的）的性能。
  3. **T2R等现有转换方法不稳定且无法扩展**：Kasai et al. (2021)的T2R方法通过MLP近似attention，但大规模uptraining时出现数值不稳定性（分母归一化、梯度问题），仅在小模型（~100M）上验证，且需约20%预训练tokens。
  4. **线性注意力本身的归一化问题**：传统线性注意力的分母归一化（除以Σsim(q,k)）在长序列中可能发散或数值不稳定，如TransNormer (Qin et al., 2022a)所指出的。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SUPRA（Scalable UPtraining for Recurrent Attention），将强预训练Transformer通过有限的继续训练（uptraining，仅需~5%预训练tokens）转换为RNN，而非从头训练。核心设计：(a) 用可学习MLP kernel（φ(x)=ReLU(Wx+b)，Q和K共享权重）替换softmax；(b) 用GroupNorm替换传统线性注意力的分母归一化（借鉴RetNet），解决训练稳定性；(c) 引入RoPE相对位置编码增强位置建模能力；(d) 使用固定衰减向量γ（借鉴RetNet）给更近的token更高权重。

  论文方法全栈执行例子（SUPRA推理时生成一个token，7B参数）：
  - 算法pipeline：token → embedding lookup → L层SUPRA block（每层: RMSNorm → Q/K/V投影 W_Q/W_K/W_V → MLP kernel: φ_q=ReLU(RoPE(qW+b)), φ_k=ReLU(RoPE(kW+b)) → 循环状态更新: s_i=diag(γ)·s_{i-1}+φ_k_i·v_i^T [O(1) per token，s∈R^{d_h×d_h}，固定大小] → GroupNorm(φ_q_i^T·s_i) → W_O output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。每生成一个token仅需O(1)计算，内存固定（s矩阵），无KV-Cache增长。
  - 系统框架：PyTorch + OpenLM fork（https://github.com/TRI-ML/linear_open_lm），集成Lightning Attention 2的Triton kernel。训练用FSDP分布式策略在H100集群上运行。
  - 编译框架：论文未明确说明。
  - kernel调度：使用Lightning Attention 2的Triton kernel实现高效线性注意力计算（训练时沿序列维度并行）。推理时切换为循环模式，O(1) per-token。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(N)推理成本）→ 线性注意力的循环形式将KV-Cache替换为固定大小的matrix state s∈R^{d_h×d_h}，每次更新为s_i=diag(γ)·s_{i-1}+φ(k_i)·v_i^T，读取为φ(q_i)^T·s_i，均为O(d_h²)常量操作。推理时无随序列长度增长的内存或计算开销。这使得SUPRA在理论上具备RNN的无限长度推理能力。
  - 缺陷2（从零训练成本高）→ SUPRA不需要从头训练。它从强预训练Transformer（Llama2/Mistral）初始化，仅用20B-100B tokens uptraining（约占5%预训练成本），即可达到与从零训练1.2T tokens的Mamba-7B竞争的性能（Mistral-SUPRA +100B avg 64.0 vs Mamba-7B avg 64.7）。这大大降低了研究线性模型的实验成本。
  - 缺陷3（T2R不稳定）→ 三项关键改进：(a) **归一化替换**：T2R用分母Σsim(q,k)做归一化，该分母在训练中可能发散/变为零导致梯度不稳定。SUPRA用GroupNorm（per-head, h个group）替代，每个head的输出独立做减均值除标准差，数值范围稳定可预测。Table 3消融直接证明了归一化策略的关键性——T2R uptraining 1B模型性能崩溃（HellaSwag 40.6），而SUPRA保持57.0（接近原始模型的62.1）；(b) **位置编码**：引入RoPE作为相对位置编码（φ(k)和φ(q)在MLP kernel后进行旋转），而T2R缺乏显式位置建模；(c) **decay因子**：γ^{i-j}衰减给近端token更高权重，模拟softmax中的位置偏置，同时短上下文性能更好。
  - 缺陷4（线性注意力归一化发散）→ GroupNorm方案有两个优势：一是数值稳定（每个head独立归一化，无累积操作），二是无需像RetNet那样维护额外的归一化状态（z_i向量）。消融实验（Table 3）证明：与T2R的基于除法的归一化相比，GroupNorm使uptraining从性能崩溃中恢复。

  训练时的并行-循环对偶性：
  - 训练（并行模式）：利用线性注意力的可并行性，使用Lightning Attention 2的Triton kernel做沿序列维度的并行计算（类似标准Transformer训练），避免BPTT。这使训练效率与标准Transformer可比。
  - 推理（循环模式）：纯RNN形式，s_i = diag(γ)·s_{i-1} + φ(k_i)·v_i^T，GroupNorm(φ(q_i)^T·s_i)，O(1) per token。
  - 两种模式在数学上等价，切换无需额外微调。

  不足之处（论文明确记录）：
  - **MMLU/In-context learning退化**：Mistral-SUPRA +100B的MMLU 5-shot仅34.2（vs Mistral原模型62.4），这是线性模型的已知弱点（Akyurek et al., 2024），表明线性化后失去了in-context learning能力。
  - **长上下文性能不达理论预期**：Table 2显示虽然SUPRA模型在超出训练长度后性能不崩溃（vs Transformer），但绝对性能仍显著低于经过位置编码扩展的Transformer（Llama2 + YaRN），且"decay因子限制了有效上下文窗口"。

  从SUPRA的训练流程看设计理念：
  ```
  Step 0: 选择最强的可用预训练Transformer（Mistral-7B）
  Step 1: 在每层attention中添加MLP kernel参数 (W, b)
  Step 2: 将attention计算替换为: GroupNorm(Σγ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)
  Step 3: 5%预训练tokens的uptraining（RefinedWeb, 100B tokens）
          Adam optimizer, cosine LR 3e-5→1e-5, 1000步warmup, FSDP
  Step 4: 推理时自动支持循环模式（数学等价，无需额外处理）
  Step 5: 获得约64% avg的7B RNN，训练成本仅为从零训练的1/20
  ```
  核心洞察：**不需要去近似softmax attention，直接替换为线性attention同时通过uptraining让模型适应新的计算范式**。附录A的热力图分析证明SUPRA的线性attention矩阵与原始softmax矩阵差异很大——模型学到的是不同的计算策略而非softmax的近似。

## GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

- baseline方法是什么？
  Baseline是标准Llama Transformer（Multi-Head Attention + SwiGLU FFN + RoPE位置编码）和Finch (RWKV-6)（线性注意力RNN，O(1) per-token推理）。Llama的MHA需要存储per-layer KV-Cache（每token 2·d_model·n_layer个元素），在256K context、32层、4096 hidden dim下需128GB VRAM；Finch虽无KV-Cache但RNN固定大小的hidden state限制了长程记忆能力（在MQAR associative recall任务中性能显著差于attention模型）。

  Baseline全栈执行例子（Llama Transformer推理时生成一个token，24层为例）：
  - 算法pipeline：token → embedding lookup (1×d_model) → L层Transformer（每层: RMSNorm → MHA: W^Q/W^K/W^V投影→QKV各∈R^{d_model} → RoPE应用到Q/K → causal attention score=softmax(QK^T/√d) → weighted sum V → output projection W^O → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。每生成一个token需O(N) attention计算（N=context length），每层存K/V cache各∈R^{N×d_model}，总KV-Cache=2·d_model·n_layer·N。
  - 系统框架：PyTorch + standard LM training scripts。推理时使用FlashAttention-2优化attention计算，但KV-Cache存储量不变。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用FlashAttention等标准GPU kernel）。
  - 硬件架构：NVIDIA RTX 4090 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. KV-Cache存储爆炸：传统Transformer每层需要独立的K/V cache（2·d_model·n_layer elements per token），长序列（100K+ tokens）时VRAM需求极高（如256K context, 32层, 4096 dim → 128GB），超出consumer GPU能力
  2. Pre-fill计算O(N²)：首次处理输入context时需对每个token计算attention（O(N) per token总计O(N²)），处理超长context时pre-fill延迟高
  3. RNN状态容量有限：Finch等线性注意力RNN虽无KV-Cache，但其固定大小hidden state (wkv ∈ R^{H×H}) 限制了有效记忆容量，在AR等需要精确长程检索的任务中性能显著下降（MQAR gap）
  4. GQA压缩有性能代价：Llama3的GQA虽减少了KV-Cache（8·d_head·n_layer vs 2·d_model·n_layer），但引入性能退化
  5. Per-layer cache冗余：每层独立存储K/V，层间信息高度冗余但无共享机制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出GoldFinch——混合RNN-Attention架构，通过将Finch-C2 RNN层的输出压缩为全局共享的极小型key cache（TokenCat机制），供后1/3的GOLD Transformer层共享消费，实现极致KV-Cache压缩（756-2550×缩小）和O(1) pre-fill，同时保持>Llama的下游性能。

  论文方法全栈执行例子（GoldFinch推理时生成一个token，L24 D2048为例）：
  - 算法pipeline：token idx_t → embedding lookup → Finch-C2层(L0-L15): ddlerp token shift → W^K·(1-w_t) key with adaptive decay → WKV linear attention (recurrent: wkv_t=diag(w_t)·wkv_{t-1}+k_t^T·v_t, O(H²) state) → LayerNorm across heads → concat(r_t·wkv_t+u'_t) → output → Finch channel mixer (ReLU² FFN) → residual → 最后一层Finch-C2输出x_t被压缩: c_t=x_t·W^{KD}∈R^{D/16} 存入全局compressed key cache → GOLD层(L16-L23): 从cache取c_t与原始embedding x_t^0拼接→TokenCat: k_t^D=RMSNorm(concat(x_t^0,c_t)·W^{KU})→DDLoRAdapt生成每层k_t和从embedding生成v_t→MHA over所有历史keys/values→output→Finch channel mixer→residual→LM head→next token。每token: Finch-C2部分O(1)，GOLD部分O(N) attention但总VRAM仅需(D/16+2) bytes per token（cache+index）。
  - 系统框架：PyTorch + 修改版Linear Attention Arena代码仓库（https://github.com/recursal/GoldFinch-paper, Apache 2.0）。支持分块增量attention计算以进一步降低VRAM峰值。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用标准CUDA kernel + 部分自定义CUDA实现，代码含8.9% CUDA）。
  - 硬件架构：NVIDIA RTX 4090 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（KV-Cache存储爆炸）→ TokenCat全局共享压缩cache：仅最后1/3层运行attention但共享同一个压缩key cache（c_t∈R^{D/16} per token），而非每层独立存储。结合从embedding生成value（无需value cache），总cache = (D/16 + 2) bytes per token。256K context, 32层, 4096 dim: GoldFinch仅需0.068GB vs Llama的128GB（约1882×缩小）。编码KV-Cache压缩比例为n_layer×(2d_model)/(1+d_model/16) = 756-2550× for common model sizes。
  - 缺陷2（Pre-fill O(N²)）→ Finch-C2 O(1) pre-fill：预填充时仅需运行前2/3的Finch-C2 RNN层（每token O(1)），只需在最后2G-1个token（G=GOLD层数）运行完整模型以准备token shift所需的previous hidden state。这使得处理超长document时pre-fill近乎线性。
  - 缺陷3（RNN容量有限）→ GOLD attention补全长程检索：后1/3层使用完整MHA（非线性attention），可通过压缩key cache访问所有历史token，在MQAR任务中达到完美分数（100% recall），与纯attention模型持平。GoldFinch ppl (48.2) 远优于Finch (81.9) 和Llama (71.7) on lambada。
  - 缺陷4/5（GQA性能代价、per-layer冗余）→ TokenCat的LoRA式压缩+层间共享：W^{KD}将D维压缩至D/16（类似LoRA的低秩分解），再通过concat(x_t^0, c_t)·W^{KU}解压，参数高效。16:1压缩vs 1:1压缩loss差异可忽略（均为2.2762），证明压缩几乎无损。所有GOLD层共享同一key cache和proto-keys (k_t^D)，每层通过DDLoRAdapt (loradapt_k)施加少量参数实现层特异性。
  - 额外创新（Finch-C2改进）→ 移除gate（减少参数，用第二Value补偿性能），k×(1-w)乘积保持kv-state行归一化，LayerNorm across heads替代GroupNorm改善训练稳定性。这些改进使Finch-C2参数更少但性能优于Finch。
  - 额外创新（GPTAlpha改进）→ RWKV channel mixer替代FFN + token shift增强attention层 + 额外LayerNorm，可独立作为改进版Transformer使用。
  - Position encoding → Finch-C2的RNN特性自动编码位置信息（训练context长度内无需显式位置编码），GOLD层可选RoPE用于extrapolation。Long context实验表明RoPE + interpolation可使GoldFinch在65536 context保持低loss，远超训练时的1024 context。Fine-tuning仅更新GOLD层（冻结Finch-C2部分）即可适应更长context，节省约3× FLOPs。

- baseline方法是什么？
  Baseline是标准BPE-tokenized Transformer（GPT-3 Large/XL scale, GPT-2 tokenizer, Llama architecture with RoPE, SwiGLU, RMSNorm）。tokenization作为handcrafted预处理步骤将raw text压缩为固定词表的token序列。此外还有其他byte-level baseline：(a) isotropic模型（MambaByte, LlamaByte）直接对raw bytes建模但无hierarchy；(b) hierarchical static chunking（如MegaByte, Hourglass Transformer）使用固定k-width pooling压缩，不依赖数据内容；(c) hierarchical external chunking（SpaceByte, BLT）使用delimiter或entropy等外部启发式规则决定chunk边界，需要auxiliary boundary predictor。

  Baseline全栈执行例子（BPE-tokenized Transformer推理时生成一个token）：
  - 算法pipeline：raw bytes → GPT-2 BPE tokenizer（离线固定词表编码）→ token embedding lookup → 24层Transformer block（每层: RMSNorm → Multi-Head Attention(QKV投影, RoPE位置编码, causal softmax(QK^T/√d)·V) → residual → RMSNorm → SwiGLU MLP → residual）→ LM head → logits → softmax → sample token → GPT-2 tokenizer detokenize → raw bytes。Tokenization是独立的预处理步骤，不可学习，词表固定。每生成一个token需O(L²) attention计算，KV cache O(L)。
  - 系统框架：PyTorch + standard LM training scripts。使用FlashAttention-2实现高效attention。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用FlashAttention融合kernel等标准实现）。
  - 硬件架构：NVIDIA GPU集群，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. Tokenization是handcrafted预处理：BPE词表通过统计频率算法生成，不能与模型联合优化。词表固定意味着chunk策略不能根据内容/上下文动态调整
  2. Tokenization导致character-level理解弱：固定词表对罕见字符、拼写错误、噪声输入鲁棒性差（如HellaSwag扰动测试中BPE Transformer Robustness Score仅22.2）
  3. Tokenization对不同语言不公平：中文、代码、DNA等缺乏自然分词线索的模态中BPE性能差（如中文XWinograd BPE Transformer仅59.9%）
  4. 现有byte-level方法的chunk策略不是端到端学习的：(a) isotropic模型计算成本高（O(L²)或线性RNN状态压缩损失信息）；(b) static pooling不考虑内容边界，在语义单元中间截断；(c) external delimiter/entropy方法依赖模态特定的启发式规则，不可多级递归
  5. 之前的可学习chunk方法（如DPT with Gumbel-Softmax）训练不稳定，无法扩展到多级hierarchy或大模型

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出H-Net——端到端的Hierarchical Network，通过Dynamic Chunking (DC)学习数据依赖的分割策略，完全替代tokenization。核心设计：(a) 基于cosine similarity的routing module预测chunk边界；(b) EMA-based smoothing module将离散chunk操作转为连续可微计算；(c) ratio loss控制目标压缩比；(d) 多层信号传播技术（Norm Balance, Separation of Two Streams, LR Modulation）保证训练稳定性。H-Net的M可递归嵌套实现多级hierarchy（S-stage H-Net）。

  论文方法全栈执行例子（H-Net 1-stage, byte-level, 推理时生成一个byte）：
  - 算法pipeline：raw byte x_t → Encoder E⁰ (4×Mamba-2层, selective SSM scan: h_t = A_t·h_{t-1} + B_t·x_t → O(1) per token) → Routing Module: 计算cosine similarity边界概率p_t → 决定是否需要main network处理（DC step）→ 若需处理: 当前字节和之前已被压缩的所有字节通过Main Network M (Transformer, QKV投影 → causal self-attention over compressed chunks → SwiGLU MLP) → Dechunking Layer: Smoothing Module (z̄_t = P_t·ẑ_t + (1-P_t)·z̄_{t-1}, EMA插值) → Upsampler: 将压缩表示扩展回原始分辨率 → Decoder D⁰ (4×Mamba-2层) → logits → next byte。每字节可选择性地使用或不使用main network，实现per-token动态计算分配。Encoder/Decoder仅在原始分辨率上操作，Main Network仅在被路由模块选中的chunk上操作，总计算量与BPE Transformer可比但消除tokenization。
  - 系统框架：PyTorch + FlashAttention-2（处理变长序列）+ Mamba-2 kernels。当前实现比isotropic模型慢约2×（动态序列长度带来batch效率损失，类似MoE的工程挑战）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。使用FlashAttention-2和Mamba-2的高效并行scan实现。
  - 硬件架构：NVIDIA GPU集群，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（handcrafted tokenization）→ DC的Routing Module基于相邻encoder输出的cosine similarity（q_t·k_{t-1}/||q_t||·||k_{t-1}||）学习边界决策，p_t和b_t由模型参数通过gradient descent联合优化。可视化证明H-Net (1-stage)自动学习将边界放在whitespace字符处（与SpaceByte的delimiter等价但无需人工规则），H-Net (2-stage)进一步学习到语义层次的chunk（如"the backbone"和"such as"等multi-word phrase），完全通过端到端训练获得
  - 缺陷2（character-level鲁棒性差）→ H-Net (2-stage)在HellaSwag 5种扰动测试中取得39.0/42.8 Robustness Score（Large/XL scale），远超BPE Transformer的22.2和MambaByte的34.5。这是因为H-Net直接操作raw bytes，每个字符都直接参与模型计算，而非通过固定词表映射
  - 缺陷3（跨语言不公平）→ 中文实验中H-Net (2-stage)的BPB仅25B bytes即超越BPE Transformer，XWinograd-zh从59.9提高到66.3（+6.4%绝对提升）。CODE中H-Net (space)和H-Net (2-stage)均远超BPE Transformer。DNA实验中H-Net (1-stage)仅需3.6×更少数据即达到isotropic模型相同perplexity。DC的content-adaptive特性使其在任何模态上都能自动发现适合的chunk策略
  - 缺陷4（之前chunk策略不可端到端学习）→ Smoothing Module通过EMA (z̄_t = P_t·ẑ_t + (1-P_t)·z̄_{t-1})将离散边界决策转化为连续插值：高置信度边界(P_t≈1.0)保持离散行为(z̄_t≈ẑ_t)，低置信度(P_t≈0.5)产生平滑过渡。这使得整个DC pipeline可通过标准backpropagation训练，无需Gumbel-Softmax等stochastic exploration。消融实验证实移除smoothing module导致压缩比剧烈波动和显著性能下降
  - 缺陷5（训练不稳定无法多级扩展）→ (a) Norm Balance在每个网络输出后添加RMSNorm平衡residual stream和深层网络特征；(b) Separation of Two Streams仅在residual path加projection保持main path梯度畅通；(c) LR Modulation按√(batch_size)·1/√(D)为每个stage缩放学习率。这些技术使H-Net稳定训练到1.6B参数、2级hierarchy，且2-stage持续优于1-stage
  - 端到端学习vs外部heuristic → Ratio Loss (L_ratio = N/(N-1)·((N-1)FG+(1-F)(1-G)))同时优化压缩比F和置信度G，使模型学会在保持目标压缩比的同时自适应分配压缩密度（信息量大处保留更多chunk）。与SpaceByte的固定spacelike规则和BLT的entropy阈值不同，DC的边界决策完全由下游LM任务驱动，可随训练过程持续优化
  - 递归hierarchy → H-Net的M可以是另一个H-Net，实现S-stage递归。2-stage H-Net每stage目标N=3（总计~9×压缩），但实际BPIC=7.0（vs 1-stage的4.8），因DC根据内容自适应调整。2-stage学到的chunk策略：(Stage 0) 边界在spacelike+词的起始字符 → (Stage 1) 边界在semantic groups（如multi-word phrases）。这展示了递归DC自然学习语言学层次的能力

## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- baseline方法是什么？
  Baseline是RWKV-4（Receptance Weighted Key Value），一种基于线性注意力的RNN架构，具有O(1) per-token推理和O(N)可并行训练的特性。RWKV-4核心机制：(1) Token Shift（静态learned lerp: lerp_□(a,b) = a + (b-a)⊙μ_□），使模型按channel分配新旧信息比例；(2) WKV attention with channel-wise additive decay: wkv_t = (Σ_{i=1}^{t-1} exp(-(t-1-i)w+k_i)⊙v_i + exp(u+k_t)⊙v_t) / (Σ exp(·))，每channel有独立learned decay rate w；(3) Sigmoid receptance作为归一化门控；(4) vector-valued state s∈R^D (head size=1, 相当于per-channel scalar state)。

  Baseline全栈执行例子（RWKV-4推理时生成一个token）：
  - 算法pipeline：输入token x_t → embedding lookup → L层RWKV block（每层: Pre-LayerNorm → Time Mixing [Token Shift(lerp): r_t,k_t,v_t = lerp_□(x_t,x_{t-1})W_□ → WKV: scalar decay, vector state s∈R^D的分母归一化 → σ(r_t)⊙wkv_t] → residual → Pre-LayerNorm → Channel Mixing [Token Shift → key/value projection → ReLU²(v_t) → σ(r'_t)⊙v'_t] → residual）→ LM head → logits → next token。每token计算O(1)，state size=5DL。
  - 系统框架：PyTorch + HuggingFace Transformers。训练支持time-parallel（沿序列维度并行，因RWKV-4的WKV可写成前缀和形式）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（RWKV-4使用标准PyTorch实现或基础CUDA kernel，无SRAM-resident优化）。
  - 硬件架构：NVIDIA GPU集群，论文未涉及RTL/模拟器层面。

  Baseline (RWKV-4) 缺陷：
  1. **状态表达力受限**：RWKV-4使用vector-valued state s∈R^D（head size=1），每个channel是标量state。这限制了模型记住和区分不同类型信息的能力，因为所有特征维度共享同一标量状态空间
  2. **分母归一化不稳定**：RWKV-4的WKV使用分母归一化（类似attention中的softmax分母），数值上可能在长序列中不稳定，且分母的除法操作增加计算开销
  3. **静态decay缺乏上下文感知**：decay rate w是learned但static的vector，对所有输入token使用相同decay行为，无法根据token内容动态调整信息保留/遗忘策略
  4. **Token Shift是静态的**：RWKV-4的Token Shift使用learned但data-independent的μ_□向量，新旧信息分配比例与输入内容无关
  5. **Sigmoid receptance限制梯度流**：Sigmoid激活在饱和区梯度接近零，可能限制深层网络的训练效率
  6. **MQAR（多查询联想记忆）能力不足**：Arora et al. (2023)实验表明RWKV-4在MQAR任务上存在性能差距，模型维度与序列长度之间存在相关性限制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两步渐进式改进：Eagle (RWKV-5) 和 Finch (RWKV-6)。

  **Eagle (RWKV-5) 的创新**：
  (a) Matrix-valued states: head size从1扩展至64（h=D/64），state变为s∈R^{(D/h)×(D/h)}的矩阵，每个head独立维护K^TV矩阵作为记忆库。这使state大小从5DL暴增至66DL（~13倍），提供更丰富的记忆存储空间
  (b) LayerNorm替代分母归一化：用per-head LayerNorm（等价GroupNorm with h groups）替代attention分母，消除除法操作，数值更稳定
  (c) SiLU gating + 移除Sigmoid receptance: receptance直接作为线性注意力中的query（无激活函数），添加独立的SiLU gate控制输出
  (d) 改进的参数初始化：针对不同参数类型使用差异化初始化策略（如time_decay初始化为-6+5·(i/(D-1))^{0.7+1.3r₀}），确保训练初期良好的数值分布

  **Finch (RWKV-6) 的创新**：
  (e) Data-dependent Token Shift (ddlerp): 将静态lerp替换为ddlerp_□(a,b) = a + (b-a)⊙lora_□(a+(b-a)⊙μ_x)，其中lora(x) = λ + tanh(xA)B。A∈R^{D×32}, B∈R^{32×D}是低秩矩阵，使token shift量成为输入内容的数据依赖函数
  (f) Time-varying decay w_t: decay从静态w=exp(-exp(ω))变为w_t=exp(-exp(d_t))，其中d_t = lora_d(ddlerp_d(x_t, x_{t-1}))，每个channel的decay rate在每个时间步根据当前和前一token的内容动态变化

  论文方法全栈执行例子（Finch推理时生成一个token）：
  - 算法pipeline：输入token x_t → embedding lookup → L层Finch block（每层: Pre-LayerNorm → Time Mixing [ddlerp Token Shift: r_t,k_t,v_t,g_t = ddlerp_□(x_t,x_{t-1})W_□, LoRA A∈R^{D×32}/B∈R^{32×D} → 计算d_t → w_t = exp(-exp(d_t)) → WKV: 矩阵state s∈R^{(D/h)×(D/h)}, s_t = diag(w_t)·s_{t-1} + k_t^T·v_t, wkv_cur = diag(u)·k_t^T·v_t → LayerNorm(r_t·(wkv_cur+s_{t-1})) → SiLU(g_t)⊙output → concat所有head → W_o output projection] → residual → Pre-LayerNorm → Channel Mixing [ddlerp → LoRA-augmented key/value → ReLU² → sigmoid gate] → residual）→ LM head → next token。每token O(1)计算+O(D²/h) state memory。Decay w_t和Token Shift现在都是data-dependent的。
  - 系统框架：PyTorch + HuggingFace Transformers。训练时有custom CUDA kernel将state操作保持在SRAM中，沿非时间维度并行。也有纯PyTorch time-parallel实现（基于GLA的associative scan方法）。
  - 编译框架：论文未明确说明。
  - kernel调度：Custom CUDA kernel for WKV computation：沿非时间维度并行+SRAM-resident state管理，避免反复HBM↔SRAM传输。Finch kernel在16k序列时比Flash Attention v2快4.2×，比Mamba省17%内存（A100 80GB）。
  - 硬件架构：NVIDIA A100/H800 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（状态表达力受限）→ Eagle引入matrix-valued states（每head的K^TV∈R^{64×64}），这等价于为每个head提供64个独立通道的记忆存储，每个通道可独立编码不同类型的信息模式。直觉上，K作为行选择器（input gate），V作为行值，矩阵的每个元素存储特定(输入通道, 值类型)pair的记忆。内部state从5DL膨胀至66DL（~13×），模型记忆容量大幅提升。Table 18消融（170M模型在Pile上训练330B tokens）证实RWKV6-Pile（avg 50.7%）超越RWKV4-Pile（47.7%）和Pythia（47.9%），接近Mamba（50.1%），证明了矩阵state的收益。
  - 缺陷2（分母归一化不稳定）→ Eagle用per-head LayerNorm替代分母除法。LayerNorm对每个head的WKV输出做减均值除标准差的归一化，数值范围稳定可预测，消除长序列中分母可能发散的风险。同时LayerNorm等效于GroupNorm on h groups，不引入跨head依赖。
  - 缺陷3（静态decay缺乏上下文感知）→ Finch引入data-dependent time-varying decay w_t = exp(-exp(d_t))。d_t由LoRA（低秩矩阵A∈R^{D×64}, B∈R^{64×D}）基于ddlerp后的输入生成，使decay rate在每个时间步、每个channel上根据输入内容动态调整。直觉上，重要token可以"标记"自己为需要更长保留时间（减小decay），不重要token可以加速遗忘（增大decay）。这使模型具备选择性记忆能力：在需要精确回忆的历史tokens上保持低decay，在无关tokens上加快遗忘。MQAR实验（Figure 4）显示Finch在MQAR任务上显著超越所有已知的非Transformer架构。
  - 缺陷4（Token Shift是静态的）→ Finch引入ddlerp：首先Eagle token shift（a+(b-a)⊙μ_x）对输入进行静态预调制，然后lora(a+(b-a)⊙μ_x) = λ + tanh((a+(b-a)⊙μ_x)A)B产生数据依赖的调制偏移。A,B是低秩矩阵（rank=32），参数开销小（~2×32×D per token-shift）。这允许模型根据输入内容决定从历史和当前token各吸收多少信息。Table 19 DDLerp消融证实完整DDLerp（loss=2.91）优于仅decay上DDLerp（2.923）和完全无DDLerp（2.926）。
  - 缺陷5（Sigmoid receptance限制梯度流）→ Eagle移除receptance的Sigmoid激活，使其直接作为线性注意力中的query项（类似标准attention的Q），梯度可通畅传播。同时引入独立的SiLU gate g_t来控制输出幅度，SiLU具有非饱和梯度的优势（x>0区域梯度线性）。这改善了训练效率和模型表达能力。
  - 缺陷6（MQAR能力不足）→ Finch在MQAR上的高准确率来自两个机制协同：(1) matrix-valued state提供更丰富的记忆存储；(2) data-dependent decay允许模型对关键token做选择性记忆增强。在MQAR任务中，模型可通过ddlerp识别key token并保留其信息（降低对应channel的w_t），通过matrix state的对应行存储value信息，最终通过receptance作为query精确检索。

  渐进式改进的效果证据：
  - RWKV-4→Eagle: Table 3多语言avg从51.8→54.3 (1.5B) / 53.9→56.5 (3B) / 56.4→58.2 (7B); Table 4英语avg从59.2→62.4 (1.5B) / 64.1→66.0 (3B) / 67.3→71.5 (7B)。Figure 5长上下文loss显著下降。
  - Eagle→Finch: Table 3多语言avg从54.3→55.0 (1.5/1.6B) / 56.5→57.1 (3B); Table 4英语avg从62.4→62.9 (1.5/1.6B) / 66.0→67.5 (3B)。MQAR上Finch达到极高高精度。

## An_Empirical_Study_of_Mamba-based_Language_Models

- baseline方法是什么？
  Baseline是标准GPT3风格的8B参数Transformer模型（32层，hidden dim 4096，32 attention heads，128 KV-channels，4x MLP expansion，SwiGLU activation，LayerNorm，RoPE位置编码，untied embeddings，无bias，无Dropout）。训练使用1.1T/3.5T tokens数据（70% English + 15% non-English + 15% code）、BF16精度、Adam优化器（β1=0.9, β2=0.95, weight decay=0.1）、cosine LR schedule。实现基于NVIDIA Megatron-LM框架，支持tensor/sequence/pipeline parallelism在H100 GPU集群上训练。
  
  Baseline全栈执行例子（推理时生成一个token）：
  - 算法pipeline：输入token → token embedding lookup → 32层Transformer block（每层: RMSNorm → Multi-Head Attention(QKV投影, RoPE位置编码, softmax(QK^T/√d), attention over V) → residual → RMSNorm → SwiGLU MLP → residual）→ LM head projection → logits → softmax → 采样输出token。每生成一个token，attention需要计算与所有历史token的QK内积（O(n²)计算），且KV cache随序列长度线性增长。
  - 系统框架：Megatron-LM的tensor parallelism（每层1次all-reduce）将模型分片到多GPU，data parallelism将batch分布到多节点。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用Megatron-LM内置的cuBLAS/NCCL kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是在8B参数规模下探索三类替代Transformer的架构：纯Mamba（56层，state dim 128，GELU，RMSNorm，无位置编码）、纯Mamba-2（56层，head dim 64，8 groups，expansion factor 2，conv window 4）以及Mamba-2-Hybrid（56层中24 Mamba-2 + 4 GQA Self-Attention + 28 MLP均匀分布，无RoPE）。核心创新在于通过大规模受控实验回答"SSM能否在>3B参数、>1T tokens规模匹敌Transformer"，并发现少量self-attention层（~7%）即可弥补SSM在in-context learning和copying任务上的短板，形成Hybrid设计。
  
  Baseline（Transformer）缺陷：
  1. 自注意力O(n²)计算复杂度和O(n) KV cache内存需求 → 长序列训练推理效率低
  2. 对极长上下文（如Phonebook）的KV cache存储压力大

  论文方法全栈执行例子（Mamba-2-Hybrid推理时生成一个token）：
  - 算法pipeline：输入token → embedding → 56层hybrid block（Mamba-2层: RMSNorm → input projection(expand 2x) → causal conv1d(window=4) → SiLU → selective SSM scan(O(1) per token via recurrent state) → SiLU gating → output projection → residual; Self-Attention层: RMSNorm → GQA(32Q/8KV, 无RoPE) → output projection → residual; MLP层: RMSNorm → GELU(4x expansion) → output projection → residual）→ LM head → logits。Mamba-2层仅需O(1)计算量和常量state memory（128维内部状态）生成每个token，无需KV cache。Self-attention层仅4/56=7.1%，其KV cache仅需存储4层的key-value（vs Transformer的32层），对长序列大幅减少内存。首层为Mamba层，天然学习位置信息，无需显式位置编码，因此模型可在训练序列长度之外泛化（Phonebook上128K模型在>150K tokens仍100%准确）。
  - 系统框架：Megatron-LM中Mamba-2每层仅需1次all-reduce（vs Mamba的2次），与Transformer的all-reduce量持平，MFU达29.9%（接近Transformer的30.7%）。推理时Hybrid模型在长上下文下生成速度预计达Transformer的8x（batch size 32）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。Mamba-2的selective scan使用硬件感知算法实现高效并行（比Mamba scan快8x）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(n²)计算/O(n)内存）→ Mamba-2层的SSM scan提供O(1) per-token生成，消除attention的二次复杂度
  - 缺陷2（长序列KV cache压力）→ 仅4/56层需要KV cache（GQA with 8 KV groups），其余层用常量大小的SSM state
  - 纯SSM不足（in-context learning/copying弱）→ 混合7.1% self-attention层恢复信息路由和上下文复制能力，MMLU 5-shot提升到53.60（超Transformer的50.07）
  - 训练效率不足（原Mamba 3x慢于Mamba-2）→ 选用Mamba-2而非Mamba作为SSM骨干，将大state dim的scan开销降低8x
  - 位置编码导致长上下文泛化受限 → 取消RoPE，依赖首层Mamba学习位置编码，使128K模型可泛化到>150K tokens

## Associative_Recurrent_Memory_Transformer

- baseline方法是什么？
  Baseline是RMT（Recurrent Memory Transformer），一种基于segment-level recurrence的Transformer扩展。RMT使用特殊memory tokens在segment间传递信息：每个segment处理后，memory tokens作为下一segment的额外输入token，实现跨segment的信息流动。Memory tokens的hidden states通过Transformer的self-attention与当前segment所有token交互，然后传递到下一segment。
  
  Baseline（RMT）全栈执行例子（处理一个长序列，BABILong QA任务）：
  - 算法pipeline：输入序列被切分为512-token segments → 对每个segment s: 将memory tokens M_{s-1}（上一segment的输出）拼接到当前segment tokens X_s前 → 通过GPT-2的Transformer layers（12层，137M参数）进行self-attention（仅在当前segment + memory tokens内，即O(seg_len²)而非O(total_len²)）→ 取出更新后的memory tokens M_s → 送入下一segment → 最终segment后通过LM head预测答案。Memory tokens在segment间通过backpropagation through time (BPTT)训练，跨越所有segment和所有层。
  - 系统框架：PyTorch + Hugging Face Transformers。Sequential segment processing，无并行化。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用PyTorch标准kernel）。
  - 硬件架构：论文未明确说明具体GPU型号。致谢中提及SberDevices提供计算资源。
  
  Baseline缺陷：
  1. 记忆容量有限——RMT的memory tokens（通常仅几个token的hidden states）作为信息瓶颈，可存储的跨segment信息量受限于memory token数量×hidden dim。在Associative Retrieval任务中，RMT的key-value存储容量显著低于ARMT。
  2. 训练困难——BPTT需跨所有segment和所有层反向传播，随着segment数增加，梯度传播路径极长，训练不稳定。
  3. Memory token缺乏专门化的记忆机制——RMT的memory tokens通过标准self-attention读写，没有专门的写入/擦除/读取操作原语，信息混合在attention中，缺乏结构化记忆更新能力。
  4. 长度外推受限——RMT虽能处理超过训练长度的序列（如BABILong 11M），但在极长序列上的性能衰减明显（QA1上从99.1% @128k降至76.4% @10M）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ARMT（Associative Recurrent Memory Transformer），在RMT的segment-level recurrence基础上，每层添加一个基于quasi-linear key-value memory（delta-rule）的层间关联记忆模块。核心创新：(1) 用D×D关联矩阵A_s^l替代memory token hidden states传递信息，存储容量从O(mem_tokens×D)提升到O(D²)；(2) 显式的写入(v_i)、擦除(δ-rule v_i-v̄_i)和读取(y_j)操作原语；(3) γ-correction解决delta-rule中的灾难性遗忘；(4) 层独立关联矩阵实现hierarchical memory。

  论文方法全栈执行例子（ARMT处理一个长序列，BABILong QA任务）：
  - 算法pipeline：输入序列切分为512-token segments → 对每个segment s、每层l: [Step 1: Memory Recall] 将当前segment的memory tokens M_s^l和input tokens X_s^l拼接到关联记忆矩阵A_s^l读取关联向量 y_j = A_s^l φ(q_j) / (z_s^l)^T φ(q_j)（仅需O(D²)计算，与历史segments数无关）→ [Step 2: Transformer Processing] [X_s^{l+1}; M_s^{l+1}] = TransformerBlock([X_s^l + y_X; M_s^l + y_M])（local self-attention仅在当前segment内，O(seg_len²)，与总序列长度无关）→ [Step 3: Memory Update] 用新产生的memory tokens M_s^{l+1}以delta-rule更新关联矩阵: A_s^l = A_{s-1}^l + Σ_i β_i(v_i - v̄_i) ⊗ φ(k_i)（外积更新，O(mem_tokens × D²)），同时用γ-correction更新归一化向量 z_s^l = z_{s-1}^l + Σ_i γ_i φ(k_i) → A_s^l, z_s^l传至下一segment同层，M_s^{l+1}传至同segment下一层 → 最终segment后通过LM head预测答案。Attention仅在当前segment内计算（local self-attention），历史信息通过固定大小的关联矩阵A_s^l（每层D×D）而非KV cache存储。
  - 系统框架：PyTorch + Hugging Face Transformers + Accelerate。Sequential segment processing（论文承认"lack of efficient parallel implementation... have to process all segments consecutively"），但在短中等长度序列（<300k tokens）上比Mamba/RWKV慢。开源地址：https://github.com/RodkinIvan/associative-recurrent-memory-transformer
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。关联记忆操作（矩阵-向量乘、外积更新）使用PyTorch标准操作，无自定义CUDA kernel。
  - 硬件架构：论文未明确说明具体GPU型号。

  关键设计选择映射到缺陷：
  - 缺陷1（记忆容量有限）→ 用D×D关联矩阵（per layer）替代memory token hidden states。对GPT-2 137M (D=768)，单层关联矩阵容量为768²≈590k浮点数，而RMT memory tokens（假设4个memory tokens × 768=3072浮点数）小约192倍。实验验证：ARMT在Associative Retrieval Remember任务上存储的key-value对数是RMT的数倍。
  - 缺陷2（BPTT训练困难）→ 层间关联记忆使得梯度可通过A矩阵在segment间短路传播，不完全依赖BPTT跨所有层和segments的完整路径。但论文也承认LM训练仍具挑战性（ARMT倾向于只保持最后一个segment的信息）。
  - 缺陷3（缺乏结构化记忆操作）→ Delta-rule提供了三种显式记忆原语：(a) 写入β_i v_i ⊗ φ(k_i) — 重要性加权外积存储；(b) 擦除—通过v_i - v̄_i计算delta实现旧信息覆盖；(c) 读取y_j = A φ(q_j) / z^T φ(q_j) — 归一化的key-value查找。γ-correction确保擦除操作同时清除归一化向量中的旧key痕迹，防止灾难性遗忘。消融实验显示去除γ-correction会导致大量rewrite操作后记忆崩溃（Fig. 4a）。
  - 缺陷4（长度外推衰减）→ 关联矩阵的固定大小存储+增量更新使得ARMT处理50M tokens与处理16k tokens的计算/存储成本完全相同（每segment O(D²)常量操作）。实验：ARMT在BABILong QA1上从16k训练长度外推到50M tokens仍达79.9%，外推比>3000x。Mamba仅能外推8x（128k/16k），ARMT达60x+（1M/16k on QA3-QA5）。
  - Mamba/RWKV等SSM在复制/记忆任务上薄弱 → ARMT保留完整local self-attention（在segment内），提供对局部上下文的直接访问能力（类似working memory），同时关联记忆提供对遥远历史的结构化访问（类似long-term memory），两者互补。SSM无此双记忆系统。
  - PRMT消融（仅有层间memory token传递，无关联矩阵）→ PRMT不改善RMT性能（Fig. 4b），证明关联矩阵（非层间传递本身）是ARMT性能的关键贡献因素。

- baseline方法是什么？
  Baseline方法是标准的Full Attention Transformer（Qwen2.5-Instruct 3B/7B/14B），以及滑窗attention变体：Sinks + SWA（attention sinks + sliding window attention, 32k window）和Compressive Transformer（CT-Max/Average，使用max/average pooling以4x压缩率压缩窗口外token，压缩记忆大小等于AHN hidden state大小）。所有方法分配相同的lossless memory budget（32k tokens, 128 attention sinks + 32640 sliding window）以便公平比较。

  Baseline全栈执行例子（Qwen2.5-3B-Instruct Full Attention推理时生成一个token，128k序列）：
  - 算法pipeline：输入token x_t → embedding → 36层Transformer block（每层: RMSNorm → QKV投影 → causal self-attention: softmax(Q_t {K_{1:t}}^T / √d) over full 128k KV cache → output projection → residual → RMSNorm → SwiGLU MLP → residual）→ LM head → logits。每token需O(L)=O(128k) QK内积和O(L) attention over V，完整序列总FLOPs为O(L²)。KV cache存储所有历史token的K/V（128k × num_layers × num_kv_heads × head_dim），内存随L线性增长。
  - 系统框架：PyTorch + Flash Attention（减少attention内存占用但仍O(L)增长）、LLaMA-Factory。长序列下GPU内存随L线性膨胀（PG19 57k example中base model峰值GPU内存持续增长）。
  - 编译框架：论文未明确说明。
  - kernel调度：Flash Attention kernel（fused attention），但长序列下kernel计算量仍为O(L²)。
  - 硬件架构：NVIDIA A100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. 全注意力O(L²)计算复杂度 → 超长序列（128k）推理FLOPs极高
  2. KV cache O(L)内存增长 → 超长序列内存瓶颈（128k时3B模型约9.4GB, 14B模型约50GB）
  3. 滑窗baseline丢弃窗口外信息 → 损失长程依赖（LV-Eval avg: 4.59 vs Full Attn 4.41, 仍低于AHN的5.13-5.88）
  4. Compressive Transformer的max/average pooling压缩过于粗糙 → 信息损失大、不支持可学习的记忆更新

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Artificial Hippocampus Networks (AHNs)：受认知科学Multi-Store Model启发，将RNN-like模块作为可学习的"海马体"，将滑窗外的KV cache连续压缩为固定大小的压缩记忆，同时保留窗口内attention的lossless short-term memory。AHN仅在L > W(默认32k)时激活，短序列下模型等同于标准Transformer。

  论文方法全栈执行例子（AHN-GDN + Qwen2.5-3B-Instruct推理时生成一个token，128k序列，W=32k）：
  - 算法pipeline：输入token x_t → embedding → 36层Transformer block（每层: RMSNorm → QKV投影 → [Branch 1: causal sliding window attention - 仅对窗口内32640个token计算softmax(Q_t {K_{t-W+1:t}}^T/√d) → O(W)=O(32k)计算量] + [Branch 2: AHN-GDN - 对离开窗口的token (k_{t-W}, v_{t-W}) 执行gated delta rule更新压缩记忆 h_{t-W}=α(I-β kk^T)h_{t-W-1} + β k^T v → O(1) per-token计算量; 然后用当前query q_t读取压缩记忆 y_AHN=γ q h W_o → O(1)] → 两分支求和 y_t = y_attn + y_AHN → output projection → residual → MLP → residual）→ LM head → logits。每token总计算量O(W)=O(32k)，完整序列总FLOPs O(WL)。Memory cache: O(W × num_layers × num_kv_heads × head_dim + H²) = 常量，不随L增长（128k时仅为full attention的26.0%）。当L≤W时AHN不激活，模型=标准Transformer。
  - 系统框架：PyTorch + Flash Linear Attention（用于AHN的线性注意力高效实现，https://github.com/fla-org/flash-linear-attention）+ LLaMA-Factory。训练仅需32 A100 GPUs ~10小时（训练AHN for 7B模型）、仅1B tokens、740步、仅优化~0.4%参数。Self-distillation中teacher（full attention）和student（window+AHN）共享base LLM参数，仅AHN参数可训练。
  - 编译框架：论文未明确说明。
  - kernel调度：Flash Linear Attention（FLA）库——基于Triton的线性注意力高效实现。AHN的gated delta rule通过FLA实现高效的recurrent状态更新。论文未详细描述kernel设计。
  - 硬件架构：NVIDIA A100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(L²)计算）→ 大窗口(32k)+AHN design: attention仅在固定窗口内做O(W)计算，AHN以O(1)压缩窗口外token，总复杂度从O(L²)→O(WL)。128k序列下mixing FLOPs降至46.7%（vs full attention），model FLOPs降至59.4%。
  - 缺陷2（O(L) KV cache）→ 窗口外KV pair被压缩后可直接丢弃，仅保留窗口内KV cache（常量大小）。128k序列下memory cache降至26.0%（3B: 2.45GB vs 9.44GB, 7B: 3.81GB vs 14.7GB, 14B: 13.01GB vs 50.33GB）。
  - 缺陷3（滑窗丢弃信息）→ AHN的可学习RNN压缩机制（gated delta rule）能动态控制记忆衰减（α gate控制遗忘率）和写入强度（β gate控制新信息写入），实现token-level选择性记忆（梯度可视化证实AHN倾向保留数学符号和数字而忽略代词/sp token），相比SWA的粗暴丢弃显著提升长程性能（LV-Eval: AHN-GDN 5.88 vs SWA 4.59）。
  - 缺陷4（pooling压缩粗糙）→ GatedDeltaNet的α/β/γ三gate机制提供比static pooling精细得多的可学习压缩，且通过self-distillation学习模仿full attention的输出分布（而非粗糙的下游CE loss），训练信号的dense程度远超pooling。
  - 训练效率 → Self-distillation方案：冻结99.6%参数、仅训练~0.4%参数、1B tokens、740步、32 A100 GPUs ~10小时的极简训练管线，远低于从头训练或全参数微调的开销。
  - 短上下文兼容 → "32k大窗口+AHN仅在超窗时激活"设计保证短序列性能与full attention完全相同（AHN不激活=标准Transformer），无需像MiL/LoLCATs等额外优化短上下文性能。
  - 窗口随机化训练 → 训练时随机化sink size和window size（从多个候选中采样），使AHN学会泛化到不同上下文长度，测试时可在1k-96k窗口范围稳定工作。

## Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

- baseline方法是什么？
  Baseline有两个核心对比对象：(a) Mamba2——使用gated更新规则S_t = α_t S_{t-1} + v_t k_t^T，α_t∈(0,1)统一衰减所有key-value关联，提供全局遗忘能力但缺乏精确的key-value级更新；(b) DeltaNet——使用delta update rule S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T，通过Householder变换精确替换特定key-value对，提供精确记忆更新但缺乏全局遗忘机制。Mamba2的优势在于gating可快速清除过期信息（context switch场景），但缺陷是遗忘均匀作用于所有记忆——无法选择性地保留重要信息；DeltaNet的优势在于delta rule可精确修改特定key-value关联（memorization场景），但缺陷是只能每次修改一个key-value对，缺乏快速批量清除过期信息的能力——在需要过滤大量无关信息的真实场景中性能中等。

  Baseline全栈执行例子（Mamba2推理时生成一个token，单层单head）：
  - 算法pipeline：输入token x_t → embedding → 线性投影生成q_t, k_t, v_t, α_t（short conv + SiLU激活）→ S_t = α_t S_{t-1} + v_t k_t^T（O(d²)矩阵更新）→ o_t = S_t q_t → output gate → 输出投影 → 下一层。每次更新将所有key-value对乘以α_t衰减，新v_t k_t^T添加到state中。
  - 系统框架：论文未明确说明（PyTorch + 自定义chunkwise kernel实现SSD算法）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。Mamba2的SSD分解将矩阵乘法分配到tensor core上训练。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  Baseline (Mamba2) 缺陷：
  1. **均匀遗忘无选择性**：Mamba2的gating α_t对所有key-value关联施加相同衰减，无法根据信息重要性差异化保留。在实际文本中，不同信息有不同的保留价值，如essay context中每个句子重要性不同
  2. **记忆碰撞无精确解决**：线性Transformer受限于维度d_k的存储容量，当序列长度超过d_k时发生"memory collision"，Mamba2通过衰减缓解但不能精确覆盖（因为只衰减不替换）
  3. **长序列记忆保持不足**：S-NIAH-1实验显示Mamba2在>2K序列上性能崩溃（8K仅30.4%），因为decay累积效应使早期信息丧失殆尽

  Baseline (DeltaNet) 缺陷：
  1. **缺乏全局遗忘**：DeltaNet只能逐个key-value对修改，无法一次清除大量过期信息。在需要过滤上下文噪声的数据（如S-NIAH-2/3）中，由于固定state大小下的记忆叠加，性能大幅下降（S-NIAH-2 8K仅17.0, S-NIAH-3 8K仅17.0）
  2. **真实世界任务性能中等**：DeltaNet在真实检索和语言建模任务上落后于Mamba2（Table 3: DeltaNet avg 52.14 vs Mamba2 54.89），验证了缺乏遗忘机制对现实非合成任务的限制

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Gated DeltaNet——将gating和delta rule统一为一个"gated delta rule"：S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T。α_t控制全局衰减，β_t控制精确更新，两者独立且互补。同时提出基于WY表示和chunkwise并行的硬件高效训练算法，以及混合SWA/Mamba2层的hybrid架构。

  论文方法全栈执行例子（Gated DeltaNet推理时生成一个token，单层单head）：
  - 算法pipeline：输入token x_t → embedding → q_t/k_t通过线性投影→short conv→SiLU→L2 norm; v_t通过线性投影→short conv→SiLU; α_t/β_t通过sigmoid(线性投影) → S_t = α_t·S_{t-1}·(I - β_t k_t k_t^T) + β_t v_t k_t^T (O(d²)) → o_t = S_t q_t → RMSNorm(o_t) ⊙ SiLU(gate) → 输出投影 → 下一层。S-NIAH实验验证：(a) 在需要纯记忆保持的S-NIAH-1上，α_t可→1让delta rule主导（接近DeltaNet性能）；(b) 在需要过滤噪声的S-NIAH-2/3上，α_t可减小让gating清除无关信息（接近Mamba2性能）；(c) 在需要复杂模式记忆的S-NIAH-3上，delta rule提供优于Mamba2的记忆质量。
  - 系统框架：论文未明确说明（推测基于Flash Linear Attention库: https://github.com/fla-org/flash-linear-attention）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。chunkwise算法将WY表示+decay mask的计算分解为matmul+triangular solve，利用tensor core实现硬件高效训练。吞吐量与DeltaNet几乎相同（图3），仅比Mamba2稍慢2-3K tokens/sec。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - Mamba2缺陷1（均匀遗忘）→ gated delta rule中的α_t和β_t独立参数化：α_t控制全局衰减率（对所有key-value对均匀），β_t控制特定key-value对的更新精度。当模型遇到重要信息需要保留时，可减小α_t的衰减效应同时增大β_t的写入强度；当遇到无关信息时，可增大α_t快速清除。两者协同而非互斥
  - Mamba2缺陷2（记忆碰撞）→ delta rule通过Householder变换(I - β k k^T)实现精确的key-value替换——先计算旧值S_{t-1}k_t，再用β_t(v_t - 旧值)作为增量写入，实际上是用新key-value对**替换**旧key-value对（而非Mamba2的简单**叠加**），从根本上缓解memory collision
  - Mamba2缺陷3（长序列性能崩溃）→ 结合delta rule的记忆保持能力，Gated DeltaNet在S-NIAH-1 8K上达91.8%（vs Mamba2 30.4%），验证了delta rule在长序列记忆保持上远优于纯gating
  - DeltaNet缺陷1（缺乏全局遗忘）→ α_t门控使模型在需要时（α_t→0）可以快速擦除全部记忆，比DeltaNet只能逐个修改的效率高得多。S-NIAH-2 8K上Gated DeltaNet 91.8%（vs DeltaNet 98.8%的drop pattern不同但绝对值仍高——注意表2中DeltaNet在S-NIAH-2 8K仅17.0，说明需要遗忘场景下delta rule确实失败，而Gated DeltaNet保持高准确率）
  - DeltaNet缺陷2（真实任务差）→ Gaited DeltaNet在Table 3的1.3B常识推理avg 55.32超越Mamba2 54.89和DeltaNet 52.14，在Table 4的真实检索avg 30.6超越Mamba2 29.8和DeltaNet 26.2，在LongBench avg 16.6超越Mamba2 13.5和DeltaNet 13.6，验证了gated delta rule在实际任务上的一致优越
  - 在线学习理论视角：从Table 1可见，Mamba2优化||S_t - α_t S_{t-1}||² - 2⟨S_t k_t, v_t⟩（仅有衰减+内积loss），DeltaNet优化更丰富的||S_t - S_{t-1}||² - 2⟨S_t k_t, β_t(v_t - S_{t-1}k_t)⟩（精确回归loss），而Gated DeltaNet优化||S_t - α_t S_{t-1}||² - 2⟨S_t k_t, β_t(v_t - α_t S_{t-1}k_t)⟩（同时具备衰减和精确回归），理论上也优于两者
  - Hybrid架构进一步弥补：线性RNN在局部模式建模和检索任务上有固有局限（state size固定）。GatedDeltaNet-H1通过交替SWA层提供O(1)的窗口内精确attention（弥补局部模式缺陷），GatedDeltaNet-H2通过Mamba2层提供互补的记忆机制。最优Hybrid顺序为Mamba2→GatedDeltaNet→SWA（Table S.2消融验证，avg 48.73 vs 其他顺序47.54-47.92）
  - 训练效率保持：chunkwise算法将gated delta rule的WY表示与decay mask（Γ_{[t]}）结合，只需修改T矩阵的计算（加入Γ_{[t]} ⊙ K_{[t]} K_{[t]}^T），其余计算流程与DeltaNet一致，因此训练吞吐量几乎无额外开销（图3）

## Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

- baseline方法是什么？
  Baseline是标准causal decoder-only循环语言模型（Mamba、Based、GLA、Mamba-2等），均以因果自回归方式处理输入——从左到右逐个token处理，使用固定大小的recurrent state存储历史信息。标准ICL格式为Ŷ = A(C, Q)，其中context C在questions Q之前出现。

  Baseline全栈执行例子（Based causal LM推理时生成一个token, 360M/1.3B参数）：
  - 算法pipeline：输入token → embedding → L层Based block（每层交替: gated short convolution(kernel=3) → sliding window attention(window=128) → causal linear attention(Taylor feature map, feature dim=16, 2nd-order: φ(q)^Tφ(k)=1+q^Tk+(q^Tk)²/2 → y_i=φ(q_i)·Σ_{j=1}^{i}φ(k_j)^Tv_j / φ(q_i)·Σ_{j=1}^{i}φ(k_j)) → residual → SwiGLU MLP → residual）→ LM head → logits → next token。每token: linear attention decode O(1)，sliding window attention O(W=128)，recurrent state s ∈ R^{d×d̃}固定大小。Causal cumsum使每个token仅能看到之前的信息——当context长且问题在后面时，模型必须在看到问题前就决定存储什么。
  - 系统框架：PyTorch + FlashAttention训练代码库（https://github.com/Dao-AILab/flash-attention/tree/main）+ LM-Eval Harness推理。开源模型从HuggingFace获取。
  - 编译框架：论文未明确说明。
  - kernel调度：Based Custom CUDA kernel (ThunderKittens)，warp-register分区存储KV-state实现IO-aware prefill。
  - 硬件架构：NVIDIA A100-80GB (训练) / H100 (推理benchmark)，论文未涉及RTL/模拟器层面。

  Baseline (Causal Decoder-only循环LM) 缺陷：
  1. **数据顺序依赖导致ICL脆弱**：Causal模型从左到右处理，若context（如长文档）出现在问题之前，模型必须在未看到问题时就预测应该存储哪些信息。错误的存储决策导致信息丢失，后续无法回忆。例如[D, Q]顺序下模型需记住文档中所有事实，而[Q, D]顺序下只需记住一个。
  2. **内存容量-回忆能力tradeoff**：O(1)推理内存使循环LM在理论上无法记住所有上下文信息（Arvind et al. 2023 proved Ω(N) lower bound for recall）。虽然增加recurrent state size可以提升回忆能力，但硬件效率下降。
  3. **选择机制的局限性**：现有改进（LSMT gates, decay rates, delta rules等）通过架构偏置优化"存储/丢弃"决策，但未利用数据顺序简化选择难度。
  4. **循环LM在recall-intensive ICL任务上显著落后于Transformer**：2.8B Mamba (300B tokens) 比1.3B Transformer (50B tokens) 平均低5个点。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两种互补方法：(1) JRT-Prompt——重复context使模型看到所有数据顺序；(2) JRT-RNN——非因果Prefix Linear Attention架构+联合训练目标。

  论文方法全栈执行例子：

  **JRT-Prompt** (推理时生成一个token, 与baseline相同架构，仅prompt改变)：
  - 算法pipeline：构造prompt Ŷ = A(C, Q, C, Q)（context重复两次）→ embedding → causal decoding。第二轮出现时模型已condition on完整context（包括前面的Q），此时决定存储什么时能看到全部信息——等价于让模型学到最优存储策略。缺点：context长度翻倍。但sub-quadratic架构使得2N长度仍渐进快于Transformer的N长度。
  - 系统框架：同baseline（使用开源模型权重，无需额外训练）。通过LM-Eval Harness调用。JRT-Prompt需要将原prompt中的context和question各重复一次，对基于HuggingFace的模型interface无额外修改。
  - kernel调度：同baseline Based kernel，但prefill长度由N变为2N（即2× prefill time）。
  - 效果：16个模型×6任务平均+11.0±1.3点提升。N=32768, B=16, H100上11.9×于FA2的prefill吞吐量（因为linear attention的2N仍远快于attention的N）。

  **JRT-RNN** (推理时生成一个token, PLA层)：
  - 算法pipeline：输入token(前M=1024为encoder区域，后N-M为decoder区域) → embedding → L层block（gated convolution/sliding window同Based；PLA层: encoder区域k_e/v_e非因果sum in parallel → decoder区域q_d/k_d/v_d causal cumsum → y_i = φ(q_i)(Σ_{j=1}^{i}k_d[j]^Tv_d[j]+Σ_{j=1}^{M}k_e[j]^Tv_e[j]) / φ(q_i)(Σ_{j=1}^{i}k_d[j]+Σ_{j=1}^{M}k_e[j])）。Pre-fill: 并行计算encoder初始state s_M = Σ_{j=1}^{M}(k_e[j]^Tv_e[j]+k_d[j]^Tv_d[j])。Decoding (i>M): O(1) standard causal linear attention。训练: L = (w1·L_NTP + w2·L_MLM)/(w1+w2)，encoder区域随机mask P比例token计算MLM loss。
  - 系统框架：PyTorch + Based代码库（https://github.com/HazyResearch/based）。训练在FlashAttention代码库上进行。开源权重在HuggingFace。
  - 编译框架：论文未明确说明。
  - kernel调度：扩展Based Custom CUDA kernel (ThunderKittens): fnbased(k_e,v_e)先计算encoder KV-state→寄存器，再fnbased(q_d,k_d,v_d)从该state续算decoder→SRAM→HBM。PLA decode O(1)无额外修改。JRT-RNN CUDA prefill达19.2×于FA2 (N=32768)。
  - 硬件架构：NVIDIA A100-80GB (训练) / H100 (推理benchmark)，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（数据顺序依赖）→ 理论形式化（SD问题+通信复杂度）证明数据顺序决定memory requirement为Ω(min(|A|,|B|))；JRT-Prompt通过重复context使模型看到所有数据顺序；JRT-RNN通过非因果encoder处理prompt使模型可以同时看到全部context信息。
  - 缺陷2（memory-recall tradeoff）→ JRT-Prompt在理论上将memory下界从Ω(max(|A|,|B|))降为Ω(min(|A|,|B|)/p)（p为重复次数）；JRT-RNN的PLA decoder O(1) memory不变但encoder非因果sum让模型充分利用decoder的有限memory（先看到问题和答案再决定存什么）。
  - 缺陷3（选择机制局限）→ 不从修改gate/decay入手，而是通过改变数据呈现方式（JRT-Prompt）或架构causality（JRT-RNN）从根本上降低选择难度。Encoder-decoder分离KV投影让encoder和decoder各自优化不同的信息处理策略。
  - 缺陷4（ICL质量差距）→ JRT-RNN 360M/30B达Transformer++同参数92%的质量（avg 42.9 vs 43.4），1.3B/50B达96%（49.5 vs 51.4）。JRT-Prompt使Based+JRT超越Transformer++ with standard prompting。
  - 训练效率 → JRT-RNN decoder区域标准NTP loss（50%数据量于纯decoder模型），encoder区域MLM loss补偿。PLA decode O(1)与causal LM完全相同。
  - 理论洞察 → BaseConv等纯卷积架构即使有JRT-prompt也无法降低memory下限（Theorem G.6/G.7/G.11），说明JRT方法的效果是架构依赖的——需要linear attention类的关联记忆机制（IP kernel + input-dependent shift）。

## LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

- baseline方法是什么？
  Baseline是vanilla Mamba（Gu & Dao, 2023），一种selective state space model (SSM)，通过time-variant的隐藏状态更新实现线性复杂度的序列建模。Mamba每个block的计算流程: X = σ(Conv1D(Linear₁(I))) → Y = SSM(X) → O = Linear₃(σ(Linear₂(I)) ⊙ Y)。SSM核心递归公式为 H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t, Y_t = C_t^T H_t，其中Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e}是隐藏状态衰减因子（A为负矩阵保证Ā_t<1），B̄_t = Δ_t ⊗ B_t决定当前token的更新量。此外对比DeciMamba（Ben-Kish et al., 2024），一种逐层token pruning方法，在更深层逐步减少序列长度。

  Baseline全栈执行例子（vanilla Mamba-1.4B推理时处理长序列，S=16000 tokens）：
  - 算法pipeline：输入序列I ∈ R^{S×d_m} → 逐层Mamba block处理（每层: Linear₁投影 → Conv1D(因果卷积, kernel=4) → SiLU激活 → SSM递归计算 [对每个token t: 输入X_t → 计算Δ_t=Softplus(X_t), B_t/C_t=Linear₄(X_t) → Ā_t=exp(Δ_t⊙A), B̄_t=Δ_t⊗B_t → H_t=Ā_t⊙H_{t-1}+B̄_t⊙X_t (H_t∈R^{d_s×d_e}, 固定大小隐藏状态) → Y_t=C_t^T H_t] → ⊙ SiLU(Linear₂(I)) → Linear₃输出投影 → residual）→ LM head → next token。每token O(1)计算，隐藏状态H∈R^{d_s×d_e}固定大小。但当S≫L（训练长度=2k）时，全局通道的累积衰减∏_{k=1}^S Ā_k因指数衰减趋向于零（Eq.12: exp((ΣΔ_k)⊙A) with A<0），导致隐藏状态H_t中早期token的信息完全丧失，全局通道的感受野无法扩展到全序列长度（图1b）。
  - 系统框架：PyTorch + HuggingFace Transformers（mamba-ssm库），直接加载官方预训练checkpoint。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用Mamba官方CUDA kernel的selective scan实现）。
  - 硬件架构：NVIDIA A5000/A100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **全局通道感受野无法泛化到更长序列**：通过per-channel attention map可视化（图1），论文发现Mamba隐藏状态通道可分为局部通道（感受野短于训练长度，仅关注临近上下文）和全局通道（感受野覆盖训练长度，捕获全局信息）。但当输入序列长度显著超过训练长度（如16k vs 2k），全局通道的感受野无法自适应扩展（图1b：全局通道(iv)/(v)在2k长度上的红色边框在16k长度下萎缩），导致它们失去了捕获全局信息的能力——这是Mamba长上下文性能差的关键瓶颈。
  2. **指数衰减导致隐藏状态记忆消失**：Mamba的Ā_t∈(0,1)使得每次更新都在衰减历史信息。累积衰减 ∏_{k=1}^S Ā_k = exp((Σ_{k=1}^S Δ_k) ⊙ A) 随S增大而指数级趋近于零（A为负矩阵）。当S≫L时，早期token对H_S的贡献几乎为零，即使在全局通道中也是如此。
  3. **DeciMamba的token pruning无差别对待所有通道**：DeciMamba对所有隐藏状态通道无差别地prune token，没有区分局部和全局通道的不同需求——局部通道本来就不需要处理长上下文（它们专门处理局部信息），强制在它们上面prune不如集中优化全局通道的感受野。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LongMamba——一种training-free技术，分为两个步骤：(a) 通过训练长度上的累积衰减 ∏_{k=1}^L Ā_k > θ 来识别全局通道；(b) 对于全局通道，通过token filtering（跳过Δ_t低于阈值g(S)的token的隐藏状态更新）来扩大感受野，使筛选后的累积衰减与训练长度对齐：∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i。

  论文方法全栈执行例子（LongMamba-enhanced Mamba推理时处理长序列，S=16000 tokens，训练长度L=2000）：
  - 算法pipeline：
    Step 1 (离线标定，仅运行一次): 从Pile采样5条序列(各2000 tokens) → 计算每个通道c在训练长度L上的累积衰减 decay_c = ∏_{k=1}^L Ā_k[c] → 若decay_c > θ则标记为全局通道 → 记录全局通道中各token的Δ_t分布 → Clamp极值到top C% → 数值求解每个S=1000,2000,...下的阈值g_c(S)使得∏_{i=1}^S Ā'_i(g)[c] ≈ decay_c
    Step 2 (推理，对每个token t):
    - 标准Mamba预处理：X_t → Conv1D → SiLU → Δ_t, B_t, C_t → Ā_t, B̄_t
    - 对每个通道c:
      if c是全局通道:
        查表得 g = g_c(round_to_nearest_1000(S))
        if Δ_t[c] < g:
          Ā'_t[c] = 1, B̄'_t[c] = 0  # 跳过该token：H_t[c] = H_{t-1}[c]
        else:
          Ā'_t[c] = Ā_t[c], B̄'_t[c] = B̄_t[c]  # 正常更新
      else:  # 局部通道保持原样
        Ā'_t[c] = Ā_t[c], B̄'_t[c] = B̄_t[c]
    - H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t
    - 标准Mamba输出：Y_t = C_t^T H_t → ⊙ gating → output
  - 系统框架：PyTorch + 修改的Mamba前向传播（在SSM核心循环中插入token filtering逻辑）。代码开源：https://github.com/GATECH-EIC/LongMamba。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。LongMamba在SSM的递归循环中插入了per-channel的条件判断，不影响底层scan kernel。
  - 硬件架构：NVIDIA A5000/A100 GPU，论文未涉及RTL/模拟器层面。延迟开销≤4.5%（表6-7）。

  关键设计选择映射到缺陷：
  - 缺陷1（全局通道感受野无法泛化）→ 通过累积衰减阈值分类识别全局通道，仅对这些通道施加token filtering。直觉上，全局通道的隐藏状态需要存储长期信息，而每个不重要token的贡献都会累积额外的衰减（Ā_t < 1），导致历史信息迅速消失。通过跳过不重要token（Δ_t<g → Ā'_t=1），有效减少了衰减累积次数，使隐藏状态能在更长序列上保持早期信息。图1可视化对比展示了LongMamba处理后全局通道的感受野可扩展到16k tokens。
  - 缺陷2（指数衰减导致记忆消失）→ LongMamba通过token filtering使"有效衰减步数"保持在≈L的规模，而非S。核心对齐公式 ∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i 将OOD长序列输入统计量变换为ID样本统计量。具体地，通过设置Ā'_t=1（即不衰减也不更新）来"跳过"那些不重要token——这些token的比例约等于(S-L)/S，使得筛选后的累积衰减（仅计算被保留token的Ā_t）与训练时的衰减量相似。
  - 缺陷3（DeciMamba无差别pruning）→ LongMamba区分对待全局和局部通道。局部通道专门处理局部上下文，不需要长感受野——因此它们完全不施加token filtering，保持对局部上下文的完整建模能力。对比DeciMamba的逐层pruning对所有通道同等对待，LongMamba的差异化策略带来显著性能优势（LongBench-E上Mamba-1.4B: LongMamba 17.33% vs DeciMamba 13.38%，提升3.95个百分点）。

  标定机制的自适应能力：
  - g(S)查找表per-channel构建，使每个全局通道根据其自身的Δ_t分布获得个性化的过滤阈值
  - Δ_t可解释为token的"重要性度量"（Mamba中Δ_t越大，该token对隐藏状态的更新贡献越大）。因此过滤Δ_t<g的token等价于"只让重要token更新全局通道的隐藏状态"
  - 标定使用Pile训练集数据，建立的是"训练分布下Δ_t的统计量"，推理时用此统计量决定哪些token值得全局通道记住
  - 消融实验（表4）验证了标定序列选择的鲁棒性——10组不同随机种子采样的校准序列在LongBench-E上STD仅为0.42%，说明Δ_t分布在不同序列间高度一致

  模型间的策略差异（体现方法灵活性）：
  - Mamba-1.4B：θ=10⁻³⁰, C=20（非常极端的阈值——只有极少数通道是全局通道，且大量clamping）
  - Mamba2-1.3B：θ=5×10⁻², C=5（较宽松的阈值——较多通道被识别为全局通道，少量clamping）
  - Zamba2-1.2B：θ=10⁻⁵, C=5（中等阈值——因混合Transformer-SSM架构影响通道分布）
  这些差异表明LongMamba自动适应不同模型的内部通道统计特性，无需人工调整。

## Attamba__Attending_To_Multi-Token_States

- baseline方法是什么？
  Baseline是标准Transformer（GPT-style，60M参数，8层8 heads，512 model dimension）。标准attention的全栈执行例子（生成一个token）：
  - 算法pipeline：输入token → embedding → 8层Transformer block（每层: RMSNorm → QKV线性投影 → causal self-attention: softmax(QK^T/√d)·V over所有历史token → residual → MLP → residual）→ LM head → logits。Attention需计算与所有历史token的QK内积，复杂度O(n²)，KV cache大小O(n)。
  - 系统框架：Meta Lingua（PyTorch LLM训练库）。单卡A6000 GPU训练。标准PyTorch autograd + Adam优化器。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用PyTorch标准attention kernel实现，如Flash Attention）。
  - 硬件架构：NVIDIA RTX A6000 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. Attention计算量O(n²)随序列长度平方增长，长序列下FLOPs和内存开销巨大
  2. KV-Cache随序列长度线性增长O(n)，autoregressive推理时内存瓶颈
  3. L² attention map的激活值占用大量显存（iso-activation条件下论文甚至无法找到等价的transformer设计，见公式11中(1-P)/P的负项）
  4. 现有KV-Cache压缩（如低秩分解Palu、ShadowKV）和稀疏attention（BigBird、LinFormer）等方法会牺牲attention表达能力

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Attamba，核心思想是用SSM block替换Transformer attention中的K/V投影矩阵，让SSM将连续P个token压缩为单个表示，然后attention仅在这些压缩表示上进行计算。Query投影保持不变以保证自回归训练的causality。配套设计包括cyclic chunk boundary（逐层偏移消除固定边界偏差）、leading tokens（保留对最近token的完整attention模拟sliding window）、pseudo-chunking（可选的仅替换投影不裁剪attention mask模式）。

  论文方法全栈执行例子（Attamba P=4, L=4推理时生成一个token）：
  - 算法pipeline：输入token → embedding → 8层Attamba block（每层: RMSNorm → Q = X·W_Q 标准Query投影 → 将KV序列分为P=4 token的chunks → SSM_K, SSM_V 在每个chunk上autoregressive扫描: h_t = A_t·h_{t-1} + B_t·x_t, k_t/v_t = C_t·h_t → 仅保存每个chunk的最后L=4个输出（即完整当前chunk的SSM输出）→ 用chunk attention mask: 仅attend已完成chunk的边界+当前chunk内causal → Softmax(Q·K_SSM^T/√d)·V_SSM → residual → MLP → residual）→ LM head → logits。Attention map从L×L缩减为L×(L/P+L)≈L²/4。KV-Cache仅保留chunk边界+L个leading token的K/V：n/4×e(每个chunk)+4×e(当前chunk L tokens) vs baseline n×e。每层不同chunk边界偏移量（cyclic: layer_idx%4），使SSM学习到位置鲁棒的压缩。
  - 系统框架：Meta Lingua + Mamba library（cu_seqlens处理变长chunk，无需padding）。开源：https://github.com/abdelfattah-lab/attamba（BSD-3-Clause）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。SSM扫描使用Mamba库的selective scan并行实现。
  - 硬件架构：NVIDIA RTX A6000 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(n²)计算）→ SSM压缩P个token为1个后，attention仅需对n/P个压缩表示计算attention，FLOPs从O(n²)降至O(n²/P)。同时论文提出Attamba-Linear方案：将序列分为固定数量P个chunk，无论序列多长，attention map大小恒定，实现O(n)复杂度（类似BigBird但保留可变chunk边界能力）。
  - 缺陷2（O(n) KV-Cache）→ 推理时仅缓存每个chunk的最终SSM输出K(p)[-1]和V(p)[-1]（而非全部P个token的KV），KV-Cache从2nE降至2(n/P+L)E（L为leading tokens数）。P=8时约8×压缩，P=128时128×压缩。
  - 缺陷3（L²激活值）→ Attamba的attention map大小为n×(n/P+L)，而非n×n。公式11推导显示(1-P)/P项为负，正是Attamba消除L² activation的结果，使得Transformer在同等激活值预算下无法匹敌Attamba。
  - 缺陷4（压缩损失attention表达能力）→ 与稀疏attention（BigBird）或低秩分解（LinFormer）不同，Attamba的SSM压缩是可学习的、data-dependent的。每个chunk内P个token通过SSM的selective mechanism（A_t, B_t, C_t依赖输入）被自适应压缩，而非简单丢弃或平均。实验表明：(a) Attamba在iso-KV+SWA条件下的困惑度远优于Transformer（图7）；(b) 随机chunk边界与均匀分块效果相当（图14），说明SSM压缩对chunk划分方式鲁棒；(c) cyclic chunking额外提升5%，不同层压缩不同的token组，增强模型对不同上下文模式的覆盖；(d) pseudo-chunking（仅替换投影，不裁剪attention）可略微优于标准Transformer（图16），说明SSM-based K/V投影本身比线性投影有更好的表示能力。
  - SSM state collapse（长序列信息丢失）→ 与纯SSM不同，Attamba的SSM仅需处理固定长度chunk（P个token），不会遇到state collapse问题。注意力在压缩chunk表示上进行，SSM不需要在任意长序列上维护state。
  - 固定chunk边界偏差 → cyclic chunking：第layer层从layer%P偏移开始分chunk，确保不同层处理不同的token分组模式，分布边界效应。实验表明cyclic比uniform/FAttn/FSSM均更优。
  - FFN不受益 → 论文诚实指出FFN层无优化（Query序列长度不变以保持自回归训练），但attention的FLOPs和KV-Cache已在主要开销上获得显著压缩。

  Baseline→Attamba的迁移路径（以P=4为例）：
  1. 保持所有非attention组件不变（embedding, MLP, LM head, RMSNorm）
  2. 将每个attention层中的W_K, W_V线性投影替换为SSM_K, SSM_V block（约4M参数，残差连接保留）
  3. 在训练时构造chunk attention mask（公式5），测试时仅缓存chunk边界（公式7）
  4. 逐层偏移chunk边界（cyclic: layer_idx % P）
  5. 可选：保留L个leading tokens的完整attention（模拟sliding window），增加可控的质量-效率trade-off

## M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

- baseline方法是什么？
  Baseline是DeepSeek-R1-Distill-Qwen-1.5B，一种基于标准Transformer架构的推理模型。Qwen2.5-Math-1.5B通过超过1T MATH tokens的SFT在Qwen2.5基础模型上训练，然后通过DeepSeek-R1的蒸馏流程（从大R1模型进行token-level蒸馏）获得推理能力。在推理时，模型通过生成长chain-of-thought（平均4k-5k tokens per MATH question）来解决复杂数学问题。

  Baseline全栈执行例子（DeepSeek-R1-Distill-Qwen-1.5B推理时生成一个MATH token，batch_size=512）：
  - 算法pipeline：token → embedding lookup → L层Transformer block（每层: RMSNorm → Multi-Head Attention with GQA: QKV投影 → RoPE位置编码 → causal softmax(QK^T/√d) over full KV cache → weighted sum V → output projection → residual → RMSNorm → SwiGLU FFN → residual）→ LM head → logits → next token。对于长chain-of-thought推理，每生成一个token需要对所有历史token计算QK内积（O(N)计算量），KV cache随生成长度线性增长。当生成4k-5k tokens的推理链时，batch size=512的KV cache总量约512×5000×2×n_layers×d_head字节（1.5B约28层×1536 hidden dim → 每token约86KB per layer → 28层×86KB≈2.4MB/token → 512batch×5000tokens≈6TB总KV cache，显存需求极大）。
  - 系统框架：vLLM 0.6.3推理引擎 + PagedAttention KV cache管理。VeRL框架用于rollout生成。推理时compute-bound的prefill（处理短prompt）和memory-bound的decode（生成长chain-of-thought）两阶段分离。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用vLLM内置的FlashAttention等优化kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. **Transformer的KV cache内存爆炸**：长chain-of-thought生成（4k-32k tokens）时，per-token KV cache随生成长度线性增长。大batch（512）推理时，KV cache总内存需求远超GPU HBM（H100 80GB），成为batch size和推理吞吐量的硬瓶颈。解码过程memory-bound导致GPU计算单元利用率低。
  2. **Transformer的二次计算复杂度限制test-time scaling**：生成k个样本做self-consistency/majority voting时，每个样本的每个token需attend所有历史token（O(N²)）。长chain-of-thought（32k tokens）×大样本数（64 samples）× batch推理的总FLOPs极高，限制了test-time compute scaling的实践可行性。
  3. **跨架构推理蒸馏效果未知**：DeepSeek-R1系列仅蒸馏到Transformer架构（Qwen/Llama），能否将推理能力迁移到sub-quadratic架构（如Mamba）并保持性能是未解问题。直接尝试从R1蒸馏到Mamba效果差（38% MATH500, 3.3% AIME24），说明需要创新的训练方案。
  4. **线性RNN在推理任务上的有效性不确定**：虽hybrid RNN模型在通用LM上表现良好，但现代推理模型需要生成长chain-of-thought（包含subtask分解、多尝试、回溯），线性RNN的固定大小hidden state是否能支撑这种复杂推理模式未知。
  5. **RL训练中生成长度受限**：RL训练（GRPO）需要高效生成长序列rollout。Transformer在RL训练的rollout阶段成为瓶颈——生成时间超过actor权重更新（forward+backward）的3倍（DeepScaleR时间分析），训练效率严重受限于生成速度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出M1——基于Mamba架构的hybrid线性RNN推理模型，通过三阶段training pipeline（蒸馏+SFT+RL）将Transformer推理能力迁移到Mamba，实现3x推理加速并将加速转化为test-time compute scaling的准确率增益。

  论文方法全栈执行例子（M1-3B推理时生成一个MATH token，batch_size=512）：
  - 算法pipeline：token → embedding → 28层hybrid block（22 Mamba层 + 6 interleaved Attention层。Mamba层: RMSNorm → input projection expand 2x → causal Conv1d kernel=4 → SiLU → selective SSM scan with state size=16, groups=192: Δ_t=softplus(Linear(x)+bias), A_bar/B_bar=discretize(A, B_t, Δ_t), h_t=A_bar⊙h_{t-1}+B_bar⊗x [O(1) per token, state h∈R^{16×192}] → C_t^T h_t → SiLU gating → output projection → residual。Attention层: 标准MHA/GQA保留。MLP: SwiGLU）→ LM head → logits → next token。每生成一个token，Mamba层仅需O(1)计算和常量内存（h_t固定16×192维），无KV cache增长。仅6/28=21%的attention层需要KV cache。Batch=512时Mamba层内存≈512×16×192×22层×4 bytes≈137MB（远小于Transformer的GB级别）。
  - 系统框架：vLLM 0.6.3推理引擎（利用PagedAttention管理attention层的KV cache）。VeRL框架用于GRPO RL训练的rollout生成（修复了CUDA graph+FSDP兼容性，5x训练加速）。训练框架：Axolotl（蒸馏/SFT阶段）。开源代码：https://github.com/jxiw/M1。
  - 编译框架：论文未明确说明。
  - kernel调度：论文修复了VeRL中Mamba+CUDA graph+PyTorch FSDP的兼容性问题，使CUDA graph启用后Mamba生成速度提升5x。Mamba的selective scan使用硬件高效并行实现（沿非时间维度并行+SRAM resident state）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（KV cache内存爆炸）→ Mamba层的固定大小hidden state替代KV cache：每层Mamba仅需维护h_t∈R^{N×N'}（SSM state 16×192≈3072维），而非per-token K/V cache。batch=512×seqlen=4096时，22层Mamba的state内存=512×3072×22×4≈137MB vs 22层attention的KV cache约512×4096×1536×22×4≈276GB（2000x+缩减）。这使得大batch下GPU内存不再是瓶颈，**解码从memory-bound转为compute-bound**，GPU利用率提升，实测3x吞吐量提升（15169 T/s vs 7263 T/s）。

  - 缺陷2（二次复杂度限制test-time scaling）→ M1 decode O(1) per token：生成k个样本的cost=k×seqlen×O(1)（M1）vs k×seqlen²（Transformer）。固定时间预算下M1可生成更多样本（或更长的chain-of-thought），使self-consistency voting等test-time scaling技术更实用。**速度增益直接转化为准确率增益**（Figure 3 right: 同等时间预算下M1 majority voting accuracy更高；Figure 4 right: M1在同等生成时间下4/5个长度点的accuracy更高）。

  - 缺陷3（跨架构推理蒸馏未知）→ 三阶段pipeline的创新策略：(a) **先做通用MATH蒸馏再做推理SFT**而非直接从R1蒸馏：先用OpenMathInstruct-2（Llama系列generated）将hybrid Mamba训练成强MATH模型（MATH500 45%→74%），再用10B reasoning tokens做推理SFT（74%→82%）。这克服了直接跨架构推理蒸馏的数据不足问题（仅10B reasoning tokens不够，需先建立math基础）；(b) **Reverse KL divergence**用于蒸馏（mode-seeking特性），比forward KL更适合将teacher的概率分布模式集中到student有限的表达能力中；(c) **GQA→full heads expansion**：用额外线性层将Transformer的GQA KV heads扩展到Mamba的full heads，补偿Mamba无KV cache造成的表达能力损失。

  - 缺陷4（线性RNN推理有效性未知）→ Table 1/2证明M1在AIME25/AIME24/MATH500/AMC23/OlympiadBench上**全面匹配**DeepSeek-R1-Distill-Qwen-1.5B（甚至OlympiadBench上M1 47.3 vs R1 43.3），且仅用<50B tokens训练（vs R1的>1T MATH tokens）。这证明了：(a) hybrid Mamba足以支持复杂数学推理；(b) 6个保留attention层（21%）足以弥补纯Mamba在长程信息路由上的不足；(c) GRPO RL对Mamba架构同样有效。

  - 缺陷5（RL训练生成瓶颈）→ M1的3x生成加速使RL训练rollout阶段大幅缩短：(a) 训练时生成长度可扩展至32k（vs Transformer受限于生成速度），更长的chain-of-thought在RL中带来更高准确率（Figure 5: max_len 4096→<10% accuracy, 24k→23%）；(b) CUDA graph+FSDP修复使Mamba生成额外5x加速（在VeRL框架内）；(c) 论文分析指出RL训练中"生成速度>3x actor更新速度"的瓶颈可被线性RNN架构缓解。

  **Stage-by-stage ablation分析**（Table 3）：
  | Stage | MATH500 | AIME24 | 增益分析 |
  |-------|---------|--------|---------|
  | Distill | 38 | 0 | 基础跨架构迁移，无推理能力 |
  | +SFT(MATH) | 45 | 0 | 通用MATH能力建立，无推理 |
  | +SFT(Reason) | 74 | 22 | **最大增益**：推理数据带来+29/+22 |
  | +RL (GRPO) | 82 | 28 | RL进一步+8/+6，巩固推理能力 |

  **为什么先蒸馏Llama而非直接蒸馏R1？**
  论文做了直接蒸馏R1的实验（Distill from DeepSeek-R1-Qwen-1.5B + SFT on 10B reasoning data）→ 仅38%/3.3%（MATH500/AIME24）。假设原因是10B reasoning tokens不足以进行有效的跨架构推理迁移。替代策略：先用OpenMathInstruct-2（Llama系列data）建立Mamba MATH基础模型，再用reasoning data做推理SFT——这种"先通用后专项"的分阶段迁移策略仅需少量reasoning tokens即可获得强推理性能。

  **Test-time compute scaling的设计哲学**：
  - 速度→准确率转换：M1的15000+ T/s吞吐量 ≈ 每秒可生成约2个完整的8K推理链。对比R1-1.5B的7200 T/s，**同等时间预算下M1可生成2x+样本或2x+生成长度**。
  - Majority voting场景：32 samples时M1仅需~16秒（32×8K/15K），R1需要~35秒。在16秒时间预算下，R1只能生成~15个样本。**M1用更多样本弥补了单样本quality的微小差距**。
  - 生成长度场景：M1用更长时间生成更长的chain-of-thought → accuracy monotonically增加。同等时间下M1的更长chain-of-thought → 更高的accuracy。

  **论文的局限性（Limitations中承认）**：
  - 3x speedup尚未利用最新的NVIDIA hybrid Mamba kernel（可进一步提升）
  - Attention层未使用vLLM的attention优化（集成后可进一步提升）
  - 未尝试从Qwen2.5-Math模型蒸馏（因该模型的cross entropy loss on OpenMathInstruct太高，需Qwen系列数据）
  - RL训练速度改进已出现（DeepSeek R1），但Mamba架构可进一步加速这一趋势

## ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

- baseline方法是什么？
  Baseline是标准Transformer-based MLLM（如LLaVA-1.5，使用Vicuna-7B/13B作为LLM backbone，CLIP作为视觉编码器，MLP projector做模态对齐），以及基于Mamba-1的MLLM（如VL-Mamba、Cobra，使用Mamba-1 LLM backbone + 视觉编码器 + 连接器）。标准Transformer MLLM的核心瓶颈是self-attention的O(n²)计算复杂度——每生成一个token需要attend所有历史token，导致长视觉序列（数百个visual tokens + 长文本生成）下推理速度慢、内存消耗大。

  Baseline全栈执行例子（LLaVA-1.5 7B推理时回答图片问题）：
  - 算法pipeline：图片 → CLIP ViT-L/336px → 576个visual tokens → MLP Projector（两层Linear + GELU）→ 拼接文本token → Transformer Decoder（Vicuna-7B: 32层multi-head self-attention, 每层对全部(L_text+576)个token做causal softmax(QK^T/√d)·V, RoPE位置编码, SwiGLU FFN）→ 自回归生成答案。每生成一个token需O(L_text+576) attention计算，KV-Cache随序列增长线性膨胀。
  - 系统框架：PyTorch + HuggingFace Transformers + DeepSpeed ZeRO。训练用LLaVA框架（https://github.com/haotian-liu/LLaVA）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用FlashAttention优化attention kernel）。
  - 硬件架构：NVIDIA A100 GPU，论文未涉及RTL/模拟器层面。

  Baseline缺陷：
  1. Transformer的O(n²)注意力计算导致推理速度慢：TinyLLaVA 3B（Phi-2 backbone）仅38 tokens/s，MobileVLM v2 3B（MobileLLaMA）虽经多项轻量化优化也仅50 tokens/s。长视觉token序列（576-729个visual tokens + 长文本生成）下attention计算量急剧增加。
  2. Mamba-1 based MLLM（VL-Mamba, Cobra）虽用线性SSM替换了Transformer，但Mamba-1的selective scan在长序列上效率不如Mamba-2（Mamba-2的核心SSD层比Mamba-1快2-8倍），且这些工作未充分探索2D视觉特征与SSM的适配——视觉patch生成的序列缺乏自然因果顺序，直接展平为1D序列送入SSM牺牲了2D空间关系。
  3. 现有多模态连接器（如纯MLP）将所有visual token视为独立的1D序列元素，无法建模2D patch之间的空间关系（上下左右相邻patch间的局部上下文）。
  4. 传统视觉编码器（单独使用CLIP/SigLIP）可能丢失低层空间细节信息，因为CLIP类模型优化的是语义匹配而非空间特征保留。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ML-Mamba——用Mamba-2（比Mamba-1快2-8倍的最新一代SSM）替换Transformer backbone构建MLLM，并设计Mamba-2 Scan Connector（MSC）解决2D视觉特征与1D SSM因果建模的gap。

  论文方法全栈执行例子（ML-Mamba推理时回答图片问题，Mamba-2 2.7B backbone）：
  - 算法pipeline：图片 → 双视觉编码器DINOv2（ViT-Large）+ SigLIP → concat[V_siglip; V_dino] → 729个visual tokens → Mamba-2 Scan Connector (MSC-MLP Advanced, BSM): MVSS模块将visual tokens沿前后两个方向各扫描一次（Mamba-2 Block处理1D序列，前向+后向scan合并捕获上下文）→ SwiGLU gated feature extraction → 三层MLP Projector → 拼接文本token → Mamba-2 LLM（2.7B参数, SSD核心层: x_expand=2×, causal Conv1d窗口=4, SiLU → 数据依赖Δ/B/C → ZOH离散化 → recurrent h_t = A_bar⊙h_{t-1}+B_bar⊗x_t, O(1) per token → gating → output）→ 自回归生成答案。每生成一个token仅需O(1)计算和固定大小hidden state，无KV-Cache增长。
  - 系统框架：PyTorch FSDP + HuggingFace Transformers + 自定义Mamba-2 kernels。代码：https://github.com/WenjunHuang94/ML-Mamba（MIT License）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用Mamba-2官方SSD kernel的高效selective scan实现）。
  - 硬件架构：8× NVIDIA A100 80GB GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（Transformer O(n²)推理慢）→ Mamba-2的SSD层实现O(1) per-token生成。ML-Mamba达成171 tokens/s（vs TinyLLaVA 38 tokens/s, MobileVLM v2 50 tokens/s），即使处理729个visual tokens（多于baseline的576和144），总推理时间仅1.47s（vs 6.45s和5.15s），提速约3.5-4.4倍。Mamba-2的RNN-like特性使内存使用不随visual token数量增加而增大，固定大小hidden state存储全部历史信息。

  - 缺陷2（Mamba-1效率不足 + 2D适配缺失）→ (a) 选用Mamba-2而非Mamba-1：Mamba-2的核心SSD层结构化状态空间对偶性使scan效率提升2-8倍，且通过head dim=64的多头设计增强了表达能力；(b) MSC的2D扫描机制（BSM和CSM）：BSM沿前后两个方向处理visual patch序列（前向：原始grid顺序扫描，后向：反转扫描），V_scan = V_f + flip(V_b)，使每个patch能在1D SSM中"看到"2D前后文。CSM沿四个对角线方向扫描，捕获更丰富的2D spatial context。消融实验（Table 7）证明BSM在大多数benchmark上优于CSM。

  - 缺陷3（纯MLP连接器无法建模2D空间）→ MSC模块（Mamba-2 Scan Connector）在MLP之前引入Mamba-2层进行2D visual selective scan。对比三种连接器变体（Table 6消融）：MLP only (VQAv2 73.42) → MSC-MLP Basic (75.09, +1.67) → MSC-MLP Advanced含SwiGLU (75.26, +1.84)。SwiGLU通过SiLU gating + 线性投影提供更复杂的特征提取和模式学习。MVSS模块的2D scan使visual tokens在进入LLM之前通过Mamba-2的selective mechanism（数据依赖的Δ/B/C）自适应地融合局部和全局空间信息。

  - 缺陷4（单编码器空间信息丢失）→ 双视觉编码器DINOv2 + SigLIP组合（Table 5消融）：DINOv2单独 (VQAv2 73.73) + SigLIP单独 (74.61) → 组合 (75.26, +0.65-0.9)。DINOv2提供低层空间特征（self-supervised ViT trained for dense feature matching），SigLIP提供高层语义特征（language-aligned via sigmoid loss），两者互补——前者保留fine-grained spatial detail，后者提供semantic alignment to language。

  推理速度优势的本质：
  - Transformer MLLM：prefill阶段处理所有visual+text tokens O((V+T)²)，decode阶段每token O(V+T) attention。生成的token越多（长答案），KV-Cache越大，decode越慢。
  - ML-Mamba：prefill阶段Mamba-2 scan O(V+T) linear，decode阶段每token O(1)。生成的token越多，速度优势越明显——固定hidden state不增长。
  - 数量化对比：ML-Mamba的256 token生成仅需1.47s → 171 tokens/s。TinyLLaVA（Phi-2）需要6.45s → 38 tokens/s。ML-Mamba速度是TinyLLaVA的4.5倍。

  消融实验关键发现：
  - 语言模型规模（Table 4）：Mamba2-2.7B在所有benchmark上全面超越780m和1.3b变体，证明更大SSM backbone的收益类似Transformer scaling law。
  - 视觉编码器组合（Table 5）：DINOv2 + SigLIP组合在所有6个benchmark上排名第一，单一编码器之间存在互补而非替代关系。
  - 连接器结构（Table 6）：MSC-MLP Advanced > MSC-MLP Basic > MLP only，证明Mamba-2 scan + SwiGLU均为有效贡献。
  - 扫描机制（Table 7）：BSM和CSM在不同任务上互有胜负（BSM在VQAv2/GQA/VizWiz/VSR上更优，CSM在TextVQA/POPE上有优势），两者性能接近，BSM整体略优。

## Rethinking_Token_Reduction_for_State_Space_Models

- baseline方法是什么？
  Baseline是将现有Transformer的SOTA Token Reduction方法（EViT的pruning、PuMer的bipartite merging）直接应用于Mamba SSM模型。这些方法原为Transformer设计但直接迁移到SSM时严重失败（如Mamba-2.8B在20% FLOPS reduction下，EViT准确率从63.3%降至43.6%，PuMer降至37.2%）。

  全栈执行例子——以EViT pruning方法处理一个输入token序列为例：

  **算法pipeline层**: Mamba-2.8B正常加载，EViT对当前层所有token按attention值（[CLS] token对其他token的attentiveness）排序，直接删除最不重要的20% token，进入下一层处理。
  **系统框架层**: PyTorch + HuggingFace Transformers的标准推理pipeline，无Serving框架修改，token reduction通过hook插入Mamba block之间。
  **编译框架层**: 论文未明确说明（无编译框架修改）。
  **kernel调度层**: 标准PyTorch CUDA kernel，无自定义kernel。Mamba中的SSM selective scan使用Mamba原生的triton kernel（来自mamba-ssm库），token reduction后调用相同的核函数处理更少的token。
  **硬件架构层**: NVIDIA A100 80GB GPU。未涉及硬件修改。

  Baseline的核心缺陷：
  - **Pruning缺陷**：pruning删除的"低重要性"token仍然包含不可恢复的信息，这些信息在SSM的序列化递推计算（h_t = A̅h_{t-1} + B̅x_t）中被逐token放大传播，删除任何一个token都会影响后续所有token的hidden state累积。
  - **Merging缺陷**：bipartite merging将token均匀分成两组，盲目将一组merge进另一组，完全忽视不同token的内在重要性差异。在SSM中，某些关键token对等式y = x * K̅的卷积结果有决定性影响，将其merge到不重要token中会导致输出发生根本偏移。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出了UTRC（Unified Token Reduction by token importance Classification），通过"重要性分类→相似度匹配→混合prune/merge"的三阶段pipeline解决baseline的缺陷：

  **（1）SSM原生Token重要性度量**：从SSM的hidden state y计算重要性，而非从attention值。具体地，`S_i = Σ_d max(0, y_{i,d}) / D'`，利用SSM高维通道空间（D'）中每个通道对各token的细粒度激活响应。对比发现clip操作（max(0,·)）优于ℓ1/ℓ2 norm，说明只关注正向激活的通道更有信息量。
  **（2）Token重要性分类**：按重要性将token分为M_A（低重要性50%）和M_B（高重要性50%），然后为M_A中每个token寻找M_B中最相似的对应token。只有相似度最高的p%连接对被保留处理，不相似的连接直接丢弃（即对应的不重要token不被保护）。
  **（3）混合Pruning + Merging（UTR）**：对保留的连接对，q比例的做pruning（仅删除M_A中token），(1-q)比例的做merging（将M_A token平均值融合进M_B对应token）。这精确平衡了"信息保留"（merging保留语义信息）和"冗余消除"（pruning清除纯粹冗余），q=0.5效果最优。
  **（4）Intra-layer分支解耦**：Hidden states上使用hybrid策略（q=0.5），Residual connections上仅用merging。原因是残差连接传递的是前一层原始信息，pruning会导致关键残差信息永久丢失，而merging可以将多路径信息融合保留。这解决了baseline中hidden和residual token不同步减少导致的index misalignment问题。
  **（5）层次化reduction**：不每层都reduction（相邻层重要性变化小），而是每5层应用一次，从第10~12层开始（前几层token representation尚未充分成熟）。

  全栈执行例子——以Mamba-2-2.7B在[12,17,22,27,32,37,42]层执行20% FLOPS reduction为例：

  **算法pipeline层**：在第12层SSM block的selective scan输出处，hook截取hidden state y ∈ R^{B×N×D'}，执行：(a) 计算S = max(0, y).sum(dim=-1) / D'得到每个token的重要程度；(b) 按S中位数二分，重要性低的进入M_A；(c) 计算M_A中每个token与M_B的cosine相似度矩阵；(d) 保留最相似的p%连接；(e) 对保留连接中前50%（q=0.5）执行pruning——仅从hidden states移除M_A token，M_B token保持；后50%执行merging——`T[f_i] = (T[a_i] + T[f_i]) / 2`。残差路径仅执行merging（保留所有残差信号的贡献）。第13~16层不执行reduction（间隔=5层逻辑），直到第17层再次执行。N从2048 tokens逐步减少至约1638 tokens（减少~20%FLOPs）。
  **系统框架层**: PyTorch 2.x + HuggingFace Transformers推理pipeline。UTRC作为hook注入每个Mamba block的SSM输出位置（在Linear投影和残差加法之前），不修改Mamba模型权重或架构本身。评估时用特殊logit裁剪：token数减少后PPL/Accuracy只在对应长度的非压缩token上计算。
  **编译框架层**: 论文未明确说明（无编译框架修改）。
  **kernel调度层**: 标准PyTorch CUDA kernel + Mamba原生的selective scan triton kernel（来自mamba-ssm库）。Token reduction后输入SSM scan的序列变短，直接享受更少的scan步数加速。无自定义kernel。GPU峰值内存因token数减少下降14.4%~40.0%（10%~30% FLOPS reduction对应），吞吐提升1.07×~1.37×。
  **硬件架构层**: NVIDIA A100 80GB GPU。未涉及硬件修改。

## Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

- baseline方法是什么？
  Baseline 是标准 Mamba-2 模型（Dao & Gu, 2024），训练在 8K 上下文长度上。Mamba-2 使用选择性状态空间模型（SSM），每层 H 个 head 并行计算，状态大小 HPN = 256d（N=128, P=64, H=2d/P），约等于同等 hidden dim Transformer 的 KV cache 大小。Mamba-2 内置指数记忆衰减机制 α_t = exp(-Δ_t·exp(A)) ∈ (0,1)，理论上 α_t → 0 可完全遗忘、α_t → 1 可完全保留。Baseline 在训练长度内表现良好（LM loss 正常，passkey retrieval 近乎完美），但 context > 8K 后：(a) LM loss 急剧爆炸（perplexity 从 ~10 升至 ~100+），(b) passkey retrieval 准确率从 >95% 骤降至 ~0%，(c) 状态分布（mean/variance）出现 outlier channel 驱动的方差爆炸，导致输出 incoherent。

  全栈执行例子（Mamba-2 370M 在 8K 训练长度下处理 32K 上下文）：
  - 算法层：输入 32K tokens → Embedding → 48 层 Mamba-2 block（每层: RMSNorm → 输入投影 expand 2× → CausalConv1d(k=4) → SiLU → SSM selective scan: Δ_t=Softplus(W_Δ u_t+b_Δ), α_t=exp(-Δ_t·exp(A)), B_t=σ(Conv(W_B u_t)), C_t=σ(Conv(W_C u_t)), x_t=SiLU(Conv(W_x u_t))^T → h_t=h_{t-1}·α_t + Δ_t·B_t·x_t → y_t=C_t·h_t + D⊙x_t → SiLU gate + 输出投影 → residual）→ LM head → logits。在 t=8K-32K 时，某些 head 的 α_{1:t} 仍 > 0.997（首 token 几乎不衰减），状态中累积了过多历史 token 的加权信息，导致 memory interference——query C_t 与状态中任意 token 的 B_i 非正交时产生检索误差，token 越多误差越大，最终 loss 爆炸。
  - 系统框架层：PyTorch + HuggingFace Transformers（Mamba-2 官方实现）。FP32 推理，greedy decoding。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Mamba-2 官方 selective scan kernel（Triton/CUDA fused kernel），并行化递归计算。状态为固定大小，O(1) per-token 计算和内存。
  - 硬件架构层：NVIDIA A800 80GB GPU。未涉及硬件修改。

  Baseline 核心缺陷：
  1. **无法遗忘（Inability to Forget）**：虽然 Mamba-2 有 α_t 衰减机制，但训练过程中模型学会了保留几乎所有信息（α_t 始终接近 1）以最小化 LM loss，而非在必要时遗忘。在 8K 训练长度内这可行，但超出后状态被"塞满"，信息干扰导致检索失败。
  2. **状态过参数化（State Overparameterization）**：状态大小 N_S = 256d 相对于 8K 训练长度过大，模型无需学习遗忘即可达到低 loss——本质是一种过拟合。更多训练数据反而加剧此问题：模型学会更激进地保留信息（Figure 8: passkey 精度随数据量增加反而下降）。
  3. **方差爆炸（Variance Explosion）**：超过训练长度后，某些 head 的状态均值和方差急剧变化，由少数 outlier channel（~5% 的 channel）驱动，这些 channel 的值在 t > T_train 时发散。
  4. **大模型更差**：780M 的长度泛化比 370M 更差（更大的状态 → 更严重的过参数化）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从诊断角度分析了 Mamba-2 的长度泛化失败根因，提出了两条高层次的解决方案方向（遗忘诱导方法）和关键训练指导原则。

  **核心发现与设计原则：**
  (1) **遗忘阈值定律**：T_forget = 5.172 · N_S − 4.469（线性关系，R² > 0.999）。训练长度必须大于此阈值，模型才能学会有效遗忘。对于 370M 模型（N_S=12.9M），T_forget ≈ 66.7K tokens。
  (2) **召回能力定律**：T_recall = 4.756 · (1.365^{N_S} − 1) − 0.742（指数关系，R² > 0.999）。即使超过遗忘阈值，模型仍可检索信息到远超训练长度的上下文——370M 模型持续预训练后在 256K passkey retrieval 上近乎完美。
  (3) **状态过参数化假说**：遗忘只会在训练上下文的信息量超过状态容量时发生。这解释了为什么更大的状态需要更长的训练长度。

  论文方法全栈执行例子（使用 Sliding Window 干预的 Mamba-2 370M 处理 32K 上下文）：
  - 算法层：与 baseline 相同的前向过程，但在每个 Mamba-2 head 的 state update 后添加 Sliding Window 后处理：(a) 正常计算 h_t；(b) 维护 Δ_sum（ Δ 的累积和）和 h_{t-w}（w 步前的状态）；(c) 计算 α_window = exp(-Δ_sum·exp(A))；(d) h_t^{(w)} = h_t − α_window·h_{t-w}（精确窗口状态）；(e) 用 h_t^{(w)} 替代 h_t 进行 query y_t = C_t·h_t^{(w)} + D⊙x_t。这等价于强制遗忘 window 之前的所有 token，将有效上下文缩短为 w tokens。RRI 方法更温和：α_t' = α_t^{0.9999}（轻微加速衰减），B_t' = 0.75·B_t（减弱新信息插入），在 32K 上将 LM loss 从 ~15 降至 ~8。两种方法均无需训练。
  - 系统框架层：修改 Mamba-2 的推理代码，在 state update 后插入干预逻辑。无需重新训练或微调。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Sliding Window 仅需额外存储 h_{t-w} 和 Δ_sum（两种浮点数，忽略不计），额外计算量为一个标量乘法和一个矩阵减法（O(NP)，极小）。RRI 的额外开销为标量乘法和指数运算，同样可忽略。
  - 硬件架构层：NVIDIA A800 80GB GPU。未涉及硬件修改。

  关键设计选择映射到缺陷：
  - 缺陷1（无法遗忘）→ **通过状态过参数化假说揭示了根因**：遗忘机制未有效学习不是因为架构缺陷，而是训练长度不足。RRI 和 Sliding Window 证明了"强制遗忘可以修复长度泛化"，提供了因果证据：问题在遗忘，不在检索。
  - 缺陷2（状态过参数化）→ **建立 T_forget ∝ N_S 的线性缩放律**：对于任意状态大小 N_S，存在最小训练长度使模型学会遗忘。这为训练长上下文 Mamba 模型提供了具体指导——8K 训练长度对 370M 模型（N_S=12.9M）严重不足，需 ~67K。
  - 缺陷3（方差爆炸）→ **通过 per-channel 状态统计揭示了 outlier channel 现象**：发现方差爆炸由少数 channel 驱动，其余 channel 保持稳定。使用 "newlines" prompt（恒定输入）来分离"上下文长度效应"和"输入变化效应"，确认爆炸是 context length 而非输入内容的函数。
  - 缺陷4（大模型更差）→ **状态大小是根本变量**：不是参数数量，而是 N_S = 256d（每层状态元素数）决定长度泛化能力。这解释了为什么 780M 比 370M 更差——本质是 N_S 更大（19.3M vs 12.9M），而非参数多。

  持续预训练的关键实践指导：
  - 训练长度应随状态大小线性增长：T_train > 5.172 · N_S − 4.469
  - 使用 Truncated BPTT（12 序列拼接，stop gradient 在序列边界）使状态初始值分布更多样化
  - WSD LR scheduler（10% decay steps）允许从中间 checkpoint 高效恢复
  - 数据过滤：丢弃短于 4K tokens 的文档，确保训练数据有足够长距离依赖
  - 验证策略：passkey retrieval 对长度泛化敏感度远高于 validation loss，应作为主要验证指标

  370M Mamba-2 持续预训练结果（256K 上下文）：
  - 在完整 256K passkey retrieval 上达到近乎完美的准确率
  - 据论文所知，这是首个 <1B 参数模型在此长度上达到如此性能
  - 超越了同参数规模的 Transformer 模型

  论文的局限与未解决问题：
  - RRI 和 Sliding Window 方法会牺牲短上下文性能（弱记忆插入）
  - Sliding Window 需要选择窗口大小 w，w 的选择对最终性能敏感
  - 论文未提供正式的训练代码仓库（依赖 Mamba-2 官方实现 + 自定义训练脚本）
  - 遗忘阈值的线性关系仅在 Mamba-2 上验证，RWKV 等其他 RNN 架构的关系未充分探索
  - 780M 模型的 T_forget 超出实验资源（>128K），未能直接验证

## VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

- baseline方法是什么？
  Baseline 是标准 Transformer-based VLM（以 LLaVA-1.5 为代表），使用 Vicuna LLM + CLIP 视觉编码器 + 投影层架构。视觉信息通过 cross-modal projector 将 CLIP 的视觉特征映射为与 LLM 同维度的 visual tokens，拼接到文本 token 序列中，通过 LLM 的 causal self-attention 实现跨模态融合。Baseline 的核心缺陷来自于 Transformer 自注意力机制的二次复杂度：(1) 推理时每生成一个 token 需要 O(N) attention 计算和 O(N) KV cache 内存增长，长序列（多轮对话、高分辨率图像、长文档）下推理延迟和内存开销快速膨胀，不适合边缘设备部署；(2) 即使使用 FlashAttention 优化，KV cache 仍随序列长度线性增长，在 24K+ tokens 时 GPU 内存成为瓶颈。

  Baseline 全栈执行例子（LLaVA-1.5 7B 推理一个包含 576 image tokens + 20 text tokens 的 VQA 请求）：
  - 算法层：输入图像 → CLIP-L(ViT-L/14, 336×336) → Z_v ∈ R^{576×1024} → Projector MLP → H_v ∈ R^{576×4096} → 拼接到 text tokens → Vicuna-7B 32 层 Decoder，每层 causal self-attention 在 (576+N_text+N_generated) 个 token 上计算 softmax(QK^T/√d)·V → 自回归逐 token 生成 → 输出 answer。每生成一个 token 需 O(576+N_text+N_gen) attention 计算，KV cache 线性增长。
  - 系统层：PyTorch + Transformers + DeepSpeed ZeRO。推理使用 FlashAttention 加速。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention kernel（IO-aware tiling），降低 HBM 访问但仍保持二次复杂度。
  - 硬件架构层：NVIDIA A100 GPU。

  Baseline 核心痛点：
  1. Transformer 自注意力使推理延迟和内存随序列长度线性以上增长（O(N²) 总计算量），限制了 VLM 在资源受限设备上的部署
  2. Linear RNN（如 RWKV）已经证明在纯文本 LLM 上可以匹配 Transformer 性能并实现 O(1) 推理，但在多模态 VLM 领域尚未被探索

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 VisualRWKV，首次将线性 RNN（RWKV）架构应用于多模态 VLM，通过三项创新设计使 RNN-based VLM 达到与 Transformer-based LLaVA-1.5 竞争的性能：

  **设计 1: Data-dependent Recurrence → 提升 RNN 模型容量**
  Baseline 中最初的 VisualRWKV-Base 使用了 RWKV-5 的数据独立 recurrence（固定 token shift μ + 固定 time decay w），模型容量有限。VisualRWKV 将 Token Shift 升级为 Data-dependent：`ddlerp(a,b) = a + (b-a) ⊙ lora(a + (b-a) ⊙ μ_x)`，其中 lora(x) = λ + tanh(xA)B 使用低秩矩阵引入输入依赖的偏移量。同时将 Time Decay 从固定 w 变为动态 `w_t = exp(-exp(lora_d(ddlerp_d(x_t, x_{t-1}))))`，使每个 token 在每个 channel 上根据内容决定遗忘速度。这一设计使模型能动态调整"记住什么、遗忘什么"，显著提升在 VQA/SQA/GQA 等 benchmark 上的表现（Table 1: VQA 51.08→65.82, SQA 41.94→46.55）。

  **设计 2: Sandwich Prompt → 解决 RNN 无法回溯的问题**
  Transformer 的 self-attention 允许模型在任何时刻从 KV cache 中检索任意历史 token，而 RNN 的序列特性意味着信息一旦被"读入"state 后，模型无法直接回溯原始 token。传统的 Image-First 或 Image-Last prompt 分别导致模型处理图像时不考虑问题（丢失上下文）或先读问题后读图像时忘记问题（RNN 信息被图像 token 覆盖）。Sandwich Prompt 将图像 token 插入 instruction token 中间（即 [Q_prefix]+[image]+[Q_suffix]），使模型：(a) 先读 prefix 激活正确的"检索意图"；(b) 带着意图处理图像信息，提取与问题相关的视觉特征；(c) 再读 suffix 完成最终回答。这种设计利用了 RNN 在短时间内（prefix 长度内）可以保持良好局部记忆的特性，同时避免了长序列中的遗忘问题。Table 3 证明 Sandwich Prompt 在所有三种 prompt 中表现最佳。

  **设计 3: 2D Image Scanning → 解决 RNN 单向性与图像多向性的矛盾**
  RWKV 本质上是为 1D 因果语言序列设计的，视觉序列（2D patch grid）无因果关系，直接进行单向扫描会丢失空间结构信息。VisualRWKV 将 RWKV blocks 的方向交替排列为 Forward → Backward → Forward → Backward（BiDir），或 Forward/Backward/Upward/Downward 四向交替（MultiDir）。每一层的扫描方向固定（训练和推理保持一致），不增加任何参数和计算量（仅改变 token 输入顺序）。这种交替设计使模型在不引入额外开销的情况下获得了 2D 空间感知能力。Table 4 证明 BiDir（VQA 65.62）和 MultiDir（66.04）均显著优于 UniDir（51.03）。

  论文方法全栈执行例子（VisualRWKV 7B 推理一个 VQA 请求）：
  - 算法层：输入图像 → CLIP-L（ViT-L/14, 336×336, 冻结）→ Z_v [576×1024] → Projector（2 层 MLP, 可训练）→ H_v [576×4096] → Sandwich Prompt 构建：[System|Q_prefix|576 image tokens|Q_suffix] → 32 层 RWKV-6 Block（每层: ddlerp Token Shift → dynamic decay w_t → WKV linear attention [matrix state S∈R^{64×64} per head, O(1) update] → SiLU gate → LayerNorm → output → channel mixing → residual）→ LM Head → logits → next token。逐 token 生成时，RWKV block 每步仅需更新 64×64 的矩阵 state（无 KV cache），计算 O(1)，内存恒定。
  - 系统层：NVIDIA PyTorch NGC Container (23.07-py3) + lightning 1.9.5 + DeepSpeed 0.12.6。开源代码：https://github.com/howard-hou/VisualRWKV。
  - 编译框架层：论文未明确说明。
  - kernel调度层：RWKV 的 WKV 计算使用 parallel scan 实现训练并行化，推理时切换为纯循环模式（O(1) per token）。RWKV-6 kernel 在 16K 序列时比 Flash Attention v2 快 4.2×（来自 RWKV-6 论文）。
  - 硬件架构层：训练 8×A100-80GB GPU；效率对比单张 L20-48GB。VisualRWKV 7B 比 LLaVA-1.5 7B 在 24K tokens 时快 3.98×，GPU 内存节省 54%。

  **效率优势的根本机制**：Transformer VLM 每生成一个 token 需要计算与所有历史 token 的 attention，随对话轮数/图像分辨率增加，延迟线性增长。VisualRWKV 通过 WKV 的递归形式 (`wkv_state_t = diag(w_t)·wkv_state_{t-1} + k_t^T·v_t`) 将所有历史信息压缩进固定大小的矩阵 state（per-head 64×64），推理时仅需 O(1) 更新和读取，且内存完全不受序列长度影响。这意味着在生成 24K tokens 的长序列时，VisualRWKV 不仅更快（3.98×），且内存使用保持在初始水平（LLaVA-1.5 此时已增长到初始的约 2.2×），使 RNN-based VLM 特别适合多轮对话、视频理解等长序列多模态场景。
