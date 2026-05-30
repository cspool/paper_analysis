## PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

- baseline方法是什么？
  Baseline 是静态稀疏注意力方法（Sliding Window、Stride Slash、Dilated Attention、LongNet）以及 Full Attention。核心缺陷：(a) Sliding Window：感受野随层数线性增长——到达距离 N 的 token 需要 O(N) 层，对 32K 上下文模型仅 28 层，远不足以覆盖全序列；(b) Stride Slash：虽然通过等间距 slash token 将路径复杂度降至 O(√N)，但 slash token 放置未优化覆盖率，效率不及最优；(c) Dilated Attention：使用膨胀滑动窗口，偶数位置可达但所有距离为 2k+1 的奇数位置 token 不可达，覆盖率仅 ~50%；(d) LongNet：多 mask 叠加设计，需要 O(log N) 层到达距离 N 但存在覆盖盲区（每段末尾 token 不可达），不可达 token 导致 passkey 检索失败。

  全栈执行例子（Sliding Window Attention baseline, Qwen2-7B, 32K context, A800）：
  - **算法层**：每个 token 仅关注前 W 个 token（9 blocks × 256 = 2304 tokens）+ 1 block sink tokens。Sequence 长度为 32768（128 blocks），最后一层的最后一个 token 仅能间接访问到距离约 9×28=252 blocks 的信息（线性扩展），前 56 blocks 的信息理论上不可达。
  - **系统框架层**：使用标准 Transformer decoder forward pass。论文未修改 serving 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 PyTorch FlexAttention 将 sliding window mask 编译为 block-sparse attention kernel。每个 query block 仅需加载 ~10 个 KV blocks 到 SRAM，计算复杂度 O(NW)，其中 W 为窗口大小。
  - **硬件架构层**：A800 GPU。block=256 对齐 GPU compute core 内存访问，但线性感受野限制模型长程依赖能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  POWERATTENTION 通过图论建模将稀疏注意力设计问题重新表述为 DAG 中最大化节点可达性的最优边集问题，提出 power-of-2 距离连接策略实现指数级感受野扩展。

  **(1) 算法层——Power-of-2 连接策略实现指数感受野扩展**：
  将注意力 mask 建模为 DAG 的邻接矩阵，约束条件为最大出度固定（sparsity 约束）。POWERATTENTION 的边集构造：每个节点 i 仅连接满足 i-j=2^k 的节点 j（即距离为 2 的幂次的位置）+ 滑动窗口 + sink tokens。关键性质：(a) 最大出度 ≤ log n（每个可能的 k 最多一条出边）；(b) 任意节点对 (i, j) 距离 ≤ log n——将距离 d=i-j 按二进制拆分，d = Σ 2^{k_t}，路径 i → (i-2^{k₁}) → ... → j，长度 = popcount(d) ≤ log n。
  
  对比 Baseline 的改进：
  - vs Sliding Window (O(N) layers)：POWERATTENTION 仅需 O(log N) layers 覆盖全序列，28 层模型在 32K 下覆盖所有 2^28 个 token（远超序列长度）
  - vs Stride Slash (O(√N))：指数增长 (2^d) 优于平方根增长 (d²)
  - vs Dilated/LongNet (incomplete coverage)：POWERATTENTION 的 DAG 边集保证所有 token 可达，无覆盖盲区

  **(2) 训练策略——Continued Pretraining + Fine-tuning 激活信息流机制**：
  先在 SlimPajama (1B tokens) 上 continued pretraining，再用 ChatQA 2（含 long-range dependencies）fine-tuning。信息流探针实验证实：未经训练的 POWERATTENTION 呈现 phase-transition 式跳跃信息传播但分类精度仅 ~56%，训练后提升至 100%。对比 sliding window 训练后反而退化（48%→37%），证明 POWERATTENTION 的 DAG 结构能有效利用训练信号激活指数感受野信息流，而 sliding window 因固有线性感受野限制无法从训练中获益。

  **(3) 系统层——Hybrid Architecture 平衡效率与性能**：
  在 RULER 评估中采用 hybrid architecture：每 7 层保留 2 层 Full Attention，其余 5 层使用 POWERATTENTION。该设计确保 attention sink 和复杂语义处理有足够的全注意力层支持，同时最大化稀疏注意力层的效率收益（稀疏度 94%）。论文指出实际部署中替换 sliding window 为 POWERATTENTION 可 deliver 进一步性能提升。

  **(4) Kernel调度层——FlexAttention Block-Sparse 编译**：
  POWERATTENTION 的 mask 定义（`(blk_qk & (blk_qk-1))==0`）完全是 block 级别的位运算，FlexAttention 直接将其编译为 block-sparse kernel。每个 query block 仅需处理 O(log n) 个 power-of-2 KV blocks + 5 window blocks + 1 sink block，总计 ~10 blocks @32K。时间复杂度 O(N log² N) 接近滑动窗口的线性复杂度，128K 时 kernel 比 Full Attention 快 21.6×，比 MInference 快 5.3×。

- baseline方法是什么？
  Baseline 是标准 Full Attention（Softmax Attention, Vaswani et al. 2017）在 GQA+MoE 架构上的实现。核心缺陷：(a) 计算复杂度 O(t²) 随序列长度平方增长——64k 序列下 attention 计算占总延迟 70-80%；(b) training/prefilling 阶段 compute-bound，decoding 阶段 memory-bound（每次解码需加载全部 KV cache），两种瓶颈机制不同但 Full Attention 无法针对性优化；(c) 现有稀疏注意力方法（H2O、Quest、InfLLM 等）多仅用于推理阶段的 KV cache 剪枝或选择，缺乏训练支持，导致 pretrain→inference 架构偏差（architectural bias）和性能退化；(d) 部分方法（ClusterKV 的 k-means、MagicPIG 的 SimHash）包含不可微操作，无法端到端训练；(e) token-granular selection 方法（HashAttention）导致非连续内存访问，无法利用 FlashAttention 的 blockwise 计算优势。

  全栈执行例子（Full Attention baseline, 27B GQA+MoE 模型, 64k 序列, A100）：
  - **算法层**：输入 X ∈ R^{65536×2560}，Q = X·W_Q ∈ R^{65536×2560}（64 heads × 40 dims, GQA 扩展），K = X·W_K ∈ R^{65536×(4·192)}（4 GQA groups），V = X·W_V ∈ R^{65536×(4·128)}。每层存储 4 groups × (192+128) × 65536 ≈ 83.9M float16 到 KV cache。per-token attention 矩阵 S = QK^T/√d_k ∈ R^{64×65536}，softmax 后乘 V。训练时所有 65536×65536 attention scores 都参与梯度计算。
  - **系统框架层**：论文未修改 serving 框架，使用标准 Transformer decoder with GQA+MoE。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 Triton 实现的 FlashAttention-2 kernel，tiling strategy 按时间连续 query block 加载到 SRAM，内循环加载 K/V tile 做 online softmax。64k 时 forward pass 的 HBM 访问量巨大，backward 需重计算 attention 矩阵。
  - **硬件架构层**：A100 GPU (312 TFLOPS FP16, 2 TB/s HBM2e bandwidth, critical arithmetic intensity ≈ 156 FLOP/byte)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  NSA 提出三层解决方案对应 Baseline 的三个核心缺陷：

  **(1) 算法层——Hierarchical Sparse Attention 替代 O(t²) 密集计算**：
  NSA 将 Full Attention 的密集 K, V 替换为三条并行的信息路径：
  - Compression path：每 32 token block 压缩为 1 个压缩 token（stride 16），全局扫描成本从 O(t) 降至 O(t/16)
  - Selection path：利用 compression attention 的中间分数免费推导 block 重要性，选 Top-n=16 个 block（每 block 64 tokens），保留 1024 个精细 token
  - Window path：独立 512 token 滑动窗口处理局部模式，防止局部 shortcut 压制全局学习
  三条路径通过可学习门控 g_t^c 融合，使用独立 K, V 投影矩阵。关键设计：(a) Compression 和 Selection 共享 attention score 计算——p_t^{cmp} 直接用于推导 selection block 重要性，零额外开销；(b) Blockwise selection（非 token-granular）确保连续内存访问，匹配 FlashAttention 范式；(c) GQA 兼容——跨 head 聚合重要性分数确保 group 内 KV block 选择一致，解码时一次加载供所有 head 共享。

  **(2) 训练可行性——end-to-end Trainable with Differentiable Operators**：
  所有操作（MLP compression、softmax attention、gating）全程可微。Selection 的 Top-n 操作在 forward 中做离散选择（只计算选中 block 的 attention），backward 时由于梯度仅对选中 block 的非零 attention score 传播，形成隐式的直通估计（straight-through estimation）。对比 ClusterKV（k-means 不可微）、Quest（heuristic min-max 无梯度）、HashAttention（token-granular 非连续），NSA 实现了原生的端到端训练。

  **(3) Kernel调度层——Group-Centric Sparse Attention Kernel**：
  针对 selection attention 设计专用 Triton kernel：
  - **问题**：FlashAttention 的「按时间连续 query block 加载」策略导致 query block 内不同 position 的 I_t 不同，KV block 访问碎片化
  - **NSA Kernel**：改为按 GQA group 加载——每个 grid program 处理一个 query 位置 t，加载该 GQA group 内所有 H 个 heads 的 Q ∈ R^{[H, d_k]} 到 SRAM，然后按 I_t 顺序加载连续 KV block，一次 KV 加载服务 H 个 heads
  - **效率**：算术强度 = H × B_k × (2d_k+3d_v) / (B_k×(d_k+d_v)) ≈ 14× H=16 ≈ 14，超越 A100 critical arithmetic intensity，从 memory-bound 转为 compute-bound

  全栈执行例子（NSA, 27B GQA+MoE, 64k 序列, A100）：
  - **算法层**：每个 query token t 不计算全部 65536 个 key，而是：(a) 与 ~4096 个 compression tokens 计算 attention（65536/16）；(b) 与 1024 个 selected fine-grained tokens 计算 attention；(c) 与 512 个 window tokens 计算 attention。总计 N_t ≈ 5632 ≪ 65536。
  - **系统框架层**：论文未修改 serving 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：compression/window attention 复用 FA2 kernel；selection attention 使用 NSA 专用 group-centric kernel。Forward: Grid loop per query position → load GQA group Q → inner loop 遍历 I_t (n=16 blocks, 4 iterations when B_k=256) → 每 iteration 加载连续 KV block 到 SRAM → 对 H heads 同时做 S = Q @ K^T 和 online softmax。数据流：HBM Q → SRAM Q(一次) → 内循环 HBM K/V → SRAM K/V(每次) → Tensor Core S = Q@K^T → SRAM P=softmax(S) → Tensor Core P@V → SRAM O → HBM O。64k forward 9.0× speedup, backward 6.0× speedup vs FA2 Triton 实现。
  - **硬件架构层**：A100 GPU 8卡系统，Triton kernel 通过 group-centric 加载使 arithmetic intensity 超临界值，利用 Tensor Core 做 dense matmul，HBM 带宽不再是瓶颈。解码时 KV cache 加载量从 65536 降至 5632 等效 token 量，预期 11.6× speedup。

- baseline方法是什么？
  MTLA 的 baseline 是标准 Multi-Head Attention (MHA) 和 Multi-Head Latent Attention (MLA)。MHA 每个 attention head 保留独立的 KV cache，per-token KV cache 大小为 2·d_h·n_h·l。MLA 将 K, V 压缩到低秩 latent space C ∈ R^{T×r}（r ≪ d），将 per-token KV cache 大小降为 9d_h·l/2 ≈ 4.5d_h·l 个元素。然而 MLA 仅压缩了 latent 维度（head 和 dimension），未触及 temporal 维度——KV cache 序列长度仍为 T，在长序列场景（如 speech T≈数千帧）下 KV cache 仍然占用大量 GPU 内存，且 per-token attention 复杂度为 O(T)。

  全栈执行例子（MHA baseline, ST task, T=2048 speech frames, d=512, n_h=8, RTX 6000 Ada）：

  - **算法层**：输入 X ∈ R^{2048×512}。Q = X·W_Q ∈ R^{2048×512}（每 head 64 dims），K = X·W_K ∈ R^{2048×512}，V = X·W_V ∈ R^{2048×512}。每一层每个 token 存储 2·d_h·n_h = 2·64·8 = 1024 个标量到 KV cache。self-attention per-token 需与全部 2048 个历史 token 计算 dot-product：softmax(QK^T/√d_h) ∈ R^{2048×2048}。训练可并行，推理需逐一解码 token 并每次 attend 整个 KV cache。核心缺陷：(1) KV cache 内存爆炸——2048 tokens × 9 layers × 1024 elems × 2 bytes ≈ 37.7 MB 仅 KV cache；speech 序列往往更长（数千帧）；(2) per-token 解码复杂度 O(T) 随序列线性增长；(3) GPU 内存带宽成为瓶颈——每次解码需加载全部 KV cache。

  - **Serving/框架层**：基于 Fairseq toolkit 的 Transformer decoder，使用 beam search 解码。论文未修改 serving 调度逻辑，仅替换 self-attention 模块。

  - **kernel调度层**：标准 attention 使用 PyTorch 原生实现（无 FlashAttention 时为慢路径）或 FlashAttention-2（快路径），均在完整 K, V 矩阵上进行 tiled softmax。无 temporal compression 优化。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. KV cache 沿 temporal 维度线性增长 O(T)，长序列下内存占用巨大
  2. Per-token 解码复杂度 O(T)，推理时间随序列长度线性增长
  3. MLA 只压缩了 latent 维度（head 数降 + 低秩压缩），未利用 temporal 冗余——相邻 token 的 KV 信息高度相关但被独立存储

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MTLA 在 MLA 的低秩 latent 压缩基础上，进一步沿 temporal 维度压缩 KV cache，通过 hyper-network 动态合并相邻 KV cache vectors，将序列长度从 T 降至 t = ⌈T/s⌉。

  MTLA 全栈执行例子（ST task, s=2, T=2048, d=512, r=256, n_h=8, RTX 6000 Ada）：

  - **算法层**：
    1. **低秩 latent 压缩**（同 MLA）：C = LayerNorm(X @ W_r) ∈ R^{2048×256}，r=256 ≪ d=512。
    2. **Temporal 压缩 via hyper-network**：对每 s=2 个 latent vectors，hyper-network 通过 w_i = Sigmoid(Linear(c_i) · Linear(pe_j)) 生成动态 merge weights，合并 ĉ_j = w_1·c_1 + w_2·c_2 等，得到 Ĉ ∈ R^{1024×256}，序列长度减半。
    3. **Stride-aware causal mask**（训练时）：解决并行训练中 compressed KV cache 与 incremental inference attention pattern 不匹配的问题。mask[n, m] = 0 iff n==m or (m < n and m % s == 0)，确保 training 时每个 query 的 attention pattern 与 inference 时一致。
    4. **Absorbed attention**：利用矩阵乘法的结合律，将 W_K 吸收进 W_Q、W_V 吸收进 W_O：attention = softmax(X·(W_Q·W_K^T)·Ĉ^T/√d_h) · Ĉ·(W_V·W_O)，避免显式计算完整 K, V 矩阵。
    5. **Decoupled RoPE temporal compression**：RoPE key 同样沿 temporal 维压缩，仅保留 position-specific key 用于 attention score 增强（每 s 个 token 共享同一 RoPE key 的最新值）。
    6. **Per-token KV cache**: 9d_h·l/(2s) = 144l elements（s=2），接近 MQA 水平（128l），远低于 MHA（1024l）和 MLA（288l）。

  - **kernel调度层**：扩展 FlashAttention-2，自定义 CUDA kernel 适配 temporal-compressed KV cache。Kernel 计算 scores = (X @ W_Q_absorbed) @ Ĉ^T / √d_h，以 Ĉ ∈ R^{t×r} 作为 attention 的"key"，避免显式 up-projection。Stride-aware mask 在 kernel 内联实现。tiling 策略可利用 s 倍的数据复用（每 s 个 query 对应同一 Ĉ 行）。内存访问量降低 2×（r=256 vs n_h·d_h=512）且 KV cache 加载量降为 1/s。

  - **Serving/框架层**：基于 Fairseq 实现，未修改 serving 调度。MTLA 作为一个 self-attention 模块替换即可使用。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → MTLA 设计映射：
  1. KV cache 内存爆炸（O(T)）→ Temporal compression（O(T/s)）：hyper-network 动态合并相邻 KV cache，s=2 时 per-token cache 从 288l（MLA）降为 144l，s=4 时降为 72l。实验结果：ST task GPU memory 从 18646 MiB（MHA）降至 2835 MiB（s=2）/ 1921 MiB（s=4），reduction factor 6.58×/9.71×。
  2. Per-token 解码 O(T) → O(T/s)：compressed cache 长度降为 1/s，per-token attention 计算量线性减少。ST inference time 从 281.3s（MHA）降至 65.6s（s=2, 4.29× speedup）/ 48.7s（s=4, 5.78× speedup）。
  3. Temporal 冗余未被利用 → Hyper-network 学习性 merge：不同于简单的 pooling 或 fixed window averaging，hyper-network 以输入序列 C 为条件动态生成 per-position merge weight w_i（Eq. 13, Sigmoid(gate)），使 merge 策略自适应于输入内容。结合 stride-aware causal mask 确保训练推理一致性，避免简单的 pre-downsampling（Fig. 2(b)）导致的 attention pattern 不匹配问题。

- baseline方法是什么？
  Baseline 是现有的线性序列建模方法（Linear Attention、State Space Models、Linear RNNs），它们都将整个输入序列压缩为单一固定大小的 memory state。典型代表包括：RetNet（恒定遗忘门 M_t = γM_{t-1} + k_t^T v_t）、GLA（数据依赖遗忘门）、HGRN2、Gated DeltaNet（delta rule update: M_t = (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t）、Mamba2 等。这些方法的共同特征是：所有 token 都更新同一个 memory state，在 recall-intensive 任务上表现远不如 Transformer。

  全栈执行例子（以 Gated DeltaNet baseline, 380M, T=2048, 32×A800）：

  - **算法层**：输入 X ∈ R^{2048×d}，所有 token 共享同一组 K/V projection W_k, W_v ∈ R^{d×d}。每个 token x_t 经 W_k, W_v 投影产生 k_t, v_t ∈ R^d，用 delta rule 更新单一 memory M_t ∈ R^{d×d}：M_t = a_t(I - k_t^T k_t)M_{t-1} + b_t k_t^T v_t。输出 o_t = q_t M_t。整个序列的信息被压缩到单个 R^{d×d} 矩阵中。d=1024 时 memory 容量为 1024×1024 = 1M 个标量。核心缺陷：(a) Memory interference — 新 token 的 k_t^T v_t 更新会覆盖 M 中先前存储的信息，即使有 forget gate a_t 也只做整体衰减无法精确隔离不同信息；(b) Limited capacity — 单一 memory 难以同时保存序列中多方面的信息（如专有名词、语义逻辑、问句结构等）；(c) 即使用增大 v 维度的方式扩展 memory（如 d_v → 2d_v），仍是"大杂烩"式的混合存储，缺乏结构化分离。

  - **kernel调度层**：使用 Triton kernel 进行 chunk-wise parallel scan 计算，每个 chunk 内并行处理后跨 chunk 传递 memory state。单 memory 场景下所有 token 顺序依赖，chunk 设计相对简单（无需 varlen 和 token reordering）。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. 单一固定大小 memory state 导致 memory interference — 新信息覆盖旧信息
  2. 有限 memory capacity — 无法同时存储序列中多面信息
  3. Gating mechanism（forget gate）只能整体衰减旧信息，无法选择性保留
  4. Recall-intensive 任务上与 Transformer 差距巨大（如 FDA 上 Gated DeltaNet 20.53 vs Transformer++ 46.14）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoM 的核心思想：用多个独立 memory state 替代单一 memory，通过 router 将不同 token 路由到不同 memory，实现"信息隔离存储"，类似人脑海马体 theta-gamma 振荡的多项目记忆编码机制。这与 gating mechanism 完全不同——gating 是"选择性遗忘"，MoM 是"选择性分离存储"。

  MoM 全栈执行例子（380M, T=2048, M=4 memories+1 shared, top-k=2, 32×A800）：

  - **算法层**：
    1. **Router**：x_t ∈ R^{1024} → W_g ∈ R^{1024×4} → softmax → TopK(k=2) → g_t ∈ R^2（归一化 importance scores）。Router 学习到不同类型 token 应路由到不同 memory——实验 Table 5 证实 Memory-1 偏好基础名词/动词/介词，Memory-2 偏好专有名词/科技术语，Memory-3 偏好技术术语/形容词，Memory-4 偏好疑问词/不完整名词。
    2. **Memory-specific projections**：每 memory m 有独立 W_k^m, W_v^m ∈ R^{1024×1024}。token 仅对 top-k 个激活 memory 计算 k_t^m, v_t^m，非激活 memory 保持 M_{t-1}^m 不变——从根本上避免了当前 token 对无关 memory 的干扰。
    3. **Memory update**：对每个激活 memory m，M_t^m = a_t^m (I - (k_t^m)^T k_t^m) M_{t-1}^m + b_t^m (k_t^m)^T v_t^m（Gated DeltaNet rule）。
    4. **Shared memory**：额外的一个 memory 始终被所有 token 激活，用于存储全局序列信息，弥补分离式 memory 可能丢失的跨 memory 长程依赖。
    5. **Memory mixing**：输出前先做 M̃_t = Σ_m g_t^{(m)} M_t^m 得到混合 memory，或等价地先逐 memory 计算 o_t^m = q_t M_t^m 再加权求和。
    6. **训练辅助**：auxiliary loss（类似 Switch Transformer）确保各 memory 负载均衡（Fig 5 热力图验证近乎均匀分布）。

  - **kernel调度层**：MoM 的硬件高效实现通过 Triton varlen kernel 实现。具体流程（Fig 2）：① tokens 按 routing 结果分组到各自 memory bucket；② 同 bucket tokens concat 为 varlen 序列 X̃；③ Triton kernel F_m 对每个 segment 独立并行计算（chunk-wise parallel scan，复用已有 linear model 算子）；④ 输出 o 返回各 bucket；⑤ 按原始 token 顺序拆分；⑥ weighted sum 恢复最终输出。复杂度：training O(n)（每个 memory 处理对应 sub-sequence，总计算量仍与总 token 数成线性），inference O(1)（每个 memory 维护固定大小的 state）。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → MoM 设计映射：
  1. Memory interference → 多 memory 隔离：不同 token 路由到不同 memory，非激活 memory 不被更新，token 间互不干扰
  2. Limited capacity → 多 memory 扩容：4 个 d×d 的 memory state = 4×1M = 4M 标量（vs baseline 1M），且 scaling 实验（Fig 6/8）证实 memory 从 1→8 持续提升
  3. Forget gate 的粗粒度衰减 → 稀疏激活 + 混合：不靠"遗忘"来管理内存冲突，而是通过 router 学习将不同信息分配到专门的 memory，结合 mixed memory 统一检索
  4. Recall-intensive 差距 → 结构化容量带来显著提升：380M MoM 在 Recall-intensive avg 上 28.16 vs Gated DeltaNet 24.78，1.3B MoM 达到 36.04 接近 Transformer++ 37.31
