## DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

- baseline方法是什么？
  Baseline 是静态/预定义稀疏注意力方法（以 MoA、StreamingLLM、H2O 为代表）以及 Full Attention（dense），其全栈执行例子如下：
  - **算法层**：MoA 为不同 layer 和 head 分配预定义稀疏模式（如 sliding window、global attention），但所有 attention map 使用固定模式集合，无法捕捉输入相关的异构 attention。StreamingLLM 保留初始 token（attention sink）和最近 token 的滑动窗口，丢弃中间 token，在超长序列中长距离检索精度急剧下降（LongEval >20K token 时精度降至 0.356）。H2O 通过累积 attention 分数识别 "heavy hitter" token 保留在 KV cache 中，但在 1B 模型上几乎立即退化。Full Attention 则对所有 token pair 计算 O(n²) attention，在 A100 40GB 上 LLaMA 3.2 3B 处理 4K token 即 OOM。
  - **系统框架层**：论文未明确说明。Baseline 方法通常在 HuggingFace Transformers 推理 pipeline 中修改 attention 层实现（替换 attention mask 或 KV cache eviction 逻辑）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention 通过 tiling 和 recompute 优化 dense attention 的 GPU memory access，但其性能在输入长度与 block size 对齐时出现非线性 scaling（如 8192 = 64×128 时吞吐量突增），无法保证稳定的长序列 scaling 行为。Static sparse attention (MoA/StreamingLLM/H2O) 的 mask 应用仍通过标准 attention kernel 完成，mask 模式固定不随输入变化。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DAM 通过捕捉真实 attention 分布中的结构模式并动态生成自适应 mask，解决了 static sparse attention 无法适应异构 attention 的问题，全栈执行例子如下：
  - **算法层（核心创新）**：
    (a) **Attention 模式观察与捕获**：LLaMA 3.2 3B 的 attention map 可视化表明，不同 layer 和 head 呈现不同的结构模式（sliding window、对角线倾斜、垂直条带），且随序列长度演变。DAM 通过冻结模型在 PCL（≤512）范围内提取完整 attention map，捕捉这些异构模式。
    (b) **Box-Cox 特征放大**：原始 attention 分布严重偏斜（少数连接主导，大量小值被淹没），Box-Cox 变换（λ=0.5）在保留大值的同时放大中小 attention 值，使隐藏的结构模式（对角线、垂直条带）可见。相比 square-root 变换（max≈150, std≈22），Box-Cox 产生紧凑、有界的值范围（max≈2.0, mean≈0.27, std≈0.35），便于阈值化。
    (c) **True Mask + 结构模式匹配**：通过阈值 τ=0.3 二值化得到 true mask，再与对角线模式（P_diag,r: j = i-r）和垂直模式（P_vert,c: j=c, i≥c）进行匹配（γ_k = overlap / pattern_size），阈值 μ=0.8 决定匹配。匹配的模式可直接外推至任意超 PCL 的长度。
    (d) **免微调**：整个 mask 生成过程仅需冻结模型的 attention map，不需任何模型权重更新或 fine-tuning。
    (e) **复杂度降低**：将 FLOPs 从 O(L²) 降至 O(sL)，s 为每 query 平均保留 key 数且 s ≪ L，同时保留长距离关键连接。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers 实现。Stage 1 离线运行，为每个 {layer, head} 生成 mask 文件；Stage 2 在推理时加载 mask，在 attention 层的 softmax 前应用 mask（mask 位置置为 -∞）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：DAM 当前使用标准 attention 计算（mask 通过 Hadamard product ⊙ 应用于 attention score）。论文明确指出 DAM 的稀疏布局与 tile-based GPU 执行兼容，未来可与 FlashAttention 等 memory-efficient kernel 融合，实现进一步的 kernel 级加速。当前效果：LLaMA 1B@4K 达到 941 tokens/sec（vs FlashAttention 633 tokens/sec），延迟降低 33%。
  - **硬件架构层**：论文未明确说明。

  **对比 baseline 的关键差异**：MoA 为不同 head/layer 分配预定义 mask 但无法适应输入变化，DAM 从真实 attention 分布中捕获结构模式并生成 mask——LongEval 上 DAM 平均精度 0.7966 接近 Full Attention 0.8011，而 MoA 在 >20K 时降至 0.394。StreamingLLM/H2O 通过 KV cache eviction 选择保留 token，但在 1B 模型/长序列下迅速退化，DAM 在 1B@33K 仍保持高精度。FlashAttention 优化 dense attention 的 memory access 但仍是 O(n²) FLOPs，DAM 通过结构化稀疏将 FLOPs 降至 O(sL) 且 scaling 可预测。DAM 唯一开销是 Stage 1 的离线 mask 生成（需在 Multi-News 上运行冻结模型提取 attention map），但这是一次性成本，Stage 2 推理无额外开销。

- baseline方法是什么？
  Baseline 是以 KIVI、KVQuant 为代表的逐标量 KV cache 量化方法，其全栈执行例子如下：
  - **算法层**：KIVI 采用非对称 per-channel 量化，对 Key cache 使用 per-channel 量化（channel-wise min/max），对 Value cache 使用 per-token 量化。量化公式：K_quant = round((K - zero) / scale)，反量化：K̂ = K_quant × scale + zero。每个标量独立处理，未利用向量内或向量间的结构信息。KVQuant 类似但使用非均匀量化（nuQ）和 grouped quantization。2-bit 量化时这些方法精度明显下降，1-bit 量化几乎不可用。
  - **系统框架层**：论文未明确说明。KV cache 的量化/反量化在 HuggingFace Transformers 框架中以在线方式执行——prefill 阶段生成 FP16 KV cache 后立即量化存储，decoding 阶段逐 token 反量化后参与 attention 计算。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：反量化操作在 GPU 上实现为逐元素的 scale+zero 乘加 kernel（KIVI）或查表 kernel（KVQuant），复杂度为 O(d N) per attention layer。但每个 decoding step 都需对全量 KV cache 执行反量化，无法融入 self-attention 的矩阵乘法中以实现计算复用。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CommVQ 将 KV cache 量化从"逐标量"提升为"逐向量"级别，并通过 RoPE-可交换码本将解码融合进 self-attention 计算，全栈执行例子如下：
  - **算法层（核心创新）**：
    (a) **向量级加法量化**：不再逐标量量化，而是将每个 token 的 key/value 向量 t_i ∈ R^d 整体编码为二进制序列 s_i ∈ {0,1}^{N_c}（Value）或量化索引 s_i^j ∈ {0,...,N_c'-1}^2（Key 的 2D 子空间）。解码通过码本矩阵乘法 t̂_i = s_i C 完成，使用加性而非标量量化，MSE loss 驱动端到端训练。
    (b) **RoPE-可交换码本**：利用 RoPE 矩阵的 2×2 块对角结构，设计 C_K^{jl} = [[x, y], [-y, x]] 形式的子码本，满足 R_i^j C_K^{jl} = C_K^{jl} R_i^j（Property 1）。这使得 key-query 计算 α_i = Σ_j,l (q^j R_t^j) C_K^{jlT} R_i^{jT} [s_i^j=l]^T 中，(q^j R_t^j) C_K^{jlT} 跨所有 token 仅需计算一次，解码开销从 O(2d N_c N) 降至与 self-attention 同量级。
    (c) **EM 算法训练**：在 FineWeb-Edu 校准集上用 EM 算法（含 soft clustering assignment + temperature annealing）优化子空间码本，避免死聚类中心。
    (d) **Value 解码重排**：将 Softmax(A) V 改写为 (Softmax(A) S_V) C_V，先做小矩阵乘再做码本乘，复杂度从 O(d N_c N + dN) 降至 O(N_c N + d N_c)。
  - **系统框架层**：基于 PyTorch + Triton 实现。Prefill 阶段正常计算 QKV 投影后将 K/V 输入编码器得到量化表示并存储；Decoding 阶段通过融合 kernel 加载压缩的 KV cache S 并结合码本 C 完成 attention 计算，替代标准的 FP16 KV cache 加载-计算流程。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层（Triton 实现）**：实现自定义 Triton kernel 将 key 解码融合进 attention score 的逐 token 计算，以及 value 解码的重排矩阵乘法。在 H100 上测试，优化实现相比 naive "decode-then-attention" 实现在 128K context 下获得 9.6× 加速（每层每 token 延迟从 36.6ms 降至 3.8ms）。CommVQ-1bit 使 LLaMA-3.1-8B 在 H100 上 120K context 仅需 20 GB 显存（FP16 需 60 GB），RTX 4090 上可运行 128K context 推理。
  - **硬件架构层**：论文未明确说明。

  **对比 baseline 的关键差异**：Baseline 的逐标量量化在 2-bit 和 1-bit 下信息损失严重（1-bit KIVI 在 LongBench 平均分仅 16.70 vs FP16 的 48.05），而 CommVQ 通过向量级加性量化 + 码本学习保留了更多 KV cache 结构信息（CommVQ-1bit 平均分 44.94）。Baseline 的解码开销与 self-attention 正交且独立（O(d N_c N)），而 CommVQ 通过 RoPE-可交换码本将解码融入 attention 计算，使解码开销可控（约 (R+1)/2 倍 self-attention 开销），且 R 一般在 11-21 之间。

- baseline方法是什么？
  Baseline 是现有 head-level KV cache 分配方法（以 HeadKV-R2 为代表），其执行流程为：
  - **算法层**：HeadKV-R2 独立评估每个 attention head 的 retrieval-reasoning 能力（通过分析 head 的 attention pattern 和检索功能），为每个 head 赋予独立的重要性分数，然后按分数比例分配 cache budget。评估时假设 head 的重要性与其个体检索和推理能力成正比，不依赖其他 head 的状态。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，在每层 prefill 完成后执行 cache eviction。每个 head 独立地根据其分配的 budget c_i，使用 SnapKV 的 local window attention pooling 机制选择保留 token。GQA 架构下，同一 group 内 4 个 query head 共享同一 KV cache，重要性分数取 group 内平均。
  - **执行层级**：对于 Llama-3-8B-Instruct（32 layers × 8 KV groups = 256 groups），推理时：input tokens 经 embedding → 逐层 Transformer block → 每层 prefill 计算 Q,K,V → 使用 FlashAttention + GQA → prefill 完毕后，对每 group 执行 cache eviction（保留 local window 8 tokens + top-c_i 高 attention 分 token）→ decode 阶段使用压缩后 cache。Baseline 的缺陷在于：(a) head 重要性评估是孤立的——每个 head 的好坏仅由自身判断，忽略了 head 之间的协同效应（例如两个"中等" head 组合可能比一个"高分" head 更重要）；(b) 评估是 task-agnostic 的——同一套 head 重要性分数应用于所有任务，无法反映不同任务对 head 的不同依赖模式。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CoKV 将 attention head 之间的协作建模为合作博弈（Cooperative Game），使用 Shapley Value 评估每个 head 在协作中的贡献，并提出 Sliced Shapley Value (SSV) 降低计算复杂度：
  - **算法层**：将每个 attention head 视为博弈玩家，定义效用函数 U(S) 为 coalition S 中 head 未被 mask、N\S 中 head 被 mask 时模型在验证集上的准确率。Shapley Value SV_i 衡量 head h_i 在所有可能 coalition 中的期望 marginal contribution。但精确计算 Shapley Value 需要枚举 2^n 个 coalition（#P-hard）。CoKV 利用 complementary contribution U(S) - U(N\S) 可同时更新 S 中所有 head 的估计值的性质，并提出 Sliced Shapley Value——观察到不同 coalition size j 下 head 的 expected complementary contribution 分布高度相关（对称于 n/2），因此只需在少量代表 size（H={32,64,96,128}）上采样即可。具体：每次迭代随机排列 heads，选 coalition size j∈H，构造 coalition S 包含前 j 个 heads，计算 u = U(S) - U(N\S)，并将 u 累加到 S 中所有 head 的 SV_{i,j} 估计中。250 samples/coalition size 时 MAE < 1/256，耗时 ~21h（8×3090）。得到 SSV 分数后，用 min-max 归一化并设 α 个最低分 head 的分数为 0（不分配额外 cache），其余按比例分配共享 budget B，最终 head h_i 的 cache size c_i = B·(NSV_i/ΣNSV_j) + s（local window）。
  - **系统框架层**：与 baseline 相同的 PyTorch/HuggingFace 推理 pipeline。区别在于：(a) prefill 前根据目标 task 加载对应的 SSV 分数表（每个 task 有独立的重要性分布）；(b) 每层每 group 的 cache budget c_i 由 SSV 决定的 NSV_i 按比例计算，而非 HeadKV 的 retrieval-reasoning 独立评分；(c) head 内 token selection 仍使用 SnapKV 的 local window attention pooling。CoKV 与 GQA 和 FlashAttention 完全兼容。
  - **执行层级**：对于 Llama-3-8B-Instruct（32 layers × 8 KV groups），推理前先按 task 加载预计算的 256 维 SSV 向量 → 计算 NSV 并确定各 group 的 c_i → 推理时：input tokens → embedding → 逐层 prefill（FlashAttention + GQA）→ 每层 prefill 后对该层的 8 groups 分别执行 eviction：local window Q 对所有前缀 K 计算 attention → max-mean pooling 得 token score → 保留 top-c_i 高 score token → decode 阶段使用压缩后 KV。CoKV 解决的 baseline 缺陷：(a) 协作评估——SSV 通过 complementary contribution（同时评估 coalition S 和 N\S）捕捉 head 之间的协同效应，能识别"单独重要但组合冗余"的 head 和"单独中等的组合关键"的 head；(b) 任务感知——SSV 在每个 task 的验证集上独立计算（不同 task 的 SSV 分布差异显著，但同 task 类型内泛化性好），推理时按用户所选 task 加载对应分数，实现任务定制化的 cache 分配。

- baseline方法是什么？
  Baseline 是标准的多模态 LLM 推理流程：Visual Encoder → Adapter → 全部 N⁰ 个 visual tokens 直接与 text tokens 拼接后送入 LLM，经过全部 L 层 Transformer 推理。对于视频任务（LLaVA-OV-7B），32 frames 产生数千 visual tokens（N⁰≈2304），每个 token 在每层都参与 Self-Attention 计算，总计 FLOPs 达 99.63 TB。这种 baseline 的缺陷在于：(a) 大量视觉 token 高度冗余，相似区域/相邻位置的 token 携带重复信息；(b) LLM 各层对 visual token 的需求不同——早期层做 cross-modal fusion 需要更多 visual token，后期层主要做 text reasoning 对 visual token 依赖很低。

  全栈执行例子（Baseline / LLaVA-OV-7B 标准推理）：
  - 算法pipeline：Visual tokens 全部保留，不做任何剪枝或合并，经过完整 28 层 Qwen2-7B Self-Attention + FFN
  - 系统框架：标准 HuggingFace Transformers 推理流程，LLaVA 框架（ViT→MLP Adapter→Qwen2 LLM→Text Decoder）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明具体 GPU 型号（FLOPs 由 LLM-Viewer 库估算）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 AIM 提出 training-free 两阶段 token 压缩——LLM 前做 Token Merging 消除相似 token 冗余，LLM 内部做 Token Pruning 逐步移除已完成 cross-modal fusion 后不再需要的 visual token。

  **对应解决 Baseline 缺陷的设计**：
  1. **Token Merging 解决冗余性**：基于余弦相似度迭代合并 LLM 输入前的视觉 token，每轮对最相似的 token pair 取均值。实验表明仅需 25% visual token 即可维持 video 性能。视频任务仅在帧内（spatial）合并，保留帧间时序信息。
  2. **Token Pruning 解决层间需求差异**：基于 PageRank 在 Self-Attention 图上计算 token 重要性，早期层（l < l₁）全保留以做 cross-modal fusion，中期层（l₁~l₂）线性递减 visual token 数量，后期层完全移除 visual token。这一分层策略源于消融实验发现：在第 8 层移除 visual token 导致性能骤降（58.0→41.9），而在第 22 层移除几乎无影响（58.0→58.1）。
  3. **仅剪枝 visual token、保留 text token**：消融实验表明剪枝 text token 导致 VideoMME 从 58.2 暴跌至 45.7，因为 LLM 本质是 text-centric 推理。
  4. **极低额外开销**：Token Merging + Pruning 的额外 FLOPs 仅 92.43 GFLOPs，占 Qwen2-7B 推理 FLOPs（14757 GFLOPs）的 0.6%。

  全栈执行例子（AIM）：
  - 算法pipeline：Two-stage——Stage 1: 在 LLM 输入侧将 N⁰ 个 visual tokens 按余弦相似度迭代合并（每轮对半合并），保留 r_merge=25%（video）/ r_merge=12.5%（image）；Stage 2: 在 LLM 的 l₁~l₂ 层间按 PageRank 分数线性递减 visual token 数，l > l₂ 层完全无 visual token；text token 全程不剪
  - 系统框架：即插即用集成到 LLaVA 推理流程中（LLaVA-OV-7B / LLaVA-1.5-7B），无需 fine-tuning，不与 FlashAttention 兼容（需显式 Attention 权重做 PageRank），但与量化、稀疏注意力兼容；支持自适应推理——通过调节 r_merge、l₁、l₂ 参数实现 2.5%~100% FLOPs 范围（40× FLOPs span，<13% 准确率损失）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明具体 GPU 型号

  **额外创新发现（对后续研究有指导意义）**：
  - 在同一 FLOPs budget 下用更多帧（如 192 frames vs 32 frames）可提升长视频理解性能（MLVU +4.6），因为去冗余后的 token 序列可以容纳更密集的时间采样
  - 跨帧（temporal）token merging 损害性能，而同帧内（spatial）merging 不影响——因为跨帧合并破坏时序信息
