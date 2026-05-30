## TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

- baseline方法是什么？
  Baseline 方法分为两类：(A) post-RoPE attention-based 方法（SnapKV, H2O, R-KV, LazyEviction）——使用最近 token 的 post-RoPE attention scores 估算 key 重要性，但 query 经 RoPE 旋转后朝向随位置变化，只有最近约 25 个 query 保持"当前"朝向，观察窗口极小，导致重要 key 未被检测到就被 evict；(B) post-RoPE norm-based 方法（VATP）——只使用 vector 范数而忽略方向信息，因为 post-RoPE 空间中方向与位置旋转纠缠，难以利用方向信号。两类方法的核心缺陷相同：都基于 post-RoPE 空间操作，受位置旋转限制。
  
  全栈执行例子（Qwen3-8B 使用 R-KV 进行 KV 压缩推理 on A100 80GB）：
  **算法pipeline**：R-KV 每 128 tokens 触发一次剪枝——收集最近 N 个 query 对所有 key 的 attention scores（post-RoPE QK^T），沿 query 维度聚合评分，结合 redundancy detection（hash similarity between adjacent tokens）标记冗余 token，保留 top-B 个非冗余 token。问题：(a) 最近 N 个 query 中大部分位置因 RoPE 旋转而方向过时，仅约 25 个有效——这对 retrieval head 特别致命，相关 token 可能沉寂数千步后才被需要；(b) 注意力分数在 25-query 窗口内缺乏统计稳健性，噪声主导选择。AIME25 上 R-KV 准确率仅 17.5%（Full Attention 40.8%）。
  **系统框架**：HuggingFace Transformers + FlashAttention-2，模型权重加载 Qwen3-8B，每次 decode step 计算 full attention（O(T)），每 128 步标记一次剪枝。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2 fused kernel，tiled QK^T + online softmax，在 A100 80GB 上 batch decode。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TriAttention 回到 pre-RoPE 空间，利用发现的 Q/K 浓度现象（pre-RoPE Q/K 围绕固定中心高浓度聚集，R ≈ 0.98，跨位置稳定）来解决 post-RoPE 的不稳定性：
  
  1. **绕过位置旋转限制**：pre-RoPE 向量不受 RoPE 旋转影响，Q 中心跨所有位置稳定——不再受限于仅 25 个有效 query 的观察窗口。校准数据中一次计算的 Q/K 中心可泛化到任意推理长度。
  
  2. **三角函数级数评分替代注意力观测**：当 Q/K 聚集时，attention logit 退化为仅依赖 Q-K 距离 Δ 的三角函数级数 Σ_f ‖q̄_f‖·‖k̄_f‖·cos(ω_f·Δ+φ̄_f)。用 Q 中心替代未来 query，直接用该级数对 key 打分 (S_trig)——无需观察实际 attention scores，避免了小窗口噪声。
  
  3. **范数信号补充分离方向与规模**：pre-RoPE 空间的方向和范数是分离的（浓度度量 R 捕获方向聚集度，范数独立变化），因此 S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖ 在浓度低的 head 中平滑补充范数信息——而 post-RoPE 中方向被旋转污染，难以利用。
  
  4. **自适应加权**：R_f 直接作为加权因子——R_f 高（浓度强，方向预测可靠）时 (1-R_f) 小，S_trig 主导；R_f 低时 Snorm 贡献更大。无需超参调节。
  
  AIME25 上 TriAttention 准确率 32.9%（R-KV 17.5%），几乎翻倍。
  
  全栈执行例子（TriAttention on Qwen3-8B，推理 on A100 80GB）：
  **算法pipeline**：(1) 离线校准：使用少量校准数据（50K tokens 即可，编码/聊天/HTML 均稳定）计算各 head 各频段的 E[q_f], E[k_f], E[‖q_f‖], R_f。(2) 推理阶段每 128 tokens：遍历 cache 中每个 key k，对其每个未来距离 δ∈{1,2,4,...,2^16} 计算 S_trig (k, Δ+δ) = Σ_f ‖E[q_f]‖·‖k_f‖·cos(ω_f(Δ+δ)+(arg(E[q_f])-arg(k_f)))，加上 S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖，平均所有 δ 得最终评分。(3) GQA 场景：per-head z-score normalize 后 max 聚合。(4) 保留 top-B，裁剪 KV cache。关键：S_trig 不依赖任何实际 attention 计算——只用到离线预计算的 E[q_f] 和 cache 中已有的 k_f。
  **系统框架**：vLLM plugin（triattention/vllm/runtime/integration_monkeypatch.py），自动发现激活——通过 monkeypatch scheduler 和 worker 注入剪枝逻辑。也支持 SGLang 集成和 MLX (Apple Silicon) 部署。论文中使用 HuggingFace Transformers + FlashAttention-2 进行评估。
  **kernel调度**：FlashAttention-2 标准 fused attention kernel，TriAttention 不修改 kernel 层——剪枝操作在 attention 计算前进行，仅减少输入的 KV 数量，不改变 attention 本身的计算图。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

## Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

- baseline方法是什么？
  Baseline 是标准 full attention（O(n²) 全对全注意力计算），以及三类高效注意力方法：(1) **结构化稀疏**（Longformer, BigBird）：使用固定位置模式（局部窗口+block/global pattern），内容无关，retrofit 到预训练模型时丢失长程依赖；(2) **近似方法**（Performer, Linformer）：用 kernel 近似或低秩投影替代 softmax 注意力矩阵，近似误差逐层累积，retrofit 时 PPL 退化惨重（Performer +75.6 PPL）；(3) **Token 选择方法**（SparQ, MagicPIG）：选择 top-k 最相关 token 但 PPL 退化 5-10 点。三者的共同缺陷是**不能学习哪些 token pair 真正需要互相关注**——固定模式缺乏内容感知，近似方法损失信息，token 选择方法缺少全局分组结构。

  全栈执行例子（GPT-2 124M 推理，full attention + FlashAttention on H100-80GB）：
  **算法pipeline**：对于长度为 T 的序列，每层计算 Q,K,V ∈ R^{T×d}，通过 softmax(QK^T/√d) 计算全部 T² 个 token pair 的注意力分数，复杂度 O(T²)。每个 token 的 softmax 概率分布在全部 T 个 token 上，导致：(a) softmax 稀释——一个代词要与其先行词竞争注意力权重，必须与数百个无关 token 共享概率质量；(b) 噪声累积——无关 KV pair 对注意力输出贡献微小噪声，12 层 × 12 heads 累积后显著降质。
  **系统框架**：HuggingFace Transformers / PyTorch 加载 GPT-2 124M 权重，使用 FlashAttention fused kernel 完成 attention 计算。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention 做 tiled QK^T + online softmax + V 加权，O(T²d) 计算 + O(T²) 中间结果，在 H100-80GB 上 T=1M 时约 1.5s。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 Focus —— 添加少量 learnable centroid 向量（K 个 centroid，dg=16，仅 148K 参数）到每层注意力中，centroid 决定哪些 token pair 可以互相关注（routing），原 QKV 注意力决定关注多少（content）。核心设计：(1) **可学习 centroid + Sinkhorn 归一化**：通过投影 W_g 将 token 映射到 centroid 空间，Sinkhorn 迭代强制双随机均衡分组，阻止 group dominance（类似 MoE 中的 expert collapse）；(2) **门控注意力**：s_ij = q_i^T k_j · (1_local + (1-1_local)·σ(λ·g_i^T g_j))，局部窗口内全注意力，远距离仅同组 pair 保留；(3) **分离 routing 与 attention**：centroid 仅控制谁关注谁，内容流经预训练 QKV 不变——这是 composability 的关键，原始权重完全冻结；(4) **推理时 FlashAttention 分解**：将稀疏 mask 分解为两个不相交 FA 调用（same-group causal + cross-group local），logsumexp 精确合并，8.6× 加速无自定义 kernel。

  全栈执行例子（Focus on GPT-2 124M，推理 on H100-80GB）：
  **算法pipeline**：对长度为 T 的序列，每层首先计算 g = sinkhorn(W_g·h^T · C / τ, N=10) 得到每个 token 的 group assignment。局部窗口 w=128 内 token 全注意力；远距离 token 仅当 g_i^T g_j ≈ 1（同组）时参与 softmax。结果是 softmax 概率质量集中在较小但更相关的 token 子集上：(a) 消除 softmax 稀释——同组内竞争 token 更少且语义相关；(b) 消除噪声——无关跨组 pair 不参与注意力计算（而非被缩放到近零值）。效果：124M 上 PPL 30.3 vs full attention 31.4（稀疏超越密集），所有 benchmark 零退化。
  **系统框架**：HuggingFace Transformers + PyTorch，加载 GPT-2 权重、添加 centroid 参数（148K），仅 centroid 训练（4000 steps on PG-19），原权重冻结。
  **编译框架**：论文未明确说明。
  **kernel调度**：推理时 token 按 group 做 stable sort，reshape 为 K 个独立序列，对每个调用 flash_attn_func(causal=True)；同时计算 cross-group local 窗口注意力；两个输出通过 logsumexp merge 精确合并。A ∩ B = ∅（无重复计数），A ∪ B = 全部应关注的 pair。O(T²/K) + O(Tw)，T=1M, K=8 时 8.6× 加速。Sort overhead ~12ms 常数，长序列下可忽略。320 行 Python，无自定义 CUDA kernel。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

## TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

- baseline方法是什么？
  Baseline 是现有 within-LLM token 剪枝方法（FastV、PDrop、SparseVLM 等），它们都依赖 **attention-based** 或 **similarity-based** 准则来评估 visual token 重要性。具体缺陷：
  (1) **Attention 存在位置偏差（Positional Bias）**：attention 机制对序列开头和结尾的 token 分配更高分数（源于 causal attention mask 的三角结构），而这些位置的图像 token 往往语义信息较少；
  (2) **Attention 过度关注视觉显著但语义无关的区域**：attention 可能被高对比度或视觉突出的区域吸引，而这些区域与用户指令无关；
  (3) **Similarity-based 方法是任务无关的（Task-Agnostic）**：基于 token 表征相似度合并 token（如 VisionZip）无法区分哪些 token 对特定下游任务真正重要。

  全栈执行例子（LLaVA-v1.5-7B + FastV on A100 GPU）：
  **算法pipeline**：LLaVA 使用 CLIP-ViT-L/14 编码 336×336 图像 → 576 个 visual tokens。每个 visual token 经 projector（两层 MLP）映射后与 instruction tokens 拼接，送入 LLaMA-7B（32 层 Transformer）。FastV 在 layer 2 之后根据最后一个 token 对 visual tokens 的 attention scores 排序，剪除低 attention 的 K% tokens，后续层仅处理保留的 tokens。由于 attention 计算受 causal mask 影响，位置靠前和靠后的 visual tokens 系统性获得更高 attention scores，导致中间位置的重要语义 token 被错误剪除。
  **系统框架**：HuggingFace Transformers 加载 LLaVA 模型，推理使用 FlashAttention kernel。token pruning 通过在前向传播中动态缩减 key-value cache 实现。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention 实现 memory-efficient exact attention，通过 tiling 减少 HBM 访问。Pruning 后序列长度减少，后续层 attention 计算量等比例下降。
  **硬件架构**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TransPrune 提出基于 **token transition**（token 在模型各模块中的表征变化）的新视角来评估 token 重要性，结合 TTV 和 IGA 两种互补准则：
  
  (1) **TTV（Token Transition Variation）解决 attention 位置偏差问题**：TTV 仅测量每个 token 自身的输入→输出表征变化（L2 norm 比率 + cosine similarity），不计算 token 间依赖关系，因此天然避免了 attention 三角 mask 引入的位置偏差。论文实验（Figure 4）证实 TTV 的保留 token 位置分布更均匀地集中在图像中央语义丰富区域，而 IGA（attention-based）明显偏向序列首尾。
  
  (2) **IGA（Instruction-Guided Attention）解决任务无关性问题**：通过计算 instruction tokens → image tokens 的 attention，引入任务相关的语义监督，使剪枝对用户指令敏感。仅计算 instruction→image 的单向 attention（非完整 attention map），计算开销极小。
  
  (3) **Accumulation 机制解决 TTV 层间不稳定性**：TTV 模式在浅层、中层、深层各不相同，仅中层（layers 7-12）的 transition 最能反映语义重要性（因为中层在全局特征和局部细节之间整合信息）。通过跨层累积 TTV 值，每个 pruning layer 的决策基于 token 的完整"transition 历史"，避免单层 TTV 噪声导致的错误剪枝。

  全栈执行例子（TransPrune on LLaVA-v1.5-7B on A100 GPU, FlashAttention）：
  **算法pipeline**：LLaVA 编码相同 576 visual tokens + instructions → LLaMA-7B 32 层。在每层 Transformer 中：对于 self-attention 模块，输入 token 表征 T_in → T_out（经 QKV projection + FlashAttention），对每个 visual token 计算 TTV_attn = Softmax(1-|cos(T_out,T_in)|) · (||T_out||₂/||T_in||₂)；对于 FFN 模块，输入 T'_in → T'_out，计算 TTV_ffn = Softmax(1-|cos(T'_out,T'_in)|) · (||T'_out||₂/||T'_in||₂)；该层 TTV = TTV_attn + TTV_ffn。在 pruning layers (7, 9, 12)，累积从 layer 7 到当前层的所有 TTV → TTV_acc；同时用 layer l+1 的 attention 权重 Q_inst @ K_img^T 计算 IGA（mean over instruction tokens）。最终 Score = 0.5·TTV_acc + 0.5·IGA，按 Score 排序保留 top-K tokens。TTV 累积计算额外开销约为 O(sd)，FLOPs 占比可忽略。
  **系统框架**：HuggingFace Transformers + FlashAttention。TTV 计算仅需 hook 模块的输入和输出 tensor，IGA 仅计算 instruction→image 的单向 attention（无需完整 N×N attention map），因此与 FlashAttention 的内存优化完全兼容。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention kernel 不变。TTV 计算（L2 norm、cosine similarity）使用标准 PyTorch 算子，无需 custom kernel。
  **硬件架构**：论文未明确说明。

- baseline方法是什么？
  Baseline 是 **GQA（Group-Query Attention）**：h 个 query head 分为 g 组，每组共享同一个 KV head。每个 key/value head 独立计算 RoPE 位置编码。GQA 通过共享 KV head 减少了 KV cache 大小（比例 g/h），但：
  (1) **表达能力受限**：GQA 仅能产生 g 个独立 key/value 向量，每个 query head 只能与其所属 group 的 KV head 交互，跨 group 信息无法流动；
  (2) **RoPE 阻止进一步压缩**：每个 key head 都携带 RoPE，使得 key 无法被吸收（Absorb）到 query projection 中——如果尝试吸收，RoPE 的位置相关旋转会破坏矩阵乘法的结合性；
  (3) **转换成本高**：已有大量投资优化 GQA 模型（LLaMA、Qwen 等），从头训练 MLA 模型成本极高。

  全栈执行例子（LLaMA-2-7B GQA on vLLM GPU serving）：
  **算法pipeline**：h=32 query heads, g=8 KV groups, d=128。每个 token：x_t → W^Q x_t → [q_1...q_32]；x_t → W^K x_t → [k_1...k_8]；x_t → W^V x_t → [v_1...v_8]。对每个 query head i（属于 group j=⌈i/4⌉）：q_i 先做 RoPE，k_j 先做 RoPE，然后 attention score = q_i^R · k_j^R / √d。KV cache 存储 2×8×128=2048 维/token/layer。推理时每步从 cache 加载全部 KV。
  **Serving调度**：vLLM PagedAttention + continuous batching，GQA 模式下每个 block 存储 g 个 KV head。长上下文下 KV cache 内存带宽成为瓶颈。
  **kernel调度**：FlashAttention-2/3，标准 GQA kernel（query heads 分组共享 KV）。
  **编译框架/硬件架构/芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **TransMLA** 通过三个关键技术将 GQA 模型等价转换为 MLA 模型，不改变 attention 输出（training-free 转换），仅需少量 fine-tuning 恢复性能：

  **对应 Baseline 缺陷 1（表达能力受限）**：论文在 Appendix A 中理论证明 MLA > GQA 的表达能力——给定相同 KV cache 大小，GQA 仅能表达 MLA 的一个稀疏子集（W^{UK}/W^{UV} 必须是 block-selector 矩阵），而 MLA 的 dense W^{UK}/W^{UV} 允许跨 head 混合信息，产生更丰富的交互模式。转换步骤：首先将 g 个 KV head 合并为一个 latent head（引入 W_i^{UK} 作为 selector 矩阵，保持等价性），然后通过 BKV-PCA 将 [K_nope; V] 压缩到 r_kv 维 latent 空间。

  **对应 Baseline 缺陷 2（RoPE 阻止 Absorb 操作）**：提出 RoRoPE——利用正交旋转 U_l 在 RoPE 内积下的不变性（Theorem/Equation 19），将各 head 中相同 RoPE 频率维度的 key 分量做 PCA 旋转，使位置信息集中到第一个 head（K_rope），其余 head（K_nope）的位置信息可忽略，移除其 RoPE。K_nope 无 RoPE 后，W^{UK}（RoPE-free part）可直接吸收到 query projection 中（Absorb 操作），实现 MLA 推理范式。

  **对应 Baseline 缺陷 3（转换成本高）**：提供了无缝转换流程——merge KV heads → RoRoPE → FreqFold → BKV-PCA → 轻量 fine-tuning。训练仅需 300M-6B tokens（相比原始预训练的 1T-2T tokens），2 小时训练即可超越 MHA2MLA 6B tokens 的性能。

  **对应额外痛点（key/value norm 不平衡导致 PCA 偏差）**：发现 K_nope 的 ℓ₂-norm 远大于 V，直接联合 PCA 会导致主成分完全由 key 主导。BKV 解法：计算 α 缩放因子使两者 norm 对齐后再 PCA，显著提升压缩质量（Figure 4）。

  全栈执行例子（TransMLA 转换后的 LLaMA-2-7B on vLLM GPU serving）：
  **算法pipeline**：输入 token x_t → W^{DKV'} x_t → c_t^{KV} ∈ R^{r_kv}（低秩 latent，例如 r_kv=144 = 92.97% 压缩）。K_rope head：单独保留 d 维 RoPE key（不参与压缩）。推理时 Absorb 操作：q̂_{t,i} = [(W_i^{UK})^T q_{t,i}^C; q_{t,i}^R]（W_i^{UK} 吸收进 query），k̂_t = [c_t^{KV}; k_t^R]（共享 latent key），所有 head 共享一个 KV cache（类似 MQA 模式），仅需缓存 c_t^{KV}（r_kv 维而非 2gd 维）。
  **Serving调度**：vLLM + DeepSeek MLA kernel。转换后模型直接兼容 DeepSeek 代码库（vLLM/SGlang/FlashMLA），无需修改推理框架。长上下文下 KV cache 大幅减小 → 更多 batch 可同时驻留 GPU 显存 → 吞吐量提升。8K context 时相对原始 GQA 模型最大 10.6× 加速（165.2 TFLOPS GPU）；16K context 时原始模型 OOM 但 TransMLA 仍可运行 414.41 tok/s。
  **kernel调度**：DeepSeek FlashMLA kernel，利用低秩 latent KV 减少全局内存访问。论文未自定义 kernel，直接复用 DeepSeek 生态的优化。
  **编译框架/硬件架构/芯片设计**：论文未明确说明。

## The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

- baseline方法是什么？
  Baseline 是**标准密集 attention（dense attention）**：所有 query-key 对参与完整的 scaled dot-product attention 计算。Prefilling 阶段计算完整下三角 attention 矩阵（O(n²) FLOPs），decoding 阶段每步从内存加载全部 KV cache（O(n) memory transfers per step）。典型部署中，长序列下 attention 成本占主导——128K tokens 时 prefilling 中 attention 占 80% FLOPs，batch size 64 解码时 KV cache 加载占 80-97% memory。

  全栈执行例子（dense attention 在 vLLM H100 serving）：
  **算法pipeline**：标准 Transformer self-attention，QKV 投影后执行完整 FlashAttention-2 kernel，所有 query 对所有 key 计算点积 → softmax → 加权求和。
  **Serving调度**：vLLM continuous batching + PagedAttention，所有请求的 KV cache 全量存储和加载，无选择性加载。长上下文（128K）下 TTFT 由 O(n²) prefill FLOPs 主导，TPOT 由 KV cache 内存带宽主导。
  **kernel调度**：FlashAttention-2/3 kernel，完整 dense attention 的前向传播。
  **编译框架/硬件架构/芯片设计**：论文未明确说明（纯软件方案，硬件无关）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是对 **training-free 稀疏注意力方法的系统实证分析**，而非提出单一新方法。核心贡献是从四个设计轴（sparsification unit、importance estimation、budget allocation、KV cache management）对六种代表性方法进行统一归类和 harmonized 实现，然后在最大规模上（3 个模型家族、4-72B 参数、16K-128K 序列、0-0.95 稀疏度、9 个任务、7065 配置）系统回答三个基本问题。

  **对应 Baseline 缺陷的解决路径**：
  1. **Baseline 缺陷：dense attention 的成本随序列长度平方增长，无选择性计算**。论文通过分类学证明，不同的稀疏策略适用于不同推理阶段和任务类型——prefill 阶段用 Vertical-Slash（细粒度 token 选择，适合检索任务）或 Block-Sparse（块级选择，适合推理/聚合任务）；decode 阶段用 Quest（token-to-page 选择，通用性最佳，0.95 稀疏度仍优于小 dense 模型）。
  2. **Baseline 缺陷：production 中固定 sparse budget 未考虑序列长度效应**。论文发现更长的序列容忍更高稀疏度——64K 时 1/20 budget 的相对误差（0.20）低于 16K 时（0.33）。最优 token budget 应**次线性增长**（sublinear scaling），而非固定或线性增长。
  3. **Baseline 缺陷：评估不足，缺乏对方法选择的实践指导**。论文建立了 per-task 方法推荐——fine-grained token selection 用于检索、chunk-based 用于推理、page-based decoding 作为通用解码方案。

  全栈执行例子（推荐组合：Vertical-Slash prefill + Quest decode on vLLM H100）：
  **算法pipeline**：prefill 阶段——仅对 Q_recent（近似窗口 256/512 tokens）× K_full 的 attention 分数做重要性估计，选出 top-k verticals（全局共享列）+ slashes（对角线），在所选 QK 子集上做精确 FlashAttention。Decode 阶段——每步仅 1 query，对 page-min/max key 做近似相似度计算选 top-k pages，仅加载所选 page 内 KV 做精确 attention（保持全 KV cache 不 eviction）。
  **Serving调度**：vLLM PagedAttention + AbstractAttention 拦截层。Prefill: 重要性估计（Vertical-Slash indexing FLOPs 含 Q×K 近似 + sorting + block 选择，公式见 Eq.9）→ 稀疏 attention 执行。Decode: Quest indexing（仅加载 page 级 min/max key 表示，memory overhead 极小，公式见 Eq.10）→ 精确 attention on selected pages。
  **kernel调度**：FlashAttention-2 block-sparse 模式执行所选 QK block 的 attention。论文未自定义 kernel，使用 vLLM 原生 kernel。
  **编译框架/硬件架构/芯片设计**：论文未明确说明。

- baseline方法是什么？
  Baseline 是现有 KV Cache 压缩方法的组合代表：**eviction 方法** (StreamingLLM/SnapKV) 永久丢弃 KV cache token，**selection/offloading 方法** (Quest/PQCache) 虽保留全部 KV cache 但受限于 GPU 内存或 PCIe 带宽瓶颈，**quantization 方法** (KIVI) 对所有层统一量化但在低精度下性能严重退化。

  全栈执行例子（以 PQCache 为典型 baseline）：
  **算法pipeline**：对所有 Transformer 层统一使用 Product Quantization (PQ) 压缩 KV cache——将 key/value 向量空间划分为 2 个 partition，每个 partition 用 6-bit PQ codes 编码。prefilling 阶段 KV cache 存储为 PQ codes + codebooks；decoding 阶段使用 K-Means 聚类在 CPU 端对 compressed key 做检索选出近似 Top-K tokens，然后从 CPU gather 完整 KV 传输到 GPU 做精确 attention。所有层统一对待，无层间差异化策略。
  **Serving调度**：基于 HuggingFace Transformers 的标准推理管线，CPU-GPU 协同执行。prefill 阶段 KV cache 存 GPU；decode 阶段每层从 CPU 计算近似 attention → gather Top-K tokens → CPU→GPU 传输 → GPU attention。K-Means 聚类在 CPU 做，但长序列下 clustering overhead 随时间增长（128k+ 需限制为 1 次迭代）。论文未修改标准 serving 框架。
  **kernel调度**：使用 FlashAttention-2 做 GPU 端 attention，CPU 端使用 K-Means 聚类做近似检索。论文未明确说明自定义 kernel。
  **编译框架/硬件架构/芯片设计**：论文未明确说明（纯软件方案）。

  Baseline 缺陷：(1) **统一策略忽视层间差异**：所有层应用相同压缩方法，但浅层（dense attention）需要保留全局信息、量化更合适，深层（sparse attention）仅有少量 dominant tokens 是关键、稀疏选择更合适。KIVI 1-bit 量化使性能从 F1=91.6 暴跌至 18.6（TriviaQA，Table 1）；(2) **CPU-GPU 通信瓶颈**：PCIe 带宽远低于 GPU 计算吞吐——单层 KV (~8GB) 通过 PCIe 1.0 (4GB/s) 传输需 ~2s，而 GPU attention 计算仅 ~10ms。OffloadCache 和 PQCache 需要传输较多 token 数据（~20%），latency 由 I/O 主导；(3) **CPU 端计算延迟**：PQCache 的 K-Means clustering 在 CPU 上执行，长序列下 clustering overhead 显著增长；(4) **检索精度 vs token 数量的 trade-off**：为覆盖关键信息需传输更多 token，但传输更多又加剧延迟。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TailorKV 提出 layer-specific 混合压缩框架，核心洞察：**不同层有不同的 compression preference**——浅层注意力密集适合量化，深层注意力稀疏适合动态检索。通过离线层分类 + 在线混合执行，实现极致压缩同时保持近无损精度。

  **算法pipeline**（核心创新）：

  (1) **Offline Identification（层分类）→ 解决统一策略忽视层差异**：定义 dense preference score P，使用最近 n_q 个 query 和全部 key 计算 attention，取 Top-k attention scores 的和取补数作为"密集度"指标。P 值高的层（浅层 0，有时含层 1）注意力分布均匀→ quantization-friendly，低的层（深层）注意力集中在少量 token → sparsity-friendly。该 metric 跨数据集一致（Appendix C, Figure 12），离线一次计算即可。实验：仅量化 layer 0 (Q={0}) 时 1-bit 性能从 F1=18.6 恢复到 F1=92.1（TriviaQA，Table 1），量化更深层反而损害性能。

  (2) **Static Quantization for Quantization-Friendly Layers → 解决量化性能退化**：对 quantization-friendly 层使用 1-bit 或 2-bit 静态量化（per-channel key + per-token value）。这些层本身注意力密集均匀，对量化误差不敏感。配合 FP16×INT1 GEMV kernel 保持硬件效率。与 KIVI 等面向所有层量化的方案不同，TailorKV 只在"适合量化"的层量化，使 1-bit 极低精度成为可能。

  (3) **Dynamic Retrieval for Sparsity-Friendly Layers → 解决通信瓶颈和检索精度 trade-off**：识别 query/key 中 outlier channels 与 attention score 的相关性（Figure 2），利用 inter-layer similarity（cosine similarity of hidden states, 附录 B Figure 11）在当前层预估算下一层 query，选出 critical channels (d_s=8~12)，仅预取 critical key cache 到 GPU，在 GPU 上近似 attention scores 后精准选出 Top-K tokens (1%~3% of total)。只传输极少量关键 token，大幅降低 PCIe 通信量。

  **kernel调度/系统设计**：

  (4) **异步 Pipeline + Double Buffering → 隐藏 CPU-GPU 通信延迟**：layer l-1 计算时异步预取 layer l 的 critical key cache，使用读写双缓冲区实现 computation 与 communication 的 overlap。唯一不可 overlap 的步骤是 Top-K token 的 fetch（依赖当前层 query 确定哪些 token）。Figure 5 时间线显示 decode 流程高度并行化。

  (5) **DGL 直接行传输 → 避免 CPU gather 开销**：使用 DGL 从 CPU tensor 直接按行索引传输到 GPU，避免 PQCache 的"先在 CPU gather 成连续内存再传输"两步操作。相比 PQCache：retrieval latency 降 27.8%~40.5%，data transfer latency 降 82.2%~83.5%（Figure 8）。

  **Serving调度**：基于 HuggingFace Transformers 4.46.1 修改推理管线。prefill 阶段逐层 offload KV cache 到 CPU（sparsity-friendly 层）或 GPU 上量化存储（quantization-friendly 层）。decode 阶段对每层根据类型分支执行静态量化 attention 或动态检索 attention。多线程实现异步任务执行。

  **编译框架/硬件架构/芯片设计**：论文未明确说明。

  效果：Llama-3.1-8B 128k context 在单 RTX 3090 (24GB) 上以 82ms/token 解码，peak GPU memory 降低 53.7%（结合 AWQ 4-bit weight quantization），相比 Full Cache A100 上降低 73.8% memory。LongBench 上性能近无损（TailorKV-1: 52.6 vs Full Cache: 53.8, Llama-3.1-8B）。

## StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

- baseline方法是什么？
  Baseline 是 **ReKV**（Di et al., 2025, ICLR），第一个为 Streaming Video QA 引入 KV-cache 检索机制的 Online Video-LLM。ReKV 全栈执行过程：

  **算法pipeline**：将视频流按固定帧数均匀切分为 uniform segments，每段编码后存储完整 KV cache（不做压缩）。收到用户问题时，基于问题 query vector 对所有历史存储的 KV caches 做 similarity-based 检索，选出 query-relevant 的 KV blocks 送入 LLM 生成答案。检索策略为 uniform allocation：每层分配相同数量的 KV blocks。

  **Serving调度**：基于 LLaVA-OneVision-Qwen2-7B-OV，NVIDIA H20 GPU (96GB)，0.5 FPS 处理视频流，local window = 15K tokens。均匀分段 → 顺序编码 → 全部 KV cache 存入 KV Bank → 问题到来时检索 → 生成答案。

  **kernel调度/编译框架/硬件架构/芯片设计**：论文未明确说明（使用标准 PyTorch + HuggingFace Transformers 推理）。

  Baseline 缺陷：(1) **均匀分段打断语义连续性**：固定帧数切分无视视频内容的语义边界，可能在关键事件中间切断，破坏语义信息完整性；(2) **存储全部 KV cache 导致显存爆炸**：长视频下累积存储所有历史 KV caches，无压缩机制，显存随视频时长线性增长；(3) **检索策略僵化**：uniform per-layer allocation 未考虑不同 transformer 层信息分布的差异性，低效利用检索预算；(4) **检索精度不足**：需要检索更多帧才能确保相关信息被包含（Figure 4），引入噪声信息反而降低 QA 准确率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  StreamKV 提出 training-free 框架，通过四个核心设计解决 ReKV 的缺陷：

  **算法pipeline**（全栈改进）：

  (1) **语义分段划分 → 解决均匀分段破坏语义连续性**：基于相邻帧 ViT embedding 的 cosine similarity 检测语义边界（s_t = cos_sim(f_{t-1}, f_t), threshold=0.99），配合 exclusion window (m=4) 和 segment merging (M=64) 保证段长合理。使分段尊重视频内容的自然语义结构，避免关键信息被切分。每个段计算 summary vector（空间位置平均）保留 segment-level information。Table 2 消融显示语义分段在所有压缩率下均优于均匀分段（如 50% 压缩率: 59.07% vs 57.32%）。

  (2) **Guidance Prompt 驱动的 KV 压缩 → 解决显存爆炸**：每段编码后立即应用压缩（非解码阶段离线压缩），引入 guidance prompt 捕获段内关键语义元素——salient entities（人/物体/场景）、key events/actions（发生了什么）、temporal/causal relationships（事件时序因果链）、contextual cues（场景切换/对话/叙事变化）、factual details（计数/摘要等）。以 guidance prompt 的平均 query vector 作为 selection criterion，选出每段最 informative 的 KV blocks 保留，压缩率 0%-90% 可调。实验显示 60% 压缩率下 StreamKV Overall 58.9% vs ReKV (无压缩) 53.5%，90% 极端压缩下仍保持 56.7%。

  (3) **Unified Layer-Adaptive KV Selection Module → 解决检索策略僵化**：将压缩和检索统一为同一模块。每层计算 softmax-normalized 相似度分布，通过 binary search 确定全局 cumulative score threshold p，使每层预算 K_l 与该层信息集中度成正比——信息越集中的层获得越多预算。Table 4 消融：Ada.+Ada.（压缩和检索均自适应）在 50% 压缩率下准确率 59.07%，优于 Uni.+Uni. 的 58.12%，单独对压缩或检索使用自适应也优于全 uniform。

  (4) **Precise Retrieval Strategy → 解决检索精度不足**：基于 question vector 作为 selection criterion 的层自适应检索。Figure 4 显示 StreamKV 仅需检索 8 帧即达最优准确率，检索更多帧反而因引入不相关噪声导致性能下降（"Lost in the Middle" 效应），与 ReKV 需要检索更多帧才能覆盖相关信息的趋势完全相反。这证明了 StreamKV 检索的高精度。

  **Serving调度**：基于 LLaVA-OneVision-Qwen2-7B-OV，NVIDIA H20 GPU (96GB)，FP16。0.5 FPS 处理帧率，local window = 15K。语义分段 → sliding-window encoding（含 summary vector）→ 即时 KV 压缩 → 存入 KV Bank。收到问题 → KV 检索 → 生成答案。Figure 1 显示 StreamKV 在准确率、显存、延迟三个维度均优于 ReKV，显存约为 ReKV 的 50-60%（60% 压缩率下）。

  **RoPE 策略**：encoding 阶段 RoPE 仅应用于 local window 内（inspired by LM-Infinite），QA 阶段基于 relative positions 应用 RoPE，缓解长序列下 RoPE 远距离 attention 衰减问题。

  **kernel调度/编译框架/硬件架构/芯片设计**：论文未明确说明。

## Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

- baseline方法是什么？
  Baseline 是 **Ring Attention**（Liu et al., 2024a），一种分布式全局注意力机制。它将输入序列分块后在各 host 上以 ring 模式循环传递 KV cache，每个 host 对当前持有的 KV cache block 计算 local attention 后传递给下一 host，最终聚合为 global attention。Baseline 全栈执行过程：

  **算法pipeline**：输入 c = [c1, c2, ..., cn]。Prefill 阶段：H 个 hosts 以 ring 拓扑互相传递 KV cache blocks，每个 host 依次接收前一 host 的 KV → 对当前 query 计算 local attention → 传递 KV 到下一 host → 循环直至每个 host 都见过所有 blocks。每个 block 的 attention 为 global self-attention：A_i = softmax(QK_i^T/sqrt(d)) V_i。Decode 阶段：每步生成同样需要 ring communication 传递 KV cache，query 与所有前序 tokens 做 global attention。

  **Serving调度**：基于 HuggingFace Transformers 和 TRT-LLM。多 GPU（8-32 A100）以 ring 拓扑连接。Prefill: KV cache 在 H 个 GPU 之间循环传递 H 次（每层 × H 轮通信）。Decode: 每个 token 生成时同样需要 H 轮 KV cache 传递（每层每 token）。通信量 O(L×d) per layer，延迟随 host 数线性增长。

  **kernel调度**：使用 Flash Attention（Dao, 2024）作为每个 local block attention 的 kernel，通过 blockwise IO-aware tiling 避免完整 attention matrix 显存驻留。

  **编译框架/硬件架构/芯片设计**：论文未明确说明。

  Baseline 缺陷：(1) Ring Attention 在 prefilling 和 decoding 阶段都需 full quadratic attention 计算，O(L^2) 复杂度；(2) ring communication 要求 KV cache 在 H 个 hosts 间顺序传递，通信延迟与 host 数和序列长度成正比，成为长序列推理的瓶颈；(3) 长序列下 vanilla 自回归生成遇到 OOM（>64K tokens on 8×A100）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Star Attention 将 attention 分为两阶段：阶段一 context encoding 用 blockwise-local attention + anchor block（无跨 host 通信），阶段二 query encoding 用 distributed global attention（仅传递 scalar + vector per token）。全栈执行过程：

  **算法pipeline**（对应解决 O(L^2) 复杂度和通信问题）：

  *阶段一：Context Encoding — blockwise-local attention with anchor block*
  - 解决 Ring Attention 的 prefilling quadratic 开销：将 context 切为 n blocks，每 block 前缀拼接 anchor block c1 → 对 2b-token augmented block 做 local self-attention（O(n·b^2) vs full O((n·b)^2)）。多 hosts 完全并行，zero inter-host communication（vs Ring Attention 的 H 轮 ring exchange）。
  - Anchor block 机制（核心创新）：blockwise-only attention 会产生多个 attention sink（每个 block 起始一个 spike），与 global attention 的单 sink 分布不一致。将 c1 作为 anchor prefix 到每个 block，使 attention sink 集中在 anchor token → 丢弃 anchor KV 后分布逼近 global attention。消融实验（Table 4）：无 anchor 时 64K NIAH 准确率从 99.5% 降至 60.1%（−39.6%）；有 anchor 时 97.6%（−1.9%）。

  *阶段二：Query Encoding & Token Generation — distributed softmax*
  - 解决 Ring Attention 的 decode 阶段通信瓶颈：各 host 独立对 query 做 local attention（用 Flash Attention）→ 仅传递 softmax 统计量（scalar s_h + vector A_h per token）到 query-host → online softmax 聚合为 A_global。
  - 通信量为 O(d) per token per host，与 context 长度无关（vs Ring Attention 的 O(L×d) per layer）。
  - 仅 query-host 更新 KV cache，context hosts 的 cache 保持冻结，避免 decode 阶段全局 KV cache 同步。

  **Serving调度**（对应解决 OOM 和多 GPU 扩展性）：
  - 8B 模型：16K-128K（8 GPU×4 workers），256K-512K（16 GPU×8 workers），1M（32 GPU×16 workers）。
  - 阶段一多 host 并行无通信（embarrassingly parallel），阶段二仅 gather-reduce 标量/向量。
  - 实现于 HuggingFace Transformers 和 TRT-LLM，集成 Flash Attention 加速。
  - 128K tokens 时 vanilla 生成 OOM（8×A100），Star Attn 仅 20s/sample（vs Ring 53s），加速 2.7×。
  - 1M tokens 时 Star Attn 加速 16.9× vs Ring Attention（block size=32K fixed），精度仅降 5.32%。

  **kernel调度**：使用 Flash Attention（Dao, 2024）处理阶段一中 2b-token block 的 self-attention 和阶段二中 local global attention 计算，利用 blockwise tiling 减少 HBM 访问。Star Attention 自身不开发新 kernel，与 Flash Attention 正交结合。

  **编译框架/硬件架构/芯片设计**：论文未明确说明。

  为什么有效：Star Attention 利用 long-context 推理中"context token 只需 local context，query token 需要全局"的观察，将 quadratic 的全量 attention 分解为 block-local + distributed global 两阶段。Anchor block 机制是保证阶段一 local attention 能正确近似 global attention 的关键——它通过控制 attention sink 分布，将 blockwise-only 的多 sink 模式成功转化为逼近 global attention 的单 sink 模式。Distributed softmax 的 log-sum-exp 聚合将通信开销从 O(L) 降至 O(1) per token。

## SageAttention2++: A More Efficient Implementation of SageAttention2

- baseline方法是什么？
  Baseline 是 SageAttention2，一种基于量化的 attention 加速方法。SageAttention2 将 Q,K 量化为 INT4/INT8（per-block），P̃ 量化为 FP8 E4M3（per-block），V 量化为 FP8 E4M3（per-channel），加速 attention 中的两次矩阵乘法。全栈执行过程：

  **算法pipeline**：输入 Q,K,V ∈ R^{N×d}。Step 1: Q,K 使用 INT4/INT8 per-block 量化，通过 INT Tensor Core 计算 QK^T → P̃。Step 2: 对 P̃ 的在线 softmax（online softmax tiling）。Step 3: P̃ 量化到 FP8 E4M3（per-block, δ_P = max(|P̃|)/448），V 量化到 FP8 E4M3（per-channel, δ_V = colmax(|V|)/448）。Step 4: P×V = P̂V̂ × δ_P × δ_V，使用 mma.f32.f8.f8.f32 指令（FP32 accumulator）在 Tensor Core 上计算。Step 5: 输出反量化 O = P×V。

  **系统框架**：论文未明确说明（直接替换 PyTorch attention 调用为 SageAttention2 CUDA kernel）。

  **编译框架/kernel调度**：SageAttention2 基于 FlashAttention 的 tiling 策略和 online softmax，使用 CUDA 编写自定义 kernel。P×V Matmul 使用 mma.m16n8k32 形状的 Tensor Core MMA 指令，但累加器类型为 FP32（mma.f32.f8.f8.f32），相对 FP16 仅 2× 加速。基线 FlashAttention2 完全在 FP16 精度下运行。

  **硬件架构**：NVIDIA RTX 4090 (Ada Lovelace) / RTX 5090 (Blackwell)，利用 Tensor Core 进行低比特 Matmul 加速。FP8 Tensor Core 在 Ada/Blackwell 架构上提供两种指令：FP32 accumulator（2× FP16）和 FP16 accumulator（4× FP16）。

  Baseline 缺陷：SageAttention2 的 P×V 计算仅获得 2× 加速（vs FP16），未能充分利用 GPU 上 FP8 Matmul with FP16 accumulator 提供的 4× 加速能力。原因在于 FP32 accumulator 指令虽然数值范围安全，但理论吞吐仅为 FP16 accumulator 指令的一半。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SageAttention2++ 将 P×V 的 MMA 指令从 mma.f32.f8.f8.f32 替换为 mma.f16.f8.f8.f16，并配合量化范围压缩和延迟 FP32 缓冲来保证数值安全。核心设计：

  1. **Baseline 缺陷：FP8 Matmul with FP32 accumulator 仅 2× 加速。**
     解法：改用 FP16 accumulator 指令（mma.f16.f8.f8.f16），理论加速 2× 提升。关键是解决 FP16 累加器溢出问题——32 次 p×v 乘积累加（mma.m16n8k32）可能超出 FP16 最大值 65504。

  2. **设计对应：Narrowing FP8 Quantization Range。**
     原 SageAttention2: δ_P = max(|P̃|)/448, δ_V = max(|V|)/448（E4M3 完整范围）。SageAttention2++: δ_P = max(|P̃|)/224, δ_V = max(|V|)/4.5，满足 $P_r × V_r ≤ 2047/2$。推导：$|32 × p_max × v_max| = |32 × 224 × 4.5| = 32256 ≤ 65504$。实验表明（Table 2），量化范围从 (448, 448) 缩小到 (224, 4.5) 后，attention 输出的 CosSim 和 L1 相对全精度几乎无损（99.97% CosSim）。

  3. **设计对应：Delayed FP32 Buffering。**
     FP16→FP32 转换需要额外 PTX 指令（cvt.f32.f16）。为减少此开销，连续两次 mma.m16n8k32 结果在 FP16 中累加后再统一转换到 FP32，转换次数减半。额外约束 $P_r × V_r ≤ 2047/2 = 1023.5$，选 $(224, 4.5)$: $224×4.5=1008 ≤ 1023.5$。

  **论文方法全栈执行过程**：

  **算法pipeline**：Q,K 量化步骤同 SageAttention2（INT4/INT8 per-block → INT Tensor Core QK^T → online softmax → P̃）。差异在 P×V 步骤：(1) 缩小 FP8 量化 scale：δ_P = max(|P̃|)/224, δ_V = colmax(|V|)/4.5；(2) P̂ = round(P̃/δ_P) to FP8 E4M3 in [-224,224]，V̂ = round(V/δ_V) to FP8 E4M3 in [-4.5,4.5]；(3) Tensor Core MMA: mma.f16.f8.f8.f16，每 32 元素内积在 FP16 中累加，|32×224×4.5|=32256<65504；(4) Delayed FP32 Buffering：每两次 MMA 结果 FP16 累加后 cvt to FP32，减少转换 PTX 指令开销 50%；(5) O = P̂V̂ × δ_P × δ_V。

  **系统框架**：论文未明确说明（与 SageAttention2 相同，直接替换 PyTorch attention 调用）。

  **kernel调度**：CUDA kernel 在 RTX4090 (Ada) / RTX5090 (Blackwell) 上运行。P×V 使用 mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16 PTX 指令，配合缩小量化范围和 delayed FP32 buffering。Kernel 输出 O 完全在 FP16/FP32 精度内。实测：SageAttn2++(4+8) ≈ 3.9× FlashAttention2，SageAttn2++(8+8) ≈ 3.0× FlashAttention2（RTX4090, headdim=128）。

  **编译框架/硬件架构/芯片设计**：论文未明确说明。

  为什么有效：利用 Ada/Blackwell GPU 上 FP8 Tensor Core 的 FP16 accumulator 变体（4× FP16 理论加速），通过数学约束保证数值安全而不牺牲精度。缩小量化范围是"无痛"优化（Softmax 输出的 P 天然在小值范围，V 的缩小可以通过 P 的放大来平衡，Table 2 验证了 (224, 4.5) 与 (448, 448) 精度等价）。延迟 FP32 缓冲进一步减少类型转换指令开销，在 kernel 级别上对已快的 MMA 路径做微调。

## SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

- baseline方法是什么？
  Baseline 是 PyramidKV / PyramidInfer 等基于注意力权重的 token eviction 方法。这些方法的核心思想是：计算每个 token 的累积注意力分数（accumulated attention score），根据分数从低到高淘汰 token，且在各层间采用金字塔形分配（浅层保留多、深层保留少）。Baseline 全栈执行过程：

  **算法pipeline**：对每个 prefill request，在每层计算完整 attention 矩阵 $A = \text{softmax}(QK^T/\sqrt{d_h})$，然后基于观察窗口 $l_w$ 内的累积注意力分数 $ac_{i,a} = \sum_{b=l-l_w}^{l-1} A_{i,a,b} / (l-a)$ 选择 Top-K token 保留 KV cache。GQA 模型中对同组内所有 Q head 的 ac 取平均（这会丢失 per-head 精度）。每层的保留 token 数按 $\lambda$ 层深线性递减（金字塔形）。后续 decode 阶段仅使用保留的 KV cache 子集计算 attention。

  **系统框架 (Serving)**：论文未明确说明 serving 框架。推理过程为标准 HuggingFace Transformers 流程：prefill 阶段计算全量 attention 并 evict 低分 token → decode 阶段仅在保留的 KV cache 子集上计算 attention。

  **编译框架**：论文未明确说明。

  **kernel调度**：论文未明确说明。eviction 操作通过 PyTorch 张量索引完成（argTopK + gather）。

  **硬件架构/芯片设计**：论文未明确说明。仅提到在单张 RTX 3090 GPU 上测推理速度。

  Baseline 缺陷：(1) 浅层压缩效果差——浅层 attention 分布均匀（不稀疏），eviction 会丢弃大量仍有用的 token；(2) GQA 兼容性差——对同组内 Q head 取平均 ac 后统一淘汰会丢失细粒度 head 差异信息；(3) 忽略了浅层 KV cache 的"构成性冗余"（constituent redundancy），即不同 token 的 KV 向量之间存在高余弦相似性，可通过分解为基础向量组合来压缩。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SpindleKV 的核心设计是"双重策略平衡浅层和深层"：(1) 深层继续使用 attention weight based eviction（与 baseline 一致但有 GQA 优化）；(2) 浅层使用基于余弦相似度的 codebook replacement 压缩构成性冗余；(3) GQA 处理通过"先 unfold（repeat KV vectors）→ eviction → codebook 压缩"三步走，利用 unfold 引入的冗余被后续 codebook 压缩消除。

  **算法pipeline**：
  - 深层 eviction：与 PyramidKV 类似的金字塔形分配 $r_c(\lambda) = r_c(0) + (r_c(m-1)-r_c(0))/(m-1) \cdot \lambda$，但 GQA 模型选择 unfold KV head（repeat $h_n$ 次）而非取平均 ac，从而避免 per-head 信息丢失。
  - 浅层 codebook：计算所有保留 KV token 的两两余弦相似度矩阵 $S_\Gamma$，设阈值 $\theta_K=0.98, \theta_V=0.95$ 构建邻接图 $G_\Gamma$。贪心迭代：每次选图中度数最高节点加入 codebook $C_\Gamma$，将其邻居节点全部映射到该 codebook entry，通过 `matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])` mask 从图中移除已处理节点。同时记录每个 token 的 L2 magnitude $m_\Gamma$ 和 codebook 引用索引 $r_\Gamma$。最终存储开销 = |codebook entries| + |indices (int)| + |magnitudes (float)|，远小于原始 KV cache。
  - 推理重建：$\Gamma_r = C_\Gamma[r_\Gamma] \otimes m_\Gamma$，然后对重建的 K 重新应用 RoPE（论文论证 RoPE 是稀疏矩阵乘法，不增加显著时间开销）。
  - GQA 全流程：unfold KV → eviction → codebook → 压缩。Unfold 增加的 KV cache 大小被后续 codebook 压缩抵消（unfold 引入的重复向量余弦相似度为 1，极易被 codebook 合并）。

  **系统框架**：论文未明确说明。

  **编译框架/kernel调度/硬件架构/芯片设计**：论文未明确说明。

  为什么有效：浅层的 KV cache 中 token 向量之间余弦相似度极高（超过 0.9），这些 token 虽然 attention 分数不稀疏（eviction 难以淘汰），但其 KV 向量可被少数几个"基础向量"（codebook entries）线性表示。深层的 attention 存在强稀疏性，eviction 即可有效压缩。两者互补，使得在不同 KV cache 保留率下均优于 PyramidKV/PyramidInfer。例如在 LLaMA3-8B 上 40% KV cache 保留率时，SpindleKV LongBench 平均分 41.13 vs PyramidKV 39.86；在 15% KV cache 保留率时，Needle-in-a-Haystack 准确率 0.979 vs PyramidKV 0.938。

## Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

- baseline方法是什么？
  Baseline 是标准 Llama 模型通过 vLLM 进行 prefill + decode 推理，以及对比方法 RAG-LLAMA（sentence-level RAG）、LLMLingua（文本级压缩）、MInference（sparse attention）。Baseline 全栈执行过程：

  **算法pipeline**：Transformer 标准 prefill 对所有 prompt token 执行完整的 attention + MLP 计算，每层计算 $A = \text{softmax}(QK^T/\sqrt{d_k})V$ 和 MLP 投影。RAG-LLAMA 基于 sentence embedding 相似度检索相关句子拼接后送入主模型。LLMLingua 使用小型模型做困惑度估计压缩 prompt 文本。MInference 使用离线搜索的稀疏 attention mask pattern 跳过部分 attention 计算，但不减少 MLP 计算量。

  **系统框架 (Serving)**：vLLM 0.6.3.post1，TP=8，enforce_eager=True。请求到达后直接进入 prefill phase，所有 prompt token 的 KV cache 被计算并写入 PagedAttention KV blocks，prefill 完成后再进行 decode phase。TP 组内各 GPU 通过 NCCL all-reduce 同步 MLP 和 attention 输出。

  **编译框架**：论文未明确说明（使用 vLLM 内置 CUDA kernels，无自定义编译优化）。

  **kernel调度**：论文未明确说明（使用 vLLM 默认 FlashAttention/PagedAttention kernel，TP=8 下自动并行调度）。

  **硬件架构**：论文未明确说明（标准 8×H200 服务器，各 GPU 通过 NVLink 478.1 GB/s + PCIe 5.0 x16 互联，无定制硬件加速器）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 SPECPREFILL 利用一个小型 speculator 模型（Llama-3.1-8B）在 prefill 前预测哪些 prompt token 是"局部重要的"，仅将筛选出的 token 子集送入主模型，从而跳过大量 attention + MLP 计算。核心设计对应 Baseline 缺陷：

  1. **Baseline 缺陷：Prefill 是 compute-bound，MLP 占比大且无法被 sparse attention 加速。**
     SPECPREFILL 解法：直接丢弃不重要 token，同时跳过 attention 和 MLP 计算（以及 TP all-reduce 通信），计算量正比于 token 保持率。对于 405B 模型 10% 保持率，理论 FLOPS 降至原来的 ~12.96%（speculator FLOPS 仅为主模型的 2.96%），实测 TTFT 加速达 7.66×，QPS 提升 7×。

  2. **Baseline 缺陷：MInference 等 sparse attention 方法在大 batch、短到中等长度 prompt 下 overhead 大。**
     SPECPREFILL 解法：跳过 attention + MLP 双重计算，在大 batch size 下优势更明显，相对 MInference 加速 2.54×-6.54×（同时保持 99.5% 质量）。

  3. **Baseline 缺陷：SwiftKV 需要轻量微调，GemFilter 需要两次完整 forward pass 开销大。**
     SPECPREFILL 解法：完全 training-free（利用同系列模型间的 token importance transferability），speculator overhead 随主模型增大而可忽略（405B 时仅 2.96% FLOPS），且可被 speculative decoding 复用摊销。

  **论文方法全栈执行过程**：

  **算法pipeline**：请求 prompt 先经过 speculator (8B) 的 N=8 步 look-ahead forward → 从各层各头提取 [N=8, L=32, S, H] 注意力张量 → max over H, L → mean over N → 得每 token 标量重要性分数 [S] → 1D avg pool 平滑 → 按 chunk 取平均 → Top-K chunks 选出 token 子集 → 保持各 token 原始 position IDs → 仅将 token 子集送入主模型 (70B/405B) 的完整 forward（包括该子集 token 的 attention + MLP 全流程）。

  **系统框架 (Serving)**：在 vLLM engine 初始化前 monkey patch 插入 speculator 加载与 token 选择逻辑。请求到达后：(1) 先由 speculator 处理（含 N 步 look-ahead decoding），利用 vLLM 的 slot mapping 机制追踪 query 数据；(2) N 步结束后 tp_gather_qk 收集 TP 组内 Q、K 分片；(3) 聚合注意力分数 → chunk selection → 筛选 token 子集；(4) 将筛选后的 token（含原始 position IDs）合并 decode 请求送入 base model forward。SPECPREFILL 与 speculative decoding 天然兼容：small model 同时服务于 prefill 阶段的 token 选择和 decode 阶段的 draft proposal，为"小型 speculator 完全辅助推理"范式铺路。

  **编译框架**：论文未明确说明（无自定义编译优化，沿用 vLLM 默认 CUDA graphs/kernels）。

  **kernel调度**：论文未明确说明（使用 vLLM 默认 FlashAttention + PagedAttention kernels，无额外 kernel 定制）。

  **硬件架构**：论文未明确说明（标准 8×H200/H100 GPU 节点，使用 NVLink + PCIe 5.0 互联，无定制硬件加速器）。

## LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

- baseline方法是什么？
  Baseline 是标准 Transformer（所有层使用 full causal self-attention，维护完整的 KV cache），以及现有的层级 KV cache 缩减方法（StreamingLLM、MiniCache、SqueezeAttention），其全栈执行过程如下：

  **算法pipeline**：标准 Transformer 每层执行 full attention，计算 $A_i = \text{softmax}(Q_i K_i^T / \sqrt{d_k} + M) V_i$，所有 token 的 KV 对都被存储和参与后续 decoding。StreamingLLM 将所有层的 attention 替换为 streaming attention（仅保留 sink + recent token 的 KV cache）。MiniCache 在模型后半部分每相邻两层间通过 SLERP 合并 KV cache。SqueezeAttention 按层分配不同的 KV cache 预算，但需完成所有层 prefilling 后才能压缩（无法降低峰值内存）。

  **系统框架**：论文未明确说明（基于 PyTorch + HuggingFace Transformers + FlashAttention）。

  **编译框架**：论文未明确说明。

  **kernel调度**：使用 FlashAttention 加速 attention 计算（NVIDIA A100 GPU），Flex Attention 用于 LightTransfer-TRAIN 的优化训练。论文未涉及自定义 kernel 实现。

  **硬件架构**：论文未明确说明（使用 NVIDIA A100 GPU 进行所有实验）。

  Baseline 的核心缺陷：（1）标准 Transformer 的 KV cache 随层数和序列长度线性增长，成为长上下文推理的内存瓶颈；（2）StreamingLLM 将所有层都替换为 streaming attention，严重损害模型的全局信息捕获能力（LongBench 上平均下降 3.5-11.5%）；（3）MiniCache 和 SqueezeAttention 仅从 KV cache 相似性或粗粒度预算分配角度进行压缩，未深入理解不同层的功能差异——前者最多压缩 25% 层，后者无法降低 prefilling 峰值内存；（4）从 scratch 训练 Hybrid 模型（如 Jamba、Gemma 2）需要大量计算资源，而将预训练 Transformer 转换为 Hybrid 的方法（如 LongGen）仍需超过 2TB 的重训练数据。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 LightTransfer，通过识别 Transformer 层的功能差异（lazy vs non-lazy），将标准 Transformer 无损转换为 Hybrid 架构。其核心洞察是：（1）某些层表现出"懒惰"行为——注意力主要集中在初始 sink token 和最近 token 上；（2）层行为对给定输入具有一致的跨 token 稳定性；（3）因此可以在 prefilling 阶段动态识别懒惰层并替换其 attention 机制。

  对应解决 Baseline 缺陷的全栈执行：

  **算法pipeline**：定义 lazy ratio $r_i = \frac{1}{w_{\text{last}}} \sum_{\hat{x} \in X_{\text{last}}} \sum_{x \in \{X_{\text{initial}}, X_{\text{recent}}\}} A_i(\hat{x}, x)$ 量化每层的懒惰程度。利用 FlashAttention 的 LSE（log-sum-exp）输出值计算 r_i，避免重算完整注意力矩阵——仅需一次 O(w_last × (w_sink+w_recent)) 的小矩阵乘法。使用最大堆优先队列（大小 P = 50% 总层数）在 prefilling 中动态选择 lazy ratio 最低的层保留 full attention，其余替换为 streaming attention（仅保留 $w_{\text{sink}}=4$ + $w_{\text{recent}}=1020$ 的 KV cache）。这直接优化了 Theorem 5.1 中网络输出误差的上界（误差 ≤ 被丢弃 KV 对的注意力分数之和 × 常数）。

  **Prefilling 阶段的 KV cache 管理流程**：
  ```
  输入 tokens → Layer 0: 计算 full attention → 计算 lazy ratio r_0 → 入堆
             → Layer 1: 计算 full attention → 计算 lazy ratio r_1 → 入堆
             → ...
             → Layer k (堆满 P 后): 弹出 ratio 最高的层 L_lazy
               → L_lazy 的 KV cache 缩减为 {sink tokens + recent tokens}
               → 释放的显存用于存储当前层 KV cache
             → ... → 最后一层
  输出: 只保留 P 层的完整 KV cache + 其余层的缩减 KV cache
  ```
  对比 StreamingLLM 的全层替换，LightTransfer 保留了 P 层 full attention 作为全局信息"锚点"，同时 lazy 层的 streaming attention 固定大小（~1K tokens），实现 2.17× 吞吐提升且 LongBench 仅下降 <1.5%。

  **系统框架**：LightTransfer-TEST 完全在 test-time 运行，无需任何训练数据或校准集，通过 FlashAttention 的 LSE API 实现零额外开销的懒惰层识别（相对吞吐仅降低 0.0014-0.0058×）。压缩在 prefilling 期间完成，因此同步降低了峰值内存使用——这是 SqueezeAttention（需所有层完成 prefilling 后压缩）无法实现的。LightTransfer-TRAIN 仅需 ~5K 训练样本（原用于蒸馏的数据）进行 SFT，远少于 LongGen 的 2TB+ 重训练数据。

  **编译框架**：论文未明确说明。

  **kernel调度**：利用 FlashAttention 的 `return_lse=True` 参数获取 log-sum-exp 值作为"免费"的注意力分布代理，避免了完整 attention matrix 的 O(n²) 重计算。lazy ratio 计算仅需一次 batched matmul：`log_lazy_ratio = matmul(q_last, k_comb.transpose).logsumexp(-1) - lse`。Flex Attention 用于 LightTransfer-TRAIN 的 SFT 阶段优化混合 attention 模式的训练效率。

  **硬件架构**：在 8×A100 40G 节点上验证了 layer-wise（而非 head-wise）hybrid 设计的必要性：head-wise hybrid 在 TP 下因不同 head 的 KV cache 大小不一致导致同步瓶颈；DP+TP 方案因注意力层参数复制消耗额外 157.5 GB 显存，吞吐仅为纯 TP 的 0.0735×，最大支持序列长度降至 1/128×。

- baseline方法是什么？
  Baseline 是标准 dense attention（RoPE/p-RoPE），以及现有的长上下文泛化方法（LogN scaling/SSMax、ALiBi、NTK-aware scaling、YaRN），其执行全栈过程如下：
  
  **算法pipeline**：标准 attention 计算 logits 为 $L_t = S_t$（无位置依赖变换），或使用位置无关的全局缩放（LogN：$L_t = s \log N \cdot S_t$），或加性偏置（ALiBi：$L_t = S_t - m \cdot t$）。这些方法将 softmax 后的 $A_t$ 作为权重对 V 加权求和。
  **系统框架**：论文未明确说明（使用 PyTorch 标准实现）。
  **编译框架**：论文未明确说明。
  **kernel调度**：使用 PyTorch 的 FlexAttention API 生成 GPU kernel；162M 模型单卡 A100，304M 模型 4×H100 DDP。
  **硬件架构**：A100/H100 GPU，论文未明确说明底层架构细节。
  
  Baseline 的核心缺陷：（1）无缩放/标准 attention 在长上下文时 attention 分布变得极度扩散（高熵），大量注意力被分散到远距离的不相关 token 上，局部上下文的注意力权重极速衰减；（2）LogN 通过全局缩放降低了熵，保持了稀疏性，但以牺牲局部上下文注意力为代价——对所有位置同等缩放，导致即使是近 100 token 的注意力也被压缩；（3）ALiBi 的线性偏置过于刚性，无法灵活控制不同 token range 的注意力分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 Scale-invariant Attention：根据与当前 token 的距离 $t$，对 attention logits 施加位置依赖的乘性缩放 $a_t$ 和加性偏置 $m_t$，使得：
  - $a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}$ （标准差随 $\log t$ 增长→分布更尖锐）
  - $m_t = -a_t^2 + \beta/\alpha$ （均值随 $-\log t$ 下降→压低远距离总权重）
  
  对应解决 Baseline 缺陷的全栈执行：
  
  **算法pipeline**：在每个 attention head 中，对已施加 p-RoPE 的 score $S_t$ 计算 $L_t = a_t \cdot S_t + m_t$，其中 $t$ 是 query-key 距离。当 $t \ll \tau$ 时 $a_t \approx 1, m_t \approx 0$（局部上下文 ≈ 标准 attention）；当 $t \gg \tau$ 时 $a_t$ 对数增长使分布尖锐（稀疏化远距离 attention）而 $m_t$ 对数下降控制远距离 token 总体权重不过大。Softmax 归一化后加权求和 V。该设计直接实现两个数学性质：scale-invariant total attention（$\mathbb{E}[Z_t^{t\Delta}] = \Theta(1)$，使各 token range 的注意力总量渐进恒定）和 weak scale-invariant attention sparsity（$\mathbb{E}[H_t^{t\Delta}] = o(\log t)$，注意力稀疏性随上下文变长而增加）。
  
  **系统框架**：论文未明确说明（基于 PyTorch + modded-nanogpt 实现）。
  **编译框架**：论文未明确说明。
  **kernel调度**：使用 FlexAttention API 自定义 attention score modification，在 GPU 上高效实现位置依赖的 logit 变换；训练用单卡 A100 或 4×H100 DDP。
  **硬件架构**：A100/H100 GPU，论文未明确说明底层架构细节。
  
  对比 baseline LogN 的关键区别：LogN 使用位置无关的 $s\log N$ 缩放所有位置，导致近处 token（$t$ 小）的 attention 也被不当缩放；而 Scale-invariant Attention 的 $a_t$ 和 $m_t$ 是位置依赖的，仅缩放远距离 token，保留局部上下文的完整注意力能力。

- baseline方法是什么？
  Baseline 是 KV cache 压缩研究的现有评估实践，存在三大缺失：
  
  **Missing Piece 1 — 仅有 TRL 框架测吞吐**：大多数压缩研究仅在 Transformers library (TRL) 上测量吞吐性能，忽略 FlashAttention 和 PagedAttention 等生产级 serving 技术。TRL 上测到的加速比（如 StreamingLLM 在 TRL 上 2-3× decoding speedup）在 LMDeploy（含 FlashAttention + PagedAttention）上大幅缩水甚至消失。
  
  **Missing Piece 2 — 固定响应长度测吞吐**：现有工作以固定响应长度评估计算效率，忽略压缩算法导致 LLM 生成更长/更 verbose 的输出，从而增加端到端延迟。测到的 throughput speedup 可能被 longer output 抵消。
  
  **Missing Piece 3 — 只看平均 accuracy 不看 individual samples**：绝大多数评估只报告整体 accuracy（如 LongBench average score），隐藏了压缩算法对不同 task type 和 individual samples 的不均衡退化。long-context 任务（summarization、QA）特别脆弱。
  
  **全栈执行例子（baseline = LLaMA-7B + KIVI-4bit on TRL without FlashAttention/PagedAttention）**：
  - **算法层**：KIVI 使用 per-channel key quantization (group_size=32) + per-token value quantization (INT4)，保留最近 128 token 为 FP16。Pre-fill 不量化，decode 每步对新 token K/V 量化后追加。Attention 计算时从 DRAM 加载 quantized K/V → dequantize → 与 Q 做 matmul。评测仅测 throughput (tokens/s) 和 memory reduction。
  - **系统框架层**：直接调用 HuggingFace Transformers 的 `model.generate()`，无 PagedAttention、无 continuous batching、无 KV cache page 管理。GPU memory 预分配至 max_length。
  - **编译框架层**：论文未明确说明（TRL 使用 PyTorch eager mode，无定制 compilation）。
  - **kernel调度层**：TRL 默认使用 PyTorch SDPA（`torch.nn.functional.scaled_dot_product_attention`），可能回退到 FlashAttention 2 的 fused kernel，但 KIVI 的量化/反量化操作为独立的 Python-level 操作，打破了 fused attention kernel 的端到端优化。
  - **硬件架构层**：NVIDIA A6000 48GB。GPU 执行流程：quantize kernel (低利用率，element-wise op) → dequantize kernel → SDPA attention kernel → 中间结果在 HBM 和 SRAM 间传输，量化带来的 memory foot-print 减少被额外的 quant/dequant kernel launch overhead 和 irregular memory access 抵消。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本论文不是提出新的压缩算法，而是**重新定义 KV cache 压缩的评估框架和部署工具**，通过三个新评估维度和一套工具链解决 baseline 的三大缺失。
  
  **(1) 评估维度一 — 在生产 serving 框架上测吞吐**：
  - 将 KIVI、GEAR、StreamingLLM、H2O 四种算法集成到 LMDeploy v6.0.1（原生支持 FlashAttention 2.5.6 + PagedAttention）。
  - 在不同 batch size (1-32)、prompt length (512-8192)、tensor parallelism (1/2/4) 下全面测量 prefill 和 decoding 吞吐。
  - **发现**：KV cache 压缩在轻量级设置下无加速甚至负加速。例如，GEAR 在 TP=1 prefill 仅 0.86× FP16 baseline、KIVI 在 TP=2 decode 仅 0.88×。Tensor parallelism 会削弱压缩收益（因 TP 缓解了 per-GPU memory bandwidth contention）。
  
  **(2) 评估维度二 — 响应长度分布分析**：
  - 比较压缩算法 vs 温度参数调整对输出长度的影响。定义 D = (L_un - L_cs)/L_un（负值=压缩导致更长输出）。
  - **发现**：KIVI/GEAR/H2O/StreamingLLM 均导致 >20% 样本的响应长度增加 ≥50%（1.55-1.76× length increase）。高压缩比（更低 bit / 更短 KV cache）加剧 verbose 输出。
  - 结合长度和吞吐评估端到端延迟 CDF，GEAR 甚至出现更高的 tail latency。
  
  **(3) 评估维度三 — Negative sample 分析**：
  - 使用 Algorithm 1 定义 negative sample：benign sample 在压缩后 relative accuracy loss > threshold。
  - **发现**：即使整体 accuracy 损失很小（<1%），仍存在大量 negative samples（threshold=10% 时数百个）。Summarization 和 QA 任务特别脆弱。
  
  **(4) 三件工具**：
  - **Throughput Predictor**：基于 Vidur offline-profiled attention operator runtime，预测任意 (batch_size, seq_len, stage) 组合的吞吐，精度 >85%。
  - **Length Predictor**：LongFormer-based BERT classifier，输入 prompt text，预测 response_length/prompt_length ratio，精度 >85%。
  - **Request Router**：在 4 GPU 混合部署（1 FP16 + 3 compressed）下，结合两个 predictor 路由请求到估计 E2E latency 最小的 GPU，实现 1.45-1.80× E2E latency speedup vs load-balancing baseline。
  
  **全栈执行例子（论文方法 = LLaMA-7B + KIVI-4bit on LMDeploy + Request Router, 4× A6000）**：
  - **算法层**：与 baseline 相同的 KIVI 量化 pipeline（per-channel key quant + per-token value quant）。不同的是在 serving 框架内评估，量化开销与 FlashAttention 的 fused kernel 交互。
  - **系统框架层**：LMDeploy v6.0.1 管理 KV cache → PagedAttention allocates fixed-size page blocks → FlashAttention executes tiled one-pass attention。KIVI window-based quantization（保留最近 128 token FP16）与 PagedAttention 的 fixed-type page blocks 不兼容，导致需要同时管理 FP16（window）和 INT4（历史）两类 tensor → 非结构化计算模式 → GPU 利用率下降。**Request Router**：Throughput Predictor 离线 profile attention op runtime → 在线查表预测 decode throughput；Length Predictor (LongFormer) 预测 response/prompt ratio → 路由到最小 E2E latency GPU。
  - **编译框架层**：论文未明确说明。LMDeploy 使用 TurboMind C++ backend，量化 kernel 为 custom CUDA kernel。
  - **kernel调度层**：LMDeploy 的 4-bit 量化 kernel（比 vLLM 更高效，BentoML benchmark 验证）→ quantize element-wise (low GPU occupancy) → dequantize → FlashAttention fused kernel。Profiling 表明 attention layer execution time 在 prefill 阶段 KIVI 接近 FP16 baseline（因 prefill 不量化），但在 decode 阶段量化 kernel overhead 随 KV length 增长而显著。
  - **硬件架构层**：4× NVIDIA A6000 + NVLink。TP=4 时 KIVI decode 仅 0.9× FP16 baseline（因 TP 已经分摊了 per-GPU memory bandwidth，压缩的 memory reduction 收益被稀释）。Request Router 在 4 GPU 上测试：1 GPU = FP16, 3 GPU = KIVI-4bit。w/ Both 策略平均 E2E latency = 6.3s (KIVI) vs Baseline (load-balancing) = 9.1s (KIVI) vs FP16 = 11.4s → 1.80× speedup over KIVI with simple load-balancing。

## Rectified Sparse Attention

- baseline方法是什么？
  Baseline 是直接使用 block-sparse attention 进行全长度稀疏解码（如 Quest [23]、ClusterKV [18]、InfLLM [24] 等 query-aware training-free sparse attention 方法）。核心缺陷：

  **KV Cache 误差累积**：Sparse decoding 每一步都基于近似 attention 计算，产生的预测 token 及其 KV cache 条目包含近似误差。这些误差随 decoding 步数累积在 KV cache 中，导致后续 attention 计算基于越来越不准确的 KV cache，形成"误差累积"恶性循环。如图 1 所示，sparse decoding 性能随 decoding length 增长持续下降。Quest 尝试通过跳过前两层的策略缓解，但效果有限（Table 1 中 Sparse_dense2 vs Sparse 差异不显著）。

  全栈执行例子（Quest-style Sparse Decoding, Qwen2.5 7B, p=0.9, A100）：
  - **算法层**：prefill 后用 dense attention 构建初始 KV cache。Decode 阶段每一步用 query-aware block-sparse attention 近似 full attention——将 KV cache 划分为 block，用 min/max 描述符做近似匹配，每步选择 top-n block attended。新生成的 token 追加到 KV cache。全过程无 rectification，KV cache 的误差随步数单调递增。
  - **系统框架层**：基于 PyTorch 实现，无特殊框架修改。可与 vLLM 等 serving 框架集成。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 Flash Decoding kernel 进行 sparse attention 的 split-execution。无 rectification 操作，kernel 专注 sparse attention computation。
  - **硬件架构层**：NVIDIA A100-80G GPU。Sparse attention 减少 HBM 访问，但 KV cache 质量退化导致 math reasoning 准确率下降（Qwen2.5 7B avg: Dense 60.72, Sparse 57.72, gap=3.0）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReSA 通过引入周期性 dense rectification 直接解决 KV cache 误差累积问题，核心设计如下：

  **(1) 算法层——Group Block Sparse Attention + Dense Rectification**：

  **GBSA**：继承 Quest 的 block-sparse attention（block 描述符、query-dependent top-n selection），新增 GQA group 内共享 attention pattern（来自 NSA 的 shared grouping），减少 block selection 开销。因为同一个 GQA group 的 query heads 共享 KV，复用同一组 block indices 避免了为每个 head 单独做 block selection 的重复计算和 memory access。

  **Dense Rectification**：每 f=32 个 decode step 后，将这 f 个最近生成的 token 批量通过 dense attention 并行重编码，刷新 KV cache 和 block key cache。这从根本上约束了稀疏误差的累积范围——每 f 步后 KV cache 被"校正"到 dense 精度，误差窗口被限制在 f 以内（而非随时间无限累积）。

  **Decoding Procedure**（Algorithm 1）：
  ```
  Prefill(P) → dense KV cache K
  for i in 1..T:
      t = SparseForward(G[i-1], K, B)  # 快速稀疏生成
      append t, update K and B
      if i % f == 0:
          K, B = DenseForward(G[i-f:i], K, B)  # 批量 dense rectification
  ```

  - vs Baseline 的无界误差累积：ReSA 将误差窗口限制在 f 以内。即使 f=128 仍保留大部分性能增益（Fig 9），表明 rectification 对频率不敏感，鲁棒性高。
  - vs Quest 的跳过前两层：ReSA 在所有层都应用 sparse attention + rectification，无需特殊处理特定层。实验显示 Sparse_dense2（前两层 dense）改善不显著，而 ReSA 显著改善。
  - vs Self-Speculation：ReSA 无需 per-token accept/reject 决策和 resampling，比 sparse KV-based self-speculation 平均快 1.92×（Table 3）。

  **(2) Memory Access 模型**：
  公式 Avg(mem) = mem(KV cache) × (1/b + p + 1/f)，显式地量化了 block 粒度 b、sparsity ratio p、rectification frequency f 对 memory access 的影响，提供了理论加速上界。

  **(3) Kernel调度层——Flash Decoding + Block-Sparse Kernel**：
  Custom kernel 采用 GQA-aware SM 分配和 block-level workload splitting，在每个 SM 上独立 fetch 其负责的 block subset 并执行 sparse attention。关键优化：
  - Sparse attention loop 仅遍历 selected blocks（由 GBSA block selection 产生），而非全部 KV blocks。
  - Intra-GQA key 共享减少 HBM 访问：同一 group 的 query heads 复用加载的 KV 数据。
  - Block key cache 在线增量更新，新 token 追加时仅更新对应 block 的 min/max 描述符，O(1) per token。

  实验表明 sparse estimation 和 attention computation 耗时相当（均 ≈ mem(KV cache) × 0.9），这是 kernel 效率设计的关键平衡点。

  **(4) 系统框架兼容性**：
  Rectification 天然兼容 continuous batching 和 chunked prefill（如 Sarathi、DeepSpeed-FastGen），仅需周期性批量重编码，无需引入特殊同步屏障。

  全栈执行例子（ReSA, Qwen2.5 7B, p=0.9, f=32, A100-80G）：
  - **算法层**：Prefill (dense) → decode (GBSA, p=0.9) × 32 steps → rectification (dense forward over last 32 tokens) → decode × 32 → ... 循环。Math reasoning avg accuracy: 60.52 vs Dense 60.72 (gap=0.20)，近乎无损。
  - **系统框架层**：基于 PyTorch，可通过 continuous batching 实现 rectification 的批处理。256K context 下 INT4 end-to-end 2.44× speedup。
  - **编译框架层**：TileLang 库辅助实现 GBSA kernel（Acknowledgments 中致谢）。
  - **kernel调度层**：Custom Flash Decoding kernel with block-sparse support。256K 下 rectification overhead 仅 32.7%，延迟随序列长度接近线性而非二次。
  - **硬件架构层**：NVIDIA A100-80G GPU。Sparse + 量化 (INT4 Marlin) 正交组合，256K 达 2.44× speedup。

- baseline方法是什么？
  Baseline 是 Palu（Chang et al., 2024）的 G-LRD（Grouped Low-Rank Decomposition）变体，即对 KV projection 矩阵直接做 group-wise SVD 低秩分解来压缩 KV cache hidden dimension。Palu 的核心缺陷：
  (a) **忽略 Key-Value 不对称性**：Palu 对 Key 和 Value 使用相同的 SVD 压缩策略，未区分 Keys 承载位置编码（RoPE）信息需重建 → 有额外计算开销，Values 承载语义信息 Fisher Information 显著更高 → 需更高保真度保留。
  (b) **Head 分组随机**：Palu 的 group-wise SVD 将相邻 head 按物理索引简单分组（无相似性考量），不同 head 的 left singular subspace 差异大 → 组内 SVD 近似误差高。
  (c) **Value SVD 无校准**：标准 SVD 分解未优化重建误差 E = ||L_v R_v X - W_v X||_F^2，在高压缩率下精度下降显著。Fisher Information 分析显示 Value projection 矩阵的 Fisher Information 显著高于 Key projection，表明 Value 对模型行为更关键，简单的 SVD 截断会引入较大性能退化。

  全栈执行例子（Palu G-LRD, LLaMA-2-7B, MHA, 70% 压缩率, A800）：
  - **算法层**：对 Key 和 Value projection 按物理相邻 4 个 head 分组，每组做 SVD 低秩分解（group_size=4），压缩后 KV cache hidden dim 减至 30% 原始大小。Key 和 Value 使用相同的压缩率和相同的 group-wise SVD 策略，未区分两者不同特性。70% 压缩率下 WikiText2 困惑度从 5.47 升至 8.62。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers，在模型加载后 offline 修改 Key/Value projection 权重矩阵，推理时无额外框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用标准 FlashAttention kernel，不对低秩 Key/Value 路径做 kernel 级融合优化。Key 重建和 Value 重建在 kernel 外部完成。
  - **硬件架构层**：NVIDIA A800 GPU。KV cache 内存占用减少但无针对性 kernel 优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReCalKV 通过分析 Key 和 Value 在注意力机制中的不同角色，分别设计了 HSR（针对 Key）和 OVC（针对 Value）两种差异化压缩策略：

  **(1) 算法层——差异化 Key-Value 压缩策略**：

  **HSR (Keys)**：利用 CKA 相似度衡量 head 的表征子空间相似性，将相似度高的 head 通过贪心算法分为一组（而非简单按物理索引分组），组内做 group SVD。相似的 head 共享更多表征成分 → SVD 低秩近似误差更低。推理时需执行在线 inverse reordering 恢复原始 head 顺序。
  - vs Palu 的随机分组：HSR 将 CKA 相似度最高的 head 聚为一组，lower approximation error from shared left singular subspace。
  - 消融实验（80% 压缩率）：HSR alone 将 WikiText2 困惑度从 9.34 降至 9.01，LongBench Avg Acc 从 9.01% 升至 12.44%。

  **OVC (Values)**：先对 Value projection 做 SVD，再用标定数据 X 按闭式解校准 L_v 和 R_v，最小化重建误差 E。校准后 R_v 通过 Matrix Fusion 融合进 W_o 中，推理时无需在线重建 Value。
  - vs Palu 的未校准 SVD：OVC 的闭式校准直接最小化 ||L_v R_v X - W_v X||_F^2，比标准 SVD 截断更精确。
  - 消融实验（80% 压缩率）：OVC alone 将 WikiText2 困惑度从 9.34 降至 8.91，LongBench Avg Acc 从 9.01% 升至 13.09%。
  - HSR + OVC 联合：WikiText2 困惑度降至 8.48，LongBench 升至 15.40%。

  **(2) Fisher Information 引导的压缩率分配**：
  借鉴 Palu 的 Fisher Information 策略，按每层的重要性分配不同的压缩 rank。高 Fisher 层保留更多 rank，低 Fisher 层可更激进压缩。这确保关键层（如承载长程依赖的中间层）的近似质量优先保证。

  **(3) Kernel调度层——Triton Fused Attention Kernel**：
  自定义 Triton fused attention kernel 将 HSR 的在线 head permutation 和 OVC 的 Matrix Fusion 整合到单一 kernel 执行路径中。Key 路径：X·L_k → K_latent → 重建 → inverse reorder → RoPE → attention scores。Value 路径：X·L_v → V_latent（存入 cache）→ fused output。Kernel 融合避免中间结果的 HBM 往返，70% 压缩率下 65K prompt 达到 1.80× 加速。

  **(4) 正交兼容性**：
  ReCalKV 与量化技术正交——可与 4-bit/3-bit per-token quantization + Hadamard transform 组合进一步压缩（Section 4.4），70% low-rank + 3-bit quant 仍维持 7.01 WikiText2 困惑度。

  全栈执行例子（ReCalKV, LLaMA-2-7B, MHA, 70% 压缩率, A800）：
  - **算法层**：Fisher 分配 rank → Key 做 HSR (CKA → greedy grouping → group SVD) → Value 做 OVC (SVD → closed-form L_v/R_v calibration on 256 WikiText2 samples → Matrix Fusion)。70% 压缩率下 WikiText2 困惑度 6.75（vs Palu 8.62），零样本 QA 平均准确率 59.90%（vs Palu 52.14%）。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers，offline 修改模型权重。无额外 serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Triton fused attention kernel，集成 HSR 在线置换和离线 Matrix Fusion。70% + 65K 加速 1.80×。测试在 A800 上进行。
  - **硬件架构层**：NVIDIA A800 GPU (80GB)。低秩 KV cache 减少 HBM 占用和带宽压力，长 prompt 下效果更显著。

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

## Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

- baseline方法是什么？
  Baseline 有三类：(1) **Dense self-attention**；(2) **Fixed Sparse Attention**（位置固定稀疏，stride-based）；(3) **Routing Transformer**（online K-means 聚类）。

  全栈执行例子（以 Dense baseline, Small model 113M, T=1024, 单 A100）：

  - **算法层（Dense baseline）**：标准 multi-head self-attention，9 heads，每 head 计算完整 Q=XW^Q, K=XW^K, V=XW^V（各 ∈ R^{1024×64}），然后 A = softmax(QK^T/√64 + M) @ V（QK^T ∈ R^{1024×1024}），FLOPs = 9×(8×1024×64×1024 + 4×64×1024²) ≈ 9×(0.537×10^9 + 0.268×10^9) = 7.25 GFLOPs/层。KV-cache: T × H = 1024 × 9 = 9.2K key-value pairs。所有 token 都参与计算，无论其重要性如何。

  - **算法层（Fixed Sparse baseline, ρ=32, k=32）**：每个 head 固定选择位置 [0, 32, 64, ..., 992] 的 32 个 token，计算 Q/K/V 投影和 attention。关键缺陷：(a) 稀疏模式与内容无关，无法根据当前输入动态调整关注点；(b) 预选 token 必须在早期层聚合周围信息，在后续层再将信息路由回原始位置——这一信息路由开销限制了模型的表达能力；(c) 所有 head 使用完全相同的 token 选择，缺乏 head 间专业化。

  - **算法层（Routing Transformer baseline）**：每 head 用 online K-means 将 tokens 聚为 ρ 个簇（各 k 个 token），基于 dot-product 距离将 token 分配给最近簇中心。簇中心通过移动平均更新。关键缺陷：(a) online K-means 收敛极慢 [Bottou & Bengio, 1994]，即使在数十万步训练后簇分配仍不稳定；(b) 必须计算所有 T 个 token 的 Q 和 K 投影才能聚类（2hh'T = 2×1024×64×1024 ≈ 0.134 GFLOPs overhead per head），且每一 Routing head 的 FLOPs ≈ ρ 个 MoSA head；(c) 每 head 内所有 ρ 个簇共享同一套线性变换 W^Q/W^K/W^V/W^O；(d) 为让 source 和 destination 选同样的 token，必须设 W^Q=W^K，限制灵活性。

  - **kernel调度层**：PyTorch native einsum/scatter/gather 操作，无专用 CUDA kernel。FlashAttention 可用于 dense head 但 MoSA 的 sparse attention 未被 FlashAttention 定制优化。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Dense attention O(T²) 的 FLOPs 和 KV-cache 随序列长度平方增长，造成训练和推理的巨大开销
  2. Fixed sparse attention 无法内容感知——固定稀疏模式对某些任务（如需要精确 retrieval）无效，且所有 head 共用同一选择 = 无 head 专业化
  3. Routing Transformer 的 online K-means 收敛慢、投影开销 T 级别（无法减至仅 k 个 token）、权重共享限制 expressiveness
  4. 现有稀疏方法在 IsoFLOP 比较下无法超越 dense baseline（论文实验证实 Fixed 和 Routing 均表现更差）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoSA 的核心理念：将 Expert-Choice Routing 的思想应用于 attention 机制——让每个 attention head 作为一个"专家"，从输入序列中学习选择自己需要处理的 k 个 token。这使得稀疏模式是**可学习、内容感知、head 专属**的。

  MoSA 全栈执行例子（以 Small model, ρ=32, k=32, hybrid: 4 dense heads + 381 MoSA heads, T=1024, 单 A100）：

  - **算法层**：
    1. 输入 X ∈ R^{1024×1024}，对每个 MoSA head i：
    2. Router 计算 r = σ(X @ W^r_i) ∈ R^1024（sigmoid 非竞争激活，遵照 σ-MoE 的发现）
    3. TopK(r, k=32) → r_topk ∈ R^32, 索引 I ∈ {0..1023}^32
    4. X^s = gather(X, I) ∈ R^{32×1024}——**仅对被选的 32 个 token 执行后续计算**
    5. Q/K/V = X^s @ W^Q/K/V_i ∈ R^{32×64}
    6. Causal mask M_{a,b}=0 if I_a≥I_b else -∞（保持了自回归约束）
    7. A = softmax(QK^T/√64 + M) @ V ∈ R^{32×64}
    8. X^o = diag(r_topk) @ A @ W^O_i——router score 乘到输出上，使路由梯度可反向传播
    9. Y = scatter(X^o, I) ∈ R^{1024×1024}——放回原位置，未选中位置填 0
    10. 最终输出 = Σ dense heads + Σ MoSA heads

    FLOPs: 4 dense heads × 0.805 GFLOPs + 381 MoSA heads × (8×1024×64×32 + 4×64×32² + 2×1024×1024 + 64×32) ≈ 3.22 + 381 × (16.78M + 0.26M + 2.10M + 0.002M) ≈ 3.22 + 381 × 19.14M ≈ 3.22 + 7.29 = 10.51 GFLOPs/层（vs dense baseline 7.25 GFLOPs/层的 9 dense heads, FLOP-matched）。

  - **kernel调度层**：纯 PyTorch einsum/scatter/gather 实现（无专用 CUDA kernel）。论文指出可结合 FlashAttention 加速 dense head 中的标准 attention 计算，并可开发专用 CUDA kernel 进一步加速 MoSA 的 sparse attention。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  MoSA 对 baseline 缺陷的解决（设计-缺陷映射）：

  1. **O(T²)→O(k²+T)**：MoSA 将每 head 的 Q/K/V/O 投影从 T 个 token 减至 k 个，attention 从 T×T 减至 k×k。例如 T=1024, k=32 时，投影成本降至 3.1%，attention 成本降至 0.1%。节省的 FLOPs 用于增加 head 数（从 9→385），实现更细粒度的专业化。

  2. **内容感知选 token → 解决 Fixed sparse 的内容无关性**：Router W^r 通过语言模型目标（cross-entropy loss）的梯度联合训练，直接学习哪些 token 对当前 head 最重要。这避免了 Fixed sparse 的固定 stride 选择，可以动态跳转到任意位置的关键 token。

  3. **Expert-Choice 完美负载均衡 → 解决 MoE routing collapse**：每个 head（专家）独立选择自己的 top-k token，天然保证每个 head 处理恰好 k 个 token。无需 auxiliary load-balancing loss，避免 token-choice routing 中的 expert collapse 问题。

  4. **仅对 k 个 token 做投影 → 解决 Routing Transformer 的 T 级投影开销**：Routing Transformer 在聚类前必须计算所有 token 的 Q/K，而 MoSA 先 router 选 token 再做投影，使投影成本正比于 k 而非 T。这使 MoSA 的 FLOP 成本约等于 Fixed sparse（内容感知却无额外计算开销）。

  5. **每个 head 独立权重 → 解决 Routing Transformer 的权重共享**：每个 MoSA head 有自己的 W^Q/W^K/W^V/W^O/W^r，head 间无共享，允许不同 head 专注于不同类型的 token 模式。384 个 head 各自学习独特的稀疏模式——这在 Routing Transformer 中不可行（因为单 head 内 ρ 个簇共享权重）。

  6. **混合架构（4 dense + M MoSA heads）→ 解决纯 MoSA 训练不稳定**：Router 和 attention weights 需联合学习，初期 router 随机选择导致 attention 学不到有用模式 → 恶性循环。4 个 dense head 提供稳定的全局信息流，稳定训练。实验证明 0 dense head 时性能崩溃（perplexity 从 22.46 升至 29.76 at ρ=16），4 dense head 时 -27% perplexity。

  7. **Perplexity-matched 资源节省**：即使无专用 CUDA kernel（仅 PyTorch），MoSA 在匹配相同 perplexity 时同步减少 wall-clock time（-2.1%~-12.9%）、GPU memory（-1.6%~-10.0%）和 KV-cache（-51.1%~-69.5%）。

## MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- baseline方法是什么？
  Baseline 是现有的免训练稀疏注意力方法和 full attention。全栈执行例子（以 LLaMA-3-8B-262K, 128K context, 单 A100 推理为例）：

  - **算法层（Full Attention baseline）**：标准 dense self-attention，pre-filling 阶段计算完整的 $A = \text{Softmax}(QK^T/\sqrt{d}) V$。对于 128K context 的输入，$Q, K, V \in \mathbb{R}^{131072 \times 128}$，$QK^T$ 矩阵大小为 $131072 \times 131072$，FLOPs 量级为 $O(S^2 d_h) = O(2.2 \times 10^{11})$。attention 计算占 pre-filling 总延迟的 >90%。对于 1M token 的 prompt，仅在单 A100 上 pre-filling 就需要 30 分钟。

  - **算法层（StreamingLLM baseline）**：固定保留 attention sink（初始 1K tokens）+ 滑动局部窗口（最近 4K tokens），丢弃中间的绝大部分 token。模式等同于论文中的 A-shape pattern（仅 static structured spatial distribution）。核心缺陷：(a) 无法处理需要在中间位置检索的动态信息（如 KV retrieval 任务中 PassKey 在 token 10000-50000 范围时完全失效，Retr.KV 得分从 Full Attention 的 14.4 降至 0.8）；(b) 对 Vertical-Slash 和 Block-Sparse 模式的注意力分配无法覆盖——例如 vertical lines（特定 token 被广泛 attend）和 slash lines（固定间隔的周期性注意力）不在 local window 范围内时会被丢弃。

  - **算法层（InfLLM baseline）**：使用 memory unit 处理流式长序列，128 global tokens + 8K local windows。与 StreamingLLM 类似，在 KV retrieval 等需要非局部信息的任务中表现不佳（Retr.KV=1.2 for LLaMA-3-262K），且有效 context window 仅 4K-8K。

  - **算法层（Ours w/ static baseline）**：在 Vertical-Slash 和 Block-Sparse heads 使用静态稀疏索引。在动态任务（如 KV retrieval）中性能崩溃（Retr.KV 接近 0），证明了 sparse indices 必须动态适配不同输入的必要性。

  - **kernel调度层**：FlashAttention（Triton 实现）——标准的 tiled dense attention kernel。对 $S \times S$ attention matrix 执行完整计算，无稀疏优化。虽然 FlashAttention 通过 tiling 和 recomputation 优化了 HBM 访问，但仍执行 O(S²) 的 FLOPs。

  - **系统框架层**：PyTorch + HuggingFace Transformers 标准推理 pipeline。原始 PyTorch LLaMA 实现在 prompt >50K tokens 时即触发单 A100 OOM。

  - **编译框架层/硬件架构层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. **静态模式失效**：StreamingLLM 的固定 A-shape pattern 无法覆盖 Vertical-Slash 和 Block-Sparse 模式的注意力分布，在 retrieval、multi-hop 等需要全局上下文的任务中准确率崩溃
  2. **动态性的双重挑战**：(a) attention 分布高度 dynamic（同一位置在不同 prompt 下 attend 的 token 完全不同，top-4K 列在另一 prompt 上 recall 从 96.8% 降至 83.7%）；(b) 但 attention pattern（A-shape/VS/BS 类型）在同一 head 上跨 prompt 保持 consistent
  3. **Top-K 方法 GPU 不友好**：直接 top-K 选择（fine-grained dynamic）在 GPU 上 latency 高，因为非结构化的稀疏索引导致不规则内存访问和低 tensor core 利用率
  4. **在线估计开销过大**：现有动态稀疏方法（如 SparQ Attention）使用 low-rank hidden states 估计注意力模式，开销过大，不适用于长上下文场景

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MInference 的核心洞察：attention heads 的稀疏性虽然动态变化，但其**空间分布模式（pattern type）在跨 prompt 时保持一致**——即每个 head 在不同输入下都表现为同一类模式（A-shape / Vertical-Slash / Block-Sparse），但具体哪些 token 被选中是动态变化的。基于此，MInference 将"模式识别"离线完成，"稀疏索引构建"在线完成，实现了低开销的动态稀疏注意力。

  MInference 的全栈执行例子（以 LLaMA-3-8B-262K, 128K context, 单 A100）：

  - **算法层（核心创新——三步动态稀疏注意力）**：
    1. **离线 Kernel-Aware Sparse Pattern Search → 解决 Baseline 缺陷 1（静态模式）和 缺陷 3（GPU 不友好）**：
       不是使用单一静态稀疏模式，而是对每个 attention head 在 kernel-aware 搜索空间中搜索最优模式（A-shape/VS/BS 之一）及其参数。关键设计：(a) "kernel-aware"——搜索空间中的 FLOPs 以真实 GPU kernel FLOPs 为准（非概念估计），确保搜索结果在 GPU 上有实际的加速效果；(b) 以 attention output recall（$\min |y_i - y|$，$y_i$ 是稀疏 attention 输出，$y$ 是 dense attention 输出）为优化目标，而非 attention score recall；(c) 仅需一条 30K tokens 的 reference sample，15 分钟即可完成搜索，且同一模型的不同 context 长度版本可复用。搜索结果显示 >90% 的 heads 被分配为 Vertical-Slash 模式。

    2. **在线动态稀疏索引近似 → 解决 Baseline 缺陷 2（动态性挑战）**：
       - **Vertical-Slash heads**：仅使用最后 64 个 query 向量（$Q_{[-64:]}$）与完整 K 矩阵计算近似注意力 $\hat{A}$，然后沿垂直和斜线方向求和，取 top-k 垂直列和斜线索引。开销极小（仅 $64 \times S$ 的 matmul，占 5-15% 时间），但准确估计了全 attention matrix 的垂直和斜线分布。
       - **Block-Sparse heads**：对 Q 和 K 做 block_size=64 的 mean pooling，然后在 block 级别计算注意力并取 top-k blocks。利用了 mean pooling 和 matmul 的交换性（$\text{MeanPool}(Q) \cdot \text{MeanPool}(K)^T \approx \text{MeanPool}(QK^T)$），以极少开销近似 block-level 注意力分布。
       - **A-shape heads**：静态稀疏 mask，零开销。

    3. **三种结构化的 GPU 友好稀疏模式 → 解决 Baseline 缺陷 3（GPU 不友好）**：
       放弃 fine-grained top-K 选择（GPU latency 高），改用三种结构化稀疏模式：(a) A-shape——block-level 的 structured static 模式，直接利用 FlashAttention block tiling；(b) Vertical-Slash——混合 block-level（斜线用 $64\times64$ blocks）+ column-level（垂直线用 $1\times64$ blocks），通过 point-range two-way merge 算法高效构建索引；(c) Block-Sparse——$64\times64$ block-level top-K 选择，Block-Sparse FlashAttention kernel 延迟与 block 数量线性相关。

  - **kernel调度层 → 解决 Baseline 缺陷 3（GPU 不友好）和 缺陷 4（在线估计开销）**：
    实现了三个高度优化的 GPU kernel（Triton + PIT + FlashAttention）：
    - **Block-Sparse FlashAttention kernel**：每个 thread block 循环遍历每行的 top-K blocks（64×64），延迟与 block 数量线性相关。1M context 下仅需计算 ~1% 的原始 FLOPs，kernel 级加速 30×。
    - **Vertical-Slash FlashAttention kernel**：混合 kernel——第一部分使用 Block-Sparse FlashAttention 处理斜线 blocks，第二部分使用 PIT（Permutation Invariant Transformation）sparse attention 处理垂直列。PIT 将非连续 column data 通过排列不变变换加载到 dense compute blocks，最大化 tensor core 利用率。
    - **A-shape kernel**：静态结构，使用 FlashAttention 仅计算固定区域的 attention。
    - 1M context 下 kernel 实际稀疏度 >95%（Block-Sparse 和 VS），理论加速 >15×。动态索引构建开销 <25%（大部分被稀疏计算节省的 FLOPs 所覆盖）。

  - **系统框架层**：
    基于 PyTorch + HuggingFace Transformers。做了三项单 A100 优化以支持 1M token 推理：(a) Tensor Splitting——按 head 拆分 Attention、按 sequence 拆分 MLP，保持 GPU 利用率 100%；(b) 消除中间变量——mask logic 在 kernel 内实现 causal mask；(c) 仅计算最后 token 的 LM Head logits（pre-filling 阶段只需最后 token 的 logits）。与 KV cache 压缩方法（SnapKV）兼容，可叠加使用。

  - **编译框架层**：PIT（Permutation Invariant Transformation）——动态稀疏编译器，将稀疏数据加载到 dense compute blocks。论文通过 PIT 实现 column-level sparse attention 的高效计算。论文未明确说明修改了哪些编译框架。

  - **硬件架构层**：NVIDIA A100 80GB GPU。论文未修改硬件架构。

  方法 vs Baseline 缺陷的对应关系：
  1. **静态模式失效** → 三种模式（A-shape/VS/BS）可覆盖所有 attention head 的稀疏分布特征，Pattern Search 确保每个 head 分配到最优模式
  2. **动态性挑战** → 在线动态稀疏索引近似（VS: last_64 query estimation, BS: mean pooling approximation），既捕获了动态性，又保持了低开销
  3. **GPU 不友好** → 结构化稀疏模式（block-level/column-level 而非 fine-grained per-token），三个定制 kernel 在 A100 上实现显著 speedup
  4. **在线估计开销** → 极简估计方法（VS 仅用 64 个 query，BS 仅用 mean pooling），总开销 5-25%，随 context 增长占比下降

## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- baseline方法是什么？
  Baseline 是现有的免训练稀疏注意力方法（TidalDecode、Quest、StreamingLLM）和需训练方法（SeerAttention-r）。这些方法的全栈执行例子如下：
  - **算法层**：TidalDecode 采用 per-head 独立 token 选择——每个 attention head 基于各自的 attention scores 独立选择 top-k token 子集，在不同层间进行周期性重新选择，各 head 维护独立的 token 索引。Quest 使用 hybrid attention layers 和 chunk-based block size（16/32），在所有层应用稀疏 attention。StreamingLLM 保留 attention sink（初始 token）+ 固定大小的 sliding window（最近 token），丢弃中间 token。这些方法的核心缺陷：(a) per-head 独立选择假设 attention heads 需要完全不同的 token 子集，但推理中 token 重要性实质上高度跨 head 重叠（cross-head spatial locality），导致选择效率低下和 selection error 的 head 间不一致传播；(b) 局部选择策略（per-head / per-layer / per-step）在成千上万个 decoding step 中产生累积误差——attention recall 从 ~95% 逐步退化至 ~65%（图 1a），导致推理链逻辑不一致和生成长度膨胀（表 2：TidalDecode 2K budget 生成 17.4K vs Full Attention 14.8K）；(c) StreamingLLM 的固定大小滑动窗口不支持按 token budget 比例适配不同近邻需求。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，采用 FlashInfer attention kernel。TidalDecode 需要 per-head token 索引管理，Quest 需要 block-based sparse attention mask。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashInfer attention kernel。TidalDecode/Quest 在 GQA 模型下需要 per-head 独立的 KV cache 子集加载——同一 KV group 的不同 query head 选择不同 token 集合时，需加载更多 token 的 KV，导致 global-to-shared memory 传输冗余（TidalDecode: 2.34MB vs LessIsMore: 1.04MB, Table 4）。
  - **硬件架构层**：NVIDIA A100 80GB / A5000 GPU，无专用硬件修改。

  Baseline 核心缺陷总结：
  1. **假设错误**：假设 token 重要性是 head-local 属性，忽略了推理中跨 head 空间局部性和时间近邻局部性
  2. **误差累积**：局部最优的 per-head/per-layer 选择在长程 decoding 中误差循环累积，导致 attention recall 退化、推理链不一致和生成长度膨胀
  3. **KV loading 冗余**：GQA 下 per-head 独立选择导致 KV cache 加载冗余

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LessIsMore 基于一个核心洞察重新设计稀疏注意力：推理模型中 token 重要性是全局属性而非 head-local 属性，由此直接推导出两个设计需求——(a) token 选择必须跨 head 全局一致，(b) token 选择必须跨层稳定且显式保留近邻上下文。

  LessIsMore 的全栈执行例子：
  - **算法层（核心创新——Cross-Head Unified Sparse Attention, CUSA）**：
    1. **跨 head 统一 token 选择 → 解决 Baseline 缺陷 1（head-local 假设）**：
       各 attention head 独立提案 top-k 候选 token（基于精确 attention score P = q·C.K^T），但通过 UnionFlatten 将所有 head 的候选聚合为统一候选集，全局排名后取 top K·(1-r)。关键设计：(a) 不假设 head 功能完全一致，而是利用观察到的 token 重要性跨 head 重叠；(b) 聚合步骤通过多数 head 赞同的方式消除个别 head 的噪声选择，降低 selection variance；(c) 同一 KV group 内所有 query head 共享最终 token 索引 ρ，消除 per-head 独立选择带来的 KV loading 冗余。

    2. **稳定近邻保留 → 解决 Baseline 缺陷 1 的 recency 部分**：
       固定比例 r=0.25 的 token budget 分配给最近 K·r 个 token（而非 baseline 的固定大小 sliding window）。此设计直接源于观察：近邻 token 占总关键 token 的比例在 decoding 全程保持稳定，因此比例性分配比固定窗口更好地适应不同 token budget 和序列长度。

    3. **低频 token 重选 → 解决 Baseline 缺陷 2（误差累积）**：
       Token 选择仅在一层（token selection layer，如 Layer 12）执行，产生的统一索引 ρ 跨后续所有 Sparse Attention Layers 复用。图 4 验证：CUSA 仅在 Layer 2 选择 vs 每层都选择，attention recall 几乎无差异（~95% vs ~96%），而 per-head 方法从 ~96% 降至 ~65%。原因：CUSA 的全局 token 重要性跨层高度稳定（由 cross-head spatial locality + temporal recency locality 共同保证），局部方法的高频重选反而引入更多 noise。
  - **系统框架层**：集成到 SGLang + FlashInfer。三种 layer 类型（Full Attention / Token Selection / Sparse Attention）分层执行，token selection layer 更新 ρ，sparse attention layers 复用，无需 per-head 独立 token 索引管理。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：定制化 FlashInfer kernel 利用统一 token 索引实现更高效的 GQA 稀疏 attention。全 query heads 共享 K[ρ]/V[ρ]（仅 K 个 token），单次 global-to-shared memory 加载，减少 G2S 传输 55%（1.04MB vs 2.34MB for TidalDecode），kernel latency 从 32.1µs 降至 20.1µs（1.6× speedup）。
  - **硬件架构层**：NVIDIA A100 80GB / A5000 GPU，无专用硬件修改。

  **对比 baseline 的关键差异**：
  - **Baseline**（TidalDecode）：per-head 独立选择 → 每个 head 维护独立的 token 索引 → GQA 下同一 KV group 的不同 query head 选择不同 token → KV loading 冗余（2.34MB G2S）→ attention recall 退化（~75% at 32K）→ 生成长度膨胀（17.4K vs 14.8K）→ kernel latency 32.1µs
  - **LessIsMore**：跨 head 统一选择 → 全局统一 token 索引 ρ → GQA 下所有 query head 共享 ρ → KV loading 最优（1.04MB G2S）→ attention recall 稳定（~90% at 32K vs ~75% TidalDecode, 图 1a）→ 生成长度几乎无膨胀（15.8K vs 14.8K, 表 2）→ kernel latency 20.1µs

  **效果量化**：
  - AIME-24 on Qwen3-8B: 2K budget 达 73.8% 准确率（vs Full Attention 74.5%, TidalDecode 53.3%, Quest 18.2%）
  - 87.5% sparsity 零精度损失
  - 端到端 decode speedup up to 1.6×（64K context, Figure 6a）
  - Kernel 级 sparse attention speedup up to 1.72× vs TidalDecode（Figure 6b）
  - LongBench 4K token budget avg F1 44.78（vs Full Attention 44.08, TidalDecode 44.56, Quest 42.62）
  - MHA 模型 LongChat-7B-32k on NIAH: 256-token budget 达 100%（vs TidalDecode 99%, Quest 99%, H2O 1%, StreamingLLM 3%）

## LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

- baseline方法是什么？
  Baseline 包括 Ring Attention、Megatron-SP（针对标准 attention 的 SP）和 LASP-1（专门针对线性注意力的 SP）。

  Ring Attention / Megatron-SP / LASP-1 的全栈执行例子：
  - **算法层**：Ring Attention 将序列切分后使用 ring-style P2P 通信逐设备传递 K、V blocks，每个设备在收到相邻设备的 KV block 后计算局部 attention 输出。Megatron-SP 在标准 attention 上实现类似的 ring-style 通信重叠。LASP-1 针对线性注意力做了定制化：使用 ring-style P2P 通信在各设备间顺序传递 memory state M_t（d×d 大小），每步（共 W-1 步）执行一次 send & receive → 计算 O_{t,inter} → 更新 M_t 的顺序操作。
  - **系统框架层**：Ring Attention / LASP-1 基于 Megatron-Core，使用 NCCL P2P send/recv 原语，ring 拓扑按 rank 排列。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Ring Attention 使用 FlashAttention-2 kernel 做局部 attention；LASP-1 使用 Triton 加速线性注意力计算。P2P send/recv 通信需逐个 launch 大量小算子。
  - **硬件架构层**：DGX-A100 集群（NVSwitch 600 GB/s 互联），无专用硬件修改。

  Baseline 核心缺陷：
  1. **Ring-style 通信导致计算并行度低**：LASP-1 的 ring-style P2P 通信需要按顺序从 rank i-1 接收 M_{t-1} → 计算 O_t → 更新 M_t → 发送 M_t 到 rank i+1，这 W-1 步完全串行，导致后续设备大量空闲等待。
  2. **通信-计算 overlap 困难**：大量细粒度 P2P send/recv 算子使得通信与计算的重叠调度复杂且低效，实际 overlap 程度远低于理论值。
  3. **通信步骤过多**：每 iteration 共 2(W-1) 个通信步骤（forward W-1 + backward W-1），随设备数线性增长，在大规模集群中通信开销显著。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LASP-2 通过重新设计通信-计算工作流，用单次 AllGather 集合通信替代 ring-style P2P。

  LASP-2 的全栈执行例子：
  - **算法层**：将序列切分后各设备并行计算 Q_t, K_t, V_t 和 local M_t = K_t^T V_t，然后通过**单次 AllGather 集体通信**将所有 M_t 同步到所有设备（而非依次传递）。各设备本地累加 M_{1:T} = Sum([M_t]_1^T)，再本地计算 O_t = Q_t M_{1:T}。通信量仍为 BHd^2，但通信步骤从 2(W-1) 降至 2（forward 1 + backward 1）。对有 causal mask 的情况，采用计算分解：intra-chunk 保持 quadratic 左乘，inter-chunk 用线性右乘，且 AllGather 与 intra-chunk 计算可通过不同 CUDA stream overlap。
  - **系统框架层**：基于 Megatron-Core 0.9.0，使用 NCCL AllGather 集体通信原语替代 P2P send/recv，通信组基于 SP group。支持与 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP/ZeRO/FSDP) 混合使用。LASP-2H 对混合模型的 standard attention 层使用 AllGather K_t, V_t 的 Context Parallelism。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Triton 2.3.1 加速 GPU 上的线性注意力计算（chunked intra-chunk attention）。FlashAttention-2 用于标准 attention。AllGather 为 NCCL 高度优化的集体通信算子，相比 P2P 更易于与计算 overlap（单次大粒度通信 vs 多次小粒度通信）。
  - **硬件架构层**：DGX-A100 集群，无专用硬件修改。

  **对应解决 Baseline 缺陷的具体设计**：

  1. **单次 AllGather 替代 Ring P2P → 解决串行依赖**：LASP-1 需要 rank i 等待 rank i-1 完成并发送 M_{t-1} 后才能开始计算，形成严格的串行链。LASP-2 通过 AllGather 一次性地将所有设备的 M_t 并发同步到所有设备，消除了逐设备传递的串行依赖。通信步骤从 2(W-1) 降至 2，计算并行度从"逐个设备串行"变为"全部设备并行"。

  2. **通信粒度从细粒度 P2P 变为单次大粒度集体通信 → 解决 overlap 困难**：LASP-1 需要 launch W-1 次小粒度的 send/recv 算子对，调度复杂且有大量 kernel launch 开销。LASP-2 仅需 1 次 AllGather，通信粒度大、调度简单，在有 mask 的场景下可直接与 intra-chunk 计算 overlap（不同 CUDA stream 并发执行 line 7 AllGather 和 line 8 intra-chunk 计算）。

  3. **Memory state 通信量与序列长度无关 → 长序列场景优势放大**：M_t ∈ R^{d×d} 的大小仅取决于 hidden dim，与 chunk/sequence 长度无关。在序列长度 2048K 时，通信数据量不变，但计算量（intra-chunk quadratic 部分）随 chunk 大小增长，因此通信-计算比进一步降低，LASP-2 的优势更加显著。

  4. **实际效果**：在 64 A100 GPU、序列长度 2048K 上，LASP-2 比 Ring Attention 快 36.6%，比 LASP-1 快 15.2%。序列长度 ≥64K 时优势开始显现，序列越长越显著。LASP-2 支持线性扩展：每 GPU 内存使用恒定（~25.6-57.8 GB）下，增加 GPU 数量即可扩展支持更长序列（如 128 GPU 支持 2048K）。

- baseline方法是什么？
  Baseline 是 query-aware KV cache 淘汰方法，包括 SnapKV、PyramidKV、H2O。
  
  SnapKV / PyramidKV / H2O 的全栈执行例子：
  - **算法层**：在 prefill 阶段利用 trailing context window 中的 query token 计算 attention-based 重要性分数（SnapKV: max pooling over observation window; PyramidKV: pyramidal layer-budget; H2O: cumulative attention scores during prefill），选择性地保留与当前 query 相关的 KV pairs。核心假设：对当前 query 重要的 KV pairs 对后续也重要。
  - **系统框架层**：论文未明确说明特定 serving 框架，方法可集成到任意支持 KV cache 的推理框架（HuggingFace Transformers、vLLM 等）。使用 FlashAttention-2 加速注意力计算。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：使用 FlashAttention-2 kernel。SnapKV 使用 max pooling kernel_size=7 对注意力分数平滑。论文未明确说明其他自定义 kernel。
  - **硬件架构层**：在 NVIDIA A100 80GB GPU 上运行，无专用硬件修改。
  
  Baseline 核心缺陷：
  1. **Query 过拟合**：在 prefill 时依赖当前 query 信息决定 KV pair 保留策略，压缩后的 KV cache 对初始 query 过拟合。在多查询场景下，复用该压缩 cache 处理不同 query 时性能显著下降（Figure 2：SnapKV 在 SQuAD multi-QA 中复用压缩 cache，准确率大幅衰减）。
  2. **重复 prefill 开销**：若每个 query 独立执行 prefill + evict（Figure 1a），则每个 query 都需要完整 prefill 计算，总开销随查询数量线性增长。
  3. **Self-attention 稀疏性不匹配**：H2O 使用 prefill 阶段的 self-attention scores 作为重要性指标，但 prefill 阶段的 self-attention 模式比 cross-attention 更密集（Figure 5），且与下游任务 attention 模式重叠度低（Figure 13e），导致无法有效识别冗余 KV pairs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KVzip 提出 query-agnostic KV cache 淘汰策略，通过上下文重建（context reconstruction）评估 KV pair 重要性。
  
  KVzip 的全栈执行例子：
  - **算法层**：将 "Repeat the previous context:" prompt + 原始 context chunk 拼接后通过 LLM forward pass，模拟 teacher-forced decoding 重建上下文。对每个 KV pair 取其在重建过程中收到的最大 cross-attention score 作为重要性分数 S ∈ R^{L×H×n_c}。保留 top r% 高分 KV pairs，淘汰其余。核心 insight：(1) Transformer 天然作为 encoder-decoder——将 context 编码进 KV pairs（类比 Zip 压缩）；(2) 重建上下文所需的关键 KV pairs，恰好也是 QA、摘要、推理等多种下游任务所需（Figure 6 2D histogram 显示下三角区域集中，即重建高分 KV pairs 在各任务中也高分）；(3) 基于重建的自监督 proxy task 能泛化到多种下游任务（类似 BERT/MAE 范式）。Chunked scoring 将复杂度从 O(n_c²) 降至 O(m·n_c)，m=2K 固定。
  - **系统框架层**：与 FlashAttention-2 集成。non-uniform head-budget allocation（跨所有 head 取 top r% 而非 per-head 均匀分配）。支持两种模式：(a) context-dependent eviction——per-context 压缩，高压缩比（可低至 30% budget）但有一次 ~2× prefill 的压缩开销；(b) context-independent eviction——预计算 head-level score S_head ∈ R^{L×H}（单次 88K-token 样本），部署时零开销，应用 DuoAttention 的 head-level KV eviction 策略，显著优于 DuoAttention 原生 head-score（KVzip 用 1 GPU 一分钟 vs DuoAttention 需 8 GPU 数小时优化）。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：主要使用 FlashAttention-2。chunked scoring 在 FlashAttention 中引入 key subsampling（仅取当前 chunk 对应的 keys）。附录 C.3 提出 softmax-free 变体，通过定制 Triton-based CUDA kernel 将评分嵌入 fused attention kernel，消除 ~10% 评分开销（代价是压缩比下降 ~10%）。与 QServe W8A8KV4 量化无缝集成。
  - **硬件架构层**：在 NVIDIA A100 80GB GPU 上运行，无专用硬件修改。
  
  **对应解决 Baseline 缺陷的具体设计**：
  
  1. **Query-agnostic 评分 → 解决 Query 过拟合**：重要性评分不依赖任何 query，仅基于 context 自身的重建能力。压缩后的 KV cache 可跨任意 query 复用（Figure 1c），无需重复 prefill。实验证明（Figure 2），KVzip 在单次 prefill + 多 query 场景下性能稳定，而 SnapKV 复用压缩 cache 时性能显著退化。
  
  2. **Cross-attention 稀疏性 → 解决 H2O self-attention 密度问题**：上下文重建过程中的 cross-attention 比 prefill self-attention 显著稀疏（Figure 5 直方图对比），因为模型可以高效利用 KV_c 中的高层表示 + 自身权重中的知识，减少不必要的注意力查找。这种稀疏性使 KVzip 能更精准地识别可淘汰的冗余 KV pairs。
  
  3. **重建驱动的评分原理 → 保证多任务泛化**：实验证明重建所需的 KV pairs 与 QA、摘要、推理等下游任务的注意力模式高度重叠（Figure 6 前三张 2D histogram 的 lower-right triangular region），而不同 QA 任务之间的注意力模式却呈现 query-specific 差异（第四张 heatmap 沿 x/y 轴分散）。这表明重建作为一个通用的 proxy task，能够捕获跨任务的通用关键信息。
  
  4. **实际效果**：Baseline 方法在 90% cache budget（仅淘汰 10%）时即出现性能退化，KVzip 在 30% budget（淘汰 70%）下仍保持接近无损性能。FlashAttention 解码延迟降低约 2×，KV cache 大小减少 3-4×。结合 4-bit 量化后，16-bit 124K-token KV cache 从 16.3GB 降至 1.2GB。

- baseline方法是什么？
  Baseline 是 Full Attention（完整 KV cache）+ 现有 KV cache 压缩方法（H2O、TOVA、StreamingLLM、FastGen），其全栈执行例子如下：
  - **算法层**：Full Attention 对所有 token pair 计算 O(n²) attention，KV cache 随序列长度线性增长（BF16 下 Llama-3-8B 处理 1M tokens 需要 ~137 GB KV cache 仅此项就超出单卡 80GB 容量）。H2O 基于累积 attention scores 识别 heavy-hitter token 保留在 KV cache 中，TOVA 基于 attention scores 贪心 evict 不重要的 token，StreamingLLM 保留初始 token（attention sink）+ 最近 token 的 sliding window。这些方法的核心缺陷：(a) 不分 head 类型差异——对所有 attention head 使用相同的压缩策略，抹杀了不同 head 的功能异质性（retrieval vs streaming）；(b) 仅依赖 attention scores 做逐 token eviction，忽略了 value states 的影响和跨层跨 head 的 attention 分布差异；(c) 在长上下文 benchmark（NIAH/LongBench）上严重退化——H2O/TOVA/StreamingLLM 在 NIAH 上几乎完全失败（无法在不同序列深度正确检索），因为它们在 pre-filling 阶段需要 materialized attention scores 但 FlashAttention 不物化这些 scores，导致 pre-filling 阶段无法 evict tokens、造成 OOM；(d) 无法降低 pre-filling 的计算和内存开销——这些方法仅在 decoding 阶段减少 KV cache，pre-filling 仍是 full computation。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，在 prefill 后/decoding 中执行 KV cache eviction。H2O/TOVA 需修改以兼容 FlashAttention（prefilling 用 exact attention，仅 decoding 阶段 evict）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention。H2O/TOVA token eviction 在 GPU 上执行 TopK + index gather 操作。
  - **硬件架构层**：NVIDIA A100 GPU（80GB）。长上下文（≥128K）时 baseline 方法由于 KV cache 爆炸导致 OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DuoAttention 通过三个核心设计解决 baseline 缺陷：

  **1. Head 功能二分（Retrieval vs Streaming Heads）→ 解决缺陷(a)**：
  观察到 attention heads 呈现两种不同功能模式：Retrieval Heads（少数）关注跨长距离的语义相关 token，对长上下文处理至关重要；Streaming Heads（多数）主要关注 attention sink（首 token）和最近 token，不依赖完整历史。利用这一观察，为两类 head 分配不同的 KV cache 策略，而非 baseline 的统一处理。

  **2. 基于优化的 Retrieval Head 识别（优化-based + 合成数据）→ 解决缺陷(b)**：
  不再依赖 attention scores profiling（如 FastGen、RazorAttention 所用），而是直接测量输出偏差——当 KV cache 压缩为仅 sink+recent 时导致输出偏差显著增大的 head 即为 retrieval head。用可训练 gate value α_{i,j} 混合 full 和 streaming attention 输出，在合成 passkey retrieval 数据集上以 L2 distillation loss + L1 regularization 端到端优化。合成数据确保每个监督信号都与最终压缩策略相关（passkey recall 需要长上下文能力），优于 natural language modeling（自然文本中跨长距离的监督信号稀疏）。与 attention profiling 相比：直接测量 output deviation 能捕捉 attention scores 上看不到的 retrieval heads、考虑 value states 的影响、以及跨层跨 head 的分布差异。

  **3. Chunked Pre-filling 中的 streaming head 优化 → 解决缺陷(d)**：
  Streaming heads 的 pre-filling 计算中，每个 chunk 的 KV 计算完毕后立即 prune 仅保留 sink+recent tokens，下一 chunk 仅需 attend 到 constant number 的历史 token。Pre-filling 复杂度从 O(L²) 降至 O(LK)，memory 从 O(L) 降至 O(K)。

  **全栈执行例子（DuoAttention on Llama-2-7B-32K-Instruct, 25% retrieval ratio, 1×A100）**：
  - **算法层（核心创新）**：
    (a) Offline Phase：8×A100 上 2,000 steps gate value 训练（仅数千参数，模型权重冻结）→ synthetic passkey dataset（BookSum + 10×32-word passkeys）→ L2 distillation loss on last hidden states + L1 regularization (λ=0.05) → AdamW (lr=0.02 warmup→decay)。
    (b) Binarization：按 sparsity quantile τ 将 α_{i,j} 二值化为 {retrieval, streaming}，head 重排 Q/K/V 权重使两类连续。
    (c) Decoding：retrieval heads → full KV cache (all tokens) + FlashAttention；streaming heads → constant KV cache (64 sink + 256 recent) + streaming mask attention。
    (d) Chunked Pre-filling：chunk_size=32K，每 chunk 后 streaming head 的 KV cache 立即 prune → 下一 chunk 仅 attend 到 O(K) 而非 O(L) tokens。
  - **系统框架层**：基于 PyTorch + FlashInfer (RoPE/RMSNorm kernels) + FlashAttention-2。支持 chunked pre-filling，与 GQA 完全兼容（per KV group gate value）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 执行 retrieval heads 的 full attention 和 streaming heads 的 constant-length attention。Chunked pre-filling 中 streaming heads 的 attention 利用标准 FlashAttention kernel（仅 mask 改变为 Λ-like pattern），无需特殊 kernel。与 FlashInfer 的 RoPE/RMSNorm kernel 配合使用。
  - **硬件架构层**：单 NVIDIA A100-80G GPU。DuoAttention + QServe (W8A8KV4 quantization) → Llama-3-8B 容纳 3.3M contextual tokens（6.4× capacity vs full attention BF16）。

  **对比 baseline 的关键差异**：
  - Baseline 统一处理所有 head → DuoAttention 区分 retrieval/streaming，仅 retrieval heads 保留 full KV cache，retrieval ratio 25%（MHA）/ 50%（GQA）即保持 accuracy，其余 memory 大幅减少
  - Baseline 依赖 attention scores eviction（H2O/TOVA/FastGen）; DuoAttention 用优化-based 输出偏差方法识别，更准确（ablation 图 13(1) 证明优于 attention profiling 和 language modeling）
  - Baseline 在 NIAH 上完全失败（Figure 6: H2O/TOVA/StreamingLLM 在不同深度无法检索）; DuoAttention 在所有深度保持接近 full attention 的性能（因 retrieval heads 保留完整 KV cache）
  - Baseline pre-filling 无优化 → DuoAttention streaming heads pre-filling O(LK) 时间 + O(K) 内存（vs baseline O(L²) + O(L)）
  - MHA 模型 memory reduction up to 2.55×, latency reduction up to 2.18×（decoding）和 1.73×（pre-filling）
  - GQA 模型 memory reduction up to 1.67×, latency reduction up to 1.50×（decoding）和 1.63×（pre-filling）
  - GQA 模型的 retrieval head ratio (50%) 高于 MHA (25%)，因为 GQA 中 per-group gate value 绑定多个 query head，必须保守压缩; MHA 中每个 head 独立 gate，压缩更激进

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

## A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

- baseline方法是什么？
  Baseline 方法为 H2O（Heavy-Hitter Oracle），其核心是基于 Accumulative Attention Score (A2S) 的 token 剪枝技术。在 Decoder 模型中，A2S 沿 Generation Step 累积每个 token 的 Attention Score：A_{n,k}^{l,h} = Σ_{q=k}^{n} S_{q,k}^{l,h}（公式 4）。该方法的缺陷在于：由于 Causal Mask 导致早期 token 的 Attention Score 累积次数远多于近期 token（第 k 个 token 累积 n-k 次，第 k+10 个 token 累积 n-k-10 次），使得早期生成但实际不重要的 token 获得虚高的 A2S 分数，导致不应被剪枝的重要近期 token 被错误剪除。

  全栈执行例子（H2O baseline）：
  - 算法pipeline：每层每头沿 generation step 直接累加 Softmax 输出的 Attention Score，无时间衰减；cache ratio 一半用于 Local Attention（保留最近 token），一半用于 A2S-based selective eviction
  - 系统框架：论文未明确说明（H2O 可集成到 HuggingFace Transformers 推理流程中）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：在 RTX 3090 GPU 上运行 FP16 推理

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 A2SF 在 A2S 累积过程中引入 Forgetting Factor α（0 < α < 1），每次新生成 token 时，所有历史的 Attention Score 乘以 α 后再累加：A_{n,k}^h = Σ_{q=1}^{n} α^{n-q} × S_{q,k}^h（公式 5）。该设计直接回应 Baseline 的缺陷：

  1. **解决累积次数不平衡**：通过 α^{n-q} 指数衰减，早期 token 虽然累积次数多，但每次累加的权重以 α 的幂次递减趋近于 0，使得早期和近期 token 的有效累积总量趋于公平。
  2. **保持 Attention Sink**：即使施加遗忘因子，Attention Sink token（首 token）因每步都产生极大 Attention Score，衰减后仍保持高分，不会被误删。
  3. **可调节的历史依赖**：α 值可调节模型对历史的依赖程度——α→0 仅看最近趋势，α→1 等价于原始 A2S。
  4. **全部预算用于选择性剪枝**：由于 A2SF 天然关注近期趋势，不再需要 Local Attention 分走一半缓存预算。

  全栈执行例子（A2SF）：
  - 算法pipeline：每层每头沿 generation step 按 A_{n,k}^h = Σ α^{n-q} × S_{q,k}^h 累积带衰减的 Attention Score；全部 cache budget 用于 selective eviction（无 local cache 分配）；最优 α ∈ [0.1, 0.3]
  - 系统框架：与 H2O 相同，即插即用式集成到 HuggingFace Transformers 推理流程，无需额外训练，与 No Token Left Behind（量化不重要token）、Get More with LESS（低秩分解）、Keyformer（Gumbel-Softmax）等技术兼容
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：在 RTX 3090 GPU 上运行 FP16 推理

  **核心创新**：从 "累积次数越多的 token 越重要" 的错误假设，转变为 "近期被关注的 token 更可能当前重要"——以指数遗忘因子的简单乘法实现公平比较，无需额外训练或复杂结构。

## AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

- baseline方法是什么？
  Baseline 是已有的 layer-wise skipping 策略，可分为三类：(1) **Early Skipping (SkipDecode)**：始终跳过模型前几层（除第一层），策略固定——无论模型和上下文如何变化都跳同样位置；(2) **Periodic Skipping (Unified Skipping)**：在中间层按固定频率周期性地跳层（每几层跳一层），策略同样固定；(3) **Early Exit**：在每层计算后判断条件（如置信度），一旦满足条件立即退出，跳过后续所有层。这些 baseline 有三个核心缺陷：(a) 跳层位置固定，忽略了不同模型和上下文中 layer importance 分布的巨大差异；(b) 按整层（Transformer Layer）粒度跳过，忽略了 Attention sublayer 和 FFN sublayer 有独立的重要性分布——Attention 在长上下文中通常有更高 IO Similarity（输出更接近输入），意味着 attention 可以被更多地被跳过，且跳过 attention 能节省更多 KV cache；(c) 所有 baseline 仅针对 decoding 阶段设计，无法优化 prefilling 阶段的 TTFT 和 KV cache 存储。

  全栈执行例子（Baseline / Unified Skipping in long-context inference on LLaMA3.1-8B-128k）：
  - 算法pipeline：每 N 层跳 1 层（固定频率），按整层跳过（同时跳过 attention + FFN），跳层位置不随模型/上下文改变。例如在 32 层模型中，跳过 4 层则每 8 层跳 1 层的 attention + FFN 两个 sublayer。跳层后输出由残差连接直接传递
  - 系统框架：即插即用到 HuggingFace Transformers 推理流程中，在指定层插入 skip（identity shortcut）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：单张 NVIDIA L20 GPU, CUDA 12.1

- 论文方法是什么？如何对应解决Baseline的缺陷？
  AdaSkip 提出 training-free、自适应的 sublayer-wise skipping，三项设计分别对应 baseline 的三个缺陷：

  **1. 自适应（Adaptive）→ 解决模型间 importance 差异（Observation 1）**：
  通过 Offline Importance Learning 从历史推理中学习每个模型的 IO Similarity 分布，而非使用固定规则。因不同模型（LLaMA vs InternLM vs Vicuna）的 layer importance 分布差异很大（如 InternLM 高 IO Similarity 层在中部，LLaMA 在尾部），必须为每个模型单独学习。Offline 学习的特征在不同数据集间有高 hit rate（跨数据集 top-10 hit rate 9.31-9.90），说明该特征具有泛化性。

  **2. Sublayer-wise Skipping → 解决整层跳过的次优性（Observation 2）**：
  独立评估 Attention 和 FFN 两个 sublayer 的 IO Similarity。Attention sublayer 的 IO Similarity 在长上下文中平均更高且更集中（如 LLaMA3.1-8B-128k 最后 11 层 attention 平均 Similarity ~0.97，FFN 仅 ~0.95），说明更多 attention sublayer 可被跳过，且跳过 attention 还能节省 KV cache。AdaSkip 按 sublayer 粒度（而非 layer 粒度）排序并选择，每次 skip 可能是 attention 也可能是 FFN，更细粒度地匹配实际的 importance 分布。

  **3. Prefilling + Decoding 双阶段支持 → 解决仅 decoding 优化的局限（Observation 3）**：
  - Prefilling 阶段：使用 Offline Importance Learning（历史 IO Similarity + Scale Factor 补偿）确定 skip set，因为 prefilling 前没有可用的 IO 信息
  - Decoding 阶段：复用 prefill 的 skip set + 额外跳过 FFN sublayer——利用前 P 个 token 的 online learning window 计算当前上下文的 IO Similarity，通过阈值 β（skip set 中最小的 Similarity）筛选出当前上下文中同样高 IO Similarity 的额外 FFN sublayer（Observation 3 发现 FFN 在 decoding 阶段 IO Similarity 高于 prefill 阶段，有更多跳过机会）

  全栈执行例子（AdaSkip on LLaMA3.1-8B-128k, α=1.14, skip 8 sublayers）：
  - 算法pipeline：
    1. Offline phase：在历史数据集（TriviaQA/MFieldQA/Wiki 等）上跑 prefill，累积各 sublayer 的 Simi_j 和 Scale_j，按 Simi_j 降序排 sorted list
    2. Prefilling phase：根据 α 确定跳过 2m 个 sublayer，取 sorted[0:2m] 为 skipped set；inference 时遇到这些 sublayer 即 skip（identity shortcut），用 Scale_j * a 补偿
    3. Decoding phase：前 P 个 token 全 sublayer 执行（online learning window）→ 计算当前 Simi_j^P → 用阈值 β 筛选额外 FFN sublayer → 合并 skipped^P → 后续 token 跳过 skipped^P 中的 sublayer；同样用 Scale_j 补偿
    4. 在每个 Transformer layer 中独立判断：该层 Attention sublayer ∈ skipped? skip. 该层 FFN sublayer ∈ skipped^P? skip.
  - 系统框架：即插即用到 HuggingFace Transformers，无需训练/微调模型参数。可配合 batching 使用，与 KV cache compression 方法（H2O, SnapKV, PyramidKV）正交互补
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：单张 NVIDIA L20 GPU, CUDA 12.1

  关键结果（End-to-End, skip 16 sublayers, LLaMA3.1-8B-128k）：
  AdaSkip GovReport Rouge-L 18.9 / MultiNews 17.8
  vs Early Exit 4.3/4.4, SkipDecode 0.0/0.0, Unified Skipping 0.0/0.1
  证明自适应 sublayer-wise skipping 在 prefill+decode 双阶段跳层时维持生成质量的能力远超固定层跳过策略。

## APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

- baseline方法是什么？
  Baseline 方法是现有的长上下文推理加速方案，可分为两类：(1) 序列并行方法（RINGATTN、ULYSSES）——通过将长序列分布到多个 GPU 并行加速 attention 计算，但保持精确 attention（FULLATTN），计算量不变；(2) 近似注意力方法（MINFERENCE）——通过稀疏 attention pattern 减少单 GPU 计算量，但缺乏序列并行支持，长输入下扩展性差。STARATTN 首次合并两者，通过在每 host 上 prepend 一个与 local block 等大的 anchor block 并取消通信实现近似分布式注意力。Baseline 的核心缺陷：
  - Challenge 1: Localized Attention Pruning——现有近似注意力方法（H2O、SNAPKV）依赖全局序列的 attention score 来剪枝 KV cache，这与序列并行中各 host 仅持有部分上下文的架构冲突；
  - Challenge 2: Multi-host Scalability——序列并行受限于 attention head 数量（ULYSSES 的 head-splitting 方式），且 STARATTN 随 host 数量增加导致大量 middle context 不可见，性能持续退化；
  - STARATTN 的额外开销：anchor block = local block size（即 l_a = l_b），导致 FFN 中 anchor block 的计算开销过大，限制了加速收益。

  全栈执行例子（Baseline / STARATTN on 8 hosts, 128K input）：
  - 算法pipeline：每 host 持有 l_b=16K 的 local block + l_a=16K 的 anchor block（文档首 16K token 在所有 host 上复制）；仅计算 anchor↔local 之间的 attention，无跨 host 通信；passing block 不处理
  - 系统框架：基于 FLASHATTN kernel 的 HuggingFace Transformers 推理，分布式执行（8-GPU per node）
  - 编译框架：论文未明确说明
  - kernel调度：FLASHATTN kernel（标准 attention mask），无通信调度
  - 硬件架构：8× NVIDIA A800-80GB, NVLink 3.0 + HDR InfiniBand

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 APB，通过三项核心设计解决 baseline 缺陷：

  1. Localized KV Cache Compression（解决 Challenge 1）：使用 LOCRET 的 retaining heads R（小型训练 MLP），在每个 host 上独立对 local KV cache 打分，无需全局序列视图。取 Top-l_p 作为压缩后的 passing block。该设计与序列并行的分布式架构完全兼容。

  2. Compressed Passing Blocks（解决 Challenge 2 的 STARATTN 退化问题）：通过 AllGather 只共享压缩后的 Top-l_p KV pair（l_p << l_b），构造 passing block P_h 作为前序 host 的"关键信息摘要"。即使 host 数量增加，每个 host 仍能通过 passing blocks 获取前序上下文的精华，维持跨 host 的长距离依赖。消融实验（Table 3）证明：移除 passing block 导致 E.MC 从 72 降至 64（-8%），移除 anchor block 则导致任务完全失败（降至 28）。

  3. Smaller Anchor Blocks（解决 STARATTN 的 FFN 开销问题）：APB 使用 l_a = l_b/4 或 l_b/8 的小 anchor block（STARATTN 为 l_a = l_b），大幅减少 anchor block 在 FFN 中的重复计算开销。Wall-time 分解（Figure 5）显示 APB 的 FFN 时间（30.76 ms/block）显著低于 STARATTN（50.01 ms/block）。

  4. Query-Embedded Anchor Block：将 query q 嵌入 anchor block 头部，使 retaining heads 能够感知查询相关信息以更精准地选择相关 KV pair。消融实验（Table 3, No.1-3）表明 query embedding 需与 retaining heads 配合使用才有效果。

  全栈执行例子（APB on 8 hosts, 128K input, l_a=4K, l_p=2K）：
  - 算法pipeline：Context Splitting（l_b=16K, H=8）→ Block Compression（retaining heads 打分 → Top-2K KV pair）→ AllGather Communication（K^C, V^C）→ Modified Attention（[A, P_h, B_h] 三部分联合计算）→ FFN（仅 A+B_h，P_h 丢弃）；Decoding 用 STARATTN stage-2 accurate attention
  - 系统框架：基于 HuggingFace Transformers + 定制 FLASHATTN kernel（修改 attention mask M'）+ NCCL AllGather 通信调度
  - 编译框架：论文未明确说明
  - kernel调度：每层 Transformer：QKV Proj (4.01ms) → Retaining Head (1.72ms) → AllGather K^C+V^C (0.62ms) → Modified FLASHATTN Attention (34.07ms) → O Proj (2.67ms) → FFN (30.76ms)。通信占比仅 ~0.8%
  - 硬件架构：8× NVIDIA A800-80GB, NVLink 3.0 + HDR InfiniBand

  核心创新：APB 在序列并行框架中引入"压缩-传递"机制——每个 host 独立压缩自己的 KV cache，仅将最重要的 Top-l_p 个 KV pair 通过 AllGather 传递给后续 host 作为 passing block。这同时解决了三个问题：(a) KV cache 剪枝不需要全局 attention score（localized scoring by retaining heads）；(b) 多 host 扩展时 passing blocks 确保跨 host 上下文可见性不丢失；(c) 小的 anchor block 减少 FFN 重复计算开销。实验结果证明：passing block + retaining heads + anchor block + query embedding 四组件共同起作用（Table 3 No.0 vs No.6-8），缺一不可。

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- baseline方法是什么？
  现有低比特 KV cache 推理系统分为两类，均未能高效利用 Tensor Cores：(1) **Non-fused attention with separated kernels（Kivi）**：将 mixed-precision attention 分解为多个独立 kernel（dequantization kernel + attention kernel），各 kernel 独立 launch，中间数据反复读写 global memory。虽然灵活支持多种 attention variant，但增加了 launch overhead、inflated memory traffic，破坏了 on-chip data reuse；(2) **Fused attention on CUDA Cores only（Atom, QServe）**：将 dequantization 和 matmul 都实现在 CUDA Cores 上（FMA 指令），虽避免了 non-fused 的中间数据问题，但完全忽略了 Tensor Cores——现代 GPU 的主要算力来源（A100: Tensor Cores 312 TFLOPS FP16 vs CUDA Cores 19.5 TFLOPS FP32）。CUDA Cores 同时处理 dequantization（memory-bound）和 matmul（compute-bound），导致 register bandwidth 竞争、L1/L2 争抢、occupancy 下降，尤其在 GQA 等 arithmetic intensity 较高的 attention variant 下性能严重退化（QServe 在 GQA 下 speedup 从 MHA 的 3.5× 跌至 1.4×）。两类 baseline 都未能解决三个关键挑战：(C1) 低比特数据 layout 与 Tensor Core fragment layout 不匹配——量化后的 packed 数据直接 dequantize 会产生乱序的 register 分布，无法直接送入 Tensor Core mma；(C2) dequantization 频繁 stall warp 执行——FlashAttention 原始的单 warp 沿 N 维策略使 dequantization 序列化；(C3) 缺乏通用的系统级优化——不同量化算法使用不同的 scaling granularity（tensor-wise vs channel-wise），现有 mixed-precision kernel（Marlin, Ladder）仅针对静态权重，无法处理动态生成的低比特 KV cache。

  全栈执行例子（Baseline / QServe on A100, 128K context, LLaMA-3.1-8B GQA decode）：
  - 算法pipeline：在线 INT4 quantization（tensor-wise/channel-wise）→ KV cache 存储为 packed INT4 → decode 时每 token 执行 CUDA Core-only fused attention kernel（FMA dequantization + FMA GEMV/GEMM）。因 CUDA Cores 同时承担 dequantization（memory-bound，~50% kernel time）和 matmul（compute-bound），Tensor Cores 完全闲置
  - 系统框架：基于 FlashAttention kernel 修改，集成到 HuggingFace Transformers/vLLM serving pipeline；支持 paged attention memory management
  - 编译框架：论文未明确说明（CUDA 手工 kernel）
  - kernel调度：Block-wise tiling（Q tile + KV tile）→ cp.async 加载 packed KV + 量化参数 → CUDA Core FMA dequantization（INT4→FP16, per-element scale+zp）→ CUDA Core FMA matmul（QK^T, PV）→ online softmax → output write-back。全程仅在 CUDA Cores 上执行，Tensor Cores 无负载。Dequantization 消耗近 50% 的 kernel execution time
  - 硬件架构：NVIDIA A100 GPU（80GB HBM, 312 TFLOPS Tensor Cores FP16, 19.5 TFLOPS CUDA Cores FP32），Tensor Cores 利用率 ~0%

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BitDecoding 通过 **cooperative use of Tensor Cores + CUDA Cores** 将 Tensor Cores 引入低比特 KV cache 解码。四项设计分别对应 baseline 的三个核心挑战：

  **1. Layout Induction via Hardware Instructions（解决 C1: Layout 不匹配）**：
  利用 ldmatrix 的 thread-to-register 映射天然产生 Tensor Core 的 interleaved fragment layout。在 Residual Kernel 中，ldmatrix 加载 FP16 KV tile 后，各线程在寄存器内完成量化和 INT16 packing——因为 ldmatrix 建立的 interleaved 映射在打包过程中被"隐式保存"。Packing Kernel 以相同 ldmatrix 配置加载 packed 数据后，解量化结果自动对齐 Tensor Core 寄存器，无需全局 reshape。这比 Marlin 的离线 layout transformation kernel 和 Ladder 的迭代搜索快 3 个数量级（prefill: 0.06ms vs 58ms/4.79ms, decode: 0.008ms vs 0.41ms/0.65ms, Table II）。

  **2. Warp Parallelism Strategy（解决 C2: Dequantization stall）**：
  将 FlashAttention 的单 warp 沿 N 维改为多 warp（W_m=1, W_n>1），SM warp scheduler 交替调度多个 warp 执行 dequantization，消除单 warp 的序列化 stall。配合 Cooperative Softmax（register→shared memory→register cross-warp reduction），以仅 0.5% overhead 恢复多 warp 下的计算正确性。TC utilization 从 10.91%（W_n=1）提升到 19.66%（W_n=4）+ correctness valid（Table III）。

  **3. Asynchronous Pipeline（解决 C2 进阶: CUDA-Tensor Core 协调）**：
  Packing Kernel 中实现 register-level 异步流水线：第 i 个 tile 的 Tensor Core mma 与第 i+1 个 tile 的 ldmatrix + CUDA Core dequantization 重叠执行，持续 producer-consumer flow。Dequantization overhead 从 baseline 的 ~50% 降至 <15%（4-bit）和 <35%（2-bit）（Fig. 15）。

  **4. Residual Kernel with Unified Quantization（解决 C3: 通用性）**：
  基于 N_r 对齐的 KV cache partitioning 统一支持 tensor-wise 和 channel-wise 量化——沿 seq_len 维做 channel-wise，沿 hidden dim 维做 tensor-wise，均在 residual block 内执行。Warp-level reduction（__shfl_xor_sync + shared memory buffer）高效计算 scale/zp。支持 MHA/MQA/GQA 全 attention variant（通过 Query Transformation reshape）。

  **5. Architecture-specific Optimizations**：
  - Hopper：利用 STSM + wgmma_SS 指令对，dequantized 数据经 shared memory 直接供 Tensor Core 消费，wgmma 异步执行
  - Blackwell：利用原生 MXFP4/NVFP4 mma 指令，直接在 packed 4-bit 数据上做 GEMM，完全消除 dequantization

  全栈执行例子（BitDecoding on H100, 128K context, LLaMA-3.1-8B GQA decode）：
  - 算法pipeline：
    1. Prefill 后 KV Cache Partitioning：N_r = 8 × W_n × R（e.g., R=4 for 4-bit, W_n=4 → N_r=128），X_pack = X[:L - (L mod 128)]（量化+pack），X_res = X[L-128:]（FP16 residual）
    2. Per decode step:
       a. Query Transformation: Q [1, 4, 8] → [4, 8]（gq=4 for LLaMA-3.1-8B GQA hq=32, hkv=8）
       b. Packing Kernel: cp.async 异步加载 Q tile + K_pack/V_pack tiles + K_p/V_p params → ldmatrix 加载 packed data + lop3 75316420 remapping（CUDA Cores, 与上一 tile 的 mma 重叠）→ mma QK^T（Tensor Cores wgmma_SS, B from shared memory via STSM）→ Cooperative Softmax（cross-warp sTMP reduction）→ P → sAcc → ldmatrix reload → mma PV（Tensor Cores）→ output
       c. Residual Kernel: 若 res_len == N_r，将满的 residual block 量化+pack → 追加到 packed cache
    3. Decode 持续至 EOS
  - 系统框架：CUDA/PTX 手工 kernel 集成到 HuggingFace Transformers attention backend；与 FlashAttention-3 兼容（Hopper warp-specialized pipeline）；支持 paged attention
  - 编译框架：论文未明确说明（手工 CUDA kernel，无编译框架修改）
  - kernel调度：Packing Kernel 异步流水线（ldmatrix+Dequant [CUDA Cores] || mma [Tensor Cores]）；Cooperative Softmax（register/shared memory cross-warp sync）；Residual Kernel（ldmatrix→quantize→pack→write，fused 单 kernel）；Hopper 优化（STSM+wgmma_SS, TMA 异步数据加载）
  - 硬件架构：H100 GPU（80GB HBM, 989 TFLOPS Tensor Cores FP16, 60 TFLOPS CUDA Cores FP32）；Tensor Cores 利用率 ~19.66%（实测，受 dequantization 限制但远超 CUDA Core-only baseline 的 ~0%）

  核心创新总结：BitDecoding 不是简单地"把 dequantization 放到 CUDA Cores、matmul 放到 Tensor Cores"，而是通过 **layout induction（ldmatrix→quantize→ldmatrix→dequant→mma 的闭环对齐）** 和 **warp-level parallelism（多 warp dequantization + cooperative softmax + 异步流水线）** 两个系统级设计，使这种分工真正高效。这种设计对任意低比特位宽、任意量化粒度、任意 attention variant、任意 GPU 代数都是高效和通用的。

## AdaSplash: Adaptive Sparse Flash Attention

- baseline方法是什么？
  Baseline 是标准的基于 softmax 的稠密注意力机制，使用 FlashAttention-2 进行硬件优化的 tiling 和 recomputation。具体而言：
  - attention 概率通过 softmax(s_i) = exp(s_i)/Σ_j exp(s_j) 计算，对所有 token 分配非零概率
  - FlashAttention-2 通过 block-wise tiling 将 Q,K,V 分块加载到 SRAM，online softmax 计算避免 materialize 完整 S ∈ R^{n×n} 和 P ∈ R^{n×n}
  - 反向 pass 利用存储的 O 和 online softmax 的 lse (log-sum-exp) 计算梯度
  - 稠密 attention 的缺陷：(a) 对所有 token 分配非零概率导致 attention 分散 (dispersion)，尤其是长上下文场景下小概率累积会稀释重要 token 的贡献；(b) 无法利用 attention 权重的自然稀疏性（实验表明 ~3% entries 覆盖 96% attention mass）来进一步减少计算

  全栈执行例子（Baseline / FlashAttention-2 softmax attention）：
  - 算法pipeline：QK^T/√d → softmax (dense, 每行和为 1) → PV。softmax 强制输出稠密概率分布，FlashAttention-2 通过 tiling 保证 O(n) memory 但无法减少 FLOPs（始终 O(n²) 的计算复杂度）
  - 系统框架：CUDA/Triton kernel，集成到 PyTorch (torch.nn.functional.scaled_dot_product_attention)，HuggingFace Transformers
  - 编译框架：torch.compile 可用但不适用于 attention 的复杂 memory access pattern；FlashAttention-2 使用手工 CUDA kernel
  - kernel调度：FlashAttention-2 kernel 分块加载 Q,K,V → SRAM compute S → online softmax → rescale O → write back。前向仅需 1 次 K,V 加载，反向用 recomputation 避免 store S
  - 硬件架构：Nvidia H100 (80GB) / RTX A6000 (48GB)，利用 GPU 层级内存（HBM → SRAM）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ADASPLASH 用 α-entmax 替代 softmax 作为 attention 的概率变换，结合 Hybrid Halley-Bisection 算法和自定义 Triton kernel，实现自适应稀疏注意力在训练时的实际加速。

  **对应解决 Baseline 缺陷的三项核心设计**：

  1. **α-entmax 替代 softmax → 解决 attention 分散和无法利用稀疏性问题**：
     α-entmax 通过参数 α > 1 产生真正稀疏的概率分布（α=1 退化为 softmax, α=1.5 约 95% sparsity, α=2 即 sparsemax 约 99% sparsity）。由 [(α-1)s - τ]_+^{1/(α-1)} 公式可知，score 低于 τ/(α-1) 的 token 获得精确零概率——不仅消除了小概率残差的干扰（解决 dispersion），还创造了可利用的稀疏性。

  2. **Hybrid Halley-Bisection 算法 → 解决 α-entmax 计算本身太慢的问题**：
     α-entmax 需通过迭代求解 τ（f(τ) 的根），传统 bisection 需 23 次迭代且每次需完整遍历 S 导致大量 HBM 读写。Halley-bisection 利用 f 的二阶导数实现 cubic convergence rate，仅需 3 次迭代到 machine precision；且 block 版本在 SRAM 中累积 f/f'/f''，不 materialize S。Fail-safe 机制保证即使 Halley 发散也回退到 bisection，确保最坏情况下仍收敛。结果是 15× 加速（2.38ms vs 36.67ms at n=8192）和 1.75× 内存节省。

  3. **Sparsity-aware Triton kernel (block masking + lookup tables) → 真正利用稀疏性减少计算**：
     FlashAttention-2 虽然可以用 block-sparse 变体，但 mask 必须预先定义，而 α-entmax 的稀疏模式是数据依赖的（dynamic）。ADASPLASH 在 Halley-bisection 最后迭代中动态检测哪些 Q,K block pair 产生非零 P，构造 binary block mask M 和 pointer-increment lookup tables（K_j, Q_i）。后续前向和反向 pass 通过 lookup tables 跳过 null blocks 的 HBM 加载和 GEMM 计算，实现真正的稀疏加速。当稀疏度足够高时，ADASPLASH 的 wall-clock time 可超越 FlashAttention-2（后者 runtime 对稀疏度无反应，始终执行 full computation）。

  全栈执行例子（ADASPLASH α-entmax attention, α=1.5, Triton kernel on H100）：
  - 算法pipeline：QK^T/√d → α-entmax（Halley-bisection 求 τ, 仅 3 迭代）→ 稀疏 P = [(α-1)S-τ]_+^{1/(α-1)}（预测 ~95% zeros）→ PV；训练时 α 从 1.0 线性 anneal 到 1.5（over 1B tokens）确保 dense→sparse 平滑过渡
  - 系统框架：Triton kernel 替代 torch.nn.functional.scaled_dot_product_attention，PyTorch + HuggingFace Transformers（在 attention 层替换 fa2 → adasplash）；训练用 fp16/bf16 mixed precision, AdamW optimizer
  - 编译框架：论文未明确说明（Triton 自身是 JIT-compiled 到 GPU）
  - kernel调度：前向：(1) Halley-bisection block kernel → τ (3 passes over K, 仅需此额外开销) → (2) 构造 M 和 lookup tables → (3) 仅对 M_{ij}=1 的 blocks 计算 O_i += P_i^{(j)} V_j。反向：(1) dK/dV kernel 用 K_j lookup 仅迭代有效 Q_i；(2) dQ kernel 用 Q_i lookup 仅迭代有效 K_j。利用 α-entmax 的稀疏 Jacobian (Diag(u) - uu^T/||u||_1, u = p^{2-α}) 替代 softmax 的稠密 Jacobian
  - 硬件架构：Nvidia H100 (80GB) / RTX A6000 (48GB)；Triton kernel 利用 SRAM (on-chip) 做 block-wise computation；block mask M 为 binary 值，跨 attention 层可共享 memory

  **关键 trade-off**：前向 pass 比 FlashAttention-2 多 ~2 次 K 加载（用于 τ 计算），故在低稀疏度下慢于 FA2；但随着序列变长/稀疏度增加，跳过 null blocks 的收益超过额外 τ 计算开销，最终超越 FA2（Figure 1）。内存复杂度在启用 block masking 时变为 O(n + T_r×T_c)（额外 mask 存储），但仍远小于完整的 O(n²)。

  **实验验证的核心结论**：
  - GPT-2 (124M, 1024 ctx, H100): ADASPLASH 1.03 s/step vs FA2 0.98 s/step — 仅慢 5%，但 Torch bisection (7.78 s/step) 和 sorting (3.61 s/step) 不可用
  - ModernBERT (149M, 8192 ctx, A6000): ADASPLASH 1.53s，超越 Halley-bisection without masking (1.61s)，碾压 Torch bisection (4.99s)
  - 下游任务精度无显著损失：GLUE avg RoBERTa α=1.5 → 83.9 (vs softmax 83.9); ModernBERT α=1.5 → 83.5 (vs softmax 83.7); BEIR nDCG@10 多项超越 dense counterpart
  - GPT-2 α=1.5 validation loss 3.263 (vs softmax 3.283)，HellaSwag 30.6 (vs 30.4)

## Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

- baseline方法是什么？
  Baseline是现有的KV cache eviction方法，分为三类：(1) Post-fill eviction方法（PyramidKV, SnapKV, H2O）——在pre-filling全部完成后才基于attention scores evict KV，导致pre-filling期间的高peak memory和pre-fill stage几乎无KV footprint reduction；(2) Recency eviction方法（DuoAttention）——将attention heads分为retrieval heads和streaming heads，但依赖L2 reconstruction loss（而非next-token prediction loss）训练、continuous gating variable带来train-test gap、仅用synthetic passkey训练数据无力捕获复杂long-range dependencies；(3) Dynamic sparsity方法（NSA, MoBA, MInference）——仅减少inactive attention weights但不实际evict KV，无法降低KV memory。

  全栈执行例子（Baseline / DuoAttention on Llama-3.1-8B-Instruct, 128K context）：
  - 算法pipeline：训练时用synthetic passkey retrieval数据+L2 reconstruction loss重建hidden states，学习continuous gating z∈[0,1]；训练后按sparsity threshold rounding z→{0,1}产生train-test gap；inference时retrieval heads保留完整KV cache，streaming heads仅保留W=1024 local+S=128 sinks
  - 系统框架：基于PyTorch推理，支持chunked pre-filling（chunk_size=32K），无serving框架修改
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明（使用标准FlashAttention）
  - 硬件架构：通用GPU推理，论文未明确说明具体GPU型号

- 论文方法是什么？如何对应解决Baseline的缺陷？

  论文提出**(1) KV Footprint统一度量**和**(2) PruLong**以及**(3) Chunked Eviction**，分别对应baseline的三大缺陷：

  **1. KV Footprint → 解决fair comparison缺失（第2节）**：
  定义KV footprint = 所有timestep的un-evicted active+inactive KV entries的time-integrated sum，归一化至full causal attention。与KV cache size（instantaneous metric）不同，KV footprint同时捕获pre-filling和decoding两阶段的memory usage。引入critical KV footprint = 保留≥90% full attention性能的最小footprint，使不同方法在公平的utility-efficiency trade-off上可比。Appendix A展示了peak KV作为alternative metric，结论一致。

  **2. PruLong → 解决DuoAttention的三个核心缺陷（第4节）**：
  - **Next-token prediction loss替代L2 reconstruction**：直接优化语言模型的实际使用目标（token generation quality），而非proxy loss（hidden state reconstruction）；实验证明即使DuoAttention用4倍训练steps也无法在natural data上收敛，而PruLong轻松收敛（Recall 91.4 vs 38.6 at 70% sparsity）
  - **Hard concrete + Bernoulli masks替代continuous gating**：用hard concrete reparameterization [Louizos et al., 2018]将z建模为Bernoulli随机变量，训练时端到端采样离散mask并优化，消除train-test rounding gap；配合Lagrangian penalty实现精确target sparsity regularization，支持训练后任意sparsity extraction
  - **Natural long-context pre-training data替代synthetic passkey**：使用Gao et al. (2025)的continued pre-training mix（code repositories + books），包含多样化的long-range dependencies，使PruLong的head assignment能泛化到recall/RAG/re-ranking/ICL/summarization等多种task types

  **3. Chunked Eviction → 解决post-fill eviction的pre-fill高peak memory（第3节）**：
  将PyramidKV/SnapKV的eviction heuristic从"pre-fill后一次性执行"改为"chunked pre-filling的每个chunk后执行"：
  - Naive Chunked Eviction：每个chunk独立计算最后k个token的attention score → evict bottom KV
  - Patched Chunked Eviction：每个chunk末尾拼接prompt的最后k个token作为query → 用完整prompt的重要性信号指导eviction
  - 同时修复GQA下的KV replication issue：在KV group内mean-pool attention后再选择统一KV set，节省8×内存
  - Patched PyramidKV在RAG（<34% footprint）和LongQA（<35%）上取得所有方法中最优结果

  全栈执行例子（PruLong on Llama-3.1-8B-Instruct, 128K context, 70% streaming heads）：
  - 算法pipeline：1000 steps训练（batch 1M tokens, seq_len 131K, LR=1.0 for log α, LR=1.0 for λ1/λ2, model weights frozen）→ target sparsity warmup 0→0.7 over 800 steps → Lagrangian penalty驱动收敛 → 训练后按log α排序取top 30%为retrieval heads → inference时retrieval heads用full KV cache，其余仅保留W=1024+S=128 → KV footprint ~30%（critical KV footprint在Recall上46%，比DuoAttention低12 points）
  - 系统框架：基于PyTorch推理，支持chunked pre-filling（chunk_size=32K）；streaming heads的fixed-size KV cache使decoding阶段memory恒定；可应用于pre-SFT或post-SFT stage
  - 编译框架：论文未明确说明
  - kernel调度：标准FlashAttention（无自定义kernel），streaming heads的attention mask变化不影响FlashAttention tiling
  - 硬件架构：通用GPU推理；peak memory PruLong 26.3 GiB（Recall task at 70% sparsity）vs PyramidKV+P+C 33.7 GiB；throughput PruLong 10.8×10⁻² req/s vs DuoAttention 10.0×10⁻²

  **核心创新总结**：
  - KV footprint作为time-integrated memory metric是conceptual contribution，为KV eviction方法的fair comparison奠定基础
  - PruLong的三项设计（NTP loss + hard concrete masks + natural data）分别攻克DuoAttention的三项缺陷，在recall task上实现12 points critical footprint improvement
  - Chunked eviction让post-fill methods在chunked pre-filling时代重获竞争力——patched PyramidKV在RAG和ICL上最优
  - 关键发现：没有任何一种方法在所有task上最优（PruLong强于recall，PyramidKV+Patched强于RAG/ICL），揭示KV eviction没有one-size-fits-all解决方案
  - Pre-filling chunk size sensitivity是未预期的挑战——PruLong/DuoAttention在8K vs 32K chunk size下performance差异达20%，比PyramidKV更敏感

## Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

- baseline方法是什么？
  Baseline 是现有基于 self-attention scores 的统一 KV cache 剪枝方法（以 SnapKV、H2O 为代表），它们将视觉 token 和文本 token 在长序列中一视同仁地进行剪枝，全栈执行例子如下：
  - **算法层**：SnapKV 使用所有 attention head 的末尾 observation window attention scores，通过 max-mean pooling 评估每个 token 的重要性，选取 top-N 高 attention token 保留 KV cache。H2O 基于累积 attention scores（A2S）识别 "heavy-hitter" token 并动态 evict。这些方法在整个混合序列（visual + text）上统一计算 attention scores 用于重要性估计。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，处理 LLaVA 等多模态 VLM。输入先经 Visual Encoder（如 CLIP）+ MLP Adapter 将图像转为视觉 token，再与文本 token 拼接送入 LLM。KV cache 管理与纯文本 LLM 一致——prefill 后执行 eviction，decode 时使用压缩 cache。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速 attention 计算。token selection 为 GPU 上的 TopK 和索引 gather 操作。
  - **硬件架构层**：RTX 4090 / A100 GPU。
  
  Baseline 的**核心缺陷**：多模态场景下，self-attention（同一模态内）和 cross-attention（跨模态间）具有显著不同的注意力分布——文本 token 的 self-attention scores 通常大于视觉 token，导致统一的重要性估计偏向文本模态，造成**关键视觉 token 被过度剪枝**，破坏跨模态交互，最终降低多模态推理性能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CSP 通过两项核心设计解决 baseline 的模态分布偏差问题：

  **1. Cross-Self Attention Decomposition → 解决模态间分布偏差导致的 token 重要性误估**：
  将原始注意力矩阵分解为 intra-modality attention（A^{st} 文本→文本 + A^{sv} 视觉→视觉）和 inter-modality attention（A^{ct} 视觉→文本 + A^{cv} 文本→视觉），独立计算各自的重要性分数（沿 query 轴求和），并独立进行 top-K 选择（M^s 和 M^c）。最终保留的 token 必须同时在这两个维度上被判定为重要（M = M^s ∧ M^c）。这确保：视觉 token 虽然可能 self-attention score 较低，但如果在 cross-attention 中被文本 token 关注（说明跨模态信息重要），仍会被保留。反之亦然。

  **2. n-Softmax Smoothness Recovery → 解决剪枝后注意力分布锐化导致的性能退化**：
  剪枝后 softmax 的 denominator 变小（去掉了被剪枝 token 的贡献），导致剩余 token 的注意力分数被"放大"，分布变尖锐。n-Softmax 通过加入偏置项 n：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})，有效恢复原始分布的平滑性，n=1 在所有实验中固定使用。

  全栈执行例子（CSP on LLaVA-v1.5-7b, RTX 4090）：
  - **算法层**：
    1. Visual Encoder (CLIP) → MLP Adapter → 视觉 token L_v 个 + 文本 token L_t 个 = L 个 token 输入 LLM
    2. Prefill 阶段：正常计算 QKV 投影 + FlashAttention，所有 KV 存入 cache
    3. 首次需要剪枝时（L_k ≥ T）：
       a. 取最近 O 个 query token 的 attention logits：A = n-Softmax(O[-O:, :-R])（n=1）
       b. 分解 A → A^{st}, A^{sv}, A^{ct}, A^{cv}
       c. A^s = Σ_{query} A^{st} ⊕ Σ_{query} A^{sv}（intra-importance）
       d. A^c = Σ_{query} A^{ct} ⊕ Σ_{query} A^{cv}（inter-importance）
       e. M^s = TopK(A^s, K^s), M^c = TopK(A^c, K^c)
       f. M = M^s ∧ M^c
       g. K = (K ⊙ M) ⊕ K[-R:], V = (V ⊙ M) ⊕ V[-R:]
    4. Decode 阶段：使用压缩后的 KV cache 进行 attention，新 token 的 KV pair 追加到 cache
  - **系统框架层**：即插即用集成到 LLaVA 推理流程，仅修改 attention 层的 token selection 逻辑，无需重新训练模型。默认配置：n=1, cross_ratio=0.5（平衡 intra/inter），recent window R 由 cache budget 决定。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速 attention。token selection（TopK + mask + gather）在 GPU 上执行，额外开销极小。
  - **硬件架构层**：LLaVA-v1.5-7b 在 RTX 4090（24GB）上：60% budget 时 1.207 GiB GPU 内存、24.377 ms/token；10% budget 时 0.208 GiB、16.287 ms/token（37% 加速 + 87% 内存节省）。

  **对比 baseline 的关键差异**：
  - Baseline 统一对待所有 token → CSP 将 intra/inter 分离独立选择
  - Baseline 直接 softmax → CSP n-Softmax 补偿剪枝导致的分布锐化
  - Baseline 的 mask 是单一维度 → CSP 的 M = M^s ∧ M^c（双维度交集），确保跨模态交互完整性
  - CSP 在 MileBench 上：LLaVA-v1.5-7b 的 T-3 提升 4.5%、S-5 提升 7.2%、NH 提升；LLaVA-v1.5-13b 的 T-3 提升 8.3%、T-4 提升 7.2%、IR 提升 9.6%

## CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

- baseline方法是什么？
  Baseline 是标准的 LLM 长上下文推理流程：每个 token 的 Key 和 Value 以完整维度 hout（通常等于 hidden_size/num_heads）存储在 KV Cache 中，KV Cache 大小随序列长度 n 线性增长（2 × n_layers × n × hout × dtype_size）。在 200K token 场景下（LLaMA-2-7B），KV Cache 约占用 100GB。现有训练无关压缩方法（如 StreamingLLM 的 token pruning + 保留 attention sink；H2O 的 Heavy-Hitter Oracle token pruning）面临压缩率上限，高压缩率时因丢弃关键 token 导致检索任务（如 LongEval）性能崩溃。ASVD（训练无关 channel shrinking via SVD）在高压缩率（80%）时导致模型丧失语言建模能力，输出不可解析的乱码 token。训练依赖方法（如 MLA/DeepSeek-V2）虽压缩率高但需从零重训整个模型，无法适配已有预训练模型。

  全栈执行例子（Baseline / 标准 LLaMA-2-7B 长上下文推理，200K tokens）：
  - 算法pipeline：每个 Transformer 层的 W^K, W^V 将输入 X 映射为 K ∈ R^{n×hout}, V ∈ R^{n×hout}，全部存入 KV Cache，不做任何压缩。Attention: softmax(QK^T/√d) × V。
  - 系统框架：标准 HuggingFace Transformers 推理流程，KV Cache 为 autoregressive 解码的 cache 机制，每步 decode 将新 KV append 到 cache。
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：记忆瓶颈——200K tokens 时 KV Cache ~100GB，远超 A100-80G 或 RTX 4090 24GB 显存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 CSKV 提出 training-efficient 的通道收缩 KV Cache 压缩方法，通过低秩分解 + 双分支缓存 + SVD-based 逐层微调，实现 80% 压缩且保持长上下文能力。

  **对应解决 Baseline 缺陷的设计**：

  1. **低秩分解解决通道维度冗余**：观察到 KV Cache 的奇异值呈长尾分布——移除最小的 50% 奇异值仅导致 MMLU 平均精度损失 <1%（0.458→0.449）。将 W^K ∈ R^{hin×hout} 分解为 A^K ∈ R^{hin×hcomp} 和 B^K ∈ R^{hcomp×hout}，仅存储压缩特征 hcomp 维（而非完整 hout 维），内存从 O(n×hout) 降至 O(n×hcomp)。

  2. **双分支 KV Cache 解决信息损失**：近期 token（窗口大小 m=32）保留完整精度，确保局部上下文预测质量不受影响；历史 token 从压缩特征通过 B^K 重建用于 attention。这避免了 token pruning 方法（如 StreamingLLM、H2O）完全丢弃 token 导致的检索信息丢失问题。

  3. **ASVD 初始化 + 逐层 MSE 重建微调解决训练代价问题**：ASVD（Activation-aware SVD）使用标定数据（256 样本）计算缩放矩阵 S，使低秩分解关注激活值大的维度。仅优化逐层重建损失（MSE(K, K_hat)+MSE(V, V_hat)）而非端到端语言建模损失，训练仅需 90 分钟/单 A100（vs 从零重训练的数天/数月）。随机初始化在此设置下完全无法收敛（Loss ~1e9），证明 SVD-based 初始化的必要性。

  4. **量化正交兼容**：通道压缩（channel shrinking）与量化（quantization）是正交维度，可与 KIVI 4-bit QAT 无缝结合，达到 95% 总压缩率（80% channel + 4-bit = 95% total），保持 >90% 长上下文能力。

  全栈执行例子（CSKV）：
  - 算法pipeline：(1) Prefilling：X → K_full = XW^K（attention 计算用），K_C = XA^K（存入 Compressed Cache），K_local = K_full[-m:, :]（保留 m 个完整 token）；(2) Decoding：新 token → 更新两个 cache → 从 Compressed Cache 用 B^K 重建历史 token 的 K_hat → concat([K_hat, K_local]) 用于 attention → 从 Full Cache 移除最旧 token 保持窗口 m；(3) 训练：ASVD 初始化 A^K/B^K/A^V/B^V → 逐层 MSE(K, XA^KB^K) + MSE(V, XA^VB^V) → AdamW 微调
  - 系统框架：即插即用集成到 HuggingFace 推理流程——仅修改 attention 层的 Key/Value 投影和 cache 管理，不改 LLM backbone 结构
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明
  - 硬件架构：训练在单 A100-80G 上 90 分钟完成；推理 KV Cache 从 ~100GB 降至 ~20GB（80% 压缩），可适配 RTX 4090 24GB

  **消融发现的关键 insights**：
  - 随机初始化低秩矩阵导致训练完全失败（Avg.Acc=0.00），ASVD 初始化为关键使能技术
  - 窗口大小 m 与性能正相关，但 m>32 后收益递减（m=32 Avg.Acc=0.92, m=4096 Avg.Acc=0.96），m=32 已足够保留局部信息
  - Key cache 比 Value cache 对压缩更不敏感——在固定 budget 下应给 Key 分配更高压缩率（K 87.5% + V 12.5%: Avg.Acc=0.97 vs K 12.5% + V 87.5%: Avg.Acc=0.80, 均在 50% 总压缩率下）
  - 与量化直接 PTQ 结合导致性能崩溃，需 QAT 才能保持性能（80% 通道压缩 + 4-bit QAT → Avg.Acc=0.90 vs PTQ → 0.00）

## CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

- baseline方法是什么？
  Baseline 是 SnapKV（代表性 KV cache eviction 方法），其全栈执行例子如下：
  - **算法层**：SnapKV 在 prefill 阶段计算所有 attention head 的 attention scores，使用末尾 observation window（默认 8 tokens）内的 attention scores 通过 clustering（per-head 或 per-GQA-group 的 max-mean pooling）来评估每个 token 的重要性。选择 top-N 高 attention 的 token 保留其 KV cache，其余 evict。所有 head 同等对待——对 GQA group 内的多头 attention scores 求和后统一判断。问题：(a) 当 GQA group 内 Streaming Head 占主导时，仅保留首尾 token 的 KV cache，evict 中间关键 token；(b) 每层使用相同的固定 cache budget，不考虑层间差异。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline。Prefill 阶段计算 QKV → 所有 KV 存入 cache → 计算 observation window 的 attention → voting/select top-N → evict 非重要 KV。Decoding 阶段使用压缩后 cache 进行 attention，新 KV pair 追加。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 FlashAttention-2 加速 attention 计算，eviction 操作在 GPU 上实现为索引选择和内存拷贝。
  - **硬件架构层**：运行于 NVIDIA A100 GPU。

  对于 PyramidKV 和 CAKE（扩展 baseline）：在 SnapKV 的 token 选择基础上增加了层级自适应 cache 分配。PyramidKV 按金字塔形分配（浅层少、深层多），CAKE 使用 attention entropy 和 variance 在线计算层级重要性。问题：(a) 依赖 attention 分布的统计量（entropy/variance），计算开销大且跨模型泛化性差；(b) 仍使用所有 head 的 attention scores 做 token 选择，Streaming Head 主导问题未解决。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  CompressKV 通过两个核心设计解决 baseline 缺陷：(1) SRH 驱动的 token 选择解决"Streaming Head 主导 eviction"问题；(2) Error-Aware 层级自适应分配解决"层级无差异化/依赖 attention 统计量"问题。

  全栈执行例子（CompressKV）：
  - **算法层（核心创新）**：
    (a) **SRH 识别**：不要求 head 的 top-1 attention 精确落在正确答案 token 上（传统 Retrieval Head 标准），而是将 head 在整个 answer span 上的 attention scores 求和作为评估指标。公式：SemanticRetrievalScore(h) = Σ_{t} I[y_t∈A] Σ_{j∈A} a_{t,j}^h。这能捕捉到对 "sandwich" 周边语义相关 token（如 "eat", "a thing"）有高 attention 的 head——这些 head 即使 top-1 attention 不在 "sandwich" 上，仍然具有语义检索能力。
    (b) **SRH 驱动的 Token 选择**：每层仅使用 top-4 SRH（而非全部 head）来判断 token 重要性——对这些 SRH 的 attention scores 在 observation window 上求和、1D average pooling（kernel=5）、取平均后选出 top-N token。因为 SRH 不太受首尾 token 的 "attention sink" 影响，所以选出的 token 更均衡地覆盖了文本中间的语义关键信息，避免了 Streaming Head 主导导致的仅保留首尾 token 的问题。
    (c) **Error-Aware 层级分配**：离线在 LongBench 上模拟极端压缩（每层仅保留 32 tokens，约 0.3%），计算每层 attention output 的 Frobenius norm 重建误差 e^(l) = Σ_t ||O_comp,t^l - O_full,t^l||_F / ||O_full,t^l||_F。跨数据集归一化平均后得到层级重要性分数 ẽ^(l)。在线推理时按 ẽ^(l) 比例分配 cache budget，设置 per-layer 上下界 [m=32, M=3×B_per-layer]。与 CAKE/PyramidKV 不同，该分数离线计算、无需在线 attention 统计量计算、且基于真实压缩误差而非注意力分布统计量——因此泛化性更好。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，与 SnapKV 的集成方式相同。额外步骤：(a) 推理前加载预计算的 SRH 索引和 ẽ 层级分数；(b) Prefill 阶段 token 选择时，仅聚合 top-4 SRH 的 attention scores 而非全部 head；(c) 各层的 cache budget B_i 由 ẽ 分数和 Algorithm 1 确定，而非均分。与 GQA 和 FlashAttention-2 完全兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：包含自定义 CUDA kernel（`adakv`，位于 `methods/adakv/`），需要单独编译。使用 FlashAttention-2 加速标准 attention 计算。eviction 操作在 GPU 上完成。
  - **硬件架构层**：运行于 NVIDIA A100 GPU，实验覆盖 4K-128K context length。

  **对比 baseline 的关键差异**：
  - Baseline (SnapKV) 使用全部 head 的 attention sum 做 token 选择，Streaming Head 的影响导致仅保留首尾 token → CompressKV 仅使用 top-4 SRH，避免 Streaming Head 主导，更均衡地保留中间关键 token。消融实验：在 SnapKV 基础上加入 SRH Selection → LongBench 准确率从 43.76% 提升至 44.96%（+1.20 pp）。
  - Baseline (SnapKV) 每层固定相同 cache budget → CompressKV 使用 error-aware 层级自适应分配。消融实验：SRH Selection + Layer Allocation → 准确率从 44.96% 进一步提升至 45.43%（+0.47 pp）。
  - Baseline (PyramidKV/CAKE) 使用 attention 统计量（entropy/variance）做层级分配，需在线计算且模型泛化性差 → CompressKV 使用 offline 计算的 Frobenius norm 重建误差，无在线开销，基于真实压缩效果而非代理统计量，跨模型泛化性更好。
  - 极端压缩下优势更明显：LongBench 上 128 KV cache budget 时，CompressKV 领先 CAKE 0.26 pp（Llama-3.1-8B）和 0.52 pp（Mistral-7B）；NIAH 上 256 KV entries（0.07% 容量）达到 90% full-cache 准确率。

## Cost-Optimal Grouped-Query Attention for Long-Context LLMs

- baseline方法是什么？
  Baseline 是当前广泛采用的 Llama-3 GQA 配置方式：(1) 强制 nh × dh = d（如 d=1536, dh=64 → nh=32），head 数量由 hidden size 唯一确定，不可独立调整；(2) 固定 nkv=8（Llama-3 全系列统一），不随上下文长度或目标 loss 变化；(3) 模型大小 N 和 GQA 配置独立决定，不考虑推理上下文长度 T 对 time-variant cost（attention FLOPs + KV cache memory）的影响。

  缺陷：(a) nh × dh = d 是原 Transformer 论文的随意选择（Vaswani et al., 2017），无理论基础，导致 attention FLOPs（4TL dh nh）不可调——当 T 很大时 attention 占主导但无法减少；(b) nkv=8 在长上下文场景（T=128K）下导致 KV cache 内存（2TL dh nkv）巨大——128K 时 ~90% 推理内存被 KV cache 占用、仅 ~10% 用于模型参数（Figure 8）；(c) 现有 scaling law（Hoffmann et al., Kaplan et al.）仅考虑训练 FLOPs 不考虑推理成本和上下文长度。实验表明 Llama-3 GQA 在 T=128K 时 "highly suboptimal"——用 cost-optimal 配置可减少 >50% memory 和 FLOPs 且 loss 相等。

  全栈执行例子（Baseline / Llama-3 GQA, T=128K, 1.2B model）：
  - 算法pipeline：nh=32, nkv=8, dh=64 → 32 query heads 共享 8 KV heads（每组 4 query heads 共享 1 KV）。每层 attention FLOPs = 4TL dh nh = 4 × 128K × 36 × 64 × 32 ≈ 37.7G FLOPs（仅 attention softmax 部分）；KV cache = 2TL dh nkv = 2 × 128K × 36 × 64 × 8 = 4.7B floats ≈ 9.4GB（BF16）。模型参数 N=1.2B → 参数内存 ~2.4GB。总内存 ~11.8GB，KV cache 占 ~80%。
  - 系统框架：标准 HuggingFace Transformers 或 vLLM 推理；KV cache 为 autoregressive decode 的 key-value 缓存，每步追加新 token 的 KV。
  - 编译框架：论文未明确说明。
  - kernel调度：使用标准 FlashAttention-2 加速 attention。
  - 硬件架构：NVIDIA A800 GPU（80GB）；长上下文下 memory bandwidth bound——KV cache 读取是 bottleneck。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两个关键改变 + 三步搜索过程来寻找 cost-optimal GQA 配置：

  **Change 1: 解耦 nh 与 d** → 解决 Baseline 缺陷(a)。解除 nh × dh = d 约束，使 nh 成为独立超参数自由控制 time-variant FLOPs。这允许在长上下文时使用更少的 query head（如 nh=8 替代 nh=32）大幅减少 attention FLOPs，同时通过增加模型大小 N 来补偿 loss（增加 time-invariant FLOPs 但远小于 attention 节省）。

  **Change 2: 联合优化 N 与 GQA 配置** → 解决 Baseline 缺陷(b)(c)。将推理成本分解为 time-invariant（N：模型参数 FLOPs 2N + 内存 N）和 time-variant（T: attention FLOPs 4TL dh nh + KV cache 2TL dh nkv）。通过同时调整 N, nh, nkv 优化推理资源分配——长上下文下 time-variant cost 主导，应减少 head 数、增大模型 size 以更高效利用硬件。

  **三步搜索过程** → 系统性解决 "给定 target loss L* 和 context length T，什么 GQA 配置最省推理成本" 这一问题：
  - Step 1 (Candidate Selection): 定义 H_cand = {nh=1,2,4,...,32} × {nkv=1,2,4,...,32, nkv≤nh} = 21 个候选
  - Step 2 (Scaling Curve Fitting): 对每个 H 训练 3M→1.2B 模型，拟合 L(N;H) = (a/N)^b + E（R²>0.999）
  - Step 3 (Cost Minimization): 对每个 H 求 N*(H) 满足 L*，计算硬件感知成本 Z = 0.9·M^0.5 + 0.1·C^(1/3)，选 Z 最小者

  关键理论洞察：上下文长度 T 对 loss 的影响与 N 和 H 相独立（Section 5.7 verified），因此可以用短上下文（T=8K）的 scaling curve 外推至长上下文，大幅节省算力。

  全栈执行例子（Cost-Optimal GQA, T=128K, L*=2.615）：
  - 算法pipeline：搜索得 H*=(nh=8, nkv=1), N*=1.8B。8 query heads 全部共享 1 KV head（退化为 MQA）。每层 attention FLOPs = 4TL dh nh = 4 × 128K × 36 × 64 × 8 ≈ 9.4G（节省 75% vs baseline 37.7G）；KV cache = 2TL dh nkv = 2 × 128K × 36 × 64 × 1 = 589M floats ≈ 1.18GB（节省 87.5% vs baseline 9.4GB）。模型参数 N=1.8B → ~3.6GB。总内存 ~4.8GB，节省 ~60% vs baseline。尽管模型大了 50%，但因 KV cache 从 8 KV heads 降为 1，总推理资源大幅减少。Downstream accuracy 几乎不变（common-sense 45.5% vs 45.7%，NIAH 略优）。
  - 系统框架：同 baseline——标准推理 pipeline，仅改变模型配置（L,d,nh,nkv）；与 FlashAttention-2 和现有推理框架完全兼容。
  - 编译框架：论文未明确说明。
  - kernel调度：标准 FlashAttention-2——因 head 数减少，单个 attention 计算的 tiling 效率反而可能提升（更大 tile、更少 kernel launch overhead）。
  - 硬件架构：NVIDIA A800 GPU；减少 KV cache → 更少 HBM 读写 → 从 memory bandwidth bound 转向 compute bound → 更高效利用 GPU 算力。

  对应解决的完整映射：
  - Baseline 缺陷(a) nh×dh=d 限制 → Change 1 解除 → nh 从 32→8（128K 时），attention FLOPs 降 75%
  - Baseline 缺陷(b) nkv=8 固定 → Change 2 + Step 3 → nkv 从 8→1（128K 时），KV cache 内存降 87.5%
  - Baseline 缺陷(c) scaling law 不考虑推理 → 引入 M_infer(T) + C_infer(T) + Z 统一成本函数，Step 2 scaling law 拟合 → 精确量化 N vs. H 的 tradeoff

  核心发现：
  - 长上下文下应使用 **更少的 head + 更大的模型**（更多 time-invariant 资源），因 time-variant cost 主导
  - 常用 Llama-3 GQA (d/dh, 8) 仅对特定 (L*, T) 组合最优，大多数情况下 suboptimal
  - nh 比 nkv 对 loss 更重要（相同参数增量下 nh 增加带来更大的 loss 降低），两者均呈 diminishing returns
  - loss 与 nh 呈 power-plus-constant 关系：L(nh) = a·nh^b + c，与 model size 和 context length 独立
  - 对齐 training FLOPs 时，用更少 head 可获更多训练数据，优势更大（88%/83% memory/FLOPs 节省）
  - 成本函数默认权重 λ=0.9（偏重 memory）可调整以适配不同部署约束

## Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

- baseline方法是什么？
  Baseline 是标准 attention（全量 KV cache）+ SnapKV / H2O 两种 KV cache 压缩方法。标准 attention 流程：所有 m 层 Transformer 对全部 n 个 input token 做 prompt computation（计算完整 KV cache），然后在 iterative generation 阶段使用预计算的 KV cache 逐 token 生成。SnapKV 和 H2O 流程：在 prompt computation 阶段同样计算全部 n 个 token 的完整 KV cache（与标准 attention 相同的 Θ(mhn²d) 计算量），但仅选择性保留部分 KV cache（如 k=1024 个 token）供 generation 阶段使用——通过不同策略选择重要 KV：SnapKV 利用 observation window 内 token 的 attention pattern 聚类选择，H2O 基于累积 attention scores 贪心地保留 heavy-hitter token。

  全栈执行例子（LLaMA 3.1 8B Instruct, n=128K, k=1024）：
  - 算法pipeline：标准 attention（all KV cache）→ 所有 32 层均处理全部 128K token → 每层产生 128K × (64×8) × 2 = ~32MB KV cache → 32 层共 ~1GB KV cache。SnapKV/H2O 同流程，在每层/每头后额外执行 token selection heuristic → 最终每层保留 k=1024 个 KV pair，但 prompt computation 仍完整执行。
  - 系统框架：HuggingFace v4.43 PyTorch 推理 pipeline，FlashAttention-2 加速 attention 计算，标准 causal generation (greedy, num_beams=1)。
  - 编译框架：论文未明确说明。
  - kernel调度：FlashAttention-2 kernel 处理 attention 计算（QK^T + softmax + PV），标准 tiling + recomputation 策略。无自定义 kernel。
  - 硬件架构：NVIDIA A100-40GB（双卡，因单卡无法容纳 128K 的 full KV cache），H100-80GB（timing 实验）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法（GemFilter）利用关键发现——LLM 在早期层（如 LLaMA 3.1 的第 13 层）的 attention 矩阵中即可定位与 query 相关的 token——设计了一种两遍推理策略：
  
  **Baseline 缺陷 1**：Standard attention、SnapKV、H2O 在 prompt computation 阶段都必须运行全部 m 层处理全部 n 个 token，时间复杂度和 GPU 内存消耗与被压缩前的全部输入成正比。即使 SnapKV/H2O 压缩了 KV cache，prompt computation 的计算量并未减少。
  → **GemFilter 解决**：第一遍仅运行前 r 层（r << m，如 r=13 vs m=32），这两层处理全部 n 个 token 仅仅是为了识别重要 token；prompt computation 的时间复杂度从 Θ(mhn²d) 降至 Θ(rhn²d)，约节省 60% 的计算量。GPU 内存只需加载前 r 层的权重（rw vs mw）。

  **Baseline 缺陷 2**：SnapKV/H2O 为每层每头维护独立的 token 索引集（m·h 套索引），不仅增加了元数据开销，还使得选中的 token 序列难以被人类理解——不同层/头选中的 token 不一致，无法给出一个统一的"模型关注什么"的解释。
  → **GemFilter 解决**：使用单一 token 索引集 J（仅从 filter layer 的所有 head 聚合 attention scores 后取 top-k），压缩后的 token 序列 T_J 是人类可读的完整文本。例如在 Figure 1 中，GemFilter 选中的 100 个 token 包含完整的 initial instruction、key message 和 query——用户可以直接检查模型是否关注了正确内容。这是可解释性优势。

  **Baseline 缺陷 3**：SnapKV/H2O 保留原始长上下文的位置编码（position embedding distance = n + t），导致模型仍需处理长距离的 RoPE 编码。
  → **GemFilter 解决**：第二遍推理时输入长度从 n 降为 k（如 128K→1024），RoPE 重新计算，最大位置编码距离从 n+t 降为 k+t，使模型在更短、更自然的输入分布上生成，有助于提高质量。

  **Baseline 缺陷 4**：H2O 的累积 attention score 策略与 FlashAttention 不兼容（FlashAttention 不物化完整 attention matrix），因此 H2O 无法处理超长输入（本论文因此将 H2O 排除在 Needle in a Haystack 对比之外）。
  → **GemFilter 解决**：GemFilter 仅需要 filter layer 的 attention scores 做 token 选择（在 FlashAttention 中可以通过一次额外的前向 pass 获得），与 FlashAttention 兼容，可处理 128K 输入。

  全栈执行例子（GemFilter, LLaMA 3.1 8B, n=128K, r=13, k=1024）：
  - 算法pipeline：**第一遍**——前 13 层做 forward pass on 128K tokens → 第 13 层取得所有 head 的 attention scores → 取最后一 query token 对所有 key token 的 scores → 跨 head 求和 → 1D avg_pooling (kernel=5) → top-k=1024 索引 J → 排序回原始顺序。**第二遍**——构造 T_J（仅 1024 个 token）送入完整 32 层 LLM → 标准 greedy generation。关键张量形状变化：第一遍 attention score [1, h, 1, n=128K] → pooling + topk → [1, 1, 1, k=1024]；第二遍整个 forward 的序列长度仅为 k=1024（vs baseline 的 128K）。
  - 系统框架：HuggingFace v4.43 PyTorch + FlashAttention-2（仅支持标准 attention 部分，GemFilter 改动了 forward pass 调用模式——两次 forward，第一次仅前 r 层，第二次完整模型但输入缩短）。
  - 编译框架：论文未明确说明。
  - kernel调度：FlashAttention-2 kernel——第一遍中处理 128K 长度但仅前 r=13 层；第二遍处理 k=1024 长度的所有 m=32 层。整体 kernel 调用量远少于 baseline（第一遍减少 (m-r) 层长序列 attention，第二遍减少全部层长序列 attention 替换为短序列 attention）。
  - 硬件架构：NVIDIA A100-40GB（双卡）——GemFilter 在第一遍仅需加载前 13 层权重到 GPU（rw vs mw），第二遍加载 32 层权重但 sequence length 仅 1024。实测 GPU 内存减少 30%（vs SnapKV）和 70%（vs Standard）。

  对应解决的完整映射：
  - Baseline 缺陷 1（prompt computation 计算量过大）→ 仅运行前 r 层处理长输入：prompt time Θ(mhn²d)→Θ(rhn²d)，GPU mem mw+2mhnd→rw+2hnd
  - Baseline 缺陷 2（m·h 套索引不可解释）→ 单索引集 J：用户可直接打印 T_J 审查，提供可解释性
  - Baseline 缺陷 3（长距离 RoPE 编码）→ 第二遍短输入：position distance 从 n+t 降为 k+t，分布更自然
  - Baseline 缺陷 4（H2O 与 FlashAttention 不兼容）→ GemFilter 与 FlashAttention 兼容，可处理 128K+ 上下文

## Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

- baseline方法是什么？
  Baseline 是 Full-context LLaVA-1.5（7B/13B）和现有的仅视觉 token 缩减方法（以 FastV 为代表）。其全栈执行例子如下：
  - **算法层**：LLaVA-1.5 在 prefill 阶段使用 CLIP ViT-L/14@336px 将图像编码为 576 个 image token，与 text token 拼接送入 Vicuna-LLM 的 32/40 层 decoder。Prefill 阶段计算：S_{l+1}^P = FFN(MHA(S_l^P, S_l^P, S_l^P))，其中 S_l^P = S_l^I ∪ S_l^T。FastV 等方法在第 2 层之后基于 attention scores 选择保留 k=3 个 attention heads 中 attention score 最高的 r=0.75 比例 image token，将 576 token 减至 144 token。**核心缺陷**：(a) 仅稀疏化 vision context——Eq. 4 表明随着输出 token 增长，Computation(Decoding_w/o_cache)_l ∝ |S_l^{OT}| → ∞，Memory(Decoding_w/cache)_l ∝ |S_l^{OT}| → ∞，image token 减少的收益在 decoding 阶段逐渐湮没；(b) 仅作用于 prefill 阶段一次，对 decoding 阶段无持续优化；(c) 现有 LLM KV cache 压缩方法 H2O（基于 attention score 丢弃历史 KV cache）在混合模态场景下严重退化——丢弃多模态混合的 KV cache 导致 SciQA 下降 16.3%、MMBench 仅 1.4；(d) 无 batch-parallel 稀疏化推理优化。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers 的 LLaVA 推理 pipeline。FastV 在第 2 层后插入 token pruning 操作（TopK attention scores across heads）。H2O 在每次 decoding step 计算 Q 与历史 KV cache 的 attention scores 以决定 eviction。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Standard PyTorch MHA + FFN 实现。FastV 的 token pruning 涉及跨 head attention score 聚合 + TopK + index gather（在 GPU 上以 PyTorch op 实现）。
  - **硬件架构层**：1× NVIDIA A100 (80G)。LLaVA-1.5-13B 在 batch=8 生成 4K tokens 时 OOM（Table 4）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Dynamic-LLaVA 通过三个核心设计系统性地解决 baseline 缺陷：

  **1. 同时稀疏化 Vision 和 Language Context → 解决缺陷(a)**：
  Baseline 仅稀疏化 vision token S_l^I，Dynamic-LLaVA 同时稀疏化 vision token S_l^I 和 output text token S_l^{OT}。在 prefill 阶段 image predictor 将 576 个 image token 减至 ~115（保留 r^I=20%）；在 decoding 阶段 output predictor 将 output text token 减至 ~50%（保留 r^OT=50%）。Eq. 4 中的三项目标同时缩小：Computation(Prefill)_l ∝ |S_l^{I*}| ≪ |S_l^I|，Computation(Decoding_w/o)_l ∝ |S_l^{OT*}| ≪ |S_l^{OT}|，Memory(Decoding_w/)_l ∝ |S_l^{OT*}| ≪ |S_l^{OT}|。实际效果：decoding w/o KV cache TFLOPs ↓~50%，decoding w/ KV cache GPU memory ↓~50%。

  **2. 可学习 Predictor + 端到端训练（MaskedSoftmax + Gumbel-Softmax + STE）→ 解决缺陷(b)+(c)**：
  使用两个轻量神经网络 predictor（image predictor: 2×ViT blocks + MLP 512→256→128→2；output predictor: MLP 512→256→128→2），从第 l=2 层 token 特征直接预测 keep/discard 决策，不依赖 attention scores 的启发式规则。训练时：(i) MaskedSoftmax（Eq. 7）替代直接置零——在不破坏自回归过程的前提下，通过 mask 矩阵 G 隔离非必要 token 对必要 token 的 attention 影响；(ii) Gumbel-Softmax（τ: 1→0.1 衰减）+ STE 解决 argmax 不可微问题；(iii) 约束正则项 R（Eq. 10）使 mask 保留率接近 r^I 和 r^OT。这使得 predictor 能端到端学习哪些 token 对最终任务重要，避免了 H2O 在混合模态场景下的严重退化。

  **3. 三模态定制化稀疏推理 + Batch-Parallel 策略 → 解决缺陷(d)**：
  针对三种推理模式分别设计：prefill → 仅 image token 稀疏化（Eq. 5）；decoding w/o KV cache → vision + language 稀疏化（Eq. 2 修改）；decoding w/ KV cache → output predictor 逐 token 决定 KV 是否加入 cache（Eq. 6，在线 KV cache 压缩）。通过 LeftPadding + TopkArgmax（Eq. 11-12）实现 mini-batch 内变长 token 集合的并行预测和 GPU 批量计算。

  全栈执行例子（Dynamic-LLaVA-13B_{I|T}，1×A100 80G，batch=8，生成 2K tokens）：
  - **算法层**：第 1-2 层处理完整 576 image tokens → 第 2 层后 image predictor P^I 输出 D^I ∈ R^{576×2} → argmax 生成 M^I → TopkArgmax 保留 r^I=20% (~115) image token → 剩余 38 层仅处理 115 image tokens + text tokens。Decoding with KV cache：每个 output token 的 embedding 经 P^{OT} → argmax → M^{OT}_{N^{OT}} ∈ {0,1} → 决定该 token 的 K,V 是否加入 KV cache（Eq. 6）。关键张量变化：prefill attention 从 [B, 576+text_len, 4096] 变为 [B, 115+text_len, 4096]；KV cache 从 |S^{OT}| tokens 的 KV 减至 ~0.5|S^{OT}| tokens 的 KV。实际数值：prefill 0.83s→0.37s，decoding 4117s→2382s，GPU mem 58G→42G。
  - **系统框架层**：基于 LLaVA-1.5 PyTorch 代码库。在第 2 层 decoder 后插入 predictor 调用（约 1% 额外计算）。Predictor 决策共享至所有后续层。Batch-parallel 通过 LeftPadding + TopkArgmax 实现。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：所有操作基于 PyTorch 原生 op（Linear, Attention, FFN）。Predictor 为小型神经网络（ViT blocks + MLP），计算开销 <1%。没有自定义 CUDA kernel。
  - **硬件架构层**：1× NVIDIA A100 (80G)。训练：8× A100 (80G)。LLaVA-1.5-13B baseline 在 batch=8 生成 4K tokens 时 OOM，Dynamic-LLaVA 可完成 4K tokens 生成（仅 56G GPU memory）。

  对应解决的完整映射：
  - Baseline 缺陷(a)（vision sparsification 收益在 decoding 中湮没）→ 同时稀疏化 vision + language context：decoding TFLOPs ↓~50%，GPU memory ↓~50%，PPL 仅增加 <0.3
  - Baseline 缺陷(b)（无 decoding 阶段持续优化）→ Output predictor 在每次 decoding 动态决策，持续优化整个生成过程；长输出场景（ShareGPT4V-VQA，平均 1555 tokens）收益更显著（Table 8）
  - Baseline 缺陷(c)（H2O 在混合模态下退化）→ 可学习 predictor 端到端训练，不依赖 attention scores；保留 ratio=50% 的 KV cache 在 LVIS-VQA 上 PPL=4.90 vs H2O 的 78.95，METEOR=0.3108 vs H2O 的 0.0381
  - Baseline 缺陷(d)（无批量并行优化）→ LeftPadding + TopkArgmax 实现 batch-parallel 稀疏化推理，batch=8 时可充分利用 GPU 并行度

## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- baseline方法是什么？
  Baseline 是现有的 hybrid attention 方法（DuoAttention、PruLong、InfLLM-V2）以及静态稀疏注意力方法（MoBA、NSA），它们的核心问题是使用**固定计算比例**（static computation ratios），无法根据输入任务动态调整稀疏度。全栈执行例子如下：

  - **算法pipeline层**：DuoAttention 在训练时将 attention heads 分为 retrieval heads（FA）和 streaming heads（SA），head 分配在训练后固定，推理时所有任务使用相同的 Ω_MSR（如始终 0.70）。PruLong 使用 hard concrete reparameterization + Lagrangian penalty 学习 head 二分类，head 分配同样训练后固定。MoBA/NSA 将 attention 计算限制在预定义的 block/chunk 内，sparsity pattern 由固定超参数（block_size, top-k）决定。这些方法在 sparsity-sensitive tasks（如 QA）和 sparsity-robust tasks（如 summarization）上使用相同的计算模式，导致要么 sparsity-sensitive 任务性能下降（sparsity 过高时），要么 sparsity-robust 任务计算冗余（sparsity 过低时）。实验表明：随着 Ω_MSR 从 0 增至 1.0，sparsity-sensitive 任务（Single-Doc QA）性能从 100% 降至 56%，而 sparsity-robust 任务（Code）性能始终 >93%。

  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline。所有 baseline 的 head 分配模式在训练后固定，推理时不随输入任务变化。DuoAttention/PruLong 的 retrieval/streaming head mask 是预先计算好的常数，MoBA/NSA 的 block selection 由固定规则决定。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：DuoAttention 使用 FlashAttention-2 分别对 retrieval 和 streaming heads 执行 attention，需要两次 kernel launch。MoBA/InfLLM-V2 需 reserve 部分计算预算用于 pre-compute sequence-level features。部分 baseline（如 NSA、InfLLM-V2）对 KV attention heads 数量有严格可除性要求（如 16 的倍数），与 Llama-3.1-8B 的 GQA 配置不兼容。

  - **硬件架构层**：单 NVIDIA A800/A100 GPU。MoBA 和 InfLLM-V2 在 256K context 下因 sequence-level feature pre-computation 导致 GPU OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 Elastic Attention 通过**test-time 自适应稀疏度分配**解决 baseline 的静态计算比例问题。核心观察：下游任务自然分为两类——sparsity-robust（粗粒度上下文即可完成，如 summarization）和 sparsity-sensitive（需细粒度证据检索，如 QA）。基于此，引入轻量级 Attention Router 根据输入动态调整每个 head 的计算模式。

  全栈执行例子（Elastic Attention on Llama-3.1-8B-Instruct, FA-SSA setting）：

  - **算法pipeline层（核心创新）**：
    (a) **Attention Router**（每层 0.27M 参数）：接受 Key hidden states x_K，通过 Boundary Pooling（仅取序列首部和尾部各 100 tokens）得到 task representation → Task MLP 提取 task-specific 特征 → Router MLP 输出 head-wise 二值路由决策 r_hard ∈ {0,1}。r=0 的 head 使用 FA (retrieval)，r=1 的 head 使用 SA (sparse)。训练使用 Gumbel-Softmax 连续松弛 + STE 解决 argmax 不可微问题，温度 τ 从 τ_init 指数衰减至 τ_min。
    (b) **Lagrangian 约束训练**：min-max 优化 L = L_language + λ1·(Ω_MSR - t) + λ2·(Ω_MSR - t)²。t 为 task-dependent 非紧约束（sparsity-sensitive t=0.7, sparsity-robust t=1.0），λ 为可训练 Lagrange 乘子。backbone 参数完全冻结，仅训练 Router（12h on 8×A800）。
    (c) **任务自适应性**：训练时 Router 从随机初始化自动学会区分 sparsity-sensitive 和 sparsity-robust 任务——训练曲线显示 Code 和 ICL 任务的 Ω_MSR 收敛到 ~0.80-0.85（较低 sparsity = 保留更多 FA），QA 任务的 Ω_MSR 收敛到接近 t (~0.65-0.7)。推理时 Router 根据输入的 task representation 动态分配每个 head 的计算模式，无需 task label。

  - **系统框架层**：基于 PyTorch + HuggingFace Transformers + LOOM-Eval 推理框架。Router 作为可插拔模块集成到已有预训练模型中，无需修改 backbone 参数。Router 的 pooling 仅处理 boundary tokens（100+100），复杂度独立于序列长度。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：实现基于 Block Sparse Attention (BSA) Kernel 的 fused hybrid attention kernel。将 routing decisions m 直接传给 kernel 作为 metadata，kernel 内部通过 thread-block level branching 判断每个 head 的类型并执行对应的 attention 逻辑。Single kernel launch 处理所有 heads，消除 tensor splitting 的内存开销和多次 kernel launch 的调度开销。保持 grid 完整性（Batch×Heads×Sequence Blocks），GPU SM 可最优调度。

  - **硬件架构层**：8× A800 训练，单 GPU 推理。Router 延迟仅 ~0.196ms/router call，且不随序列长度增长。

  **对比 baseline 的关键差异**：
  - Baseline 静态 sparsity → Elastic Attention test-time 自适应 Ω_MSR：同一模型在 Code 任务上 Ω_MSR ~0.82（高 sparsity），在 QA 任务上 Ω_MSR ~0.68（低 sparsity），动态匹配任务需求。LongBench-E 上 Elastic Attention (FA-SSA) 在 Llama-3.1-8B 上平均分 53.35 > backbone 53.28（因 sparsity 减少 attention dispersion 反而可能提升性能）。
  - Baseline 需 per-task 调参 → Elastic Attention 自动分配：训练时仅需设置两类 target sparsity（t_robust=1.0, t_sensitive=0.7），无需 per-task 超参搜索。
  - Baseline kernel inefficiency（多次 launch + tensor split）→ Elastic Attention fused BSA kernel：prefill 阶段加速显著，减少 memory copy 和 kernel launch overhead。
  - Router 的 Task MLP 隐式学会任务判别：通过对 hidden states 的 pairwise cosine similarity 分析，经过 Task MLP 后不同任务的 representation 被映射到近似正交的子空间（M_uv ≈ 0），使 Router MLP 能做出准确的 head 分配决策——尽管训练中从未提供 explicit task label。
  - 极低训练成本：12h on 8×A800 vs 同类方法需更新 backbone 参数，且 Elastic Attention 冻结 backbone 保持原有能力。
  - RULER 长度外推：256K context 下 FA-XA 仍保持 68.51（Llama-3.1-8B）vs MoBA/NSA 在 128K+ 严重退化（near-zero accuracy）。
  - 与 XA-SSA（全部 head 用 SA）的 scalability：Qwen3-4B 在三个 benchmark 上平均性能差距 <1 点（48.45 vs 48.14），证明全 SA 配置在极限压缩下仍可控。

## Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

- baseline方法是什么？
  Baseline 是 Full Attention（dense attention, 完整 KV cache on GPU）+ KV cache offloading（FlexGen）+ cache eviction 方法（StreamingLLM, H2O），其全栈执行例子如下：
  - **算法层**：Full Attention 对所有 token pair 计算 O(N²) 的 attention matrix，KV cache 占用 O(N·D·L) GPU 显存（BF16 下 Llama-3 8B 在 N=100K 时 KV cache = 52GB，超出 commodity GPU 16GB 内存）。FlexGen 将 KV cache offload 到 CPU 内存并在每个 decoding step 将数据搬回 GPU，但 8B 模型单层 KV cache 为 1.6GB，需往返搬运数百次，数据移动成本本身即为瓶颈。StreamingLLM 通过 sliding window attention + attention sink（保留最初 4 个 token）维持固定 KV cache 大小，但 evict 中间 token 会导致长上下文任务失败（无法检索 window 之外的 token）。H2O 使用累积 attention scores 识别 heavy-hitter token 优先保留，基于历史信息 evict token，但策略非 query-aware——一旦 token 被 evict 即不可恢复，后续 query 可能因需要已 evict token 而失败。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline。FlexGen 实现 CPU offloading engine（quadratically scheduled offloading）；StreamingLLM/H2O 在 attention 层插入 KV cache 管理逻辑（eviction policy）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention（Dao et al., 2022）减少 attention matrix 的 GPU memory footprint（tiling + recomputation），但仍需对所有 key 计算 attention scores（事后可在 softmax 后丢弃无关 scores，但 score computation 本身已全部执行）。FlexGen 涉及 CPU↔GPU 数据传输调度。vLLM PagedAttention 优化多请求吞吐量。
  - **硬件架构层**：Commodity GPU（~16GB VRAM）或 A100/H100（datacenter）。Full attention 在 commodity GPU 上因 O(N) KV cache 爆炸导致 OOM；FlexGen 因 CPU-GPU 带宽瓶颈导致吞吐量极低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出基于 attention 稀疏性的 top-k attention 机制，核心是通过 Faiss 向量数据库在 CPU 上执行 approximate nearest neighbor search（ANN），仅将 top-k 个最相关的 key-value pair 从 CPU 传输到 GPU 进行 attention 计算，从根本上解耦 attention score computation 和 FFN computation。

  **核心创新 1：Query-aware top-k selection via ANN search → 解决 cache eviction 的 query-unaware 缺陷**
  Baseline（StreamingLLM/H2O）的 eviction 策略对所有 query 使用相同的缓存，无法感知不同 query 的关注点差异。Top-k attention 对每个新 query 向量在 CPU 端执行 ANN search（inner product metric 直接对应 attention score），动态选择与该 query 最相关的 k 个 key，不受历史 eviction 影响。实验中 1M token NIAH：k=1 即可 100% 成功检索任意位置的 needle（vs StreamingLLM 因 token 被 evict 而完全失败，early sink tokens + local window 之外无法检索）。

  **核心创新 2：CPU-GPU 解耦，仅传输 k 个 value → 解决 FlexGen 的全量数据搬运瓶颈**
  Baseline FlexGen 每 step 将所有层的完整 KV cache 在 CPU↔GPU 间搬运（8B 模型单层 1.6GB×L=32≈51GB 需搬运数百次），数据移动本身成为瓶颈。Top-k attention 每层仅传输 k 个 value 向量（k << N），单层数据传输量从 O(N·D) 降至 O(k·D)。k=2% of N 即可保持 95% dense attention 性能，数据搬运量降低 50×。

  **核心创新 3：分离 prefill 和 decoding 的计算资源需求 → 解决 commodity GPU 无法处理长上下文的瓶颈**
  Prefill 阶段（一次性，高计算需求）使用 H100 GPU + FlashAttention + chunking strategy 构建完整 KV cache；decoding 阶段（重复执行，低成本）在 commodity GPU（~16GB VRAM）上仅需 O(k) GPU memory。将一次性高成本计算 amortize 到多次低成本的 query 上，实现 "rent cloud compute for prefill, run queries on local hardware" 的使用模式。

  **核心创新 4：Layer-wise adaptive k budget → 利用不同层的 attention 稀疏度异质性**
  观察到 attention entropy 随 layer depth 增加而降低（第一层 entropy 最高，后续层 attention 更集中）。利用此现象，在 fixed total k budget 下按 linear increasing from first to last layer 分配 k_ℓ，在不增加总计算量的前提下获得 non-trivial RULER performance boost（vs uniform k 分配）。

  **全栈执行例子（Llama-3-8B, 1M token context, commodity ~16GB GPU）**：
  - **算法层（核心创新）**：
    (a) Prefill（一次性）：FlashAttention on H100 计算所有 32 层的 KV cache → 存储于 CPU host memory。
    (b) Index 构建：对每层每个 head 的 key 向量构建 Faiss IndexFlatIP（exact inner product search）。
    (c) Decoding loop：每 token → GPU QKV projection → q 传 CPU → Faiss ANN search（inner product 距离）→ k 个 indices → 传输对应的 V[I] 和 vals 到 GPU → softmax(vals/√d_k)·V[I] + GPU 本地 window attention（已生成 token 的 KV on GPU）→ 合并输出。
    (d) k 选择：k = 2% of N 实现 >95% dense attention 性能；k = 0.001% 足以完成 NIAH 任务。
  - **系统框架层**：PyTorch + Faiss (Facebook AI Similarity Search) 作为 vector database + FlashAttention（prefill）。支持 exact (IndexFlatIP) 或 approximate (HNSW, IndexHNSWFlat) ANN search。无额外 serving framework 修改，实现为独立的 generation script。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Standard PyTorch attention kernel on GPU（仅计算 k 个 token 的 attention）；Faiss CPU-based ANN search kernel；CPU↔GPU 数据拷贝（仅 k·D 元素，用户可控）。与 FlashAttention 兼容（prefill 阶段使用），对 decoding top-k 部分无特殊 kernel 要求。
  - **硬件架构层**：Prefill: 1× H100 (80GB)；Decoding: 1× commodity GPU (~16GB VRAM) + CPU host memory。1M token context 解码仅需 ~16GB GPU RAM（vs full attention 的 ~520GB KV cache 需求）。

  **对比 baseline 的关键差异**：
  - Baseline Full Attention GPU O(N) → Top-k O(k) GPU memory，k=0.001%-2% of N
  - Baseline FlexGen 全量 KV cache 往返搬运 → Top-k 仅传输 k 个 V vectors，数据搬运量降 50×+
  - Baseline StreamingLLM/H2O eviction query-unaware → Top-k per-query ANN search 动态选择
  - Baseline StreamingLLM 在 1M token NIAH 上完全失败 → Top-k k=1 即 100% 成功（Figure 8 红色 cells vs top-k）
  - Baseline 平均分配计算 budget across layers → Top-k 支持 layer-wise adaptive k，在固定 total budget 下获得 ~2% 性能提升
  - 方法通用性：支持任意 Faiss index（IndexFlatIP, IndexHNSWFlat 等），与模型/架构无关（无需训练或微调），已在 Llama-1/2/3/3.1/3.2 全系列验证

## FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

- baseline方法是什么？
  Baseline 分为两类：(1) **仅解码加速方法**（StreamingLLM、H2O、SnapKV）：预填充阶段处理完整上下文构建完整 KV cache，解码阶段根据注意力重要性 score 裁剪 KV cache。它们不减少预填充计算量，在长上下文（128K）下预填充时延占总时延主导部分，且 H2O 因需导出完整 attention map 而无法使用 FlashAttention-2、在 8K+ 上下文即 OOM。(2) **预填充感知加速方法**（GemFilter、PyramidInfer）：在预填充阶段减少处理的 token 数量。GemFilter 在单个 filter layer 处从完整上下文中选择关键 token，然后仅用这些 token 重新预填充所有层——导致早期层（注意力分布各异）被迫使用同一 token 子集，信息丢失严重；PyramidInfer 从第一层即开始按 cosine schedule 逐步减少 token，在上下文稳定性建立之前就过早丢弃 token。这两类 baseline 的核心缺陷是**预填充计算量减少与 KV cache 预算刚性耦合**——要减小 KV cache 必须更激进地减少预填充计算，导致准确率大幅下降。

  **Baseline 全栈执行例子（以 GemFilter 为例，LLaMA-3.1-8B，128K 输入）：**
  - **算法层**：Embedding → Layer 0-13 full-context prefill（计算 128K×128K attention map）→ Layer 13 收集所有 head 的 attention scores → 跨 head 平均计算 saliency → TopK 选择 20% token（25600 tokens）→ 丢弃其余 102400 tokens 的信息 → Layer 0 到 Layer 31 仅用 25600 tokens 重新 prefill → 输出 logits。被丢弃 token 的语义信息永远不会被任何层处理。
  - **系统框架层**：HuggingFace Transformers + FlashAttention-2 kernel。GemFilter 需执行两轮 prefill（首次 128K + 二次 25.6K），相当于 1.6 轮完整前向传播。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 on A100 SXM。首次 prefill 计算完整 128K attention；二次 prefill 仅计算 25.6K。FlashAttention-2 的 tiled 计算和 SRAM 优化对两次 prefill 均适用。
  - **硬件架构层**：单张 NVIDIA A100 SXM GPU（80GB）。KV cache 在 GQA 下每层每 token 约占用 2×8 KV heads × 128 head_dim × 2 bytes = 4KB/token/layer（FP16），128K×32 层 ≈ 16.4GB。压缩至 20% 后约 3.3GB。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FastKV**，核心设计：(1) **Token-Selective Propagation (TSP)**：在模型中部的 TSP 层（LLaMA-3.1-8B 的 layer 15），基于 window tokens 的注意力权重计算每 token 的 saliency score，仅向后续层传播 top-R_TSP 个 token 的 hidden states——但早期层（0 到 TSP）保持完整上下文计算，确保每个早期 layer 可自由关注其偏好的 token 子集。(2) **TSP rate 与 KV retention rate 完全解耦**：TSP rate 控制预填充计算量（等于 1 - Σ_{l>TSP}(1 - R_TSP)，约 60%），KV retention rate 独立控制解码时每层保留的 KV cache 比例（10% 或 20%），二者互不约束——可在保护准确率的同时激进压缩 KV cache。

  **如何解决 Baseline 缺陷：**
  - **vs GemFilter**：GemFilter 的 filter layer 决定所有层使用同一 token 子集 → FastKV 的前 TSP 层保留完整上下文，每层可独立关注不同 token；GemFilter 中丢弃 token 的信息完全丢失 → FastKV 中被 TSP 丢弃的 token 已在早期层的注意力计算中将其语义融合到传播 token 中（Figure 7 可视化）。GemFilter 预填充=KV retention → FastKV 完全解耦两个比率。
  - **vs PyramidInfer**：PyramidInfer 从 layer 0 即开始减少 token → FastKV 仅在 layer 15（上下文稳定后）开始 TSP。PyramidInfer 不压缩 KV cache（KV retention = prefill compute rate）→ FastKV 独立设为 10%，大幅减小解码时延。
  - **vs SnapKV/StreamingLLM/H2O**：这些方法预填充完整上下文后压缩 KV → FastKV 在预填充阶段即减少后续层的计算量，同时独立压缩 KV cache，同时加速预填充和解码。

  **FastKV 全栈执行例子（LLaMA-3.1-8B，128K 输入，R_TSP=0.2，R_KV=0.1）：**
  - **算法层**：Embedding → Layer 0-15 全量 prefill（128K 上下文，构建 K_X, V_X）→ 每层完成 prefill 后立即执行 KV_Compress：对每 KV group 计算 group-wise saliency（head-wise attention scores 在 group 内平均），保留 top-10% 关键 token 的 KV entries → Layer 15 同时执行 TSP：基于最后 N_obs=8 个 window tokens 的注意力权重，MaxPooling(kernel=7)+跨 head 平均计算 saliency score → 选择 top-20% token 的 hidden states + 所有 window token → 仅传播这 25608 个 hidden states 到 layer 16 → Layer 16-31 在 25608 个 token 上计算注意力并各自压缩 KV cache → LMHead 输出 logits。总 prefill FLOPs ≈ 完整上下文 × (15/32 + 17/32×0.2) ≈ 60%。
  - **系统框架层**：HuggingFace Transformers self-attention 层被修改：在 TSP 层增加 HiddenCompress 步骤；每个 decoder layer 增加 KV_Compress 步骤。与 FlashAttention-2 完全兼容（不使用需要导出完整 attention map 的操作）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 on A100 SXM。TSP 层的 saliency scoring 仅基于 8 个 window token query 的行进行 MaxPooling + averaging，不加载完整 attention map，额外开销仅 0.15s（128K 下占总 prefill 0.88%）。KV compression 后每层仅需存储和访问 10% 的 KV entries，解码时 attention 的 memory-bound 瓶颈显著缓解。
  - **硬件架构层**：单张 NVIDIA A100 SXM（80GB）。128K 输入下：prefill 阶段，layer 0-15 计算完整 128K attention → 每层 KV cache 压缩至 12800 tokens → 总 KV cache ≈ 128K×15×0.1 + 128K×0.2×17×0.1 ≈ 2.4GB（vs 完整 16.4GB）。128K 上下文生成 256 token 时：端到端时延从 18.81s 降至 6.63s（约 2.84× 加速）。

  **FastKV 通过观察 "早期层注意力不稳定、后期层注意力收敛" 的层依赖上下文动态特性，将 TSP 和 KV compression 置于正确的时机——早期保留完整上下文满足每层异构注意力需求，后期识别并传播稳定关键 token 获得预填充加速——同时将两个压缩比例解耦，实现了第一个同时加速预填充和解码且保持高准确率的 KV cache 压缩框架。**

## GTA__Grouped-head_latenT_Attention

- baseline方法是什么？
  现有高效注意力机制在效率与表达力之间存在根本权衡：

  (1) **MHA**：每个 head 独立计算 Q_i K_i^T 并独立存储 K_i、V_i。KV cache = 2n_h d_h N（如 1B 模型的 2560 dims/token/layer），预填充 FLOPs = 2n_h d_h N^2 + 2H^2 N。表达力最强但因 KV cache 和注意力计算随序列长度线性/二次扩张，长文本推理受限于显存和计算带宽。解码时每个新 token 需加载全部历史 K、V 计算 attention，I/O 密集。

  (2) **GQA**：将 heads 分为 n_k 个 KV groups，group 内共享 K、V。KV cache = 2n_k d_h N（低于 MHA），但注意力计算仍为 n_h d_h N^2（每个 head 仍独立计算 QK^T）。缺陷：KV cache 节省来自减少 KV head 数，但 attention 计算未减少；且共享 key-value 会损失 attention 粒度，在下游任务上可能退化。

  (3) **MLA**：引入低秩联合压缩 latent vector c^{KV}，将 K、V 压缩至低维（d_c），再通过 up-projection 解压回各 head 的 K_i、V_i。KV cache = (d_c + d_{rope})N，显著减少。缺陷：解压需要 n_h 次 up-projection（W_{UK,i} c^{KV} 和 W_{UV,i} c^{KV}），prefill 时线性计算项 O(n_h d_c d_{nope} N) 较重；decode 时仍需为每个 head 从 latent vector 解压 K 和 V，计算开销限制了其在资源受限设备上的部署。

  **Baseline 全栈执行例子（以 GQA-1B 为例，n_k=5, d_h=64, H=1280, N=2048）：**
  - **算法层**：X → Q=XW_Q (N×1280)、K=XW_K (N×320, 5 groups × 64)、V=XW_V (N×320) → 每个 head i 从 Q 取第 i 组 (64 dims)，从 K 取第 i mod 5 组 (64 dims)，从 V 取第 i mod 5 组 (64 dims) → 计算 score = Q_i K_{i mod 5}^T / 8 → softmax → O_i = softmax @ V_{i mod 5} → W_O 投影 → 求和。FLOPs_attention = 2 × 20 heads × 64 × 2048^2 = 10.7G FLOPs。KV cache = 2 × 5 × 64 × 2048 = 1.3M elements/layer。
  - **系统框架层**：PyTorch + HuggingFace Transformers（论文使用 transformers v4.36.0 进行实际推理评估）。
  - **编译框架层**：论文未明确说明（训练直接使用 PyTorch 自动微分，推理使用 transformers DynamicCache / OffloadedStaticCache）。
  - **kernel调度层**：论文未明确说明（未开发自定义 kernel，依赖 PyTorch 默认 GPU kernel 和 transformers 内置 attention 实现）。
  - **硬件架构层**：NVIDIA A800 80GB（训练），NVIDIA H100 80GB（推理模拟和实际测试）。GQA-1B 在 H100 上，2048 token prefill 时延约 100ms，decode 128 tokens 需约 10s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **GTA (Grouped-head latenT Attention)**，通过两个核心设计打破效率-表达力权衡：

  **(1) Shared Attention Map（共享注意力矩阵）**：将 query heads 和 key heads 分别分组成 n_q 和 n_k 组（n_q << n_h），同一 Q group 内的 heads 复用相同的 QK^T 计算。**解决 GQA 缺陷**：GQA 虽共享 K/V 以减少 cache，但 attention 计算仍是每个 head 独立 QK^T → GTA 将 QK^T 计算次数从 n_h 降至 n_q，预填充 FLOPs 从 2n_h d_h N^2 降至 n_q (d_h + d_l) N^2。同时 GTA 的 key 共享 + attention 共享比 GQA 更激进，KV cache 更小。

  **(2) Nonlinear Value Decoder（非线性值解码器）**：引入压缩 latent value C ∈ R^{N × n_c × d_l}（共享 latent 空间），每个 head 的 V_i 由 (C @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i}) 动态生成，而非直接存储独立的 V_i。**解决 MLA 缺陷**：MLA 的解压需要为每个 head 做两次 up-projection（key 和 value 各一次）→ GTA 仅需一次 head-specific 投影（W_{P,i}）加一个轻量级 gate（W_{G,i} 输入仅为当前 token x_t），且 decode 时无需从 latent 解压所有 history value（Eq 8 将 attention 放在 latent space 计算）。Gate 的 Sigmoid 非线性确保 value 表示具有高有效秩（full-rank projection），相比 MLA 的纯线性解压表达力更强。论文的消融实验证实：Sigmoid > Silu > ReLU²，因稀疏激活降低了有效秩。

  **解决 MLA 的 Prefill 计算重问题：** MLA prefill 线性项含 (d_c+d_{rope})NH + n_h(d_{nope}+d_{rope})NH + 2n_h d_c d_{nope} N，项数多且最后一项与 n_h 正比。GTA prefill 线性项为 2NH^2 + (n_q+n_k+n_c d_l + d_l)NH，无与 n_h d_c d_{nope} 等价的项——因 GTA 无解压-Upprojection 步骤，仅需直接投影 + gate。论文 Table 4 对比表明 GTA 的 attention 计算为 n_q(d_h+d_l)N^2，MLA 为 n_h(d_{rope}+2d_{nope})N^2，当 n_q << n_h 时显著更低。

  **GTA 全栈执行例子（GTA-1B，n_h=20, n_q=5, n_k=1, n_c=1, d_h=64, d_l=128, N=2048）：**
  - **算法层**：X → Q=XW_Q (N×320, 5 groups)、K=XW_K (N×64, 1 group)、C=XW_C (N×128, 1 latent group) → 每个 head i 从 Q 取第 q(i) 组、K 共享、C 共享 → 对 5 个 Q groups 分别计算 attention（而非 20 个 head 各自计算）：attn_g = softmax(Q_g @ K^T / 8) → 5 组 attention weights → 每组内所有 head 共用该组 attention weight 对 C 做加权：O_i_raw = attn_{q(i)} @ C → O_i = (O_i_raw @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i}) → W_{O,i} 输出投影。FLOPs_attention = n_q(d_h+d_l)N^2 = 5 × 192 × 2048^2 ≈ 4.0G FLOPs（vs GQA 的 10.7G）。KV cache = (n_k d_h + n_c d_l)N = (1×64+1×128)×2048 = 393K elements/layer（vs GQA 的 1.3M）。
  - **系统框架层**：PyTorch + HuggingFace Transformers v4.36.0。DynamicCache 存储 K(64 dims) + C(128 dims) 共 192 dims/token。OffloadedStaticCache 模式同样兼容。未修改 serving 框架，纯算法层替代 self-attention 模块。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。实际推理未开发自定义 kernel——其 FLOPs 减少和 cache 减少直接转换成 PyTorch 默认 kernel 上的速度提升。LLM-Viewer 模拟（roofline model）验证了理论效率增益能在实际硬件上体现（H100 上 prefill 和 decode 时延均低于 GQA-1B）。论文坦承缺乏工程优化（见 Conclusion："The limitation stems from our lack of engineering-focused optimization efforts, which prevents us from achieving the theoretical upper bound of efficiency gains"）。
  - **硬件架构层**：NVIDIA H100 80GB、A800 80GB、RTX 3060、Apple M2、BCM2712。2048 token prefill：GTA-1B 比 GQA-1B 快约 1.5-2×（H100 上 prefill time ~50ms vs ~100ms）。cache offload 场景（GPU↔CPU 传输）：GTA-1B 的 I/O 优势更明显（cache 仅 30%），decode 128 tokens 时延比 GQA-1B 减少 30-50%。

  **GTA 通过 "shared attention map + nonlinear latent value decoder" 的双重创新，同时压缩了注意力计算的 QK^T 次数和 KV cache 的存储量，用非线性 gate 替代 MLA 的线性解压以提升表达力，实现了第一个在不牺牲模型质量前提下同时加速 prefill 和解码的注意力机制。**

## HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

- baseline方法是什么？
  Baseline是以Loki(low-rank)、Quest(block-level)为代表的现有top-k attention方法，以及MagicPIG(LSH-based)、KVCache压缩方法(StreamingLLM/H2O/SnapKV)。其全栈执行例子如下：
  - **算法层**：Loki通过PCA投影到前R个channel的低维子空间计算近似qk scores（low-rank方法带来的维度-精度trade-off，保留足够channel需大量计算），Quest将keys分block并估计block-level qk score上界（coarse-grained estimation可能漏掉分散在blocks间的关键token，同时block内irrelevant token被不必要地加载）。两类方法都基于一个强假设——精确数值估计qk scores是复现full attention效果的前提——因此投入大量计算/内存开销来最小化绝对qk score的近似误差。MagicPIG使用LSH但需要1500-bit hash bits才能保证精度，速度受限且牺牲准确率。
  - **系统框架层**：基于PyTorch推理pipeline，在attention模块中修改QK^T计算方式(Loki)/token selection策略(Quest)。未使用专用serving框架（Quest有开源高性能实现基于mit-han-lab/Quest）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Loki使用Triton实现低秩score计算kernel（HATA论文附录C提供了优化的Triton implementation，含fused gather-FlashAttention + Static KVCache优化）；Quest提供open-source CUDA kernel实现（仅支持MHA和batch=1）。
  - **硬件架构层**：48GB HBM GPU，149.7 TFLOPS FP16。Baseline方法在长序列和large batch下受限于KV cache loading的memory bandwidth瓶颈。

  Baseline的核心缺陷：(a) 追求精确qk score估计导致高计算/内存开销——Loki的channel extraction需大量dot product计算，Quest的block-level搜索仍需加载大量KV；(b) 精确数值估计是做top-k selection的overkill——关键需求仅是相对排序而非绝对数值；(c) LSH方法需要大量hash bits（1500 bits）才保证精度，效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HATA通过将top-k attention重新定义为"轻量级序数比较任务"而非"数值回归任务"，引入learning-to-hash产生紧凑的128-bit binary hash codes进行key retrieval：

  **1. 重新定义问题 → 解决缺陷(a)的overhead问题**：
  发现top-k attention的核心需求不是精确qk score估计，而是知道qk scores的相对排序。因此将"qk score精确回归"松弛为"哪个key与query更近的序数比较"，消除高精度score近似所需的计算/内存成本。关键insight：精确score magnitude对ranking outcome无关紧要。

  **2. Learning-to-Hash → 解决缺陷(b)的overkill问题**：
  学习hash函数h(x)=2·Sigmoid(σ·xW_H)-1将连续的query/key向量映射为紧凑的128-bit二进制hash code。通过优化min Σ s_i||h(q)-h(k_i)||²（相似性保持）+ bits balance/uncorrelation约束，确保相似q/k pair被赋予Hamming距离小的hash codes。训练数据由prefill阶段的正负qk pair采样构建（top 10%正样本标签[1,20]，90%负样本标签-1），每head独立训练W_H∈R^{d×128}。HashEncode复杂度O(s×d×128)，rbit=128≪s→prefill overhead<1%。

  **3. Hardware-Efficient Optimization → 解决缺陷(c)的LSH效率问题**：
  128-bit hash code vs MagicPIG的1500-bit LSH——HATA的128-bit足够精确（通过learning-to-hash而非random projection实现）。三项GPU优化：(a) Kernel Fusion将HashEncode的MatMul-Sign-BitPack-CacheUpdate融合为单CUDA kernel，消除CPU-GPU同步开销；(b) Hamming Score Operator使用XOR+popc指令和coalesced memory access，O(s×4)而非O(s×128)复杂度；(c) Fused Gather+FlashAttention消除selected KV的冗余HBM↔SRAM传输。

  全栈执行例子（HATA on Llama-3.1-8B-Instruct, 128K context, 1.56% token budget）：
  - **算法层（核心创新）**：
    (a) Offline Hash Training：从Qasper/RepoBench-P/LSHT/LongBench-v2采样150K-300K qk pairs → per-head训练W_H (SGD lr=0.1 15 epochs×20 iters) → 学习128-bit hash function
    (b) Prefill：标准dense attention + HashEncode K → 缓存K_H (128-bit=4 INT32) + K/V
    (c) Decode：HashEncode Q→Q_H, K→K_H → bitwise_xor(Q_H, K_H_cache)+popc→Hamming distance S → TopK(S, N)→sparse FlashAttention with selected K/V
  - **系统框架层**：基于PyTorch 2.4 + FlashInfer，pluggable集成——用户仅需替换标准attention为HATA attention。支持MHA和GQA（GQA时aggregate S across shared KV head queries）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Fused Hash Encode kernel → Hamming Score kernel (XOR+popc) → TopK (GPU sort) → Fused Gather+FlashAttention kernel。三项CUDA kernel融合将Simple PyTorch实现加速6.53×。Score operator贡献最大(53.2% latency reduction)，FusedAttn次之(23.8%)，Encode Fusion更小(7.6%)但critical for end-to-end latency。
  - **硬件架构层**：48GB HBM GPU, 149.7 TFLOPS FP16。batch=8 seq=32K时7.2× speedup over Dense；batch=1 seq=256K时6.51× over Dense。

  **对比baseline的关键差异**：
  - Baseline (Loki/Quest) 精确估计qk scores → HATA 仅需序数比较(ordinal comparison)，消除了precision-vs-cost trade-off
  - Baseline (Loki) O(R×d) channel extraction → HATA O(rbit×d/32)=O(4×d) hash encoding
  - Baseline (MagicPIG) 1500-bit LSH → HATA 128-bit learned hash codes，compact + precise
  - Baseline 在超低token budget下accuracy退化 → HATA在0.4% token ratio仍维持可接受accuracy（LongBench-e Llama2 avg 34.60 vs Dense 34.47 at 1.56% budget）
  - HATA-off (with KVCache offloading): 6.04×/2.54× faster prefill/decode than MagicPIG on Llama2

  **关键ablation发现**：
  - Hash bits rbit=128是最优配置（32→128 accuracy持续提升，128+仅微小波动），平衡精度和效率
  - 前两层保留vanilla attention为标准做法（attention outlier layers）
  - Token budget reduction: HATA accuracy degradation远小于Quest和Loki

## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- baseline方法是什么？
  Baseline 是 DeepSeek Sparse Attention (DSA) 的 flat token scan indexer，用于 DeepSeek-V3.2 和 GLM-5 的 token-level 稀疏注意力。DSA 包含两个组件：(1) **Token-wise Indexer**：对每层每个 query，用轻量 indexing heads（含 gating weights）对全前缀 L 个 token 逐一打分 I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)，选出 top-k 个最高分 token 的索引 T_t；(2) **Sparse MLA**：仅在 T_t 中的 k 个 token 上执行 attention 计算。DSA 的设计使得下游 attention 是稀疏且廉价的（O(Lk)），但 indexer 本身需要扫描全前缀（per-query O(L)，per-layer O(L²)），在超长上下文（128K-1M tokens）下 indexer 从可忽略开销转变为主导瓶颈。block-sparse 方法（如 MoBA、NSA）虽然硬件友好，但 block 级别的粗粒度选择会丢失 token 级别的重要度差异——block 内所有 token 必须整体保留或丢弃，浪费预算在不重要的 token 上且可能遗漏关键 token。

  全栈执行例子（Baseline / DSA on DeepSeek-V3.2）：
  - **算法pipeline**：全前缀 token scan → indexer 逐 token 计算 relevance score I_{t,s} → TopK(k) token selection → Sparse MLA (MQA mode, 单 KV latent entry 共享于所有 query heads) → 输出。复杂度 per-layer O(L²)。
  - **系统框架**：vLLM online serving framework，FP8 精度，支持 continuous batching
  - **编译框架**：论文未明确说明
  - **kernel调度**：TileLang kernel（https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32）实现 DSA indexer。在 A100 上 64K context 时 indexer ~5.6 ms，Sparse MLA ~1.6 ms——瓶颈在 indexer
  - **硬件架构**：NVIDIA A100 GPU

- 论文方法是什么？如何对应解决Baseline的缺陷？
  HISA 将 DSA 的 flat token scan 替换为两阶段层级搜索（coarse-to-fine），通过"先粗筛、后精排"的策略，在保持 token-level 细粒度选择的同时大幅降低索引复杂度。

  **1. Block-level 粗过滤 → 解决 indexer O(L²) 复杂度瓶颈**：
  将前缀划分为大小为 B 的连续 block，每个 block 用 mean pooling 生成一个代表向量 k̃_b^I。query 仅需对 M = ⌈L/B⌉ 个 block 代表打分（而非 L 个 token），选出 top-m blocks。这步复杂度 O(L/B)，将搜索空间从 L 压缩到最多 mB（m ≪ M）。由于 block pooling 信息可增量维护在 KV cache 旁边，额外开销可忽略。

  **2. Token-level 精筛 → 解决 block-sparse 方法的粗粒度缺陷**：
  在粗过滤选出的候选 block 内，使用与原始 DSA 完全相同的 token-level scoring 机制，从 mB 个候选 token 中选出最终 k 个 token。这一步保留了 DSA 的 token-level 细粒度——block 内的 token 不再"全留或全弃"，而是逐 token 竞争。block-sparse baseline（仅 Stage 1 无 Stage 2）在 NIAH（needle 在中间位置）和 LongBench（Synthetic 任务）上显著退化，而 HISA 的 token 精筛弥补了这一差距。

  **3. 即插即用、免训练 → 解决工程落地门槛**：
  HISA 输出与 DSA indexer 完全相同的数据结构（每个 query 的 k 个 token 索引集），下游 Sparse MLA 完全不变，KV cache 布局不变，模型权重不变。可直接替换 DeepSeek-V3.2 和 GLM-5 的 indexer 模块，无需任何 fine-tuning。

  全栈执行例子（HISA on DeepSeek-V3.2, B=128, m=64, k=2048）：
  - **算法pipeline**：
    1. Block 划分: L 个 token → M = ⌈L/B⌉ 个 block，每 block 128 tokens
    2. Block pooling: MeanPool(k_s^I, s ∈ B_b) → k̃_b^I（增量维护）
    3. Stage 1 (Block Filter): query 对所有 block 代表打分 → J_{t,b} → TopK(m) blocks + 强制首尾 → 候选集 Ω_t（≤8192 tokens）
    4. Stage 2 (Token Refine): 在 Ω_t 上用 DSA 机制逐 token 打分 → I_{t,s} → TopK(k=2048) → T_t
    5. Sparse MLA: Attn(h_t, {c_s | s ∈ T_t})（与 DSA 完全相同）
    复杂度 per-layer: O(L²/B + LmB) = O(L²/128 + 8192L) vs DSA O(L²)
  - **系统框架**：vLLM + indexer 模块替换，FP8 精度，无需改 Sparse MLA 和 KV cache
  - **编译框架**：论文未明确说明
  - **kernel调度**：TileLang 实现两阶段 kernel：(a) Block filtering kernel: M × d matmul → TopK；(b) Token refine kernel: mB × d matmul（仅候选 token）→ TopK。在 fixed 8K budget 下第二阶段计算量恒定，更易优化
  - **硬件架构**：NVIDIA A100 GPU

  **对比 baseline 的关键差异**：
  - DSA indexer 扫描全前缀 O(L²) → HISA O(L²/B + LmB)，64K 时加速 2.16×-3.75×
  - Block-sparse (MoBA/NSA) 仅 block 级粗选 → HISA block 粗选 + token 精排，NIAH 和 LongBench 上接近 DSA 质量
  - HISA 的 token 精筛使 block 内低质 token 被剔除（而 block-sparse 全留），关键 token 即使所在 block 排名不高也能通过在候选池中的 token 级竞争被选中
  - 首尾 block 强制保留策略处理 attention sink 和局部上下文，避免关键信息丢失
  - 候选池 mB=8192 远大于输出 k=2048，提供 4:1 过采样率以保证精筛质量
  - block size B 与 top-m 的 trade-off: B 越大粗过滤越快但 block 代理越粗糙，m 越小效率越高但遗漏风险增

  核心创新：HISA 发现 DSA 的瓶颈不在 Sparse MLA 而在 indexer，并通过层级索引将 indexer 的搜索路径从 flat scan 改写为 coarse-to-fine——保留了 token-level 的细粒度稀疏模式，同时将搜索成本从 O(L²) 降至亚二次方。这一设计使 HISA 能够作为 DSA 的免训练 drop-in replacement，直接在 DeepSeek-V3.2 和 GLM-5 上使用。

## Hardware-Efficient Attention for Fast Decoding

- baseline方法是什么？
  Baseline 是现有的硬件高效注意力变体（GQA、MQA、MLA），它们在解码阶段面临算术强度不足和并行扩展受限的问题。全栈执行例子如下：
  - **算法层**：MQA 缓存单一 KV head 供所有 query head 共享，将 KV cache 降至最低但模型质量显著下降，且 TP 时每设备需复制该单头，内存节省无效。GQA 将 query head 分组共享 distinct KV head，通过适中的 group 数（如 gq=4）保持质量，但当 TP 度低时（如 TP=2），每设备仍需存放大量 KV cache（例如 GQA-8 TP=2 时每 token 存 2dh 元素）。MLA（DeepSeek V2/V3）通过低秩压缩将 KV 投影为单头 latent c^{KV}（维度 4d_h），解码时吸收 up-projection 矩阵，算术强度达 ~2h_q——是 MQA 的 2 倍。但 MLA 的致命缺陷是**单头 latent 在 TP 时在所有设备上复制**，TP=2 时 KV cache 不减半，TP=4 时仍是每设备 4d_h，并行扩展性受限。MLA 只能通过混合 TP+DP 来缓解——将不同 batch 序列分配给不同 DP group，但这在序列长度不均匀时产生严重的 straggler 效应（一个 DP rank 处理长序列阻塞所有其他 rank）。
  - **系统框架层**：基于 SGLang / vLLM 等 serving 框架，使用 chunked prefill (Agrawal et al., 2023) 和 PagedAttention (Kwon et al., 2023)。MLA 在 TP-only 配置下 latent 被复制，在 TP+DP 混合下 attention 子模块跨 DP group 复制。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-3 / FlashMLA kernel（Li, 2025）——使用 warp specialization、TMA、软件流水线优化 MLA 解码。但 MLA 的算术强度 ~2h_q 使 kernel 在 L_q=1 时接近 compute bound（H100 上达 610 TFLOPS），在 L_q=2（推测解码）时超出 compute roof 变为 memory-bound。
  - **硬件架构层**：NVIDIA H100 80GB SXM5 GPU（989 TFLOPS BF16, 3350 GB/s HBM）。HBM 带宽与计算能力的差距持续扩大（FLOPs ~3×/2yr vs 带宽 ~1.6×/2yr），使 memory-bound decoding 问题日益严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过"以算术强度为设计视角"重新设计注意力，提出 GTA 和 GLA 两个变体，辅以低层次 kernel 优化，解决 basline 的三个核心缺陷：

  **缺陷 1: GQA 的 KV cache 在低 TP 度时仍然过大 → GTA 通过 KV tying 解决**
  GTA 将 GQA 中独立的 Key 和 Value 投影合并为单一的 *tied KV* 状态（m_kv=1 vs GQA 的 m_kv=2），基于以下洞察：(a) key 在加 RoPE 前位于极低秩子空间（Yu et al., 2024a），(b) 仅部分 head 维度需要 RoPE 用于位置区分（Black et al., 2022; Barbero et al., 2025）。GTA 将 tied KV 的前半维度用作未旋转的 key（K_NoPE），另加单头 RoPE 投影广播到所有 group 作为 key 的旋转部分。这使得每 token KV cache 从 hkv×2×d_h 降至 hkv×1.5×d_h（含广播的 RoPE 部分 0.5×d_h），算术强度从 gq 翻倍至 2gq。关键设计选择：对 tied 部分加 RoPE 再反旋转会损害质量，因此 tied 部分永不旋转。

  **缺陷 2: MLA 的单头 latent 无法分片到 TP rank → GLA 通过多 latent head 分组解决**
  GLA 将 MLA 的单头 latent c^{KV} (d_c=4d_h) 拆分为 h_c 个 latent head（每 head d_c=2d_h），每个 latent head 负责一组 query head。这一改动的关键后果是：(a) latent head 可以在 TP rank 间分片而无需复制——TP=2 时每设备仅存 2d_h（vs MLA 的 4d_h），TP=4 时每设备存 0.5×d_h/head；(b) 不牺牲算术强度——每 group 内的算术强度仍为 ~2gq（与 MLA 的 2h_q 可比）；(c) 保留模型质量——通过 per-group 的 up-projection 矩阵为每组 query head 学习专属的 K/V 特征。训练时每组有独立的 W^{UK}_i 和 W^{UV}_i；解码时这些矩阵被吸收进 Q/O 投影。

  **缺陷 3: 解码 kernel 未充分利用现代 GPU → 异步流水线 + 分布式偏移量计算 + Cooperative Softmax**
  论文将算法创新与低层次系统优化配对：(a) Warp Specialization 将 producer（内存加载）和 consumer（Tensor Core MMA）分配到不同 warp，利用 GPU warp scheduler 的异步性重叠执行；(b) Distributed Offset Calculation 解决 Paged KV 下 cp.async 指令的地址计算瓶颈——通过 warp 内多线程协作分摊 64-bit 整数地址计算，使 page size 1 的 kernel 速度匹配 page size 64（解锁 RadixAttention prefix caching）；(c) Cooperative Softmax 支持多 warp 并行下的正确 online softmax。

  全栈执行例子（GLA-2, TP=2, H100 serving）：
  - **算法层**：X → Q_{0,1} 投影 → c_{0,1}^{KV} latent heads（各 2d_h，分片到 rank 0 和 1）→ 吸收后 attention: O_i = softmax(Q_i @ c_i^{KV}^T) @ c_i^{KV} → partial output @ W^{VO}_i → AllReduce → 最终 O。每 token KV cache per device: 2d_h（vs MLA 的 4d_h）。
  - **系统框架层**：基于 SGLang live server 模式（含 HTTP 解析、动态队列、GPU kernel 调用的全程计时），使用 FlashAttention-3 kernel + chunked prefill (tile=8192)。GLA-8 (h_c=8, TP=8) 在 64 并发 8K/4K prefill/decode 下达到 1461 tok/s（vs MLA TP=8 的 859 tok/s，+70%）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：CUDA/PTX kernel：warp specialization（producer TMA/cp.async + consumer mma），distributed offset calculation（128 threads 分 8 组协作计算 paged KV 地址），cooperative softmax（sTMP cross-warp reduction）。GLA kernel L_q=2 时 2× faster than FlashMLA。
  - **硬件架构层**：NVIDIA H100 80GB SXM5。GLA kernel 达 93% peak memory bandwidth、70% peak TFLOPS。因 GLA 每 device KV cache 小，在 131K prefill + 不平衡负载下纯 TP=8 的 GLA-8 完成 2.7× MLA 混合 TP+DP 的吞吐。

  对比 baseline 的关键差异：
  - GTA vs GQA：每 token KV cache 减半、算术强度翻倍，同规模模型上 PPL 更低（XL: 10.129 vs 10.202）
  - GLA vs MLA：每 device KV cache 减半（TP≥2）、算术强度相当（~2gq vs ~2h_q），可纯 TP 部署（无需 DP）、对不平衡负载鲁棒（无 DP straggler），质量匹配或略优（XL downstream: 60.0% vs 59.1%）
  - GLA vs FlashMLA kernel：标准解码快 20%、推测解码快 2×，page size 1 无减速（解锁 prefix caching）
  - 设计哲学：算术强度最大化 = 多为每加载的字节做计算 = 更好利用现代 GPU 的算力过剩——通过减少 m_kv（tying KV→1）、增加 gq（分组）、引入 h_c（多 latent head 可分片）三个独立维度实现

## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- baseline方法是什么？
  Baseline 是三种现有方法的组合缺陷：(A) **Full Attention (FlashAttention2)** + (B) **HiP Attention** 的层次化剪枝 + (C) **SelfExtend/NTK** 的 RoPE 外推。

  (A) FA2 的缺陷：通过 tiling+recompute 优化 memory access 但不减少 FLOPs——1M context 解码时每次 attention 需 4,645 µs，KV cache 在 BF16 下 1M tokens 需约 64GB 远超单卡容量，不支持 OOL generalization（仅在训练长度内有效）。
  
  (B) HiP Attention 缺陷：(a) 层次化剪枝算法的迭代式 top-k 涉及大量 global thread synchronizations，阻碍 GPU 并行——SelectRep 的每次迭代需全局同步，无法利用 key sequence dimension parallelism（类似 FlashDecode 的 split-KV）；(b) 启发式剪枝精度不足——top-k 估计的 recall 较 InfiniteHiP 低 4.72%；(c) 无 per-stage mask 缓存——解码时每次都要运行完整的 costly initial pruning stage（O(T_kv)），导致解码延迟高；(d) KV cache offloading 的驱逐策略简单，未使用 LRU。
  
  (C) RoPE 外推缺陷：SelfExtend 使用统一的 group-size 缩放 RoPE，不区分不同层/head 的 attention pattern 差异——早期层倾向于 dynamic sliding window（关注相对位置），后期层依赖语义信息，统一处理导致 OOL 性能不佳。

  全栈执行例子（baseline: HiP Attention + SelfExtend, Llama 3.1 8B, 128K context, RTX 4090）：
  - **算法层**：输入 128K tokens → HiP 层次化剪枝（heap-based top-k selection across all heads, 需 global sync）→ 从 128K 缩减到 ~1K token → Block Sparse Attention on ~1K token → 输出 token。SelfExtend 对全部层使用相同 group_size 缩放 RoPE。
  - **系统框架层**：基于 PyTorch 自定义 attention 层，HiP 的 offloading 使用 UVM，无 SGLang 集成。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：HiP kernel：迭代式 SelectRep + 全局 top-k → 每个 stage 内 global thread synchronization → 解码 450 µs/token @ 1M context。FA2 kernel：全量 dense attention，无剪枝优化。
  - **硬件架构层**：NVIDIA RTX 4090 24GB。HiP+UVM offloading 在 256K+ 上下文时因 offload 开销大，解码延迟高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  InfiniteHiP 通过三个核心设计解决 baseline 缺陷：

  **1. 高度并行的模块化层次化剪枝（Modular Hierarchical Pruning）→ 解决缺陷 (B-a, B-b)**
  将 HiP 的迭代式 top-k 替换为 per-chunk top-1（SelectRep）+ max chunk score + top-K chunk selection 的单 kernel 流程。关键创新：SelectRep 算法每次迭代仅访问 2 个 token（左右分支首 token）→ 无全局同步即可实现 → 可利用 key sequence dimension parallelism（类似 FlashDecode split-KV）→ 在 A100/4090 等现代 GPU 上实际运行速度更快。同时由于使用 per-query-block 的动态代表 token 选择（而非 InfLLM 的预选固定代表 token），top-k 估计 recall 比 HiP 高 4.72%，比 InfLLM 高 1.57%。

  **2. 分层动态 RoPE 调整（Layer-wise Dynamic RoPE）→ 解决缺陷 (C)**
  观察到 LLM 早期层（layer ≤ 5）呈现 dynamic sliding window-like attention pattern（关注相对位置），后期层依赖语义信息。因此：前 3 层使用 Chunk-indexed RoPE（每 chunk 赋相同 position ID，引导滑窗式 mask）、后续层使用 Relative-style RoPE（分左右分支赋不同 position offset，依赖内容信息）。Block Sparse Attention 阶段使用 StreamingLLM-style RoPE（保持 causality 和相对顺序）。这种异构 RoPE 策略使 En.MC OOL 评测从纯 Relative 的 68.55% 提升到混合 Chunk-indexed+Relative 的 74.23%（+5.68%）。

  **3. Per-stage Mask 缓存 + LRU KV Offloading → 解决缺陷 (B-c, B-d, A)**
  (a) 每个剪枝 stage 维护独立的稀疏注意力 mask 缓存，利用 temporal locality（query 的 mask 在连续 decoding step 中变化缓慢），以 configurable refresh interval 更新。效果：256K context 解码从 no cache 的 9,803 µs 降至 all cache 的 110 µs（89× speedup）。
  (b) 将 KV cache offloading 驱逐策略从 HiP 的简单策略改为 LRU，更精确地识别 cold token。同时实现为 graph-capturable CUDA 操作避免 CPU launch overhead。
  (c) 所有组件集成到 SGLang serving 框架中，使用 Triton 实现跨硬件可移植。

  全栈执行例子（InfiniteHiP, Llama 3.1 8B, 3M context, L40S 48GB, 3K-Flash preset）：
  - **算法层**：输入 3M tokens → 保留 n_sink=256 + n_stream=1024 → Stage 0: 全量 key 分 chunk(l_c=256) → SelectRep(top-1 per chunk，每次 2 token 点积) → max chunk score → top 32K key → Stage 1: 32K key 分 chunk(l_c=32) → SelectRep → top 8K key → Stage 2: 8K key 分 chunk(l_c=8) → SelectRep → top 4K key（前 3 层）/ 2K key（后 29 层）→ BSA on ~3K selected keys → Dynamic RoPE（前 3 层 Chunk-indexed / 后 29 层 Relative-style / BSA 阶段 StreamingLLM-style）→ 输出 token。复杂度从 O(T²) 降至 O(T*q)：3M 解码时仅对约 3K token 执行完整 attention。
  - **系统框架层**：基于 SGLang（https://github.com/DeepAuto-AI/sglang/），在 SGLang 的 attention 计算路径中替换为 InfiniteHiP pipeline。支持 mask cache 按 stage 独立管理、refresh interval configurable（flash preset: 96/24/8）、单 batch 长上下文 serving。End-to-end 解码吞吐：L40S 上 3M context 达 23.8 tok/s（Flash offload），比 SRT 估计值快 7.25×。
  - **编译框架层**：论文未明确说明。Triton kernel 通过 Triton compiler 编译为 GPU 代码。
  - **kernel调度层**：(a) Pruning Stage Triton Kernel：单一 kernel 实现完整 stage，parameterized by (l_c, b_q, k)，key sequence dim parallel（类似 FlashDecode split-KV），无全局同步。(b) BSA Triton Kernel：FlashAttention-style（prefill）+ FlashDecoding-style（decoding）+ PagedAttention（block KV memory）。(c) UVM Offloading：CUDA UVM dynamic page migration + LRU eviction on GPU key bank。Latency breakdown（1M context, 3K, 无 mask cache）：Stage 0 28.2% + Stage 1 4.0% + Stage 2 5.3% + BSA 2.2% + Extra(offload overhead) 60.3%。
  - **硬件架构层**：NVIDIA RTX 4090 24GB（消费级）和 L40S 48GB（云端性价比），单 GPU。PCIe 4.0 x8 连接 CPU RAM（访问延迟 31.5× VRAM）。3M context KV cache 约需 192GB → 远超显存 → 通过 UVM offloading 使用 CPU RAM。

  对比 baseline 的关键差异：
  - HiP 的迭代 top-k → InfiniteHiP 的 per-chunk top-1 + max score + top-K chunk（无全局同步，key dim 并行）
  - HiP 无 mask 缓存 → per-stage mask caching（89× 解码加速）
  - HiP 简单 offloading 驱逐 → LRU 驱逐 + graph-capturable CUDA operation
  - SelfExtend 统一 RoPE → 分层异构 RoPE（Chunk-indexed + Relative + StreamingLLM，En.MC +5.68%）
  - InfLLM 预选固定代表 token → 动态 per-query-block 代表 token 选择（recall +1.57%）
  - InfLLM 不在 attention kernel 内访问 CPU memory（牺牲精度）→ InfiniteHiP 在 kernel 内访问 CPU memory（保持精度，但 offload 开销 60.3%）
  - FA2 不支持 OOL generalization + KV 随 T 线性增长无法 fit GPU → InfiniteHiP 训练无关 OOL + UVM offloading 支持 3M tokens on 48GB GPU

## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- baseline方法是什么？
  Baseline 是现有 KV cache eviction 方法（H2O、SnapKV、PyramidKV、Ada-SnapKV、Ada-PyramidKV），其全栈执行例子如下：
  - **算法层**：H2O 基于全部过去 queries 的累积 attention score（A2S）识别 "heavy-hitter" KVs 并 evict 低分 KVs，uniform eviction rate 跨所有 heads；SnapKV 使用有限 observation window（w=8）内的 queries 聚合 attention + max-pooling（p=7），保留 top-N high-attention KVs，同样 uniform eviction across heads；PyramidKV 使用 SnapKV 的 observation window 但按金字塔形分配各层 eviction rate（浅层少 evict、深层多 evict）；Ada-SnapKV/Ada-PyramidKV 允许每 head 可变 eviction rate（cross-head eviction），但该方案在现有推理框架中仅增加 cache 碎片化而无法实际减少物理内存占用。所有 baseline 的共同缺陷：(a) 针对 MHA 模型设计，在 GQA 模型实现中先将 KV cache repeat 到 query head 数量再进行压缩，导致 cache 中 3/4 (Mistral/Llama-3 gqa ratio r=4) 的 KVs 为重复数据，压缩效率低下——需压缩率超过 r 才能改善已有 GQA 提供的压缩效果；(b) Uniform eviction rate across heads 忽略不同 attention head 对 KV cache 的异质性需求，限制了理论压缩率；(c) Ada-SnapKV 虽提出 variable-head-rate 但缺乏使碎片化被 PagedAttention 管理的实现。
  - **系统框架层**：HuggingFace Transformers 或 PyramidKV 实现（https://github.com/IsaacRe/PyramidKV），在标准 attention 计算后执行 KV selection/index-gather。非 paged attention，KV cache 为 contiguous tensor，uniform eviction 通过缩小 L 维度实现内存节省。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention。Eviction 操作（sort, top-k, gather）在 GPU 上通过 PyTorch 算子执行。
  - **硬件架构层**：NVIDIA L4 / H100 GPU（baseline 评测）；NVIDIA A100（Ada-SnapKV/PyramidKV 原论文评测）。长上下文下 KV cache 内存为 throughput bottleneck。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KV-Compress 通过三个核心设计解决 baseline 缺陷：

  **1. Query-Group Compression + Non-repeated GQA Cache → 解决缺陷(a)**：
  将 KV eviction metric 的聚合范围从 "所有已 repeat 的 query heads" 改为 "key 所属 query group 内的 queries"：M_{h_k,j} = Σ_{h∈H_k} Σ_i (A_{h,i,j})^2，其中 H_k = {h: r·h_k ≤ h < r·(h_k+1)}。直接在非 repeat 的 GQA KV cache（shape H_kv × L × d）上执行压缩，而非先 repeat 到 H_q × L × d。对于 Mistral/Llama-3 (r=4)，同样 max-cache-size C 下 KV-Compress 持有 1/4 的 KVs——即实现 4x 额外有效压缩率，LongBench 上以 1/4 KVs 达到 state-of-the-art。

  **2. Paged Block Eviction + Variable-Head-Rate 实现 → 解决缺陷(b)(c)**：
  PagedAttention 扩展：将 block 从 "每个 block 存储所有 layer×all heads 的 KVs" 改为 "每个 block 仅存储单 head 的 KVs"，block table 扩展为 B×l×H×L_max/b。这使得不同 head 可以有不同数量的 allocated blocks——variable-head-rate eviction 可以实际释放 evicted blocks 的物理内存。MoveCache 算法重排物理 cache 使得被 evicted 的 blocks 在物理上连续可释放。Block eviction 选择：跨 head 排序候选 block eviction（按每 block 最大 eviction metric），选择总 metric 最低的 E_s blocks 进行 eviction。Variable per-layer compression rate 通过相同机制实现。

  **3. Squared Attention (L2) Metric → 改善 eviction 质量**：
  使用 Σ(A_hij)² 替代 ΣA_hij 作为 eviction metric——等价于最小化未来 attention 的 L2 error 而非 L1 error。在 LongBench 所有 max-cache-size 和所有变体（KVC-w, KVC-full）中 L2 一致优于 L1。

  全栈执行例子（KV-Compress on Llama-3.1-8B-Instruct, vLLM modified, compression rate 32x, L4 GPU）：
  - **算法层（核心创新）**：
    (a) Prefill 阶段：正常计算 attention → 对 observation window w=8 的 queries 计算 squared attention 聚合（GQA group-wise）→ max-pooling p=7 得到 M_{h_k,j} → 跨 (head, seq_len) 排序 → 按 head 分组 reshape 为 [N, b] → 每 head 各 block 的最大 eviction metric m(h,e) → 跨 head 排序候选 eviction blocks → 选择 E_s blocks → MoveCache 重排物理 cache → 释放 E_s 个 blocks。
    (b) Decoding 阶段：每 step 累积新 token 的 squared attention → 按需（preemption 即将发生时）触发基于 updated metric 的再次压缩。
    (c) 调度：prefill 后 + preemption 前压缩，实现 dynamic cache size management。
  - **系统框架层**：vLLM v0.6.0 修改版（开源 https://github.com/IsaacRe/vllm-kvcompress），GPU 端 block manager 替代 CPU scheduler（消除 l×H 倍 block 增长的 CPU 调度开销），block table 扩展支持 per-head per-layer 索引，unified physical KV cache K_u, V_u ∈ R^{N×16×d}。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：PyTorch sort API 用于 metric 排序（主要 overhead，额外内存 ~8× sorted tensor 大小）。PagedAttention kernel 通过修改后的 block table 索引读取 per-head KVs。Eager mode（no CUDA graph）。
  - **硬件架构层**：NVIDIA L4 24GB 和 H100 80GB。GPU block manager 利用 SIMT parallelism 实现 block 分配和释放的并行化。

  **对比 baseline 的关键差异**：
  - Baseline 所有 head uniform eviction → KV-Compress variable per-head per-layer eviction，实际释放内存（非仅碎片）
  - Baseline GQA 先 repeat KV 再压缩 → KV-Compress query-group compression 在非 repeat cache 上压缩（4x 额外效率）
  - Baseline L1 attention aggregation → KV-Compress L2 squared attention aggregation（LongBench 平均分 KVC-w8-L2 vs KVC-w8-L1: +1.3 pp for Llama-8B C=128）
  - KV-Compress 8B C=128: avg score 46.26（state-of-the-art）vs SnapKV 45.93, PyramidKV 45.97（且仅使用 1/4 KVs）
  - Llama-3.1-70B-FP8 64x compression: 多数 non-summarization 任务保持 >90% full-cache 性能
  - Throughput on L4, L_c=6000: 4.93x (32x) / 5.18x (64x) over vanilla vLLM
  - Summarization 任务（GovReport, QMSum）对压缩最敏感，低压缩率即显著退化

## KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

- baseline方法是什么？
  Baseline 分为两类：(a) 训练无关方法：H2O (H2A/H2I) 和 SnapKV。H2O 基于累积注意力分数选出 "heavy-hitter" tokens 作为 KV cache 中保留的 top-k 键值对。H2A（问题感知）将问题和上下文拼接后计算累积注意力，能利用问题扫描上下文中的关键信息；H2I（问题无关）仅在上下文内部计算累积注意力。SnapKV 使用最近 token 窗口的注意力模式选择重要 token。(b) 可训练方法：ICAE 使用 auto-encoding + language modeling 目标预训练上下文压缩器，将长上下文编码为少量 memory slots，再用 frozen LLM 解码。DODO 将 KV cache 子选择为 "nugget" tokens，训练时使用 auto-encoding 或 LM 目标，但压缩率固定。

  全栈执行例子（H2O on LLAMA-3 8B，上下文 N=6000 tokens，20% retention，问题无关范式）：
  - **算法层**：将上下文 tokens 通过 LLAMA-3 forward pass 得到所有层的 attention weights → 对每层每头累加 attention scores → 选 top-k=1200 tokens 保留在 KV cache → 被 evict 的 token KV 从 cache 删除。解码时：每个新 token 仅对保留的 1200 tokens attend（而非 6000）。
  - **系统框架层**：标准 HuggingFace Transformers 的 KV cache 机制，修改 cache 的 `past_key_values` 元组在 prefill 后裁剪。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Flash Attention kernel（PyTorch SDPA 或 flash-attn），无特殊 kernel 修改。
  - **硬件架构层**：NVIDIA A100 80GB / H100。

  Baseline 的关键缺陷：
  (a) H2I 在问题无关范式下性能急剧下降——因为缺少问题信号引导，上下文内部的 heavy-hitter 分布与实际需要回答的问题无关。例如 LLAMA-3 SQuAD: H2I 25% retention 准确率仅 56.6%（vs uncompressed 87.6%）。
  (b) ICAE/DODO 使用 auto-encoding 预训练目标，与下游推理时 next-token prediction 存在分布不一致（pretraining-inference mismatch），导致高压缩率下性能损失大。
  (c) 训练无关方法无法利用领域先验知识进一步提升压缩性能（H2O 没有 fine-tuning 机制）。
  (d) ICAE/DODO 的压缩率固定，不支持灵活的任意压缩率推理。
  (e) 训练无关方法的 token 选择仅在 prefill 后执行一次，被丢弃的 token 中的信息永久丢失，未被选中的 token 无法向被选中的 token "传递"信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KV-Distill 提出三组件协同的 KV cache 压缩框架：

  **1. Learnable Token Importance Scorer + Conditional LoRA Routing → 解决缺陷(a)(e)**：
  训练一个 FFN scorer 从第 η 层的 hidden states 预测每个 token 的重要性分数 s = FFN_θ(X'_η)，取 top-k 作为保留 token。更重要的是，通过 LoRA-adapted LM_θ 进行条件计算路由：被选中的 token 使用可训练的 LoRA W^Q/W^O 矩阵，使其能 "吸收" 来自未选中 token 的信息（通过 cross-attention 机制），实现被选中 token 表示的语义增强。这与训练无关方法的纯 token 选择有本质区别——KV-Distill 不仅选择 token，还增强了被选 token 的表示质量。

  **2. Forward + Reverse KL Divergence Distillation → 解决缺陷(b)**：
  使用加权 KL 散度 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p) 直接匹配压缩前后的 next-token 分布（p=完整 cache 的分布，q_θ=压缩 cache 的分布）。forward KL (λ=0.6 主导) 为 mean-seeking 行为，确保压缩模型覆盖完整模型的所有可能输出；reverse KL 为 mode-seeking 行为，避免分布模式坍缩。这与 auto-encoding loss 有本质区别：KL 散度直接在 token 预测分布层面优化，与下游推理时的 next-token prediction 任务一致，消除了 pretraining-inference mismatch。

  **3. Multi-Ratio Training → 解决缺陷(c)(d)**：
  训练时随机采样 KV retention ratio ∈ [0.1%, 80%]，使单一 KV-Distill 模型支持任意压缩率推理。此外支持在领域数据上通过相同 KL 损失 fine-tune，进一步提升特定领域下的压缩率上限（GovReport 上 1% retention fine-tuned ROUGE-L=22.8 vs uncompressed 23.7）。

  全栈执行例子（KV-Distill on LLAMA-3 8B, N=6000 tokens, 20% retention=1200 tokens）：
  - **算法层**：
    (a) Pre-compression: context tokens → LM_θ 第 6 层 hidden states X'_6 → FFN scorer 输出 s ∈ R^6000 → top-1200 索引。
    (b) Encoding with routing: context tokens 通过 LoRA-adapted LM_θ (rank=128, Q/K/V/O 应用 LoRA) 编码。每层中：被选 token i 的 query 通过 LoRA W^Q 计算 → attention weights α_i = (z_i @ W^Q_lora)(K)^T → 可选地通过 α'_i = σ(s_i) ⊙ α_i 衰减 attention → output 通过 LoRA W^O 变换。未选 token j 使用冻结原始 W^Q/W^O 计算（其 KV 会参与被选 token 的 attention 计算，传递信息，但自身 KV 最终被丢弃）。
    (c) KL distillation: compressed KV ˜X (仅 1200 个 token) → LM 解码 → p_compressed；full KV X → LM 解码 → p_full；L = 0.6·D_KL(p_full||p_compressed) + 0.4·D_KL(p_compressed||p_full)，反向传播更新 LoRA + FFN scorer 的 150M 参数。
    (d) 解码时：使用原始 frozen LM，仅对 compressed KV cache ˜X 做 attention，无额外计算开销（LoRA adapter 已融合到 encoding 阶段）。
    (e) 长上下文折叠：N>1536 的上下文 pad 到 1536 的倍数，reshape 为 batch 后分别压缩再 unfold，保持 1536 的训练上下文窗口。
  - **系统框架层**：HuggingFace Transformers + DeepSpeed Stage 2 (8×A100 80GB) + LoRA adapter 管理。压缩后的 KV cache 格式与原始 KV cache 完全相同（仅 sequence length 维度缩小），因此任何支持 KV cache 的推理框架可直接使用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Flash Attention (SDPA)，无特殊 kernel 修改。训练时梯度传播通过 α' = σ(s) ⊙ α 实现 scorer 的可微分（绕过 non-differentiable topk）。推理时 zero overhead——compressed KV cache 直接配合标准 attention kernel。
  - **硬件架构层**：训练：8 × NVIDIA A100 80GB。推理：任意支持 HuggingFace Transformers 的 GPU 平台。

  **对比 baseline 的关键差异**：
  - H2I 纯 attention-based token selection → KV-Distill 可训练 scorer + conditional LoRA routing 增强选中 token 表示（信息从丢弃 token 传递到保留 token）
  - ICAE/DODO 的 auto-encoding loss → KV-Distill 的 next-token KL divergence loss（消除 pretraining-inference mismatch）
  - H2O/SnapKV 无领域适应能力 → KV-Distill 支持领域 fine-tune 实现 100x 压缩
  - ICAE/DODO 固定压缩率 → KV-Distill 单模型支持 0.1%-100% 任意压缩率
  - LLAMA-3 8B SQuAD 20% retention: KVD 86.0% vs H2I 51.7%, H2A 83.0%, DODO 73.3%
  - GovReport 1% retention fine-tuned: ROUGE-L 22.8 vs H2I 18.3
  - 1000x compression 下仍能产生有意义输出（定性分析）

## KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

- baseline方法是什么？
  Baseline 是全 KV cache 推理（Full KV Cache）和已有的 intra-layer KV cache 压缩方法（H2O、StreamingLLM、PyramidInfer 等）。全栈执行例子如下：
  - **算法层**：Full KV Cache 为每个 Transformer 层的每个 token 保存完整的 Key 和 Value 向量（2 × n_layers × seq_len × d_head × num_kv_heads），KV cache 在推理时占总内存 >80%。已有的 intra-layer 压缩方法（H2O 基于累积 attention scores 保留 heavy-hitter token、StreamingLLM 仅保留 attention sink + 滑动窗口、PyramidInfer 仅对重要 token 计算 KV）均在单层内通过丢弃 token 实现 KV cache 稀疏化。核心缺陷：(a) 所有压缩都在层内（intra-layer），忽略了层间（layer-wise）压缩的潜力——即是否可以不计算某些层的 KV cache 而直接复用其他层的 KV cache；(b) 已有的极少层间 KV cache 压缩工作（MiniCache、LCKV、CLA、YOCO）都需要额外训练模型，无法作为 plug-and-play 方法直接应用于已训练好的 LLM。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline，KV cache 按 autoregressive 方式逐层存储和更新。Intra-layer 压缩方法在 prefill 后/decoding 中执行 KV cache eviction。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 FlashAttention 或 PyTorch 原生 attention 实现。KV cache eviction 在 GPU 上执行 TopK + index gather 操作。
  - **硬件架构层**：NVIDIA A100 80GB GPU。Llama2-13B-Chat 在 512+2048 序列长度下 Full KV cache 内存占用 51639 MB。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KVSharer 通过一个反直觉的发现——共享不相似的 KV cache 比共享相似的 KV cache 能更好地保持模型性能——提出全新的层间 KV cache 共享策略，全栈执行例子如下：
  - **算法层（核心创新）**：
    (a) **反直觉发现**：传统参数共享/注意力共享方法都基于替换相似的值（相似度越高越好），但 KVSharer 首次发现在 KV cache 的上下文中，共享不相似的层间 KV cache 反而能有效保持模型性能。消融实验（Figure 6）证明按 dissimilarity（欧氏距离降序）共享的 PPL 远低于按 similarity（升序）共享，差距达近 2 倍以上。
    (b) **Strategy Searching 启发式搜索**：在校准数据集（30 句 Wikipedia，每句 64 tokens）上计算任意两层之间 KV cache 的欧氏距离（分别 flatten keys 和 values 为 1D 向量后取平均），按距离降序排列。依次尝试用靠近输入端的层的 KV cache 替换靠近输出端的层的 KV cache，若替换后模型最后一层 hidden state 与原始模型的余弦相似度超过阈值 T=0.5 则保留该替换。不可逆方向设计（仅输出端被输入端替换）是因为靠近输入的层更敏感。
    (c) **非 task-specific**：一次搜索的策略可通用于所有下游任务。
    (d) **正交兼容性**：作为层间压缩方法，与现有 intra-layer 压缩方法（H2O、PyramidInfer）正交兼容，可叠加使用进一步降低内存。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，即插即用——在 forward pass 中根据共享策略 Z 将被替换层的 KV cache 从前层直接拷贝，不修改模型权重。与 FlashAttention 兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：无自定义 kernel。KV cache 拷贝操作为 GPU 上简单的张量赋值（memcpy），无额外计算开销。prefill 阶段无加速，generation 阶段因跳过部分层的 KV cache 计算实现加速。
  - **硬件架构层**：4× NVIDIA A100 80GB GPU。Llama2-13B-Chat 在 25% 压缩率 + 512+2048 序列下内存从 51639 MB 降至 37049 MB（72%），generation 速度从 18.2 tokens/s 提升至 30.0 tokens/s（1.65× 加速）。叠加 PyramidInfer 后内存降至 30141 MB（58%），generation 速度达 34.1 tokens/s（1.87× 加速）。

  **对比 baseline 的关键差异**：
  - Baseline (intra-layer compression) 仅在单层内丢弃 token → KVSharer 在层间共享 KV cache，完全不计算被替换层的 KV cache，从源头减少 KV cache 计算量
  - Baseline (MiniCache/LCKV/CLA) 需要额外训练 → KVSharer 是 plug-and-play 方法，一次搜索（~60 秒，4×A100）即可应用，无需任何训练
  - Baseline 的相似度共享直觉（替换相似的）→ KVSharer 反直觉地共享不相似的 KV cache，并通过消融实验（Figure 6）证明了 dissimilarity-based 策略显著优于 similarity-based 策略
  - Baseline 层间方法不可与 intra-layer 方法兼容 → KVSharer 与 H2O/PyramidInfer 叠加使用，同时享受层间 + 层内压缩的双重收益
  - 随机共享的 PPL 在 Llama2-13B 上高达 51.41（vs KVSharer 的 9.11），benchmark 下降约 30%，证明并非任意共享都有效——KVSharer 的搜索策略是关键
  - 25% 压缩率下模型性能保持 >90%（Llama2-7B 97.9%、InternLM2-7B 97.0%、InternLM2-20B 97.4%），12.5% 压缩率下 Llama2-7B 甚至提升至 113.6%

## LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference

- baseline方法是什么？
  Baseline 是现有的 text-only KV cache eviction 方法（H2O、SnapKV、RoCo），它们在多模态长上下文场景下直接被应用。

  H2O / SnapKV / RoCo 的全栈执行例子：
  - **算法层**：H2O 基于累积 attention scores（A2S）识别 "heavy-hitter" token 保留在 KV cache 中；SnapKV 使用末尾 observation window 的 attention scores 通过 max-mean pooling 评估 token 重要性；RoCo 基于 mean attention scores 进行 eviction。这些方法将所有 token（文本 + 图像）无差别地按 attention score 统一排序和淘汰，核心假设是"attention score 高的 token 更重要"，但这在多模态场景下失效——文本 token 通常比图像 token 获得更高的 attention scores（Figure 2 观察），导致图像 token 被系统性过度剪枝，破坏跨模态交互完整性。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline，处理 LLaVA 等多模态 VLM。输入先经 Visual Encoder (CLIP ViT-L/14) → MLP Adapter 将图像转为 576 个 visual tokens，再与文本 token 拼接送入 LLM (Vicuna)。KV cache 管理与纯文本 LLM 一致——prefill 后执行 eviction，decode 时使用压缩 cache。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速 attention 计算。token selection（TopK + index gather）在 GPU 上执行。
  - **硬件架构层**：NVIDIA A100 80GB / RTX 3090 24GB。
  - **芯片设计层**：论文未明确说明。

  Baseline 核心缺陷：
  1. **多模态注意力分布偏差被忽略**：在多模态 prompt encoding 阶段，模型对文本部分表现出更强的注意力偏好（Figure 2 可视化），文本 token 天然获得更高的累积 attention score。H2O/SnapKV/RoCo 无差别地按 attention score 排序淘汰 token，导致大量图像 token 虽然对跨模态理解至关重要却被淘汰，同时文本 token 中即使有冗余也被优先保留。
  2. **跨模态交互信息丢失**：传统 text-only 方法不考虑文本 token 和图像 token 之间的 cross-modal attention 交互（如 "a dog" 文本 token 和狗图像 token 区域的 attention 关联），仅看单向的 self-attention 分数。
  3. **被 evicted 的 token 信息永久丢失**：H2O/SnapKV 等方法简单丢弃 evicted tokens，不尝试保留其中蕴含的上下文信息。在多模态长上下文中，被丢弃的图像 token 可能包含关键视觉细节，导致 hallucination 和 contextual inconsistencies。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LOOK-M 通过两项核心设计解决 baseline 缺陷：

  **1. Text-Prior KV Pair Eviction → 解决缺陷1（多模态注意力分布偏差）**：
  在累积 attention scores 的基础上，显式地为所有文本 token 加入 text-prior T_p = Max(A_s)（即全局最大 attention score）。这使得文本 token 的最终分数 = 原始累积 attention + text-prior，从而在排序保留时天然占据优势位置（事实上被"锁定"为保守 token），图像 token 中仅 attention score 最高的 Top-N 个才会被保留。这与 Figure 2 的观察一致——模型在 prompt encoding 阶段优先关注文本特征来理解全局视觉内容，因此文本 KV pair 应优先保留，图像 KV pair 中的冗余部分可被淘汰。

  **2. KV Pairs Merging Strategies → 解决缺陷3（evicted token 信息丢失）**：
  对被 evicted 的 KV pair（K_e），通过 many-to-one nearest-neighbor matching 找到与 conserved token（K_c）最相似的对应关系，然后通过三种可选的合并策略将 evicted token 的信息融入 conserved token：
  - Averaged Merging：直接对 evicted + conserved 求均值
  - Pivotal Merging：先 evicted↔closest 二元融合产生 pivotal token，再 pivotal↔conserved 均值合并，强调 conserved token 的权重比例
  - Weighted Merging：基于 similarity matrix 动态分配权重

  LOOK-M 的全栈执行例子（LLaVA-v1.5-7B, TP+P-Merge, α¹=α²=0.1, RTX 3090）：
  - **算法层**：
    1. Visual Encoder (CLIP ViT-L/14@336px) → MLP Adapter → 576 visual tokens/图 + text tokens = L_prompt tokens 输入 LLM
    2. Prefill 阶段：逐层计算 QKV 投影 → attention A_p
    3. 累积 attention score A_s = Σ_i A_p[i,:]，对 text token 施加 text-prior
    4. 保留：最近 M = 0.1×L_prompt 个 token + 前 L_prompt-M 中 top-N（N = 0.1×L_prompt）高 attention 的非文本 token
    5. 对 evicted token K_e 执行 nearest-neighbor matching → Pivotal Merging 融入 conserved token
    6. Decode 阶段：使用 (N+M) 个压缩后的 KV pair 进行 attention，新 token 的 KV 正常追加
  - **系统框架层**：即插即用集成到 LLaVA 推理流程，仅修改 attention 层的 KV cache 管理和 eviction 逻辑，无需任何 fine-tuning。与 HuggingFace Transformers 推理 pipeline 完全兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速 attention 计算。KV pair 合并（cosine similarity 矩阵 + nearest-neighbor matching + 加权平均）在 GPU 上执行。
  - **硬件架构层**：NVIDIA A100 80GB / RTX 3090 24GB。LOOK-M (20% budget) 在 RTX 3090 上 decoding latency 从 28.16 ms/token 降至 20.98 ms/token（1.34× 加速），GPU memory 从 1.52 GiB 降至 0.32 GiB（~80% 减少）。

  **对比 baseline 的关键差异**：
  - Baseline 无差别按 attention score 淘汰 → LOOK-M Text-Prior 显式优先保留文本 token
  - Baseline 直接丢弃 evicted token → LOOK-M 通过 Pivotal Merging 将 evicted token 信息融入 conserved token
  - Baseline 在 Needle in a Haystack 任务上几乎完全失败（Full Cache NH score 4.7, H2O 仅 1.4, SnapKV 1.4） → LOOK-M TP+P-Merge NH score 5.3（超越 Full Cache），因为 merging 策略补偿了 eviction 后的局部上下文信息丢失
  - Baseline 跨不同 MLLM 架构性能不稳定 → LOOK-M 在 LLaVA-v1.5-7B/13B、MobileVLM-V2-3B、InternVL-v1.5-7B 上一致优于 baselines
  - 在 99% 极端压缩率下，H2O/SnapKV/RoCo 全面崩溃（多个子任务降至 0.0），LOOK-M 仍接近 Full Cache 性能

## LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

- baseline方法是什么？
  Baseline 包括两类：(1) Recency-based 方法（StreamingLLM）——所有层统一保留固定大小的最近 token 滑动窗口（含 attention sink），O(1) 内存复杂度支持无限连续生成，但在长上下文任务上精度严重下降（Llama2-7B-Chat 512 budget 下 1K decoding length PPL 退化 35% vs full cache）；(2) Retrieval/importance-based 方法（H2O、TOVA、SnapKV、PyramidInfer）——基于 attention scores 动态选择重要 token 保留，精度较好但依赖完整 attention maps，与 FlashAttention 不兼容，导致实际设备上 throughput 低，且缓存全量 KV cache 导致 O(T) 内存复杂度，长序列 OOM。

  StreamingLLM / H2O 的全栈执行例子：
  - **算法层**：StreamingLLM 所有层保留相同的最远 k 个 token（attention sink + sliding window），每步新 token 进入时淘汰最早的非 sink token。H2O 在 prefill 阶段计算累积 attention scores（A2S），取 top-k 高分的 "heavy hitter" token 保留在 KV cache 中，其余淘汰。
  - **系统框架层**：基于 PyTorch/HuggingFace Transformers 推理 pipeline。H2O 需修改 attention 实现以获取 prefill 阶段的 attention scores。可使用 FlashAttention-2 加速部分计算。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 用于标准 attention。H2O/TOVA 的 token selection (TopK + index gather) 在 GPU 上执行，但 prefill 阶段需要 materialized attention scores，与 FlashAttention 的 online softmax 冲突（FlashAttention 不产出完整 S 矩阵）。
  - **硬件架构层**：NVIDIA A100 80GB / H200 GPU，无专用硬件修改。

  Baseline 核心缺陷：
  1. **Recency 方法精度差**：StreamingLLM 在所有层缓存相同的近期 token，在固定 cache budget 下无法覆盖更早的关键长距离依赖 token，导致长上下文理解任务精度大幅下降。
  2. **Importance 方法不兼容 FlashAttention**：H2O/TOVA/SnapKV 依赖 prefill 阶段完整的 attention maps 来评估 token 重要性，与 FlashAttention 的 IO-aware tiling 和 online softmax 设计冲突（FlashAttention 不物化完整的 S ∈ R^{n×n}），导致这些方法在实际设备上要么无法使用 FlashAttention（慢）要么需要特殊适配（复杂）。
  3. **Importance 方法内存不可控**：H2O/TOVA 需先缓存全量 KV 再做淘汰，KV cache 内存复杂度 O(T)，长序列必然 OOM，无法支持连续无限生成长度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LaCache 提出两个核心设计——Ladder-Shaped KV Cache Pattern 和 Iterative Compaction——分别解决 baseline 的精度和内存矛盾。

  **设计 1: Ladder-Shaped Pattern → 解决 Recency 方法的精度问题**：
  核心 Insight：近期 token 对生成很重要，但其 KV 状态不需要所有层都处理。不同层可以存储不同位置 token 的 KV cache——浅层存早期 token，深层存近期 token。这形成阶梯状（ladder）的 KV 存储模式：每层保留 O 个 token 的 KV cache，但每层保留的 token 集合不同步右移 (S-O) 个位置。在相同总 cache budget C = L × O 下，ladder pattern 覆盖的 token 跨度远大于 StreamingLLM（后者所有层缓存同一组 token，有效跨度仅为 O）。形式化：每个 token 至少被 S 个不同层覆盖，信息保留下界得到保证。

  **设计 2: Attention-Free Eviction → 解决 Importance 方法的 FlashAttention 不兼容问题**：
  LaCache 故意不依赖 attention maps 进行 token 重要性评估。Ladder pattern 是静态的（基于位置的），不随输入改变，因此无需 materialize attention scores。这使 LaCache 与 FlashAttention 天然兼容，在实际设备上实现高 throughput（Fig. 7：Ladder pattern achieves Pareto-optimal score-throughput trade-off on H200）。

  **设计 3: Iterative Compaction → 解决连续生成的内存 OOM 问题**：
  当 KV cache 达到预设容量后，对已压缩的 cache 再次应用 ladder pattern eviction，释放空间给新 token。随着迭代次数增加，老 token 被越来越激进地压缩，新 token 保留更多。这实现 O(1) 内存复杂度的连续无限生成，且天然遵循 recency bias（近期 token 信息保留更多）。

  LaCache 的全栈执行例子（Llama2-7B-Chat, cache budget 512, 16K decoding length）：
  - **算法层**：
    1. Prefill 阶段：正常计算 Q,K,V 投影，生成完整 KV cache
    2. Ladder pattern eviction：对每层 l = 1..L，确定保留范围 [start_l, end_l)，其中 start_l = (l-1)×(S-O), end_l = start_l + O。每层仅保留该范围内的 KV 状态。S ≈ num_layers × compression_ratio（理解任务）或 S = L/4（语言建模）。O = S/2（语言建模，保证语义连续性）。
    3. Decode 阶段：使用压缩后的 ladder KV cache 进行 attention，新 token 的 KV 追加入 cache
    4. Iterative compaction：当 cache 满时，对已有压缩 cache 再执行 step 2。Ladder pattern 天然淘汰最早 token（ladder 左端），释放空间
  - **系统框架层**：基于 PyTorch 实现。与 FlashAttention 完全兼容——ladder pattern 通过 mask/索引裁剪实现，不干扰 FlashAttention 的 tiling。代码集成到 HuggingFace Transformers attention 模块。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 用于标准 attention 计算。Ladder KV 裁剪通过构建 token indices 实现索引选择（K_cache[l] = K_full[l, indices[l]]），无额外自定义 kernel。
  - **硬件架构层**：NVIDIA A100 80GB GPU 单卡评估。无专用硬件修改。

  **对应解决 Baseline 缺陷的具体设计**：

  1. **Ladder pattern 的"信息覆盖跨度最大化"→ 解决精度问题**：StreamingLLM 在 budget 512 下覆盖跨度仅 512 tokens，LaCache 通过跨层错位存储可覆盖更长上下文。实证：StreamingLLM PPL 退化 35%（512 budget, 1K length），LaCache 仅退化 5%。

  2. **Attention-free 静态 pattern → 解决 FlashAttention 兼容性**：H2O/SnapKV 需 materialized attention maps → 与 FlashAttention 冲突 → 实际设备 throughput 受限。LaCache 的 ladder pattern 无需任何 attention score → 无缝使用 FlashAttention → Fig. 7 实验证明 LaCache 在 score-throughput Pareto 边界上优于所有 attention-based baselines。

  3. **Iterative compaction 的渐进压缩 → 解决 OOM 问题**：H2O 需全量 KV cache → O(T) 内存。LaCache constant cache size → O(1) 内存，实证支持持续生成超 10M tokens（PG19 全量 concatenated，Llama3-8B）。

  4. **关键超参数 S/O 的物理含义和校准**：
     - Span S 决定信息保留下界：S 越大 → 每个 token 被更多层覆盖 → 信息丢失风险越低 → 存储成本越高
     - Overlap O 决定语义连续性：O 越大 → 相邻层间重合 token 越多 → 信息过渡越平滑 → 适合需要全局语义的任务（synthetic tasks）；O 越小 → 信息集中 → 适合局部依赖任务（QA tasks）
     - 消融验证：随机生成 1500+ 种 KV cache pattern 并评估 PPL-cache size trade-off，ladder pattern 位于 Pareto optimality boundary (Fig. 3)

  5. **实证效果（关键数据）**：
     - NIAH (50% cache): LaCache 99.16% accuracy vs StreamingLLM 54.54% on Llama3.2-3B-Instruct-128k
     - RULER (50% cache): LaCache avg 50.88 vs StreamingLLM 44.82 on LongChat-7b-v1.5-32k
     - PG19 (10M tokens): LaCache maintains reasonable PPL throughout, full cache OOM at 160K tokens
     - LongBench (50% budget): avg degradation reduced from StreamingLLM's 2.4→1.5 on Llama2-13B-Chat

## LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

- baseline方法是什么？
  Baseline 是两类主流 KV cache 压缩方法：
  1. **基于 Attention Weight 的方法**（SnapKV、H2O、StreamingLLM 等）：利用 attention weight/score 判断 token 重要性进行驱逐。
  2. **量化方法**（KIVI 等）：压缩 KV 精度但不减少 token 数量，计算量不变。
  3. **滑动窗口方法**（StreamingLLM）：保留 attention sink + 滑动窗口，丢弃中间 token。

  Baseline 在模型推理全栈的执行例子（以 SnapKV/H2O 为例）：
  - **算法层**：Prefill 完成后，对每层计算 attention weight，根据 query-key attention score 选重要 token，中间 token 被驱逐。
  - **系统框架层**：集成于 HuggingFace Transformers 或 vLLM 等推理框架，需要 hook attention 模块获取 attention weight → 与 FlashAttention（不显式 materialize attention matrix）不兼容或需额外开销。
  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：GQA（Grouped Query Attention）模型在 GPU 上执行标准 attention kernel；attention-based 方法的 scoring 需要额外 CUDA kernel 或 PyTorch 操作。
  - **硬件架构层**：论文未明确说明。

  痛点：
  - **Instruction Dependence（指令依赖偏差）**：基于 attention weight 的方法依赖末尾 query（instruction）来评估 token 重要性，导致压缩方向被问题本身引导 → 改变了原始 prompt 的语义分布。
  - **与 FlashAttention 不兼容**：attention weight 需要 materialize attention score matrix → 无法直接兼容 FlashAttention，需要额外计算开销。
  - **高压缩比下 passkey 检索退化严重**：H2O 在 64-digit passkey 任务中（4× 压缩 Llama-3），exact match 仅 35%，partial match 仅 70.8%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **LagKV**，一种完全不依赖 attention weight 的 KV cache 驱逐方法。核心机制：
  - **递归分区 + 滞后参考**：将 KV cache 按 lag size L 分区，每个分区使用**下一个相邻分区**（lag chunk）的统计量（token-wise max/min）作为参考来归一化当前分区，计算 channel-wise 标准差 → softmax 得到 token 重要性分数。
  - **K+V 联合评分**：同时对 Key 和 Value 做归一化和评分，求和得到最终 token score，top-K 保留。
  - **Attention Sink + 滑动窗口保留**：始终保留前 S 个 token 和最后一个分区（作为滑动窗口）。

  论文方法在模型推理全栈的执行例子：

  - **算法层**：
    1. Prefill 阶段：标准 prefill 完成后，在每层对 KV cache 执行 LagKV 压缩（或 chunk-by-chunk prefill 模式下边 prefill 边压缩）。
    2. 分区压缩：设 S=16, L=128。将 KV cache 分为 [sink(0:16)] + [分区0(16:144), 分区1(144:272), ..., 滑动窗口(最后 L+mod)]。对分区 p，用分区 p+1 的 K/V 统计量归一化分区 p，计算 score 后保留 rL 个 token（如 r=0.5 则保留 64 tokens/chunk）。
    3. Decode 阶段：新生成的 token 累积到长度满 L 后参与递归压缩。
    4. 与 attention weight 完全解耦 → 可按任意顺序处理 token，兼容 FlashAttention。

  - **系统框架层**：
    - 集成于 **NVIDIA KVPress**（开源框架 https://github.com/NVIDIA/kvpress），使用 `KVPressTextGenerationPipeline` 包装 HuggingFace model。
    - KVPress 在 `generate()` 过程中 hook 每层的 `past_key_values`，在每次 forward 后对 KV cache 应用 LagKV 压缩策略。
    - 使用示例（论文推断）：
      ```
      from kvpress import KVPressTextGenerationPipeline
      pipeline = KVPressTextGenerationPipeline(
          model=model, tokenizer=tokenizer,
          press=GreedyPress(strategy=LagKVPress(lag_size=128, retention=0.5))
      )
      output = pipeline(prompt, max_new_tokens=256)
      ```
    - 对比 baseline：无需额外 attention weight 计算 → 在 decode 阶段零额外 attention 开销，仅需 O(d_h) per channel 的统计计算。

  - **编译框架层**：论文未明确说明。

  - **Kernel 调度层**：论文未明确说明具体的 kernel 优化或 CUDA kernel 实现。但 LagKV 的计算模式（channel-wise max/min/std + top-K）是简单的归约操作，可在 PyTorch 层面高效实现，不需要修改 attention kernel。论文承诺与 FlashAttention 兼容。

  - **硬件架构层**：论文未明确说明。

  核心创新与对比：
  | 维度 | Baseline (SnapKV/H2O) | LagKV |
  |------|----------------------|-------|
  | 重要性度量 | Attention weight (query-dependent) | KV channel-wise std after lag-normalize (query-free) |
  | 压缩时机 | Prefill 后一次性评估 | 递归分区，prefill + decode 持续压缩 |
  | FlashAttention 兼容 | 不兼容（需 materialize attn） | 兼容（不访问 attn matrix） |
  | 64-digit passkey (4× Llama) | H2O exact match 35% | LagKV exact match 89% (L=1024, r=4×) |
  | 指令依赖性 | 有（末尾 query 决定保留方向） | 无（仅依赖 KV 局部统计） |

## LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

- baseline方法是什么？
  Baseline 是现有 KV cache 淘汰方法（H2O、SNAPKV、SIRLLM）以及 offloading 方法（InfLLM）、量化方法（HF-2BITS）和 sparse attention 方法（MINFERENCE），其全栈执行例子如下：
  - **算法层**：H2O 基于累积 attention score 识别 heavy-hitter token——在 prefill 阶段计算完整的 attention score 矩阵，累加每 token 被各 query 关注的 attention weight 作为重要性分数，保留 top-k 高分 token。SNAPKV 使用 observation window 内的 query token 计算 attention-based 重要性并通过 voting 机制选择保留 token。SIRLLM 使用 token-level entropy 作为重要性度量进行 eviction。这些方法的**核心缺陷**：(a) H2O 和 SNAPKV 的重要性评分是 non-causal 的——依赖后续 token 的 attention score 来判断当前 token 的重要性，导致在 chunked prefill 中只能看到当前 chunk 时严重低估某些未来会被关注的 token 的重要性（local-global discrepancy），Figure 1 显示 H2O/SNAPKV 的 local-global consistency 远低于 LOCRET；(b) H2O 与 FlashAttention 不兼容——需要 materialize 完整的 attention score 矩阵，无法利用高效 attention kernel；(c) SIRLLM 虽为 causal 评分但准确性不足，在 R.PassKey 和 R.Number 等需要精确检索的任务上表现差（准确率 <4%）；(d) InfLLM 依赖 CPU offloading 存储全量 KV cache，CPU-GPU 通信成为瓶颈；(e) MINFERENCE 虽加速 attention 计算但不压缩 KV cache 大小，内存需求仍为全量 KV cache。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline + chunked prefill。FULLATTN 使用 vLLM（含 tensor parallelism）。SIRLLM 使用官方实现（含 CPU 重要性排序操作）。H2O 使用 layer-wise chunked prefill（chunk_size=1024，更大的 chunk 会 OOM）因需全序列 attention scores。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention（FULLATTN、MINFERENCE）。H2O 使用 PyTorch vanilla attention（不兼容 FlashAttention）。SIRLLM 在 GPU 上执行 TopK + index gather + CPU 排序。
  - **硬件架构层**：NVIDIA A800/H800/4090 GPU，无专用硬件修改。

  Baseline 核心缺陷总结：
  1. **Non-causal 重要性评分导致 local-global discrepancy**：H2O/SNAPKV 需要后续 token 信息才能准确评分，在 chunked prefill 中评分不准确，导致关键 token 被错误 evict。
  2. **与高效 attention kernel 不兼容**：H2O 需要 materialize attention matrix，无法使用 FlashAttention，prefill 极慢（464 tok/s vs LOCRET 5080 tok/s）。
  3. **准确性不足**：SIRLLM 的 token-entropy 评分在精确检索任务上失效；量化方法（HF-2BITS）在所有任务上严重退化。
  4. **不压缩 KV cache 大小**：MINFERENCE 通过 sparse attention 减少计算但不压缩 KV cache 存储，总内存仍为 model weights + full KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LOCRET 通过训练小型 retaining heads 预测 causal importance score (CIS)，实现 causal 的、准确的、FlashAttention 兼容的 KV cache eviction。

  LOCRET 的全栈执行例子：
  - **算法层（核心创新——Trained Retaining Heads + CIS-based Eviction）**：
    1. **训练式 Causal 重要性评分 → 解决 Baseline 缺陷 1（local-global discrepancy）**：
       在每层注入两层 MLP 作为 retaining head R，输入为 [Q, K, V] 拼接，输出为每 token 在各 head 的预测 CIS Ŝ。Ground truth 定义为 answer token 对该 prefix token 的最大 pre-softmax attention score：S[k]_j = max_p (Q_j K_j^T)_{p,k}。因 CIS 仅依赖当前及之前的 token（causal），一旦计算即不变，在 chunked prefill 中评分始终准确，无需等待后续 token。Figure 1 证明 LOCRET 的 local-global consistency 远高于 H2O/SNAPKV。

    2. **Stabilizers 机制 → 解决上下文不连续性**：
       每次 chunked prefill 后，将最后 n_s 个 token 的 CIS 强制设为 +∞（永不被 evict），保证局部连续上下文。Figure 3 消融实验：n_s=0 时模型完全失败（R.Number 准确率 0%），n_s=2500 时恢复。

    3. **与 FlashAttention 兼容 → 解决 Baseline 缺陷 2**：
       Retaining head 的 CIS 预测在前向 pass 中作为附加输出计算，不需要 materialize attention matrix。推理时使用 FlashAttention 加速全部 attention 计算。

    4. **LOCRET-Q 感知 Query → 解决 query-driven task 失效**：
       训练时将 query token 前置，CIS label 基于包含 query 的 attention。推理时 query 在序列首部确保所有 eviction 感知 query。RULER 上 LOCRET-Q 达 75.54%（vs LOCRET 34.33%）。

    5. **训练开销极小**：仅需 <1 GPU 小时（Phi-3-mini-128K: 0.47h, Llama-3.1-8B: 0.80h），可训练参数仅占 8% 和 2.5%。训练对各种超参数和数据集鲁棒（Figure 6, Table 15-17）。

  - **系统框架层**：基于 PyTorch + HuggingFace Transformers 实现 chunked prefill 推理。推理流程遵循 Algorithm 1：split chunks → per-chunk forward with retaining heads → concat KV cache + scores → Top-b selection → evict low-score units → stabilizers protected → final n_loc tokens → decoding。支持所有 decoder-only LLM（MHA/GQA）。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：FlashAttention-2 加速 attention 计算。Retaining head 的小型 MLP 不引入明显延迟（Table 20: w/ R 19153 tok/s vs w/o R 20304 tok/s at 4096 ctx，差距来自系统波动非 overhead）。Cache eviction 的 TopK + gather 为 GPU 标准操作。

  - **硬件架构层**：NVIDIA A800/H800 GPU（训练+主要评估）；NVIDIA 4090 24GB（消费级设备验证）；无专用硬件修改。

  **对比 baseline 的关键差异**：
  - **H2O/SNAPKV (non-causal)** → LOCRET (causal by training)：H2O 在 ∞Bench chunked prefill 时 accuracy 从 FULLATTN 的 48.40 降至 21.18，LOCRET 保持 47.57。特别是 R.Number：H2O 3.39、SNAPKV 2.54 vs LOCRET 97.46。
  - **SIRLLM (token-entropy)** → LOCRET (trained CIS)：SIRLLM R.PassKey 1.69 vs LOCRET 100.00（Phi-3-mini-128K on 4090）。
  - **H2O (vanilla attention)** → LOCRET (FlashAttention)：4090 上 R.PassKey 128K 推理速度 LOCRET 5080 tok/s vs H2O 464 tok/s（~11× speedup）。
  - **MINFERENCE (full KV cache)** → LOCRET (compressed KV cache)：Peak GPU memory LOCRET 17.71 GB vs MINFERENCE 27.63 GB (LongBench, Table 9)。
  - **HF-2BITS (quantization)** → LOCRET (eviction)：∞Bench avg Phi-3-mini-128K: HF-2BITS 2.03 vs LOCRET 34.73。
  - **10M token context**：LOCRET 1747.6× 压缩比，R.PassKey 100% 准确率。
  - **LOCRET 与 pooling 方法正交**：LoCoCo + LOCRET combination 28.70 > LoCoCo 26.01 / LOCRET 27.96 (Table 11)。
  - **LOCRET 与 quantization 兼容**：LOCRET-4bits 平均退化仅 0.85 vs FULLATTN-4bits 退化 0.56 (Table 10)。

## LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

- baseline方法是什么？
  KiVi (Liu et al., 2024c)：一种 training-free 的非对称 2-bit KV Cache 量化方法。KiVi 仅保留最近 R 个 token 为 BF16 全精度，将所有更早的 token 量化为 2-bit。核心假设是"最近的 token 总是最重要的"，因此采用简单的均匀窗选择策略。Per-channel 量化，group size 64。

  Baseline 全栈执行例子（以 Llama3.1-8B-Instruct 解码阶段为例）：
  - **算法层**：KiVi 的非对称 INT2 量化。KV Cache 中第 1 到第 (L-R) 个 token 的 K/V 被量化为 INT2（per-channel, group=64），最近 R=128 个 token 保持 BF16。解码时对量化 token 做 dequantize→BF16，与全精度 token 拼接后计算标准 Scaled Dot-Product Attention。
  - **系统框架层**：HuggingFace transformers pipeline。Cache 类统一管理 KV 存储与量化/反量化。每次生成新 token 时触发量化逻辑——当 reserved tokens 超过 R 时将最早的全精度 token 量化为 INT2。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch GPU kernels（matmul + softmax）。反量化操作（INT2→BF16）插入在 attention 计算前，每个 decoding step 的 K/V 加载后执行，无 custom fusion。
  - **硬件架构层**：NVIDIA GPU（H100 用于效率测试）。KV Cache 存储在 GPU HBM 中，每个 decoding step 从 HBM 加载全部 K/V 到计算单元。量化 KV 通过反量化恢复为 BF16 后参与 attention 计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LogQuant 将 KiVi 的"均匀最近窗"替换为"对数分布 token 选择"，核心洞察是：attention spikes（高注意力分数的位置）遵循对数分布——距离当前位置越远的 token，其 attention spikes 的密度越稀疏（见图 1）。基于此，LogQuant 以递减密度保留 token：最新 W 个 token 密度 p，次新 W 个 token 密度 p/2，再次新 W 个 token 密度 p/4……以此类推，形成自然对数稀疏性。

  三个具体设计解决 KiVi 的三大缺陷：

  **缺陷 1：KiVi 的均匀窗硬截断会丢失远处的关键 token**。KiVi 保留最近 128 个 token 但丢弃了所有更早的 token，然而许多问题（如"法国的首都是什么？"）的答案 token 可能位于远处。LogQuant 的对数选择在远处仍有稀疏保留——每 2^k 个位置保留 1 个 token——捕获了这些"远处但重要"的 token。实验证明对数选择方案的 token coverage（公式 1：所选 token 的平均 attention score）在所有模型上均优于 KiVi/StreamingLLM/H2O（图 4）。

  **缺陷 2：KiVi 未区分量化与 eviction 的信息损失差异**。论文证明在相同的对数选择方案下，量化（降低精度）比 eviction（移除 token）保留更多信息。这是因为 softmax 归一化使得 eviction 移除 token 后剩余 token 的 attention 权重被重新分配，造成更大的 attention 分布偏差。量化仅降低单个 token 的数值精度，不改变 attention 的归一化结构。实验验证：L1 attention 误差——LogQuant (2-bit) 432.50 vs LogQuant (Eviction) 1076.70，证明量化策略优于 eviction（表 2）。

  **缺陷 3：KiVi 未利用 attention 计算的置换不变性优化内存布局**。LogQuant 证明 A·V = A_P·V_P（P 为任意置换），即 K/V Cache 中 token 的排列顺序不影响 attention 输出。这使得 LogQuant 可以将全精度 token 和量化 token 分别连续存储（而非按原始位置交错存储），改善内存局部性，减少碎片化，无需额外计算开销。

  论文方法全栈执行例子（以 Llama3.1-8B-Instruct 解码阶段为例）：
  - **算法层**：LogQuant 的 log₂-分布式 2-bit 量化。Algorithm 1 的 APPENDTOKEN 过程：KV Cache 起始为空，依次追加 token。当 cache 长度 < 3W 时直接追加（全精度）。当长度达到 3W 时，将前 2W 个 token 做步长=2 子采样（保留一半），与新 token 拼接，总长回到 2W。反复执行后，cache 中 token 的保留密度自然呈现：Window_0 密度 p，Window_1 密度 p/2，Window_2 密度 p/4……非保留 token 量化为 INT2。W = ⌊KiVi_R/3⌋ = ⌊128/3⌋ = 42，实际最多 126 个全精度 token。压缩率 ≈ 16L / (2(L-126) + 16×126)。
  - **系统框架层**：继承 HuggingFace transformers Cache 类的 derived class LogQuantCache。量化后端使用 Quanto（Key-per-channel 策略，也可切换 HQQ）。position-agnostic 重组：将全精度 K/V 与量化 K/V 分别连续存储。与 HuggingFace 推理 pipeline 无缝兼容。batch size 比 BF16 baseline 增加 60%。
  - **编译框架层**：论文未明确说明。未来工作提及 operator fusion——将 dequantization 与 attention 计算融合为单一 kernel，直接在量化数据上计算 attention，消除反量化开销。
  - **kernel调度层**：标准 PyTorch attention kernels。每个 decoding step 中，量化 K/V 需先反量化至 BF16 再参与 attention 计算——反量化操作是当前 throughput 瓶颈之一。25% 吞吐量提升主要来自更大 batch size（内存节省释放了 batch 扩展空间）而非单步计算加速。论文明确指出可进一步通过 fused kernel 优化。
  - **硬件架构层**：NVIDIA H100 48G MIG。KV Cache 存储于 GPU HBM——2-bit 量化将非保留 token 的内存从 16-bit/entry 压缩至 2-bit/entry（~8× reduction）。全精度保留 126 token（BF16），其余 INT2。48GB 内存限制下，batch size 从 baseline 的 X 增至 1.6X。Dequantization 在 HBM→SRAM 加载后、Tensor Core 计算前执行。

  对比 baseline 的关键差异：
  - **KiVi 均匀窗 (uniform recent window)** → **LogQuant 对数窗 (log-sparse window)**：KiVi 保留最近 128 token 但完全丢弃更早 token；LogQuant 以 log₂ 递减密度在更远位置保留 token，捕获跨距离的关键信息。
  - **KiVi 仅量化无选择性** → **LogQuant 量化+对数选择联合设计**：KiVi 对所有非保留 token 一视同仁地量化；LogQuant 先通过对数分布选择保留更重要的 token（全精度），再对剩余 token 量化——重要性判断基于 attention spike 的位置分布规律而非单一时间衰减假设。
  - **未利用置换不变性** → **利用 A·V = A_P·V_P 重排**：KiVi 按原始位置存储 K/V；LogQuant 将全精度和量化 token 分别连续存储，改善内存局部性。

## MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference

- baseline方法是什么？
  现有 KV cache 压缩方法（H2O、SnapKV、PyramidKV）和 LOOK-M（多模态 KV cache 压缩 baseline），其全栈执行例子如下：

  - **算法层**：H2O 和 SnapKV 采用 eviction-based 策略——基于累积注意力分数（H2O）或观察窗口注意力（SnapKV）选择保留 heavy-hitter token，丢弃低分 token。PyramidKV 使用静态渐进式层间缩减——前层保留较多 KV cache，后层线性递减，不考虑跨层注意力密度的变化。LOOK-M 针对多模态场景做了优化但使用**固定（uniform）分配策略**——所有层分配相同的 KV cache 大小。所有 baseline 的核心缺陷：(a) 忽视层间注意力密度差异——如 Figure 2 所示，早期层（如 Layer 1）注意力密度高（高熵），深层（如 Layer 24）注意力集中于少数关键 token（低熵），统一分配导致密集层信息丢失或稀疏层资源浪费；(b) 丢弃低分 token（eviction-based 方法）或仅保留高分 token（LOOK-M），完全丢失了被丢弃 token 中可能包含的上下文信息；(c) 未充分利用跨模态（文本↔视觉）注意力分布特征来指导 cache 分配——多模态场景下文本-视觉的跨模态交互产生与纯文本注意力不同的分布模式。

  全栈执行例子（Baseline / H2O on MLLM）：
  - 算法pipeline：prefill 阶段计算累积注意力 A_s = Σ_i A[i,:]，按 A_s 排序保留 top-N token 组成 KV cache，丢弃其余 token。各层统一使用相同 budget。
  - 系统框架：HuggingFace Transformers 推理 pipeline，修改 attention 层以支持 KV cache eviction。与 MLLM（LLaVA系列、InternVL等）的视觉 encoder + LLM decoder 架构兼容。
  - 编译框架：论文未明确说明。
  - kernel调度：FlashAttention（GPU 标准 attention kernel），eviction 操作在 GPU 上执行 TopK + index gather。H2O/SnapKV 需要 prefilling 阶段 materialize attention scores（与 FlashAttention 的 online 计算冲突）。
  - 硬件架构：NVIDIA A100 GPU，无专用硬件修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MEDA 提出三个核心设计，分别对应 baseline 的三项缺陷：

  **1. 跨模态注意力熵引导的动态层间分配 → 解决缺陷(a)层间密度差异**：
  观察到不同层的跨模态注意力密度差异显著（Figure 2：Layer 1 注意力熵高、分布分散；Layer 24 注意力熵低、集中于关键 token）。引入**跨模态注意力熵 E_CM^l**（公式 6：E_CM^l = -(E_TV^l + E_VT^l)，分别衡量文本→视觉和视觉→文本的注意力不确定性），使用 **inverse entropy softmax allocation**（公式 7：α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ）为每层动态分配 KV cache——注意力集中的层（低熵）分配少，注意力分散的层（高熵）分配多。对比 baseline 的统一分配：PyramidKV 的静态线性递减（不考虑实际密度变化）、LOOK-M 的固定分配（所有层相同），MEDA 能自适应匹配每层的注意力分布特征，在总 budget 不变下实现更优的性能。

  **2. KV pair 合并替代丢弃 → 解决缺陷(b)信息丢失**：
  不同于 H2O/SnapKV 直接丢弃低分 token 的做法，MEDA 对未选中的 less important tokens 执行 **many-to-one nearest-neighbor matching**（公式 11：基于 key token 的 cosine similarity 将每个不重要 token 匹配到最近的保守 token），然后通过**平均合并**（公式 12：k_j ← (k_j + Σ k_i) / (|N_j| + 1)）将被丢弃 token 的信息整合进保守 token 中。这一设计保留了全局上下文的完整性——即使是低注意力分的 token，其携带的视觉细节或文本语义信息也不会完全丢失，而是融入到相似的保守 token 中。ablation 验证：移除 average merging 导致 CLEVR-Change ROUGE-L 从 18.9 降至 18.2，Spot-the-Diff 从 18.2 降至 17.3（Table 5）。

  **3. 多模态 text-prior 选择策略 → 解决缺陷(c)跨模态特征利用不足**：
  在 KV pair 选择阶段，对文本 token 的累积注意力分数加 max(A_s) 偏置（公式 9），确保关键的文本语义 token 在高压缩比下仍被优先保留。同时保留最近 M 个 token 的上下文窗口（recent context window），支持模型对近期上下文的记忆需求。这一策略专门针对多模态场景设计：文本 token 通常在语义上更关键（如问题描述、指令），优先保留它们确保语义连贯性，同时视觉 token 通过合并而非丢弃保留信息。

  **全栈执行例子（MEDA on LLaVA-NeXT-7B, ρ=0.1, single A100）**：
  - **算法层（核心创新）**：
    1. Prefill 阶段：LLaVA-NeXT-7B（32 layers, CLIP ViT vision encoder + Vicuna-7B LLM）处理多模态输入（text + multi-images 或 text + video frames）。每层计算跨模态注意力熵 E_CM^l。
    2. 动态分配：根据 E_CM^l 用 softmax 公式计算 α_l 和 S_l。早期层（高熵、注意力分散）获得较大 S_l，深层（低熵、注意力集中）获得较小 S_l。
    3. KV 选择与合并：每层按 S_l 执行 text-prior 累积注意力选择 + nearest-neighbor 平均合并，生成压缩后的 (K_c, V_c)。
    4. Decoding：使用压缩 cache 逐 token 生成，新 token 的 KV 追加到压缩 cache。
  - **系统框架**：基于 HuggingFace Transformers 实现，即插即用兼容 LLaVA-v1.5-13B、LLaVA-NeXT-7B、InternVL-v1.5-7B（multi-images）以及 LLaVA-Video-7B/32B、LongVA-7B、LongVILA-8B（long-video）。无需额外 fine-tuning。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：使用标准 FlashAttention。跨模态熵计算为额外 O(n_T · n_V) per layer（仅 prefill 时执行一次），KV 选择为 O(L_prompt) TopK。无自定义 kernel。
  - **硬件架构**：NVIDIA A100 GPU，无专用硬件修改。

  **对比 baseline 的关键差异**：
  - **H2O/SnapKV (uniform allocation + eviction)** → **MEDA (dynamic allocation + merging)**：H2O 在所有层统一丢弃低分 token，在 ρ=0.1 时 LLaVA-NeXT-7B MileBench 大幅低于 Full Cache（T-1: 42.0 vs 45.8）。MEDA 动态分配 + 合并保留信息，MileBench 全面接近 Full Cache（T-1: 45.4 vs 45.8），并在 NH（5.5→4.8）和 IR（7.6→7.4）上几乎无损。
  - **PyramidKV (static progressive reduction)** → **MEDA (entropy-guided dynamic allocation)**：PyramidKV 按固定线性递减分配 KV cache（前层多后层少），不考虑各层实际注意力密度。MEDA 基于实时计算的跨模态注意力熵分配，在高熵层（信息密集型）分配更多、低熵层（信息集中型）分配更少。ρ=0.1 时 PyramidKV LLaVA-NeXT-7B IR 仅 3.2 vs MEDA 7.4。
  - **LOOK-M (fixed multimodal allocation)** → **MEDA (dynamic multimodal allocation)**：LOOK-M 虽考虑多模态但使用固定统一分配，MEDA 的熵引导分配在所有 11 个 MileBench sub-task 上均优于 LOOK-M（LLaVA-NeXT-7B: avg ~5-6 points improvement）。
  - **关键效果量化**：ρ=0.1 时 LLaVA-NeXT-7B MileBench 全面接近 Full Cache（多数 sub-task 差距 <2 points）。ρ=0.2 时 LongVA-7B Video-ChatGPT Correctness 2.16 vs Full Cache 2.24（H2O: 1.93）。20% KV cache budget 下 GPU memory 从 2.42 GiB 降至 0.67 GiB（72% 减少），decoding latency 14.61→8.23 ms/token（1.78× speedup）。5% budget 时 latency 降至 5.18 ms/token（2.82× speedup）。

## MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

- baseline方法是什么？
  - **Softmax Self-Attention**：对每个 query q_i，计算与所有 key k_j 的 pairwise 相似度 exp(q_i·k_j/√d)，对所有 value v_j 加权求和。复杂度 O(N²d)，内存 O(N²)。全栈执行例子：输入 token 序列 X → 线性投影 Q,K,V → 计算 QK^T/√d → softmax 归一化 → ×V 输出 → 结果传入下游 FFN/layer norm。当 N 增长时，QK^T 矩阵占据 O(N²) 内存，成为瓶颈。
  - **Linear Attention**：将 softmax kernel 替换为可结合的 feature map φ(·)，使得 Sim(Q_i, K_j) ≈ φ(Q_i)φ(K_j)^T。通过先计算全局 KV summary G = Σ_j φ(K_j)^T V_j ∈ R^(d×d)，查询只需 q̃^T G / q̃^T z。复杂度 O(Nd²)，内存 O(d²)。全栈执行例子：输入 X → 投影 Q,K,V → 应用 φ 得 Q̃, K̃ → 计算全局 summary G = Σ_j K̃_j^T V_j 和 normalizer z = Σ_j K̃_j → 对每个 q̃_i 计算 o_i = (q̃_i^T G) / (q̃_i^T z)。**缺陷**：所有 token 被压缩进同一个 d×d 的全局 summary，rank ≤ d（通常 d_h ≤ 72），导致"全局上下文坍缩"——当 N >> d 时注意力矩阵 rank 不足、熵升高（分布趋于均匀），失去 query-conditioned 的选择性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MHLA (Multi-Head Linear Attention)**：沿 token 维度将序列分为 M 个 non-overlapping blocks，为每个 block 独立计算局部 KV summary，再通过可学习的系数矩阵 Mc 使每个 query block 生成专属的混合 summary。全栈执行例子：
    1. 输入 X ∈ R^(N×d) → 投影 Q,K,V → 应用 φ 得 Q̃, K̃
    2. 沿 spatial/spatiotemporal 维度将 N 个 token 划分为 M 个 blocks
    3. 每 block b 独立计算局部 summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)
    4. 通过可学习系数矩阵 Mc ∈ R^(M×M)（初始化为 locality-biased：m_{i,j}^(0) ∝ 1-dist(i,j)/max_dist），query block i 混合所有 summary：S̃_i = Σ_b m_{i,b} S_b
    5. 输出：o = (q̃^T S̃_i) / (q̃^T z̃_i)，其中 token 级别贡献为 m_{i,b(t)} (q̃^T K̃_t) V_t^T
    6. 结果传入下游 FFN/layer norm → 下一层
  - **解决 Baseline 缺陷的对应关系**：
    - **Rank 限制**（基线：d → MHLA：Σ_b min(N_b, d)）：将单个 d×d summary 拆成 M 个局部 summary，再通过学习混合恢复多样性。当各 block 的 row spaces 线性独立时，rank 可接近 Σ_b min(N_b, d)，远超 d。实测 MHLA 的 attention score rank 显著高于所有 linear attention 变体（Fig. 3b）。
    - **注意力熵过高/稀疏性丧失**（基线：uniform distribution → MHLA：concentrated distribution）：Mc 允许每个 query block 选择性关注相关 blocks（block-level pruning），block 内部再通过 q̃^T K̃_t 区分 token 贡献（token-level reweighting），两阶段机制恢复 query-conditioned 的尖锐注意力分布。实测 MHLA 熵低于 linear attention 甚至 softmax attention（Fig. 3b）。
    - **不引入额外模块**：与 Focused LA（加 DW-Conv）、Inline Attn（加卷积+gating）等不同，MHLA 仅需标准 GEMM 操作和可学习系数矩阵，额外开销 O(M²d²)。当 M² ≤ N 时，主导项仍为 O(Nd²)。
  - **对比 baseline 的关键差异**：
    - **Linear Attention (单个全局 summary)** → **MHLA (M 个局部 summary + 可学习混合)**：Linear Attention 对所有 query 共享同一个 G，导致 query 间无差异（rank ≤ d, entropy 高）。MHLA 每个 query block 通过专有系数 m_i 混合各 block summary，恢复 query-conditioned 选择性。ImageNet DeiT-T: Linear Attn 69.8% → MHLA 75.8%（+6.0%）；DiT-S/2 FID: Linear Attn 89.72 → MHLA 59.80（↓33%）；Wan2.1 视频生成: Linear Attn Total 58.24 → MHLA 82.62（+41%）。
    - **Self-Attention（O(N²) pairwise）** → **MHLA（O(N) block-level + token reweighting）**：Self-Attention 通过 pairwise softmax 实现完全 query-conditioned 但 O(N²) 复杂度。MHLA 通过两阶段（block 选择 × token 重加权）逼近相同效果。DiT-XL/2 256px FID: Self Attn 19.47 → MHLA 20.32（w/o CFG 相近），512px 下 MHLA 吞吐量是 Self Attn 的 2×；视频生成 31500 tokens: MHLA latency 81s vs Wan2.1-FA 166s（2.1× speedup）。

## MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

- baseline方法是什么？
  Baseline 是标准 LLM 推理流程（Standard）和现有内存优化方法（Chunked Prefill、KV Cache Offloading alone）。

  **Standard（无优化）**：全量 prefill + 全量 KV cache on GPU。全栈执行例子（Llama-3-8B, S=128K, d=4096, I=4d=16384, L 层）：
  - **算法pipeline**：输入 X → 逐层 Transformer Block → Attention(X) 使用 FlashAttention → MLP(X) 计算 SwiGLU（W_gate, W_up, W_down），中间激活大小 S×I → LM_Head(last_token)。每层 MLP 产生峰值中间内存 S×I = 128K×16384 ≈ 2.1B floats ≈ 4.2GB (bfloat16)。
  - **系统框架**：PyTorch + HuggingFace Transformers 标准推理流程。FlashAttention-2 优化 attention。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：FlashAttention-2 kernel 做 attention。MLP 使用标准 cuBLAS GEMM。
  - **硬件架构**：单张 NVIDIA A100 80GB GPU。Standard 在 155K tokens 时 OOM（超出 80GB 显存）。

  Baseline 的核心缺陷：(1) **MLP 中间激活主导 peak memory**——prefill 阶段 MLP 层的中间激活 S×I（I≈4d）是峰值内存的最大贡献者，远超 attention 优化后的 KV cache 和 attention 计算内存；(2) **Chunked Prefill 重复 forward-pass 开销**——将整个 prefill 切分为多个 chunk 串行处理，每个 chunk 需完整 forward（attention+MLP+LM Head），导致重复 kernel launch 和 extra computation；(3) **Offloading alone 收益有限**——仅 offloading KV cache 不降低 MLP 中间激活，因此 peak memory 减少不明显（因为 MLP 中间激活仍是瓶颈）。

  Chunked Prefill 的全栈执行例子：
  - **算法pipeline**：输入 X 按 chunk_size=C 切分为多个 chunk → 对每个 chunk X^(i) 执行完整 Transformer forward → 累积 KV cache → 所有 chunk 完成后进行 decode。每 chunk 计算 attention+MLP+LM Head 全部子层。
  - **系统框架**：PyTorch + HuggingFace。通过多次 forward 调用模拟分批处理。TensorRT-LLM 中也有实现。
  - **kernel调度**：每次 forward 有独立的 kernel launch overhead，多次 forward 累积开销显著。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MOM 通过两项核心设计解决 baseline 缺陷：

  **1. Mini-Sequence MLP Partitioning → 解决缺陷(1)（MLP 中间激活主导 peak memory）和缺陷(2)（Chunked Prefill 重复开销）**：
  Attention 层保持完整序列处理，仅将 MLP 层输入沿序列维度切分为 M 个 mini-sequences（每个 N=S/M），逐个通过 MLP。由于 attention 层不变，KV cache 的生成和使用不受影响。Mini-sequences 的中间激活从 S×I 降至 (S/M)×I，且均在单次 forward pass 中完成（非多次 forward），无 Chunked Prefill 的重复开销。最后一个 MLP 层和 LM Head 仅处理最后一个 token，进一步减少计算。

  全栈执行例子（MOM, Llama-3-8B, S=128K, C=8192）：
  - **算法pipeline**：
    1. Attention 层：完整处理 S=128K，使用 FlashAttention/GQA 保持不变，生成完整 A ∈ R^{128K×4096}
    2. KV cache：更新后 offload 到 CPU（每层 attention 完成时立即 offload）
    3. 非最后 MLP 层：将 A 按 C=8192 切分为 16 个 mini-sequences A_i ∈ R^{8192×4096}，逐个计算 MLP(A_i)，拼接后传入下一层
    4. 最后 MLP 层：仅取 A_last = A[-1:] ∈ R^{1×4096}，计算 MLP(A_last)
    5. LM_Head(A_last) → logits → 开始 decode
    6. Decode 前：将所有层 KV cache 从 CPU reload 到 GPU
  - **系统框架**：HuggingFace Transformers，使用 OffloadedCache 管理 KV cache 的 CPU/GPU 传输。仅需修改 MLP 层和 LM Head 的输入处理逻辑，attention 和其余组件不变。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：MLP 的 GEMM 操作由原本的 [S, d]×[d, I] 变为 [N, d]×[d, I]（N=S/M），更小的矩阵乘更利于 GPU L2 cache 命中。论文观察到 Mini-sequence 甚至能提升吞吐量（因为 shorter sequence chunks fit better into GPU cache）。
  - **硬件架构**：单 A100 80GB GPU。MOM 将最大 context 从 155K 扩展至 455K（~3×），内存节省 >50%。

  **2. KV Cache Offloading → 解决缺陷(3)（Offloading alone 收益有限）**：
  当 MLP 中间激活被 Mini-sequence 大幅降低后，KV cache 成为剩余内存中的主要占用者。此时 offloading KV cache 到 CPU 变得有意义——因为 MLP 内存不再是瓶颈，offloading 能进一步释放 GPU 内存供更长的序列使用。Offloading 与 Mini-sequence 结合产生协同效应：Mini-sequence 降低 MLP 中间激活 → offloading 降低 KV cache → 两者叠加释放的 GPU 内存远多于各自单独使用。

  **对比 Chunked Prefill 的关键差异**：
  - Chunked Prefill：多次 forward pass，每 chunk 重复 attention + MLP + LM Head → overhead 随 chunk 数增加
  - MOM：单次 forward pass，attention 一次完成，仅 MLP 逐 mini-sequence 循环 → 无重复 forward overhead
  - MOM 比 Chunked Prefill 延长 context 35% more（455K vs Chunked Prefill 的扩展量）
  - MOM + offloading throughput 远优于 Chunked Prefill + offloading（后者数据传输开销 >75%）

  **效果量化**（Llama-3.2-8B, A100 80GB）：
  - Peak memory @155K: Standard 72GB → MOM 35GB（~51% reduction）
  - Max context: Standard 155K → MOM 455K（~3×）
  - Prefill TTFT @144K: Standard 34.9s → Mini-sequence 34.0s（slightly faster），MOM 37.3s
  - Decode speed @144K: Standard 11.63 tok/s → MOM 11.60 tok/s（几乎无退化）
  - Accuracy: Logits identical，Needle test 等同

## MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

- baseline方法是什么？
  Baseline 是标准 autoregressive decoding（AR）+ 传统 speculative decoding 的小 draft model 方法。全栈执行例子（以 LLaMA-3.1-8B, batch=128, S=32000, 8×H100 为例）：

  - **算法层（Autoregressive Decoding baseline）**：标准逐 token autoregressive generation，每步生成一个 token。每步需加载完整 KV cache（~25.2 GB for B=128, S=32K）和 model weights 并执行一次完整 forward pass。延迟 = KV loading time + MLP compute time + attention compute time。在大 batch + 长序列下，KV cache loading 成为主导瓶颈（memory-bound），但每个 token 仍需独立生成，吞吐 = B / T_per_token。

  - **算法层（传统 SD + 小 draft model baseline，如 LLaMA-3.2-1B draft for LLaMA-3.1-8B）**：使用小型独立模型作为 draft，生成 γ 个候选 token 后由 target model 并行验证。短序列下有效（减少参数加载摊销），但长序列 + 大 batch 时出现三个缺陷：
    (a) **验证成本过高**：大 batch + 短序列时推理变为 compute-bound，$T_V/T_T$ 显著上升（Figure 2b, S=1000 时 $T_V/T_T$ 从 1.0 升至 ~3.5），因为验证需对所有 draft tokens 做完整的 attention+FFN 计算
    (b) **小 draft model 的 KV cache 占比大**：长序列下 KV cache 超过参数内存，小 draft model 的 KV cache 可能达到 target model 的 38%~140%（如 LLaMA-3.1-8B/LLaMA-3.1-70B pair, Figure 4a），draft 成本 $T_D/T_T$ 不降反升
    (c) **接受率不足**：model compression 的 draft-target pair（如 LLaMA-3.2-1B → LLaMA-3.1-8B）接受率 < 85%（Figure 1c），频繁的 rejected verifications 浪费计算资源

  - **系统框架层（传统 SD serving baseline）**：标准 speculative decoding pipeline —— draft model 生成 γ 个 token → target model 并行验证 → greedy matching 确定接受数 → 重复。现有研究（Liu et al., 2024a; Su et al., 2023; Miao et al., 2023）显示 SD 在大 batch 下失效（speedup < 1），因此 serving 系统通常仅在小 batch 下启用 SD，大 batch 回退到 AR。

  - **编译框架层/硬件架构层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. **传统 SD 的 batch-size 限制**：现有认知认为 SD 仅对小 batch 有效，大 batch 下验证成本过高导致 speedup < 1，这限制了 SD 在 high-throughput serving 中的应用
  2. **小 draft model 在长上下文下的内存劣势**：长序列 KV cache 膨胀使小 draft model 的 KV 内存占比超过模型参数压缩带来的优势
  3. **Model compression 接受率天花板低**：压缩模型权重的接受率难以突破 90%，而高接受率是大 batch SD 效率的关键
  4. **静态 KV compression（如 StreamingLLM）接受率低**：虽无搜索开销但接受率上界低，影响 speedup 上限

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MagicDec 的核心洞察：在长上下文 + 大 batch 场景下，KV cache 加载（而非计算）成为推理瓶颈，此时 speculative decoding 的验证成本 $T_V$ 与正常解码 $T_T$ 共享相同的 KV loading 成本，$T_V/T_T \approx 1$。同时，通过压缩 KV cache（而非压缩模型权重）实现 self-speculation，既能获得接近 90%+ 的接受率，又能使 draft 的 KV 远小于 target 完整 KV（$T_D/T_T \to 0$），从而在大 batch 下实现 speedup > 1 甚至随 batch 增大而提升。

  MagicDec 的全栈执行例子（以 LLaMA-3.1-8B SnapKV self-speculation, batch=128, S=32000, 8×H100）：

  - **算法层——Bottleneck Shifting 识别（Section 3.2）**：
    通过 roofline 模型分析，随着 context length 增长超过临界值 $S_{\text{inflection}}$（LLaMA-3.1-8B 上约 4000 tokens，Figure 2c），推理从 compute-bound 转向 memory-bound。此时 KV cache loading 成为瓶颈，$T_V/T_T \approx 1$ 因为 verify 和 decode 共享相同的 KV 预算。同时 draft 使用压缩 KV（budget K=512~2049 << S=32000），$T_D/T_T$ 随 batch 增大而下降（Figure 2a），因为 target 受 KV 瓶颈影响更大。结果：speedup 随 batch 增大反而提升。
    → **解决缺陷 1**：证明了 SD 在大 batch 下有效，条件是 $S > S_{\text{inflection}}$

  - **算法层——压缩 KV 的自推测（Self-Speculation, Section 3.3）**：
    使用 target model 自身 + 稀疏 KV cache（SnapKV/StreamingLLM）作为 draft。关键：(a) KV compression 接受率远超 model compression（Figure 1c：Top-K KV 在 >90% 接受率 vs model compression <85%）；(b) 压缩 KV 使 draft 内存远小于 target，$T_D/T_T$ 随 S 增长趋于 0（Figure 3a）；(c) draft model 是 target 自身，共享 weights，无额外参数加载开销。
    → **解决缺陷 2 和 3**：KV cache compression 替代 model compression，记忆效率更高且接受率更高

  - **算法层——最优 Drafting 策略选择（Section 4, 公式 4）**：
    $$\min_{T_{select}, K, \gamma, \alpha} \left[ \frac{1}{\Omega(\gamma, \alpha)} \left( \frac{\gamma \cdot (T_D(B, K) + T_{select}(B, S, K))}{T_T(B, S)} + \frac{T_V(B, S, \gamma)}{T_T(B, S)} \right) \right]$$
    考虑三个维度的 trade-off：(a) draft model size（self-speculation vs 小 draft model vs 混合）；(b) draft KV budget K（小 K 降低 $T_D$ 但降低 α，大 K 提高 α 但增加 cost，Figure 5c）；(c) KV compression algorithm type（static SnapKV/StreamingLLM vs dynamic PQCache/TopK，前者无搜索开销 $T_{select}=0$ 但接受率上限较低，后者接受率高但 $T_{select}$ 随 batch 增长）。根据任务特征（检索型需高接受率 → dynamic 可能更优；生成型接受率差异小 → static 成本低更优）和 batch size（大 batch 下 $T_{select}$ 成本放大 → static 更优）选择。
    → **解决缺陷 4**：不依赖单一 KV 方法，而是根据 model/hardware/task 特征自适应选择最优策略

  - **系统框架层（Self-implemented backend + MLC-LLM）**：
    Prefill 阶段：dense FlashInfer attention + SnapKV selection → 生成压缩 KV cache。Decode 阶段：CUDA graph 封装的 draft-verify loop。Draft 用压缩 KV + torch.compile + Triton matmul + TP-embedding 加速；Verify 用完整 KV + FlashInfer。所有结果通过 greedy decoding 验证（lossless — 与 AR 输出完全一致）。
    
    Speedup 结果：LLaMA-3.1-8B SnapKV self-speculation, 8×H100: batch=41, S=100K → 2.51x (cwe); batch=128, S=32K → 2.01x; batch=64, S=64K → 2.36x (cwe)。

  - **编译框架层/kernel调度层**：torch.compile 编译模型 + Triton-based matmul 加速 MLP + FlashInfer attention kernel + CUDA graphs 消除 launch overhead。非论文核心贡献，作为基础设施使用。

  - **硬件架构层/芯片设计层**：论文未明确说明。

  效果量化总结：
  - LLaMA-3.1-8B, SnapKV self-spec, 8×H100: batch=41, S=100K, cwe → 2.51x speedup
  - LLaMA-3.1-8B, SnapKV self-spec, 8×H100: batch=64, S=64K, cwe → 2.36x speedup
  - LLaMA-2-7B-32K, StreamingLLM self-spec, 8×A100: batch=64, S=8K → 1.43x; batch=128, S=8K → 1.55x
  - Mistral-7B-v0.3: up to 2.06x; Qwen-2.5-7B: up to 1.89x; Qwen-2.5-32B: up to 1.51x
  - 关键性质：所有 speedup 均为 lossless（greedy decoding, 与 AR 输出完全一致）

## MagicPIG: LSH Sampling for Efficient LLM Generation

- baseline方法是什么？
  Baseline是TopK attention及其搜索近似变体（如Quest的block-level dynamic sparsity）。TopK attention仅选择attention scores最高的K个key-value对参与注意力计算，本质上是一个有偏估计（biased estimator）。其全栈执行例子：
  - 算法层：Quest使用page-level分块，计算q与每个page summary的内积近似TopK选择，page_size=16时Cost_1=1/16(搜索开销)+Cost_2=手动控制(稀疏计算开销)
  - 系统框架层：全注意力在GPU上执行，KV cache全部驻留在GPU HBM，FlashAttention / FlashDecoding进行IO-aware加速
  - Kernel层：GPU执行标准Softmax(qK^T/√d)V，memory-bound瓶颈
  - 缺陷：(1) TopK丢弃了低attention score tokens中大量有效信息(长尾分布下Top20% token仅覆盖70-80% attention score)，在聚合任务(CWE, FWE)中准确率显著下降；(2) 搜索TopK本身开销大(IVF需访问30% key才能获得精确TopK)；(3) KV cache仍全量驻留GPU显存，限制了最大batch size和context length。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MagicPIG提出基于LSH的采样方法来估计attention输出，替代TopK的确定性选择。核心思想是将attention output视为从attention score分布w中采样的期望值o=E_{i~w}[v_i]，通过Self-normalized Importance Sampling + LSH SimHash实现高效的无偏/近似无偏估计。全栈执行例子：
  - 算法层：q在GPU上计算K×L bit SimHash码 → CPU上查询L张哈希表，收集至少2表中碰撞的key集合S → 计算每个采样key的碰撞概率u_i = 1-(1-p_i^K)^L-L·p_i^K·(1-p_i^K)^{L-1} → 注意力估计ō=Softmax(qK_S^T/√d - log(u))·V_S。关键创新：(a) centering预处理解决q和k方向几乎相反导致LSH失效的问题，(b) 至少2表碰撞机制提升采样质量，(c) on-device cache保留sink+local tokens避免丢失关键信息。
  - 系统框架层：GPU执行compute-bound的线性投影和HashEncode(Cost_1≈0)，CPU执行memory-bound的哈希表查询和稀疏注意力(Cost_2=2%~5%全注意力FLOPs)，KV cache完整offload到CPU DRAM
  - Kernel层：GPU PyTorch执行线性层+随机投影(3.8%~8.5%额外计算)，CPU FBGEMM bfloat16执行稀疏qK^T和weighted sum
  - 对应解决：(a) 采样比TopK更准确——oracle sampling减少4×估计误差，在CWE和FWE任务上MagicPIG甚至超过exact TopK 3-8%；(b) LSH采样Cost_1≈0，远低于TopK搜索的3-6%；(c) KV cache offload到CPU DRAM使batch size达baseline的12×，突破GPU显存限制。

## MoBA: Mixture of Block Attention for Long-Context LLMs

- baseline方法是什么？
  Baseline 是标准 Full Attention（Transformer self-attention）以及静态稀疏注意力方法（Sliding Window Attention / Attention Sink / 各类 static sparse patterns）和线性注意力方法（Mamba / RWKV / RetNet）。

  全栈执行例子（以 Full Attention baseline, Llama 8B, 1M context, single GPU prefill）：
  
  - **算法层（Full Attention）**：标准 scaled dot-product attention: O = Softmax(QK^T/√d)V。对于 1M context, Q,K,V ∈ R^{1M×128×8}（per head），QK^T ∈ R^{1M×1M}，FLOPs = O(N²d) ≈ 2×10^12 per head。prefill 1M tokens 需 ~30 分钟（FlashAttention applied）。KV cache = 2×L×N×d×h_kv ≈ 2×32×1M×128×8 = 64GB (BF16)。复杂度 O(N²) 在长 context 下计算和内存开销 prohibitive。

  - **算法层（Sliding Window Attention baseline）**：每个 query 仅关注最近 W 个 token（如 W=4096）。可视为 MoBA 特例：gating network 固定选择最近 blocks。缺陷：(a) task-specific——对需要跨长距离检索的任务（如 Needle in a Haystack）性能崩溃；(b) 丢失中间 context 信息，模型无法利用非局部的 key information；(c) 需要对 W 外的 token 做因果关系的信息路由，但缺乏有效机制。

  - **算法层（Attention Sink baseline）**：每个 query 关注初始 token (sink) + 最近 token。可视为 MoBA 特例：gating 固定选择首尾 blocks。缺陷：(a) 同样 task-specific；(b) 丢弃中间 token 可能包含关键检索信息；(c) "为什么初始 token 重要" 缺乏充分理论基础。

  - **算法层（线性注意力 baseline: Mamba/RWKV）**：将 Softmax attention 替换为线性近似 O_t = Q_t Σ_{i=1}^t K_i^T V_i。缺陷：(a) 与现有 Transformer 预训练模型不兼容——转换成本高（H. Liu et al. 2023）或需从头训练（A. Li et al. 2025）；(b) 复杂推理任务上的有效性缺乏充分证据。

  - **kernel调度层（FlashAttention baseline）**：tiled online softmax attention kernel，O(N²) FLOPs 但 O(N) memory。长 context 下仍受限于 O(N²) 计算量。1M prefill 需 30 分钟。
  
  - **系统框架层**：PyTorch + FlashAttention + HuggingFace Transformers。
  
  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Full Attention O(N²) 计算复杂度导致长 context 训练和推理成本 prohibitive
  2. 静态稀疏 attention（SWA/Attention Sink）task-specific，缺乏内容感知能力，无法自适应不同输入
  3. 线性注意力与现有 Transformer 生态不兼容，复杂推理能力未充分验证
  4. 现有动态稀疏方法（Quest, MInference）仅优化推理而非训练，无法降低长 context 训练成本

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoBA 将 MoE 的 "专家路由" 原理从 FFN 层迁移到 attention 层，通过 block-level top-k gating 实现内容感知的动态稀疏注意力，同时保持与 full attention 的参数等价性（0 参数增量），支持训练和推理阶段的无缝切换。

  MoBA 全栈执行例子（以 Llama-8B-1M-MoBA, 1M context, block size=4096, top-k=12 prefill）：

  - **算法层（核心创新——Block Partitioning + MoE-style Routing）**：
    1. **Block Partitioning**：将 1M context 划分为 n = 1M/4096 ≈ 244 个 block，每 block 4096 tokens。K, V 按 block 划分后 mean_pool 得到 block-level key representation K̄ ∈ R^{n×h×d}（每 block 的 4096 个 K vectors 的均值）。
    
    2. **Gating Score**：每个 query q 与 K̄ 中 n 个 block representation 做内积 s_i = ⟨q, mean_pool(K[I_i])⟩，得到 query-to-block affinity scores S ∈ R^{N×h×n}。计算量为 O(N·n·d) vs Full Attention O(N²·d)，n=N/B ≪ N。

    3. **Top-k Gating + Causality**：在 S 上施加 causal mask（future blocks = -∞）后取 top-k（k=12），每个 query 仅关注 (k+1) 个 blocks（k 个历史 + 1 个当前）。Sparsity = 1 - 4096×13/1M = 94.7%。

    4. **Hybrid Design**：MoBA 与 full attention 参数等价（无参数增减），支持：
       - 两阶段训练：90% tokens MoBA + 10% tokens Full Attention → 接近 full attention 的 loss
       - Layer-wise hybrid：最后 3 层 full attention + 其余 MoBA → SFT 性能显著恢复
       - 推理切换：prefill 用 MoBA（快速处理长 prompt），generation 用 full attention（保证生成质量）

    5. **Fine-Grained Block Segmentation**：类似 MoE 的 fine-grained expert segmentation，将 32K context 从 8 blocks 细分至 128 blocks（维持 sparsity 75%），性能提升 ~0.01 LM loss。

  - **kernel调度层（FlashAttention + MoE 融合）**：
    1. **Block-based query grouping**：根据 top-k gating 结果将 queries 按分配的 KV blocks 重排分组（类似 MoE 的 token dispatch）
    2. **Varlen FlashAttention**：对每个 (query_group, kv_block) 对使用 FlashAttention varlen 分别计算 block-wise attention
    3. **Online Softmax Combining**：将 self-attention output（当前 block）和 MoBA output（历史 blocks）用 online softmax tiling 合并
    4. **Tensor Parallelism for Extreme Length**：将 K/V broadcast 到不同 query heads 解决 10M context 显存限制
    
    Speedup: 1M context → 6.5× (vs FlashAttention), 10M context → 16× speedup. 复杂度 sub-quadratic.

  - **系统框架层**：
    基于 PyTorch + FlashAttention + DeepSpeed-MoE。MoBA layer 可直接替换标准 attention layer，无需修改模型其他部分。训练和推理使用同一套代码，MoBA/full attention 动态切换。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  对应解决 Baseline 缺陷的设计-缺陷映射：

  1. **O(N²)→O(k·B·N) → 解决缺陷 1（Full Attention 计算量）**：通过 block-level top-k routing，每个 query 仅关注 (k+1)B tokens 而非 N tokens。例如 1M context, k=12, B=4096 → 仅关注 ~53K tokens（5.3%），计算量降低 ~20×。

  2. **Content-Aware Gating → 解决缺陷 2（静态稀疏 task-specific）**：gating score s_i = ⟨q, mean_pool(K[I_i])⟩ 是 query-dependent 且 content-dependent 的——不同 query 根据自身语义选择不同的历史 blocks。SWA 和 Attention Sink 被证明是 MoBA 的特例（gating 固定选择最近/首尾 blocks），MoBA 表达力更强且可自适应学习。

  3. **参数等价 + 无缝切换 → 解决缺陷 3（线性注意力不兼容）**：MoBA 不改变 Transformer 架构，不引入或删除参数。现有预训练模型可无痛转换（全 attention→MoBA），训练中可动态切换。已部署于 Kimi 长 context 请求服务。

  4. **训练+推理双重加速 → 解决缺陷 4（仅推理优化）**：与 Quest/MInference 等仅推理优化的方法不同，MoBA 同时降低训练计算量——scaling law 实验（5 个模型规模, Chinchilla scaling）证明 MoBA 的训练 loss 与 full attention 高度一致（差值 < 1e-3），但训练 FLOPs 大幅降低。

  5. **"Less Structure" 原则**：MoBA 让模型自己学习 attention pattern，而非预设固定结构（SWA/Sink/Strided）。这符合论文的核心理念：attention 稀疏性应由数据驱动而非人工设计。

## Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

- baseline方法是什么？
  Baseline 是 ThinK [44] 结构化剪枝（per-channel structured pruning of KV cache）。ThinK 以整个 channel 为粒度剪枝 Key cache（per-channel, output-aware scoring），只剪枝 Key cache 无法有效剪枝 Value cache（Value cache 元素分布均匀，无显著 channel-wise outliers）。ThinK 报告的 Value cache 剪枝上限仅为 30% 稀疏度。此外，ThinK 结构化剪枝的稀疏 pattern 受限于 channel 对齐，导致大量冗余元素被迫保留，且 GPU 上无法将 channel-wise 剪枝直接转换为内存带宽节省（需要实际减少矩阵维度）。

  全栈执行例子（ThinK baseline, Llama-3-8B-Instruct, T=4096, RTX 6000 Ada）：

  - **算法层**：Key cache 剪枝：对每个 channel c，计算 S_c = Σ_{t} |Q_t| · |K[:, c]|（最近32 Q 的 L1 累加 × channel K），保留 top-k channels。Value cache 不剪枝（或仅 30% 稀疏度）。结构化剪枝后 Key cache 矩阵维度从 R^{T×d} 降为 R^{T×d'}（d' = d×(1-s)），仍为稠密矩阵——本质上仍是 dense 矩阵乘法。核心缺陷：(1) Value cache 几乎无法剪枝，KV cache 总体压缩率受限于 Key-only 剪枝；(2) channel-wise 剪枝忽略 token 内不同元素的差异，一个 channel 整体被保留/丢掉，粒度太粗；(3) 结构化稀疏的限制：即使剪枝 70% channels，dense 矩阵仍需整体加载进行计算。

  - **Serving/框架层**：使用 HuggingFace Transformers 推理。论文未修改 serving 框架调度。ThinK 在 attention 计算前对 KV cache 执行 channel-wise 选择，不影响计算 graph 其余部分。

  - **kernel调度层**：标准 PyTorch/cuBLAS batch GEMV 或 FlashAttention decode kernel。ThinK 减少 K 的 channel 维度后，QK^T 计算仍是标准 dense matmul（维度减小但仍是稠密计算），无自定义 kernel。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Value cache 剪枝困难——均匀分布无 channel outliers，结构化剪枝仅 30% 稀疏度上限
  2. Channel-wise 粒度过粗——整 channel 剪枝丢弃了 channel 内有价值的元素，同时被迫保留冗余元素
  3. 结构化稀疏无法直接转换为 GPU memory bandwidth 节省——剪枝后仍以 dense matmul 计算
  4. 需要 output-awareness 计算额外 pruning score 开销

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Mustafar 提出 per-token magnitude-based unstructured pruning + bitmap sparse format + custom CUDA SpMV kernel，实现端到端的非结构化 KV cache 剪枝加速。

  Mustafar 全栈执行例子（Llama-3-8B-Instruct, T=4096, K_s=0.5, V_s=0.5, RTX 6000 Ada）：

  - **算法层**：
    1. **Per-token magnitude-based pruning**：对每个 token 的 KV vector 独立按元素绝对值排序，保留 top-(1-s) 元素。Key cache 受益于 outlier channels（高 magnitude 元素集中在特定 channel），Value cache 虽分布均匀但 per-token 按 magnitude 剪枝等价于 per-token output-aware 剪枝（因 attention 中 V 每个元素乘以同一个 attention score）。
    2. **Key cache 结论**：无结构约束的非结构化剪枝在 70% 稀疏度下精度优于 ThinK 50% 结构化剪枝。output-awareness 带来微小提升但 magnitude-only 已足够。
    3. **Value cache 结论**：per-token magnitude-based 在 70% 稀疏度保持精度，远超 ThinK 的 30% 结构化上限。per-channel output-aware 可达到近似精度但需额外重算 attention scores（FlashAttention 不物化完整 attention matrix）。
    4. **Local dense window**：最近 32 token 保留稠密不剪枝，确保近期上下文的完整 attention 质量。
    5. **模块化兼容**：per-token 粒度允许与 token eviction (H2O) 无缝整合（evict token 后，剩余 token 各自独立剪枝）；与 KV cache quantization (KIVI) 叠加（先 prune 再 quantize）。

  - **kernel调度层**：
    1. **Triton 压缩 kernel**：GPU 并行将稀疏 KV cache 实时压缩为 bitmap-based 格式（每 1×64 tile 一个 bitmap + nonzeros）。
    2. **Custom CUDA SpMV kernel**：基于 Coruscant 的 bitmap sparse format，采用 FlashLLM 的 load-as-compressed, compute-as-dense 范式——在 GPU SM 上完成 compressed→register→shared memory decompress→Tensor Core dense compute pipeline。Memory-bound decode attention 中 global memory 数据搬移量大幅减少（仅加载非零元素 + bitmap + offset，而非 full dense tile）。
    3. **Decode attention 重新分拆**：SpMV 处理压缩历史 KV cache + dense MV 处理 local window，两部分结果 concat 后 softmax 再分别加权求和。
    4. **KV cache tile 管理**：Key cache column-tile 沿 token 维度，Value cache column-tile 沿 channel 维度；channel-major 遍历确保新 token 压缩数据可尾部追加。

  - **Serving/框架层**：基于 PyTorch + Triton + CUDA 实现，作为可插拔 attention 后端。Prefill 使用 FlashAttention，Decode 使用 Mustafar kernel。未修改 serving 调度逻辑。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → Mustafar 设计映射：
  1. **Value cache 无法剪枝（ThinK 仅 30%）→ Per-token unstructured pruning（70% 保持精度）**：非结构化剪枝解除 channel 对齐约束，允许在每个 token 内独立选择最佳元素保留。即使 Value cache 分布均匀，per-token magnitude 等价于 output-aware，因每个 value 元素在 attention output 中的贡献正比于其 magnitude。
  2. **Channel-wise 粒度过粗 → Element-wise per-token pruning**：以元素而非 channel 为剪枝单位，实现真正细粒度选择。实验证明在同等 70% 稀疏度下（减少 70% 元素），非结构化剪枝精度（LongBench avg 41.55-42.84）远超结构化剪枝（26.55-38.53）。
  3. **结构化稀疏无法转换为 bandwidth 节省 → Bitmap compressed format + custom SpMV kernel**：将无规则稀疏 pattern 压缩为 tile-wise bitmap 表示，SpMV kernel 以压缩格式加载、解压后 dense 计算，memory-bound attention 中 HBM 数据搬运量减少（50% sparsity: 65% compression ratio, 70% sparsity: 45% compression ratio）。
  4. **Output-awareness 计算开销 → Magnitude-only 剪枝避免额外计算**：Key cache 无需 output-awareness 即可达到 competitive 精度；Value cache per-token magnitude 天然等于 output-aware。避免了 ThinK 的 per-channel output-aware score 计算和 attention score 重算。
  5. **与正交方法兼容性差 → Per-token granularity 的模块化设计**：与 token eviction (H2O) 和 quantization (KIVI) 无缝叠加，允许不同程度的联合压缩（如 H2O 20% budget + Mustafar 50% sparsity）。
  6. **Batch=1 下 GPU 利用率不足**：论文明确指出的当前限制——小 batch 下 SpMV kernel 的 threadblock 数少于 SM 数量导致 SM underutilization。在 batch≥4 时性能优势显著（batch=8 时 2.23× throughput）。

## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- baseline方法是什么？
  Baseline 是 H2O（Heavy-Hitter Oracle）和 MSRNN 等基于 attention score 的 KV cache 淘汰方法。H2O 在 generation 阶段每步贪心淘汰 KV cache：对每个 token 计算 F_score = Σ_{all past tokens} Softmax(A[i, :])（累加全部历史 attention scores），保留 top-C 最高分 token + 最近 token。MSRNN 仅用当前 token 的 attention score 做淘汰。

  全栈执行例子（H2O baseline, LLaMA2-7B-Chat, 4K context, 单 A100 80GB）：

  - **算法层**：输入 x_prompt ∈ R^{4096×4096}。encoding 阶段正常 prefill 存储完整 KV cache。generation 阶段每生成一个 token，计算当前 token 对所有 cache 中 key 的 attention scores，累加到历史 accumulated attention scores，按总分排序淘汰低分 token。核心缺陷：
    1. **Attention bias problem**：attention scores 高度集中在初始 token 和最近 token，中间 token 即使关键（如 passkey）也因低 attention 被 H2O 淘汰（Fig. 2）。
    2. **Step-by-step 贪心淘汰**：每步基于局部信息做淘汰决策，无法全局优化；时间复杂度 O(p+T) per token。
    3. **冗余信息干扰**：H2O 累加全部 token 的 attention scores，大量无关 token 的 scores 引入噪声，稀释了真正重要的 task-specific scores。
    4. **Perplexity 不可靠**：H2O 用 PPL 作为主要指标，但在 long-text 实际任务（如 LongBench passkey retrieval）中 PPL 表现好但任务准确率差。

  - **kernel调度层**：使用标准 FlashAttention-2 计算 attention，KV cache 淘汰操作为纯 PyTorch tensor indexing。无自定义 kernel。

  - **Serving/框架层**：论文未明确说明 serving 框架。淘汰逻辑在 HuggingFace Transformers 推理 pipeline 中以 Python hook 实现。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Attention bias（H2O 偏向初始/最近 token，MSRNN 仅看当前 token）→ 中间关键 token 被误淘汰
  2. Step-by-step 贪心淘汰 O(p+T) → 长 context 下淘汰本身成为瓶颈
  3. 全量 attention score 累加引入冗余信息 → 评分不精准
  4. PPL 评估不反映真实 long-text 任务性能 → 方法效果被高估
  5. 确定性淘汰缺乏鲁棒性 → 一旦关键 token 被淘汰无法恢复

- 论文方法是什么？如何对应解决Baseline的缺陷？
  NACL 提出混合 KV cache 淘汰框架：PROXY-TOKENS EVICTION（基于 proxy tokens 的全局最优淘汰）+ RANDOM EVICTION（per-head 概率采样淘汰），在 encoding 阶段一次性完成淘汰。

  NACL 全栈执行例子（LLaMA2-7B-Chat, 4K context, C=20%, C_p=6%, C_r=12%, 单 A100 80GB）：

  - **算法层**：
    1. **Encoding phase one-eviction**：将淘汰从 generation 阶段移至 encoding 阶段一次性完成。encoding 阶段计算完整 attention matrix A ∈ R^{p×p}，利用全局信息做最优淘汰 S_encoding = F_score(A, C)，随后 compressed KV cache 用于全部 generation steps。时间复杂度从 O(p+T) 降至 O(1)（T ≪ p）。
    2. **PROXY-TOKENS EVICTION**：选取输入末尾 ~10% token 作为 proxy tokens P（对应用户问题部分），F_score = Σ_{x_p∈P} Softmax(A[x_p, :])，仅聚合 proxy tokens 的 attention 信号。proxy tokens 天然携带 task-specific 信息，其 attention pattern 更精准地反映哪些 token 对任务关键。淘汰建模为组合优化：S_t = argmax_{S⊂R} Σ_{x∈S} F_score(A, C_p) ∪ P（proxy tokens 默认保留）。
    3. **RANDOM EVICTION**：将 F_score 经 Softmax 归一化得到概率分布 P_prompt，从该分布中采样 C_r 个 token 保留。每个 head 使用不同 seed → head-wise 多样化采样。在 LLaMA-7B 32层×32头、budget=20% 下，token 在至少一个 head 中保留概率为 1-(C_h)^l，即使 C=1% 也 >99.99%。
    4. **Hybrid budget allocation**：C = C_p + C_r，典型比例 20% total = 6% proxy eviction + 12% random + 2% protect proxy（Tab. 4）。

  - **kernel调度层**：实现 Reduce Attention Scores CUDA kernel 兼容 FlashAttention-2。利用 FlashAttention-2 forward 输出的 log-sum-exp 重算 attention scores 并做 column-wise reduce（Algorithm 2）。或仅对 proxy tokens（~10%）重算 attention scores，开销可忽略。128K context 下 evict 20% 维持 ~15GB 稳定显存。

  - **Serving/框架层**：论文未明确说明 serving 框架。NACL 作为 pluggable eviction policy 可在 HuggingFace Transformers 中以 hook 形式实现。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → NACL 设计映射：
  1. **Attention bias（H2O 偏向初始/最近）→ PROXY-TOKENS EVICTION 用 task-specific proxy tokens**：H2O 累加全部 token attention（含大量非任务相关的 attention scores），导致评分偏向位置而非语义重要性。NACL 仅用末尾 ~10% 的 proxy tokens（用户问题）评分，这些 token 的 attention pattern 反映"哪些 prefix token 对回答当前问题有用"，从而更精准定位关键信息。实验证据：NACL 在 passkey retrieval（PR-Zh/PR-En）上显著优于 H2O（30% budget: NACL 6.8/9.0 vs H2O 3.7/5.0），证明 proxy tokens 能保留位于中间的 passkey 而 H2O 的 attention bias 将其淘汰。
  2. **Step-by-step 贪心 O(p+T) → Encoding phase one-eviction O(1)**：H2O 每 generation step 做一次淘汰，长 context 下淘汰开销占总推理时间可观。NACL 在 encoding 阶段利用完整 attention matrix 一次性全局优化淘汰，generation 阶段仅每 m 步做轻量淘汰。消融：移除 global eviction 导致 short-text -1.3%、long-text -1.5%。
  3. **全量 attention 累加引入噪声 → Proxy tokens 子集精准评分**：H2O 的 F_score = Σ_{all} Softmax(A[i,:]) 中大量 non-task-related token 的 attention 贡献噪音。NACL 的 F_score = Σ_{P} Softmax(A[x_p,:]) 仅聚合 proxy tokens。消融：移除 PROXY-TOKENS EVICTION 后 short-text -28.1%、long-text -6.0%，证明该策略是最核心贡献。极端情况：0% proxy budget = MSRNN（仅当前 token），100% = H2O（全量 token），~10% 最优。
  4. **确定性淘汰无鲁棒性 → RANDOM EVICTION head-wise 多样化采样**：H2O 的确定性 top-K 一旦丢弃关键 token 无法恢复。NACL 的 head-wise probability sampling 确保每个 token 在多个 head 的 KV cache 中有独立被保留的机会。消融：移除 RANDOM EVICTION 后 short-text -1.2%、long-text -9.2%（long-text 下随机性更重要！）。Uniform sampling 替代 attention-weighted sampling 后 long-text -1.1%。证明随机性 + attention 引导的组合最优。
  5. **PPL 不可靠 → LongBench + lm-eval-harness 真实任务评估**：论文重新评估了 H2O、MSRNN、Attention Sink 在 short-text（7 任务，5-shot/25-shot）和 long-text（LongBench 7 任务，budget 10%/20%/30%）上的真实表现，揭示 PPL 与实际任务准确率的系统性偏差。

  **关键性能**：
  - NACL 20% short-text avg 63.8 vs Full 64.6 (-0.8) vs H2O 60.3 (-4.3) — 80% improvement over H2O
  - NACL 20% long-text avg 30.8 vs Full 31.5 (-0.7) vs H2O 28.6 (-2.9) — 76% improvement over H2O
  - KV cache up to 5× reduction with >95% performance maintenance
  - LLaMA2-7B, batch=4, 32K seq: 64GB → NACL 20% ≈ 12.8GB

## PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- baseline方法是什么？
  Baseline 是传统后训练 KV Cache 量化方法（KIVI、RotateKV、MiKV、SKVQ 等），这些方法为短上下文场景（<8K tokens）设计，直接应用于长 CoT LLM 会导致严重性能退化。

  **Baseline 全栈执行例子（以 KIVI 为例，DeepSeek-R1-Distill-Qwen-7B，2-bit KV Cache）：**

  - 模型推理算法层：每步解码时直接用非对称分组量化将 KV Cache 压缩到 2-bit。Key Cache 使用 per-channel 量化，Value Cache 使用 per-token group-wise 量化（group size=128），保留首 token 和最近 token 为高精度。标定数据使用短序列（512 tokens），通道重参数化因子 λ_i 在短序列上标定，无法捕获 RoPE 低频通道（周期 > 54000 tokens）在长上下文下的完整数据分布。
  - 系统框架层：论文未明确说明（使用标准 HuggingFace Transformers 推理流程，未修改推理引擎或 Serving 框架）。
  - 编译框架层：论文未明确说明（未涉及编译器修改）。
  - kernel调度层：论文未明确说明（fake quantization 实验，非真实部署 kernel）。
  - 硬件架构层：论文未明确说明（纯软件层量化，不涉及硬件修改）。

  Baseline 的两个核心痛点：
  1. **大累积误差**：每步直接量化到目标 2-bit，内存预算未被充分利用（前期存在大量空闲内存），但量化误差随生成长度线性累积，长 CoT 场景（最大 32K tokens）下精度严重退化。
  2. **短标定数据无法反映长上下文分布**：RoPE 低频通道周期长达 54K tokens，512-token 标定无法覆盖这些通道在长序列下的正弦分布特征，导致通道重参数化因子不准确。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PM-KVQ 提出三项技术分别解决 Baseline 的两个缺陷。

  **PM-KVQ 全栈执行例子（DeepSeek-R1-Distill-Qwen-7B，Fbit=2-bit）：**

  - 模型推理算法层：
    1. **Progressive Quantization（解决累积误差）**：不直接量化到 2-bit，而是按 16→8→4→2 bit 逐步降低。初期以 16-bit 高精度存储，当内存预算耗尽时执行 Equivalent Right Shift：X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b。该策略等价于先反量化到浮点再重新量化，在长 CoT 前期（内存未满时）保持零量化误差，后期再有损压缩早期 token。
    2. **Block-wise Memory Allocation（解决内存利用不均）**：不采用统一位宽，而根据各 block 对量化的敏感度（一阶泰勒近似 s_{i,b}）分配不同位宽，建模为整数规划问题并用 CVXPY 求解（几秒内完成）。敏感 block（深层 + 第一层）分配高位宽，不敏感 block 分配低位宽，在相同总内存预算下最大化精度。
    3. **Calibration with Positional Interpolation（解决短标定问题）**：在 RoPE 中对位置索引 m 乘以缩放因子 s（实验中 s=4），使 2048-token 标定数据携带 8192-token 的位置信息，覆盖 RoPE 低频通道的更完整周期分布，从而准确标定通道重参数化因子 λ_i。

  - 系统框架层：论文未明确说明（使用 HuggingFace Transformers + fake quantization 进行评测，未修改推理引擎）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明（为 fake quantization 实验，但论文指出实际推理时 Equivalent Right Shift 可通过整数加法和移位高效实现，无需浮点反量化）。
  - 硬件架构层：论文未明确说明。

  **关键性能对比（Baseline vs PM-KVQ）：**
  - DeepSeek-Qwen-7B (2-bit) AIME-2024 pass@1：KIVI 32.08% → PM-KVQ 40.00%（+7.92%）
  - DeepSeek-LLaMA-8B (4-bit) AIME-2024 pass@1：KIVI 41.25% → PM-KVQ (BS=6, block-wise) 47.71%（+6.46%，超 16-bit 的 44.17%）
  - DeepSeek-LLaMA-70B (2-bit) AIME-2024 pass@1：KIVI 51.88% → PM-KVQ 64.79%（+12.91%）
  - Voting accuracy 提升更显著：DeepSeek-Qwen-7B voting KIVI 43.33% → PM-KVQ (BS=32) 66.67%（+23.34%）

  **设计思路映射（缺陷→方法）：**
  - 累积误差大 → Progressive Quantization：用时间换精度，前期高精度存储，内存满后再逐渐降位宽
  - 内存仍浪费（块间敏感度不均） → Block-wise Memory Allocation：敏感块多分配内存，不敏感块少分配
  - 短标定不能反映长上下文 RoPE 分布 → Positional Interpolation in Calibration：在短序列 RoPE 中嵌入长位置信息

## PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- baseline方法是什么？
  Baseline 是所有层使用固定相同 KV cache size 的压缩方法：H2O（基于累积 attention score 动态淘汰，保留 recent + heavy hitter tokens）、SnapKV（基于 instruction token attention score 选择/clustering 重要 KV positions）、StreamingLLM（仅保留初始 sink tokens + 局部 window）。核心缺陷：(a) **跨层 uniform cache budget**：所有 baseline 对每一层分配相同 KV cache size，忽略底层 attention 分散（需要更多 cache 覆盖全局信息）和高层 attention 集中（少量 cache 即可）的差异；(b) **底层信息丢失**：底层 uniform small budget 下，分散的 attention 中许多关键 token 被错误淘汰，高层虽然 attention 已集中但 uniform budget 仍保留大量不重要 token，资源浪费。

  全栈执行例子（H2O baseline, LLaMa-3-8B-Instruct, 8K context, A100, KV size=64）：
  - **算法层**：prefill 时对所有 32 layers 每层每 head 保留最后 α 个 instruction token + 选出 top-64 个 heavy hitter tokens（基于累积 attention score）。底层 layer 0 仅保留 64 tokens（占 8K 的 0.8%），大量分散在全局的 attention 信息被丢弃。顶层 layer 31 同样保留 64 tokens，但此时 attention 已集中在极少数关键 token 上，保留的 64 tokens 中包含许多不必要的 token。总计 KV cache memory = 64 × 32 layers × 2 (K,V) × d_model × 2 bytes(fp16)。
  - **系统框架层**：HuggingFace Transformers 推理 pipeline 或 vLLM paged attention。KV cache 为 contiguous tensor，每层 uniform eviction。与 FlashAttention-2 兼容。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention 计算。Eviction 操作（sort, top-k, gather）在 GPU 上通过 PyTorch 算子执行（torch.topk, torch.gather），无自定义 kernel。
  - **硬件架构层**：NVIDIA A100 GPU，fp16 精度。8K context 下 Full KV cache 占用 ~6848M，64 KV size 下仅 ~428M（6.3%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PyramidKV 通过观察 LLM 中跨层的"Pyramidal Information Funneling"注意力模式（底层→均匀分布/广播模式，中层→局部聚拢，顶层→massive attention 集中于少量关键 token），提出跨层不均匀 KV cache budget 分配和 attention score-based token 选择。

  **(1) 算法层——跨层 Pyramid-Shaped Budget Allocation 解决 uniform budget 缺陷**：
  核心思想：底层 attention 分散 → 分配更多 cache budget；顶层 attention 集中 → 分配更少 cache budget，形成金字塔形分配。

  Budget 分配公式（arithmetic sequence）：
  ```
  k^{m-1} = k^{total} / (β·m)         # 顶层（最少）
  k^0 = 2·k^{total}/m - k^{m-1}       # 底层（最多）
  k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l  # 中间层 linear decay
  ```
  超参 β=20 控制顶层陡峭程度，α=8 为各层固定保留的 instruction token 数。**Token 选择**：每层每 head 保留 instruction tokens + top-(k^l - α) 个按 attention score（来自 instruction tokens 的 attention sum）排序的最高分 token。

  **对比 Baseline 的改进**：
  - vs H2O/SnapKV/StreamingLLM (uniform budget)：PyramidKV 底层多分配 cache（layer 0: ~2× 平均 budget），充分保留分散注意力所需的信息；顶层少分配 cache（layer 31: ~0.1× 平均 budget），仅保留 massive attention 集中的关键 token。LongBench avg: KV size=64 时 PyramidKV 34.76 vs H2O 33.89 vs SnapKV 33.05 (LLaMa-3-8B)；KV size=128 时 PyramidKV 37.25 vs H2O 35.37 vs SnapKV 35.50。
  - 极端压缩下优势更显著：TREC 任务 KV size=64 时 PyramidKV 58.00 vs H2O 38.00/SnapKV 38.50（+20.5/19.5 pp）。
  - Needle-in-a-Haystack：LLaMa-3-70B KV size=128 时 PyramidKV 100.0 Acc = FullKV 100.0，vs H2O 82.3 / SnapKV 98.6。

  **(2) Arithmetic Sequence 选择优于其他衰退策略**：
  论文 ablation 比较了线性(arithmetic)、几何(geometric)、指数(exponential)衰退以及自适应分配策略（entropy-based, Gini coefficient-based）。线性策略 LongBench avg=34.76 优于几何(34.36)和指数(34.23)，且远优于 entropy(32.71)和 Gini(32.58)。论文认为线性衰退与观察到的注意力模式的自然渐进收窄更吻合，计算开销最小（budget 一次性预计算，无在线 entropy/Gini 计算开销）。

  **(3) vLLM 集成——Per-Layer Block Table 解决 cache fragmentation**：
  Naive vLLM 实现中不同层不同 budget 导致小 chunk 内存分配/释放/移动/访问的碎片化和低效。PyramidKV 将每个 sequence 的 block table 扩展为 per-layer block table，使得每层独立 page-out KV cache，避免固定内存偏移限制。Throughput 显示 compression 下相对 throughput 随 input context length 增加而降低（因新 sequence 需等 decoding batch 加入），需进一步优化。

  全栈执行例子（PyramidKV on LLaMa-3-8B-Instruct, 8K context, A100, KV size=64 avg）：
  - **算法层**：(a) 预计算 32 layers 的 budget：底层 layer 0 ~100 tokens，layer 31 ~10 tokens（instruction tokens 不算在内）；(b) Prefill 阶段计算 attention scores，各层选 top-k^l tokens + 8 个 instruction tokens；(c) torch.gather 执行非 in-place eviction，释放原 tensor。总计 KV cache memory 与 baseline uniform 64 相同（平均 budget 相等），但分配更匹配信息流。
  - **系统框架层**：HuggingFace Transformers 即插即用（无需 training/fine-tuning）。论文开源实现：https://github.com/Zefan-Cai/PyramidKV。vLLM 集成通过 per-layer block table 实现。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 标准 attention。Eviction 使用 torch.gather（非 in-place，需额外临时 tensor），allocation time < 0.000006s（占总 inference time 可忽略），selection time ~0.013s。PyramidKV 延迟与 H2O/SnapKV/StreamingLLM 可比（e.g., prompt 4096 + gen 4096: PyramidKV 138.87s vs H2O 139.87s vs SnapKV 138.57s）。
  - **硬件架构层**：NVIDIA A100 GPU，fp16。KV cache size=2048 时 memory=1712M（25% of Full 6848M），performance match/exceed FullKV（LongBench avg 41.49 vs FullKV 41.46 on LLaMa-3-8B）。

## Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

- baseline方法是什么？
  Baseline 方法包括：(1) **StreamingLLM**：始终保留前几个 token（attention sink）和最近 n 个 token 的滑动窗口，丢弃中间 token；(2) **K-Norm**（Devoto et al., 2024）：基于 Key 的 L2 范数评估 KV pair 重要性，保留低范数的 KV pairs；(3) **SnapKV**：利用 prompt 末尾部分的注意力分数选择重要 KV pairs，需物化注意力矩阵，因此与 FlashAttention 不兼容。
  
  全栈执行例子（以 K-Norm 为代表性 baseline）：
  - **算法层**：计算每个 Key 向量的 L2 范数 $||K_t^h||_2$，保留范数最小的 KV pairs。这个启发式基于经验观察（低范数 Key 对应高平均注意力），但忽略了 Key 向量在 Query 主方向上的角度分量，近似精度有限。
  - **系统框架层**：使用 HuggingFace Transformers + KVPress 库，在 prefill 完成后或生成过程中对 KV Cache 进行压缩。论文未明确说明 Serving 框架层面的修改。
  - **编译框架/算子层**：不涉及编译框架修改。K-Norm 的范数计算需要在每次推理时显式计算，每次需要 $O(L \times d_H)$ 的浮点操作，与 Q-Filters 的标量积计算复杂度相当。论文未明确说明 kernel 层面的修改。
  - **硬件架构/芯片设计层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Q-Filters**，通过分析 Query-Key 几何特性，发现 Query 分布具有各向异性（anisotropic），存在一个单一主方向 $u^h$（由 Query 矩阵 SVD 的第一右奇异向量给出），Key 在该方向上的投影可以精确近似期望注意力分数。与 baseline 的对比：
  
  | 缺陷 | Q-Filters 解决方案 |
  |------|-------------------|
  | K-Norm 仅用 L2 范数忽略角度信息 | Q-Filters 同时捕捉 Key 在 Query 主方向上的投影（含范数和角度），与注意力分数的 Spearman 相关性显著高于 K-Norm |
  | SnapKV 需物化注意力矩阵，与 FlashAttention 不兼容 | Q-Filters 仅需一次标量积投影，不访问注意力权重，完全兼容 FlashAttention |
  | StreamingLLM 固定保留 attention sink + 滑动窗口，丢弃中间关键信息 | Q-Filters 基于数据驱动的每头重要性估计，动态选择全局最重要的 KV pairs |
  | 许多方法需要微调 | Q-Filters 完全训练无关，仅需一次离线 SVD 校准（<3 分钟） |

  全栈执行例子（Q-Filters 对比 K-Norm baseline）：
  - **算法层**：(a) 离线从校准数据收集各层各头的 Query 激活矩阵 $Q^h$，SVD 分解得 Q-Filter $v_1^+$；(b) 推理时计算 $s_t^h = \langle K_t^h, v_1^+ \rangle$ 作为重要性得分，取 top-k 保留。定理保证 $\mathbb{E}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $\kappa^h > 0$ 为常数。对 GQA，组内 Q-Filters 取平均。这比 K-Norm 多捕捉了 Key 在 Query 主方向上的投影角度分量 $\cos(K_j^h, u^h)$。
  - **系统框架层**：基于 KVPress 库实现，作为 KV Cache 压缩的 plugin 插入 HuggingFace 推理 pipeline。Q-Filters 校准只需前向传播若干样本提取 Query 激活（无需反向传播），推理时在每次 KV Cache 更新后执行 top-k 筛选。论文未明确说明 Serving 框架层面的进一步修改。
  - **编译框架/算子层**：Q-Filters 的标量积计算 $K \cdot v_1^+$ 与 K-Norm 的 L2 范数计算复杂度相当（均为 $O(L \times d_H)$），但 Q-Filters 避免了 FlashAttention 之外显式物化注意力矩阵的需求。FlashAttention 兼容性意味着 kernel 执行路径更短：prefill 用 FlashAttention 高效计算 attention，压缩仅需额外的矩阵-向量乘法和 top-k，不破坏 FlashAttention 的内存优化。论文未明确说明编译框架或 kernel 层面的修改。
  - **硬件架构/芯片设计层**：论文未明确说明。

## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- baseline方法是什么？
  Baseline 包含两类：(A) Full Attention（FlashInfer 实现）：decode 阶段每步加载完整 KV cache，对 Llama2-7B @ 32K context，KV cache 16GB，内存加载占 decode 阶段 53% 以上时间；(B) KV Cache 驱逐算法：H2O（基于累积历史 attention score 裁减 token）、TOVA（基于当前 query 的 attention score 决定丢弃哪些 token）、StreamingLLM（仅保留 attention sink + 滑动窗口）。核心缺陷：

  | 缺陷 | 具体表现 |
  |------|----------|
  | Full Attention 内存瓶颈 | 长 context 下 KV cache 加载量随 seq_len 线性增长，decode 阶段严重 memory-bound，32K context 下 KV 加载耗时 >50% |
  | 历史信息驱逐不可逆 | H2O/TOVA 丢弃的 token 可能对未来 query 关键（如 passkey 在 question 之前），导致 passkey retrieval 准确率近乎 0%（Tab. 1, 10K/100K tests） |
  | 静态窗口无法覆盖长依赖 | StreamingLLM 仅关注最近 window，passkey 在 window 之外的 100K 测试完全失败 |
  | Query-agnostic 假设错误 | H2O 假设历史 attention score 高的 token 对将来也关键，但 Fig. 2 证明同一 token 对不同 query 的关键性差异巨大 |

  全栈执行例子（Full Attention baseline, Llama2-7B, 32K context, RTX 4090）：
  - **算法层**：输入 prompt tokens 已编码为 KV cache（32 layers × 32 heads × 32768 tokens × 128 dim × FP16 = 16GB）。每步 decode：取最后一个 token 的 Q（1×128），加载全部 32768 个 K 向量，计算 S = QK^T/√128 ∈ R^{1×32768}，softmax 后乘全部 V 向量得 O ∈ R^{1×128}。attention 计算量 O(seq_len × d_head) = O(32768×128) ≈ 4.2M FLOPs，但 KV cache 加载量 2 × 32768 × 128 × 2 bytes ≈ 16.8MB/head/layer（memory-bound）。
  - **系统框架层**：FlashInfer 作为 attention kernel 库。单 batch decode，所有 KV cache 驻留 GPU HBM。FlashInfer 使用 FlashAttention 的 tiling 策略（分 tile 加载 K,V 到 SRAM 做 online softmax rescaling），但 tile 数量随 seq_len 线性增长。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashInfer CUDA kernels。Decode 阶段：加载 Q（registers），循环加载 K/V tiles 从 HBM → SRAM（每个 tile B_r × d_head），compute QK^T（Tensor Cores），online softmax update（CUDA Cores），accumulate P×V（Tensor Cores），write O 到 HBM。Memory-bound：arithmetic intensity ≈ 4.2M FLOPs / (2×32768×128×2 bytes) ≈ 0.25 FLOPs/byte，远低于 RTX 4090 的 ~200 FLOPs/byte 拐点。
  - **硬件架构层**：NVIDIA RTX 4090（24GB GDDR6X, 1.0 TB/s memory bandwidth, 82.6 TFLOPS FP16）。Decode 阶段仅利用 ~2.5% peak FLOPs（因 memory-bound）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Quest 提出 **query-aware KV cache sparsity**：不丢弃任何 KV cache token，而是在每步 decode 时基于当前 Query 动态估计 token 关键性，仅加载 Top-K 关键 page 的 K、V 参与 attention。核心设计：

  **(1) Page 粒度元数据（1/PageSize 存储开销）**：对每 page（默认 16 tokens），维护 per-channel Key 向量的最小值 m_i 和最大值 M_i。元数据大小 = 2 × d_head / page_size × KV cache size ≈ 12.5% KV cache for page_size=16。插入新 token 时 M_i = max(M_i, k_i), m_i = min(m_i, k_i)，O(d_head) overhead。

  **(2) Query-Aware Upper-Bound Criticality Estimation**：给定当前 Query Q，每 page 的关键性上界 s_p = Σ_{i=1}^{d} max(Q_i · m_i^p, Q_i · M_i^p)。该上界保证 s_p ≥ Q_i · K_i^{(t)} 对 page 内任意 token t 成立，因此选 score 最高的 K 个 page 不会遗漏高 attention 的 token。这与 query-agnostic 方法（H2O/TOVA）根本不同——H2O 基于历史 attention 裁减可能错误丢弃对未来 query 关键的 token。

  **(3) Two-Stage Attention Execution（内存减载的核心）**：
  - Stage 1：加载 metadata（2 × num_pages × d_head，而非完整 KV cache），计算 per-page criticality scores → Top-K page indices
  - Stage 2：仅加载 Top-K pages 的完整 K、V 到 SRAM → 标准 FlashAttention
  - 内存加载减少：1/PageSize + K/PageNum of total KV cache。如 page_size=16, 64K context (4096 pages), K=256 → 加载量仅为完整 KV cache 的 ~12.5%，约 8× 减少。

  **(4) 前两层豁免**：观察到前两层 attention sparsity < 10%（Fig. 3），对前两层保持 full attention，其余层使用 Quest。

  全栈执行例子（Quest, Llama2-7B, 32K context, 2048 token budget, RTX 4090）：
  - **算法层**：与 baseline 相同的 KV cache 存储（不丢弃任何 token）。每步 decode：(a) 加载 page metadata（2 × 2048 pages × 128 dim × FP16 ≈ 1MB，vs baseline 16MB K cache）→ 计算 per-page criticality upper-bound scores；(b) Top-K 选 128 pages（2048 tokens = token budget）；(c) 加载 128 pages × 16 tokens × 128 dim × FP16 ≈ 512KB K + 512KB V → FlashAttention → 输出 O。总内存加载 ~2MB vs baseline ~32MB/layer，16× 减少。
  - **系统框架层**：基于 FlashInfer 实现。PageAttention 兼容性使 sparse page loading 可直接通过 FlashInfer page table indirection 实现，无需额外数据重组。单 batch decode 评估。论文未修改 serving 框架的多请求调度逻辑。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：三个 CUDA kernel 在 FlashInfer 中实现：(a) Criticality estimation kernel —— element-wise max + reduce-sum, memory-bound on metadata；(b) RAFT batched Top-K —— 5-10 µs latency, compute-bound；(c) Approximate attention —— FlashInfer PageAttention with sparse page indices。32K seq_len, 2048 budget → self-attention 7.03× speedup（Fig. 9）。
  - **硬件架构层**：NVIDIA RTX 4090，无专用硬件修改。利用现有 GPU 的 HBM 带宽和 Tensor Cores。

  效果量化：
  - Passkey Retrieval (10K)：Quest 64-token budget 达 100%（Tab. 1），H2O 256-budget 仅 1%
  - Passkey Retrieval (100K)：Quest 1024-token budget 达 100%，H2O 4096-budget 仅 4%
  - LongBench 六数据集：Quest 1K budget 达 full cache 可比性能，H2O/TOVA/StreamingLLM 即使在更大 budget 下仍有明显差距
  - Self-attention speedup：7.03× @ 32K, 2048 budget vs FlashInfer（Fig. 9）
  - End-to-end speedup：2.23× @ 32K, 2048 budget, 4-bit weight quantization（Fig. 10）
  - 同等精度约束下：Quest 比 TOVA 减少 2.6-7.7× latency（Fig. 11b）

## R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

- baseline方法是什么？
  Baseline 是标准 attention-based KV cache eviction 方法，以 SnapKV 为代表。SnapKV 的 token 选择完全依赖 attention scores：计算最后 α 个 observation tokens 对 key tokens 的注意力，通过滑动窗口 max-pooling 稳定化后取平均作为 per-token importance score，保留 importance 最高的 B_budget 个 token。核心缺陷：推理模型（如 DeepSeek-R1）的长 CoT 生成中含有大量冗余内容——反复的自我验证、迭代推理、冗长的自言自语。这些重复内容"self-attend 到自己"，产生高 attention score，导致 SnapKV 保留大量语义冗余 token，挤占了真正关键的推理中间步骤。论文观察表明：推理模型生成长度是 ground truth 的 8-14×，1-gram 和 2-gram 重复频率是 ground truth 的 5-7×（Fig. 2）。在 SnapKV 的 selected token 可视化中（Fig. 3），大量被选中的 token 集中在对同一结论的反复重述（如 "10% of 30 is 3. So 3 students are leaving early" 被重复数十次并被高 attention 选中）。

  全栈执行例子（SnapKV baseline, DeepSeek-R1-Distill-Llama-8B, 16K generation, A100 80G）：
  - **算法层**：每 B_buffer=128 步触发压缩。选取最后 α=8 个 tokens 作为 Q_obs，对 GQA 的各 query head 分别计算 attention A^{h,g} = softmax(Q^{h,g}·(K^h)^T/√d)，对所有 query head 做 mean-pooling 聚合得到 final attention matrix。滑动窗口 max-pooling 后取均值得到 I_i^h = (1/α)·Σ_{j=0}^{α-1} max(A_{j,i-W:h,i+W})。选择 I_i^h 最高的 B_budget 个 token。问题：如果一个冗余 token 反复出现，同一个 key vector 会被周围的 query token 频繁 attend（因为语义高度相似），导致 I_i^h 膨胀，over-retain 冗余内容。
  - **系统框架层**：未修改 serving 框架，直接在 HuggingFace Transformers 的 forward pass 中插入 KV cache selection 逻辑。压缩后内存需要重新分配（移动压缩后的 KV cache），可能引入内存管理 overhead。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch matmul + attention kernel。压缩操作（top-k 选择）为 PyTorch CPU/GPU 操作，非定制 CUDA kernel。SnapKV 与 R-KV 在 kernel 层等价——差异仅在 selection score 的计算方式。
  - **硬件架构层**：NVIDIA A100 80G。无专用硬件修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  R-KV 在 attention-based importance scoring 的基础上额外引入 **redundancy estimation via cosine similarity of key vectors**，通过 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性（I）和去冗余性（R），从根源解决 SnapKV 的冗余过保留问题。

  **(1) 算法层——冗余感知的 Joint Selection Score**：
  R-KV 的三阶段计算：

  **阶段 A：Importance Scoring**（继承并改进 SnapKV）
  - 对 GQA 使用 max-pooling 替代 mean-pooling 聚合 query head 的 attention（实现细节，详见 Appendix A.2）。Max-pooling 能更好地保留每个 query head 中最重要的 token，避免 mean-pooling 抹平关键信号。
  - 滑动窗口 max-pooling 稳定化 + 取均值得到 I_i^h（与 SnapKV 相同）。
  - 这一阶段确保 critical reasoning context（如问题中的数值、关键中间步骤）被保留。

  **阶段 B：Redundancy Estimation via Key Vector Cosine Similarity**
  - 这是 R-KV 的核心创新。对每层每 head，将 key vectors L2 归一化后计算余弦相似度矩阵 S = K̄ K̄^T ∈ R^{n×n}（Eq. 5）。
  - 对角线置零（防止 self-redundancy）；对每个 token i，找到 S_{:,i} > T 的高相似 token 集合，保留其中最近的 β 个（largest indices），将其 similarity 置零——这确保即使 token 高度重复，最近出现的那几个仍被保留（因为它们接近当前解码位置，contextual relevance 更高）。
  - 计算每 token 的平均相似度 S̄_i = mean(S_{:,i})，再通过 softmax 归一化得到 R_i^h ∈ [0,1]（Eq. 6）。高 R_i^h 表示 token i 的 key vector 与许多其他 token 高度相似 → 冗余。

  核心洞察：冗余 token 的 key vectors 在向量空间中高度聚集。通过余弦相似度矩阵 S，R-KV 在向量空间层面（而非 token 表面）捕捉语义冗余。这解决了 SnapKV 仅看 attention weight（标量）无法区分的"高 attention 但高度冗余"的 token：这些 token 的 I_i^h 高但 R_i^h 也高，joint score Z_i^h = λ·I_i^h − (1−λ)·R_i^h 被拉低，从而在 top-B_budget 选择中自然被淘汰。

  **阶段 C：Joint Selection 与跨 Head 聚合**
  - Z_i^h = λ·I_i^h − (1−λ)·R_i^h（Eq. 7），λ=0.1（通过消融确定，Fig. 5-6）。
  - 跨 head 聚合：AggScore_k = mean_h(Z_{k,h})，取 top-B_budget 保留。
  - 为何 λ 偏小（0.1）？因为 I_i^h 的值分布高度稀疏（少数 outlier 主导），而 R_i^h 经过 softmax 后分布较均匀。λ=0.1 时 redundancy 项的权重(1−λ)=0.9 足以有效抑制冗余，λ 增大到 0.5 以上则退化为近似纯 attention-based selection。

  **(2) 对比 Baseline 的改进**：

  | 维度 | SnapKV（baseline） | R-KV（论文方法） |
  |------|-------------------|-----------------|
  | Token 选择信号 | 仅 attention weight（I_i^h） | Joint: λ·I_i^h − (1−λ)·R_i^h |
  | 冗余检测 | 无。冗余 token 常因高 attention 被误保留 | Key vector 余弦相似度显式测量冗余 |
  | GQA 聚合 | Mean-pooling of attention scores | Max-pooling（更好保留每 query head 关键 token）|
  | 近期 token 保护 | 仅保留最后 α 个 observation tokens | observation tokens + 显式保留最近 β 个高相似 token |
  | 推理模型适配 | 未针对长 CoT 冗余特性优化 | 专门针对推理模型的重复/反射模式设计 |
  | AIME24 (R1-Llama-8B, 10% budget) | ~20% pass@1 vs FullKV 49.79% | ~51.56% pass@1（lossless @1536 budget）|

  **(3) 全栈执行例子（R-KV, DeepSeek-R1-Distill-Llama-8B, 16K generation, A100 80G）**：
  - **算法层**：每 128 tokens 触发压缩（B_buffer=128, B_budget=1536, α=8, λ=0.1）。Importance scoring：对 32 heads × 8 attention groups (GQA group_size=4)，每个 GQA group 内 4 个 query heads 独立计算 attention → max-pooling 聚合 → sliding window max-pooling → 得到 I_i^h ∈ R^{1536+128-8=1656}。Redundancy estimation：对同一批 1656 个 key vectors L2 归一化 → 1656×1656 余弦相似度矩阵 → 抑制对角线和最近 β 个高相似 token → 行均值 → softmax 归一化 → R_i^h。Joint: Z_i^h = 0.1·I_i^h − 0.9·R_i^h → mean 跨 head 聚合 → 取 top 1536 个 token + 8 observation tokens = 1544 KV tokens 保留。相比 FullKV 的 16000 KV tokens，压缩比 ~10%，节省 ~90% KV cache 内存。
  - **系统框架层**：未修改 serving 框架。论文在 Limitations（Appendix D）中明确指出现有 serving 框架（如 vLLM）若不提供 KV cache compression 专用接口，则需要 reallocate 内存来存放压缩后的 cache 并 deallocate 原始 cache，这会引入额外开销。论文表示通过 dedicated KV compression interfaces 可以避免此问题。目前实现为 HuggingFace 级别的 prototype。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch 操作。重要性计算 overhead O(α·B_budget)（~8·1536 量级），冗余估计 overhead O(B_budget²)（~1536² 量级），总计约 ~2.4M FLOPs per compression step（每 128 tokens 一次），远小于 attention 计算量。无定制 CUDA kernel。
  - **硬件架构层**：NVIDIA A100 80G。R-KV 通过压缩 KV cache 释放 memory 来增加 batch size（因为 batch size 受 KV cache 内存限制），从 batch=30 (FullKV @16K) 提升至 batch=402 (R-KV @16K, fixed budget=1024)，端到端 throughput 从 347 tok/s 提升至 3189 tok/s（9.2×）。单独 batch=1 时 throughput 提升有限（80.95 vs 69.41 tok/s），证明 primary gain 来自 batch size scaling 而非 per-step latency reduction。

  效果量化（R1-Llama-8B）：
  - MATH-500: 34% KV cache budget → lossless（82.34% vs FullKV 82.38%）, 16% budget → 105% of FullKV
  - AIME24: 10% KV cache budget → lossless（51.56% vs FullKV 49.79%, pass@1）, 16% budget → 52.29%（超越 FullKV）
  - 8K generation, 10% ratio budget: 90% memory saving, 479 vs 62 max batch size, 6.6× throughput
  - 16K generation, fixed 1024 budget: 93.75% memory saving, 402 vs 30 max batch size, 9.2× throughput

## SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

- baseline方法是什么？
  Baseline 是标准的 Full Attention（dense decoding），即在长序列推理模型（如 Qwen3, DeepSeek-R1-Distill）的自回归解码阶段，每个新 token 都需要与完整的 KV cache 计算 attention。全栈执行例子（沿一个 decode token 的路径）：

  算法层：Qwen3/DeepSeek-R1 模型使用标准 Multi-Head Attention with GQA。每个 decode step，Q（单 token, multi-head）与完整 K, V（seq_len 长度）计算 scaled dot-product attention，再 softmax，输出 O。复杂度 O(seq_len × d_head) per token，KV cache 大小 O(seq_len × num_kv_heads × d_head)。

  系统框架层：论文未明确说明（baseline 使用标准 PyTorch/HuggingFace Transformers 推理，未涉及 vLLM/SGLang 等 serving 框架的具体配置）。

  编译框架层：论文未明确说明。

  Kernel调度层：FlashAttention-3 (FA3) 的 flash decoding kernel。Kernel 采用 GQA-aware 的 split-KV 策略：沿 (batch, heads_kv, num_splits) 三维 grid launch，每个 SM 负责部分 KV sequence 的 attention 计算，最后 reduce partial results。此为 I/O-bound kernel。

  硬件架构层：NVIDIA H100 GPU，利用 Tensor Core（wgmma 指令）加速矩阵乘法。HBM 带宽为瓶颈。

  Baseline 的核心痛点：
  1. 长推理场景（AIME 平均 11k-18k tokens，max 32k），decode 阶段每 token attention 计算量与 seq_len 成正比，总体生成成本 O(n²)。
  2. 现有训练无关（training-free）稀疏方法（如 Quest）在大 block size（≥64）时准确率显著下降，无法利用粗粒度稀疏块带来的硬件效率优势。
  3. 训练无关方法依赖人为启发式规则，缺乏对 attention 稀疏模式的精确学习。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SeerAttention-R 提出基于自蒸馏 Attention Gate (AttnGate) 的 post-training 块稀疏注意力框架。核心思路：在原始模型注意力层插入轻量级 AttnGate module（可学习 gate），通过自蒸馏训练让 AttnGate 预测哪些 KV blocks 对当前 query 最重要，推理时只计算被选中的 blocks。全栈执行例子（沿一个 decode token 的路径）：

  算法层：
  - AttnGate: Q 分支通过线性层将 GQA group 内 query heads 聚合为 KV-head 数，K 分支通过 Max/Min/Avg 三种 pooling 压缩序列维度，再经线性层和 RoPE 后计算块级 attention 分数 S = softmax(Q_gate @ K_gate^T / sqrt(d_gate))。
  - Shared sparsity: 同一 GQA group 内所有 query heads 共享相同的 sparsity 选择（与 NSA、SAAP 一致），提升硬件效率。
  - 训练（自蒸馏）：只训练 AttnGate 参数（冻结原始模型权重）。修改版 FA2 kernel 同时生成 full attention output 和 1D column-wise maxpooled ground truth，用 KL divergence 训练 AttnGate。仅需 0.4B tokens（OpenR1-MATH-220K），训练效率极高。

  系统框架层：K Compression Cache：为 AttnGate 的 K 分支维护压缩后 K 表示的 cache，每次生成 block_size 个新 token 才更新一次。block_size=64 时 K Compression Cache 仅占原始 KV cache 的 1/128 (<1%)。推理时仅需加载被选中 KV blocks，可结合 KV cache offloading（将完整 KV cache 放 CPU，按需 fetch 选中 blocks 到 GPU）。

  编译框架层：论文未明确说明。

  Kernel调度层：Block Sparse Flash Decoding Kernel（TileLang + Triton 实现）。3D grid launch (batch, heads_kv, num_splits)，仅遍历 selected_block_indices。num_splits 按 max_selected_blocks 分割而非 total_blocks，解决 sparsity 带来的 SM 负载不均衡问题。TileLang 自动应用 tiling、warp specialization、pipelining、tensorization 等优化。在 H100 上，bs=16, seqlen=128k, 90% sparsity 时达 8.6× 加速 vs FA3。

  硬件架构层：NVIDIA H100 GPU，利用 wgmma 指令和 TileLang 自动优化。论文未涉及 RTL 或芯片级修改。

  Baseline 缺陷 → 方法对应的具体设计选择：

  | Baseline 缺陷 | SeerAttention-R 设计 |
  |---|---|
  | Training-free 方法在大 block size 下准确率崩溃 | 自蒸馏 AttnGate 学习精确的 block-level 稀疏模式，block_size=64/128 仍保持 near-lossless 准确率 |
  | 每 token 需计算完整 attention (O(n)) | K Compression Cache + AttnGate 预测选中的 blocks，只计算 O(k) attention（k=token budget） |
  | Training-free 方法无法利用 GQA 共享 sparsity | AttnGate Q 分支聚合 GQA group 内 heads，实现 group 内共享 sparsity |
  | 长序列 KV cache 内存压力大 | K Compression Cache 仅占 <1% KV cache，支持 KV cache offloading |

  Oracle sparsity 实验验证：Qwen3-14B 在 block_size=64、2k token budget 时即可达到 near-lossless 准确率，说明 attention 本身具有内在稀疏性。

## SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching

- baseline方法是什么？
  Baseline 方法主要包括两类：**(a) Token-level eviction 方法**（如 H2O、SnapKV）在 prefilling 阶段基于累积注意力分数永久驱逐 token，忽略了 token 在后续解码步骤中的动态重要性变化，导致被驱逐的关键 token 无法恢复；(b) **固定大小 chunk 检索方法**（如 Quest、ShadowKV）将 KV cache 按固定长度分块，基于当前 query 检索相关 chunk 从 CPU 加载到 GPU，但固定大小分块会打破语义边界，割裂完整的思维单元。全栈执行示例：基线方法（以 Quest 为例）在 prefilling 阶段将 32k token 输入按固定 chunk size=16 切分 → 每个 chunk 内保留 attention 分数最高的若干 token 的 KV 对 → offload 所有 chunk 的 KV 到 CPU → decoding 阶段每个 step 用当前 token 的 query 与各 chunk 的 key 计算相关性 → 仅加载 top-k 个 chunk 的 KV 到 GPU 做 sparse attention → 输出 next token。缺点：chunk size 对性能高度敏感（chunk size 从 32 降到 16，NIAH 准确率从 48.3% 跃升至 96.1%），且固定 chunk 无法感知语义边界。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SentenceKV 用**句子级语义分组 + 多 query 聚合检索**替代 token 级驱逐和固定 size chunking。(1) 按标点符号将输入文本分成句子桶，每个桶是一个完整的语义单元，避免了固定大小 chunking 对语义连贯性的破坏。(2) 在 prefilling 阶段用 observation window 计算每个 token 的重要性分数，保留 top ⌊r·τ⌋ 个 token（r>1，保留超过最终预算的 token 以维持语义丰富性），计算每个句子桶的平均 key 向量 $\bar{k}_{s,h}$ 作为 GPU 上的紧凑语义表示。(3) 在 decoding 阶段维护 sentence cache $Q_s$ 累积当前生成句子的所有 query，以平均 query $\bar{q}$（而非单 token query）与存储的句子桶语义向量做内积相似度排序，按相似度从高到低检索句子桶中的 KV 对直到填满 token budget τ。全栈执行示例：32k token 输入 → prefilling 阶段按标点分句（约数百个句子桶，每个 25-30 token）→ observation window（最后 32 token）对前 31968 个 token 计算跨所有 head 的累积注意力分数 → 选 top 2048 (r=2, τ=1024) 个 token → 对每个句子桶计算 $\bar{k}_{s,h}$ 存 GPU（tens of MB）→ 2048 个 token 的 KV 对 offload 到 CPU → decoding 阶段：每生成一个 token 将其 query 加入 $Q_s$ → 若遇到句子边界，计算 $\bar{q} = mean(Q_s)$ → CPU 端对 $\bar{q}$ 和所有 $\bar{k}_{s,h}$ 做内积排序 → 按相似度依次选句子桶直到累计 1024 token → 从 CPU 加载对应 KV 对到 GPU → 做 1024-token 的 sparse attention → 继续生成。直接解决了：(a) token 动态重要性：不永久驱逐，decoding 阶段按语义动态检索；(b) 固定 chunk 语义割裂：以完整句子为检索单元，保持语义连贯性。效果：NIAH 检索准确率 97.5%（vs SnapKV 78.2%），256k 上下文 GPU 内存 52.71GB（vs Full KV 89.71GB），延迟稳定在 17.8ms（vs Full KV 84.9ms）。

## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- baseline方法是什么？
  Baseline 方法包括：(a) **Full Attention**：完整 KV cache 保留在 GPU 显存，对每 token 做完整 attention。优点是精度无损，缺点是 KV cache 显存占用随序列长度线性增长（128K context × batch=8 时 KV cache ~800MB/layer），导致 batch size 受严重限制（60K→max batch 8，122K→max batch 4，244K→max batch 2），超出则 OOM。全栈执行示例：用户输入 prompt 128K tokens → prefill 阶段每层计算 QKV 投影 → FlashAttention 计算完整 attention → 所有层的完整 KV cache 驻留 GPU HBM → decoding 阶段每步从 HBM 读取完整 KV cache → Batch MatMul Q·K^T → Softmax → MatMul A·V → 输出投影 → next token。瓶颈：KV cache 占 GPU 显存 >80%，限制 batch size，吞吐受显存而非计算能力约束。(b) **Dynamic Sparse Attention（Quest/Loki/InfiniGen）**：保留完整 KV cache 但仅对选中的稀疏 KV 对做 attention。Quest 用 chunk-level min-max 近似选择 KV pages，Loki 用 PCA 在低维空间计算注意力分数，InfiniGen 用离线 SVD 投影做 KV 选择并做 CPU offload。全栈执行示例（以 Quest 为例）：128K token 输入 → prefill 阶段完整 FlashAttention 计算并保存完整 KV cache 在 GPU → decoding 阶段每个 query 与每个 chunk（size=16）的 min-key/max-key 做内积估算 attention 上界 → 选 top-k pages → 仅对选中的 KV 做 sparse attention。缺陷：(1) 未减少 GPU 显存占用，KV cache 仍全量存储在 GPU，batch size 无法扩容；(2) 若 offload 到 CPU（InfiniGen），需 fetch 完整 KV 对（key + value），PCIe 传输量大且 KV selection 不精确导致精度下降；(3) 未利用 key cache 的低秩特性压缩存储。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ShadowKV 通过三个核心观察驱动设计，分别解决 baseline 的显存、延迟和精度缺陷：
  
  **(1) 低秩 Key + CPU Offload 解决显存瓶颈（Observation 3.1）**：发现 pre-RoPE key cache 极低秩（rank 160 可达 6× 压缩无精度损失），且同序列内的低秩子空间高度共享但跨序列不同——因此对每序列在线做 prompt-dependent SVD 而非离线 data-independent 投影。Prefilling 阶段对 pre-RoPE K 做截断 SVD，仅保留 A∈R^{s×r} 和 B∈R^{h_kv×r×d} 在 GPU（大小从 s×h_kv×d 降至 sr + h_kv×r×d），value cache 不低秩因此直接 offload 到 CPU。与 data-independent 方法（如 Palu 训练投影矩阵）不同，在线 SVD 自适应每个 prompt 的低秩结构。
  
  **(2) Landmark 近似 + Outlier 静态缓存解决精度（Observation 3.2）**：发现 post-RoPE keys 具有空间局部性（chunk 内 cosine similarity 高），可用 chunk 均值作为 landmark 来准确近似注意力选择；同时仅 0.2-0.3% 的 chunk 是 outlier（cosine similarity 显著低），将这些 outlier 的完整 KV 对静态存储在 GPU 上。Decoding 时，用 Q 与 landmarks L 做 MatMul 近似 chunk 级 attention score → ArgTopK 选 k 个 chunk → 仅对选中的 (K+k) × C 个 token 做真实 attention。与 Quest 的 min-max 近似相比，landmark 近似更准确，允许更低的 sparse budget（1.56% vs 6.25%）同时保持精度。
  
  **(3) Multi-Stream Overlap + Cache Mechanism 解决延迟（Section 4.2）**：由于仅需从 CPU fetch value cache（key 可在 GPU 端从低秩投影重建），PCIe 传输量减半；同时 CUDA multi-stream 将 key 低秩重建（GPU compute）与 value CPU fetch（PCIe）并发执行，总延迟 ≈ max(t_recon, t_fetch) 而非两者之和。此外，KV cache temporal locality（相邻 step 命中率 ~60%）通过 cache mechanism 以 index scan 跳过已命中 chunk 的重复操作，减少 60% 重建和传输。
  
  全栈执行示例（128K context, batch=24, Llama-3.1-8B, A100）：
  - **Prefilling**：128K tokens → embedding + QKV 投影 → 对 K_pre-RoPE 做在线 SVD（rank=160，耗时仅 attention 的 3-5%）→ 存 A(128K×160)、B(8×160×128) 在 GPU → K_post-RoPE 分 chunk_size=8 → 16K chunks → 每 chunk 算均值作为 L → cosine similarity 检测出 48 个 outlier chunks → K_outlier、V_outlier 存 GPU → 其余 V offload 到 CPU → 完整 FlashAttention prefill (论文保留 exact prefilling)
  - **Decoding step**：Q(24×32×1×128) × L^T(8×16K×128) → softmax → 聚合 → TopK 选 256 chunks → cache mechanism 对比上一步 indices 发现 60% 命中 → 仅重建 miss 的 ~102 chunks → Stream1 GPU 端 A[I_miss](102×160) × B(8×160×128) → RoPE → K_sparse；Stream2 CPU→GPU cudaMemcpyAsync V[I_miss] → 总延迟 ~1.84ms (overlap后) vs 无 overlap 的 ~3ms → K=[K_outlier(48×8); K_sparse(102×8); K_new(1)] ≈ 1201 ~ 2432 tokens；FlashAttention → FFN → 输出 → next token 的 K 投影到同低秩空间保存
  
  **对比 baseline 的解决效果**：
  - 显存：KV cache GPU 占用降 6-7× → batch size 从 4 扩至 24 (6×) at 122K context
  - 吞吐：Llama-3.1-8B 从 80.78 tok/s (batch=4) 升至 245.90 tok/s (batch=24)，3.04× 加速，超过无限显存理论吞吐 134.30 tok/s
  - 精度：RULER 128K 平均 83.57 vs Full Attention 85.53（仅降 2%），远超 Quest 的 35.52、Loki 的 35.52、InfiniGen 的 59.27
  - PCIe 效率：仅 fetch value（vs InfiniGen fetch KV 对），理论等效带宽 7.2 TB/s = 3.6× A100 原生带宽

## SnapKV: LLM Knows What You are Looking for Before Generation

- baseline方法是什么？
  Baseline 是全量 KV cache 方法（Full KV），即在生成阶段保留 prompt 的完整 KV cache，每步解码需对所有 prefix tokens 计算 attention。同时论文将 H2O（Heavy-Hitter Oracle）作为对比 baseline。

  **Full KV / H2O 在模型推理全栈的执行例子（以 Mistral-7B-Instruct-v0.2，16K prompt tokens，A100-80GB 为例）**：

  - **算法层**：标准 causal attention 计算，每步解码对全部 L_prompt 个 KV pairs 做 Q·K^T → softmax → weighted sum V，时间复杂度 O(L_prompt·D) per token per layer。H2O 在解码阶段根据累积 attention scores 贪婪淘汰低分 KV pairs，但仅在解码阶段新生成的 KV 上做压缩，不压缩 prompt KV cache。

  - **系统框架层**：HuggingFace Transformers 推理 pipeline，使用 `model.generate()` 进行自回归解码。Prompt 编码后 KV cache 驻留 GPU 显存。H2O 在每步解码后按累积 attention score 更新 KV cache 的淘汰策略。

  - **kernel调度层**：标准 PyTorch matmul + attention kernel。H2O 额外执行 TopK + index gather 操作，但无需 custom CUDA kernel。论文未明确说明使用 FlashAttention 的具体版本。

  - **硬件架构层**：NVIDIA A100-80GB GPU（HBM 80GB，带宽 2TB/s）。16K prompt → 32 层 × 32 heads × 128 head_dim × 16K × 2(K+V) × 2 bytes(FP16) ≈ 1GB KV cache，随 prompt 线性增长。原生实现在 16K tokens、batch=2 时 OOM；batch=1 时解码延迟 > 100ms/token。

  **Baseline 缺陷**：
  1. Full KV：解码延迟随 prompt 长度线性增长（每步计算 Q·K^T 的复杂度与 L_prompt 成正比），KV cache 显存占用随 prompt 长度线性增长，导致 OOM 和吞吐下降。
  2. H2O：仅在解码阶段压缩新生成的 KV pair，不压缩 prompt KV cache（prompt 通常是 KV cache 的主要瓶颈）；依赖累积 attention scores 做重要性评估，缺乏对 prompt 内信息完整性的保持。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SnapKV 是一项无需微调的 KV cache 压缩技术，核心理念：**LLM 在生成之前就已经知道哪些 prompt tokens 对其回答至关重要**。通过在 prompt 末尾设置一个 "observation window"，利用该窗口内 queries 对 prefix keys 的注意力权重进行投票，选出每个 attention head 最重要的 KV 位置，并通过 1D pooling 聚类保留周围上下文，实现 prompt KV cache 压缩。

  **两大关键发现驱动方法设计**：
  1. **生成前可识别注意力模式**：prompt 最后一个 window 的注意力分配模式与生成阶段高度一致（Fig. 2）
  2. **注意力模式在生成中保持一致**：生成过程中不同 window 选出的重要特征高度重叠（Fig. 3）

  **SnapKV 在全栈的执行例子（以 Mistral-7B-Instruct-v0.2，16K prompt tokens，max_capacity=2048，A100-80GB 为例）**：

  - **算法层**：
    1. Prefill 阶段正常计算 QKV 投影
    2. 取 Q 的最后 L_obs=32 个 token（observation window，包含 prompt 末尾的指令/问题）
    3. 计算 observation window queries 对所有 prefix（前 L_prompt - L_obs 个）keys 的 softmax-normalized attention weights：W_obs ∈ R^{32×32×(L_prompt-32)}
    4. 沿 query 维度求和得到投票分数：C_h = Σ_{i=0}^{L_obs} W_obs[:, i, :] → C ∈ R^{32×(L_prompt-32)}
    5. 1D max pooling（kernel_size=7）聚合邻域信息：pool_vote = pool1d(C, kernel_size=7, padding=3, stride=1)
    6. 每个 head 独立 TopK 选择 k=2048-32=2016 个最重要 prefix 位置
    7. 压缩 prefix KV + 完整 observation window KV → 恒定 2048 个 KV pairs
    8. 解码阶段仅在这 2048 个 KV 上计算 attention，复杂度恒定 O(2048·D) per token

  - **系统框架层**：HuggingFace Transformers + 少量 monkey-patch 代码修改（替换 attention forward），无侵入式集成。Prompt 编码后执行一次 KV 压缩，后续生成直接使用压缩后 KV cache。与 Medusa 并行解码框架兼容：压缩 KV 后 draft head 和验证阶段均使用压缩 cache，解耦了解码复杂度与 prompt 长度。

  - **kernel调度层**：标准 PyTorch matmul + attention kernel，无需 custom CUDA kernel。额外操作为 TopK + index gather（PyTorch 原生操作，开销可忽略）。Pooling 使用 `torch.nn.functional.max_pool1d` 或 `avg_pool1d`。

  - **硬件架构层**：NVIDIA A100-80GB GPU。SnapKV 解码阶段 KV cache 从 1GB（16K tokens）降至固定 128MB（2048 tokens）→ 解码延迟从 >100ms 降至 <40ms（3.6× speedup），同一 GPU 可处理的序列长度从 16K 扩展到 131K（8.2× memory efficiency）。压力测试：LWM-Text-Chat-1M + SnapKV，单 A100 可处理 380K context tokens（380× 压缩比），准确检索 needle。

  **对比 baseline 的解决效果**：
  | 指标 | Full KV | H2O (4096) | SnapKV (2048) |
  |------|---------|------------|---------------|
  | LongBench avg (Mistral) | baseline | 显著下降 | 与 Full KV 持平 |
  | 解码延迟 @16K, batch=2 | >100ms/tok | 论文未明确说明 | <40ms/tok |
  | 最大 batch-2 序列长度 | 16K (OOM) | 论文未明确说明 | 131K |
  | NIAH @380K (LWM) | OOM @33K | 论文未明确说明 | 准确检索至 140K |

  **方法优势的本质**：
  1. **Observation window voting → 解决 prompt KV 压缩问题**：H2O 等只在解码阶段压缩，SnapKV 通过 prompt 末尾窗口投票，在生成前即完成 prompt KV 压缩，直接解决长 prompt 的内存和时间瓶颈。
  2. **Pooling 聚类 → 保持上下文完整性**：仅选 top attention 位置会导致信息断裂（如电话号码只取国家代码），1D pooling 通过平滑邻域保留 token 周围的上下文，保证 induction heads 能正确 copy 完整信息串。
  3. **Context-aware 动态选择 → 而非静态策略**：不同指令对同一文档的注意力模式不同（Fig. 4），SnapKV 的 observation window 机制能根据具体 query 动态调整选择，优于固定保留策略（如 StreamLLM 的 attention sink + recent window）。

## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- baseline方法是什么？
  长上下文 LLM 推理中，baseline 是 Full Attention（保留完整 KV cache 在 GPU 显存中进行标准 attention 计算）。全栈执行过程：

  **算法pipeline**：输入 prompt tokens → Token Embedding → 逐层 Transformer Block。每层：X → QKV Projection (W_qkv) → Q, K, V。K 经 RoPE 旋转位置编码。QK^T 计算 attention scores（O(n²)），Softmax 归一化，P×V 加权求和得到 attention output。残差连接后经 FFN。输出 hidden states 送入下一层。最后一层输出经 LM Head → logits → 采样下一个 token。

  **系统框架**：vLLM 系统框架，使用 PagedAttention 管理 KV cache 内存。Prefill 阶段：所有 prompt tokens 并行计算，KV cache 写入 GPU HBM。Decoding 阶段：每次生成一个 token，新 token 的 Q 与完整 KV cache 计算 attention，新 KV 对追加写入 GPU HBM。Batch 请求共享 GPU，Continuous Batching 动态调度。

  **编译框架**：论文未明确说明。PyTorch eager mode 或 torch.compile 自动图捕获。

  **kernel调度**：FlashAttention (v2) CUDA kernel，基于 tiling 和 online softmax，利用 GPU shared memory 减少 HBM 访问，将 attention 计算的内存复杂度从 O(n²) 降至 O(n)（IO 层面）。QKV Projection 使用 cuBLAS GEMM 在 Tensor Core 上执行。FFN 使用 cuBLAS GEMM。

  **硬件架构**：NVIDIA A100 GPU (80GB HBM2e)。Tensor Core 用于 GEMM 计算（312 TFLOPS FP16），HBM2e 带宽 2 TB/s。CPU-GPU 通过 PCIe 4.0 x16 连接（31.5 GB/s）。GPU 上 KV cache 占用 2×b×h_kv×s×d×sizeof(dtype) bytes。以 Llama-3.1-8B 在 128K 上下文为例：s=128K, h_kv=8, d=128, dtype=BF16 → 单层 KV cache = 2×8×128K×128×2 = 512 MB，32 层总计约 16 GB（单 batch），batch=4 即 64 GB，batch=8 即 128 GB 超出 80 GB 显存。

  Baseline 痛点：
  1. GPU 显存瓶颈：KV cache 随序列长度线性增长，长上下文（128K-1M）下即使小 batch size 也会 OOM
  2. 解码延迟高：每步需对所有 s 个 token 计算 attention，计算量 O(s×d)，访存量 O(s×d)
  3. Batch size 受限：小 batch 导致 GPU 计算资源利用率低，吞吐低下
  4. CPU offloading 朴素方案延迟大：将完整 KV cache 移至 CPU 并每次取回稀疏 KV 对，PCIe 传输成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ShadowKV 提出 GPU-CPU 异构 KV cache 存储 + 准确的稀疏 attention 选择策略，全栈执行过程：

  **算法pipeline**（核心创新）：
  - **发现**：pre-RoPE key cache 具有极低秩特性（奇异值衰减最快），同一序列内 key 的低秩子空间高度共享（内序列相似度高），不同序列间低秩子空间不同（跨序列相似度低）。因此 online SVD 比 data-independent weight decomposition 更精确。
  - **低秩 Key 存储**：pre-filling 时对 pre-RoPE K ∈ R^{s×d} 执行 SVD 保留 rank r=160 的截断分解：A ∈ R^{s×r}, B ∈ R^{h_kv×r×d}。K 通过 K ≈ A @ B（忽略奇异值对角矩阵）重建。压缩比 = d/r ≈ 128/160 不合适？实际上低秩分解将存储从 s×d 降至 s×r + h_kv×r×d。以 Llama-3.1-8B (d=128, r=160, h_kv=8) 为例：原始存储 = s×128，低秩存储 = s×160 + 8×160×128 = 160s + 163840。对 s=128K：原始 = 16.4M floats，低秩 = 20.5M + 0.16M（实际上论文称约 6× 压缩，涉及 batch 累计效应和 landmarks 替代完整 K cache 存储）。
  - **Landmark Approximate Attention**：将 post-RoPE K 分为 c=8 token 的 chunk，每 chunk 均值作为 landmark L。解码时 Q×L^T（O(n_c×d) 替代 O(s×d)）近似注意力选择 top-k chunk。
  - **Outlier 缓存**：检测 chunk 内 cosine similarity 最低的 o=48 chunk（0.3%）作为 outlier，完整保留其 KV 对在 GPU。
  - **Temporal Locality Cache**：利用相邻解码步 attention 模式高重复性（>60% chunk hit rate），跳过已缓存 chunk 的取回和重建。

  **系统框架**（Serving调度）：
  - GPU 存储：低秩 key 投影 A/B、landmarks L、outlier KV 对（总计 ~1/6 原始 KV cache 大小）
  - CPU 存储：完整 value cache V_CPU（pinned memory，快速 H2D 传输）
  - 解码调度：Q → Landmark Attention（选择 top-k chunk）→ 并行执行 key 重建（GPU GEMM）与 value 取回（PCIe H2D）→ Sparse FlashAttention → 输出
  - Temporal cache 减少 60% 重复操作

  **编译框架**：论文未明确说明。

  **kernel调度**（解决解码延迟问题）：
  - 自定义 CUDA kernel：注意力近似融合 kernel（GEMM+Softmax+TopK）、低秩 key 重建 GEMM、异步 value 取回
  - CUDA multi-stream overlap：Stream 1 执行 K_sparse = A_selected @ B（GPU Tensor Core），Stream 2 执行 V_sparse = cudaMemcpy(V_CPU[I], H2D)（PCIe）。两者通过 CUDA event 同步，net latency = max(compute, transfer) 而非 sum
  - 理论等效带宽 7.2 TB/s = 3.6× A100 原生带宽，超越假设无限显存条件下的吞吐

  **硬件架构**：NVIDIA A100 GPU + x86 CPU，PCIe 4.0 x16。无硬件修改。

  Baseline 缺陷 → ShadowKV 解决方案对照：
  1. **显存瓶颈** → 低秩 key + CPU value offloading，GPU KV cache 占用降至 1/6，batch size 提升 6×
  2. **解码延迟高** → Landmark 近似 attention 降至 O(s/c×d)，仅对 1.56% 的 sparse chunk 做精确 attention
  3. **Batch 受限** → 显存节省后 batch 从 2-8 升至 12-48，吞吐提升 2.23-3.04×
  4. **CPU offloading 延迟大** → 仅取回 value（非完整 KV），与 key 重建重叠执行，等效带宽 7.2 TB/s

## TokenButler: Token Importance is Predictable

- baseline方法是什么？
  Baseline 方法分为三类：(1) 静态策略：StreamingLLM 用 recency-based sliding window + attention sinks 固定 KV-cache budget，完全缺乏 query-awareness；(2) 自适应驱逐：H2O 累积注意力分数永久驱逐低分 token，SnapKV 在固定窗口上池化注意力分数驱逐 token，一旦 token 被驱逐就永久丢失，在 co-referential 场景（如对话中早先提到的实体被后续引用）中失败；(3) 自适应动态策略：Quest 保留完整 KV-cache 但以 page 粒度（chunk）选择性加载 token，用 query 与 page 内 min-max token 幅值的点积作为 page 重要性代理。page 粒度过粗，当关键 token 跨越 page 边界时会丢失部分信息，在 context-dense 任务中精度不足。TokenSelect 保留全量 token 并用 Q·K 点积选择重要 token，但需在完整嵌入维度 E 上计算，开销高。
  
  全栈执行例子（以 H2O / SnapKV 为代表的自适应驱逐 Baseline）：
  - 算法层：在每个 decode step，对每个 head 计算 query 与 KV-cache 中所有 key 的注意力分数，累积近期窗口内的分数作为 "长期重要性" 指标，对低分 token 永久驱逐出 KV-cache。使用 GQA (Grouped Query Attention) 时由 KV head 共享。
  - 系统/Serving 层：论文未明确说明具体 Serving 框架修改。驱逐逻辑通常在模型 forward 内部实现，拦截 attention 计算后的注意力权重，排序并裁剪 KV-cache。
  - 编译框架层：论文未明确说明。
  - Kernel 调度层：驱逐后的 KV-cache 变为非连续存储（碎片化），需要 gather/scatter 操作或重新打包。H2O/SnapKV 使用标准 FlashAttention kernel，sink token + 保留 token 可能不连续。
  - 硬件架构层：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TokenButler 训练一个轻量级预测器（二层 MLP，<1% LLM 参数量）通过 attention distillation 学习预测 token 重要性。核心设计：(1) 在固定深度间隔 G 的 producer layer 处从 hidden states 预测低维 importance queries（d'=16），与学习投影后的真实 KV-cache keys（同样 d' 维）做点积得到细粒度 token 重要性分数，替代 H2O/SnapKV 的启发式注意力累积；(2) 不驱逐任何 token，保留完整 KV-cache，而是每步动态选择 top-k token——解决 H2O/SnapKV 永久驱逐导致 co-referential retrieval 失败的缺陷；(3) 逐 token 粒度选择（非 page 粒度）——解决 Quest 跨 page 丢失 token 的缺陷；(4) prediction interval + neighbor fetching 摊销预测器开销——每 N 步运行一次预测器并通过空间邻居覆盖重要性漂移，解决 TokenSelect 每步计算高维 Q·K 点积的高开销问题。

  全栈执行例子（TokenButler 方法）：
  - 算法层：Producer layer（每 G=4 层一个）的 hidden states H ∈ R^{B×L×E} 经 LayerNorm + 二层 MLP（hidden=512）预测 slot-specific importance queries Q_imp ∈ R^{(B·H)×G×L×d'}（d'=16）。对每个 consumer layer l，其真实 key cache K ∈ R^{B×H_kv×L×D} 通过学习投影矩阵 W_K^{(l)} ∈ R^{D×d'} 降维为 K_proj ∈ R^{B×H_kv×L×d'}。计算 scores = Q_imp[slot] · K_proj^T，取 top-B token 组成 Important Buffer。最终 attention 输入 = [Sink(128) | Important(B) | Local_Window(256)]，调用标准 FlashAttention。训练目标：minimize CE(softmax(A_teacher_masked), softmax(A_pred_masked))，teacher 为冻结 LLM 每层的 masked causal attention logits。训练数据 1K seq len，预测器通过 key-cache 投影自动泛化到 64K。
  - 系统/Serving 层：KV-cache 组织为三个连续 buffer（Sink / Important / Local Window），避免碎片化。延迟投影：新 token 在 Local Window 中停留 N 步后才批量投影 key（利用 HBM 带宽）。Prediction interval i=N：预测器每 N 步运行一次，中间 N-1 步复用上次选择。Neighbor fetching 基于聚类感知算法扩展选中 token 的空间邻居，2B 个唯一位置。集成 TokenSelect 代码库进行端到端 throughput 评测。
  - 编译框架层：论文未明确说明。
  - Kernel 调度层：Attention kernel 为标准 FlashAttention，输入为三 buffer 拼接。Importance Score Computation（低维 Q·K 点积，d'=16）随 context 增长但远小于原始 attention（D=128）。Timing breakdown 显示 Attention Kernel 耗时恒定（因 sparse budget 固定），Importance Score Computation 随 context 线性增长但斜率低（低维运算）。KV gather 耗时与 budget B 成正比。CPU offloading 场景（>=256K context）：仅传输 sparse 选中的 KV pairs 从 CPU 到 GPU，减少数据传输量 8×，latency 从 Dense 的 3.2s/token 降至 0.6s/token (7.6×)。
  - 硬件架构层：论文未明确说明。

  Baseline 缺陷 → TokenButler 解决方案对照：
  1. **静态策略无 query-awareness** → 学习预测器从 hidden states 动态预测每个 query 的 token 重要性
  2. **驱逐策略永久丢失 token** → 保留完整 KV-cache，仅选择性访问，co-referential 场景 near-oracle 精度
  3. **Page 粒度过粗（跨 page token 丢失）** → 逐 token 细粒度选择，synthetic co-reference coverage 84-95% vs Quest 19-58%
  4. **启发式重要性指标不准确** → attention distillation 直接学习真实 attention distribution，Recall@50% 达 67-81%
  5. **每步预测开销高** → prediction interval (up to 16× amortization) + neighbor fetching，精度仅降 1.1%

## Trainable_Dynamic_Mask_Sparse_Attention

- baseline方法是什么？
  Baseline 是标准 **Multi-Head Attention (MHA)**，具有 O(n²d_h) 计算复杂度和 O(n²) 内存复杂度，所有 query-key 对参与完整 scaled dot-product attention。其他对比 baseline 包括：**SWA (Sliding Window Attention)**——固定局部窗口 w，O(nwd_h) 计算但无法捕获长程依赖；**MLA (Multi-Head Latent Attention)**——低秩分解压缩 KV，减少内存但丢失细粒度信息，且无法动态调整压缩策略；**NSA (Native Sparse Attention)**——硬件对齐的块稀疏但静态模式无法适应输入内容变化；**H2O/InfLLM/Quest/DAM**——内容感知的 KV cache 选择方法，但依赖启发式离散操作，不可微且与 FlashAttention 的连续内存访问不兼容。

  全栈执行例子（MHA dense attention 在 A100 GPU 上处理长序列）：
  **算法pipeline**：标准 Transformer self-attention，QKV 线性投影后执行完整 scaled dot-product attention。seq_len=n 时计算 QK^T (n×n 矩阵)，softmax 后乘以 V，复杂度 O(n²d_h)，内存 O(n²)。
  **Serving调度**：论文未明确说明（聚焦于 attention 机制层面，非 serving 框架修改）。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2/3 CUDA kernel，完整 dense attention 前向传播与反向传播，所有 K/V tiles 均加载到 SRAM 参与矩阵乘。
  **硬件架构/芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DMA 通过三个核心创新解决 baseline 缺陷：
  **(1) Content-Aware Dynamic Mask 解决静态稀疏的灵活性不足**：从 value 向量表示 δ=exp(τ(v·Δ)×A) 生成每 head 独立的动态 mask，top-w 选择保留最重要位置。相比 SWA 的固定局部窗口只能捕捉近邻依赖，DMA 的动态 mask 能自适应跳过无关 token 并直接关注远距离语义相关 token。相比 MLA 的全局低秩分解丢失细粒度信息，DMA 保留完整 KV cache 且稀疏化发生在 attention weight 计算阶段而非压缩阶段。
  **(2) Position-Aware Sparse Weights 解决内容感知方法的非可微和硬件低效**：mask 为 −∞ 的位置 attention weight≈0，kernel 在 block 级别判断——若整个 K block 的 mask 全零则直接跳过加载和 M×M 操作。相比 H2O/Quest 等的 token 级离散选择破坏内存访问连续性（与 FlashAttention 不兼容），DMA 的 block 级跳过多路复用 FlashAttention tiling，硬件友好且完全可微。
  **(3) Fully Differentiable End-to-End Training 解决训练-推理 gap**：DMA 的 mask 生成（线性变换+softplus+exp+top-w）和 sparse weight 计算全程可微，梯度在选中位置与 full attention 完全一致。训练和推理使用相同稀疏策略，消除 post-hoc pruning 导致的性能退化。复杂度从 O(n²) 降为 O(n·w)，可同时用于高效训练和推理。

  全栈执行例子（DMA 在 A100 GPU 上处理长序列）：
  **算法pipeline**：QKV 投影后，value 向量经 Δ 投影 + softplus + A 门控 + exp 得到 per-head 重要性分数 δ，top-w 选择后生成动态 mask m_t（非选中位置 −∞）。attention 计算 o_t = softmax(q_t K^T/√d_h ∘ m_t) V，仅有效 w 个 key-value 位置参与实际计算，复杂度 O(nwd_h)。
  **Serving调度**：论文未明确说明（DMA 训练和推理共用同一架构，但未涉及 serving 框架层面的调度修改）。
  **编译框架**：论文未明确说明。
  **kernel调度**：Flash DMA CUDA kernel，outer loop 中先加载 mask block 调用 Judge() 判断，active=0 则跳过整个 K/V block；active≠0 时用 FlashAttention online softmax 递推计算 S_ij=Q_i K_j^T/√d_h + M_j。backward 中 dM=dS，无需额外存储 mask 梯度，与 forward 共享 skip logic。block 级跳过多路复用保证硬件效率——Forward 在 32K seq 提速 21.5×，Decode 在 128K key 提速 92.7×。
  **硬件架构/芯片设计**：论文未明确说明。

## WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

- baseline方法是什么？
  Baseline 是三类 state-of-the-art KV cache 压缩方法：(1) **StreamingLLM (SLM)**：基于 attention sink 现象，保留最前 b-α 个 token 和最后 α 个 token 的 KV cache，所有层使用统一 cache size，但丢弃了中间大量可能包含关键语义的 token；(2) **H2O**：基于 Heavy Hitter 观察（少量 token 贡献大部分 attention score），动态保留最近 token + 历史中累积 attention scores 最高的 token，所有层统一 cache size，但 token 级选择破坏语义连贯性；(3) **PyramidKV (PKV)**：观察到底层 dense attention、顶层 sparse attention 的金字塔效应，按等差数列跨层分配不同 KV cache size（底层多、顶层少），但仍然是 token 级离散选择。三种方法共同缺陷：(a) 逐个 token 选择导致 context 语义碎片化，破坏了人类阅读的窗口级信息处理模式；(b) 对所有任务使用统一压缩策略，没有考虑不同任务（信息定位 vs 信息聚合）对语义上下文的不同需求。

  全栈执行例子（LLaMA3-8B-Instruct + H2O/PyramidKV on A100 40G，KV cache size=2048, context=7950 tokens）：
  **算法pipeline**：LLaMA3-8B-Instruct（32 层 Transformer，GQA，8 KV heads/32 Q heads）。H2O 在每层计算 attention scores 后，对每个 head 维护最近 w 个 token + top-(b-w) 个累积 attention 最高的历史 token，其余 evict。Attention 计算时仅加载保留 token 的 KV → softmax 分布在离散 token 上，相邻 token 的语义关联被打断。PyramidKV 额外根据层编号分配不同预算（底层 1024 tokens，顶层 256 tokens），但仍选离散 token。复杂度：O(n·b) per layer。
  **Serving框架**：HuggingFace Transformers 加载模型，自定义 KV cache manager 在每层 decode 后动态 evict/retain KV entries。论文未修改 serving 框架本身。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention 或 PyTorch SDPA，无自定义 CUDA kernel。KV cache 通过 tensor indexing 动态裁剪，无 kernel 级优化。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  WindowKV 提出三个关键设计解决 baseline 缺陷：

  (1) **Task-Adaptive Window Selection → 解决语义碎片化和任务无关性**：不再逐 token 选择，而是将 context 切分为固定大小的 review windows（ω tokens/window），以 window 为单位做保留/evict 决策。Window 级别的选择天然保留了 consecutive tokens 的语义连贯性。同时训练 bert-base-cased 分类器将输入任务分为 Information Localization（QA 类，p=ω，保留窗口中所有 token 以理解完整语义）和 Information Aggregation（摘要类，p<ω，从每个窗口中提取 top-p 高注意力 token）。这种任务自适应机制确保不同任务场景下 KV cache 的分配策略与任务语义需求匹配。

  (2) **Observation Window 驱动的注意力打分 → 解决选择标准单调性**：使用最后 α 个 token（observation window）作为 query 端，计算其对 review context 中各 token 的累积注意力 t_j = Σ A_ij。这不同于 H2O 的全 query 平均注意力（容易被 outliers 主导），也不同于 PyramidKV 仅用 instruction tokens 作为 query（可能遗漏 context 中的重要长程依赖）。Observation window 紧邻生成位置，天然携带当前生成阶段最相关的上下文需求。

  (3) **Intra-Group Layer KV Cache Indices Sharing + Dynamic Budget Allocation → 解决计算效率**：利用相邻层 attention 分布的相似性（Jaccard similarity 实验验证），将 m 层分为 H=m/γ 组，仅每组首层执行完整的 window selection。预算按等差数列跨组分配（底层组多、顶层组少），继承 PyramidKV 的金字塔结构优势但应用于 window 级别。实验中 γ=7（Qwen2.5）或 γ=8（LLaMA3），window selection 的计算开销降至原来的 1/γ。

  全栈执行例子（WindowKV on LLaMA3-8B-Instruct on A100 40G，KV cache size=2048, context=7950 tokens）：
  **算法pipeline**：LLaMA3-8B-Instruct 32 层分为 4 组（γ=8）。输入 7950 tokens → 分类器判定任务类型 → 选取 (ω=8, α=16, p=8) for localization 或 (ω=16, α=32, p<16) for aggregation。仅 groups 首层 (layer 0/8/16/24) 计算 full attention A [7950,7950] → observation window [7934:7950] 累积 attention → token scores t_j → window scores s_k → 按 dynamic budget (b^0=704, b^1=576, b^2=448, b^3=320) 选择 top-n windows。组内其余 28 层直接复用首层 indices。结果：prefill 后仅保留 ~2048 tokens 的 KV cache（12% of 原 7950 tokens），LongBench 平均分 41.35 vs FKV 41.51（差距 0.16），Needle-in-a-Haystack 超越所有 baseline。
  **Serving框架**：HuggingFace Transformers + PyTorch，自定义 KV cache pruning module 在 prefill stage 后执行 window selection。Throughput test 显示 Vanilla+WindowKV+Classifier 吞吐 881 token/s vs Vanilla 764 token/s（+15%），延迟 1.14 ms/token vs 1.31 ms/token（-13%）。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 PyTorch tensor operations（attention score computation + gather/scatter for KV cache selection）。无自定义 CUDA kernel。Window selection 的额外 overhead 主要来自组首层的 full attention computation 和 top-k ranking，但因仅 4/32 层执行，overhead 可控（Classifer overhead 在 throughput test 中仅 ~13 token/s 下降）。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

## X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression

- baseline方法是什么？
  Baseline 是标准 Multi-Head Attention (MHA) / Grouped-Query Attention (GQA) 的 Transformer 模型（SmolLM 和 Llama 系列）。这些模型在推理时 KV cache 需求为 2·n_h·d_h·l（MHA）或 2·n_kv_heads·d_h·l（GQA），长序列推理时显存开销巨大。现有 post-training KV cache 压缩方法（如 H2O 的 heavy-hitter 驱逐、sliding window attention、PALU 的低秩投影压缩）虽然易于部署，但存在信息损失导致性能退化；而 training-based 方法（如 DeepSeek MLA）虽然效果更好，但需要从零开始 pre-training，计算成本极高（DeepSeek-V3 需 2.664M GPU hours on H800）。

  全栈执行例子（Llama3.2-1B-Instruct 推理，GQA + HuggingFace Transformers on AMD MI300）：
  **算法pipeline**：输入序列 H ∈ R^{l×d}，每层计算 Q = HW^Q, K = HW^K, V = HW^V（GQA 下 K/V head 数少于 Q head 数）。通过 softmax(QK^T/√d_h) @ V 产生注意力输出。KV cache 存储每 token 每层的全部 K 和 V 向量，cache 总量 = 2·n_kv_heads·d_h·l，Llama3.2-1B 32 heads × 64 dims × 2 = 4096 dims/token。长序列下 KV cache 迅速超越模型权重成为显存瓶颈。
  **系统框架**：HuggingFace Transformers / PyTorch 加载模型权重，使用 FlashAttention fused kernel 执行 attention。推理时逐步追加 KV cache，显存随序列长度线性增长。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention 做 tiled QK^T + online softmax + V 加权，内存访问量 O(T²d + Td) 。
  **硬件架构**：AMD MI300 GPU，使用 ROCm 生态。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 X-EcoMLA——通过 SVD 初始化 + 知识蒸馏 + DPO 将预训练 MHA/GQA 后训练转换为 MLA，避免 costly pre-training from scratch。核心设计：(1) **SVD-based 初始化**：从预训练 Q、K、V 的 SVD 分解直接构造 MLA 的 down/up-projection 矩阵，保留原模型 dark knowledge，相比 random init 在 SmolLM 上提升 22.8-30.91% 平均分；(2) **Joint KV SVD**：对 [W^K, W^V] 做联合 SVD 而非分别分解，捕获 K 和 V 之间的相关性，提高低秩近似的保真度；(3) **Dynamic Rank Selection**：通过能量阈值 δ 自适应确定每层 rank，比固定 rank 更灵活、无需手动调参；(4) **统一共享 RoPE Key**：所有 head 共享单一 K^R 向量（与 DeepSeek MLA 一致），在固定维度预算下每个 head 获完整 d_r 维位置编码（vs MHA2MLA 的 per-head 分配仅 d_r/n_h），提供 n_h× 的位置编码容量；(5) **知识蒸馏**：使用更大 teacher 模型（如 8B teacher for 1B student）通过 KL 损失传递 dark knowledge，弥补低秩压缩的信息损失——实验表明蒸馏远比纯 CE loss 有效（52.77 vs 48.54）；(6) **DPO 偏好对齐**：以蒸馏模型自身为 reference 做 DPO，进一步提升 benchmark 表现（+0.3-1.3 分）。

  全栈执行例子（X-EcoMLA-1B on AMD MI300, r_kv=128, d_r=32）：
  **算法pipeline**：输入序列 H，每层通过 MLA 计算。C_KV = H @ W_DKV 将 hidden state 压缩到 r_kv=128 维 latent；在推理时只缓存 C_KV[128] + K_R[32] = 160 dims/token（vs baseline 4096 dims/token，25.6× per-token 压缩）。上行矩阵 W_UK 和 W_UV 在推理时被吸收进 W_Q 和 W_O，无额外计算开销。通过 SVD 初始化 + teacher 蒸馏，虽丢失部分低频分量但 teacher dark knowledge 补偿了信息损失。最终 15.6% KV size 下 8B teacher 蒸馏的 1B 模型平均分 52.94 超越 baseline 52.85。
  **系统框架**：HuggingFace Transformers / PyTorch + ROCm，加载 MLA 权重。KV cache 大幅缩小使得同硬件下 batch size 从 128 扩展到 1024（显存 28 GB vs 143 GB），吞吐达 1.7-2×。
  **编译框架**：论文未明确说明。
  **kernel调度**：MLA 的 down/up-projection 为常规矩阵乘法（GEMM），无自定义 kernel。Attention 部分仍可用 FlashAttention 加速。上行矩阵吸收后推理等价于修改后的 MHA forward，额外开销仅为一个 d×r_kv 的 down-projection GEMM。
  **硬件架构**：AMD MI300 GPU，ROCm 生态。
  **芯片设计**：论文未明确说明。

## XAttention: Block Sparse Attention with Antidiagonal Scoring

- baseline方法是什么？
  Baseline 方法为基于 FlashInfer 的 FlashAttention（dense full attention），以及其他 training-free 稀疏注意力方法 MInference 和 FlexPrefill。

  **Baseline（MInference/FlexPrefill）的执行流程**：
  - **算法 Pipeline**：MInference 和 FlexPrefill 使用 "Vertical-Slash" 稀疏模式——通过分析输入序列末端的 query 来识别重要的"垂直列"和"斜线"注意力模式索引，然后用这些索引构建稀疏 mask 并执行 sparse attention。这种方法依赖 pooling（mean/sum pooling）来估计 block 重要性，但 pooling 在 block 内仅有少量显著垂直/斜线模式时会严重低估重要性。
  - **Serving 调度**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：MInference 使用 Vertical-Slash Sparse Index kernel（基于 point-range two-way merge 算法，在 GPU 上并行构建 per-row block indices）和 Vertical-Slash FlashAttention kernel（混合 block-sparse + column-sparse 计算）。Pattern selection 的 index search 开销巨大，尤其在短上下文时（pattern selection overhead 占比高）。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：
  1. Pattern selection 计算开销大——MInference 的 vertical-slash index search 和 FlexPrefill 的复杂超参数搜索在短上下文时反而成为瓶颈。
  2. Pooling 方法不准确——mean/sum pooling 无法有效检测 block 内的稀疏但关键的垂直/斜线模式。
  3. 固定稀疏度策略（Top-K 或 Top-Ratio）无法适应不同序列长度和输入内容。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  XAttention 提出使用注意力矩阵的**反对角线值之和**（antidiagonal sum）作为 block 重要性的轻量级代理。核心洞察：反对角线天然交叉 block 内所有可能的垂直和斜线注意力模式，因此反对角线和能高效、准确地检测这些关键模式。

  **XAttention 的执行流程**（全栈对比 Baseline）：
  - **算法 Pipeline**：三步流程——(1) Strided Antidiagonal Scoring：按步长 S 沿反对角线采样并求和；(2) Threshold Block Selection：基于累积 softmax 概率选择 block，实现动态稀疏度（不同头、不同输入稀疏度自适应）；(3) Minimum Threshold Prediction：通过动态规划为每个注意力头离线搜索最优阈值 τ_h，进一步优化稀疏度-准确率平衡。

  - **Serving 调度**：论文未明确说明。

  - **编译框架**：论文未明确说明。

  - **Kernel 调度**：基于 FlashInfer 框架实现。反对角线 scoring 的计算复杂度仅 O(L×d/S²)（vs MInference 的 O(L×k_v×k_s)），通过简单的 Q/K reshape + 小矩阵乘法完成近似注意力分数计算。Block selection 使用贪心累积阈值算法，无需复杂的 index search。Sparse attention 直接调用 FlashInfer 的 block-sparse kernel，仅计算选中 block 对的注意力。

  - **硬件架构**：论文未明确说明。

  - **芯片设计**：论文未明确说明。

  **对比 Baseline 的关键改进**：
  1. **反对角线 scoring 解决 Pooling 不准确问题**：反对角线交叉每个 block 内所有垂直和斜线模式（Figure 2），保证了信息完整性——每个 token 至少贡献一条反对角线的值。消融实验显示 antidiagonal pattern 在同等计算量下密度最低且准确率最高（S=8: antidiagonal average 88.47 vs random 82.48 vs diagonal 81.06）。
  2. **极低 scoring 开销解决 Selection 瓶颈**：反对角线 scoring 仅需 reshape Q/K + 小矩阵乘法，计算量仅为完整注意力的 1/S²。实测 pattern selection 比 MInference 快 24.9×、比 FlexPrefill 快 5.9×。
  3. **动态阈值解决固定稀疏度问题**：Threshold block selection 按累积概率 τ 自适应决定稀疏度——短序列（注意力密集）自然保留更多 block，长序列（注意力稀疏）自动提高稀疏度。128k 时密度仅 6.89%（S=8），4k 时密度 52.16%。DP-based per-head threshold 进一步优化。
  4. **训练自由且即插即用**：与需要 costly pretraining 的 SeerAttention 不同，XAttention 无需任何训练，可直接替换任意 Transformer 模型的注意力模块。

## ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

- baseline方法是什么？
  Baseline 是现有 KV cache 管理方法，分为三类：
  (1) **稀疏驱逐（Sparse Eviction）**：H2O 基于累积 attention score 阈值动态驱逐低贡献 token + 保留最近 token；StreamingLLM 固定保留 attention sinks（初始 token）+ 滑动窗口；SnapKV 在 prefilling 阶段做一次性 KV cache 剪枝。共同缺陷：**永久丢弃 token 导致不可逆信息损失和 attention distribution drift**——被驱逐 token 的 KV 信息完全丢失，后续解码步无法恢复，错误在长序列中累积传播（Figure 4 中纯驱逐在 5% cache 下相对误差远高于残差合并）。
  (2) **Token 合并（Token Merging）**：LESS 使用辅助网络学习压缩 token，通过 recurrent merging 合并相似 token 并做 attention rectification。缺陷：**需要额外网络 + 训练数据（C4）微调**，跨模型架构和新任务泛化差（Falcon-7B 上 5% cache 下 ROUGE-1 从 27.06 暴跌至 7.75），训练开销大。
  (3) **上下文无关优化**：GQA/MQA（减少 KV head）、KV cache 量化（KV cache 低精度存储）、低秩近似。缺陷：模型架构修改或精度损失。

  全栈执行例子（LLaMA2-7B + H2O on NVIDIA A800-80GB）：
  **算法pipeline**：LLaMA2-7B 32 层 × 32 heads × 128 dim，在解码步 T 时 KV cache 为 K_T, V_T ∈ R^{T×d_head}。H2O 维护每个 token t 的累积 attention score Σa_t，在每个 decoding step 保留 top-k "heavy hitter" tokens + 最近 w 个 tokens，其余永久驱逐。被驱逐 token 的 key-value 信息完全删除，后续所有 query 无法访问。当 T 增长到 54K 时，即使只保留 5% tokens（~2700），剩余 tokens 的 attention distribution 也显著偏移——heavy hitters 筛选依赖历史 attention pattern，而该 pattern 本身因之前驱逐而被扭曲（compounding error）。
  **系统框架**：HuggingFace Transformers 加载 LLaMA2-7B，使用 PyTorch SDPA。H2O 在每个 attention layer 的前向传播中插入 KV cache 管理逻辑。
  **编译框架**：论文未明确说明。
  **kernel调度**：论文未明确说明。使用标准 PyTorch scaled_dot_product_attention。
  **硬件架构**：NVIDIA A800-80GB，VRAM 25GB 用于模型参数加载，KV cache 随 token 线性增长 ~1MB/token。FullKV 在 54K tokens 时 KV cache 消耗 54GB，总 VRAM ~79GB > 80GB 导致 OOM。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ZSMerge 提出"稀疏+残差"混合压缩，在不引入参数/训练的前提下解决驱逐和合并方法的缺陷：
  
  (1) **残差合并解决驱逐的信息损失**（核心创新）：Baseline 驱逐方法永久删除低贡献 token → ZSMerge 动态将被驱逐 token 合并入 Br 个残差 slot。每次驱逐 token (k_t, v_t)，通过 k_r·k_t 找最相似 residual slot，用增量均值聚合更新 slot（Eq. 7）。这等价于将多个原始 token 压缩编码为一个 slot，而非丢弃——解除了"驱逐即信息永久丢失"的根本约束。Figure 4 验证：残差合并比纯驱逐减少 37-89% 的 attention 输出误差。
  
  (2) **补偿注意力解决合并后表示偏差**（Theorem 1）：Token 合并产生 mismatch——k_r 是多个 token 的均值，与对应的 v 分布不匹配。ZSMerge 在 softmax logit 中加 α·log w_r 补偿项（Eq. 8），其中 w_r 为 slot r 合并的 token 数。log w_r 将"此 slot 代表多个 token"的信息注入 attention 评分。Theorem 1 证明 â_i ≥ a_i（∀ 未压缩 token i），即未压缩 token 的 attention 占比在压缩后不降低，防止压缩 token 因 log w 补偿而"过度放大"。
  
  (3) **三分区 budget 优于已有方法的二分法**：H2O 仅重排 heavy hitters + 最近 token，LESS 用合并网络。ZSMerge 的三分区（proximity + context + residual）将"保持局部上下文"、"保留全局关键信息"、"压缩冗余"三者解耦分配预算——这是纯驱逐或纯合并无法同时做到的。
  
  (4) **零样本无参数设计解决泛化问题**：LESS 的辅助合并网络需在 C4 上训练，泛化到 Falcon 架构时性能崩溃。ZSMerge 仅依赖 token 间 key 相似度（dot product）和累积 attention score（Eq. 5），无任何学习参数，可直接应用于 MHA/MQA/GQA 任意架构。

  全栈执行例子（ZSMerge on LLaMA2-7B, NVIDIA A800-80GB）：
  **算法pipeline**：LLaMA2-7B 相同 32 层，每层每个 head 独立运行 ZSMerge。解码步 T：(a) 更新 s_t = 0.98·s_t + a_t（每个 token 累积 attention 衰减和）；(b) 分配 B=B_p+B_c+B_r，B_p 保留最近 token（proximity），B_c 选 top-B_c 按 s 排序（context），剩余 token 按 Eq. 6-7 合并入 B_r 个 residual slot；(c) K_B = [K_p∥K_c∥K_r] 拼接后，用 Eq. 8 计算补偿注意力（每个 slot k_r 带 log w_r 偏置）；(d) softmax + V_B 加权输出。复杂度 O(T + B·d)，线性于 T。54K tokens 下仅需 18K cache budget（~3:1 压缩），VRAM 恒定 43GB（82% 减少），解码吞吐维持 9 tokens/sec（3× FullKV）。
  **系统框架**：HuggingFace Transformers，仅全局替换 `scaled_dot_product_attention` 函数。支持 LLaMA/Falcon/Mistral 系列。`change_mode` 方法支持运行中切换压缩模式。prefill 阶段通过 `window_size` 参数限制 s 初始化范围（类似 SnapKV）。
  **编译框架**：论文未明确说明。
  **kernel调度**：论文未明确说明。使用 PyTorch 标准 SDPA 或 Flash Attention v2（通过 KVCache-Factory）。无自定义 CUDA kernel。
  **硬件架构**：NVIDIA A800-80GB。实验使用单 GPU 推理。
  **芯片设计**：论文未明确说明。

## dKV-Cache: The Cache for Diffusion Language Models

- baseline方法是什么？
  Baseline 是标准的 diffusion language model（DLM）推理，具体为 LLaDA-8B-Instruct 和 Dream-Base-7B 在无 KV-Cache 下的全序列去噪推理。DLM 推理中，每个去噪步需要完整编码长度为 L 的全部 token（双向注意力），生成 L 个 token 需 T 个去噪步，复杂度为 O(L³)（AR 模型带 KV-Cache 仅为 O(L²)）。加速 baseline 为 Few-Steps/Half-Steps（减少去噪步数到 50-62.5%），但以生成质量为代价。

  全栈执行例子（LLaDA-8B-Instruct 标准推理，A6000 GPU）：
  **算法pipeline**：输入全 [MASK] 序列 x^{1:L}_{c(T)}，每步 t ∈ [T, 1]：调用 p_θ 预测全部 L 个位置的 x_0，根据 confidence/random 策略选择部分位置 remask，其余位置保持已解码 token 不变。每步需对全部 L 个 token 计算双向 self-attention（QKV ∈ R^{L×d}），softmax 在全部 L 个位置上归一化。L=256, T=256 时总计算量 = 256 步 × O(256²) = O(256³)，与 L=256 的 AR（256 步 × O(256²) cumulative = O(256³)）理论相当，但因缺少 KV-Cache 导致每步都从头计算全部 token 的 K/V。
  **系统框架**：HuggingFace Transformers + PyTorch，LLaDA-Model 加载 8B 参数，使用标准 Transformer forward（无 caching），每步完整计算所有 hidden states。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2 kernel，每步执行 O(L²d) 的完整双向注意力。无 KV-Cache 的情况下每次都需要从 HBM 读取完整的 K/V 矩阵，memory-bound 严重。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 delated KV-Cache (dKV-Cache) —— 首个用于 DLMs 的 KV-Cache 机制。核心思想源于对 DLM 去噪过程中 token 表征动态的实证观察（Figure 2）：(1) 已解码 token 的 K/V 表征在后续步趋于稳定，而 [MASK] token 持续波动；(2) 相邻步间 K/V 相似度整体较高。据此设计延迟缓存策略：(a) **延迟缓存（delayed caching）**：仅缓存已解码 token 的 K/V 并跨步复用，掩码 token 每步重新计算；(b) **一步延迟（one-step delayed caching）**：使用上一步 M_{t-1} 而非当前步 M_t 决定缓存集合，避免刚解码 token 在表征剧变时被过早缓存，这是保证质量的关键设计（Figure 3 证明无延迟时性能崩溃）；(c) **缓存刷新（cache refreshing）**：每 N 步清空缓存重新计算全序列 K/V，避免长时间累积的缓存误差导致质量退化。两种变体覆盖不同场景：dKV-Cache-Decode（近乎无损，refresh=4-8）和 dKV-Cache-Greedy（O(L²) 复杂度，refresh=2，加局部窗口 w≤6）。

  全栈执行例子（dKV-Cache-Decode on LLaDA-8B-Instruct, A6000 GPU）：
  **算法pipeline**：每步 t，根据上一步掩码集 M_{t-1} 确定已解码 token（缓存复用）和仍在掩码的 token（重新计算）。重排序列：将缓存 token 置于左侧（连续块），掩码 token 置于右侧，同时调整位置编码。Transformer 仅计算掩码 token 的 Q/K/V（|M_{t-1}| 个 token），已解码 token 的 K/V 从缓存直接拼接（concat），完整 K^I / V^I 参与注意力计算。注意力输出 scatter 回原始位置。每 8 步刷新缓存。与 Baseline 对比：每步计算量从 O(L) 降至 O(|M_t|)，cache ratio 从 0 提升到逐渐接近 1。|M_t| 从 L 递减到 0，累计加速约 2-3.5×。
  **系统框架**：PyTorch + HuggingFace Transformers，修改 LLaDA 模型的 forward 函数增加 concat_reorder 逻辑。concat_reorder 实现将索引操作从 K/V 矩阵层级（[B,L,D]）转移到 token 层级（[B,L]），大幅减少内存碎片。Generation 脚本修改为 step-by-step 调用并管理缓存状态。
  **编译框架**：论文未明确说明。
  **kernel调度**：使用标准 FlashAttention。concat_reorder 通过重排使缓存 token 连续，从而可用简单 concat 和 slice 操作替代高开销的 gather/scatter 索引。位置编码重排仅每步一次、跨层共享，开销可忽略。数据移动仍为关键瓶颈，batch size=1 时 memory-bound 导致加速有限甚至退步。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

  dKV-Cache-Greedy 的额外设计：将缓存集合从 I \ M_{t-1} 激进缩减为 M_t = {D_t} ∪ {D_{t-1}} ∪ W(D_{t-1})（3 个组件共最多 8 个 token），将每步计算量固定为 O(w·L)（w 为窗口大小 ≤6），复杂度从 O(L³) 降至 O(L²)，以轻微性能下降换取更大加速（1.51-1.73× vs baseline speed 即 1.63-1.70× speedup on LLaDA）。

## xKV: Cross-Layer SVD for KV-Cache Compression

- baseline方法是什么？
  Baseline 分为两类：(1) **MiniCache**：利用相邻层 KV-Cache 的 token-wise cosine similarity 假设，通过 SLERP 合并相邻层的 KV 对来实现跨层压缩。执行流程：从 Transformer 中间层到末尾层，对半数的相邻层对执行 SLERP merging；每对合并后共享一份 KV-Cache。(2) **Single SVD (per-layer SVD)**：对每一层的 KV-Cache 独立执行 SVD 分解，保留 top-r 奇异值/向量，压缩每层的 KV-Cache，但未利用跨层冗余。

  **全栈执行例子（Baseline: MiniCache）**：
  - 算法层：计算相邻层 KV-Cache 的 token-wise cosine similarity → SLERP 插值合并 → 用合并后 KV 替换原两个 layer 的 cache
  - Serving层：HuggingFace 推理，prefill 后执行合并，decode 阶段直接使用合并后的 KV-Cache（论文未说明修改特定 Serving 框架）
  - 编译框架：未涉及
  - Kernel调度：未涉及
  - 硬件架构：未涉及

  **Baseline 痛点**：MiniCache 依赖 token-wise cosine similarity 假设，但实际中相邻层 embedding 的 token 级相似度很低（Figure 2a），导致高压缩比下准确率急剧下降（Qwen2.5-7B 上 1.3× 压缩就崩到 avg 5.7%）；Single SVD 到 8× 压缩时同样发生灾难性性能退化（Llama-3.1-8B 上 avg 仅 35.3%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  xKV 发现：尽管 token-wise cosine similarity 很小，但相邻层 KV-Cache 的**主导奇异向量（dominant left singular vectors）**高度对齐（通过 CKA 度量验证，Figure 2b）。基于此，xKV 将多层 KV-Cache 水平拼接后做一次统一的跨层 SVD，提取共享的 left singular vectors（共享基 A），各层只保留独立的 reconstruction matrix（B_ℓ_i）。

  **全栈执行例子（xKV）**：
  - 算法层：Stride-based 分组（相邻 G 层一组）→ 每层 pre-RoPE key/value states 水平拼接 → 对拼接矩阵做 SVD → 保留共享基 A (= U_r @ S_r) 和层独立 B_ℓ_i（Vt_r 分块）→ decode 时 A @ B_ℓ_i 重构 → 对重构 key 重新施加 RoPE
  - Serving层：HuggingFace 推理实现，prefill 阶段在线 SVD 分解（<10% prefill time at 128K），decode 阶段逐 token 重构并查询（论文未说明修改特定 Serving 框架）
  - 编译框架：未涉及
  - Kernel调度：未涉及（论文未实现 custom kernel，仅用 PyTorch/HuggingFace 原生算子）
  - 硬件架构：未涉及

  **关键设计 vs Baseline 缺陷映射**：
  1. **MiniCache 的 token-wise cosine 假设不成立** → xKV 改用 CKA 发现跨层奇异向量对齐，用 SVD 共享子空间替代 token 级合并
  2. **Single SVD 仅利用单层低秩性** → xKV 通过跨层拼接 SVD 利用多层共享基，相同 rank 下保留更多信息（Figure 2c: 层越多所需相对 rank 越低）
  3. **跨层合并受限于 pairwise 操作（2 层一组）** → xKV 的 stride-based grouping 支持任意组大小 G（论文验证了 G=2,4），组越大共享子空间越丰富
  4. **离线统计无法适配不同上下文** → xKV 采用在线 SVD（per-request），捕捉上下文动态变化
  5. **MLA 架构压缩困难** → xKV 直接对 MLA latent representations 做跨层 SVD（non-RoPE 部分），在已压缩的 latent cache 上再获得 3× 压缩

## TreeKV: Smooth Key-Value Cache Compression with Tree Structures

- baseline方法是什么？
  Baseline 是三类 KV cache 压缩方法：(1) **位置驱动方法（StreamingLLM, LM-Infinite）**：仅保留 initial tokens（attention sinks）和 recent tokens（sliding window），丢弃所有中间 token。缺陷：可能漏掉"预定义区域"之外的重要信息，如位置靠前但与当前生成高度相关的上下文。(2) **全局重要性方法（H2O, TOVA, Scissorhands）**：基于 attention weights 的全局排序（H2O 用累积 attention、TOVA 用最后 token 的 attention、Scissorhands 二值化）选择高分 token。缺陷：产生强烈的**区域偏差（regional bias）**——图 1 显示 H2O 和 TOVA 在特定位置区域集中选择 token，无法覆盖序列全局，导致 KV cache 失去全局视图，损害需要完整上下文的复杂任务。(3) **Prefilling-only 方法（SnapKV, PyramidKV）**：仅优化 prefill 阶段，通过 funnel-like 跨层策略选择关键 token。缺陷：只覆盖一个阶段，decoding 阶段仍需额外策略。

  全栈执行例子（H2O on Llama-2-7B decoding stage, RTX 4090）：
  **算法pipeline**：对生成 step t，标准 QKV projection 后追加新 KV pair。H2O 在每个 head 独立计算 token i 的累积 attention score = Σ_j a_j[i]（a_j 是 step j 的 attention weights vector），对所有历史 token 的累积 attention 做全局排序，保留 top-k。复杂度：每 step 需 O(t) 空间存储 per-head attention scores + O(t log t) 排序。缺陷：由于每次都在全局范围贪心选高分 token，attention scores 相关性强 → 被选 token 在位置空间中集聚在少数区域（如高注意力密集段），未被选区域的好 token 永久丢失。Llama-2-7B 32 layers × 32 heads → 共 1024 个独立缓存策略（因每个 head 独立选择）。
  **系统框架**：HuggingFace Transformers 原生实现，每次 decode step 对所有层同步追加新 KV 并执行独立淘汰。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention kernel 计算 QK^T attention weights。H2O 的额外开销为 per-head attention score 累积与排序，常驻内存随序列长度线性增长。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TreeKV 通过三个核心设计解决 baseline 缺陷：

  **(1) Wavelet 分析发现"平滑递增"规律 → 树形淘汰结构解决 Regional Bias**
  对 attention-weighted values 信号做 multi-level Haar wavelet 分解发现：从远到近，token 的信息贡献平滑递增，且与邻居的差异性递增（高频分量增长尤为显著）。这表明上下文存在 smooth transition from coarse-grain (distant) to fine-grain (nearby)。基于此设计"左疏右密"的 tree structure：eviction scope 在 cache 中从远（idx=1）到近（idx=c）循环移动，每次仅在相邻两个 token {idx, idx+1} 中淘汰较低的重要性分者。这保证淘汰均匀分布在整个序列上 → 与 H2O/TOVA 的"在某个区域集中淘汰"形成对比，图 1(c) 的 token distribution map 证实 TreeKV 分布更均匀。

  **(2) 循环 eviction scope 而非全局贪心 → 解决全局排序的计算负担**
  不同于 H2O 每次对所有 token 做全局排序（O(t log t)），TreeKV 每次仅比较相邻两个 token 的重要性（O(1)），且 idx 循环递增保证每轮每个 token 都有被评估的机会。这本质上在 token 间建立了二叉树竞争关系：左子树 vs 右子树逐级向上淘汰低分 token → tree structure 平滑保留各层级的"胜出者"，而非贪心取全局 top。

  **(3) Block-level prefill + observation window query → 统一双阶段**
  大多数方法（H2O 除外）只覆盖 decoding 或只覆盖 prefilling。TreeKV 在 prefill 将 prompt 切分为 blocks，用最后一个 block 做 observation window query 得到各 block importance，再在 block 级别复用 decoding 的树形淘汰。所有 blocks 并行计算。position encoding re-assignment 保证淘汰后位置编码语义连续性。

  **(4) Ablation 证实树结构才是核心，非 attention weight**
  TreeKV_Select_Left_Token 变体（每次固定淘汰左侧 token，完全不用 attention weight）在 PG19 65k token 书上与完整 TreeKV 的 perplexity 差距极小（Figure 5），而两者均远超 H2O → 树结构本身而非 attention-weight-based selection 才是性能来源。

  全栈执行例子（TreeKV on Llama-2-7B decoding, RTX 4090）：
  **算法pipeline**：每 layer 每 head 维护独立的 importance scores（S: 累积 attention、C: 计数）。step t 时：(a) 标准 QKV projection → append cache；(b) 计算 attention a = softmax(qK^T/√d)；(c) 更新 S += a, C += 1；(d) 若 |cache| > c：比较 S_avg[idx] vs S_avg[idx+1]，淘汰较低者，idx = (idx+1) mod c + 1；(e) re-assign position IDs。关键差异 vs H2O：每 step 淘汰 O(1) 比较 vs O(t log t) 全局排序，且淘汰均匀分布在序列 → 长序列下 cache 保留 coarse-to-fine 的信息层次。复杂度：额外 O(c) 存储 per head → 与 H2O 相同量级，但每 step 计算量为 O(1)。
  **系统框架**：HuggingFace Transformers + PyTorch，使用 HuggingFace 原生 LlamaForCausalLM 加载模型，在前向传播的 attention 层中插入 TreeKV cache management 逻辑（无需修改模型权重）。
  **编译框架**：论文未明确说明。
  **kernel调度**：标准 FlashAttention 计算 attention weights。TreeKV 增加的 O(1) per-step overhead（一次比较 + 一次 index 更新）可忽略。无自定义 CUDA kernel。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

## Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

- baseline方法是什么？
  Baseline 是 **Ring Attention**（Liu et al., 2023），一种将精确 attention 计算在序列维度上跨 GPU 并行化的方法。Ring Attention 将 K,V 在序列维度分片到 p 个 GPU，解码时将各 GPU 的 K,V chunk 通过 P2P 在逻辑环形拓扑中依次传递，每个 GPU 依次处理所有 chunk 的 attention 计算。其核心缺陷：
  (1) **通信步数线性增长**：每个 GPU 需要依次接收并处理所有 p 个 chunk，通信步数 O(p)，序列长度增加或 GPU 增加时延迟线性增长。
  (2) **非拓扑感知**：Ring Attention 假设均匀网络带宽的环形拓扑，但现代 GPU 集群具有两层拓扑——intra-node NVLink (900 GBps) 和 inter-node InfiniBand (400 Gbps per link)。Ring 的 uniform P2P 模式被最慢链路（inter-node）瓶颈限制，无法利用 intra-node 高带宽。
  (3) **高通信量**：每个 step 传输完整 K,V chunk (2btd elements)，总通信量 V_ring = 2btd × p，随 GPU 数和 chunk 大小线性增长。
  (4) **解码场景下通信无法 overlap**：单 token 解码时 per-GPU attention 计算仅需 ~10μs，而传输 K,V chunk 需 ~1ms (intra-node) 到 ~10ms (inter-node)，计算太快无法隐藏通信延迟。
  (5) **高峰值内存**：需存储相邻 GPU 传来的 K_chunk, V_chunk 和输出 chunk，Mem_ring = 4btd + 2bd。

  全栈执行例子（Ring Attention on 8×H100 DGX, decoding 640K context, d=2048, BF16）：
  **算法pipeline**：序列长度 N=640K 分片到 p=8 GPU，每 GPU 持有 t=80K tokens 的 K,V 分片。解码时：GPU_0 的 query q 广播到所有 GPU → 每 GPU 使用 Flash Attention 2 计算 q 与当前本地 chunk 的 attention → 通过 P2P send/recv 将当前 K,V chunk 传给下一 GPU → 重复 p=8 次 → 合并结果。总通信量 = 8 × 2 × 80000 × 2048 × 2 bytes (BF16) ≈ 5GB per token。
  **系统框架**：JAX / PyTorch + Flash Attention 2 per-GPU + NCCL P2P send/recv。Ring Attention 的 JAX 实现：https://github.com/nshepperd/flash_attn_jax。
  **编译框架**：论文未明确说明。
  **kernel调度**：每 GPU 独立执行 Flash Attention 2 kernel（tiled QK^T + online softmax），P2P communication 在 attention kernel 之间进行。GPU 间通信使用 NCCL P2P (nvlink + IB)，通信时间 >> 计算时间，无法 overlap。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Tree Attention 通过将 self-attention 表述为能量函数梯度（Observation 1: attention = ∂F/∂ζ|_{ζ=0}），利用 logsumexp 的 associative 性质设计树形归约并行化，从根本上解决 Ring Attention 的五个缺陷：

  **(1) 对数通信步数 → 解决线性增长**（Theorem 1）：
  Associative reduction (logsumexp/max) 可用 tree reduction 在 O(log p) 步完成，而非 Ring 的 O(p)。Algorithm 3 仅需 3 次 AllReduce（1×max + 2×sum），通信步数 O(log p)。实验证实：128 GPU + 5.12M 序列达到 8× speedup。

  **(2) 拓扑感知通信 → 解决非均匀带宽瓶颈**：
  AllReduce 的 tree reduction 模式天然适配 GPU 集群的两层拓扑：NCCL 自动在 intra-node 使用 ring reduce（高带宽 NVLink）、inter-node 使用 tree reduce（低带宽 InfiniBand）。Tree Attention 通过调用 NCCL AllReduce（而非手写 P2P ring）将拓扑优化委托给通信库——intra-node 快速归约后仅传递标量级中间结果跨节点，避免 Ring Attention 中每个 chunk 都需经过 inter-node 链路。

  **(3) 通信量降低 → 解决高通信量**：
  Tree Attention 传输的是部分归约结果（分子 n ∈ R^{d_h}、分母 d ∈ R^1、max m ∈ R^1），而非完整 K,V chunk。V_tree = 2(p-1)/p × (bd + 2bn_h)，与 chunk 大小 t 无关。对比 V_ring = 2btd × p，当 t 大时差异显著。以 640K/8GPU/d=2048 为例：Tree 单次通信 ~4K elements，Ring 每次 ~320K elements（~80× reduction）。

  **(4) 无需 overlap → 解决解码场景通信瓶颈**：
  解码时 per-GPU attention 计算极快（~10μs），Ring 即使尝试 overlap 也无法隐藏 ~1ms 的 chunk 传输（100× 差距）。Tree Attention 将通信变为标量级 AllReduce（~μs 级），无需 overlap 策略即可达到低延迟。论文 6.3 节明确分析："overlapping communication and computation in the decoding case is infeasible because of how fast the attention computation on a single GPU is relative to how long it takes to communicate the chunk of K,V"。

  **(5) 内存降低约 2× → 解决高峰值内存**：
  Tree Attention 不需要存储相邻 GPU 传来的 K,V chunk 和输出 chunk，峰值 Mem_tree = 2btd + 2bd + 2bn_h ≈ Mem_ring / 2（因为 2bn_h << 2btd）。实验验证：doubling hidden size from 2048 to 4096，gap doubles from 524MB to 1040MB。

  全栈执行例子（Tree Attention on 8×H100 DGX, decoding 640K context, d=2048, BF16）：
  **算法pipeline**：N=640K 分片到 p=8 GPU，t=80K。每 GPU：(a) Flash Attention 2 计算 q 与本地 K_i,V_i 的局部输出 o_i 和 lse_i；(b) AllReduce(max): 获取全局 max m_global；(c) 本地修正 n_i = o_i·exp(lse_i - m_global), d_i = exp(lse_i - m_global)；(d) AllReduce(sum): 归约全局分子 n_global, 分母 d_global；(e) z = n_global / d_global。全程 K,V 不移动，通信仅传输标量 lse (1 elem)、n (d_h elems)、d (1 elem)。总通信量 ≈ 3×(d_h+2)×2 bytes ≈ 780 bytes per token（vs Ring 的 ~5GB），6 个数量级的差异。
  **系统框架**：JAX + shard_map + Flash Attention 2 + NCCL AllReduce。KV 在序列维度分片 (`P(None, 'i', None, None)`)，query 广播到所有 GPU，AllReduce 通过 `lax.pmax`/`lax.psum` 调用 NCCL。
  **编译框架**：论文未明确说明。
  **kernel调度**：Per-GPU: Flash Attention 2 kernel（tiled QK^T + online softmax）。Cross-GPU: NCCL AllReduce（intra-node ring reduce + inter-node tree reduce），NCCL 自动选择通信拓扑。每个 AllReduce 传输标量级数据，延迟远低于 P2P chunk 传输。
  **硬件架构**：论文未明确说明。
  **芯片设计**：论文未明确说明。
