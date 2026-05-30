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
