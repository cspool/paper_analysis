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
