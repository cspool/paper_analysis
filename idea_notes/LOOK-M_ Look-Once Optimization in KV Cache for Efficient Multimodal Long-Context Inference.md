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
