## VisiPruner: Decoding Discontinuous Cross-Modal Dynamics for Efficient Multimodal LLMs

- baseline方法是什么？
  Baseline 是标准 MLLM（以 LLaVA-v1.5 7B 为典型代表）的 **dense 全 token 推理**：ViT 编码图像为 576 个视觉 token → MLP projector 映射到 LLM 维度 → 与 text token 拼接 → 32 层 LLaMA 2 transformer 逐层处理，每层对所有 visual tokens 执行 full cross-attention（Q_text 与 K_visual、V_visual 计算 softmax attention）+ visual self-attention + text self-attention + FFN，然后 autoregressive 解码。这种 dense 模式也被现有 training-free token pruning 方法（FastV、SparseVLM、PyramidDrop、FitPrune）继承，它们主要依赖 cross-attention weights 来选择保留哪些 token。

  全栈执行例子（以 LLaVA-v1.5 7B + LLaMA 2 7B 推理为例）：
  - 模型推理算法层：输入 336×336 图像 → CLIP-ViT-L/14（24层，patch size=14）编码为 576 tokens × 1024 dims → MLP Projector 映射至 4096 dims → 与 text instruction tokens（~74 tokens）拼接为 650 tokens 序列 → 送入 LLaMA 2 7B（32 layers, d=4096, 32 heads, GQA with 32 kv_heads, FFN intermediate=11008）。每层：① text tokens 对 full visual tokens 做 cross-attention（QKV 投影后 softmax 得到 N_text × 576 attention matrix），② visual tokens 之间做 self-attention，③ text tokens 之间做 causal self-attention，④ 各自过 FFN。32 层后 autoregressive 生成。首 token 延迟严重受 N_v=576 的 attention O(N_text × N_v) 影响；生成长回答时 KV cache 需存储全部 576 个 visual tokens × 32 层 = 18,432 个 visual KV entries，每个 entry 为 4096 × 2(K+V) × 2 bytes(FP16) = 16KB，总计 ~295 MB 仅用于 visual KV。
  - 系统框架层：LLaVA 标准推理 pipeline（HuggingFace Transformers 加载 LLaMA 2 + CLIP ViT），无特殊 serving 修改。推理时 visual token 的 attention 计算随 token 数平方增长。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch nn.MultiheadAttention / scaled_dot_product_attention，无自定义 kernel。
  - 硬件架构层：论文未明确说明具体 GPU 型号（仅提及 models up to 13B）。

  Baseline 的核心缺陷：
  (1) **浅层视觉计算完全冗余**：现有方法假设浅层 cross-attention 负责跨模态融合（基于高 attention score 的观察），但实验表明浅层 cross-attention scores 与输入指令无关（静态 attention pattern），mask 高 attention token 对性能无影响，视觉 token 的作用仅为 attention sink 而非信息传递——计算完全浪费。
  (2) **attention-based token 选择不可靠**：FastV/SparseVLM 等基于 attention weights 选择关键 token，但由于 visual attention sink 现象、attention 分布的分散性（难以隔离单 token 影响）、以及固定阈值缺乏任务适应性，导致选中的 token 并非真正关键。
  (3) **深层仍保留视觉 token 产生噪声**：深层模型行为已切换为纯语言 refinement，继续处理视觉 token 不仅无益，反而引入噪声（论文实验：skip layer 26 的视觉处理反而比继续处理性能更好）。
  (4) **缺乏对跨模态交互阶段性的理解**：现有方法未区分浅/中/深层的不同角色，一刀切地减少 token，导致在高压缩率（-97.6% attention）下性能崩溃（如 PDrop retained=64 时 MMB 降至 33.3）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VisiPruner** 基于对 MLLM 跨模态交互三阶段规律的实验发现，提出分层剪枝策略：
  (1) **浅层 Attention Merging**（解决缺陷 1）：在 layer 1 将所有视觉 cross-attention 合并到单个随机 token 作为 attention sink（公式 A^{(1)}_{i,j} = Σ_{v∈V} A^{(1)}_{i,v} if j=k else 0）；layer 2+ 完全跳过 cross-attention 和 visual self-attention，仅保留 FFN 对视觉 token 的处理。
  (2) **中层 Influence-based Token Selection**（解决缺陷 2）：不依赖 attention scores，而是直接评估每个视觉 token mask 后对最后输入 token 的 attention output 的影响，使用 cosine similarity（方向变化）和 L2 distance（幅度变化）双指标——这捕获了 token 的实际信息贡献而非 attention 中的权重分布。以 cosine < 0.995 定义过滤层，L2 < 0.2 丢弃，平均 576 → 10.3 tokens。
  (3) **深层 Vision Exit**（解决缺陷 3）：持续追踪保留 token 的 influence，一旦连续两层无影响则在此层移除所有视觉 token，后续层纯文本处理。
  (4) **三阶段统一框架**（解决缺陷 4）：训练无关、即插即用，不修改模型权重，在 -98.3% visual attention 压缩下 MMB 仅从 64.3→62.0（相比之下 PDrop retained=64 的 -97.6% 压缩下 MMB 降至 33.3）。

  全栈执行例子（VisiPruner + LLaVA-v1.5 7B 推理）：
  - 模型推理算法层：LLaVA-v1.5 结构不变，在前向过程中插入三个剪枝阶段。**浅层（layer 1-8）**：layer 1 cross-attention 合并到随机令牌（N_text×576 → N_text×1 attention），layer 2+ cross-attention 矩阵全零（非可视），visual self-attention mask 全零，仅 FFN 处理 H_v。此时每个 visual token 仅经历 W_up → SiLU → W_down 的 FFN 变换，无 attention 交互。**中层（layer 9+）**：在每个 layer 末尾计算最后一个 text token 对每个 visual token 的 cross-attention output 影响（逐个 mask → 重算 O'_masked → cosine+L2），第一次 cosine<0.995 时确定为过滤层。过滤层保留 L2≥0.2 的 visual tokens（平均 10.3 个），其余丢弃。后续层仅对这 10.3 个关键 tokens 做 attention。**深层（平均 layer 23.9 起）**：检测到连续两层无 influence → ℓ_exit，此后 H_v 为空，仅剩 H_t 做纯语言 causal self-attention + FFN 直到生成完成。
  - 系统框架层：在 LLaVA 推理代码中插入 `cli_pruning.py`（位于 GitHub repo 的 `llava/` 目录），对 HuggingFace LLaMA 2 模型的每层 forward 进行 hook 修改：① 修改 attention mask（cross-attention 合并/mask），② 注入 token influence 计算逻辑，③ 动态丢弃 token 并调整 KV cache。配置参数：`shallow_mid_layer`（浅层/中层分界）、`layer_threshold`（cosine threshold）、`tokens_threshold`（L2 threshold）。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：使用标准 PyTorch attention 算子，通过减少参与计算的 token 数量实现加速（减少 softmax 计算和 matmul 中的 N_v 维度），无自定义 kernel。
  - 硬件架构层：论文未明确说明。

  设计思路核心：**用 influence（对 attention output 的实际改变）替代 attention weights（softmax 概率分布）作为 token 重要性度量**。这是因为 attention weights 受 attention sink 效应污染（高 attention 不代表高信息贡献），且 softmax 归一化使得权重分布分散，难以精准定位关键 token。Influence-based 方法直接在值空间（V × attention output）操作，捕获了 token 对残差流状态的实际扰动，从而实现了比 attention-based 方法（FastV/SparseVLM/PyramidDrop）高得多的压缩精度。
