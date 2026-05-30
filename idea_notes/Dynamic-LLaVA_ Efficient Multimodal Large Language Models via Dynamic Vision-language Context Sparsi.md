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
