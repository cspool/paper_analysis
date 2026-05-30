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
