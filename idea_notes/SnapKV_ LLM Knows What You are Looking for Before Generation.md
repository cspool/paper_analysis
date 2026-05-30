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
