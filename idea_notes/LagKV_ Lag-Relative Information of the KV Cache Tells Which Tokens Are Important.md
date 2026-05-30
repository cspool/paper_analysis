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
