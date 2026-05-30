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
