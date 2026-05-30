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
