## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- baseline方法是什么？
  Baseline 是现有 MoE 模型的 expert offloading 系统（如 SwapMoE、MoE-Infinity、EdgeMoE），它们依赖观察到的"局部 expert 激活重复"现象来设计 expert cache 策略（LRU/LFU），但缺乏系统性的量化指标来衡量不同模型适合 offloading 的程度。这些系统面临的挑战是：并非所有 MoE 模型都能从 expert offloading 中受益——频繁的 cache miss 导致 CPU offload 或 on-demand loading，显著拖慢推理速度。

  Baseline 全栈执行例子：
  - 算法 Pipeline：token x ∈ R^d → Router Softmax + TopK → 选择 top-k experts → 若 expert 不在 GPU cache → CPU 加载 (memory copy overhead) 或 LRU eviction + PCIe 传输 → FFN 计算 → output。问题：如果模型局部路由一致性低，相邻 token 激活完全不同 expert，cache miss 频繁。
  - 系统框架：现有 expert offloading 系统（如 SwapMoE）采用通用 cache 策略（LRU/LFU），不做模型感知的差异化配置。无论何种 MoE 架构，均使用同一 cache policy。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：GPU（快速但有限内存）+ CPU（大容量慢速内存）异构架构。缺少对特定 MoE 架构特性与硬件适配的系统性理解。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是提出两个理论化的**模型级度量指标**——SRP 和 SCH，量化 MoE 模型的"局部路由一致性"，使开发者能够在部署前评估模型对 expert offloading 的友好程度，并给出优化后的部署配置建议。

  论文方法全栈执行例子：
  - 算法 Pipeline（两阶段）：
    **阶段 1（离线分析）**：对候选 MoE 模型在 22,528 样本语料上收集 router decisions → 统计 expert 在 segment 内的激活频率 f(e,T,p,m) → 计算 SRP（最大化 F1 的最优 threshold α_e^m）→ 计算 SCH（oracle segment cache 在缓存比 ρ 下的 hit rate）→ 生成 SCH-vs-ρ 曲线 → 识别拐点 ρ*（通常为 2× 激活 expert 数）→ 筛选高 SRP 模型部署。
    **阶段 2（部署决策）**：根据 SCH 分析选择模型 → 配置 expert cache size = ρ*×k×expert_size → 运行时用 LRU/LFU cache。

  - 系统框架层面的改进：论文提出的分析表明：（1）**无 shared experts** 的 MoE 模型（如 OLMoE、LLaMA-MoE-v2）有更高局部路由一致性——指导架构设计时避免或少用 shared experts；（2）**domain-specialized experts** 而非 vocabulary-specialized experts 贡献了主要的路由一致性——意味着针对特定领域（如代码、数学）优化时效果更好；（3）**cache size ≈ 2x 激活 expert 数** 是大多数模型的 sweet spot，超出此比例 hit rate 增益递减。这些见解可直接指导 expert offloading 系统在模型选择、缓存配置、领域适配方面的设计，而非对所有模型统一处理。

  - 编译框架：论文未明确说明。

  - Kernel 调度：论文验证了 LRU/LFU cache 与 SCH 的强相关性（m=16 时 Pearson r=90.43/88.70），证明命中率上限由模型路由行为决定。实际 offloading 系统可将 SCH 作为理论命中率上界指导 kernel 层面的 expert 调度。

  - 硬件架构：GPU 内存（A100 80GB）+ CPU（大容量）异构。结论适用于 memory-constrained 边缘设备场景。论文 insight：decoding 阶段 overhead 与局部路由一致性负相关（r≈−0.3），而 prefilling 阶段正相关（r≈0.2），因为 prefilling 瓶颈在 expert-level load balance 而非跨 token 的路由一致性。

  核心创新：将 expert offloading 系统的性能分析从**系统实现层面**（如何优化 cache/cache policy）下沉到**模型路由行为层面**（哪些模型天然适合 offloading），实现了模型设计与部署系统之间的双向指导。论文发现的关键 trade-off——局部路由一致性 vs. 局部负载均衡——解释了为什么一些模型可在保持全局负载均衡的同时获得高局部路由一致性（通过 domain-specialized experts 实现：特定领域上下文中集中激活某些 expert → 高局部一致性；不同领域激活不同 expert → 全局均衡）。
