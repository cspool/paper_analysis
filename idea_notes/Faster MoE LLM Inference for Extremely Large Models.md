## Faster MoE LLM Inference for Extremely Large Models

- baseline方法是什么？
  - **Baseline: 标准 fine-grained MoE 推理（DeepSeek-V2-Lite na=6, DeepSeek-V3 na=8）**：使用 sglang 等 serving 框架对 fine-grained MoE 模型进行标准推理，保持训练时的 expert 配置不变——所有 ne 个 expert 全部加载，每 token 激活全部 na 个 expert。MoE 层的执行与 FFN 类似，但因稀疏激活导致额外的 expert 参数加载开销，使得批次效应（batch effect）比同参数量 FFN 更弱。
  - 全栈执行例子（以 DeepSeek-V2-Lite na=6, ne=64, concurrency=512, 2×A800 为例）：
    - **模型推理算法层**：DeepSeek-V2-Lite，16B 参数，每层 64 个 routed expert（de=1408）+ 1 个 shared expert（ds=10944）。Router 使用 softmax Top-k（k=6）选择 6 个 expert，输出为 6 个 expert 的加权和 + shared expert。na=6 时激活中间维度 da=8448，共享 expert 占比 54.4%（ds/(ds+da)）。
    - **系统框架层**：sglang v0.4.4 post 1，EP=2（2×A800），continuous batching。每层执行：attention（MLA）→ MoE gate（softmax top-6）→ 6 个 expert FFN + 1 个 shared expert → all-reduce 聚合。低并发时 memory I/O bound（需加载 6 个 expert 参数），高并发时 compute bound。固定 1024 input + 1024 output tokens。
    - **编译框架层**：论文未明确说明。PyTorch + torch.compile (Section 4)，sglang 内置 CUDA kernel。
    - **kernel调度层**：论文未明确说明。sglang 使用 RadixAttention + fused MoE kernel，各 expert 作为独立 GEMM 执行。
    - **硬件架构层**：2× NVIDIA A800 80G PCI-e，EP=2，NVLink 160GB/s（单机多卡）。
  - **Baseline 痛点**：
    1. **MoE 弱化批次效应（核心痛点）**：MoE 虽降低了 FLOPS（仅激活 na/ne 的 expert），但多 token 间很少复用同一 expert，导致增加 token 数时需额外加载更多 expert 参数到显存。这使 MoE 的 arithmetic intensity 上升速度远慢于 FFN，在低/中并发时 memory I/O 瓶颈严重，peak efficiency 更难达到。
    2. **固定 na 浪费容量**：fine-grained MoE 的 na=6-8 远大于 coarse-grained MoE 的 na=2，但并非所有层/所有 token 都需要如此多的 expert。在低并发（memory I/O bound）和高并发（compute bound）场景下，固定 na 的最大值分别导致不必要的 expert 参数加载和不必要的计算开销。
    3. **全量 ne 资源浪费**：fine-grained MoE 经过 load-balancing 训练后 expert 重要性仍有巨大差异（Section 6.2 的奇偶索引实验表明偶数索引 expert 和后半部分 expert 更重要），大量低重要性 expert 占用了显存和参数加载带宽，但对模型质量贡献甚微。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：从效率分析和算法优化两个角度探索 fine-grained MoE 的推理优化空间，提出两种互为补充的优化策略：
    1. **Inference Time Expert Skipping（解决痛点 2）**：按层级别动态减少激活 expert 数 na，通过四元组 (b,h,e,p) 定义跨层的 expert 分配模式。核心洞察来自 Roofline 分析——低并发时减少 na 直接降低参数加载量（memory I/O 改善），高并发时减少 na 降低计算量（FLOPS 减少），中并发时因总量 token 足够多、减少 na 不显著减少总 expert 池而加速有限。策略层面：softmax 路由（V2）使用 descending 策略最优（早期层更多 expert 提供足够上下文，后期层特征已足够丰富可减少 expert）；sigmoid 路由（V3）使用 ascending 策略最优（sigmoid 的极化特性意味着早期需要更多 expert 探索，后期可集中到少数高权重 expert）。
    2. **Pre-Inference Expert Pruning（解决痛点 3）**：在推理前通过 calibration 数据统计各 expert 的激活频率（soft count），仅保留 top-ne' 个最活跃 expert。核心发现：fine-grained MoE 的全随机初始化使得 expert 间无结构相似性（不像 Mixtral 的 expert 从 dense checkpoint 初始化），因此 random 和 naive structured selection 完全失效。但 activation count 方法可有效识别关键 expert——仅去掉 25% expert（ne 64→48）即可获得显存节省，性能退化可控（Avg −2.7%）。
    3. **Expert Parallelism 通信优化分析（解决痛点 1 的结构性缺陷）**：尽管 MoE 单机效率不如 FFN，但在分布式部署中，EP 的通信量仅为 TP 的 na/(nd-1)（典型值 28%）。结合 fine-grained MoE 的 group-constrained routing（限制 token 仅从少数 EP group 选 expert），EP 可在跨节点 InfiniBand (50GB/s) 上实现与节点内 TP NVLink (160GB/s) 相当的延迟，从而通过跨节点 EP 补偿单节点 MoE 效率不足，允许每个节点承载更高 batch、达到更高 arithmetic intensity。

  - 全栈执行例子（与 baseline 同配置，expert skipping na=2 descending 策略 + V2-Lite）：
    - **模型推理算法层**：DeepSeek-V2-Lite，使用 descending expert skipping——首层 na=6 → 逐层递减 → 末层 na=2，平均 na≈3.3。Router 仍使用 softmax top-k，仅 k 值按层变化。相比 baseline na=6（da=8448），平均 da 降至 ≈4646（3.3×1408），shared expert 占比从 54.4% 升至 70.2%（ds/(ds+da)），意味着 baseline 中 45.6% 的 routed expert 计算被部分削减。
    - **系统框架层**：sglang 修改——在 MoE layer 初始化时按 (b,h,e,p) 四元组计算每层 na(l)，forward 时 router 动态使用对应的 top-k 值。执行流程与 baseline 一致，但每层的 expert 加载数量和 FFN 计算量减少。低并发时 benefit 来自更少的 expert 参数 I/O（na=2 时仅加载 2 个 expert + 1 shared vs baseline 的 6+1），高并发时 benefit 来自更少的 GEMM FLOPs。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明（沿用 sglang 内置 MoE kernel）。关键差异在于每个 MoE layer forward 的 expert GEMM 调用次数从 6 降到 2（含 shared 仍为 3 次），reduce 从 6-way 降为 2-way。
    - **硬件架构层**：与 baseline 相同。结果：concurrency=2 时 speedup 1.32×（memory I/O bound 场景，na 减少显著降低参数加载），concurrency=128 时 speedup 仅 1.10×（中并发，总 expert 池已由足够多 token 覆盖），concurrency=768 时 speedup 1.16×（compute bound 场景，na 减少降低 FLOPS）。

    关键性能对比（V2-Lite expert skipping）：
    - na 从 6→2：Avg benchmark 下降 7.5%（best strategy 仅 6%），但 throughput 提升 10-50%（取决于并发度）
    - na 平均 3.3：Avg benchmark 下降 <1%，零性能退化下的最佳吞吐量提升
    - V3 best strategy：throughput +10% 且零性能退化
    - Expert pruning ne 64→48：throughput up to 2.3× at low concurrency，但性能退化不可忽略

    **核心设计洞察**：这篇论文是一种"效率-性能 Pareto 探索"，而非提出单一优化方法。核心贡献在于：(1) 从 Roofline 模型出发系统性分析了 fine-grained MoE 在不同 batch size / concurrency 下的效率特征（记忆 I/O bound → compute bound 的过渡），(2) 量化了 expert skipping 和 pruning 在不同并发度下的效率收益与性能代价的 trade-off，揭示了两种方法在不同场景（低并发 vs 高并发）下的互补性，(3) 发现 sigmoid vs softmax routing 导致截然不同的 skipping 最优策略（ascending vs descending），表明 fine-grained MoE 的推理优化高度 model-dependent，不存在 universal approach。
