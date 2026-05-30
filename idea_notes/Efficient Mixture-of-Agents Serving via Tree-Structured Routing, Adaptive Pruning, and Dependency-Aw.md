## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- baseline方法是什么？
  **All-to-All 全连接 MoA**：现有 MoA 系统采用多层 all-to-all agent 连接拓扑——相邻层间的 agent 全连接，每个 agent 接收上一层所有 agent 的输出，经 aggregator 融合生成最终答案。硬件部署上，MoA serving 缺乏对 agent 间复杂数据依赖和异构延迟的支持：前驱 agent 解码与后继 agent prefilling 被视作严格串行，无 overlap 机会；且每层延迟由最慢 agent 决定（`T_ℓ^{all} = max_i t_{ℓ,i}`），导致 GPU 利用率低（pipeline stall time）。
  
  **全栈执行例子**：
  - **算法pipeline层**: 用户 query → Layer 1 所有 N 个 proposer agents（如 9 个，各用不同 LLM 骨干）并行生成答案 → Layer 1 所有输出拼接为 Layer 2 每个 aggregator agent 的输入（全连接，输入长度 = 原始 prompt + N × 输出长度）→ Layer 2 所有 aggregator 并行生成 → ... → 最终 aggregator 融合所有 Layer L-1 输出为最终答案。全连接拓扑导致：(a) 冗余连接传递无用信息；(b) 大模型（32B）与小模型（4B）同时启动但大模型慢得多，小模型完成后 GPU idle 等待；(c) aggregator 输入 context 极长（9×输出长度）。
  - **系统框架层**: Naive PD disaggregation 下，dependent agent（如 Layer 2 aggregator）必须等待所有前驱解码完成并收集输出后才能开始 prefill → 前驱解码期间后继 GPU 完全空闲。
  - **编译框架层**: 论文未明确说明（基于 SGLang/vLLM 原生 PD pipeline，无 agent-aware 优化）。
  - **Kernel调度层**: 论文未明确说明。
  - **硬件架构层**: All-to-all topology 下最慢 agent（如 32B 模型）决定层延迟 → 该层其他已完成 agent 的 GPU SM 空闲等待 barrier 同步 → 低硬件利用率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Faster-MoA** 通过三个协同设计解决上述缺陷：
  
  1. **层次化树状拓扑替换 All-to-All**：以 9-3-1 三层树结构替换全连接。Layer 1 的 9 个 agents 分为 3 个 clusters，Layer 2 的每个 agent 仅连接其对应 cluster（而非全部 9 个），Layer 3 的 root aggregator 连接所有 Layer 2 agents。效果：(a) 减少冗余连接——每个后继仅处理局部 cluster 输出而非全部；(b) 输入 context 长度从 9× 输出降至 3× 输出，prefill 成本线性下降；(c) 子树间可独立并发，不因跨 cluster 慢 agent 而被阻塞（`T_ℓ^{tree} ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c`）。

  2. **语义引导动态 Early-Exit**：利用 FrobCosSim + 几何平均置信度计算早退概率 Q。当小 agent（4B/8B）输出语义一致且高置信时，以概率 Q 终止未完成的大 agent（32B），跳过其等待时间。这解决了 all-to-all 中"必须等最慢 agent"的问题，且 Q 自适应任务难度——难任务（IFBench）大模型被调用更多，简任务（GSM8K）大模型较少被调用。

  3. **依赖感知增量 Prefilling**：Shell Router 将依赖 agent 的 prompt 按前驱 agent 输出槽分割。前缀段无依赖立即 prefill；前驱解码出的 token chunk 流式 append 到 APC → shell router 周期性 fetch → 增量 /prefill_only update（复用已驻留在 HBM 的 prefix KV，近 100% cache hit）→ prefilling 计算被前驱 decoding 重叠隐藏。解决了 naive PD disaggregation 中"依赖 agent 必须等前驱全部完成后才开始 prefill"的串行瓶颈。

  **全栈执行例子（与 Baseline 对比）**：
  - **算法pipeline层**: 用户 query → Layer 1：9 agents 分为 3 clusters 并行执行，每个 cluster 内含 4B/8B/32B 三模型 → 完成 4B 和 8B 后 → 计算 Q = √(C̄·B)^(1/τ) → 若 Q 足够高，概率性终止 32B（不再等待）→ Layer 2：3 agents 各继承自己的 cluster 输出（仅 3 个，非全部 9 个）→ Layer 3：root aggregator 融合全部 Layer 2 输出为最终答案。
  - **系统框架层**: Shell router 接收到 Layer 2 agent 的请求 → 识别依赖 Layer 1 特定 cluster 输出 → 将 prompt 按前驱输出槽分割 → 前缀段立即发 /prefill_only → 监控 APC 中前驱 agent 的 decode chunk → 增量 /prefill_only（复用 prefix KV）→ decode 完成时 prefilling 已完成（被 decode 时间隐藏）→ 转发 /generate → 前驱解码与后继 prefilling 时间重叠。
  - **编译框架层**: 论文未明确说明。基于 SGLang v0.5.3 原生 xgrammar/torch.compile 机制，增量 prefilling 通过 KV cache reuse 机制实现 token 追加而非完整重计算。
  - **Kernel调度层**: GPU PE 在执行 /prefill_only 时维护 prefix KV blocks 在 HBM，增量 token 通过 FlashAttention 仅计算新 token 的 KV 并追加，prefix 部分直接从 HBM 读取（near 100% cache hit）。
  - **硬件架构层**: 6×H200 GPU，3 组 PE/DE pair。不同 size 模型的 decode 和 prefill 在不同 GPU 上并行，空闲 SM 被增量 prefill 任务利用。最终效果：~90% E2E 延迟减少，准确率 ±1%。
