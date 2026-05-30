## PME (Parallelism-Memory Efficiency / 并行-内存效率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PME（Parallelism-Memory Efficiency）是 MoE-Lens 提出的量化指标，衡量一个推理序列将 CPU memory capacity 转化为 GPU 端并行计算 token 数的效率。核心洞察：prefill 阶段所有 prompt tokens 可同时处理（高并行），decode 阶段每次仅生成一个 token（低并行），且 KV cache 随 decode 步数增长。PME 定义为：

$$PME = \frac{\sum_{\text{gen. steps}} \text{Parallel Tokens}}{\sum_{\text{gen. steps}} \text{Sequence KV Cache Size}} = \frac{p+g}{\sum_{j=0}^{g} (p+j)} = \frac{2(p+g)}{(2p+g)g}$$

其中 p = prompt length, g = generation length。分子是所有 generation step 中可并行处理的 token 总数（prefill p + 每 decode step 1 × g steps）。分母是序列在其整个生命周期中占用的累积 KV cache memory。PME 直接代入 Stage 1 throughput 上界：$T_{max} = \min(\frac{PME \cdot M}{\delta}, T_{GPU})$（M = KV cache 总容量，δ = model weight transfer 时间，$T_{GPU}$ = GPU max throughput）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PME 在 MoE-Lens 调度决策中的作用（以 MTBench p=98, g=32 为例）：

1. PME = 2(98+32)/((2×98+32)×32) ≈ 0.056
2. 70GB KV cache (~280K tokens) → $T_{max} ≈ 0.056 × 280K / 4.8s ≈ 3267 tok/s$
3. $T_{GPU}$ ≈ 3600 tok/s → 3267 < 3600 → system is memory-capacity bound
4. Scheduler 据此确保 max parallel tokens = q(p+g) from Equation 8
5. 若 g=256, PME ≈ 0.015 → $T_{max}$ 更低 → memory bottleneck 更严重

PME 随 p/g ratio 变化：prompt 比例越高（如 RAG 数据集 p=926, g=128），PME 越高，GPU utilization 越高。Prefill-decode overlapping 通过 Equation 7 放大有效 KV cache 容量，等效提升 PME。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **使用方式**：在部署前用 PME 估算给定 hardware + workload 下的 theoretical max throughput，判断 memory-bound vs GPU-bound，指导 KV cache size 配置。
- **局限性**：PME 假设 uniform prompt/generation length，实际 variance 由 Stage 2 Model 修正。

涉及论文标题：
- MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
