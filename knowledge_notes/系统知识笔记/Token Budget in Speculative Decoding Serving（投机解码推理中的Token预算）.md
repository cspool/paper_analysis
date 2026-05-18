## Token Budget in Speculative Decoding Serving（投机解码推理中的Token预算）

术语是什么？通过联网搜索让回答具体和精准。
Token Budget 在 AdaServe 中表示当前硬件在一次 target LLM verification 中可承载的最大 token 数。受 GPU 计算能力、当前 batch 大小、KV cache 可用空间和 draft model 计算资源约束。Token budget 是 SLO-customized token tree 构造问题的核心约束：所有请求 selected token tree 节点数之和不能超过此 budget。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Token budget 调度决策流程：
1. **Budget 设定**：每轮根据活跃请求数、GPU memory 和 KV cache 状态确定 B
2. **SLO-customized allocation**：先为各请求分配满足 SLO 所需最少节点（Phase 1）
3. **Remaining budget**：B_remaining = B - Σ|Phase1_nodes|
4. **Throughput-optimized allocation**：剩余 budget 按全局 path probability 分配（Phase 2）
5. **动态调节**：活跃请求多→B缩小→d/w调小；活跃请求少→B增大→d/w调大

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
由 AdaServe scheduler 维护为运行时变量。过小 B 限制 speculation 加速空间，过大 B 增加 draft model overhead 和低质量候选 token 比例。

涉及论文标题：
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
