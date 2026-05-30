## Batched MoE Inference（批量MoE推理）

术语是什么？
Batched MoE Inference 是指 MoE 模型推理时同时处理多个请求/序列（batch size ≥ 2），以提升系统吞吐量的技术。在 LLM serving 中，batching 通过两种方式提升吞吐：(1) 将多个请求序列合并为单批，摊薄固定开销（如 kernel launch 一次而非每请求一次）；(2) 并行处理多个序列更好地利用 GPU 硬件资源。在 MoE 推理中，batching 的特殊性在于每个 batch 中各个 token 的 gating 选择可能不同，导致 batch 越大、同时激活的不同 experts 越多，从而放大 host-GPU 通信开销。

从系统架构角度拆解术语：
在 Diff-MoE 的场景中（H200 GPU, Switch-Base, XSum），batched MoE 推理的工作流：

1. **请求级 Batching**：多个独立请求（如多条 XSum 摘要任务）被分组到一个 batch。每个请求处于解码的不同迭代步（因为请求可能在不同时刻到达），形成 continuous batching。
2. **Token Batch**：batch_size=64 时，每组 decoding iteration 处理 64 个 token（每个请求贡献 1 个 token）。64 个 token 经过 attention layer 后，独立通过 gating network 选出各自的 top-1 expert。
3. **Expert Fan-out**：64 个 token 可能分到 30-34 个不同 experts（batch 越大，expert 多样性越高）。这放大了需加载的 expert 数量——batch=1 时仅 1 个 expert，batch=64 时约 34 个，通信量增长 34×。
4. **计算稳定**：GPU 并行计算能力足够应对 64 个 token 的 FFN matmul，计算时间仅从 ~1.58 ms (batch=1) 增长到 ~2.45 ms (batch=16)，仅 1.55×。但通信时间从 2.20 ms 增长到 14.37 ms，6.53×。
5. **瓶颈转移**：batch=1 时计算（1.58 ms）与通信（2.20 ms）相对均衡；batch=16 时通信（14.37ms/总 14.78ms = 97.19%）完全主导。

Diff-MoE 对此的应对：(1) HPC/MPC 三级缓存减少需传输的 experts 数量；(2) per-layer 缓存隔离防止大 batch 下跨层缓存竞争；(3) predictor 预取下一层 experts 将传输与计算重叠。

术语一般如何实现？如何使用？
主流实现方式：
- **Continuous Batching**（如 vLLM、TGI）：动态将新请求加入运行中的 batch，避免等待所有请求完成后再组批。在 MoE serving 中进一步需要 expert-aware scheduling。
- **Prefill-Decode Disaggregation**（如 Splitwise、Sarathi-Serve）：将 prefill 和 decode 阶段分到不同 GPU 实例，针对各阶段的 batch 特性优化。
- **Multi-Batch Pipeline**（如 Klotski）：将多个 batch 流水线化通过 MoE 层，利用 batch 间空闲时段预取/计算。
- 在 Diff-MoE 中，batching 是透明的——框架处理任意 batch size，缓存层自动调整 LPC 大小以容纳当前 batch 中需加载的全部 experts。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
