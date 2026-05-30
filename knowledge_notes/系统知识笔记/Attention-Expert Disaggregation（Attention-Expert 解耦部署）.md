## Attention-Expert Disaggregation（Attention-Expert 解耦部署）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention-Expert Disaggregation（Attention-Expert 解耦部署）是 AMoE 系统中 MoE 模型的分层部署策略：将 attention 层和 expert 层部署到不同的 GPU 组。Attention 层使用 Data Parallelism (DP) 部署在部分 GPU 上，expert 层使用 Expert Parallelism (EP) 部署在另一部分 GPU 上，两类 GPU 之间通过异步 P2P 通信而非 barrier all-to-all 连接。这与传统 EP 部署（所有 GPU 同时执行 attention + expert，每层 barrier）有本质区别。

从系统架构角度拆解术语，给出术语在系统架构中运转流程的具体例子。
以 8 GPU 部署 Mixtral 8x7B 为例：
- **AMoE 部署**：4 GPU 用于 attention (DP=4)，4 GPU 用于 experts (EP=4)。Token 先由 attention GPU 处理 attention 层，Dispatcher 将输出按 expert ID 分组发送到 expert GPU；expert GPU 执行 FFN 后，输出回传 attention GPU 形成循环。
- **传统 SGLang EP**：8 GPU 全部同时执行 attention (DP=8) 和 experts (EP=8)，每层 barrier all-to-all。

Disaggregation 的收益：
1. **独立扩展**：当 KV cache 容量成为瓶颈时（长上下文场景），可以为 attention 分配更多 GPU 而不影响 expert GPU 资源分配。
2. **异构硬件适配**：Attention 是 memory/bandwidth-intensive，expert 是 compute-intensive，可以分别为两类 layer 选择不同的硬件（如 PIM for attention, GPU for experts）。
3. **性能隔离**：避免 attention 和 expert 计算在同一个 GPU 上的资源竞争和 interference。

术语一般如何实现？如何使用？
AMoE 默认按照所有 block 的同类 layer 全 colocate 在各 GPU 上（e.g., GPU 持有 Expert 1 的所有 32 个 block 的层）。Attention GPU 额外负责 sampler（将上一轮输出 token embeddings 采样为下一轮输入 token）。论文未实现自动 placement optimizer，但指出 AMoE 的 disaggregated 架构为异构集群中的灵活 placement 优化打开了可能性。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
