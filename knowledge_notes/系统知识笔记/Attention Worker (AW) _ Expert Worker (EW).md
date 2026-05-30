## Attention Worker (AW) / Expert Worker (EW)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Worker (AW) 和 Expert Worker (EW) 是 decoupled attention-expert deployment 中两类功能不同的 GPU worker 进程。AW 托管 transformer 的 attention 模块（self-attention, KV cache 管理, gating network），负责与用户请求交互和维护 per-request 状态。EW 托管 expert FFN 模块（MoE 层中的 sparse expert 计算），是 stateless 的纯函数计算单元。AW 通常按 data parallelism 扩展（每个 AW 服务不相交的请求子集），EW 按 expert parallelism 扩展（partition experts across GPUs）。AW 是 stateful（持有 KV cache，prefill 初始化 + decode 增量更新），EW 是 stateless（无 per-request 持久状态，仅依赖当前输入 token embeddings 和固定权重）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AW 和 EW 在 Tarragon 中的交互（以 Mixtral-8×7B, 8 AWs + 8 EWs 为例）：
1. **AW 角色**：接收用户请求 → prefill 阶段处理 prompt 构建 KV cache → decode 阶段逐 token 生成 → 每层 attention 后通过 gating network 选 top-k experts → 将 token embeddings 发往对应 EWs → 收集 expert outputs 加权求和 → 预测 next token。
2. **EW 角色**：接收来自多个 AW 的 tokens → 按 (layer, expert_id) 聚合为 batch → 执行 expert FFN 前向 → 将结果返回各 AW。
3. **故障域分离**：AW 故障只影响该 AW 上的请求（可通过 KV cache checkpointing 恢复），EW 故障只影响 expert 容量（可通过 shadow expert 接管）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- AW 实现：基于 vLLM compute engine（Python/C++），处理 attention + KV cache + scheduling。Tarragon 增加 REFE C++ 扩展。
- EW 实现：Tarragon 中从零用 C++ 编写（libtorch + libibverbs），专注于高性能 expert FFN 执行和 RDMA 通信。
- 通信模式：many-to-many——每个 AW 可能与多个 EW 通信（取决于 gating 选择），不同于标准 NCCL all-to-all 的对称模式。Tarragon 使用点对点 RDMA RC (Reliable Connection) 双 QP。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
