## Expert Deduplication in MoE Serving（MoE服务中的专家去重）

术语是什么？
Expert Deduplication 是 MoEsaic 提出的在 multi-tenant MoE serving 场景下，通过 tensor-level hash 检测跨不同 model instance 的相同 expert，使多个 client 共享同一份 GPU 显存拷贝的技术。核心操作：对每个 expert 中的每个 tensor 计算 128-bit hash digest，存入 in-memory dictionary；后续加载的 expert 计算 hash 后查 dictionary——命中则引用已有 tensor（共享显存），未命中则分配新 GPU 显存。去重不修改任何 expert 权重，不改变模型精度或行为。

从系统架构角度拆解术语：
在 MoEsaic 的 vLLM 集成中（8×A100 40GB），Expert Deduplication 的系统流程为：

1. **张量级 Hash 计算**：模型从文件系统加载时，对每个 expert 的每个 tensor（如 gate_proj.weight, up_proj.weight, down_proj.weight）计算 128-bit hash digest（如 SHA-512/128 截断）。由于 vLLM 中 in-memory 表示与 in-file 表示不同（多个 in-file tensor 对应一个 in-memory tensor），hash 需在所有 in-file segment 完全填充到 in-memory tensor 后才计算。
2. **In-Memory Dictionary 查找**：每个 expert 的 hash 存入全局 dictionary。后续加载的 expert 计算 hash 后查询 —— 命中即引用已有 tensor 指针，不分配新显存。
3. **Expert Population Tracking**：每个 expert 独立表示（Independent Expert Representation），跟踪其 tensor 分配状态。expert 完全填充后标记为"可去重候选"。
4. **去重聚合**：初始化完成后，将共享相同底层 tensor 的 expert 合并为单一 nn.Parameter（Merged Expert Representation），供后续批处理使用。
5. **Lazy Allocation 配套**：初始化时用 tiny pseudo experts 占位（几乎零显存），加载参数时才逐步扩容。去重后的峰值内存 = 去重后模型大小 + 当前正在加载（尚未去重）的一个 expert。

以 Mixtral-8x7B 为例：每 expert 约 14 GB，7 shared experts + 1 unique → 14 model instances。去重后参数仅需 ~294 GB（baseline dedicated instances 需 ~224 GB 仅支持 2 instances），可服务 7× 更多变体。

术语一般如何实现？如何使用？
实现方式：
- **128-bit Hash**：使用加密级 hash 函数（如 BLAKE3 或 SHA-512/128）计算每个 tensor 的字节级 digest。碰撞概率极低（2^-128），适合生产环境使用。
- **离线 Hash 预计算**：MoEsaic 论文指出 hash 可在离线阶段预计算并与模型文件一同存储（hash-tensor mapping），消除在线 hash 计算开销。
- **Tensor-Parallel 扩展**：在 TP 模式下，expert shard 在 shard 级别去重。每个 Ray worker 负责加载指定 GPU 的 expert shard，shard 去重逻辑与完整 expert 相同。
- vLLM RFC #9203 中提出了 Shared Mixture of Experts 特性的上游集成方向（https://github.com/vllm-project/vllm/issues/9203），包括 shared-mixtral.py 和 xMoE interface。
- 主要挑战：(1) 去重粒度——per-tensor 级别（而非 per-expert 或 per-layer），需要每个 expert 独立表示为 nn.Parameter；(2) 初始化时 loading 时间长——在线计算 hash 导致首模型加载慢 3×（33s vs 11s for Mixtral-4x1B），离线 hash 可解决。

涉及论文标题：
- MoEsaic: Shared Mixture of Experts
