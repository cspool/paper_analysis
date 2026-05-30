## Shadow Expert

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shadow Expert（影子专家）是 Tarragon 在 EW GPU 显存中预加载但保持**inactive**的 expert 权重副本。Shadow expert 包含与 primary expert 相同的模型权重和计算 kernel，在 primary expert 所在 EW 故障时可以被立即激活，无需从存储重新加载权重（典型耗时数百毫秒到秒级）。在无故障时，shadow expert 仅占用 GPU 显存（不消耗 compute/SM 资源），因此不引入 kernel-level interference（单 expert 执行延迟与 "shadow loaded + primary active" 配置下相同）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Shadow expert 的工作流程：
1. **初始化**：在 EW GPU 显存中分配 shadow expert 权重空间，从 checkpoint/存储预加载权重。Shadow expert 的 CUDA kernel 和 metadata 就绪但不被任何 CUDA stream 调度。
2. **故障发生**：Orchestrator/REFE 检测到 primary EW 故障。
3. **激活**：Orchestrator 通知持有该 expert shadow 副本的 EW → 将 shadow expert 标记为 active → ERT 更新（该 expert ID 现在指向 shadow host EW）。
4. **接管**：AW 后续对该 expert 的请求被 REFE 路由到 shadow host EW。Shadow 已就绪，无需等待权重加载。
5. **后台恢复**：Orchestrator 在后台 provision 新 EW，加载完整 expert set 后，该 expert 从 shadow 切回新 primary（通过 ERT 更新）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- GPU 显存开销：单个 DeepSeek-R1 expert 约 2.5 GB（141B 参数 / 约 56 MoE layers），对于 A100 (40-80 GB) 或 H200 (141 GB) 可轻松容纳多个 active + shadow experts。
- 控制：shadow expert 的 inactive 状态通过不将其 CUDA kernel 提交到任何 stream 实现（或通过 CUDA MPS 的 resource limiting）。GPU 显存分配使用 `cudaMalloc` 但 CUDA kernel 不 launch。
- 激活延迟：~microseconds（仅需更新 ERT 和设置 expert active flag），vs 从 SSD 加载权重的秒级延迟。
- 局限：仅提供容量冗余而非算力冗余——shadow expert 激活后 share host GPU 的 compute resource，若 shadow host 原本已满负荷可能产生排队延迟。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
