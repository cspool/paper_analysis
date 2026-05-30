## Shell Router for MoA Orchestration (MoA 编排的 Shell 路由器)

术语是什么？
Faster-MoA 在 SGLang 标准 PD router + PE/DE 引擎之外实现的外层编排路由器。所有 agent 请求先到达 Shell Router，由其根据 agent 依赖关系进行分发和编排。Shell Router 执行四步：(1) Dependency identification——独立 agent 请求直接转发 native PD router；(2) Dependent requests handling——将依赖 agent 的 prompt 按前驱输出槽分割，立即启动前缀 prefill，并监控第一个依赖 agent 的 APC；(3) Incremental prefilling loop——周期性从 APC fetch chunk → append → /prefill_only update（基于 KV cache 复用）；(4) Forward prefill-done requests——所有槽填满后转发 /generate 请求。

从系统架构角度拆解：
Shell Router 的内部状态机和数据流：

```
Shell Router State per-dependent-request:

State: PREFIX_PREFILL
  → 发出 /prefill_only(prefix_segment)
  → 等待 PE 确认 prefix KV cached
  → 过渡到 FIRST_SLOT_WAITING

State: FIRST_SLOT_WAITING (slot k)
  → 周期性 poll APC of dependent agent k
  → 收到 chunk 时:
    → 发出 /prefill_only(chunk, base_offset=prev_end)
    → 检查 slot k 是否完整 (agent k decode 完成)
  → slot 完成 → k++ → 若 k <= total_slots: 回到 FIRST_SLOT_WAITING
  → 若 k > total_slots → 过渡到 READY

State: READY
  → 所有 slots prefilled
  → 转发 /generate 到 native PD router
  → 清理状态
```

Shell Router 不参与标准 PD pipeline 的 KV block 管理或 decode——它只负责在正确的时机发出正确的 /prefill_only 请求，其余由 SGLang 原生机制处理。

术语一般如何实现？如何使用？
- 作为独立线程/进程运行，与 SGLang native router 通过 API 通信
- 维护依赖图（从树拓扑配置计算）和 APC 引用
- 无状态：每个依赖请求创建临时状态，完成后释放
- 适用于任何有 agent 间数据依赖的多 agent 推理系统

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
