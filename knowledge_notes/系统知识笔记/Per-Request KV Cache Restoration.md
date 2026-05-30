## Per-Request KV Cache Restoration

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Request KV Cache Restoration（逐请求 KV Cache 恢复）是 Tarragon 在 AW 故障后恢复受影响请求的机制，与"全局重放"形成对比。当 AW 故障时，Orchestrator 识别该 AW 上的所有活跃请求及其最后 committed token，将每个请求分发到健康的替代 AW。对于每个请求，checkpoint store 通过 GPUDirect one-sided RDMA write 将完整的 KV cache segments 直接注入替代 AW 的 GPU 显存，替代 AW 从 committed token+1 继续 decoding。与重放 prefill+decoding 相比，恢复延迟降低最高 1800×，传输流量降低 8×，GPU 重计算消除。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
完整恢复流程：
1. **故障检测**：Orchestrator 检测到 AW_i 故障（或 REFE 上报）
2. **请求识别**：Orchestrator 查询 checkpoint store，获取 AW_i 上所有活跃请求的 `{request_id → latest_committed_token}` 映射
3. **请求重分配**：Orchestrator 将请求 round-robin 分发到健康 AWs
4. **逐请求恢复**（每个请求并行执行）：
   a. Checkpoint store → 替代 AW：committed_token_id, kv_state_size
   b. 替代 AW 分配 KV cache region → 返回 offset
   c. Checkpoint store 通过 GPUDirect RDMA write 逐层逐 segment 注入 KV cache
   d. Checkpoint store → 替代 AW：HTTP restore_complete 确认
5. **恢复完成**：替代 AW 从 committed_token+1 开始 decoding，对外表现为短暂的 token 间隔（~0.4s）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- isolation：每个请求在 AW 上拥有独立的 KV cache region，恢复可完全并行，不干扰 AW 上其他活跃请求。
- 与 Mooncake 等分布式 KV cache store 的关系：正交——Tarragon 的 checkpoint store 可与 Mooncake 结合，利用其 multi-tier caching (VRAM/DRAM/NVMe) 提升 checkpoint 容量。
- 对比 Sequential Replay：恢复时间从 O(i·L) 降至 O(1)（i 为已解码 token 数，L 为层数）。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
