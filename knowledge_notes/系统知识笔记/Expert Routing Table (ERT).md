## Expert Routing Table (ERT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Routing Table (ERT) 是 Tarragon 中将 logical expert identity 与 physical expert location（EW/GPU）解耦的关键数据结构。每个 AW 独立维护一份 ERT，由 Orchestrator 动态更新。ERT 映射为：`expert_id → [primary_ew, shadow_ew, ...]`，即每个 logical expert 可对应多个候选物理位置。这使得路由调整变成一个本地化的重映射操作（更新表项），而非系统级的全局恢复。这一设计打破了 MegaScale-Infer 等系统中 expert 与 GPU 的静态绑定，是 Tarragon 实现 worker 级故障域隔离的基础。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ERT 的生命周期：
1. **初始化**：Orchestrator 根据 expert placement 策略生成初始 ERT，广播给所有 AW
2. **正常推理**：AW 的 REFE 查询 `ERT[expert_id]` 获取 primary_ew，发起 token embedding 传输
3. **EW 故障**：AW 探测到 EW 故障 → 从 ERT 获取 `shadow_ew` → 激活 shadow expert → 重路由。同时通知 Orchestrator
4. **Orchestrator 响应**：后台 provisioning 新 EW，新 EW 就绪后 Orchestrator 更新所有 AW 的 ERT（添加新 EW 作为新的 primary 或 shadow）
5. **新 AW 加入**：新 AW 从 Orchestrator 获取当前 ERT，建立到所有 EW 的 datapath

ERT 的设计使路由更新与推理 pipeline 完全解耦：路由表更新是 control-plane 操作（无需停止数据流），请求继续由 REFE 根据最新 ERT 转发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：内存中的 hash table（expert_id → vector of EW endpoints），由 Orchestrator 通过 HTTP/gRPC 推送更新。
- 一致性模型：最终一致性——各 AW 的 ERT 更新存在短暂的时间窗口差异，但在 AW 侧自愈（超时重试 + 重新查询 ERT）的配合下不影响正确性。
- 扩展场景：ERT 也可用于非故障的 elastic scaling——当新增 EWs 就绪时，更新 ERT 即可将后续请求路由到新 EW，无需推理中断。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
