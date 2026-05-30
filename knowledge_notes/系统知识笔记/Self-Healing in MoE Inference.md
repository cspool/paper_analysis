## Self-Healing in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Self-Healing（自愈）是 Tarragon 在 MoE 推理系统中引入的故障恢复策略，核心思想是：当 worker 故障时，在**故障域内部本地响应**，令受影响请求快速转移到健康 worker，而不阻塞整个推理 pipeline。Tarragon 设计了双向自愈：
- **AW 侧自愈（容忍 EW 故障）**：AW 端的 REFE 检测到 EW 无响应 → 立即查询 ERT 找替代 EW → 重播 token embeddings + metadata（因 expert 计算是 stateless deterministic 的，重播产生相同结果）。重播请求被标记为高优先级以避免恢复 AW 成为 straggler。
- **EW 侧自愈（容忍 AW 故障）**：EW 不再等待所有 AW 的输入。当收到"足够子集"的健康 AW 的 tokens（或 batch 达到配置的最小大小）时，即开始 expert 计算，将未响应 AW 的 slots 略过。这打破了"每层 EW 必须等所有 AW"的全局同步屏障。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 AW 侧自愈（EW 故障）为例：
1. AW 的 REFE 向 EW_j 发送 token embeddings 后进入等待
2. 超时（默认 probing interval 10ms + 3 次连续超时）→ REFE 判定 EW_j fail-stop
3. REFE 查询 ERT：`alt_ew = ERT[expert_id].get_shadow()` 或 `ERT[expert_id].get_other_primary()`
4. REFE 向 alt_ew 发送相同的 token embeddings（标记 priority flag）
5. alt_ew 将重播请求插入 batch 队首（优先处理）
6. AW 收到 alt_ew 的响应 → 正常继续到 layer ℓ+1
7. 其他未向 EW_j 发请求的 AWs 完全不受影响，继续正常推理

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- AW 侧实现：依赖 REFE 的超时检测 + ERT 动态查询 + RDMA 重传。关键设计是"不等 Orchestrator"——自愈是 AW 本地的即时反应，Orchestrator 仅被告知用于后台 recovery。
- EW 侧实现：EW 维护每个 (layer, expert) 的 token buffer + health AW list。当 `len(received_tokens) >= min(batch_threshold, len(healthy_aws))` 时即触发 expert kernel。batch_threshold 参考 expert kernel 的效率拐点（NVIDIA A100 上约 256-512 tokens 达到 throughput knee point）。
- 效果：EW 故障 stall 从 ~64s（MegaScale-Infer）降至 0.3s；AW 故障 stall 降至 0.4s。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
