## Reconfigurable Forwarding Engine (REFE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reconfigurable Forwarding Engine (REFE) 是 Tarragon 中 AW 侧的运行时组件（C++ 扩展 + Python shim，约 16K 行 C++），负责 AW-EW 之间的所有通信协调和故障恢复。对外暴露简单的 `expert_io(expert_id, layer_id, token_embeddings)` API，内部运行非阻塞、事件驱动的执行循环。REFE 的核心职责：(1) 查询 ERT 将 logical expert ID 解析为物理 EW；(2) 通过双 QP（control-plane QP for liveness probe + data-plane QP for token embeddings via GPUDirect RDMA）与 EW 通信；(3) 管理 EW 响应的接收和超时检测；(4) 在检测到 EW 故障时执行 AW 侧自愈（重路由到健康 EW/shadow expert）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
REFE 在 decoding 阶段的执行流程：
1. vLLM compute engine 完成 layer ℓ 的 attention 计算 → 调用 `expert_io(expert_id, ℓ, token_embeddings)`
2. REFE 查询本地 ERT：`physical_ew = ERT[expert_id]`
3. REFE 通过 data-plane QP（GPUDirect RDMA write）将 token embeddings 直接写入目标 EW 的 GPU 显存
4. REFE 通过 control-plane QP 发送 metadata（layer_id, expert_id, request_id, token_position）
5. REFE 进入等待状态（超时窗口内），监听 data-plane QP 的响应
6. 若超时无响应：REFE 通过 control-plane QP 发 explicit probe → 确认 failure → 查询 ERT 找替代 EW → 重播请求（带优先级标记）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C++ 扩展嵌入 vLLM Python 进程，使用 libibverbs（RDMA verbs API）管理 QP 和 RDMA 操作。
- 关键实现细节：(a) 非阻塞事件循环基于 epoll/RDMA completion queue (CQ) 轮询；(b) 双 QP 隔离控制流和数据流，避免 liveness probe 被大数据传输阻塞；(c) 使用 GPUDirect RDMA 实现 zero-copy GPU-to-GPU 数据传输，无需 CPU 中转。
- 超时配置：默认 probing interval 10ms，连续 3 次超时判定为 fail-stop。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
