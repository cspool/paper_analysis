## Making MoE-based LLM Inference Resilient with Tarragon

- 属于Serving调度的实现是什么？实验比较什么？
  TARRAGON 是一个具备故障恢复能力的 MoE 推理框架（约 16K 行 C++ + 2K 行 Python）。它在解耦的 Attention Worker (AW) 与 Expert Worker (EW) 部署之上构建了三个核心机制：
  1. **可重构数据通路（REFE + ERT）**：Reconfigurable Forwarding Engine (REFE) 是 AW 侧的运行时，通过 Expert Routing Table (ERT) 将逻辑 expert ID 动态映射到物理 EW/GPU，实现故障时的请求重路由，避免全局重启。
  2. **自愈机制（Self-Healing）**：AW 侧通过超时检测 + 重播到健康 EW/shadow expert 来容忍 EW 故障；EW 侧通过部分输入批处理（不等所有 AW）来容忍 AW 故障。
  3. **后台容量恢复（Background Provisioning）**：Orchestrator 在后台启动替换 AW/EW 并集成到在线推理 pipeline 中。
  实验比较了 TARRAGON 与 MegaScale-Infer（解耦基线）和 vLLM（单体基线）在故障场景下的 stall 时间、稳态下的 TTFT/TBT/吞吐量、以及 KV cache checkpointing 的开销。

- 硬件平台是什么，配置是什么。
  Google Cloud (GCP) A3 Ultra 节点，每节点：224 vCPUs, 3 TB RAM, 8x NVIDIA H200 GPUs (141 GB 显存), 8x 400 Gbps ConnectX-7 RDMA NICs（支持 GPUDirect RDMA），节点内 NVLink 3.6 Tbps。实验使用 3 个节点：AWs 占 1 节点 (8 GPUs)，EWs 占 1 节点 (8 GPUs)，checkpoint store 占 1 节点。软件环境：Ubuntu 22.04, Linux 5.15, CUDA 12.8 (driver 580), PyTorch 2.6.0。

- 开源Serving框架是什么。修改了什么。
  基座框架：**vLLM** 作为 AW 侧的 compute engine（处理 prefill 和 decoding 的 attention 计算）。EW 侧从零用 C++ 编写（libtorch for expert computation, libibverbs for RDMA）。Orchestrator 和 Checkpoint Store 均为独立 C++ 服务。
  具体修改：
  - 在 vLLM 上层增加了 REFE（C++ 扩展 + Python shim），负责 AW-EW 间的 RDMA 通信、ERT 查询和请求分发。
  - 实现了双 QP 设计：control-plane QP 用于存活探测和自愈元数据，data-plane QP 用于 token embedding 批量传输（GPUDirect RDMA）。
  - 实现了 ERT 机制：将 expert identity 与 expert location 解耦，允许动态重映射。
  - 实现了 AW 侧自愈：超时后重路由到健康 EW/shadow expert。
  - 实现了 EW 侧自愈：收到足够 AW 的输入即开始 expert 计算，不等所有 AW。
  - 实现了 shadow expert：在 EW GPU 显存中预加载但保持 inactive 的 expert 副本。
  - 实现了异步增量 KV cache checkpointing：利用 AW-EW 通信间隙进行 one-sided RDMA write。
  - 实现了 per-request KV cache restoration：从 checkpoint store 通过 GPUDirect RDMA 直接注入到替代 AW 的 GPU 显存。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文声明将开源（"We will open-source TARRAGON"），截止论文阅读时尚未公开链接。
  
  **TARRAGON 推理全流程（以 decoding 阶段为例）**：
  1. **请求接入**：用户请求通过单一 cluster gateway 到达，被分发到某个 AW。
  2. **Attention 计算**：AW 的 compute engine（vLLM）对当前 layer ℓ 执行 attention 计算，更新 KV cache，产生 token embeddings。
  3. **Gating + 分发**：compute engine 调用 `expert_io(expert_id, layer_id, token_embeddings)` API。REFE 查询 ERT 将 logical expert ID 解析为物理 EW，通过 data-plane QP（GPUDirect RDMA）将 token embeddings 直接写入目标 EW 的 GPU 显存。
  4. **Expert 计算**：EW 收到来自多个 AW 的 tokens 后，按 layer ℓ + expert ID 聚合为 batch，调用 libtorch 执行 expert FFN 前向计算。当收到足够 AW 的输入（或达到最小 batch size）即开始计算。
  5. **结果返回**：EW 将 expert 输出通过 RDMA 写回 AW 的 GPU 显存。AW aggregate 所有 expert 输出（加权求和），进入 layer ℓ+1。
  6. **KV Cache Checkpointing**：在 AW 执行 attention 的间隙（AW-EW link idle 时），REFE 异步将新增的 KV cache segment 通过 one-sided RDMA write 写入 checkpoint store。
  7. **故障处理（AW 故障）**：Orchestrator 检测到 AW 故障 → 从 checkpoint store 恢复该 AW 上所有请求的 KV cache 到健康 AW → 健康 AW 从 committed token 继续 decoding。
  8. **故障处理（EW 故障）**：AW 侧 REFE 探测到 EW 无响应 → 查询 ERT 获取替代 EW（含 shadow expert）→ 重播 token embeddings 到替代 EW → 无需等待 orchestrator。
