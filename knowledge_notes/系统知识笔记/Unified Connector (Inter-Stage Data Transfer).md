## Unified Connector (Inter-Stage Data Transfer)

术语是什么？

Unified Connector 是 vLLM-Omni 中负责 stage 间中间数据传输的抽象层。灵感来自 vLLM 的 KV cache transfer 机制（用于 prefill-decode disaggregation），但 Unified Connector 将其泛化以处理更广泛的数据对象类型：embeddings、hidden states、audio tensors、image tensors、latents 等。它将 transport 从 model logic 中解耦，通过统一 put/get interface 支持多种传输后端。

从系统架构角度拆解术语：

Unified Connector 的传输层次：
```
┌─────────────────────────────────────────┐
│         Model Logic (Python)             │
│  transform_fn: store(key, tensor)        │
├─────────────────────────────────────────┤
│       Connector Interface (C++)          │
│  put(key, tensor) / get(key) → tensor    │
├─────────────────────────────────────────┤
│      Transport Backends                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Inline CQ │ │ Shared   │ │ Mooncake │ │
│  │(small    │ │ Memory   │ │ (TCP/    │ │
│  │ payload) │ │(large    │ │ RDMA)    │ │
│  └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────┘
```

单节点传输流程（以 Thinker → Talker shared memory 为例）：
```
Thinker stage 完成 text generation:
  Output Processor:
    1. 调用 Thinker2Talker transform function → 生成 Talker input embeddings
    2. 将 embeddings 存到 per-request intermediate data dict (CPU memory)
    3. Unified Connector: 检测 destination device (Talker GPU)
       → allocation shared memory segment
       → memcpy CPU → shared memory
       → 通过 inline control queue 发送 metadata (key, shape, dtype)
  Talker stage:
    Connector 接收 metadata → 定位 shared memory segment
    → memcpy shared memory → Talker GPU memory
    → preprocess 函数读取 intermediate data dict

传输开销:
  Thinker→Talker (shared memory): 5.49ms
  Thinker→Talker (Mooncake RDMA): 8.28ms
  Talker→Vocoder (shared memory): 0.53ms
  Talker→Vocoder (Mooncake): 3.34ms
  对比总推理 latency（tens of seconds），connector overhead negligible (<0.1%)
```

跨节点传输：Orchestrator 通过 Ray 管理跨节点 stage 分布，Mooncake-based connector 提供 TCP 或 RDMA-based transport，control plane 仅传递 lightweight metadata。

术语一般如何实现？如何使用？

Unified Connector 在 vLLM-Omni 中以 C++ 实现高性能 backend + Python wrapper。开发者无需直接使用 connector——它由 vLLM-Omni runtime 根据 stage 部署 topology 自动选择 backend（单节点→shared memory，跨节点→Mooncake RDMA）。Connector 还兼容 intra-stage EPD disaggregation（encoder→prefill MM cache transfer、prefill→decode KV cache transfer）。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models
