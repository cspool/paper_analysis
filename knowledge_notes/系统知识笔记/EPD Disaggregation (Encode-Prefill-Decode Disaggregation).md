## EPD Disaggregation (Encode-Prefill-Decode Disaggregation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

EPD Disaggregation 是 EPD-Serve 提出的多模态大语言模型推理的三阶段解耦架构。它将传统的 monolithic 推理 pipeline 拆分为三个独立可调度的实例：**Encode**（Vision Encoder 编码多模态输入为特征向量）、**Prefill**（LLM 首次前向生成首 token 并构建 KVCache）、**Decode**（自回归逐 token 生成）。阶段间通过异步 tensor 传输（E-P 特征预取 + P-D 分层 KV 传输）通信。与 PD Disaggregation（仅解耦 Prefill/Decode）相比，EPD 额外隔离了计算密集的 Encode 阶段，消除了视觉编码对文本推理的资源竞争。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

EPD-Serve 的 (E-P)-D 部署在 2 NPU 上的流程：

```
请求到达 API Server
  │
  ├── 模态感知路由: 含图像 → E-P-D 管道; 纯文本 → P-D 管道
  │
  ▼
NPU 1 (E-P 共置):
  ┌──────────────────────────────────────┐
  │ Encode: ViT(0.7B) 编码图像 → V_m    │
  │   ↓ 完成后发 hash 事件(异步)         │
  │ Prefill: listener 收到 hash→MM Store │
  │   检索 V_m → 拼接 V_m+V_t → LLM(7B)  │
  │   → 逐层 Prefill 计算 KVCache        │
  │   → 分层传输 KVCache 至 NPU 2        │
  └──────────────────────────────────────┘
                    │ P-D KV 传输(分层分组)
                    ▼
NPU 2 (D 独立):
  ┌──────────────────────────────────────┐
  │ Decode: 接收分层 KVCache              │
  │   → 自回归生成 O_i+1                 │
  │   → 至 max_length 或 <eos>           │
  │ 独立 NPU 不受 E/P 资源竞争影响        │
  └──────────────────────────────────────┘
```

部署拓扑符号: "-" = 分置不同硬件, "()" = 物理共置。支持 E-P-D, EP-D, ED-P, E-PD, (E-P)-D, (E-D)-P, (E-PD) 等拓扑按 SLO 灵活切换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EPD 解耦的关键实现要点：(1) 阶段间通信通过 Mooncake Store 的异步传输接口（基于 RDMA/TCP/Shared Memory）；(2) 逻辑隔离 + 物理共置策略——各阶段保留独立进程和调度，但可共享同一 NPU 硬件资源，通过算子级空间复用提升利用率；(3) Proxy 组件统一执行跨实例请求路由和负载均衡；(4) 部署拓扑可按 SLO 优先级动态选择。EPD-Serve 的实验结果显示全解耦 E-P-D (3 NPU) 在 10 req/s 下 SLO 达成率 94.34%，per-NPU effective throughput 为 EP-D 的 7.95 倍。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
