## Modality-Aware Multi-Path Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modality-Aware Multi-Path Scheduling 是 EPD-Serve 的请求级调度策略，根据请求是否包含非文本模态（图像/音频/视频）将其路由到不同的执行管道。多模态请求走完整的 Encode-Prefill-Decode 管道，纯文本请求直接走 Prefill-Decode 管道（绕过 Encode 阶段）。配合 instance-level 的 least-loaded-first 动态负载均衡，实现异构流量分离和资源隔离。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

多路径调度+负载均衡的运转流程：

```
API Server 接收请求
  │
  ▼
Modal Detector: 检查 inputs 是否含 image/audio/video
  │
  ├── 多模态请求 → E-P-D Pipeline（完整三阶段）
  │     └── Global Instance Status Table
  │          ├── Encode instances: [E0: queue=2, E1: queue=5] → 选 E0
  │          ├── Prefill instances: [P0: queue=1, P1: queue=3] → 选 P0
  │          └── Decode instances: [D0: queue=4, D1: queue=2] → 选 D1
  │
  └── 纯文本请求 → P-D Pipeline（仅 Prefill-Decode）
        └── Global Instance Status Table
             ├── Prefill instances: 同 P0 (queue=1)
             └── Decode instances: 同 D1 (queue=2)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) Modal Detector 在 API Server 或 Proxy 层实现，检查 request payload 的 content-type 或多模态字段；(2) Global Instance Status Table 实时追踪各阶段实例的 queue length、pending requests、resource usage 等指标；(3) Least-loaded-first dispatch：新请求发给当前负载最低的实例；(4) 关键效果——防止高负载多模态请求（需 Encode）抢占纯文本请求的 Prefill/Decode 资源，保证两种请求类型的 SLO 不被跨模态干扰。EPD-Serve 论文 evaluation 中使用了含 256 text-image + 256 text-only 的 mixed dataset (VisualWebInstruct) 验证此策略的有效性。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
