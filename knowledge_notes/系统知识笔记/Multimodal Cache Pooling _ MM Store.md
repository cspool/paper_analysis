## Multimodal Cache Pooling / MM Store

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MM Store（Multimodal Cache Pooling）是 EPD-Serve 中用于跨 Encode-Prefill 阶段共享已编码多模态特征的分布式缓存池。它以多模态输入的 hash 值为 key、对应的特征向量（feature vector）为 value 存储。核心设计目标：避免重复编码相同多模态输入，支持跨请求的特征复用，并作为 E-P 异步预取机制的数据源实现零拷贝特征传输。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MM Store 在 E-P 异步预取中的运转流程：

```
Encode 实例                    MM Store              Prefill 实例
    │                              │                      │
    │ 1. ViT 编码 I_m              │                      │
    │    → V_m ∈ R^{n×d}           │                      │
    │                              │                      │
    │ 2. hash = SHA256(I_m)        │                      │
    │    put(hash, V_m)            │                      │
    │ ──────────────────────────► │                      │
    │                              │ 存储 key-value       │
    │                              │                      │
    │ 3. 发送 hash 事件(异步)       │                      │
    │ ──────────────────────────────────────────────────► │
    │                              │     4. listener 收到  │
    │                              │        事件            │
    │                              │                      │
    │                              │  5. get(hash)        │
    │                              │ ◄───────────────────  │
    │                              │ ──── V_m ────────►  │
    │                              │                      │
    │                              │     6. 写入本地缓存    │
    │                              │     7. 若 miss:       │
    │                              │       本地重算 V_m    │
    │                              │       (fault-tolerant)│
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MM Store 基于 Mooncake Store 的分布式缓存基础设施构建。缓存淘汰策略论文未明确说明。Fault-tolerant 机制：若 Prefill 实例从 MM Store 检索失败（hash miss 或网络故障），触发本地 recomputation 生成缺失的特征向量，保证 pipeline 连续性。适用场景：(1) 多请求共享同一图像时的特征复用（如相同 logo/UI 在多轮对话中重复出现）；(2) E-P 跨 NPU 部署时的低延迟特征传输（仅传 hash，不传完整 tensor）；(3) Encode 与 Prefill 时间解耦——Encode 可提前批量预处理多模态输入并缓存特征。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend
