## RESONATOR（共生式 MLLM serving 运行时：Intra-GPU Sharing Engine + Inter-GPU Parallelism Engine）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RESONATOR 是一个面向 MLLM 推理服务的运行时系统（ISCA'26），把 vision encoder 从"干扰源"变成 LLM 的"合作者"：先在软件栈里解耦 encoder 与 decoder，再通过两级运行时控制重新耦合——①Intra-GPU Sharing Engine（单 GPU 内管理 encoder 与 LLM 的 SM/HBM 共享，双模式：complementary 场景用 SM 分区、contending 场景用 per-kernel stream binding）；②Inter-GPU Parallelism Engine（跨 GPU 按 batch 动态选 encoder 的 DP/TP 计划，用 Performance Atlas 预测 + PRISM 调度 + logical sharding 近零开销切换）。核心洞察来自两处实证：encoder 与 LLM 阶段资源互补（encoder compute-bound 低 HBM、prefill/decode 高 HBM 中 SM，Table I）且 kernel 级存在大量 SM/HBM 空洞（Figure 4）；encoder 最优 TP 度随分辨率/并发变化（低分辨率 1 GPU 最优、高分辨率 4-TP 最优，Figure 5），无单一静态计划最优。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
RESONATOR 的运行时控制流（Figure 6）：
```
Offline: Profiler 在目标 GPU 上扫代表工作点，训练 Performance Atlas（encoder 多项式 + LLM 随机森林）
Online（每请求/每 chunk 边界）:
  1) 单 GPU 内：Intra-GPU Sharing Engine 查 Atlas 判 LLM chunk 标签
     - memory-bound decode-heavy chunk（Tag=mem 且 ρ≥ρ0）→ complementary 模式：
       SM 分区 SM_dec=⌈SM_total·SM_dec_min(c)⌉，decode kernel 固定跑该切片，encoder 用其余 SM
     - compute-bound prefill-heavy chunk → contending 模式：
       per-kernel stream binding：compute-bound kernel → wide stream（全 SM），
       memory-bound/低占用 kernel → narrow stream（窄 SM 子集）
  2) 跨 GPU：Inter-GPU Parallelism Engine 对 pending encoder 请求跑 PRISM（MCKP/DP）
     选最优 batch 与各请求 TP 度，logical sharding 只改 cuBLAS/CUTLASS 的 ld 参数即完成分片
Data plane: 按上述决策经 CUDA 流/SM 控制与逻辑分片执行
```
Annotations：chunk 特征 c=(n_p,n_d,L_c)（prefill token 数、decode token 数、平均 KV 深度 bucket）；TAG(c)∈{mem,comp} 与 SM_dec_min(c) 由 Atlas 提供；模式切换带迟滞（连续多个 chunk 分类一致才切换）；contending 路径用 eager 执行逐 kernel 选流（可接受，因 compute-heavy encoder+prefill 场景 CPU launch 开销相对 kernel 时间小），complementary decode 路径保留 CUDA Graph 重放兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 SGLang-0.4.7 之上：LLM backbone 用 TP=4（7B/16B）或 TP=8（72B）的 SGLang chunked-prefill 执行；encoder 每 GPU 预载完整权重（ViT-675M 1.3GB / MoonViT 0.8GB，HBM 开销 1.6%/1.0%，Table II），运行时经 strided GEMM 逻辑分片；流到 SM 子集的绑定用 green-ctx 或 libsmctrl（论文引用 [26] Bakita & Anderson RTAS'23）。调度查询只读 Atlas，运行时开销是元数据查找+选流。效果：同 GPU 预算下相对 SGLang/vLLM 提高 mean TTFT 最高 5.1×、TPOT 3.0×、E2E 4.9×、吞吐 3.4×；相对 EPD-Serve（6×A100 分池）用 4×A100（省 33% GPU）仍全面领先。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
