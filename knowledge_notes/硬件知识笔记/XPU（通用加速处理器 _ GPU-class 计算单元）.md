## XPU（通用加速处理器 / GPU-class 计算单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XPU 指通用高性能加速处理器（GPU/TPU 类，HybridSpec 具体用 GA100 GPU core），承担计算密集与容量敏感的工作；在异构系统里与专用/近存单元（如 HB 栈）配合，各自承接不同计算-访存特征的算子或模型。HybridSpec 的 XPU 配 512GB/1.1TB/s LPDDR5X（8 包 × 64GB），跑 target 模型（大权重 + 大 KV cache），并执行目标/草稿 prefill 与全部 verification；相比 A100（80GB HBM2e + interposer），XPU 计算 die 相同但内存换为大容量低成本 LPDDR5X——因为 SD 把 target 的算术强度抬高、使其容忍低带宽。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HybridSpec 的 XPU 角色（一个 draft-verification 周期）：请求到达 → XPU 做 target+draft 的 prefill（target prefill 与 verification 联合批、draft prefill 单独执行）→ 请求移交 HB 栈 draft decode → draft 达 budget 后回传 XPU 做并行验证（拒绝采样）→ accepted token 回传 HB 栈清误推测 KV。XPU 侧调度：异步 batching 双 task pool 之一，PFS 优先 prefill、CHK 把长 prefill 切块与 verification 组批（见系统架构层对应条目）。相对"XPU+CPU（低带宽）"“XPU+不同代 XPU（强度差仅 2-3×）”等异构方案，XPU+HB 栈的强度/容量差可达一个数量级，匹配 draft/target 的极化需求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：现成 GPU/TPU 内核（GA100）配内存封装（LPDDR5X wire bonding 包，trace routing 可行性已验证）；性能评估用分段线性性能模型（对 GPU 在不同并行度/序列长度/batch 下 profiling 拟合）。使用方式：作为异构系统的"计算密集 + 大容量"侧，承接 target 模型、prefill/verification 高算术强度任务；与高带宽专用单元（HB/PIM 栈）按模型级边界（而非算子级）分工，减少跨单元搬运。可扩展性：~100B 模型用数据并行（每设备独立完成请求、无跨设备通信），更大模型（LLaMA-405B）才需 TP（因单设备 512GB 容量大、所需 TP rank 少、通信少）。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
