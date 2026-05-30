## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **Faster-MoA**，一个统一的算法-系统协同设计，针对 MoA (Mixture-of-Agents) serving 的三个Serving调度创新：(1) **Shell Router + Agent Prompt Cache (APC)**：在 SGLang 标准 PD 引擎之外实现独立的 shell router，负责任务分发和编排。APC 存储每个依赖 agent 的部分解码文本/token，使后继 agent 可增量构建 prompt。(2) **依赖感知增量 Prefilling**：shell router 将依赖 agent 的输入 prompt 按前驱 agent 输出槽分割为独立前缀+依赖段。前缀段无数据依赖可立即 prefill，依赖段随着前驱 agent 解码逐 chunk 流式到来进行增量 prefilling（基于 KV cache 复用，仅计算新增 token 的 KV），实现 decode 和 prefill 的重叠。(3) **两个 API entrypoint**：/generate（标准 PD pipeline）和 /prefill_only（仅执行 prefill 并缓存 KV blocks，不触发 PE→DE 传输）。
  实验比较：(a) 动态 Early-Exit 消融：Tree+EE vs Tree-only 的模型激活分布（4B/8B/32B 各被调用的比例）、EE 开销（仅 ~5% 额外延迟但不带来 10-50% E2E 延迟减少）; (b) 增量 Prefilling vs 三个 baseline（Naive PD disaggregation only、Data Parallelism only、DP+chunked prefill）的第二层 E2E 延迟（最大 27.4% 减少 vs baseline 仅 ~10%）; (c) 最终对比：All-to-all Baseline vs Tree-only vs Tree+Incremental Prefill vs fully-integrated Faster-MoA，E2E 延迟分别减少 ~62%、~76%、~90%，同时准确率 ≤±1%。

- 硬件平台是什么，配置是什么。
  6 张 NVIDIA H200 GPU（单台 H200 HGX Server 内），每模型配置为一台 Prefill Engine (PE) + 一台 Decode Engine (DE)，跑在两个独立 GPU 上。PE 与 DE 之间通过 NVLink 传输预填的 KV blocks。最大输出 token capped at 65535，scheduling conservativeness=0（SGLang 激进调度最大化显存利用率）。

- 开源Serving框架是什么。修改了什么。
  两个开源框架：(1) **SGLang v0.5.3**——用于精确延迟测量，修改包括：添加 /prefill_only API entrypoint、集成 Shell Router 编排逻辑、Agent Prompt Cache 机制、增量 prefilling 流程（fetch→append→incremental-prefill loop），设 concurrency=1 获取精确 per-sample 延迟。(2) **vLLM v0.11.0**——用于大规模 batch dataset-wise 验证，修改包括集成增量 prefilling 和 early-exit 逻辑，设 concurrency=32 questions/batch 加速验证。
  核心修改架构：在 SGLang 的 native PD router + PE/DE 引擎之上添加外层 **Shell Router**。Shell Router 处理四步：(1) Dependency identification——独立请求直接转发 native PD router；(2) Dependent requests handling——按前驱 agent 输出槽分割 prompt，发送前缀到 PE 开始 prefill，监控第一个依赖 agent 的 APC；(3) Incremental prefilling loop——周期性从 APC fetch text/token chunk，append 到已 prefilled 前缀后，发出轻量 /prefill_only update，利用已驻留 HBM 的 prefix KV 达到近 100% KV cache hit；(4) Forward prefill-done requests——所有槽填满后转发 /generate 请求到 native PD router。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文代码未公开（截至查询时无 GitHub 仓库）。来自 Georgia Tech + Peking University + Samsung，提交 DAC 2026。下面基于论文 (Sec. 4.3) 描述给出 Faster-MoA SGLang serving 全流程：

  **以 3-agent 依赖（agent 3 依赖 agent 1、agent 2 输出）在 9-3-1 三层树结构第二层的执行为例**：

  1. **Agent 1 和 Agent 2 先独立执行**：二者的 prompt 无数据依赖 → Shell Router 识别为独立请求 → 直接转发 native SGLang PD router → 标准 prefill+decode → decode 出的 text/token 流式写入各自 APC。

  2. **Agent 3 依赖识别**：Shell Router 接收 agent 3 请求，解析其 prompt 发现依赖 agent 1、agent 2 的输出槽 → 将 prompt 分割为三段：Segment 0 (agent 3 自身前缀)、Slot 1 (agent 1 的输出槽)、Slot 2 (agent 2 的输出槽)。

  3. **前缀 Prefilling（立即启动）**：Segment 0 无数据依赖 → Shell Router 立即发出 /prefill_only 请求到 PE → PE 计算 Segment 0 全部 token 的 KV → KV blocks 驻留 PE HBM（不传输到 DE，因 /prefill_only 跳过了 KV block 传输）。

  4. **增量 Prefilling Loop - Slot 1**：Shell Router 周期性监控 agent 1 的 APC → APC 收到 agent 1 decode 的第一个 text/token chunk → Shell Router 将 chunk 追加到 Segment 0 之后 → 发出 /prefill_only update（仅新 chunk 的 KV 需计算，prefix KV 从 HBM 复用，近 100% cache hit rate）→ 继续 fetch APC → append → incremental prefill 直至 Slot 1 填满（agent 1 decode 完成）。

  5. **增量 Prefilling Loop - Slot 2**：Slot 1 依赖 agent 1 全部输出完成后，Slot 2 依赖的 agent 2 输出段在 prompt 中紧随 Slot 1 → Shell Router 类似地 fetch agent 2 的 APC chunk → append → incremental prefill 直至 Slot 2 填满。

  6. **Prefill 完成 + Decode 启动**：所有 slot 填满 → agent 3 输入 prompt 完整且 prefilling 已在 overlap 中完成（计算被前驱 decode 时间隐藏）→ Shell Router 转发 /generate 请求到 native PD router → DE 执行标准自回归解码。

  **关键执行对比（Fig 3 底部 bubble diagram）**：
  - Vanilla MoA: Agent 1 decode → Agent 2 decode → Agent 3 prefill（等待两者完成）→ Agent 3 decode → 总时间为串行累加
  - Faster-MoA: Agent 1 decode / Agent 2 decode（并行） | Agent 3 prefix prefill（立即） | Agent 3 incremental prefill（与 Agent 1/2 decode 重叠） → Agent 3 decode → 总时间仅略长于最慢前驱

  数据流：`User Prompt → Shell Router → Dependency ID → Segment split → PE(prefill-only, store KV) → APC(poll decoded chunks) → Incremental /prefill_only(append, reuse KV) → /generate to DE → Output tokens`
