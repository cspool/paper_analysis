## Continuous Batching（连续批处理 / iteration-level 调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
连续批处理（Orca OSDI'22 提出、vLLM 推广）把调度粒度从"整批"降为"每个 forward iteration"：每个 token 迭代重新组批，新请求在 slot 空闲时立即插入、完成的请求立即离开，不等整批最慢请求（静态批处理的尾部问题）。调度器维护 waiting/running/swapped 三队列，每迭代：释放完成 → 接纳新请求 → 显存不足时抢占（swap/重算）→ 重排。工程报告效果：GPU 利用率 30–45%→85–98%、吞吐 2.5–5×、p99 相对 median 从 3–8× 降到 1.3–1.8×。与 PagedAttention 配合使"KV 按需分配、请求随时进出批"成为可能。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ConServe 中的使用：decode 吞吐差距的一部分来自"更低 TTFT 使请求更均匀进入 decode，continuous batching 调度器在同一到达率下形成更大更稳的 decode 批，SM 更忙"。multi-turn 下调度器把各 conversation 的新 turn 组进 micro-batch——同一 conversation 的 turn 可能非连续批处理，KV 须跨 turn 常驻且持续增长，这是 vAttention 单请求设计不满足的 allocator 需求（ConServe 的目标场景）。ConServe 不改调度算法本身，只替换 KV 内存管理（所有系统用同一 batching policy）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：vLLM/SGLang/TGI 均内置；配套 chunked prefill（大 prefill 分块与 decode 交错，防 prefill 饿死，vLLM V1 默认开）；in-flight batching（TRT-LLM）同批混合 prefill/decode；可与 prefix caching、投机解码组合。使用方式：在线多请求 serving 的默认调度器；到达率扫描（Poisson）下观察 TTFT 饱和点（ConServe：Yi-6B/Llama-3-8B 约 4.5–5 req/s、Yi-34B 约 2–2.5 req/s）。Web 证据：llm-inference-handbook 连续批处理章节与 vLLM 官方 blog（https://vllm.ai/blog/2025-09-05-anatomy-of-vllm）确认三队列与 iteration-level 调度流程。

EVA 补充视角（ISCA'26，硬件侧 multi-batch 支持）：EVA 的 VQ-GEMM 阶段对多个请求（batch>1）的 Tile 0 计算方式与单请求相同，但 Epilogue 阶段多个请求可以复用同一 weight tile（WI 相同），显著降低带宽消耗，为 continuous batching 提供硬件级权重复用支撑。batch≤32 时 EVA-A16W2 已高利用率（延迟随 batch 近线性增长），batch>32 后 workload 变成 GEMM 形态、非 VQ 架构阵列利用率趋近 100%，此时 INT8 计算（EVA-A8W8）反超 FP16 VQ——说明 batch 增大后连续批处理负载自然转向 GEMM 形态，VQ 加速器的优势窗口在低 batch 的 decode-heavy 场景。

HybridSpec 补充视角（ISCA'26，连续批处理在异构+SD 下的失效与扩展）：连续批处理假设各阶段模型结构相同，而 SD 破坏这一假设——prefill/verification 跑 target、decode 跑 draft，异构模型无法统一批处理；即使同步各请求的 draft/verify 阶段，请求间 draft budget 差异也引入 straggler 阻塞快请求，产生 ragged batching 气泡（图 5(b)）。HybridSpec 的解法是把连续批处理扩展到异构架构的"异步 batching"（见本库同名条目）：XPU 与 HB 栈各自维护 task pool、空闲即组批、批组成每迭代动态变化。此外：批处理下 batch>1 时低算术强度算子（attention）被批量放大、映射到高带宽 HB 栈的理论依据（HB-ATTEN baseline 思路）也源于此——但 HB 容量不足会反噬（见 KV Cache 条目）。

从系统架构角度拆解（图 8 异步 batching 时间线）：请求 T0/T1 到达 → XPU 组批 prefill → 完成即注入 HB 栈 task pool 成为 decode 任务（T2）→ HB 栈组批迭代 decode（T5 为未达 budget 的续 decode）→ 达 budget 后回传 XPU 验证（T3）→ accepted 后再派新 decode（T4）。两单元各按"空闲即组批"原则异步推进，批组成（请求集合）跨迭代可变，无全局同步屏障。

实现与使用：连续批处理是 vLLM/SGLang 默认调度（三队列 + iteration-level 组批）；HybridSpec 在扩展 SplitwiseSim 的事件驱动模拟器里实现异步 batching 变体（双 task pool + watermark 内存水位），与 CHIME 的 sub-batch 调度同属"连续批处理在异构/分离系统上的扩展"一族。

  - SHyLA 补充：SHyLA 在 PD aggregation 下采用连续 batching，约束"每个微批最多一个 prefill 请求"；GPU baseline（8× H800、4× MI300X）用 vLLM 的连续 batching + PagedAttention，batch 由 vLLM 运行时按 GPU 内存动态决定（作为对比基线，非 SHyLA 实现组件）。更大微批 → 更强 Weight 复用是 SHyLA 相对 DRAM-only 的核心收益来源之一。
涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

Tetris 补充视角（ISCA'26，CDSP 中 continuous batching 与静态 batching 的取舍）：Tetris 论证 LoongServe 式贪心静态 batching（整批 prefill 完集体进 decode、批固定）在在线场景的三个问题——(1) 长请求合批 prefill 让早到请求等整批、恶化 TTFT（应每批单请求）；(2) 局部最优缺全局负载感知、过度 SP 扩张恶化整体 TTFT 分布；(3) decode 批内请求逐批完成、资源利用率递减，静态批无法插入新请求。因此 Tetris 的 decoding 实例采用 iteration-level/continuous batching：receive manager 收齐某请求全部 KV chunk 后通知 local scheduler 把请求插入 decode batch（随时可加/可退），并在 decode 调度器扩展 Llumnix 的 "virtual usage"——把正在 KV cache 传输的请求槽位视为虚拟占用，新请求路由到 freeness rate（可用槽/活跃 batch 大小）最高的实例。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
