## Prefill-First Scheduling 与 Chunked Prefill（PFS + CHK，prefill-verification 仲裁）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HybridSpec 解决 SD 下 XPU 上 prefill 与 verification 竞争资源的调度策略。PFS（Prefill-First Scheduling）：优先 prefill 任务（内存允许时）——FIFO 下早到的 verification 会阻塞后续 prefill，而完成 prefill 才能把请求"物化"为下游可调度工作，阻塞 prefill 就减少可批请求数、限制并发与利用率；PFS 让 prefill 及时完成、请求更快进入 decode 流水，扩大可批请求池。CHK：prefill 长度差异大（几十到几千 token），短 prefill 完成快而欠利用率、长 prefill 长时间占 XPU 延误后续调度；把长 prefill 沿序列维切块（调整 attention mask 保持序列对齐，借鉴 Sarathi [1]），使批更好地匹配 XPU 计算-内存比。批类型三分：verification-only / prefill-only / mixed（短 prefill 与 verification 混批填满算力）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TTFT-TPOT 权衡（图 18，FIFO vs PFS）：FIFO 不抢占 verification，draft-verify 周转短、TPOT 略好，但新请求排在 verification 后面、TTFT 明显更差；请求率升高后 FIFO 把新请求堵在 verification 后、可批请求减少、TPOT 也变差——PFS 两维反超。利用率（图 11）：FIFO 55.63% → PFS 62.43% → CHK 66.17%（每点一迭代，CHK 让更多迭代达满占）。消融：PFS 平均 1.10×、PFS+CHK 平均 1.29×（相对 FIFO，增益随请求率上升）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器在事件驱动模拟器中实现；PFS 改任务出队优先级（prefill 优先）、CHK 加序列维切块 + attention mask 调整。用途：任何"多任务类型共享计算单元"的 SD serving 系统（prefill/verification 争 XPU）都可复用；与 Sarathi 的 chunked prefill 同源，但这里面向 SD 的 prefill-verification 竞争而非 prefill-decode 竞争。

Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载下 Chunked Prefill 的退化模式）：论文观测 Llama-405B 在 batch 达 5K 时系统无法容纳完整状态，调度器激活 Chunked Prefill（每迭代只处理部分 prompt token）作为 OOM 防御：(1) Running 曲线平台化而 Waiting 居高——系统进入"convoy"模式，新 reasoning trace 只在旧请求完成并释放 KV 块后才被准入；(2) 防止崩溃但引入 Start-Up Latency，GPU 利用率高却实质在"内存容量管理"上 stall 而非产出 token；(3) batch 4K/5K 下 KV 在 prefill 阶段即耗尽——"Reasoning Cliff"被提前到准入期。教训：chunked prefill 是容量压力的缓释而非解药，调度器应在 admission 时估计未来 KV 增长并预留 decode 容量，而非只按当前内存使用准入。
涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
