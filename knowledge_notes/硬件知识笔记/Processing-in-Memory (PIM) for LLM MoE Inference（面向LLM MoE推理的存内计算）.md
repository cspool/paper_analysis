## Processing-in-Memory (PIM) for LLM MoE Inference（面向LLM MoE推理的存内计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Processing-in-Memory (PIM) 是一种将计算逻辑放置在内存（DRAM/HBM）内部或紧邻位置的架构，通过减少数据搬运距离来降低内存带宽瓶颈。对于 LLM 推理，PIM 利用其远高于 GPU 的有效内存带宽（因计算单元靠近存储单元，数据传输路径短），特别适合 memory-bound 的计算任务。Duplex (MICRO 2024) 是代表性 HBM-based PIM 架构，专为加速 MoE 层的 expert 计算设计。PIM 设备的 Ridge Point 通常远低于 GPU（如 RP_acc = 8 Op/B），因其内存带宽极高但计算能力有限。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文通过模拟 Duplex PIM 与 GPU baseline 的对比评估 PIM 在 MLA + MoE 场景的有效性：

- **低 batch size (B < 32)**：PIM 优势明显——MoE expert FC 层的 ArI 低（因 token 少），PIM 的高 BW（4× GPU HBM）使执行时间更短。归一化吞吐量 > 1（PIM 更快）。
- **高 batch size (B > 64)**：GPU 优势明显——大 batch 使 FC 层 ArI 超过 PIM 的 RP_acc=8，PIM compute-bound；GPU 以 RP_acc=281 提供更高计算吞吐。归一化吞吐量 < 1（GPU 更快）。

论文结论：MLA + MoE 架构下，大 batch 推理已成为主流（因 MLA 释放了 KV$ 约束），PIM 仅适用于低 batch/低序列长度的推理场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主要 PIM 实现方案：(1) HBM-PIM (Samsung/AMD) — 在 HBM stack 的逻辑 die 中集成计算单元；(2) UPMEM — 在 DRAM bank 中嵌入处理单元；(3) CXL-PIM — 通过 CXL 接口连接 PIM 设备。论文引用的 Duplex 架构使用 HBM-based PIM，在 memory die 中集成小的计算单元，通过 bank-level parallelism 提供 4× 于 GPU HBM 的有效带宽。部署策略：在 PIM 上执行 MoE expert FC 层（memory-bound 部分），在 GPU 上执行 attention 和 routing（需要高计算吞吐和灵活性）。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
