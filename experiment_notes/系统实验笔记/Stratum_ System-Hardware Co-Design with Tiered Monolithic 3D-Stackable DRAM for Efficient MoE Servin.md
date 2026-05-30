## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Stratum 的 Serving 调度系统包含四个核心模块：(1) **Topic-Aware Request Scheduler**——接收包含 topic tag 的 inference requests，SLO-aware 地将同一 topic 的请求批量分组，优先 dispatch 同 topic 请求以最大化 hot expert hits。SLO 定义为 TTFT（Time to First Token），确保请求不等待过久。(2) **Request Generator（Poisson Process）**——模拟不同 topic 的请求以定义速率到达，生成 realistic serving workload。(3) **Memory and Computation Mapper**——Memory Mapper 按 Algorithm 1 聚合 batch 内所有 topic 的 expert usage 表 → 计算最大化 hot expert hit 的 expert placement。Computation Mapper 将 prefill phase 分配给 xPU、decode phase 分配给 Stratum NMP（类似 AttAcc 策略）。Memory reconfiguration（expert swap）在两次 dispatch 之间执行。(4) **Expert Swap 机制**——当 scheduler 切换到新 topic batch 时，通过 NMP 的 row-swap buffer 在 Mono3D DRAM bank 内执行 tier-to-tier expert 迁移，避免 traversing DRAM-xPU interposer 接口。
  - 实验比较：(a) System-level decoding throughput：Stratum tiering vs GPU baseline (vLLM 0.8.1) vs Stratum no-tiering vs Duplex，四种模型（OLMoE/Mixtral/Qwen2.5/Llama-4），不同 input/output lengths；(b) Energy efficiency：同样配置下的能效对比；(c) Batch size scaling (1-32)：Stratum-XL on Llama-4-Scout，4.7-9.8× throughput vs GPU baseline；(d) Expert swap overhead：<0.37% time, <0.03‰ energy；(e) SLO-aware scheduling 效果：same-topic batching 最大化 hot expert hit rates。

- 硬件平台是什么，配置是什么。
  - Stratum-S：NVIDIA RTX A6000 + 1 Mono3D DRAM chip (32GB), 16 channels, 16 banks/channel
  - Stratum-L：NVIDIA H100 SXM5 HBM3 + 6 Mono3D DRAM chips (192GB total), 1024-bit xPU-DRAM I/F @ 6.4 Gbps/pin
  - Stratum-XL：2× Stratum-L modules, 384GB total, NVLink cross-chip interconnect
  - GPU baseline：vLLM 0.8.1 on RTX A6000 / H100 SXM5 HBM3 GPUs
  - GPU energy：measured via NVIDIA-SMI tool
  - System-level simulator：自研 in-house simulator，包含 Request Generator（Poisson process）、SLO-Aware Scheduler、Memory/Computation Mapper、Stratum NMP interface

- 开源Serving框架是什么。修改了什么。
  - 论文未基于开源 Serving 框架。Stratum 使用自研 system-level simulator（非开源）和自研 NMP simulator 进行端到端 serving 评估。GPU baseline 使用 vLLM 0.8.1 但不修改其代码——仅作为对比基准。
  - System-level simulator 架构：Request Generator（Poisson 到达，topic-tagged）→ SLO-Aware Scheduler（动态 batching，优先同 topic dispatch）→ Memory Mapper（Algorithm 1 expert placement）→ Computation Mapper（prefill→xPU, decode→NMP）→ Stratum NMP Simulator（cycle-level execution + energy accumulation）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未开源 Stratum 系统代码。GPU baseline 使用开源 vLLM 0.8.1。
  - Serving 执行全过程（Stratum-L, Mixtral 8×7B, batch=1-32）：
    ```
    输入：多个 client 发送不同 topic 的 inference requests
    
    阶段 0 — 预处理：
    1. Topic Classifier (CPU, <10ms): 每个 query → DistillBERT inference → topic label (6 classes)
    2. Scheduler 按 topic 将 requests enqueue 到对应 queue，维护 per-request SLO (TTFT)
    
    阶段 1 — Batch 构建与调度：
    3. Scheduler 按 SLO 约束 periodically 检查 queue：
       - Within SLO slack: 优先等待同 topic 请求到达 → 最大化同 topic batch
       - Near SLO deadline: 立即 dispatch 现有 batch（混合 topic）
    4. Scheduler 将 batch dispatch 到 Stratum 处理系统，携带 topic tag 列表
    
    阶段 2 — Memory Mapping (Expert Placement):
    5. Memory Mapper 读取 batch topic tags → 查询 per-topic expert usage table → 聚合
       → 按 Algorithm 1 计算目标 expert placement（hot→fast tier, cold→slow tier）
    6. 若当前 placement ≠ target: 触发 expert swap（near-memory 操作，row-swap buffer）
    
    阶段 3 — 计算分配与执行：
    7. Computation Mapper: Prefill tokens → xPU (H100 GPU), Decode tokens → Stratum NMP
    8. xPU: Gating network forward (lightweight linear layer: 4096→8) → routing decisions
    9. xPU: 发送 input tokens + expert IDs + scaling weights → Mono3D DRAM → switch to NMP mode
    10. Stratum NMP: 顺序执行 activated experts (sequential, tensor-parallel across all PUs)
        - GeMM1 + GeMM2 (projection-up) → Activation (SiLU) + Hadamard → GeMM3 (projection-down)
        - Reduce-scatter 与下一 expert GeMM1 并行（通信-计算 overlap）
    11. NMP: Weighted sum of expert outputs → write back to DRAM → exit NMP mode
    12. xPU: 读取 output tokens from designated DRAM address space
    
    阶段 4 — KV Cache Management（Attention 处理）：
    13. xPU: 写入新生成 KV pairs 到对应 DRAM channels
    14. Stratum NMP: 使用 head-level parallelism 执行 attention（PU groups 分区）
        - 每 PU group 处理多个 heads，interleaved Softmax/MatMul pipeline
        - Query 通过 sub-ring all-gather 分发，减少跨 bank 访问
    
    阶段 5 — 循环与完成：
    15. 重复解码循环直至所有 requests 完成
    16. 输出 tokens → 返回客户端
    
    性能指标：
    - Decoding Throughput (tokens/s): GPU vs Stratum no-tiering vs Stratum tiering
    - Energy Efficiency (tokens/J): 同样对比
    - TTFT SLO compliance: scheduler 保证请求在约束时间内开始处理
    ```
