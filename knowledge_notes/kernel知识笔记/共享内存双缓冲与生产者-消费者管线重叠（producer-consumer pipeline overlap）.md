## 共享内存双缓冲与生产者-消费者管线重叠（producer-consumer pipeline overlap）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU kernel 内用两块（或多块）shared memory buffer 交替：生产者（解码/加载）填充 buffer A 的同时消费者（tensor core）计算 buffer B，经原子标志/barrier 同步，从而把生产延迟藏在计算之后。CUTLASS 的 multi-stage 流水（num_stages）、cuBLASDx 的 pipeline、Hopper 的 TMA + mbarrier 都是同一思想的工业化形态。本论文在双缓冲之上再叠加四阶段 shared memory ring buffer，让解码领先若干 sub-tile，消费者每步都能拿到已就绪操作数。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文的流水时序（Section V-D）：生产者（rANS 解码）与消费者（MMA）运行在 SM 内物理分立的管线上——解码 warp 只用整数（查表/状态更新）+ LSU（coalesced 补位、shared store）管线，GEMM warp 只用 shared load + tensor pipeline；warp scheduler 同周期 co-issue，互不占用对方资源。代价模型：解码成本/sub-tile 近似常数（概率表访问 + 重归一化读），MMA 成本/sub-tile 随 M-rows 增长——故小 batch 时 tensor 管线先耗尽（decoder-bound，退化为 decode-then-GEMM 行为），大 batch 时消费者永不 stall、解码完全隐藏、近完全重叠。效果：加双缓冲较单缓冲 tile 对齐版本进一步提升，总计 4.0–10.1× vs 分离两段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：`As[2][...]` 双缓冲 + cp.async/`__pipeline_memcpy_async`；或 CUTLASS num_stages=N 多级流水；Hopper 上 TMA descriptor + mbarrier 的 producer_acquire/consumer_acquire；本文用 cuda::atomic_ref<int, thread_scope_block> 的 ready 标志。注意实测边界：当瓶颈是寄存器压力/occupancy 而非访存延迟时，双缓冲收益有限（开源 benchmark 显示 4096 规模 GEMM 上双缓冲仅 1.03×）。使用：任何"内存加载/解码 vs 计算"重叠场景（GEMM 主循环、attention 流水、解压-计算融合内核）。

  - SHyLA 补充：解析模型假设 GEMM/GEMV 以双缓冲把 Weight/KVCache 加载与 MAC 计算重叠（Fig. 8），即"加载下个 tile 的 Weight 的同时计算当前 tile"；因 LLM 线性层顺序访问命中 row buffer、NVM 读利用率可达 70%（> 一般工作负载），该重叠有效。数据布局层面，NVM plane 存 NTile 与同组 DTile 的 Weight 切块、DRAM plane 存 IA 行，tile group 内专用高速链路供跨内存交换，避免 plane starvation；这些缓冲/加载调度在 GPGPU-Sim 中以 CUDA 双缓冲 + plane-aware tile 映射实现（论文未开源）。
涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
