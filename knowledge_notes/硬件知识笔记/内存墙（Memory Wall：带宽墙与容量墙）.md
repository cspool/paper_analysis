## 内存墙（Memory Wall：带宽墙与容量墙）

术语解释
LLM 推理中 off-chip 内存的两类硬约束：带宽墙（数据流率超过 HBM 带宽）与容量墙（KV cache 增长超过 HBM 容量），共同使计算单元利用率远低于峰值。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
带宽墙（bandwidth wall）：每步 decode 需读全部 KV cache 与 FFN 权重、并写回新 KV，总字节率超过 HBM 带宽时计算单元等待数据。容量墙（capacity wall）：KV cache 随上下文线性增长（LLAMA-3-70B 128k 上下文 FP16 KV 单 batch ≈39 GB，常超过权重体积），限制可驻留 batch 数。二者协同：容量墙限制 batch → GEMM 的 M 维小 → fat GEMM → 方形阵列低利用；带宽墙限制计算速率。agentic 负载（比 chatbot 平均多 100× token、极端 1000×）使两墙显著加剧。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PLENA 的度量与缓解链路：① 量化前：FP16 KV 239.26 GB、权重 129.46 GB、峰值带宽需求 8192 GB/s（OSWorld-L 90k/8k、BS=8）；② 非对称 MX 量化 4/4/4 后：KV 59.81 GB、权重 32.36 GB、峰值带宽 2048 GB/s——带宽墙与容量墙同时缓解，同 HBM 容量下 batch 4→16；③ 扁平阵列解决容量墙造成的 fat GEMM 利用率问题；④ 原生 FlashAttention 消除 QK^T 中间结果 off-chip 往返（大中间激活留片内）；⑤ 硬件预取引擎（H_LOAD_M/H_LOAD_V）把 HBM 延迟与计算重叠，逼近带宽利用率上限。系统级评估协议以"batch = 每硬件-负载组合下 HBM 容量可容最大值"对准容量墙（full HBM-capacity utilization），再在乘法器数、HBM 容量带宽对齐条件下比较 TPS 与 Tok/J。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：带宽侧——量化/稀疏/投机解码减少每 token 字节数、预取与计算重叠隐藏延迟；容量侧——KV 量化/卸载/逐出/前缀共享。论文用法：把两墙解析建模（KV 足迹表、权重存储表、峰值带宽表，Table X）并据此选择 batch 与量化配置；A100（80 GB/1.99 TB/s）、H100（80 GB/3.35 TB/s）、TPU v6e（32 GB/1.56 TB/s）与 PLENA（16 加速器系统对齐 TPU 容量带宽）归一化比较——PLENA 最高 2.23× A100、4.70× TPU v6e 吞吐、4.04× Tok/J 能效。同类概念：PIM 文献中的内存墙（处理器-内存速度鸿沟）与本文的推理专用两墙（带宽+容量）一脉相承，本文把"容量墙→batch 受限→fat GEMM 低利用"链条显式化。

DynoPipe 补充视角（ISCA'26）：内存墙在边云场景表现为跨设备不对称——边缘 DRAM（64GB、51.2 GB/s）vs 云端 HBM（140GB、3.35 TB/s）存在 65× 带宽差距，edge-only 部署 >8B 模型即内存耗尽（FlexNN）、KV 访问模式针对 HBM 优化后在 DRAM 劣化 4.2×。DynoPipe 用"跨域流水"绕过单设备内存墙而非芯片级改造：把算力密集层放 HBM 侧（云端 A40）、隐私敏感+轻量层放边缘 DRAM 侧（RTX 3090），边缘只容纳 embedding+前几层权重与局部 KV、云端 HBM 承载大头权重，把"单设备放不下"拆成两段都放得下的"分段内存墙"——代价是跨域传输延迟（网络+序列化），由 pipeline 并行的排队减少抵消（LLaMA2-7B 128K+ context、16+ 并发、70B 模型 830 tok/s）。实现依赖参数重叠缓存 + 差分 KV 同步，使边界迁移不必全量搬运状态。

NeRArch-Sim 补充视角（ISCA'26，DRAM 时序建模）：NeRArch-Sim 的模块化硬件加速器把内存子系统建模为"可配置 SRAM 块（容量/组数/端口数/访问延迟，显式算子-SRAM 绑定）+ DRAM 后端经 Ramulator 2.0 建模 DRAM timing"，构成 memory-aware duration 模型的访存侧（式 1 的 v_off/B_hw 项：每个 SRAM 的带宽/延迟、DRAM 时序决定端到端周期）。首次 PPA 运行自动克隆/构建 Ramulator 2.0（Hardware/ramulator2/，CMU-SAFARI）。评估设定（Sec. V-A）：28nm HPCP 节点、DRAM 时序统计用 Ramulator 得到。per-SRAM 数据流分析（表 VIII，Lego 场景）给出每个 SRAM 块的 SRAM Rd/Wr 与 DRAM Rd/Wr 流量（如 ICARUS Input FIFO 4KB 每帧 1.4GB、NeuRex Grid Cache 64KB DRAM Rd 320KB/Subgrid Buffer 128KB DRAM Rd 16MB、GSCore Gaussian In FIFO 8KB DRAM Rd 78.8MB），以及 bank conflict 与 stall（表 IX：NeuRex Subgrid Buffer 冲突 7.7M、stall 开销 2.25%，因细分辨率层不规则哈希查找；ICARUS 激活 ping-pong 冲突最多但 MLP 计算主导故开销 0.07%；GSCore 顺序 tile 处理几乎无争用 0.01%）。

R-Max 补充视角（ISCA'26，缓存预取桥接内存墙的上界）：R-Max 的动机正是内存墙——处理单元需求超过内存供给，预取是桥接这一差距的成熟手段，但领域内没有"预取能带来多大增益"的现实上界。论文指出 Always Hit（所有访问命中）上界过松：忽略带宽与 MSHR 约束（如 619.lbm 的 Always Hit L2 660.5% vs R-Max 10.3%、pr.kron 701.6% vs 24.8%）；R-Max 给出受带宽/容量/延迟约束的现实上界（L2 geomean 72.6%、最高 299.6%），并量化 DRAM utilization 降 47.93%（工作集可入缓存时 R-Max 靠及时预取+有未来知识的替换减少 DRAM 流量）。
STEP 补充视角（ISCA'26，CPU 侧内存墙与预取）：STEP 的动机正是经典内存墙——近十年 DRAM 容量与带宽显著改善但延迟进展缓慢，处理器频繁因访存停顿；cache 层次可隐藏大部分延迟，但 cache 不友好的数据结构（局部性差、逐出率高、强制缺失）限制了其效果，数据预取是把数据提前取入更近缓存、隐藏全内存访问延迟的成熟手段。STEP 作为空间足迹预取器，其多触发时序决策（FOE/SOE/TOE + 置信度评估）把"早机会"与"晚精度"统一：带宽受限场景（800 MT/s 最低带宽点）下 STEP 仍领先 eBingo/Gaze，说明内存墙约束下分阶段触发依然有效；受限 way 预取（1 way）下 STEP 仍 1.263× 领先，证明收益不仅来自低污染。DRAM 配置：DDR4-3200 8 banks/rank，tRP=tRCD=tCAS=12.5ns，1C-8C 通道/rank 扩展。


涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
