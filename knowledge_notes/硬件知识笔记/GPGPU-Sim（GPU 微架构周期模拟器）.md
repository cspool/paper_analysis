## GPGPU-Sim（GPU 微架构周期模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPGPU-Sim 是开源周期级 GPU 微架构模拟器（多伦多大学/Georgia Tech，https://github.com/gpgpu-sim/gpgpu-sim_distribution），逐周期模拟 warp 调度、cache/内存层次与 DRAM 控制器，广泛用于 GPU 架构研究。SHyLA 用它作为主性能模拟器：因为 16 个 chiplet 跑相同分区（PD aggregation）或 prefill/decode 实例（PD disaggregation），只需建模单 chiplet 的内存访问，片间通信单独用解析模型计算。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- SHyLA 对 GPGPU-Sim 的修改：(1) 内存控制器地址映射改为把连续地址分布到多 channel（提高带宽利用）；(2) 调整 channel 数匹配 CACTI-3DD 推导的硬件带宽（DRAM 552GB/s、PCM 1792GB/s）；(3) 按 workload 数据访问行为配置 DRAM/PCM 参数（DRAM tCL/tWR/tRCD/tRP=14/9/14/14、PCM=14/1000/120/14、541MHz）。CUDA 管理片上缓冲并实现目标映射策略。模拟聚焦计算 die 与内存控制器间的 request 级交互，以带宽与时序参数为主要建模目标，输出各 workload 运行时分解与系统 token 吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用（官方流程）：clone gpgpu-sim_distribution → 配置 GPU 微架构与内存时序（论文注入 DRAM/PCM 时序与 channel 配置）→ 编译 CUDA kernel（GEMM/GEMV + 双缓冲 + plane-aware tile 映射）→ 时序模式运行 → 读 per-kernel 周期/内存统计。模拟原理：逐周期推进 warp 调度与内存请求 → 经修改的地址映射跨 channel 交错 → 按 DRAM/PCM 时序服务 → 统计带宽利用与 stall。与解析模型互补：解析模型扫设计空间，GPGPU-Sim 校准/验证代表性配置（stall <20%、利用率 90/70/10%）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
