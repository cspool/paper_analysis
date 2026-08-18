## MGPUSim（多 GPU 周期级模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MGPUSim 是开源的周期级（cycle-accurate）多 GPU 性能模拟器（ISCA 2019，Sun et al.，Northeastern SArchLab），基于 Go 与 Project Akita 离散事件仿真框架实现，默认建模 AMD GPU（GCN3，R9 Nano；v5 起新增 CDNA3/MI300A 建模并移除 NVIDIA 支持），支持多 GPU 系统、L1/L2 cache、TLB、DRAM controller 与 RDMA（多 GPU 互连）建模，同时提供快速功能仿真与详细时序仿真两种模式。RoCC 论文用 MGPUSim 建模以 NVIDIA V100（baseline）/H100/B200 为参照的多 GPU 系统，并扩展它支持 SM-initiated kernel 与 NCCL 式 ring collective（AllReduce/AllGather/AllToAll），用于周期级验证 RoCC 的 ROP 扩展设计。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
作为硬件架构评估工具：输入 = GEMM kernel + CC 配置（tensor 大小、tile 形状、Column-Linear/RowLinear/AllToAll 并行模式、模型参数）+ 平台配置（Table III：80 SMs、L1D 128KB/L1I 32KB、L2 6MB 16-way 64 MSHR、XBar NoC 32B flit、64 MPU、每 MPU 1 ROP（1KB cache、28-cycle 数据通路、4×3-cycle ALU）、4 并发 doorbell、GPU-GPU 300GBps full-mesh、CPU-GPU PCIe Gen4 x16 ≈150 cycle、DRAM ≈900GBps）→ Akita 离散事件引擎逐事件推进（SM 执行 tiled GEMM、warp 完成 tile 后发 RoCC 指令 → doorbell manager 识别 → 译码器生成 μOp → ROP 执行 + 跨 GPU 门铃接力）→ 输出 GEMM/CC 执行时间、重叠比例（83.4%）、CC-only 延迟、争用下 GEMM 性能与各模型加速比。修改：新增 ROP_AR/ROP_AG/ROP_A2A 指令路径、RoCC descriptor buffer、doorbell manager/buffer、双译码器、collective command buffer、对称物理地址映射；以 V100/H100/B200 三套参数化配置（H100：24 ROP/132 SM/50MB L2/3.35TBps/900GBps；B200：2 chiplet/48 ROP/148 SM/126MB L2/8TBps/1.8TBps NVLink）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MGPUSim 开源（https://github.com/sarchlab/mgpusim，MIT，Go + Akita；官方文档 https://akitasim.dev/docs/mgpusim/intro）。使用示例：① 安装 Go；② `git clone https://github.com/sarchlab/mgpusim && cd mgpusim/samples/<workload>`；③ `go build`；④ `./<workload> -timing --report-all`（-timing 开启时序仿真）；⑤ 从 metrics.csv（或 akita_sim_*.sqlite3）读取 kernel 执行时间、cache/DRAM/互连统计。支持的 workload 套件：AMD APP SDK、DNN Mark、HeteroMark、Polybench、Rodinia、SHOC 等。RoCC 论文未开源其 MGPUSim 扩展（web search 未找到公开仓库，论文刚被 ISCA 2026 接收），无法确认。作用：在多 GPU 平台的周期级精度上评估架构改动（本文的 ROP 复用）对 GEMM-CC 重叠、通信延迟与争用的收益。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
