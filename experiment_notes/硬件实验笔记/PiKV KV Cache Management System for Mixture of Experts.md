## PiKV KV Cache Management System for Mixture of Experts

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现：PiKV-FPGA（§3.5）将 PiKV 的 metadata-heavy 阶段（路由 lookup、压缩 codec、调度 scoring、page table 管理）offload 到 FPGA SmartNIC（AMD Alveo U55C / Intel Agilex），通过 CXL Type-3 link 连接 disaggregated DDR 内存池。GPU 仅运行 f_enc（encoding）和 f_attn（attention）kernel，通过 32B MMIO command queue 与 FPGA 上的 PiKV-CTRL 模块通信。FPGA 内部包含四个可重构引擎：(1) Routing Engine——ScoreFuse + radix Top-k 实现 hash/TopK/load-balance/cache-aware/entropy-penalized/hierarchical 路由；(2) Compression Engine——Codec_ρ 实现 LoRA (rank-r URAM matvec)、PyramidKV (multi-level codec)、ChunkKV (block-PCA engine)、FastV (tail crop+pad)、structured prune (sparse mask gen)；(3) Scheduling Engine——u_i ≷ θ comparator 实现 H2O/sliding window/QUEST MLP/LRU recency sort/AdaKV multi-feature fuse/Duo layer-sum attention；(4) Page Table Γ——(t,e)↦addr lookup 与 miss count 跟踪。
  - 实验比较：论文 §3.5 给出资源预算分析（E=64, S=256, k=4, K=16, d=128 时 BRAM_Γ ≈ 176 KB, BRAM_meta ≈ 48 KB，可装入单 U55C SLR），以及端到端延迟公式 T_fpga = T_route + k(T_Γ + K(T_ddr + T_codec))。论文未提供与纯 GPU baseline 的 FPGA 实测性能对比数据。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 非模拟器，为实际 FPGA 实现。目标板卡：AMD Alveo U55C（xcu55c-fsvh2892-2L-e）。代码仓库 `core/fpga/` 包含完整 Vivado 工程：RTL 源码（Verilog：pikv_soc_top.v, pikv_axi_lite_slave.v, pikv_cxl_dma.v, pikv_axi_dma_master.v, pikv_top.v）、Vivado Tcl 构建脚本（create_project, build_bitstream, create_bd）、U55C + generic XDC 约束文件、host C library（libpikv_fpga.so）。论文未使用第三方模拟器。

- 模拟器模拟什么的性能，修改了什么。
  - FPGA 实现 offload 以下 metadata-heavy 阶段：(1) Routing——ScoreFuse 融合 routing penalty term 的计算 + radix Top-k 排序，O(E log k) 周期；(2) Compression——LoRA URAM matvec O(dr)、PyramidKV multi-level Codec O(d)、ChunkKV block-PCA O(dr)、FastV tail crop O(d)、structured prune sparse mask gen O(d)；(3) Scheduling——max-reduce attention O(K)、age comparator O(1)、QUEST DSP MLP O(dK)、recency sort O(K log K)、AdaKV multi-feature fuse O(K)、Duo L-way accumulate O(LK)；(4) Page Table——Γ: (t,e)↦addr 与 miss counter 维护。
  - 修改：论文提出了从纯 GPU 到 GPU+FPGA 异构的系统拓扑修改，不修改 vLLM/GPU attention kernel 本身。GPU 通过 MMIO (AXI-Lite) 发命令到 PiKV-CTRL，KV payload 通过 CXL.mem DMA 在 FPGA 管理的 DDR pool 与 GPU 之间传输。每个 decode step 中 GPU 仅接收 FPGA 打包好的 {(K̂,V̂,idx)}_{i∈P_t}，直接用于 FlashAttention。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源情况：FPGA RTL 完全开源在 https://github.com/NoakLiu/PiKV（Verilog 4.3%），可通过 `build_fpga.sh` 构建。
  - FPGA 使用流程（以 Alveo U55C + CXL.mem 为例）：

    **构建阶段**：
    1. 运行 `build_fpga.sh` → Vivado 读取 `core/fpga/vivado/scripts/create_project.tcl` → 创建 Vivado 项目。
    2. 添加 RTL 源文件（pikv_soc_top.v 为顶层，pikv_cxl_dma.v 管理 CXL.mem DMA，pikv_axi_lite_slave.v 处理 MMIO 命令，pikv_top.v 集成 routing/compression/scheduling engines）。
    3. 添加约束文件（U55C xdc + generic xdc），综合 → 布局布线 → 生成 bitstream。

    **运行时流程**（per decode step）：
    4. GPU 通过 MMIO (AXI-Lite, 32B command queue) 发送 query token q_t 和 metadata 到 PiKV-CTRL。
    5. Routing Engine：接收 q_t 的特征向量 → ScoreFuse 计算 routing weights + penalty terms → radix Top-k 选出 g_t ⊆ E（active experts）。T_route = ⌈E/16⌉ / f_fpga。
    6. Page Table Γ lookup：对 g_t 中每个 expert e，查 Γ: (t,e)↦addr，定位 KV page 在 DDR pool 中的物理地址。T_Γ = 2/f_fpga per lookup。
    7. Compression Engine（on read path）：从 DDR pool 通过 CXL.mem DMA 读取压缩 KV page data → Codec_ρ 按配置执行解压（LoRA matvec / PyramidKV 逆变换 / 等）。T_ddr = 2d'/B_mem, T_codec 取决于所选压缩方案。
    8. Scheduling Engine：读取 page metadata {r_i, f_i, a_i^(ℓ)} → 计算 utility scores u_i → u_i ≷ θ comparator 进行驱逐决策。AdaKV 的 θ 每 Δ tokens 通过 MMIO 更新（BRAM 内）。
    9. 结果打包：FPGA 通过 PCIe/CXL 将 {(K̂, V̂, idx)}_{i∈P_t} 传回 GPU。
    10. GPU 直接执行 FlashAttention(q_t, {(K̂,V̂)})。

    **资源估算**（tile: E=64, S=256, k=4, K=16, d=128）：
    - BRAM_Γ ≈ 176 KB（page table 条目：E·S·80bit）
    - BRAM_meta ≈ 48 KB（per-page metadata：k·K·S·48bit）
    - URAM_W = d·r（LoRA 权重，r=8 时 1KB）
    - 总计 ~224 KB on-chip，可装入单 U55C SLR。

    PiKV-FPGA 核心作用：将 metadata-heavy 的 routing/compression/scheduling 从 GPU 关键路径剥离到 FPGA，GPU 仅接收已过滤、解压、评分后的 KV page 子集，避免 GPU 浪费算力在 KV 管理开销上，突破 HBM 容量墙。
