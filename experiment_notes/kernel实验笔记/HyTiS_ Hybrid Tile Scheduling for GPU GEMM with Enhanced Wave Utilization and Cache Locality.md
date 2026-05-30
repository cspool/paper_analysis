## HyTiS: Hybrid Tile Scheduling for GPU GEMM with Enhanced Wave Utilization and Cache Locality

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是HyTiS，一个两层级联的混合 tile 调度框架。第一级使用吞吐量导向的 micro-kernel（large tiles）处理 full waves，最大化 SM 利用率；第二级使用延迟导向的 micro-kernel（fine-grained tiles）处理 partial wave，最小化残余 wave 的延迟。此外，HyTiS 实现自适应 tile layout 选择（group-M vs group-N），通过分析模型最小化 wave 粒度的 DRAM 到 L2 cache 数据流量。实验比较 HyTiS 与 cuBLAS、Inductor-Triton、Split-K（CUTLASS）、Stream-K（CUTLASS），以及两个消融变体 HyTiS(L1)（仅单级调度）和 HyTiS(STL)（静态 tile layout）。评估指标包括执行延迟（speedup over cuBLAS）、SM 负载均衡度 B=(max-min)/avg（通过 NSight Compute 采集 sm_cycles_active 指标）和 DRAM read 数据量（NSight Compute dram_bytes_read.sum）。

- 后端平台是什么，配置是什么。
  NVIDIA H100-PCIE (80GB, Hopper 架构, compute capability sm_90) 和 NVIDIA A100-PCIE (40GB, Ampere 架构, compute capability sm_80)。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 Triton 3.2.0 实现 HyTiS 调度框架。主要修改包括：
  1. 实现两层级联 GEMM kernel（Algorithm 1）：第一级循环以 TO micro-kernel K1 执行 n1_wave 个 full waves；第二级以 LO micro-kernel K2 处理剩余 n2_tiles。每个级别的 tile-to-output-offset 映射由 HyTiScheduler 生成的 l1_offset_fn 和 l2_offset_fn 函数控制。
  2. 离线 profiling 阶段：在目标 GPU 上对单个 data layout 执行一次 GEMM operator profiling（H100 ~19 min，A100 ~36 min），收集 SMEM 使用量、register spill 情况和执行延迟，构建 TO 候选集 S^TO 和 LO 候选集 S^LO。
  3. 自适应 tile layout：运行时根据问题形状和 tile 配置，利用分析模型计算最优 group-M/group-N 布局及 group size s_opt，选择使 wave 粒度 DRAM→L2 流量 V_tol 最小的布局。
  4. Hopper 架构上使用 persistent kernel + TMA 指令；Ampere 架构上使用传统 data-parallel launch（因 TMA 不支持且 persistent kernel 导致 register 压力过大）。
  5. 自动调优：在运行时对 TO-LO 组合搜索空间执行 auto-tuning，缓存结果以消除重复开销。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源，Zenodo DOI: 10.5281/zenodo.15244191（论文 AE 版本），另有 Zenodo DOI: 10.5281/zenodo.16674739。评估原理和流程如下：
  1. **安装**：`src/install.sh` 安装 PyTorch 2.3.1、Triton（从源码 build，应用 `patchs/triton-patchs/` 下两个 patch 以启用 HyTiS 调度）、CUTLASS 3.4.1（用于 Stream-K/Split-K baseline）、HyTiS（pip install -e .）。
  2. **整体性能基准测试**：`python exps-1.0/run_tasks.py 0`（无 NCU profiling）运行全部 3600+1024=4624 个 GEMM 测试用例，对每个 (M,N,K, layout) 组合分别跑 cuBLAS、Inductor-Triton、Split-K、Stream-K、HyTiS(L1)、HyTiS(STL) 和 HyTiS，记录执行延迟，计算 speedup 并输出到 `checkpoints/cache/`。生成 Figure 9：`python exps-1.0/fig9.py`。
  3. **Breakdown 分析**：`python exps-1.0/run_tasks.py 1`（带 NCU profiling）采集 SM balance 指标（sm_cycles_active.avg/.max/.min）和 DRAM read 量（dram_bytes_read.sum），按 low/mid/high 三区归一化到 cuBLAS。生成 Table 3：`python exps-1.0/table3.py`。
  4. **Wave quantization 专项测试**：固定 N,K 变 M（如 M ∈ [512,8192] step 64, N=1024, K=4096），在量化显著区（orange highlight）和非显著区比较 HyTiS 与 cuBLAS/Inductor-Triton 的延迟。通过 `MNs=0 MNe=120 TASK_ID=3 python run_tasks.py` 可快速评估 120 个代表性 case（~1 小时）。生成 Figure 10：`python exps-1.0/fig10.py`。
  5. **Hyperparameter 分析**：`MNs=0 MNe=1000 L1_THRES=1.3 L2_THRES=1.4 python run_tasks.py` 测试不同 virtual tile 数量下 l1、l2 阈值对搜索空间和性能的影响。生成 Figure 11：`python exps-1.0/fig11.py`。
  6. **kernel 输入到性能输出全流程**：用户调用 `hytis.matmul(a, b)` → HyTiScheduler 接收问题形状 (M,N,K) → 检查 tuning cache → 若无缓存命中，对 TO×LO 搜索空间内每个有效组合（first level tile 覆盖 full waves，second level tile 覆盖 partial wave 且 tile 数 ≤ N_SM）进行 auto-tuning → 选择最优 (K1, K2, layout) 组合 → HyTiS_GEMM kernel launch：grid_size 个 CTA 并发执行，前 n1_wave×k_tiles 个 iteration 以 K1 处理 full waves（TMA load on H100），后续以 K2 处理 n2_tiles 个 partial wave tiles → 结果写回 output tensor C。性能指标通过 CUDA event timer 采集 kernel 执行时间，NSight Compute 采集 SM 利用率和内存流量。
