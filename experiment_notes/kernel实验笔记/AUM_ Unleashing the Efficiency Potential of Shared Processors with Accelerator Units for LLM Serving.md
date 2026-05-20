## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  AUM在AU (Accelerator Unit) 选择和core分区层面进行kernel级调度：(1) **AU Selection**：根据ARI = 6(1/d + 3/BL)^(-1) (prefill) 和 6(1/d + 3/B)^(-1) (decode) 判定每个operator的最优AU——高ARI operator (prefill GEMM, dim=8192×4096×22016) 使用AMX TMUL达到40.57 TFLOPS，低ARI operator (decode GEMV, dim=16×4096×22016) 使用AVX达到更高效（仅3.87 TFLOPS via AMX），避免小矩阵AMX overhead（tile register配置开销>计算收益）；(2) **Frequency Region Division**：按AU使用率将物理核划分为High-AU region (2.1-2.5 GHz, 运行prefill)、Low-AU region (2.8-3.1 GHz, 运行decode)、None-AU region (3.2 GHz, 运行shared非AU应用)，利用AU功耗→频率反比关系避免decode phase被compute-intensive shared app频率拖累；(3) **Resource Affinity Profiling**：监控各AU operator的μarch资源bound——prefill为backend bound (92% backend), decode为memory bound (DRAM 59.9%)，指导kernel运行时资源分配。实验比较perf-per-watt效率、AU operator SLO guarantee、各频率区域划分下的AU perf degradation。

- 后端平台是什么，配置是什么。
  三台Intel Xeon AU-enabled CPU：GenA (SPR Xeon 8475B, 48核×2, AMX BF16 206.4 TFLOPS, DDR5 233.8 GB/s)；GenB (SPR Xeon Max 9468, 48核×2, AMX BF16 206.4 TFLOPS, HBM 588 GB/s)；GenC (GNR Xeon 6982P-C, 120核×1, AMX BF16+FP16 344 TFLOPS, MCR 600 GB/s)。AU单元详情：每物理核AMX单元含8×1KB TILECFG寄存器和TMUL加速器（1024 BF16 ops/cycle），m≤16, n≤64的矩阵乘。

- 评估性能的软件/脚本是什么。修改了什么。
  软件栈：xFasterTransformer (LLM serving框架) + Intel oneDNN (底层AMX算子)。表征工具：Linux perf (tma_amx_busy, tma_fp_amx, avx_insts等metrics)、pmu-tools (top-down分析)、turbostat (core频率记录)、pqos + Intel RDT CAT/MBA (LLC和内存带宽分区)。AUM Background Profiler收集operator-level AU behavior：AMX cycle ratio (prefill 14.4% vs decode 1.5%)、AMX μop ratio (prefill 3.7% vs decode 0.5%)、per-region频率下限(f_H/f_L/f_N)、per-region minimal资源需求(R_L2C, R_LLC, R_BW)。修改：通过ARI阈值判定AMX vs AVX选择→不同region设置不同U_AU threshold。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。软件使用流程：
  1. 表征阶段：在GenA上运行xft llama2-7b prefill/decode→perf stat记录per-core AMX cycle ratio→turbostat记录per-core频率→绘制Figure 6a频率vs AU core count曲线
  2. Top-down分析：pmu-tools toplev采集prefill/GEMM/decode的cycle分布→Figure 7对比AU vs非AU应用的frontend/backend bound→发现AU frontend bound显著低(5%→1%)、decoder backend DRAM bound达59.9%
  3. AUV Model profiling：对High/Low/None三个频率区域×5种敏感资源配置各10次重复执行→记录P_a (50%-ile perf)和P_t (90%-ile tail perf)→例：High U_AU 0-11核, F=2.1 GHz, R_LLC=0-2 way, R_BW=50%
  4. Runtime控制：Controller查AUV Model→按Algorithm 1决策：if P^m_H < SLO_H and P^m_L < SLO_L → aggressive harvest (δ = U_AU × SLO/P^m) → 用P_a profile分配min resource给AU、剩余给shared

