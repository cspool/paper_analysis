## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  提出AUM，一个AU-aware资源管理器，用于在AU-enabled CPU上与通用workload共享LLM serving，最大化处理器效率。核心实现包含两个协作组件：(1) **Background AU Profiler**（离线）：通过ARI (Arithmetic Intensity) 判定不同operator的AU使用率U_AU、按AU使用率将处理器划分为High-AU/Low-AU/None-AU三个频率区域、用CAT/MBA profiling不同AU使用率下的最小资源需求R_AU，最终汇总为离散AUV Model；(2) **Runtime AU Controller**（在线）：Slack-aware SLO Analyzer通过LAG分析实时量化每个serving request领先/落后于deadline的程度来设定decode SLO、Efficiency-aware Core Switcher按加权perf-per-watt最大化切换core分区分频配置、Collision-aware Allocation Tuner根据AU性能偏差δ_AU调整资源分配并优先收获对AU干扰最小的资源（如低AU operator的LLC和带宽）。实验比较CPU perf-per-watt efficiency和AU application SLO guarantee (TTFT SLO和TPOT SLO)，对比EXCLUSIVE (ALL-AU)、AUV-oblivious sharing (SMT-AU, RP-AU) 和AU-aware ablations (AU-UP, AU-FI, AU-RB)。

- 硬件平台是什么，配置是什么。
  三台商用AU-enabled CPU平台：(1) **GenA**: Intel 4th Sapphire Rapids (SPR), 2×Xeon 8475B (48核/socket, 2 socket), 2.7 GHz基频, AU TFLOPS: AVX-512 25.6 / AMX 206.4, L1-I 32KB, L1-D 48KB, L2 2MB/core, LLC 97.5MB/socket, DDR5 1TB (233.8 GB/s)；(2) **GenB**: Intel 4th Sapphire Rapids, 2×Xeon Max 9468 (48核/socket, 2 socket), 2.1 GHz基频, AU TFLOPS: AVX-512 25.6 / AMX 206.4, LLC 105MB/socket, HBM 128GB (588 GB/s)；(3) **GenC**: Intel 6th Granite Rapids (GNR), 1×Xeon 6982P-C (120核/socket, 1 socket), 2.8 GHz基频, AU TFLOPS: AVX-512 32 / AMX 344, L1-I 64KB/core, L1-D 48KB/core, L2 2MB/core, LLC 504MB/socket, MCR 768GB (600 GB/s)。主eval平台为GenA。GPU对比平台：单NVIDIA A100 80GB。

- 开源Serving框架是什么。修改了什么。
  基于Intel xFasterTransformer (xft) 实现LLM serving原型，通过两个Python组件实现AUM管理：(1) Background Profiler在专用节点上重复实验记录AUV Model（450次AU-enabled执行收敛：3 division × 3 sharing × 5 config × 10 repetition）；(2) Runtime Controller作为daemon监控SLO并tune资源分配，决策耗时<1ms。工具链使用Linux perf、pmu-tools、pqos、turbostat做AU行为表征。修改：(1) 修改xft支持基于ARI的AU选择（prefill用AMX、decode用GEMV/AVX）；(2) 通过Intel RDT (CAT/MBA) 接口实现LLC way和memory bandwidth的动态分区；(3) 实现LAG-based SLO analyzer (Algorithm 1) 和efficiency-weighted core switching决策逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源地址（HPCA 2026已录用，可能后续开源）。使用流程（以chatbot场景，GenA + SPECjbb shared为例）：
  1. 部署：GenA上运行xft serving llama2-7b/13b BF16, batch_size=16。AUM的Background Profiler离线运行→收敛后构建AUV Model table (Table III示例：High U_AU=0-11 cores, F=2.1GHz, R_LLC=2way等)→Runtime Controller作为Python daemon启动
  2. 请求到达：LLM requests到达xft→prefill tokens (高AMX usage, TTFT SLO dTTFT=250ms)→decode tokens (低AMX usage, TPOT SLO dTPOT=100ms)。AUM计算每个request的LAG = Σ(dTPOT - e_token)
  3. Runtime Controller每个control iteration：Slack Analyzer计算SLO_H = dTTFT - t_wait, SLO_L = dTPOT + LAG→Core Switcher按max E_CPU = (1.8×P_H + 0.2×P_L + γ×P_N)/W_CPU选择最佳分区分频→Allocation Tuner监控P^m，若满足SLO则用P^a aggressive harvest资源给SPECjbb，否则用P^t conservative return资源
  4. 资源调整通过CAT设置LLC ways、MBA限制memory BW→每次调整<1ms，vs 100ms-scale token latency可忽略
  5. 效果：AUM相比ALL-AU baseline efficiency提升8.8%（平均），比SMT-AU/RP-AU提升4.7%；chatbot场景下SLO guarantee 93.6%（比AUV-oblivious高11%），decode TPOT SLO比AUV-oblivious高7%

