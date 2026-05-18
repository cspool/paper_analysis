## AU-aware Resource Manager (AUM) & AUV Model

术语是什么？

AUM (AU-aware resource Manager) 是AUM论文提出的系统层AU感知资源管理器，通过两个协作组件对三维AUV做offline profiling和online control：(1) **Background AU Profiler**（离线）：用ARI划分AU usage、用频率region记录per-region频率下限、用CAT/MBA profile per-region最小资源需求，汇总为离散AUV Model；(2) **Runtime AU Controller**（在线）：三阶段决策——Slack-aware SLO Analyzer (LAG-based)、Efficiency-aware Core Switcher (加权perf-per-watt最大化)、Collision-aware Allocation Tuner (碰撞感知自适应资源调节)。**AUV Model**是Background Profiler产出的离散化table，每个bucket记录U_AU(usage level)、C_AU(core count)、F_AU(frequency)、R_AU={R_L2C,R_LLC,R_BW}、P^a(avg perf)、P^t(tail perf)、W_CPU(power)。

从系统架构角度拆解术语：

AUM的runtime控制流程（Algorithm 1）：
1. **Slack-aware SLO分析**：Slack Analyzer对每个request计算LAG = Σ(d_TPOT - e_token)——LAG<0表示behind schedule需加速→给更多AU资源；LAG≥0表示ahead→可收获资源给shared apps
2. **Core switching**：Core Switcher按max E_CPU = (α×P_H + β×P_L + γ×P_N)/W_CPU选择最佳分区分频，subject to P^t_H < SLO_H and P^t_L < d_TPOT——prefill token价值远高于decode (α=1.8 vs β=0.2)
3. **Resource tuning**：Allocation Tuner监控P^m performance——if P^m satisfies SLO→用P^a aggressive harvest (δ = U_AU × SLO/P^m)；else→用P^t conservative return；优先收获对AU干扰最小的资源（如decode低AU时先收LLC）；若δ > threshold=2→触发re-partition

术语一般如何实现？如何使用？

部署AUM到AU-enabled CPU平台（以GenA + llama2-7b + SPECjbb为例）：
1. Background Profiler offline：3 division × 3 sharing × 5 config × 10 rep = 450 AU执行次→AUV Model收敛
2. Runtime Controller作为Python daemon部署→每次control iteration <1ms查表决策→通过pqos CAT/MBA调整硬件资源
3. 效率结果：相比ALL-AU eficiency ↑ 8.8%（平均），比SMT-AU/RP-AU ↑ 4.7%；SLO guarantee ↑ 11%（chatbot场景93.6% vs 82.6% baseline）

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
