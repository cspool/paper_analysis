## Amortization Roofline Model

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Amortization Roofline Model 是 MoE-SpeQ 提出的专用性能模型，扩展经典 Roofline Model 以适应 speculative offloading 的 trade-off。两轴：(1) X 轴 Amortization Intensity I_amort(k) = E[Accepted Tokens] / E[Synchronous I/O Bytes]，量化每字节同步 I/O 产生多少有用工作；(2) Y 轴 Effective Throughput Θ(k) = k_accept(k)/T_cycle(k) tokens/sec。两 Roof：Compute Roof（水平线，I/O 完美隐藏时的最大吞吐）和 I/O Roof（斜线斜率=B_PCIe，I/O stall 主导时的吞吐上限）。在线解 argmax_k Θ(k) 确定最优 draft length。

从系统架构角度拆解术语：
```
for k in [k_min, k_SLO]:
    k_accept = sum(prod(p_1..p_i))  # EMA 更新接受概率
    T_cycle = max(T_draft(k), T_pcie_init) + T_pcie_new(k) + T_verify(k+1)
    Θ(k) = k_accept / T_cycle
k* = argmax(Θ(k))
```
T_pcie_new(k) 通过分析 ELB vs current cache state 实时计算 |E_new(k)|，T_pcie_new = overhead + |E_new| × S_expert / B_PCIe。离线 SLO profiling 确定 k_max（TTFT budget 约束），在线搜索约束在 [k_min, k_SLO] 内。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 离线 profiled：T_draft(k) 线性拟合、T_pcie_init 常数、T_verify(k+1) 插值。在线 EMA 更新 p_i 跟踪 generation context 变化。
- 与 HRM 区别：HRM 指导 CPU/GPU operator placement；Amortization Roofline 指导 speculative draft length。两者互补。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
