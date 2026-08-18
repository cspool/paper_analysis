## 工作负载感知平台分析框架（T_B/T_S/T_C 三阈值与可行性/最优性）

术语解释
- 论文（RQ3，Sec. V）的框架：给定工作负载访问间隔分布 S(T) 与平台资源，定义三个阈值——DRAM 带宽阈值 T_B、SSD 带宽阈值 T_S、DRAM 容量阈值 T_C，用 max(T_B,T_S)≤T_C 判定平台"可行"，用 τ_break-even∈[max(T_B,T_S),T_C] 判定"经济学最优"，不满足时诊断瓶颈资源并给出升级路径。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 建模：设 l_blk 为块大小、N_blk 为工作集块数、τ_i 为块 i 的平均访问间隔，S(T)={i:τ_i≤T} 为访问间隔不超过 T 的块集；缓存的吞吐 Ψ_c(T)=l_blk·Σ_{i∈S(T)}1/τ_i、未缓存 Ψ_d(T)=l_blk·Σ_{i∉S(T)}1/τ_i；零拷贝 I/O 下 DRAM 带宽需求 B_DRAM^use(T)=Ψ_c(T)+2Ψ_d(T)（未命中=SSD→DRAM DMA+一次 DRAM 读）。三阈值：T_B=最小 T 使 B_DRAM^use(T)≤B_DRAM（DRAM 带宽可行）；T_S=最小 T 使 Ψ_d(T)<B_SSD（B_SSD=l_blk·N_SSD·IOPS_SSD，SSD 带宽可行）；T_C=最大 T 使 |S(T)|·l_blk≤C_DRAM（容量可行，等于第 K=⌊C_DRAM/l_blk⌋ 小的 τ_i）。
- 从系统架构角度拆解术语：该框架把"数据放置"组织为可计算的资源可行性判定——平台对工作负载是否可行（三个资源维度同时满足）、最优 DRAM 容量是多少（C_DRAM^(V)=|S(T_v)|·l_blk、C_DRAM^(O)=|S(T_o)|·l_blk，T_v=max(T_B,T_S)、T_o=max(τ_break-even,T_v)），以及不满足时的升级方向（T_B>T_C≥T_S→加 DRAM 带宽；T_S>T_C≥T_B→加 SSD 吞吐/主机 IOPS；两者都超→加容量或降 max(T_B,T_S)）。定量结论（Fig. 6）：CPU+DDR 下 T_v=T_S（SSD IOPS 决定可行 DRAM），GPU+Storage-Next 下 T_B、T_S 都 <5s、可行 DRAM 需求低。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为配置工具——输入平台参数（DRAM 带宽/容量、SSD IOPS、主机 IOPS、成本）与工作负载 profile（访问间隔分布、读写比、总吞吐），输出可行性、最优 DRAM 容量与升级建议。论文用 lognormal 访问间隔分布、1B 块、总吞吐 200GB/s 做定量演示（CPU+DDR: DDR5-5600×12=540GB/s、100M IOPS；GPU+GDDR: GDDR6-20×8=640GB/s、400M IOPS；各 4 块 SSD、99 分位 13/17/26/44μs→ρ=90%）。论文未开源该框架实现。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
