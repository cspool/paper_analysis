## Performance Binning（性能分档）与 SECC（Sellable Effective Compute Capacity，总可售有效算力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Performance Binning 是 ConBin 提出的 WSC 分档新范式：以芯片在代表性目标负载上实测的性能（论文用平均 latency）而非最大频率/核数划分产品档位（bin），使产品档位与"可保证的性能水平"对齐。配套指标 SECC = Σ P_k·Y_k（Eqn.1）：P_k 是 bin k 的保证性能水平（由其性能阈值决定），Y_k 是该 bin 的良率占比（由芯片性能分布 f(p) 得出）。逻辑链：WSC 执行时间 = 计算 + 通信，通信对缺陷引起的拓扑不规则高度敏感（非线性能退化，Fig.1）→ 按核数/频率分档与实际性能脱节 → 性能方差大时，阈值要么保守（P_k 低）要么激进（premium bin 的 Y_k 骤降），两种情况下 SECC 都被方差封顶 → 若把性能分布收敛，同一功能良率可支撑更激进阈值 + 更高 premium-bin 占比，SECC 扩大。Web 证据：工业 binning 本质即"同一 die 按可承受频率/功耗/核数分档销售"（Intel/AMD/Apple 核数档、NVIDIA B100/B200 功耗档），与 ConBin"WSC 需按实测性能分档"的论断形成对照。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
ConBin 的分档流程：修复后每芯片在代表性负载上测平均 latency → 按性能降序排列、离散为 N=100 个百分位 → 动态规划求最优阈值：DP[b][i] = max_{j<i}{DP[b-1][j] + τ_{j+1}·(i-j)}（Eqn.8：用恰好 b 个 bin 覆盖前 i% 芯片的最大 SECC；τ_{j+1} 为第 (j+1) 百分位的性能阈值，(i-j) 为落入 bin b 的芯片比例）→ 输出 B 个 bin 的阈值（bin 数 B 由厂商定义，论文用 2/3/4/8）。复杂度 O(BN²) 时间 / O(BN) 空间，N=100 固定故可忽略。效果：128×136、8 bins 下 premium-bin（前 1/3 档）yield 2.80×、阈值收紧 2.09×、SECC 2.64×（vs CB*+SK*），Ours-ALL 最高 3.85×；bin 数 4→8 时 SECC 显著上升（更细的 bin 让 bin 感知映射/调度更精确地利用每片潜力）。注意：CB*+SK* 虽靠宽松阈值也提高 premium-bin yield，但宽松阈值限制每 bin 保证性能、SECC 受限——"阈值更激进且不损失 yield"才是性能收敛带来的真收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 = 产线端到端流程：设计期冗余模板 → 制造后修复 → pre-binning → 每芯片映射/调度 → 负载实测 → DP 阈值计算 → 分档出货。论文用 latency 作性能指标，并指出可推广到吞吐/功耗/复合 QoS（只要有对应收敛目标）。与本知识库"功能 binning"条目（CAPA）的区别：功能 binning 按缺陷禁用模块数分档、不改变性能定位；Performance Binning 直接以实测性能为分档依据，且前置了专门的性能收敛机制（硬件冗余 + 软件 bin 感知优化）。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
