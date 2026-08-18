## Thermal-Aware Refresh（温度感知刷新）与交错 Reed-Solomon ECC（3D-DRAM 高结温可靠性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Thermal-aware refresh 是 Raptor 针对 3D-DRAM 逻辑-DRAM 界面 105°C 结温的刷新策略：把刷新间隔从标称 32ms（HBM 级）缩到 4ms（8× 更密）以维持保留裕度，同时借助"深 bank 化、每 bank 仅 1364 行"（比常规 DRAM bank 少 16-32× 行）使刷新开销低于商品 DRAM——实测 4ms 刷新只损失 1.37% 带宽（99.26/102.13/103.57 TB/s 对应 1/2/4ms）。交错 ECC 是配套可靠性机制：每 bank 最后 8 列存一对 8-bit symbol 的 [144,140] Reed-Solomon codeword，按 subarray 映射交错，并与 stream-flipping 的 DBI metadata 共置；读路径先取 ECC+DBI metadata、ECC 纠正后按 DBI 翻转，写路径先算 ECC、选翻转极性、写数据再提交 metadata，另配周期背景 scrubbing 修复瞬时错误。额外收益：RowHammer 阈值 200K（旧工艺节点）下，t_RC=44ns 的连续激活需 8.8ms 才到锤击阈值，而 4ms 刷新小于该窗口 → 每行在邻居积累足够激活前就被刷新，RowHammer 被固有缓解，无需额外防护。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Raptor 热-可靠性设计中：热分析用 5 层串联热阻网络（thinned die stack 0.62mm → TIM1 k=5W/mK 100µm → 铜 lid k=390W/mK 1.5mm → TIM2 k=6W/mK 200µm → heatsink/airflow），冷却方案占 ~80% 总热阻、die stack 仅 +1.5%（0.003°C/W），因此 DRAM 堆叠不显著改变 Tj 对功耗的曲线；TE 阵列是热点（~92°C @ 106W/chiplet，优化 heatsink Rθ=0.10/Ta=35°C 下峰值 ~93°C、裕量至 ~140W，液冷 Rθ=0.02 使 Tj<60°C）。刷新调度按 bank 独立执行（每 bank 1364 行 → 单 bank 刷新延迟比常规 DRAM 低 16-32×），配合 16 个 channel 独立工作，刷新/scrub 不跨 channel 停顿；ECC+DBI metadata 共置在末 8 列，一次读先取 metadata 再取数据 flit，避免额外访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：刷新间隔按温度缩放（Tj>85°C 用 4ms、标称 32ms），由深 bank 化几何使每 bank 刷新开销可忽略；ECC 用符号级 [144,140] RS（8-bit symbol，可纠正符号错误），交错匹配 subarray 映射，与 DBI metadata 共置末 8 列；背景 scrubbing 周期巡检。使用方式：作为 3D-DRAM 在量产高温（422W/MCM、105°C 结温）下的可靠性组合——refresh 保证保持性、RS ECC 纠正制造/运行故障、4ms 刷新顺带消除 RowHammer；带宽代价可量化（4ms→1.37%）并在热预算内规划。价值：证明深 bank 化 + 刷新/ECC 共设计使 3D-DRAM 能在生产工况下稳定跑 100TB/s。
