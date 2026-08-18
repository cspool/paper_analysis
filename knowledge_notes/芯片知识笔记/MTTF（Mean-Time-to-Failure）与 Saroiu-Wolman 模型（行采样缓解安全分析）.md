## MTTF（Mean-Time-to-Failure）与 Saroiu-Wolman 模型（行采样缓解安全分析）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MTTF（平均失效时间）是 RowHammer 缓解的安全量化指标：以"诱发一次位翻转的平均时间"衡量防御强度。PrISM 论文以每 bank MTTF 10,000 年为目标（与 DRAM 固有 soft-error 率相当，32-bank 系统级 MTTF 约 417 年），并做 1K–1M 年敏感性分析（表 III）。Saroiu-Wolman 模型（Saroiu & Wolman, "How to Configure Row-Sampling-Based RowHammer Defenses", DRAMSec 2022，[79]）是行采样类缓解的标准安全分析模型：给定每窗口缓解概率 P_m、窗口大小 W 与刷新窗口 tREFW，推导所需最小 RowHammer 阈值 TRH-D，使攻击者在 tREFW（32ms）内对任一 aggressor 行积累的未缓解激活数低于翻位阈值。PrISM 用它将 (W,R,L) 参数组合的每窗口缓解概率转换为最小支持 TRH-D，并在 circular-X-rows 最坏攻击模型（X∈[W,(L+1)W] 扫描）下取最大值作为该配置的安全界。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 分析流程（芯片设计参数→安全界的完整链路）：给定 (W,R,L) → 用 SHQ 占用稳态固定点模型（两状态马尔可夫链，附录 B：P_in=(R−1)/W，P_SHQ = L·P_in/(1+L·P_in)）求持久 aggressor 在 SHQ 中的驻留概率 → 由式 (5) 得每窗口缓解概率 P_m = (1−P_SHQ^R)/W（默认缓解）+ (R/W)·P_SHQ（交集缓解资格）→ 代入 Saroiu-Wolman 模型求支持的最小 TRH-D → 在 circular-X-rows 攻击（X 个 aggressor round-robin，row(t,s)=(tW+s) mod X，单参数 X 覆盖"锤击率 vs 交集逃逸"权衡）下扫 X∈[W,(L+1)W] 取最大值 → 用 5M-epoch Monte Carlo 验证解析模型（图 5/6 吻合）。考虑 PMQ 延迟缓解后实际阈值 = 基础阈值 + TPMQ + ABO_ACT(Q)（默认 16 项 PMQ、TPMQ=4 时 +16），并计入 Feinting/Wave 式 chained-Alert 攻击的额外激活。
- 用途：用分析模型确定每个目标 TRH-D 的 (W,R,L) 配置（表 II），避免对每种配置做全量攻击仿真；MTTF 敏感性（表 III）展示同一配置在不同安全目标下的支持阈值变化（如 66 项 SHQ 配置在 10K→1M 年 MTTF 下支持阈值 731→786）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：安全分析部分开源为纯 Python（`prism/security_analysis/`，prism_circular_security_analysis.py + plot_figure5.py），输入 (W,R,L) 与 MTTF 目标，输出最小支持 TRH-D（图 5/6）与 MTTF 敏感性表（表 III）；Monte Carlo 以百万级窗口 epoch 验证解析模型。使用场景：为行采样类 in-DRAM 缓解配置安全参数、量化最坏攻击下的支持阈值，是概率缓解论文的标准安全论证方法。

涉及论文标题：
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
