## PRAC（Per-Row Activation Counting，含 ABO/ALERT/RFM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- JEDEC 为 DDR5 及后续 DRAM 引入的逐行激活计数 RowHammer 防御框架：DRAM 芯片内每行一个计数器，行激活计数达到阈值 N_BO 时进入 Alert Back-Off（ABO）协议——DRAM 拉高 ALERT 信号请求内存控制器发 RFM（Refresh Management）命令，DRAM 借此做缓解刷新。RFMab（All-Bank）会停顿全通道内存请求约 350ns（每 ABO 配 2/4 次 RFM 时 700-1400ns），故 ABO/RFM 是内存通道粒度、保守的全通道停顿机制。Web 后续研究确认其代价与改进方向：计数更新抬高 tRP/tRC 使平均性能降约 14%；QPRAC（优先队列）、PRACtical（subarray 级计数更新 + RFM_MASK 的 bank 级恢复隔离）、PRACLeak（ABO 时序侧信道 → TPRAC）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- PRAC 是把 RowHammer 防御下沉到 DRAM 芯片的工业标准接口，ColumnKeeper 大量与之交互，有三层关系：(i) **朴素改造失败**（动机基线 §2.3.2）：把 ABO 阈值降到 N_CD/S 并在触发时刷新 3 个子阵列全部行来防 ColumnDisturb——N_CD=1M、S=1024 时阈值仅 1K，即每 1K 次激活要刷 3K 行，性能/能耗不可接受；(ii) **in-DRAM 版 CK-D 复用其信令**（§8）：RPT/CT-O/CT-E 放 subarray 旁，bank 内 SP/SHC 寄存器跟踪最高计数 subarray，SHC 超 N_PR 时拉 ALERT → 控制器发 RFM → bank 按 SP 定位并一次刷 7 行；(iii) **控制器版与 PRAC 组合的盲区**：控制器不知道 RFM 影响了哪些 subarray，只能保守递增全 rank 计数器，产生多余刷新（in-DRAM 版在 16K 阈值下保持 0.97 IPC，而控制器版 0.84）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：JEDEC DDR5 标准特性，RFM 命令率可配置（如 Mithril 最高每 16 次激活 1 次 RFM）；使用场景是低阈值 RowHammer 防御与（本论文）ColumnDisturb 防御的信令底座。注意两点副作用：计数器逐激活更新抬高 tRP/tRC；ABO 停顿全通道可观测（PRACLeak 侧信道）。DejaVu（ISCA 2026）把 PRAC/PARA 作为"阈值依赖型缓解"的代表，在 Ramulator 2.0 中对 N_RH 做 −20%…+20% guardband 扫描量化 DejaVu 引起的阈值不确定性代价：N_RH=64 时 −20% 使 PARA 平均性能开销 6.3%、PRAC 相对较小（其开销主要来自计数更新抬高的时序参数）；+20% 使 PARA 性能提升 7.8%、PRAC 提升 2.1%（60 个四核 mix 几何平均）。

Sigries 视角（ISCA'26）：PRAC 已被 DDR5 采纳（可选特性）并进入 LPDDR6，且计划进入 DDR6 与 HBM5，但其距广泛使用仍需数年——一家主要 DRAM 厂商明确表示要等 DDR6 强制要求才支持 PRAC；PRAC 代表重大架构变更，早期产品可能有 bug，效果还依赖正确配置的参数与阈值。因此 Microsoft 即使使用 PRAC-enabled DRAM 也会保留 Sigries 作为 fallback 机制，并把 Sigries telemetry 作为独立信号帮助区分持续攻击与良性/瞬时事件。Sigries 评估中 PRAC 是 7 个对比防御之一（每行计数器放 DRAM、commodity workload 下无行超阈值故不动作，性能与 Sigries/Graphene 相当）。

PrISM 视角（ISCA'26，Loaded Dice 论文）：把 PRAC 作为高性能/高面积开销的对比 baseline，以 QPRAC（HPCA 2025，[101]：每个 Alert 触发 1 个 RFM + 5 项优先级服务队列，Back-Off 阈值按目标 TRH-D 调优）作为安全 PRAC 实现。量化 PRAC 开销并随接口速率放大：tRP 16→36ns（2.25×）、tRC 48→52ns，行缓冲冲突延迟 +42%、每 bank 激活率 −8%；DDR5 3200 MT/s 时 2.2% 平均 slowdown，8000 MT/s 时升至 14%（高内存强度 workload 21.8%），因高数据率下更短的 tRRD/tFAW 让控制器更快跨 bank 发激活、更多次暴露 PRAC 的固定时序惩罚；DSAC 估计每行 counter cells + 更新逻辑使 DRAM core 面积增约 9%。结论：PRAC 提供精确 per-row 追踪（RFM 频率近零）但受计时膨胀主导的开销，与概率方案（MINT/PrISM）的"缓解频率主导"开销形成对照；PrISM 以 0.2%（TRH-D=500）与 1.5%（250）平均 slowdown 全面优于 PRAC 的 ~14%，且无 per-row 计数器与面积开销。

PVAC 视角（ISCA'26，Per-Victim-row hAmmered Counting）：PVAC 指认 PRAC aggressor 计数的根本缺陷——计数器在每次激活（含良性 REF）单调递增、仅显式 RFM 才重置，导致空闲 bank 计数器也在 tREFW 内线性累积：模型显示 NBO=64、NMit=4、BR=2 时第 63 个 tREFW 计数器达阈，RFM 级联使有效带宽从 92.44% 崩到 20.56%（无恶意访问也 DoS）；且重置嵌在 DRAM 行内的 aggressor 计数器（BR=2）需至少 5 次串行激活（≈4×tRC=208ns），PRAC 被迫把重置拖到 Alert 的 350ns 窗口。PVAC 改为 victim 计数（每次 ACT 重置被激活行、递增 BR 内 victim 行计数），并用专用 Counter Subarray（CSA）在默认 DDR5 时序下并发完成每次 ACT 的 5 个计数更新（总 35.1ns 藏进 tRC=48ns，不付 PRAC 的 tRP 16→36ns 惩罚），从而消除虚假 Alert、在相同最大 HC 约束下配置更大 NBO（HC=128、NMit=1/2/4 时 PVAC 85/102/108 vs PRAC 3/15/19），性能与能耗全面优于 PRAC/QPRAC/MOAT/Chronus（HC=2048 时能耗仅 +4.08% vs PRAC +15.4%）。其最小安全 HC 也更低：HC=64 时 PRAC-1/PRAC-2 已无法配置，PVAC 仍可用 NBO=43。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
