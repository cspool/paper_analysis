## Victim-based counting（受害行计数）与 Aggressor-based counting（攻击行计数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 两类 RowHammer 缓解的计数语义，区别在于"计数器记录的是谁"：Aggressor-based counting 在每个激活的 aggressor 行自己的计数器上加 1，用于判断"这行锤了多少次"（PRAC 及 Chronus/QPRAC/MOAT 都属此类）；Victim-based counting 在被激活行激活时，给它 blast radius（BR）内的 victim 行计数器各加 1，并把被激活行自身计数器清零，记录的是"这行被邻居锤了多少次"（PVAC、ProTRR 的 victim 侧计数、ProHIT/MRLoc 等）。逻辑链：RowHammer 的物理机制是 aggressor 反复激活-预充电使相邻 victim 行位翻转，因此与物理扰动机制对齐的计数器语义应落在 victim 侧；aggressor 计数存在结构不对称——计数器在每次激活（含良性 REF）都递增、只能靠 RFM 重置，且 BR 内多个 aggressor 的扰动在 victim 处累加，aggressor 侧必须按 BR 保守压低阈值，导致计数器空闲也饱和、虚假 Alert 与级联 RFM。NRH（aggressor 侧最大锤击次数阈值）与 NRHv（victim 侧最大被锤计数阈值）是同一物理现象的两个视角，BR 越大二者差距越大。Web 证据：ProTRR（IEEE S&P 2022，ETH ComSec）的 victim-based counting + ProMG 频繁项近似；PVAC arXiv:2604.20576 的系统对比（Table IV：PVAC 是唯一 victim 计数 + ACT 时重置计数器的 PRAC 变体）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 DRAM 芯片内，计数语义直接决定计数器阵列的读写模式与安全阈值推导。运转流程对比（BR=2，行 r 被激活）：Aggressor 版（PRAC，Algorithm 1）——PRE 时对 cnt[r] read-modify-write 递增；cnt[r]>=NBO 时入 RFM 目标队列并触发 Alert；RFM 时对 r 的 Victims(r)（r±1,r±2）各做一次 RFM-induced ACT（递增它们计数），然后把 cnt[r] 清零。问题：每次 REF 也递增 cnt[r]，空闲 bank 计数器在约 63 个 tREFW 内线性到 NBO=64，触发 RFM 级联使有效带宽从 92.44% 崩到 20.56%。Victim 版（PVAC，Algorithm 2）——ACT(r) 时置 cnt[r]=0，对 r±1、r±2 各 cnt[v]++；任一 victim 计数>=NBO 才入队触发 Alert；RFM 时直接刷新队列里 4 个最高 HC 的 victim 行。因为每次激活都重置被访问行，计数器在正常刷新下有界，空闲 bank 不再触发虚假 Alert。安全阈值推导也随之改变：在相同最大 victim hammered count（HC）约束下，victim 计数不需要除以 BR——HC=128、NMit=1/2/4 时 PVAC 的 NBO=85/102/108，而 PRAC 仅 3/15/19、Chronus 31；HC=64 时 PRAC-1/PRAC-2 甚至无法配置安全 NBO，PVAC 仍可用 NBO=43。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：两种语义都要 per-row 计数器（8-bit 常见）；victim 计数每次 ACT 更新 2×BR+1 个计数器（PVAC 用 CSA 并发完成，见 CSA 条目），aggressor 计数每 PRE 更新 1 个计数器（代价是 PRAC 时序参数 tRP 16→36ns）。victim 计数还需维护"哪行被锤最多"的排序信息用于选 victim 刷新——PVAC/ProTRR 用 top-K 优先级队列 / Misra-Gries 频繁项近似（ProMG）来近似 exact victim 计数。使用：victim 计数天然把"计数目标"与"防护目标"统一，可直接配置更大 NBO、在更低 HC 下仍安全，且每次 ACT/REF 的天然重置消除了 DoS 性虚假 Alert；已被 ProTRR（S&P'22）、PVAC（ISCA'26）用于形式化证明/实证最优的防御；攻击侧对应最优攻击是 feinting/wave attack（见该条目）。论文未明确说明 victim 计数在非对称 BR（双侧不同）或 ColumnDisturb 等超宽扰动下的适配。

涉及论文标题：
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
