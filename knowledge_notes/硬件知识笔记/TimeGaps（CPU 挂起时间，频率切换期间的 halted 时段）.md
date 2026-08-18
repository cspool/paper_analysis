## TimeGaps（CPU 挂起时间，频率切换期间的 halted 时段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TimeGaps 指程序执行期间"时间戳计数器（TSC）在推进、但 CPU 实际挂起不做任何计算"的时段（ISCA'26，南方科大/北大/Georgia Tech 等）。其存在由 Intel Power Management Guide 文档化：P-state 切换时 CPU 核暂停取指执行直到电压稳定。本文系统化确认两大根因：(1) CPU P-state（频率）切换——Skylake 系共享时钟域，所有核同时停止，TimeGap 跨核同步（起止差平均 46.21 cycles、时长差 0.69%）；(2) iGPU 频率切换——首次发现，即使固定 CPU 频率，iGPU DVFS 仍经包级 PMU（power management unit）协调短暂挂起 CPU 核（见"iGPU 动态频率切换"条目）。量化：空闲系统上 TimeGaps 占 >1% 总时间（i7-9750H 10s 中 1.53%，约 0.4B/26B cycles），出现频率数百次/秒；不同频率切换对产生特征性时长分布（4.0→4.1GHz 约 35k cycles、4.4→4.3GHz 约 52k cycles）。
- 安全影响：TimeGaps 构成新的非特权侧信道——在默认 DVFS 下泄露能力与 Hertzbleed 相当（网站指纹 Chrome 98.0%），在固定 CPU 频率（Hertzbleed 失效的场景）下仍可泄露 iGPU 指令/操作数级信息，复活像素窃取、网站指纹与击键检测三类此前被认为已被固定频率缓解的攻击。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程：CPU/iGPU 请求新频率 → 包级 PMU 协调（保持安全电压/时序裕量）→ 共享时钟域的 CPU 核停止取指执行（电压不稳定期间）→ 时钟/电压稳定后恢复执行 → TSC 因以固定速率计数而在挂起期间继续推进 → 程序下一次读 TSC 时观察到异常大的跳变。iGPU 场景的证据链：固定 CPU 频率下只有 iGPU 驱动显示的平台出现 TimeGaps（i5-8259U/i7-9750H/i3-10100+iGPU，dGPU 独占显示时消失）→ 排除 OS idle（C1/C1E residency 增量恒为 0）→ 包级能量 MSR（MSR_PKG_ENERGY_STATUS/MSR_PP0_ENERGY_STATUS）显示含 TimeGap 区间 package 功率代理平均低 18.2%、core 仅低 2.9%——即功率降幅在包级而非核级，指向 iGPU DVFS 触发的包级协调。
- 检测流程（硬件视角）：反复执行 `rdtscp`（读 TSC）→ 相邻读数差 >5000 cycles（阈值用于忽略 cache/TLB miss）记为一个 timestamp jump → 用 PMC（CPU_CLK_UNHALTED.THREAD/REF_TSC 是否推进）判定 halted vs unhalted → halted 即为 TimeGap。非特权场景用 SegScope（GS 段寄存器）或浏览器空循环计数替代 PMC。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：真实 Intel 硬件（6th-10th 代 Skylake 系，如 i5-8259U/i7-9750H/i7-7700/i3-10100/i9-10940X），C 采集器 gaps_collector_pmc/gaps_collector_5ms + user_rdpmc 内核模块（开放 PMC）+ Python/JS 采集。artifact 开源：Zenodo https://doi.org/10.5281/zenodo.19450827（MIT）。使用场景：侧信道攻击（网站指纹、像素窃取、击键检测、SIKE 密钥提取、covert channel 50bps）与性能分析（I-DVFS/SUIT 从性能角度研究 halted 时间）。缓解：固定 CPU 频率仅消除 CPU 型 TimeGaps；iGPU 型无法从用户态可靠固定（即使 gt_min_freq_mhz=gt_max_freq_mhz 仍波动）；限制定时器、硬件降低 P-state 切换延迟（3rd 代 Xeon 起 ~12µs→近 0）或改用 dGPU 显示可缓解，但均不保护旧平台。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
