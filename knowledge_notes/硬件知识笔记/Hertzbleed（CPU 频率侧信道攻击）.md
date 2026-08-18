## Hertzbleed（CPU 频率侧信道攻击）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hertzbleed（Wang et al., USENIX Security 2022，本文引用 [49]）是第一个把 x86 上的功率侧信道转化为远程时序侧信道的攻击：CPU DVFS 根据负载功耗动态调频，而功耗与数据相关（Hamming Weight/Distance），因此处理不同数据导致不同 CPU 频率、进而不同执行时间，攻击者通过测量执行时间（或频率）恢复数据。本文将其作为 TimeGaps 的对比 baseline：Hertzbleed 观测的是"稳态频率变化"，而 TimeGaps 观测的是"频率切换期间的 halted 时间"。
- 防御关系：Hertzbleed 依赖 CPU 频率可自由缩放，因此固定 CPU 频率或禁用 Turbo Boost 即可使其失效。本文关键贡献之一即证明固定频率只能堵住 Hertzbleed 这类频率信道，堵不住 iGPU 频率切换产生的 TimeGaps。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程（Hertzbleed 视角）：受害者执行数据相关运算（如 SIKE 解密，m^i≠m^(i-1) 时 Montgomery ladder 产生零值 stall、功耗下降）→ 包级/核级功耗随数据变化 → CPU DVFS 感知功耗变化调整频率 → 攻击者测量受害进程执行时间或读 CPU 频率（MSR_IA32_APERF/MPERF 或 cpufreq scaling_cur_freq）→ 按频率/时序推断数据。
- TimeGaps 对照：本文在默认 DVFS 下证明 TimeGaps 泄漏能力与 Hertzbleed 相当（HD/HW 区分、SIKE 密钥提取、Chrome 网站指纹 98.0%）；在固定频率下 Hertzbleed 失效而 iGPU 诱导的 TimeGaps 依旧（像素窃取 98.2%、指纹 92.2%、击键 precision 84.6%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Hertzbleed 用循环时序测量（执行时间）或 MSR_IA32_MPERF/APERF 比例估计频率；SIKE 攻击需把 CPU 压到接近热极限以放大数据相关频率差异。使用场景：破解 SIKE（后量子 KEM，已被密码学攻击破解但仍是侧信道标杆）、网站指纹等。缓解：固定 CPU 频率、禁用 Turbo、限制定时器精度。本文扩展：不依赖 CPU 加压、不需要热极限状态，仅监测 halted 时间即可达到相当效果。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
