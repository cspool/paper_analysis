## iGPU 动态频率切换（iGPU DVFS / GT frequency transition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- iGPU 动态频率切换是 Intel 集成显卡（Graphics Technology，GT 域）的动态频率缩放机制：iGPU 负载升高时升频、负载下降时降频，利用未使用的包级功率/热余量提升应用性能（Intel Integrated Graphics Developer's Guide 与 Intel graphics dynamic frequency 文档化）。本文首次发现并验证：iGPU 频率切换会在 CPU 侧产生可观测的 TimeGaps——即使 CPU 频率固定、无 P-state 切换，iGPU DVFS 仍出现，且 TimeGaps 跨所有 CPU 核同步（i5-8259U 2300MHz 上 iGPU 频率在 1017–1050 MHz 波动时出现）。机制推断：iGPU DVFS 与 CPU P-state 由同一包级 PMU 协调（8th-Gen U 平台 datasheet 把 IA 域与 GT 域控制/状态寄存器同置于包级 PMU 寄存器空间；RAPL 把 iGPU 当作包级功率域），iGPU 请求新频率时 PMU 为保持安全时序/电压裕量短暂挂起 CPU 核。这与 prior work（Wang et al. 需把 CPU 压到热极限使 iGPU 负载成为压垮频率的最后一击）不同——本文无需 CPU 加压。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程：iGPU 渲染/OpenCL kernel 负载上升 → iGPU 请求更高频率 → 包级 PMU 协调 IA（CPU）与 GT 域 → PMU 短暂挂起 CPU 核（电压/时序裕量保护）→ CPU 恢复，TSC 跳变形成 TimeGap → 攻击者在任意 CPU 核记录到。条件性：仅当 iGPU 被用作默认显示设备时出现（i5-8259U/i7-9750H/i3-10100+iGPU 有，dGPU 独占显示时无，i9-10940X 无 iGPU 时无）。区分于 OS idle：CPU 核保持 active、C1/C1E residency 增量为零。
- 实验例子（Section III-D）：固定 CPU 频率 2300MHz，每 1 秒在 iGPU 执行一次 vector_add OpenCL kernel（0.1s 持续）→ TimeGaps 与 iGPU 频率切换对齐（Fig.3）；用 OpenCL 批量提交 kernel（CPU 仅提交+等待）把 iGPU 频率限制在 1017–1050 MHz 波动、CPU 侧活动最小，隔离出纯 iGPU DVFS 行为。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/观测：iGPU 频率经 sysfs 文件 gt_cur_freq_mhz 读取（约 0.3ms 更新，部分采样）或经 perf_event_open 跟踪 Intel i915 驱动的 PMU 事件（5ms 更新）；iGPU 功耗经 perf_event_open 的 PMU 事件 power/energy-gpu 读取；TimeGaps 用 rdtscp 采集器在 CPU 上记录。使用场景：像素窃取（渲染不同像素值产生不同 iGPU 负载→不同 TimeGaps，固定频率下 98.2% 高区分度、4.38 s/pixel）、击键检测（击键触发屏幕渲染→iGPU 频率切换→TimeGaps，precision 84.6%）、固定频率网站指纹（Chrome 92.2%）。缓解困难：i915 驱动下即使 gt_min_freq_mhz=gt_max_freq_mhz 锁死 iGPU 频率，有 iGPU 负载时频率仍波动，故该向量难以从用户态缓解。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
