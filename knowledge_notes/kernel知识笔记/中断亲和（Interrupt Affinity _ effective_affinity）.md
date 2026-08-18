## 中断亲和（Interrupt Affinity / effective_affinity）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 中断亲和是 Linux 中把每个 IRQ 绑定到特定核子集的机制。虽然名义上的 affinity mask 可包含多个核（默认所有核），但内核最终通过 effective_affinity 把每个中断绑定到单一核；effective_affinity 的更新是事件驱动的，可由 irqbalance 等守护进程根据运行条件和用户配置的 affinity mask 触发。结果：即使 mask 允许多个候选核，同一时刻中断投递也集中在单一核上。
- 在本文中：中断亲和是 Linux/x86 与 Arm 上 SPI 投递的默认行为（SPI 恒定路由到 affinity 指定的核，无论该核是否 idle）；Apple silicon + macOS 经 TIDE 反推证明**不采用**中断亲和（Observation 1），SPI 被均匀投递到所有 active core——这一差异使基于"监听固定核"的既有攻击假设失效，也让 Mwait/IdleLeak 类依赖 idle 唤醒的攻击不可行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Linux 中断投递流程（对比）：
```
# Linux（x86/Arm）：
#   设备产生 IRQ → 中断控制器 → 按 effective_affinity 选单一核 → 该核处理
#   /proc/irq/<irq>/smp_affinity 可配置 mask；irqbalance 按负载/拓扑动态改 effective_affinity
#   ⇒ 攻击者必须先确定目标中断被路由到哪个核，再在该核上检测
# Apple silicon + macOS（AIC）：
#   设备产生 SPI → AIC → 均匀投递给所有 active core（idle core 忽略）
#   ⇒ 攻击者任意 active core 上运行 TIDE 即可捕获 victim 中断，无需 core 绑定
```
- 论文验证（efficiency 指标 = 生成的 SPI 请求数 / 成功检测到的中断数）：默认设置下各机均可检测到三类 SPI；Mac mini 2023（8 核）上把 sender/receiver 用 CoreBinder 钉到不同核遍历全部 56 种组合，效率稳定在 10.73±0.20，与 sender 所在核无关——证明"均匀投递"而非"固定核亲和"。把 receiver 换成 idle 进程后，检测到的中断数随 active receiver 数成比例下降（idle 核被忽略）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Linux 内核 IRQ 子系统（smp_affinity / /proc/irq/）、irqbalance 守护进程；Apple 侧为 AIC 硬件均匀广播（无此配置接口）。使用：性能调优（把网卡中断绑定到专用核，如 Mellanox ConnectX 127 个 IRQ 绑定独立核避免与 memcached 争抢）、防御（把 SPI 绑到非攻击者核）。限制：macOS 用户态无 affinity 接口（需 CoreBinder kext 才能钉核）；Apple 的均匀广播使"绑定防御"失效。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
