## Shared Peripheral Interrupt（SPI，共享外设中断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SPI 是由处理器核外部设备（网卡、键盘、鼠标、GPU）产生、可投递到一个或多个核处理的中断类型，区别于核间中断（IPI）等处理器触发的中断。SPI 是中断侧信道攻击的主要载体：攻击者通过检测 SPI 的到达时机/模式推断 victim 行为（击键、网站访问、视频播放、GPU 活动）。Linux 上 SPI 由 IRQ affinity 路由到固定核；Apple silicon 上由 AIC 均匀投递到所有 active core（见 AIC 条目）。
- 在本文中：SPI 是 TIDE 检测的目标信号，覆盖网络中断、鼠标中断、键盘中断三类；反推实验证明 Apple 均匀投递 SPI（Observation 2），且 SPI 是 Apple 系统跨核攻击的主要载体（处理器触发的中断数量有限）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- SPI 从产生到被攻击者观测的完整流程（网站指纹场景）：
```
① victim 访问网站 → 网卡收发数据包（不同网站数据包数量/时序不同）
② 每个网络包触发一个 SPI（可能被 macOS 聚合）
③ AIC 把 SPI 均匀投递给所有 active core（本机 4 E + 4 P 核）
④ 攻击线程（任意 active core）用 TIDE 记录每次中断（x18 清零）及相邻中断间隔
⑤ 输出一条 TIDE trace（中断间隔序列）→ LSTM(32 units) 分类 → top-1 网站（closed-world 93.8%）
```
- Annotations：②的"时序不同"是分类特征来源（图 7 显示不同网站 trace 模式明显可区分）；③使攻击无需 core 绑定（与 x86/Linux 需要先确定处理核不同）；④TIDE 计数受 CPU 频率影响（消融实验用固定 24 MHz cntvct_el0 验证，准确率基本不变）；⑤分类器沿用 Cook et al. [7] 的开源实现。
- 触发实验设置：网络中断用 UDP 1B/9KB 包、200 kHz；鼠标/键盘用 C 程序调 CGEventRef 模拟（5 kHz）；efficiency 指标衡量"生成的 SPI 请求数/成功检测中断数"（表 II，网络 6.03–180.70、鼠标 0.76–1.13、键盘 0.56–4.32）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：设备（网卡/输入设备）→ 中断控制器（GIC/AIC）→ 内核 IRQ 子系统 → 核处理。使用：攻击者监听 SPI 做指纹/隐蔽信道/密钥提取；防御者依赖中断亲和把 SPI 绑到非攻击者核（Linux 可行，Apple 不可行——SPI 均匀广播）。限制：macOS 不提供 /proc/interrupts 类中断统计接口，软件检测失效，只能靠 TIDE 式硬件/架构信号。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
