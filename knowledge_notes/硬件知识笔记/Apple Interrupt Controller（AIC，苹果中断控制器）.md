## Apple Interrupt Controller（AIC，苹果中断控制器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AIC 是 Apple 自研的私有中断控制器，替代传统 Arm 架构的 Generic Interrupt Controller（GIC）。从 M1 Pro 起为 AICv2，采用基于硬件的机制把中断分发给"愿意处理"的核。闭源设计：公开信息主要来自 Asahi Linux 文档（https://asahilinux.org/docs/hw/soc/aic/）与内核 DT binding（https://www.kernel.org/doc/Documentation/devicetree/bindings/interrupt-controller/apple,aic2.yaml），仅提供简短功能描述；AIC 也是 Windows Arm64 无法原生运行在 Apple silicon 上的原因之一。
- 在本文中：TIDE 只检测本核中断，跨核攻击需要理解 SPI 投递；论文用 TIDE 反推 AIC 行为，得出两条观测：Observation 1——Apple silicon + macOS 不采用 Linux 式中断亲和，SPI 不投递到固定核；Observation 2——SPI 被均匀投递到所有 active core、忽略 idle core（P-core 与 E-core 一视同仁）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- SPI 投递流程（网络中断为例，Mac mini 2023，8 核）：
```
① sender 进程以 200 kHz 频率 UDP 发包（1B 或 9KB）
② 网卡产生 SPI → AIC 决策目标核
③ AIC 将 SPI 均匀投递给所有 active core（idle core 不收）；与 sender/receiver 所在核无关
④ 每个 active core 上的 receiver（TIDE 检测）记录中断数
⑤ 指标：efficiency = 生成的 SPI 请求数 / 成功检测到的中断数
```
- Annotations：③是本文反推的核心结论——Linux 上 SPI 经 effective_affinity 绑定单一核，Apple 上均匀广播；③还解释了图 5：56 种 sender/receiver 核组合效率稳定在 10.73±0.20（avg±std）；图 6：把部分 receiver 换成 idle 进程，检测到的中断数随 active receiver 数成比例下降，证明 idle 核被忽略；⑥额外发现 macOS 会把多个资源请求聚合进一次中断以降低上下文切换开销（效率可 >1，网络 6.03–180.70、鼠标 0.76–1.13、键盘 0.56–4.32）。
- 攻击侧含义：攻击者无需把代码钉到特定核（macOS 用户态本就无 affinity 接口），任意 active core 上运行 TIDE 即可收到 victim 中断；同时，AIC"只投 active core"的设计使依赖 idle 唤醒的 Mwait/IdleLeak 类攻击在 Apple silicon 上不可能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：M 系列 SoC 硬件中断控制器；驱动在 XNU 与 Asahi Linux（linux 内核 apple,aic2 绑定）中。使用：系统中断分发（网络/键盘/鼠标/GPU/定时器）；对攻击者是 SPI 投递策略的裁决者。限制：闭源，无法直接配置 affinity；用户态无法可靠触发/绑定 SPI（论文用 UDP 发包、CGEventRef 模拟输入触发）。
- 注意：本知识库另有条目"AIC"指 AI Cube/AI Vector（ML 加速器单元），与本文 Apple Interrupt Controller 是不同概念。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon
