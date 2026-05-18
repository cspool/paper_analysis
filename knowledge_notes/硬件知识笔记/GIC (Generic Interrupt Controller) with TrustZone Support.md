## GIC (Generic Interrupt Controller) with TrustZone Support

术语是什么？通过联网搜索让回答具体和精准。

GIC（Generic Interrupt Controller）是ARM架构中的标准中断控制器IP，管理所有外设中断的优先级、分发和路由。TrustZone-aware GIC支持将每个中断源配置为Group 0（Secure中断，以FIQ方式由Secure World处理）或Group 1（Non-Secure中断，以IRQ方式由Normal World处理）。Secure中断不能被Normal World软件拦截或屏蔽，确保TEE系统可靠响应安全事件。这是TZ-LLM co-driver NPU设计中secure NPU job completion signaling的基础——NPU完成中断必须安全路由到TEE，不能被REE截获。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

GIC在TZ-LLM co-driver NPU中的中断路由切换流程：

1. 正常REE NPU job执行：NPU_IRQ→Group 1→路由到Normal World REE driver→REE处理completion
2. Secure NPU job启动前：TEE重新配置NPU_IRQ→Group 0→路由到Secure World TEE。Normal World无法拦截此中断
3. Secure NPU job执行完成→NPU发中断→GIC路由到Secure World→CPU进入EL3 Secure Monitor→确认已在Secure World→直接转发到TEE OS→TEE NPU data plane driver interrupt handler处理
4. Secure job完成后恢复：NPU_IRQ→Group 1→后续REE job恢复正常路由

关键机制：(1)中断分组——每个SPI可通过GICD_IGROUPRn寄存器配置为Group 0/1，Secure World可修改Normal World不可；(2)中断优先级——Secure中断可抢占Non-Secure中断但反过来不行；(3)中断状态隔离——Secure/Non-Secure中断pending状态独立管理，防止信息泄漏。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常见实现包括ARM CoreLink GIC-400 (GICv2)、GIC-500/600 (GICv3)，所有支持TrustZone的ARM SoC均集成。Rockchip RK3588集成TrustZone-aware GIC，TEE OS通过GIC驱动的Secure World接口配置中断分组。TZ-LLM中GIC配置是co-driver安全机制的关键一环，每次NPU world switch通过TEE OS安全驱动完成。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

