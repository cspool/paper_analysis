## Arm TrustZone

术语是什么？通过联网搜索让回答具体和精准。

Arm TrustZone是ARM架构中的硬件级安全扩展技术，自ARMv6架构引入。它将处理器执行环境划分为两个安全世界：Secure World（安全世界）和Normal World（非安全世界/REE, Rich Execution Environment）。TrustZone通过硬件实现的隔离机制确保即使Normal World的操作系统内核被攻陷，也无法访问Secure World的代码、数据和硬件资源。TrustZone的核心硬件组件包括：处理器核的NS（Non-Secure）状态位（通过AXI总线AxPROT信号传播）、TrustZone Address Space Controller (TZASC)用于外部DRAM内存分区、TrustZone Protection Controller (TZPC)用于外设安全配置、以及Generic Interrupt Controller (GIC)支持安全和非安全中断独立路由。在ARMv8-A及之后架构中，世界切换通过EL3（Exception Level 3，最高特权级）的Secure Monitor实现，由SMC（Secure Monitor Call）指令触发。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TrustZone硬件架构关键组件和数据流（以TZ-LLM在RK3588上的secure LLM推理为例）：

1. **处理器核安全状态**：每个ARM核心内部维护NS状态位，通过AXI AxPROT信号传播到SoC互联。所有总线事务被标记为Secure或Non-Secure，从设备（内存控制器、外设、中断控制器）据此区分访问来源。

2. **内存分区**：TZASC位于DRAM控制器前端，根据物理地址范围和访问安全属性允许/阻止内存访问。TZ-LLM中，TZASC被动态配置——TA调用extend_protected时TEE OS配置TZASC将CMA分配的连续物理内存标记为Secure-only，确保明文模型参数不被REE读取。

3. **外设保护**：TZPC配置NPU等外设的安全访问权限。TZ-LLM co-driver NPU执行secure job前，TEE用TZPC阻止REE访问NPU MMIO寄存器，防止REE窥探或篡改secure job。

4. **中断路由**：GIC支持将每个中断源配置为Group 0（Secure, FIQ）或Group 1（Non-Secure, IRQ）。TZ-LLM在secure NPU job执行前将NPU完成中断路由到TEE，完成后恢复路由到REE。

5. **安全启动链**：从Boot ROM开始逐级验证固件完整性（ROM→FSBL→Trusted Firmware→TEE OS），确保TEE OS和TA未被篡改。TZ-LLM依赖此信任链保证LLM TA在TEE中的完整性。

TZ-LLM在RK3588上的具体硬件流程：LLM TA启动→REE CMA分配连续物理内存→TEE OS配置TZASC将内存标记为Secure→LLM TA在此secure memory中解密并使用模型参数→NPU job执行时TEE配置TZPC+TZASC+GIC建立安全执行环境→job完成后恢复Non-Secure配置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TrustZone在所有主流ARM移动SoC中部署。TEE OS实现包括：(1) Qualcomm QTEE：闭源，广泛用于Android设备Widevine DRM、生物识别、Keymaster等；(2) ARM Trusted Firmware (TF-A)：开源参考实现，提供EL3 Secure Monitor；(3) OP-TEE：开源TEE OS，支持GlobalPlatform API；(4) OpenHarmony TEE：华为OpenHarmony的TEE子系统，TZ-LLM原型基于此实现。编程模型：Normal World应用通过GlobalPlatform TEE Client API调用TEE；Secure World内TA通过TEE Internal Core API访问安全服务。世界切换通过SMC指令触发，开销通常数微秒至数十微秒。TZ-LLM中LLM TA的代码和明文参数均在Secure World中受TrustZone硬件保护。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

