## TZPC (TrustZone Protection Controller)

术语是什么？通过联网搜索让回答具体和精准。

TZPC（TrustZone Protection Controller）是ARM TrustZone安全体系中的外设安全配置硬件IP。它控制SoC上各外设（NPU、DMA引擎、SPI控制器、UART等）的安全访问权限——为每个外设提供安全配置位，决定该外设是Secure-only（仅Secure World可访问其MMIO寄存器）还是Non-Secure-accessible。与TZASC保护DRAM数据不同，TZPC保护外设控制路径（MMIO寄存器空间）。两者配合实现完整TrustZone硬件隔离：TZASC保护数据面，TZPC保护控制面。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TZPC在TZ-LLM co-driver NPU场景下的工作流程：

1. TEE data plane driver写TZPC将NPU设为Secure-only→阻止REE访问NPU MMIO寄存器→防止REE窥探/篡改secure job command/register序列
2. 配置GIC将NPU中断路由到TEE
3. 等待当前non-secure NPU job完成
4. 配置TZASC允许NPU DMA访问secure memory
5. 验证shadow job sequence number→写NPU MMIO launch寄存器启动secure job
6. Job完成后→恢复NPU为Non-Secure（TZPC+TZASC+GIC全部恢复）

关键硬件特性：(1)配置粒度——以单个外设或外设组为单位，各外设独立保护；(2)硬件执行——TZPC信号连接系统总线访问控制逻辑，Normal World访问Secure-only外设时总线返回错误；(3)与TZASC配合——TZPC保护外设MMIO访问，TZASC保护NPU作为bus master访问DRAM时的内存访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TZPC与TZASC一样在所有支持TrustZone的ARM SoC中集成。仅Secure World可配置TZPC寄存器。TZ-LLM中TEE OS扩展~50 LoC支持TZPC/TZASC动态配置。相比将完整NPU driver放入TEE（需引入~60K LoC进入TCB），co-driver通过TZPC/TZASC/GIC的轻量配置（~112 LoC）实现安全NPU切换，避免driver detach-attach的~32ms开销。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

