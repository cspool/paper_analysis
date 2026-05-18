## TZASC (TrustZone Address Space Controller)

术语是什么？通过联网搜索让回答具体和精准。

TZASC（TrustZone Address Space Controller）是ARM TrustZone安全体系中的DRAM访问控制硬件IP。它位于DRAM内存控制器前端，根据物理地址范围和总线事务的安全属性（通过AXI AxPROT信号携带的NS位）决定是否允许该内存访问。TZASC将DRAM地址空间划分为可配置的安全region，每个region可设置为Secure-only（仅Secure World可访问）或Non-Secure-accessible。这是TrustZone保护外部DRAM中敏感数据（如TZ-LLM中解密后的模型参数、activation、KV cache）的核心硬件——CPU内部状态和片上SRAM/TCM可在不经过TZASC的情况下受NS位保护，但DRAM必须依赖TZASC。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

TZASC在TZ-LLM secure memory动态扩展中的工作流程：

1. LLM TA决定为新computation operator扩展参数内存→调用extend_allocated(op_id, size)
2. TEE OS向REE Linux请求CMA分配size字节连续物理内存→REE返回[pa_start, pa_end)
3. TEE OS配置TZASC：新增secure region [pa_start, pa_end)→Secure-only。TZASC硬件将配置写入内部region descriptor table
4. 此后任何来自Normal World对该地址范围的DRAM访问被TZASC硬件直接拒绝（总线返回错误）
5. TEE OS将物理内存映射入TA虚拟地址空间→TA解密参数后安全使用（明文受TZASC硬件保护）

关键硬件特性：(1) Region粒度——通常支持8-16个可配置region，每个region从64KB到数GB，但要求连续物理地址。这就是TZ-LLM必须使用CMA而非buddy allocator的原因；(2) 硬件查表延迟极小（1-2总线周期），不显著增加DRAM访问延迟；(3) 仅保护DRAM，不保护CPU cache中数据（需配合cache flush/invalidate）。

TZ-LLM的特殊用法：(1) 动态extend/shrink TZASC region而非静态预留大块secure memory；(2) 利用LLM参数first-in-last-out模式保证TZASC所需连续物理内存；(3) pipeline-aware extend/shrink——在computation时后台扩展后续operator的TZASC region。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常见实现：ARM CoreLink TZC-400（最多9个region）。Rockchip RK3588集成RK定制TZASC，通过TEE OS安全驱动配置。TZ-LLM中TEE OS扩展~50 LoC支持动态TZASC配置。仅Secure World可访问TZASC配置寄存器（受TZPC保护）。局限性：(1)要求连续物理地址→受碎片影响；(2)Region数量有限→大规模应用需区域合并；(3)不防物理DRAM攻击（probing等可绕过）。

涉及论文标题：
- TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

