## Nonvolatile Memory（NVM，非易失内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NVM 是断电后仍保留数据的存储器（如闪存、FRAM、MRAM、ReRAM 等）。EHS 以 NVM 作主存（MSP430FR5994 即内嵌 FRAM），程序数据直接存于 NVM；因无缓存，load/store 直访 NVM，访存成为 EHS 最耗能的操作。NVM 的耐久性（durability）带来安全与一致性双重问题：①断电后明文残留，攻击者可在设备被丢弃/被盗后探读 NVM 提取敏感数据（本论文威胁模型）；②加密的粗粒度（16B）与 EHS 2B 原子写冲突，破坏 crash consistency。
- MANATEE 把 NVM 的角色从"主存"降为"二级存储"：NVM 只保存加密态页面（页粒度 64B、AES-XTS），SPM 中的解密副本断电即失。这样 NVM 只承担持久化职能，其全容量都可被寻址（不受 SPM 容量限制），解决了"主存容量被限制在远小于 NVM 的 SPM"的问题。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：编译器从 linker .map 获取 data section 在 NVM 的物理地址并划分为 64B 逻辑页 → 每条 load/store 携带 (NVM 页号, 颜色) hint → 页未驻留 SPM 时，page manager 从 NVM 读加密页、AES-XTS 解密载入 SPM frame → 页被驱逐或断电时，4×16B 密文块在 SPM 内凑齐 64B 后原子写回 NVM。工作负载变化实验用 STM32（Cortex-M33）+ 4MB 片外 MRAM 评估 512KB~4MB 数据集。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：EHS 中 NVM 主存直访（无 MMU），MANATEE 以页粒度加密持久化（AES-XTS，无完整性树）替代字级全加密；对比方案（Mapi-Pro/NVSRAM）在断电时整 SPM 加密写 NVM，能耗大、无法缩放到大负载。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
