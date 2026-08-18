## BRIM（Bistable Resistively-Coupled Ising Machine，双稳态阻性耦合 Ising 机）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BRIM 是 Rochester 大学 Afoakwa 等提出的全 CMOS 兼容 Ising 机（HPCA 2021）：自旋用双稳态节点表示，耦合强度 J_ij 用可编程电阻电导实现，节点间的物理电流/电压交互驱动网络自发弛豫到低能基态，不需显式算法模拟。它针对量子退火（D-Wave：庞大、需 ~15 mK 低温、~25 kW）与相干 Ising 机（光纤公里级、需大量外部数字计算）等既有方案的缺点，支持 all-to-all 耦合（免去稀疏机器的小嵌入 minor-embedding 开销），单次 max-cut 实例能耗低至 ~75.3 fJ。控制上采用 selector 逻辑：pull-down 网络按列激活 + 共享 programming bus 广播参数——这是 DS-ISA 论文所指"实现特有、未抽象"的底层控制方式。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
BRIM 的硬件运转流程：求解组合优化问题 → 问题映射为 Ising 哈密顿量 H_Ising = −Σ J_ij σ_iσ_j − Σ h_i σ_i → selector 按列激活目标耦合列 → (J_ij, h_i) 经共享编程总线写入电导 → 双稳态节点电压经阻性耦合网络自发演化（电流经电导对节点充放电）→ 网络弛豫至低能态 → 节点稳态即自旋解。DSU 族在此基础上的扩展：把线性自相互作用换成二次项得到实值连续哈密顿量 H_DS，节点从二值自旋变为实值电压变量（见"Dynamical System Unit"条目）。对比数据（DS-ISA 论文 Table II，45nm 工艺）：BRIM 2000 节点、无耦合演化、250mW、5mm²。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CMOS 双稳态节点电路 + 可编程电阻耦合 + 列选择器/编程总线；DS-ISA 论文把 selector 机制作为 DSU 的默认基础组件，在其上抽象出标准 ISA。使用方式：求解 Max-Cut/SAT 等组合优化；作为 DSU 原型供后续工作扩展实值域（DS-GL/DS-TPU/DS-TIDE/DS-LLM）。局限性（DS-ISA 论文归纳的先前控制方式共性）：按列统一配置耦合、无子集掩码控制；全局耦合演化、不可分区多任务；节点一次性初始化、中间态不可复用；无重叠执行控制。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
