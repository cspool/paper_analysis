## DJOLT（Distant Jolt Prefetcher，远距 Jolt 预取器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DJOLT（Kumar et al., NCSU，第一届指令预取锦标赛 IPC1 冠军之一）是状态型 L1I 预取器：用"指令指针（IP）历史 + 距离"的时间相关模型预测未来取指地址——把当前取指地址与历史表中的先前序列匹配，"jolt"即重放时与当前地址的距离差，据此把历史中记录的后续地址平移后作为预取目标。IPC1 前三名 EIP/DJOLT/FNLMMA 收益几乎相同。Bumper 评估（Fig.17）：DJOLT 单独较 FDIP-only baseline 有增益（额外预取覆盖），但 Bumper 单独更强（移动负载中减少 L2C 污染比增加预取覆盖更值钱）；两者互补：Bumper+DJOLT 平均 +9.7%（vs FDIP-only baseline）、较 Bumper 单独 +3.2%——DJOLT 增加覆盖的同时加剧 L2C 污染，Bumper 负责快速淘汰 useless 行并延长 useful 行生命周期。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：取指产生当前 IP → 查历史表找匹配序列 → 命中则按记录的距离（jolt）平移生成未来取指地址 → 向 L1I 发预取；与 FDIP 并存时两个预取源都可能把 useless 行带入 L2C。Bumper 叠加后：预取来的行同样以 RRPV=3 插入，只有行内指令提交才被提升，错误路径预取被快速淘汰。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：IP 历史表 + 匹配/重放逻辑（时间相关预取器范式）；IPC1（research.ece.ncsu.edu/ipc）框架下与 EIP、FNLMMA 等比较。Bumper 与之正交：Bumper 是缓存管理机制，可与任何 L1I 预取器组合，且论文强调其收益可随 L1I 预取器进步而放大。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
