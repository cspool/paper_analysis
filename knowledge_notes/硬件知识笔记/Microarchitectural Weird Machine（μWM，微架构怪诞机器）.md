## Microarchitectural Weird Machine（μWM，微架构怪诞机器）

术语解释
用微架构状态作寄存器、微架构行为作逻辑门构造的可编程计算装置：把微架构侧信道时序当作布尔逻辑运算来组合成任意电路（如 SHA-1、AES），在架构层面几乎不留痕迹。SSBench 的 MDP-Gates 是基于 Intel MDP 状态传播的新 μWM 实现，性能较 cache 版 μWM 高两个数量级。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
经典 cache μWM（[13][26][65][66]）：编码 bit 1=缓存命中、bit 0=缓存未命中；分支被训练误预测后推测执行 line 2，输入是否命中决定 out 地址解析快慢，从而决定 out 是否被推测性填进 cache——据此实现 AND/NOT/NAND 门（NAND 图灵完备，可组合任意程序）。其代价：依赖缓存驻留，受预取器、替换策略、缓存一致性干扰，且需瞬态执行（推测窗口有限）。SSBench 的 MDP-Gates 改用 Intel MDP 表项状态编码 bit（依赖=1、独立=0）：NOT 门用延迟 store-load 对，line 3 的 load 选输入项 E1，若 E1=1 则 load 被阻塞、阻止 line 7 的 load 提前解析从而不更新输出项 E2；若 E1=0 则 line 7 load 提前激活并置 E2=1——实现 E2=¬E1。NOR/NAND 门类似（用两个输入项决定输出项是否被激活）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
MDP-Gates 的硬件依据（SSBench 对 Intel 的发现）：① MDP 状态可传播（一个 MDP 引起的 load 阻塞延迟另一个 store-load 对的 store 地址生成，从而更新另一表项）；② 近期 Intel 提供 512 项 direct-mapped 表，无 hash 碰撞时各 load 的预测结果可持久保留；③ Intel 上 MDP 更新仅在 load 地址先于 store 解析时激活。据此构造：NOT 门（1 条延迟 store-load + 2 个表项）、NOR 门（2 输入表项 + 1 输出表项）、NAND 门（2 输入 load 碰撞同一输出表项）。实测（i9-13900）：MDP-Gates 各门准确率 >99%、每 10^7 门 <3.22s；4-bit 加法器/ALU 无纠错 >85% 准确率、Best-of-5 后 >97%（ALU 99.38%）；相对 cache-gates 快 3–15×（单门）与 >100×（电路级）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：需先经 SSBench 表征目标 CPU 的 MDP 参数（表大小、direct-mapped 组织、更新条件、状态传播能力）→ 构造门模板（汇编指令序列：乘法链延迟 store 地址 + 多 load 按依赖关系布局表项）→ 组合成电路（4-bit adder/ALU），可选 Best-of-5 多数投票纠错。威胁模型：攻击者用户态运行、探测 MDP 状态、把恶意计算电路（如 ALU）藏在微架构状态中逃避软件层检测。MDP-Gates 相对 cache-gates 的优势：状态传播不依赖瞬态执行（免预测器训练/回滚开销）、不依赖缓存驻留（对预取器/替换策略/一致性不敏感），因此更快更稳。

涉及论文标题：
- SSBench: Automated Characterization of Memory Dependence Predictors on Modern CPUs
