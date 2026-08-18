## 寄存器文件 Banking 与 Swizzled Bank 映射（冲突无关访问与线程展开）

术语解释
多 bank 寄存器文件：寄存器按 bank 分布，不同 bank 可同周期并行访问，避免 bank conflict。DICE 基线用"不同索引→不同 bank、同索引跨线程共享 bank"实现 II=1 的冲突无关访问；为支持编译期线程展开（thread unrolling），引入 swizzled bank 映射 bank=(r+T) mod N_r，使相邻线程错开 bank。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：II=1 要求每周期读出一个线程的全部输入寄存器 → 把不同索引的寄存器映射到不同物理 bank，同周期对同一线程的访问即天然无冲突；同索引寄存器跨线程共享同一 bank，避免为冗余副本浪费 bank。线程展开优化要求每周期同时派发多个线程（T, T+K, T+2K, …），若这些线程的同索引寄存器都在同一 bank 即冲突 → 硬件侧将线程 T 的寄存器 r 映射到 bank $((r+T) \bmod N_r)$（相邻线程整体错位一个 bank），配合固定间隔 K 的 co-dispatch 保证无冲突。DICE 参数：$N_r=32$ 逻辑寄存器/32 bank、$N_{Tmax}=4$、K=8（4× unroll）/K=16（2× unroll，3× 不支持）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
例子：4× unroll 时同周期派发线程 T、T+8、T+16、T+24；线程 T 的输入寄存器集 $\{R_{a_1},...,R_{a_{p-1}}\}$ 各落在 bank $(a_i+T) \bmod 32$，四组线程的同一逻辑寄存器分属四个不同 bank → 同周期无冲突读出。软件协同：DICE 编译器感知 swizzling 策略，按输入寄存器集合静态计算每 p-graph 的最大安全 unroll factor，bank 冲突限制时调整寄存器分配换取更高 unroll。效果：小 p-graph 利用率提升，BPNN-1/GE-1/HS 上 unroll 单独 1.24–1.35×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RF 组织为 32 个 bank（DICE 每 CP 32 banks/256KB per cluster），访问译码加入线程 ID 偏移；Dispatcher 支持多 lane 但只允许固定间隔线程同拍派发。来源：swizzled bank mapping 出自 Swizzle Inventor（Phothilimthana et al., ASPLOS'19，论文引用 [41]）。使用场景：多线程同拍访问 RF 的空间流水后端（CGRA、多路 SIMD）。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
