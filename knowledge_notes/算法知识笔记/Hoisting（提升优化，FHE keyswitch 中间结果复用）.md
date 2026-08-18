## Hoisting（提升优化，FHE keyswitch 中间结果复用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hoisting（提升）是 CKKS 程序级的算法优化（Bossuat et al. EUROCRYPT'21 提出，HE² 采用）：利用"模数可交换性质"（EWO、Autom 等 Commutative Operators 与 ModUp/ModDown 可交换执行顺序），把多个并行 keyswitch 中的冗余 ModUp/ModDown 提取合并到 PKB 的输入/输出端，使同一输入密文的 ModUp 只做一次、多个 IP 聚合结果只做一次 ModDown。本质是用"中间结果复用"换取"evk 复用"：虽然各 keyswitch 的 evk 不同，但 ModUp 结果可跨多个 IP 复用，聚合后的 IP 输出只需一次 ModDown。
- 关键权衡：hoisting 减少 ComOps 数量（ModUp/ModDown），但把 MemOps（PMul、CAdd 等）的计算顺序交换、模数域从 Q 升到 PQ 或 PQ·dnum，MemOps 计算量增加——总收益取决于削减的 ModUp/ModDown 是否超过 MemOps 增额。其收益上限受 keyswitch 并行度约束：并行 keyswitch 越多、PKB 入/出度越低，可合并的 ModUp/ModDown 越多。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 PKB（n 条并行 keyswitch，同一输入 ct）为例（HE² 论文 Fig. 2(c) 与式 (1)(2)）：
```
# hoisting 前：每条 keyswitch 独立做 ModUp/IP/ModDown
for i in 0..n-1:
    c_up[i] = ModUp(ct)            # n 次 ModUp（同输入冗余）
    ip[i]   = IP(c_up[i], evk_i)   # n 次 IP
    out[i]  = ModDown(ip[i])       # n 次 ModDown
# hoisting 后：ModUp 提到前端共享，ModDown 聚合到后端
c_up = ModUp(ct)                   # 1 次共享 ModUp
for i in 0..n-1:
    ip[i] = IP(c_up, evk_i)        # n 次 IP 复用 c_up
out = ModDown(Σ_i ip[i])           # 1 次共享 ModDown（线性组合后）
```
- Annotations：共享前提是各 keyswitch 的输入密文相同（同一 PKB 内）；ModDown 可合并到输出端线性组合（PMul/CAdd）之后；代价是 IP 结果在 PQ·dnum 域累加、MemOps 域变大（见"ModUp/ModDown"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：软件库层面在 CKKS 程序变换中做（Anaheim 对原始程序直接应用 hoisting、FAST 在低密文层应用）；HE² 的 HERO 框架把 hoisting 作为 DFG 优化的末端步骤——先识别/扩展/融合 PKB（提高并行度、压低出入度），再在 PKB 输入输出端应用 hoisting，使 ModUp/ModDown 削减最大化（相比直接 hoisting 再多削减 2.25× 计算与 2.42× 通信）。硬件影响：hoisting 后 IP/PMul 域变大、可整块卸到近存（IRF），但 evk 复用率下降——EVF 单体 ASIC 直接应用 hoisting 反而因 off-chip evk 访问 stall 性能下降（SHARP+hoisting 仅 39.4% 加速需 2.89× 片上内存），IRF 异构架构（中间结果在 xMU 侧复用）才是 hoisting 的受益场景。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
