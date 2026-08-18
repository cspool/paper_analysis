## CGRA Processor（CP）控制流水与 e-block（CS/FDR/DE/RE 四阶段 + 双缓冲配置存储 + BRT）

术语解释
DICE 中替代 warp 前端/后端的轻量控制结构：以 CTA 为调度粒度、p-graph 为执行粒度，四阶段流水 CS→FDR→DE→RE 组织 e-block（= 某 CTA 的活跃线程执行某 p-graph 的动态实例）的取配置、派发与退役。e-block 之于 p-graph 如同动态指令之于静态指令，p-graph 之于机器模型如同静态指令之于 ISA。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
四阶段组成（Fig.7a）：① CS（CTA Schedule）：Active CTA Table（存 kernel 元数据 + 每 CTA 一个 PDOM 栈）选择下一 CTA 并实例化 e-block；调度策略优先选 next PC 与最近 e-block 相同的 CTA 以复用 metadata/位流，否则 round-robin。② FDR（Fetch/Decode/Reconfig）：p-graph cache 取 metadata → 解码 → Branch Handler 按 BTFNT 预测更新 PDOM 栈 → barrier 检查（等 RE 确认全部前序 e-block 访存完成）→ 位流装载进非活跃 CM（CM0/CM1 双缓冲，重配置不打断执行）。③ DE（Dispatch/Execution）：Dispatcher（Active Thread Selection Logic + Scoreboard + Operand Collector）+ CGRA + LDST Unit；stall 条件 = scoreboard 检测操作数未就绪 / LDST FIFO 无 credit / 无线程可派。④ RE（Retire）：Block Retire Table（BRT）容纳多个 e-block 并跟踪未归访存，全部归账后 retire——CGRA 不为等访存停机。e-block ready 条件 = metadata 解码完成 ∧ 位流装载完成 ∧ active mask 解析完成 ∧ barrier 满足。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
时间线例子（Fig.8，2 CTA × 3 p-graph）：CS 选 CTA0 的 pg-0 → FDR 取 M（metadata）+B（bitstream）→ DE 派发 N 线程流水执行 → 全线程完成计算后 e-block 入 RE、访存归账即 retire；随后 CS 选 CTA1 的 pg-0（同 p-graph），其 FDR 因 metadata/位流命中大幅缩短（①）；若 scoreboard 检出对未决访存写回的依赖则 dispatch stall（②）。分支预测使 CS/FDR 可提前准备后继 e-block，但 e-block 须等 active mask 解析才进 DE，误预测 e-block 丢弃。效果：CTA 级（数百线程）摊销取 metadata/位流与重配置开销，NN 上控制能耗占比 18.1%→1.3%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：轻量四阶段控制流水（类 von Neumann 前端但一次"取指"服务数百线程）；每 CTA 一套发散状态（PDOM 栈）；BRT 解耦计算完成与访存归账。使用：为静态 CGRA 提供通用 SIMT 控制骨架；scale-out 时 CP 微架构不变（DICE-O48/O72），scale-up 改阵列尺寸（DICE-U 32 PEs/CP）。论文未明确说明各阶段缓冲深度等实现细节。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
