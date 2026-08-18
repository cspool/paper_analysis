## CoRR（Coherence of Read-Read，读读相干性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CoRR（读读相干性，又称逐地址顺序一致性 SCPL，Sequential Consistency Per Location）是多核内存一致性中的基本性质：同一 hart 对同一地址的两个 load，若中间无 store，则后一个 load 不得返回比前一个 load 更旧的值——即"内存不能对同一 hart 的时间倒退"。在 RVWMO 中它是同地址 load-load PPO 规则（手册规则 2）：若 a、b 都是 load，共享字节 x，程序序中 a、b 之间无对 x 的 store，且 a、b 返回的 x 值由不同写产生，则 a 必须在 GMO 中先于 b。RISC-V 把 SCPL 作为可由 ppo 规则 + Load Value Axiom 推出的定理（而非单独公理），并弱到足以允许真实微架构的 RSW / fri-rfi 模式（store-buffer 转发）。Web 来源：RISC-V 手册 mm-eplan（https://github.com/riscv/riscv-isa-manual/blob/20250508/src/mm-eplan.adoc ）与中文形式化证明（https://www.jos.org.cn/html/2025/9/7292.htm ）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件实现中 CoRR 由缓存一致性协议与 LSU 的按地址排序保证：同一 hart 的三个并发 load 到达 LSU 时，若 cache line 因其他 hart 的 store 被 probe 失效，已完成/已转发的 load 必须被无效化或重放。HARTBREAKER 发现的 B1 与 N1 都是 CoRR 违例：B1（图 16）——hart 1 连续三个 load 访问 x，load#1 的地址计算被多周期指令延迟，load#2 先成功；此时 hart 0 的 store 触发 coherency probe 使 cache line 失效，但 load#2 无重放路径、继续提交陈旧值，而 load#1 随后读到新值 → 年轻 load 返回旧值、年老 load 返回新值，违反 CoRR。N1（图 17）——同 hart 的 store 与后续异宽 load（sw + lw/lbu/lwu）因尺寸不匹配无法 store-to-load forwarding，store queue 条目释放时序使 load#2 标记成功且失去重放路径，跨核 probe 后 load#2 提交陈旧值。两例都需复杂结构/计算依赖与特定时序（数小时 core-time 才触发），是 litmus 测试无法构造的场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：一致性协议（MOESI 等）提供 cache line 粒度序；LSU 对已完成的 load 提供 replay 机制以应对失效（BOOM/NaxRiscv 缺失该机制正是 bug 根因）。验证侧：Table V 的 litmus 模式（P1 三连读 + P0 并发写）是最小 CoRR 违例检测模板，HARTBREAKER 把该模板融入随机程序并用 MCM solver 判"结果 (x1=0,x2=1,x3=0) 是否被 RVWMO 允许"。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
