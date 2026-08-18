## Load-Store Queue（LSQ，加载存储队列）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LSQ 是乱序发射（out-of-order-issue）处理器中按程序序保存未提交访存指令的硬件结构，由 load queue（LQ，load 队列）与 store queue（SQ，store 队列/store buffer）组成，是"一个核内所有访存指令重排发生的地方"（QED 的定位）。运转原理（QED II-B）：load 在 load queue 中可乱序发射到 cache、或从更老的同地址 store 经 store-to-load forwarding 转发值；为支持精确中断，store 必须到达 ROB 头提交后才写 cache，miss 时从 store queue 移入 store buffer 直到完成，但 store 地址一知即可预取 coherence 权限；较弱 MCM 下 store miss 可重叠、乱序完成。QED 把 MCM 验证聚焦在 LSQ：假设流水线前端（寄存器/控制流依赖）与缓存一致性（写串行化、多副本写原子性）已正确，则"其余所有访存排序"都由 LSQ 负责，验证 LSQ 即验证 MCM 的排序部分。Web 佐证：BOOMv3 的 LSQ 位于 riscv-boom 仓库 src/main/scala/loadstore/（lsu.scala、load-store-unit.scala）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
QED 把 load 条目建模为 FSM（图 12）：dispatch 分配条目 → 等地址（pending_addr）→ 地址就绪执行（addr_ready）→ 完成（succeeded）→ 到 ROB 头提交（committed）；另加 observed 态（匹配到 invalidation 的 load）。N 个 load 的可达状态约 7^N——这是 RTL 验证状态爆炸的来源，QED 用 fast-forwarding 缩减（见该条目）。验证中用的关键 RTL 信号：ldq[i].succeeded（第 i 个 load 条目完成）、ldq[i].observed（已匹配 incoming invalidation）、ldq[i].addr、commit_load_idx（正在提交的 load 索引）、stq[j].in_flight（store 写请求已发往内存、未收到响应/nack）、stq[j].committed（store 已提交）；辅助 shadow logic：load_matrix[i][j]（ldq[i]<p ldq[j]）、ooo_load_matrix[i][j]（ldq[j] 乱序先于 ldq[i] 执行）、ld_st_mtx[i][j]（ldq[i]<p stq[j]）。QED 在 BOOMv3 LSQ（128 loads/64 stores）上发现三类 bug：ld-amo 正确性 bug（amo 在 store queue pre-commit 头部提前写、未等 ROB commit）、st-st 重复写 bug（nacked store 重试后重复重发已完成的 younger store）、ld-ld 性能 bug（observed 字段在 younger load 未 succeeded 或 store-forwarded 时也被置位导致多余 squash）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BOOMv3 用 Chisel 实现（RV64GC，BSD-3-Clause，https://github.com/riscv-boom/riscv-boom，经 Chipyard https://github.com/ucb-bar/chipyard 实例化），LSQ 规模可参数化（QED 验证 128ld/64st，其他设计如 Dorado 模拟 200 项、RUNLTS 配置 1024 项）。使用/验证：QED 流程——RVWMO 生成探索树 → 决策树谓词 → 翻译 SVA（用上述 RTL 信号）→ JasperGold 无界证明；对 LSQ 尺寸 6x32x16 到更大配置扫 full proof time/depth/memory（尺寸翻倍时间约×10、深度近似线性、内存近似二次）。测试侧对比：litmus 测试只覆盖特定交错，bounded 穷举测试实践上限 7 条指令，均无法覆盖现代 LSQ 规模的重排空间。

sCROOGe 合并视角（GPU OoO 中的内存重排评估）：sCROOGe（ISCA'26，RISC-V Vortex GPGPU）把 LSQ 用于 GPU backend OoO 的内存指令重排——访存指令驻留 LSQ（而非 CU），与 CU 中的算术指令平行重排。RTL-aware 实现探索了逻辑复杂度与延迟的权衡：naive 方案需 LSQ_size²×T² 个 4-byte 地址比较器/SM（每 warp 地址全对全比较，冲突概率随 warp 大小平方增长）；sCROOGe 的轻量方案把目标地址存入中间寄存器、与其他 LSQ entry 逐项比较后更新依赖位图，引入单 cycle 派遣惩罚但把比较器降到 LSQ_size×T²。评估结论：per-warp DAG（真实寄存器依赖）分析显示内存重排加速上限平均 <1.1%（{4,16} 配置最高 1.1%），而 4 LSQ entry + 4 CU 配置下该机制面积/功耗开销达 8.7%/9.3%——收益远低于成本，故 LSQ 内存重排被排除在 sCROOGe backend 实现之外。该案例说明 GPU 上 LSQ 重排的适用性受"访存冲突检查的组合复杂度"与"收益天花板"双重限制，与 QED 验证的 CPU LSQ（重排空间大、正确性关键）形成对比：CPU 侧 LSQ 重排是性能引擎且需严格验证，GPU 侧则因冲突检查成本高而收益有限。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
