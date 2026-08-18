## 影子逻辑（Shadow Logic）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
影子逻辑是验证辅助的组合逻辑/信号：在 RTL 中推断或复制难以直接表达的关系（如程序序、乱序执行关系），供断言引用；它(1) 是验证中常规做法（Burch-Dill 流水线控制验证、ISA-Formal 等，QED 引 [15][28][64][74]），(2) 不影响功能，(3) 综合时被 elide（不进入最终电路、不影响性能）；因它比 RTL 简单得多，通常用简单 sanity-check 断言即可验证其正确性（也可加更复杂断言）。QED 中的实例：load_matrix[i][j]（返回 ldq[i]<p ldq[j]）、ooo_load_matrix[i][j]（ldq[j] 是否乱序先于 ldq[i] 执行）、ld_st_mtx[i][j]（ldq[i]<p stq[j]）——都是"从既有信号推断矩阵关系"的组合逻辑。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
RTL 里原本没有"程序序矩阵"这类信号（LSQ 的 load/store 条目间关系隐含在分配/提交逻辑中），为写 SVA 断言而添加。QED 的两处用途：(1) 谓词前置条件——cover 语句用 load_matrix[i][j] & ldq[j].succeeded & !ldq[i].succeeded 证明乱序场景可达；(2) fast-forwarding 的 shadow 逻辑——验证某直接序对时把其他 load 条目直接推进到 observed（见 fast-forwarding 条目）。若 shadow logic 与 RTL 不一致（罕见），JasperGold 会判定其前置条件不可达，从而暴露 shadow logic 或 RTL 的 bug——QED 的 ld-ld 性能 bug 正是"我们的 shadow logic 与 BOOMv3 的 observed 字段语义有出入"被发现的：BOOMv3 在 younger load 未 completed 或值来自 store-forwarding 时也置 observed，导致非必要的 squash。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 RTL 模块上加辅助组合逻辑/寄存器（如每对 load 条目一个比较器矩阵），随设计读入 FPV 工具，综合流程 elide；对其正确性加 sanity 断言（通常够，因为其逻辑远简单于 RTL）。类似方法：ASPLOS'25《RTL Verification for Secure Speculation Using Contract Shadow Logic》（[74]，用 contract shadow logic 验证安全投机）。使用要点：shadow logic 是验证观测面，不应被误认为设计逻辑；其与真实 RTL 的语义差恰恰是发现隐蔽实现 bug 的窗口。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
