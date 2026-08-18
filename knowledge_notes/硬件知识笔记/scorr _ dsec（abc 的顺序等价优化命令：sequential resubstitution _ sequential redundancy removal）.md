## scorr / dsec（abc 的顺序等价优化命令：sequential resubstitution / sequential redundancy removal）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
scorr 与 dsec 是 Berkeley abc（https://github.com/berkeley-abc/abc，Brayton & Mishchenko CAV 2010）中两个互补的顺序（sequential，带状态/时序）逻辑优化与验证命令，均基于 k-induction/BMC：
- scorr = sequential resubstitution（顺序重代入）：在可达状态空间与外部约束（assume/EDC）下检测 netlist 中功能等价的节点对，用一个替换另一个，消除等价冗余结构（与组合综合中的 resubstitution 对应，但作用域扩展到时序可达状态）。
- dsec = sequential redundancy removal / sequential equivalence checking（顺序冗余移除/顺序等价性验证）：对候选节点注入 stuck-at-0/1 故障，用 k-induction 证明 faulted 电路与原电路顺序等价——若等价，则该节点在所有可达状态下"卡在常量"、是顺序冗余的，可删除。
二者由 Mishchenko et al.（ICCAD 2008 "Scalable and scalably-verifiable sequential synthesis"）提出，实现"可扩展且可验证"的顺序综合；abc 中同类命令还有 lcorr（组合等价合并）等。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 æSIP 的 don't care 驱动 ASIP 生成器（§IV-C3，Fig.7）中 scorr/dsec 的运转流程：EDC（SVA assume：ISA 白名单 + shamt∈{1,2} + dcache miss 5 cycle 内响应）注入后的 baseline netlist 上 → ① scorr 找在约束下功能等价的节点对（如受限 shamt 使桶形移位器多条路径等价）→ 替换合并；② dsec 逐节点注入 stuck-at-0/1 故障 → k-induction（k=2..5）验证 faulted 电路在 EDC 下与原始顺序等价 → 等价则删除该节点；③ 两者共同实现"在 EDC 下证明可安全移除的门级裁剪"，产生面积更小的 ASIP netlist。论文明确"scorr and dsec are existing commands in abc"，æSIP 的贡献是自动化约束推导（静态分析+EDA 约束生成）与 error-injection-based gate pruning 的集成，而非实现这两个引擎本身。运行时：scorr 22.5-213.8s/benchmark（中位 25.0s），为框架最耗时阶段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：abc 命令行/API（读入 netlist（read）→ 设置 assume 约束 → 执行 scorr / dsec → 写回简化 netlist）；底层 SAT solver 驱动 BMC，支持多步 k-induction（-k 参数）。使用：①综合优化——RTL/netlist 级 don't care 裁剪（ASIP 生成、datapath 精简）；②等价性验证——验证优化后设计与原设计顺序等价。文献佐证（IEEE "Scalable Sequential Logic Synthesis Using Observability Don't Care Conditions"）：dsec 2-step induction 验证 <13s/benchmark（最大 74s）；sequential redundancy removal 平均降 3.40% NAND2 门、加 resubstitution 3.65%、2-step induction 3.97%，与顺序 SAT-sweeping 正交可再降 18.3% 面积。在 æSIP 中与 PdatScorrWrapper/ 仓库模块对应（don't care 裁剪封装）。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
