## Dafny（验证感知编程语言与机器检查证明）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dafny（Leino, LPAR 2010）是一种 verification-aware 编程语言：在命令式语言中内建 pre/postcondition、invariant、lemma 等规范注解，编译器把"程序 + 规范 + 证明"翻译成 Satisfiability Modulo Theories（SMT）查询交给 Z3 验证，实现部分自动化的机器检查证明。täkōFormal 用它实现 täkō 的分层迁移系统（operational model：Core+L1 / Engine+L1 / L2 / L3 slice / Memory / Network 各组件状态机 + 环境迁移 + 分层目录 MSI），并完成 61K LoC 证明注解、119 条归纳不变量子句的端到端 soundness 证明。Dafny 已知在不同系统上有 brittleness（超时/内部错误），artifact 在不同 OS 上可移植性受限。Web：Dafny 官方仓库 https://github.com/dafny-lang/dafny ，参考实现与文档 https://dafny.org/ 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子：täkōFormal 在 Dafny 中定义每个组件状态机的 Init 谓词与 Next 谓词（如 L2.Next 把 GetS 消息写入 Network），定义 ISM（构建部分执行图）与抽象函数 abs，然后用归纳法证明：(a) Init(s)⟹Inv(s)；(b) Inv(s)∧Next(s,s')⟹Inv(s')（归纳不变量）；(c) abs 保持迁移对应（refinement：operational 迁移要么是合法 ISM 迁移、要么是 NoOp）。每条证明义务经 Dafny→SMT→Z3 机器检查，最终输出"对所有程序、所有执行，operational model 的执行都被 ISA MCM 允许"。run_dafny_verification.sh 验证全部 Dafny 文件（图 6 每个公理到对应文件的映射在 README）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：装 Dafny 4.11.0（Windows 需 .NET）后运行仓库脚本；与 Alloy（检查公理/测试）互补——Dafny 侧证明"实现⊆MCM"，Alloy 侧证明"MCM 对 litmus tests 的判定符合论文声称"。它还可验证硬件算法正确性（如 Rowhammer 防御论文 From Lab to Fleet 用 Dafny 验证每行访问计数不变量）。täkōFormal artifact：GitHub https://github.com/GenericMonkey/takoFormal（MIT），Zenodo DOI https://doi.org/10.5281/zenodo.19444275。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
