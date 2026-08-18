## Alloy（轻量形式化建模与模型检查）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Alloy（Jackson, ACM TOSEM 2002）是一种轻量级形式化建模语言与模型检查工具：用关系逻辑描述系统结构（signature、relation、fact、predicate），在有限 bound 内自动搜索满足/违反约束的实例，用于发现反例与验证性质。täkōFormal 把 ISA 级 MCM 的全部公理（图 6）与 litmus tests 编码为 Alloy 模型（.als 文件），run_alloy_tests.sh 逐个确认论文声称的 allowed/forbidden/racy 结果；RISC-V 官方也把 RVWMO 形式化为 Alloy 规范（mm-formal）。Web：Alloy 官方 https://alloytools.org/ 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子：为每个 litmus test（test_paper_ex、test_mp、test_mp_rmw、test_mp_rmwcb、test_icb_sb、test_wbrace、test_wbflush、test_phir、test_phinr、test_hatsr、test_hatsnr）建立 .als——声明事件类型（R/W/Wcb/Ms/Me/Es/Ee/Fl）、关系（cbo/viscb/rf/mo/hb）、把图 6 每个公理写成 fact/predicate，程序指令作为事实加入，然后用 Alloy 在给定 bound 内搜索满足全部公理的执行（或断言某结果不可达）；hatsNR 用大 bound 搜索 race 无果，为 race-free 提供信心。Alloy 是"有界模型检查"——只保证 bound 内无反例。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：装 Alloy 6.2.0（需 Java JVM 17+，测试用 JVM 24.0.1），运行 run_alloy_tests.sh 遍历 .als 文件确认结果并打印到 console。与 Dafny 互补：Alloy 验证"MCM 公理与论文 litmus 声称一致"（程序员视角），Dafny 验证"硬件实现模型被 MCM 允许"（验证者视角）。RVWMO 的 Alloy 形式化（https://docs.riscv.org/reference/isa/v20240411/unpriv/mm-formal.html）是同类用法先例。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
