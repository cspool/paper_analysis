## Litmus test（litmus 测试，逐字测试）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Litmus test 是描述内存一致性模型行为的最小并发测试程序：多个线程/hart 各执行一小段汇编（load/store/AMO/fence），最后用 exists 断言指定一个观察到的寄存器/内存结果，由工具判定该结果在目标模型（如 RVWMO）下是否合法。经典例子（Table I store-buffer）：x、y 初始 0，P0 执行 x←1; x1←y，P1 执行 y←1; x2←x，断言 (x1=0,x2=0)——SC 禁止而 RVWMO 允许。litmus 测试工具族：diy（生成测试）、herd7（公理模型执行）、litmus（硬件上跑）、rmem（操作模型），RISC-V 官方测试集 litmus-tests-riscv 含 5000+（同尺寸非混合）测试；标准来源 https://github.com/litmus-tests/litmus-tests-riscv 、https://docs.riscv.org/reference/isa/v20240411/unpriv/mm-eplan.html 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
从硬件验证角度看，litmus 测试的优势是"直接对硬件内存子系统施加精确的排序压力"，劣势（HARTBREAKER 的动机）是：(1) 只覆盖内存一致性，无法测 IPI/中断/特权等并发通道（Table II 显示 litmus 无 exceptions/FPU/privilege switch/IPI）；(2) 指令多样性极低，无法触发特定微架构条件（store buffer 深度、LSU 重放、一致性 probe 时序）下的 bug——论文用 1,314 个 PULP/CHERI bare-metal litmus 测试实测在五个 CPU 上零 bug，而这些 bug 正是 HARTBREAKER 发现的。HARTBREAKER 的关键洞察：任何随机生成的多 hart 测试程序都可以翻译成等价 litmus 测试（图 7 依赖构造法 + 图 6 流程），从而复用 litmus 验证后端（Dartagnan 等）检查非确定性执行 trace 的合法性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：litmus 文件格式（P0/P1 指令列表 + exists 断言 + 初始化）被 herd7/Dartagnan 直接解析；硬件厂商用 diy 生成 + 硬件实测对照模型。HARTBREAKER 的翻译流程（图 6）：(1) 生成器静态提取程序中的句法依赖；(2) Spike ISS 提供确定性 store 值；(3) RTL 仿真 commit log 提供 load 返回值；(4) 为每个 store 附指令保证 litmus 中提交相同值、用临时寄存器携带依赖并 xor 清零、用 or 传递依赖到地址，最后构造 exists 断言交 solver。作用：把"验证任意长非确定性程序"降为"验证一组等价小 litmus"，绕开 testing problem 的 NP 完全性（Cantin et al.）。QED 对 litmus 方法的批判视角补充：litmus 是"测试"而非"验证"——只检查给定测试暴露的 bug，未覆盖的 bug 检测不到；穷举最小 litmus 集（[39]）跨线程 n 条指令在实践上 n>7 即爆炸，远小于现代指令窗口（数百条 in-flight）。QED 的事件序与 litmus 存在"根本反转"（fundamental reversal）：litmus 检查具体样例是否出现禁止结果，而 QED 的 <m 事件排序"蕴含存在某个合法（可能对抗性）样例"（图 5(a) 蕴含 5(b) 或 5(c)，而非反之），反例由探索树+环检测自动、穷举地发现。täkōFormal 的 MCM 使用视角补充：litmus test 不仅用于验证硬件实现，也是"程序员用 MCM 编写可编程内存层级程序"的编程工具——论文 §V 用 litmus tests（mp、mprmw、mpcb、icbsb、wbr/wbf、phiR/phiNR、hatsR/hatsnR，对应 Alloy 文件 test_paper_ex/test_mp/test_mp_rmw/test_mp_rmwcb/test_icb_sb/test_wbrace/test_wbflush/test_phir/test_phinr/test_hatsr/test_hatsnr）分析 täkō 程序，判断某结果 allowed/forbidden 或程序是否 racy（如 wbr 无 FlushRange 时 racy、wbf 加 FlushRange 后 no race 且 r1=0 forbidden；hatsR racy、hatsNR 在 OnEvict 只记录合法边后 race-free）。这些测试被编码进 Alloy 模型（run_alloy_tests.sh 逐个确认论文声称结果），与"把 trace 翻译成 litmus 交 solver"（HartBreaker）或"用 litmus 直接测硬件"互补——täkō 目前只有 closed-source 模拟器，litmus+Alloy 是唯一可执行的 MCM 验证途径。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
