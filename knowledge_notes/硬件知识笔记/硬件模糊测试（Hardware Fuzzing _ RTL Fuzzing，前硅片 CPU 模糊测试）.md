## 硬件模糊测试（Hardware Fuzzing / RTL Fuzzing，前硅片 CPU 模糊测试）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
硬件模糊测试是在芯片流片前（pre-silicon）用随机生成的测试程序在 RTL 仿真上动态验证 CPU 正确性的方法：fuzzer 生成指令流（通常带覆盖率/差分引导），在 DUT 的 RTL 模型（Verilator/VCS 仿真）上执行，用黄金模型（Spike ISS、Sail 或 reference hart）或自定义 oracle 校验结果，发现偏离正确行为的 bug。代表性工具：DifuzzRTL（差分 + 覆盖率引导）、Cascade（复杂程序生成 + Spike）、ProcessorFuzz（CSR 引导）、TestRIG、INSTILLER、RISCVuzz、TheHuzz、MorFuzz、RISCV-DV（随机指令生成器，非差分验证）等。优势是指令多样性高、能触达任意微架构状态，弥补 litmus/形式化方法的覆盖盲区。Web 来源：DifuzzRTL（S&P 2021）、Coverage-Guided Pre-Silicon Fuzzing 综述（https://ar5iv.labs.arxiv.org/html/2511.08443 ）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
既有 fuzzer 的共性局限（HARTBREAKER 的动机）：都面向单 hart 确定性执行——靠"黄金模型差分"验证，而多 hart 执行天然非确定（共享内存交错、IPI 到达时刻），黄金模型无法预测合法交错结果，差分即失效（论文 III-A 的表 III 汇总各 fuzzer 验证策略，均不支持非确定性验证）。HARTBREAKER 的运转流程（图 2/图 13）：生成（随机多 hart 汇编 + determinism anchors）→ Spike ISS（取确定 store 值）→ Verilator RTL 仿真（收集 load 返回值与 commit log）→ litmus 翻译 → Dartagnan MCM 验证；控制流 bug 表现为程序级确定性失效（超时）。与 RISCV-DV 的覆盖率对比（图 14，三 hart BOOM v3 上 15,872 个 mux select coverage points，各 858 core-hours）：HARTBREAKER 达相似覆盖率，且额外提供 IPI 支持、内存模型检查、确定性中断注入，关验证时显著更快。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：以 HARTBREAKER 为例——Docker 化（build_docker.sh base/run/covcollect/naxriscv），simulators/ 放各设计预编译二进制，环境变量 HARTATTACK_CORES/NUM_PROGRAMS/NUM_ELFS 调成本，artifact_reproduction/figure_8~14.sh 复现各图；评估机 2× AMD EPYC 7H12（256 逻辑核/1TB RAM），DUT 为 Rocket/BOOM/Toooba/NaxRiscv/XiangShan（Verilator 仿真，NaxRiscv 经 Scala/JNI 绑定 Verilator）；发现 5 个此前未知并发 bug（N1/N2/T1/B1/X1），N1/N2/X1 已获开发者确认修复。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
