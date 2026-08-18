## Spike（RISC-V ISA Simulator，指令集模拟器 / 黄金模型 ISS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Spike（riscv-isa-sim）是 RISC-V 官方参考 ISA 模拟器：功能级、逐指令精确（非周期精确）的 C++ 模型，支持 RV32/RV64、M/S/U 特权级、hypervisor 扩展、向量与多数已批准扩展，严格遵循 RISC-V ISA 规范（比 QEMU 在异常处理上更严格、更贴近规范），广泛用于 ISA 合规测试、编译器验证与软硬件协同验证。在硬件 fuzzing 生态中它是事实标准的"黄金参考模型"（GRM）：DifuzzRTL、GoldenFuzz、TestRIG、ProcessorFuzz 等都把 Spike 作为差分测试的 oracle。Web 来源：https://github.com/riscv-software-src/riscv-isa-sim 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 HARTBREAKER 流水线中 Spike 承担"确定性参照物"角色（图 13 第二阶段）：(1) 生成器产出的多 hart 测试程序先在 Spike 上执行，获得生成时未知的确定值——尤其是各 store 最终提交到内存的值（df-anchor 保证 store 值确定）——为后续 litmus 翻译提供 store 值来源；(2) 它同时也是控制流确定性的基线：程序在正确实现上必须以 Spike 可见的方式完成所有指令。之后同一二进制才在 Verilator 仿真的 DUT 上运行；两阶段分别提供"应该是什么"与"实际是什么"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：`spike -m256 pk test.elf` 或直接加载裸机 ELF 运行并 dump 架构状态（寄存器/内存签名）；fuzzer 通常对每测试运行 Spike 生成参考签名、与 DUT RTL 仿真结果逐指令或逐状态比对。HARTBREAKER 中 RISCV-DV 对比实验同样以 Spike 为参考模型（Table III 汇总各 fuzzer 的验证策略：DifuzzRTL/Cascade/ProcessorFuzz/INSTILLER/RISCV-DV 用 Spike，TestRIG 用 Spike+Sail，Trippel 等用自定义 golden）。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
