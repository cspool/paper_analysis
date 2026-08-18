## PPO（Preserved Program Order，程序顺序保持规则）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PPO（Preserved Program Order）是 RVWMO 中"必须在全局内存序（GMO）中被保持的程序内排序关系"的集合。GMO 是覆盖所有内存事件的全序，程序序（po）中绝大多数边可以被硬件重排，只有 PPO 规定的边强制在 GMO 中保持原方向。RISC-V 手册给出 13 条 PPO 规则，可归为四类：(1) 句法依赖——地址依赖（addrdep）、数据依赖（datadep）、控制依赖（ctrldep，仅约束其后 store）；(2) 流水线依赖——load-to-store、load-to-load 的转发类依赖；(3) 同地址排序——后写不得越过同地址先读/先写（写不早于先访存）、读读相干 CoRR、AMO/SC 原子性；(4) 显式同步——FENCE（PR/PW/SR/SW 六种常用编码与 FENCE.TSO）、acquire/release 注解、RCsc 对、LR/SC 配对。形式化 Alloy 规范（riscv-isa-manual 的 memory-model-alloy.tex）把 ppo 定义为 po_loc 约束 + ppo_fence + acquire/release/RCsc + 三类句法依赖 + 流水线依赖的组合，并要求 ppo ⊆ gmo。Web 来源：https://docs.riscv.org/reference/isa/v20240411/unpriv/mm-formal.html 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PPO 是"硬件可以优化到多激进、软件可以依赖到多强"的边界：实现中每条 PPO 规则对应一类具体微架构机制——地址依赖由 LSU 的地址等待/重放保证、数据依赖由寄存器旁路与序内执行保证、同地址规则由 store buffer 的按地址检查与一致性探针保证、FENCE 由流水线 drain/内存屏障单元保证。HARTBREAKER 以"PPO 规则覆盖"（图 8）作为测试质量指标：统计生成的 10,000 个测试程序中每条内存指令受哪条 PPO 规则约束的概率，确保所有规则都被施加压力（避免 bug 被测试盲区掩盖）；发现的两个 bug（B1、N1）正是同地址 load-load PPO（CoRR）被违例：年轻 load 返回了比同 hart 更早 load 更旧的值，即 PPO 边未在 GMO 中保持。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：测试侧用 litmus 测试逐一针对某条 PPO 规则构造最小违例场景（如 Table V 中 P1 的 x1←x、x2←x、x3←x 三连读测 CoRR）；生成侧在程序生成器里为内存操作随机注入/构造句法依赖（图 7：xor 清零寄存器但保留依赖、or 传递依赖到地址寄存器）；验证侧把 trace 中观察到的内存序与 PPO 规则比对，Dartagnan 判定是否存在违背规则却仍成立的执行。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
