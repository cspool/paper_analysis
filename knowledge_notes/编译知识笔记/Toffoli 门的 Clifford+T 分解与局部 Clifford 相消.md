## Toffoli 门的 Clifford+T 分解与局部 Clifford 相消

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Toffoli（CCX）门是量子算法（加法器、Shor 等算术子程序）的高频非 Clifford 门：标准 Clifford+T 分解为 7 个 T/T†、6 个 CNOT 与 2 个 Hadamard（共 8 个 Clifford + 7 个非 Clifford），即每个 Toffoli 贡献 8 个 Clifford 门。TACO 的洞察：这些 Clifford 门可基于代数结构"局部"两两相消，无需把门穿过电路（不损失并行）。流程（图 10）：①把第一个非 Clifford 门 T† 表示为 Rz(−π/4)，穿过 CNOT 再穿过 H（图 10b）；②所有非 Clifford 门换到前面后，剩余 Clifford 门按序标为 C1..C8；③C3、C4（作用于同一 target 的 CNOT）可交换对调；④对调后自逆对 (C2,C4)、(C3,C5)、(C7,C8) 相消，最后 H 对 (C1,C6) 相消（图 10d）。净效果：Toffoli 只剩 7 个非 Clifford 旋转，全部操作限制在原始三比特局部子空间内，不引入额外 commutation 步骤或深度开销。
- 论文用途（TACO）：Toffoli 分解的 Clifford 消除与单比特序列的 MA Normal Form 化简共同构成"并行保持的 Clifford reduction"，对 Toffoli-heavy 电路（adder、csla_mux、hwb、qcla_mod，来自 Op-T-mize）尤其重要——这些电路中约 70% 的 Clifford+T 门来自 Toffoli 分解。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TACO 转译器中的运转流程（单个 Toffoli，输入三比特 a,b,c）：
```
① 分解: Toffoli(a,b,c) = H(c) + T†(c) + CNOT(b,c) + T(c) + CNOT(a,c)
        + T†(c) + CNOT(b,c) + T†(a) + T(c) + CNOT(a,b) + T(b)
        + T†(a) + CNOT(a,b) + H(c) + T(a) + T(b)   # 标准 7T+8C 分解
② 非 Clifford 前置: 把 T/T†（=Rz(±π/4)）经 CNOT 与 H 换到序列开头
   （Pauli 走线: T 经 CNOT 变成带控制位的 X 旋转，经 H 变轴）
③ Clifford 分组: 剩余 Clifford 标记 C1..C8，识别可交换 CNOT 对 (C3,C4)
④ 局部相消: 对调 C3/C4 后 (C2,C4)(C3,C5)(C7,C8) CNOT/H 自逆对相消，
   H 对 (C1,C6) 相消
输出: 仅剩作用于 (a,b,c) 的 7 个非 Clifford 旋转（Rz/Rx(±π/4) 序列）
```
- 效果：Toffoli 的 Clifford 门 100% 消除（8→0），且因相消都在局部三比特子空间内，不引入跨电路 commutation、不增加深度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：作为转译器内的模式重写 pass：对每个 CCX 门展开为标准分解后，用"非 Clifford 前置 + Clifford 相消"两组规则重写（自逆对识别 = 检查连续两门是否相同且可交换）。TACO 中与单比特序列优化（MA Normal Form）并行运行，总复杂度 O(n)。使用场景：任何含大量 Toffoli/多控门的 FTQC 编译（Shor 加法器、Toffoli-heavy 基准）；hwb 电路（初始 CNOT 占比 >65%）因 TACO 保留 CNOT 而 Clifford 减少率最低（77%），但运行时仍获 3.52× 加速——说明保留 CNOT 换并行是值得的。

涉及论文标题：
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
