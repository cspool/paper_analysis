## 2Q 门规范表示（Canonical Form / KAK 分解 / Weyl chamber）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 任意两比特酉 U ∈ SU(4)（忽略全局相位）都可唯一表示为"规范形式夹心单比特门"：Can(a,b,c) := e^{-iπ/2(a·XX + b·YY + c·ZZ)}，其中规范系数 (a,b,c) 满足 1/2 ≥ a ≥ b ≥ |c|，U 与 Can(a,b,c) 局部等价（locally equivalent，记作 ~）。三个系数张成的区域是四面体 Weyl chamber，它是所有 2Q 门局部等价类的几何表示。得到规范形式的标准算法是 KAK 分解（Khaneja–Glaser 分解，源自 Cartan 分解 su(4) = su(2)⊗su(2) ⊕ Cartan subalgebra，Qiskit 中 `TwoQubitBasisDecomposer`/`UnitarySynthesis` 内部使用）。常见门在 Weyl chamber 中的位置：CX/CZ/CR 都等价 Can(1/2,0,0)；CX 族 XX(θ)~YY(θ)~ZZ(θ)~Can(θ/π,0,0)；param-SWAP 族 pSWAP(θ)~Can(1/2,1/2,1/2−θ/π)（θ=π/2 时即 iSWAP ~ Can(1/2,1/2,1/4)）。这一表示的核心价值：2Q 门的"本质交互强度"被压缩成 3 个实数系数，合成成本、交换性等性质都只依赖这些系数。
- 别名：canonical gate、magic basis/KAK coordinates、local invariants、Weyl chamber coordinates（Qiskit 的 `canonical` 与 `abc` 参数化）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 CANOPUS 中，编译框架把输入电路先 rebase 到 {Can, U3} 门集（每个 2Q 门经 KAK 分解转为规范系数），之后路由、成本计算、交换性判断全部在规范表示上进行：① 合成成本：给定目标 ISA 的 basis 门集，Can(a,b,c) 的合成成本由该点所在的多胞形决定（见 Monodromy 多胞形条目），无需显式做 rebase；② SWAP mirroring：SWAP·U 合并为复合酉后仍是规范门（如 SWAP·iSWAP ~ Can(1/2,0,0)，只需 1 个 CX 合成）；③ 交换性判断（Theorem 1）：Can(a,b,c) 与 Can(a',b',c')（共享一个 qubit）可交换当且仅当 b=b'=c=c'=0（纯 XX），无需追踪 control/target 位置。路由输出仍是 {Can, U3} DAG，最终由后端 synthesizer 做具体 ISA rebase——这正是"LLVM 式"的中间表示抽象。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：KAK 分解在 Qiskit（`qiskit.synthesis.TwoQubitBasisDecomposer`、`qiskit.quantum_info.TwoQubitBasisGate`）、TKET、Cirq 中均有实现；论文用 pytket 获得 Can/U3 rebase 后的电路、用 monodromy 库做多胞形计算。CANOPUS 的 `canopus/basics.py` 定义 CanonicalGate/BGate/SQiSWGate 等门对象，`synthesis.py` 提供 rebase_to_canonical/sqisw/zzphase/custom 等。评估时以 TKET 优化后的 {Can,U3} 电路作为各编译器统一输入（Table III）。
- 实际例子：Table III 的 12 个 benchmark（bigadder、qft、qpeexact 等）都先 TKET 逻辑级优化并 rebase 为 {Can,U3}；qft 18 qubit：#Can 153、Depth2Q 33、Ccount 306（CX ISA 下成本）。规范表示让"任何 ISA 的合成成本可预计算"成为可能，是跨 ISA 统一编译器优化的数学基础。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
