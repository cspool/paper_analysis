## Quantum ISA（量子指令集架构 / Basis Gate / Native Gate）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Quantum ISA（在编译语境下）指物理平台上原生实现并校准的 2Q basis 门集合（广义上也含 1Q 门），即设备实际能高质量执行的门。传统超导平台原生支持 CX 等价门（CZ、Cross-Resonance CR、Mølmer–Sørensen）；近年新硬件出现更强大的 native 门：√iSWAP（Huang 等，SQiSW 基础）、连续 ZZ(θ)（等价 XX(θ)/ZX(θ)/MS(0,0,θ/2)，IBM/Quantinuum/IonQ 共同采用）、iSWAP 族、CX 族分数门（如 {∛CX, √CX, CX}）、异构 basis 门（同时含 CX 族与 iSWAP 族）、以及镜像门（gate mirror，如 pSWAP 族、ECP）。这些 ISA 在合成能力（把任意 2Q 酉分解为 native 门的效率）与保真度上可超越 CX 等价门，如 SWAP 可用 "1 CX+1 iSWAP" 或 "3 √iSWAP" 实现，甚至原生高保真实现（脉冲时长仅 1.5×CZ）。
- 论文把 ISA 分为 6 类做跨 ISA 评估（Table II）：CX={CX}；ZZPhase={ZZ(π/6),ZZ(π/4),ZZ(π/2)}；SQiSW={√iSWAP,iSWAP}；ZZPhase_=ZZPhase+{pSWAP(π/6),pSWAP(π/4),pSWAP(π/2)}（镜像增强）；SQiSW_=SQiSW+{ECP,CX}（镜像增强）；Het=ZZPhase+SQiSW（异构组合）。镜像门集（ZZPhase_/SQiSW_）与 Het 都能显著降低路由开销。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架中，ISA 是成本模型的输入参数：① 定义各 basis 门单位成本（式3，基于物理时长/保真度的归一化：CX:1、ZZ(π/t):2/t、√iSWAP:0.75、iSWAP:1.5、ECP:1.25、pSWAP(π/t):2−1/t，保证 Weyl chamber 关键边上成本连续，如 pSWAP(π/2)≡iSWAP 同为 1.5）；② 用 monodromy 多胞形把 ISA 合成能力编码为 Weyl chamber 内的覆盖集；③ 路由启发式用该成本表计算 c_g、Δdepth、c_swap，做出 ISA-aware 选路；④ 换 ISA 只需更新成本表（配置步骤），算法无需改动——这是 CANOPUS "LLVM 式"统一编译的关键。评估时用一致成本表保证跨 ISA 公平比较（Ccount/Cdepth 是广义门数与深度）。
- 编译管线位置：逻辑级优化（CX 表达、模板/peephole）→ qubit routing（CANOPUS，规范表示+ISA 成本）→ 最终 ISA rebase（后端 synthesizer，如 Qiskit ZZPhase 合成功能、BQSKit 数值合成）。真机验证（ibm_marrakesh，Heron-R2，native 门 {CZ, √X, Z(θ), ZZ(θ)}）显示 CANOPUS 编译的 QFT 相对 QISKIT 默认降低 CZ/CX 门集错误 26.89%、ZZ(θ) 门集错误 34.98%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：ISA 体现在 CANOPUS 的 backends.py（backend/ISA/cost-estimator 模型）与 synthesis.py（rebase_to_sqisw/zzphase/custom 等）；Qiskit 生态中 basis gate 由 target/backend 提供，synthesis 用 `qiskit.synthesis`（如 ZZPhase 用 TwoQubitBasisDecomposer，见 Qiskit 最新合成功能 [31][55]）。扩展新 ISA：配置步骤——指定目标 basis 门单位成本。硬件现实：CZ 与 iSWAP 在主流超导平台（Google Sycamore、IBM）可原生支持；AshN 门方案（[12][13]）可任意 basis 门直接实现且达到最优门时长，为真机验证多样化 ISA 铺路。
- 论文协同设计结论：程序模式与 ISA 匹配很重要——ising（哈密顿模拟，含大量 2-local Pauli 旋转）选 ZZPhase 最佳；CX/CZ 主导的 qec9 选异构 Het 收益最大；镜像门集（ZZPhase_/SQiSW_）天然利于 SWAP mirroring。这些是 program-ISA-topology 协同探索的具体指南。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
