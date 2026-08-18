## Repeat-Until-Success Fusion（RUS 融合，重复直至成功）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Repeat-Until-Success（RUS）融合是增强融合成功率的一种 boosted fusion 方案（Lim, PRL 95, 030505, 2005；并在 Gliniasty 等的 spin-optical 架构 [21] 与 Thomas 等的融合实验 [66] 中沿用）：使用辅助光子（ancillary photons）在两个光子源之间反复施加融合操作，一旦某次融合成功即终止，从而在 caterpillar 态之间建立纠缠。其思路与冗余编码类似（多次尝试），但"成功即停"避免冗余编码的固定 m 次尝试浪费。逻辑错误率：P_fail=p_fail^m（失败率被压缩），P_eras=Σ_{i=0}^{m-1} p_fail^i·2p_eras——每次尝试暴露 2 个新量子比特给擦除，失败尝试还会累积擦除暴露。相比冗余编码，RUS 略优（擦除暴露随失败次数累积而非固定 2m），但消耗更多 ancilla 资源、耗时更长，且与冗余编码同样对擦除不容错。论文中 RUS 作为树编码融合的 baseline 之一：代码尺寸 m_RUS=6（按 [12][25] 最优容错性能），在真实硬件实验中 RUS+photonic 是 MemTree 的对比对象（PST 2.68×/IST 3.23× 落后于 MemTree）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RUS 融合的计算过程（伪代码）：
```
attempt = 0
while attempt < m:
    attempt += 1
    outcome = type2_fusion_with_ancilla(source_A, source_B)
    if outcome == SUCCESS:
        entanglement_established()      # 成功即终止
        break
    elif outcome == FAILURE:
        continue                        # 消耗 ancilla，重试
    elif outcome == ERASURE:
        # 无容错：纠缠建立与否未知，整体结果必须丢弃或依赖重试
        record_erasure()
# 逻辑错误率：
# P_fail = p_fail^m
# P_eras = Σ_{i=0}^{m-1} p_fail^i * 2p_eras   （每次尝试 2 个新 qubit 暴露给擦除）
```
例：p_fail=0.25、p_eras=0.1、m=6：P_fail≈2.4×10^-4，但 P_eras≈2×0.1×(1-0.25^6)/0.75≈0.266——擦除率显著高于失败率，成为主导错误。RUS 在论文 Fig.4(f) 模拟中相对树编码明显退化（擦除率升高时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 线性光学——需要 ancilla 光子注入与重复融合测量硬件；(2) 编译集成——论文按 [12][25] 的最新协议实现 RUS 并集成进 MemTree 编译框架作对比（m_RUS=6）；(3) 真实硬件——论文在 Quandela 云平台用 Perceval 实现 RUS+photonic 作为真实硬件实验 baseline。使用场景：需要提升概率性融合成功率的 PQC 图态生成；与树编码相比，RUS 在擦除率 >0 时性能快速退化、且 ancilla 开销高，是论文 ablation 的关键证据——MemTree 换用 RUS 后除 Grover 外全面劣于 OneAdapt-ET，证明树编码才是性能来源。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
