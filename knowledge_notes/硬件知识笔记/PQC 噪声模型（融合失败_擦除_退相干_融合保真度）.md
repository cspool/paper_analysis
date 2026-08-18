## PQC 噪声模型（融合失败/擦除/退相干/融合保真度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PQC 噪声模型是论文为 spin memory 架构模拟器建立的 realistic 误差模型，覆盖光子 MBQC 执行中的四类主导噪声：(1) 融合失败（fusion failure）——1-p_fail=0.75（无擦除时的融合成功率，对应 OneAdapt 论文 Sec 5.1 错误模型，需额外干涉测量装置实现）；(2) 融合擦除（fusion erasure）——光子丢失，真实硬件观测 p_eras≈10%，仿真 0%~10%；(3) 光子源退相干（source decoherence）——F_de=e^(-N_e·T_gen/T2)，其中 T2=2.34 μs 由 4-qubit GHZ 态 95% fidelity 反推（对照：PsiQuantum T2=2.04 μs、RLGS 4.4 μs）；(4) 融合相干误差（fusion infidelity/indistinguishability）——F_fus=σ_fus^N_fus，σ_fus=(1+V_HOM)/2=99.75%（HOM 可见度 V_HOM=99.5%，来自 PsiQuantum [52]），OSRP fidelity 99%。统一 fidelity 对比表（Table II）：OneAdapt（PsiQuantum, T2=2.04 μs, CZ 99.75%, t_cycle=8 ns）、RLGS（[57] 仿真, 4.4 μs, 99%, 10 ns）、MemTree（Quandela, 2.34 μs, 99%, 30 ns）。噪声模型的意义：让跨架构编译器对比（OneAdapt/RLGS/MemTree）在同一 fidelity 框架下公平进行，并揭示"操作频率 vs 光子利用率"的 trade-off（Table III：OneAdapt 高频低利用率→融合主导误差；RLGS 高利用率低频→退相干主导；MemTree 居中达到平衡）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
噪声模型在模拟器中的运转流程（fidelity 计算与仿真配置）：
```
# 配置（模拟器输入）:
p_fail = 0.25; p_eras = 0~10%; T2 = 2.34 μs; σ_fus = 99.75%
t_cycle = 30 ns; init = 12 ns; per_qubit = 0.6 ns; caterpillar ≤ 30 qubit
# 执行模拟（2x10^4 个 caterpillar 发射周期）:
for cycle in 1..20000:
    caterpillar = emit()                    # 含退相干: F_de = e^(-N_e*T_gen/T2)
    logical = prepare_tree_encoded(caterpillar, b=4, b_prep=6)  # 制备误差: 失败/擦除恢复
    fused = hierarchical_fuse(logical)      # 每融合乘 σ_fus, 按结果走测量模式
    if fused == target: shots += 1
# 输出指标:
avg_exec_time = total_cycles_time / shots   # 平均执行时间
photon_sources = count_photon_sources()     # 所需光子源数
F_fus = σ_fus^N_fus; F_de = e^(-N_e*T_gen/T2)   # fidelity
```
关键机制：融合错误与退相干相互放大——执行时间越长（融合错误多）→ 累积退相干与 CZ 误差越多 → QAOA 需更多调优迭代（Fig.3），因此抑制融合错误（树编码）同时改善执行时间与 fidelity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：论文自研 realistic error-aware 模拟器实现该噪声模型（未开源、未提供链接），配置取自实验工作 [6][25][29][30][43][52]（PsiQuantum/Quandela/量子点实验）；退相干模型 F_de=e^(-N_e·T_gen/T2) 沿用 RLGS [38]，T2 由 Bell/GHZ fidelity 反推（T2=-N_q·t_gen/ln(F_state)）。使用场景：任何需要"在真实硬件不可及的大规模（36~100 qubit）下评估光子 MBQC 编译/容错方案"的仿真工作；噪声参数可作为后续光子编译研究的统一对比基准。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
