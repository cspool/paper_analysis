## Qubit-Cycle Volume（qubit-cycle 体积 / 空间-时间体积）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- qubit-cycle volume 是 FTQC 资源开销的标准度量：volume =（物理 qubit 数）×（QEC cycle 数），其中 qubit 指物理 qubit、cycle 指一轮 syndrome measurement + 纠错（QEC cycle）。它同时刻画"空间"（物理 qubit/tile 数，由逻辑 qubit 数 × code distance 决定）与"时间"（QEC cycle 数，由电路关键路径决定）两个维度，是权衡不同 FTQC 架构/编译策略的统一指标。TACO 用它做架构决策目标：最小化总 space-time volume = 平衡 magic state 工厂的物理 qubit 开销与提升电路吞吐（缩短 cycles）的收益。
- 论文用途（TACO）：①每逻辑门 QEC-cycle 成本（Pauli 0、CNOT/H 3d+4、S 1.5d+3、T 2.5d+4）按关键路径累加得架构无关 cycle 上界；②cycle-accurate 模拟器逐层路由（greedy/LSQECC）得真实 cycle 数；③物理 qubit = tiles×2d²（d 由 total_tiles×total_cycles×d×p_L<0.01 反推）。对 20 比特 QFT：PBC Compact 41 tiles/28.8M cycles/体积 1.0×10^12，PBC Fast 170 tiles/2.38M cycles/2.9×10^11，TACO 404 tiles/760,901 cycles（MSD）/2.2×10^11——TACO 体积比 Compact/Fast 降 79%/24%（MSD）、95%/63%（MSC）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TACO 中的计算流程（20 比特 QFT，MSD 协议）：
```
输入: TACO 优化电路 + 架构配置（compute/memory 块数、magic state 吞吐）
① 模拟: cycle-accurate 逐层推进 → QEC cycles（如 760,901）
② 计 tile: data 29 + compute 12 + distillation 363 = 404 tiles
③ 定距离: d=19（满足 tiles×cycles×d×p_L<0.01，p_L=0.1(100p)^((d+1)/2)，p=10⁻³）
④ 换算: 物理 qubit/tile=2d²=722 → 总物理 qubit ≈ 404×722
⑤ 体积 = 物理 qubit × QEC cycles → 2.2×10^11（MSD）
对比 PBC Compact/Fast → 79%/24% 体积降
```
- 吞吐灵敏度（Table III）：magic state 4/3/2 个每 round → 物理 qubit 64.5k/47.6k/30.8k、cycles 595k/760k/1.14M，体积仍比 PBC Fast 低 2.7×/2.9×/2.99×——说明体积权衡下 TACO 鲁棒。工厂数量选择：图 15 显示 18 比特 QFT 最优 4 个 magic-state block（比 1 block 体积降 57%），再增吞吐收益递减（空间开销吃掉 cycle 收益）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：资源估计方法学（[32,34,51] 标准流程）——给定电路与物理错误率 p，先用逻辑门延迟模型/cycle 模拟得 cycle 数，再按 code distance 公式选 d、按 2d² 换算物理 qubit，最后相乘得 volume。使用场景：FTQC 架构对比（本论文 PBC vs TACO）、distillation 工厂规模优化、code distance 选择；与"逻辑门数""物理 qubit 数"等单维指标互补——它是端到端成本代理。局限：物理 qubit 数随 d² 增长、cycle 数随电路深度增长，两个方向此消彼长，故"最小体积"是比"最少 qubit"或"最快"更合理的单一目标（TACO 明确以它为架构目标）。

涉及论文标题：
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
