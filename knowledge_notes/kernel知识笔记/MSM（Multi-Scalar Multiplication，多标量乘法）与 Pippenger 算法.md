## MSM（Multi-Scalar Multiplication，多标量乘法）与 Pippenger 算法

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSM 是椭圆曲线密码学的核心原语：计算 R = Σ_{i=0}^{N-1} s_i ⊗ P_i（N 个标量 s_i 与 N 个椭圆曲线点 P_i 的带权求和），是 Groth16/HyperPlonk 的 KZG 承诺/打开的 dominant kernel（占 prover 时间 59–70%）。Pippenger 算法（FOCS 1976）用"加窗口 + 桶累加"把昂贵的 PMUL 换成便宜的 PADD：标量 s 按 c-bit 窗口切分，桶累加（bucket accumulation）阶段按窗口值把点 P 分发进对应桶并 PADD 累加，桶归约（bucket reduction）阶段把桶值按桶号加权求和 Σ i⊗B_i，最后窗口聚合按 2^c 幂加权合并。算术强度高（170 modmul/元素），是计算密集型 kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pippenger 三段调度（GenZA 映射）：
```
# ① 桶累加（可并行）：每点每窗口独立
for each point P_i, for each c-bit window w of s_i:
    bucket[window][w] += P_i          # 分发到对应桶 PADD
# ② 桶归约（先 PE 内并行、后跨 PE 串行）：
for each window: B_window = Σ i ⊗ bucket[i]   # 两个顺序 PADD 求和优化 [59]
# ③ 窗口聚合（忽略级）：R = Σ B_window ⊗ 2^(c·win)
```
- Annotations：GenZA 的调度要点：(1) 动态 window size 选择——离线成本模型按曲线 bitwidth、MSM 大小 N、片上 SRAM、带宽选 c（MNT4-753 从 2^14 的 c=11 到 2^23 的 c=16），对照 zkSpeed 固定 c=9 最高 2.90× 加速；(2) window-major 映射——桶数超片上 SRAM 时每轮从所有 window 各取子集桶（而非整 window 的桶），提高桶归约阶段并行度（同 window PE 串行归约，并行度正比于片上 window 数），免去 LegoZK 的树式归约；(3) 单 PADD 用 complete addition formula（齐次射影坐标，免模逆）需 12 模乘+2 常数乘：MNT4-753 全 32 lane 组单 PADD 单元（2 个宽乘法器时间复用 14 次模乘），BN128 每 PE 2 个 PADD 单元（4 个宽乘法器，常数乘化简为加法）；(4) 附加优化：signed-digit 把桶数从 2^c−1 减半到 2^{c-1}，sparse MSM 预累加标量 1 的点；(5) 分发由 MSM decoder & dispatcher 在 NoC 前复制点注入对应行（BN128 c=16 平均每点到 ~4 PEs），桶命中均匀无热点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU/GPU 用 cuZK/GZKP/Bellman 的 MSM（GPU 用多 GPU 或 Pippenger 并行化）；FPGA 用 CycloneMSM/PipeMSM/BSTMSM；ASIC 用 PipeZK/SZKP/zkSpeed/GenZA 的专用或统一单元。使用：任何配对型 PCS（KZG）的承诺与打开；GenZA 中由 PE 阵列 + decoder&dispatcher 执行，MSM 大小与 bitwidth 决定最优 window c，调度器离线选定后硬件配置 window/桶数。NoC 评估（packet-level 模拟）：BN128 c=16 下 dispatch stall 仅 3.97%、平均 link 利用 5.9%、最热 link 峰值 44.51%。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
