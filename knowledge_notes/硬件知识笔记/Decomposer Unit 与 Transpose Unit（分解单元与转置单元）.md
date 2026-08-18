## Decomposer Unit 与 Transpose Unit（分解单元与转置单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 两个是 FlashTFHE 中支撑外部乘积与 FFT 的辅助功能单元：(1) Decomposer（gadget 分解）——把 torus 多项式的每个元素按 2 的幂基 B 表示成整数向量、深度 l_b（multi-bit 参数 l_b 常为 8–10），分解结果与 GGSW 的整数矩阵做乘累加，是外部乘积的输入准备；硬件实现分两部分：初始 scaling 单元（分解深度>1 时可能引入 stall）+ 连续 digit-extraction 单元（每周期输出一个整数、内置 rounding，维持 FFT 单元吞吐）。(2) Transpose——两类使用场景：FFT 单元内部的转置（同一周期数据可及，只需 mux）与 FFT-A↔FFT-B 之间的大输入 divide-and-conquer 转置（中间数据需缓存），后者用两端口 SRAM bank 高密度存储 128 行×256 列的复数中间数据（每复数 96-bit）；因 256 列读耗时 2× 于 128 行，把相邻两复数分组为 128 逻辑列、按奇偶索引分发到两个 FFT-B 并行处理。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 BRU 流水线中的位置：外部乘积每迭代 = Decomposer 把 GLWE/GGSW 多项式分解为 l_b 层整数 → 各层与对应 BSK subchunk 在 VecMAC 乘累加（FFT 域）→ 转置单元在 FFT-A/FFT-B 间搬运中间数据 → 全部累计后 I-FFT。面积/功耗（16nm，Table I）：Decomposer 0.24mm²/0.65W、Transpose 0.79mm²/1.44W、VecMAC 4.27mm²/8.41W。瓶颈控制：decomposer 的 rounding 内置与每周期一整数吞吐保证不成为 FFT 吞吐瓶颈；transpose 的两端口 SRAM + 128 逻辑列分组消除吞吐不平衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Decomposer 为 scaling + digit-extraction 两段流水（类似 FHE 加速器通用的 gadget decomposition 硬件，MATCHA/Morphling 均有对应模块）；Transpose 为 mux 网络 + 双端口 SRAM 存储。使用：多 bit TFHE 的 l_b 高达 8–10 使分解工作量与中间数据搬运占比上升，两单元必须与 FFT/VecMAC 同吞吐设计以免反成瓶颈。论文未开源 RTL（联网未找到仓库）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
