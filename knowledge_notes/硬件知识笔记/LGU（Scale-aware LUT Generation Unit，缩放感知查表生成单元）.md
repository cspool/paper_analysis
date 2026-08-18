## LGU（Scale-aware LUT Generation Unit，缩放感知查表生成单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LGU 是 Omni-LUT（ISCA 2026，NYCU）LUT-based GEMM 加速器中负责预计算 scaled partial sums 并生成查找表的硬件单元。它解决的关键问题是"量化方向"：现有 LUT-based GEMM 加速器（如 FIGLUT [52]）在查表之后应用单一 scale（post-lookup scaling），只支持 column-wise 缩放（Fig.7a）；而 Weights 最优 row-wise 缩放 [73]、Keys 最优 per-channel（在 QK^T 中等价于 row-wise K）[46]、Values 最优 per-token（在 Attn×V 中等价于 row-wise V）——精度最优方向是 row-wise（Fig.7b），post-lookup 单一 scale 无法给组内 4 个元素不同缩放因子。Omni-LUT 因此把缩放放进 LUT 生成阶段（scale-aware generation）：LGU 为每个 bit-plane 先用 4 个 FP 乘法器把 4-element 激活组内每个激活乘各自的 row-/channel-/token-specific 缩放因子，再枚举所有组合写入表项；同时内嵌 zero-point 补偿——首 bit-plane 计算激活组与其 per-row zero-point 向量的点积、加进生成的每个表项，免去最终单独加法。这是把 LUT 执行从 AW-GEMM 扩展到 AA-GEMM（Key/Value 两个运行时激活操作数）而不损失精度的关键使能器。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - LGU 微架构是两段流水（Fig.8）：(1) scaling stage——4 个 FP 乘法器把当前 bit-plane 的 4 个激活分别乘其缩放因子；(2) table generation stage——用 reuse-oriented 加法/减法网络形成 4 元素组的 8 个基础条目（仅需 6 个加法器 + 6 个减法器），并物化 half-LUT [T-MAC]：PE 利用符号对称（sign symmetry）获得互补条目，即 4 激活的 2^4=16 种组合中只显式生成 8 个基础项。运转流程：activation buffer 中的输入激活与其缩放因子取到 LGU → 生成 scale-aware LUT → 流式送入 PE array；LGU 与 PE array 并行工作——生成完一个 LUT 后立即开始生成下一个激活组的 LUT，而 PE 同时消费当前 LUT，因此 LUT 生成不引入额外 stall cycle。生成后的表由 32 个 binary weight 在 PE 内并行查表（Read-and-Accumulate，RAC）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Verilog/RTL 综合（Omni-LUT 用 Synopsys Design Compiler TSMC 7nm @500MHz），面积 0.017 mm²（占 compute core 3.11%）、能量 3.43 J（占 5.36%，OPT-6.7B 8192/512 场景，Table VII）——相对 PE array 的 86.34% 面积占比很小。使用：作为 LUT-based GEMM 加速器的配套生成器，替代"查表后乘 scale + 加 zero-point"的旧式路径；每 4 激活组生成一次、表被多个 activation tile × 量化列重用。row-wise 缩放在 A×W 表述下即对激活逐元素缩放；论文注明其"row-wise"等价于 W×A 表述的"column-wise"。参考：T-MAC [67] 的 half-LUT 符号对称思想、FIGLUT [52] 的 LUT 加速器。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization
