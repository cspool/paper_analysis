## Bit-composable 架构（可组合乘法器 PE，O(n²) dynamic tax）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bit-composable（位可组合）架构指把多个小的计算单元（PE）动态融合以模拟更宽位宽操作的加速器设计，经典代表是 Bit Fusion（ISCA'18）：可重构整数 PE 按需组装成更宽的算子。这类架构的优势是可配置（一个硬件覆盖多种位宽/量化格式），但根本局限是乘法器成本随位宽平方增长——部分积生成与累加规模 O(n²)，因此融合后 PE 的吞吐随精度增加按 1/n² 下降（"dynamic tax"）。UNICORE 论文将其与加法型可组合（S-FPMA，O(n) 线性）对比：乘法器型设计在 W4A4 高效、切到 W8A8/W16A16 时吞吐塌缩，无法同时保证高位宽性能与位宽灵活性。UNICORE 的硬件 baseline OliVe、Tender、M-ANT 即属此类 bit-parallel composable-multiplier 设计。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 运转流程（以 W8A8 融合为例）：两个 4-bit 乘法器经 bit-decomposition 组合成 8-bit 乘法器，需要把操作数高低位交叉相乘（W_lo×A_lo、W_lo×A_hi、W_hi×A_lo、W_hi×A_hi）再按位权重移位累加——部分积数量随位宽平方增长；融合宽度翻倍时 PE 面积与时延近似 ×4，同面积约束下可部署的 PE 数按 1/n² 下降，因此高位宽模式吞吐相对 W4A4 按 1/4、1/16 塌缩（图 3a）。UNICORE 实验量化：W16A16 时乘法器型 Tender 的 GEMM 面积与计算密度明显劣于 S-FPMA（UNICORE 高 1.32×@W4A4 → 2.63×@W8A8 → 5.26×@W16A16）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RTL 中每 PE 含可组合乘法器（partial-product 生成网络 + 加法树/进位保存累加）+ 融合选择逻辑；Bit-Fusion 有开源实现思路（动态位级融合 PE 阵列），后续 OliVe/Tender/M-ANT 在其上扩展 outlier 处理、张量分解（Tender）与自适应数值类型（M-ANT）。使用：用于自适应量化/outlier 处理的加速器场景，适合低比特（4-bit）高吞吐需求，不适合多精度统一部署。UNICORE 论文用它作为对比 baseline 论证"乘法器型可组合的 O(n²) 缺陷"，反衬 S-FPMA 的线性可扩展。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
