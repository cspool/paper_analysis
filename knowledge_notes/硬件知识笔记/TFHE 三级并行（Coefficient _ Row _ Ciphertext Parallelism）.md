## TFHE 三级并行（Coefficient / Row / Ciphertext Parallelism）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 TFHE 加速器设计对"外部乘积（GGSW □ GLWE）"暴露出的三个可并行维度（FlashTFHE 论文系统性提出）：(1) coefficient 并行——单个 degree-N 多项式内部的 N 个系数级乘累加，由 FFT/PolyMult 流水线宽度决定；(2) row 并行——同一 GLWE 多项式可同时作用于 GGSW 密文的多个"行"（(k+1) 个独立点积，k 为 GLWE 维数），空间架构把它映射到一行并行 PE 上广播复用 post-FFT 结果；(3) ciphertext 并行——同时处理多个独立 LWE ciphertext 的多个 PBS，程度由程序决定、在程序内变化。三者的关系：Throughput = (#在飞 ciphertext) × (#每周期处理的 coefficients)。
- 该三分法解释了两个设计事实：低 bit-width TFHE（k=2~3、N 小）空间阵列靠 row + ciphertext 并行吃饱；multi-bit TFHE 参数（k=1、N 至 2^16）使 row 并行塌缩到最小 2× 复用、ciphertext 并行复制流水线代价随 N 爆炸，只剩 coefficient 并行是面积高效的缩放维度——但空间架构加宽系数吞吐又受 DRAM 带宽硬约束。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 空间架构映射（Morphling 式）：coefficient 并行在 PE 内（每 PE 一个窄 FFT/MAC），row 并行在 PE 行（4 个 PE 做 k+1=4 个点积），ciphertext 并行在行间（数组行数 = 设计时固定的 batch）。FlashTFHE 时间域映射：coefficient 并行 = 单片宽流水线（512 coef/cycle）的 FFT/VecMAC 吞吐；row 并行不再依赖阵列几何（k=1 时只保留 2× 复用的 MAC 裁剪）；ciphertext 并行 = 编译器可控的 round-robin batch 大小（12–48），无需复制流水线。Figure 8 定量对比：同带宽下 temporal 系数吞吐 2×；8 ciphertext 时同面积吞吐 2.63×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：空间设计把三维硬编码进 PE 阵列几何（batch 固定）；时间域设计把 batch 变成软件参数（adaptive batching），用 lane masking/round-robin 调度实现。使用：加速器设计者先按 workload 画像（PBS 占比、并行 ciphertext 数）决定主缩放维度；FlashTFHE 实测多数真实 workload 并行 ciphertext 均值 106–6448（DNN/XGBoost 丰富、Decision Tree/KNN 低至 8–46），据此用 adaptive batching 匹配 batch 与并行度。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
