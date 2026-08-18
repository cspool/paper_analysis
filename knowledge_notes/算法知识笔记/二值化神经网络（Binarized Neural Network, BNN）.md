## 二值化神经网络（Binarized Neural Network, BNN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 二值化神经网络（BNN）是把权重与激活都约束为 1-bit（{+1,−1}，通常用 sign 函数）的神经网络。它把浮点乘加（MAC/FMA）替换为按位 XNOR + popcount 运算，实现约 32× 存储压缩与约 58× 计算量下降，适合超低功耗边缘/微架构硬件。代表作：BinaryNet（Courbariaux & Bengio 2016，权重与激活都二值化，用 STE 训练）与 XNOR-Net（Rastegari 等 2016，引入 BWN 仅二值权重+缩放因子 α、XNOR-Network 权重激活都二值化 → 卷积退化为 XNOR+bitcount）。逻辑链：1-bit 量化 → 乘法变位逻辑 → 面积/功耗数量级下降 → 但精度受损 → 靠先进训练技巧（STE、混合精度潜在权重、缩放因子）弥补。Moirai 论文把它用于 L1D 预取器，把 TCN 变成 CaPNet：权重与激活 1-bit，前向只用 XNOR/popcount，780 Bytes 存储、1178 μm² 面积、8.5mW 功耗（ASAP7 7nm @4GHz），精度仅比 INT8/full-precision 下降 <2%（Figure 19c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BNN 在 Moirai 中的计算 pipeline（一次前向）：
  1. 二值化：W_bin = sign(W_raw)（W_raw 为混合精度潜在权重，首层 7-bit、其余 4-bit）；激活 A_bin = sign(A)；
  2. 二值卷积：Ac^k = bitcount(A_bin ⊙ W_bin^k)（式 1，XNOR 逐位比较 + popcount 计数，等价于二值向量的点积：popcount(x_b XNOR w_b) = 匹配位数，64 位字打包后一条 XNOR+POPCNT 指令算 64 个乘法）；
  3. 结果经 sign 传给下一层，通道结构 [8,4,2] 逐层下采样；
  4. 反向（训练）：梯度经 STE 绕过 sign 的非可导点更新 W_raw（式 2：ΔW_raw^k = G_{i+1} * Ac_i^k）。
- 与浮点卷积对比：常规 y = Σ x_i·w_i（64 次乘法+累加）→ BNN 变成 popcount(x_b XNOR w_b)（1 次位运算簇）。论文合成评估：Moirai 二值化相对 INT8/full-precision 仅 <2% 精度损失，换取 >8×/>32× 存储缩减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练侧实现：保留全精度潜在权重（latent/raw weights），前向时 sign 二值化，反向用 STE 更新潜在权重（BinaryConnect/BinaryNet 范式）；实践中常保留首层（特征保真度敏感）与末层为更高精度——Moirai 即给首层 7-bit、其余层 4-bit 的混合精度 W_raw。硬件侧实现：XNOR gate 阵列 + popcount 加法树（Moirai 的 FCC），反向用 BCC（条件符号翻转器 + 浅加法树）。使用场景：边缘 NPU（STM32N6 类）、FPGA、ASIC、内存内计算；Moirai 开创了把它用于 L1D 预取器的新场景（BTCP 是 L1 之外用 B-TCN 的先前工作，4.5KB/134-cycle 只能放 L2）。开源参考实现：BinaryNet.pytorch、XNOR-Net-PyTorch。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
