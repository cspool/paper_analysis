## Feature Average Bit-width (FAB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Feature Average Bit-width (FAB) 是动态量化方法中衡量量化效率的核心指标，定义为测试数据集上所有特征图在所有 patch 上的平均 bit-width。FAB 越低量化越激进，计算和存储开销越小。与固定比特量化（如 PAMS 的恒定 8-bit）不同，动态量化的 FAB 随图像内容自适应变化，反映实际使用的平均精度水平。计算方式：

$$\text{FAB} = \frac{1}{|\mathcal{D}|} \sum_{X \in \mathcal{D}} \frac{1}{M} \sum_{i=1}^{M} b_i$$

其中 $\mathcal{D}$ 为测试集，$M$ 为图像 $X$ 的 patch 数，$b_i$ 为第 $i$ 个 patch 的 bit-width。

从算法pipeline角度拆解术语，给出具体例子。

在 Granular-DQ 评估中，FAB 与 PSNR/SSIM 构成 trade-off：EDSR ×4 SR 在 Urban100 上，全精度 FAB=32.00 (PSNR 26.03dB)，PAMS FAB=8.00 (26.01dB)，CADyQ FAB=6.09 (25.94dB)，Granular-DQ FAB=4.97 (26.01dB)。Granular-DQ 以最低 FAB 实现与全精度相当的 PSNR。FAB 与 BitOPs（bit-weighted operations）配合，FAB 侧重"平均精度"，BitOPs 侧重"实际计算量"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FAB 是评估指标而非实现组件，通过统计测试集所有 patch 的 bit 分配计算。被 CADyQ、CABM、AdaBM、RefQSR 等动态量化方法广泛采用作为统一的效率度量。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues
