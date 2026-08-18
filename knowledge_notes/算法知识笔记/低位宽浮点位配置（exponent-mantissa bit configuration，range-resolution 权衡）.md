## 低位宽浮点位配置（exponent-mantissa bit configuration，range-resolution 权衡）

术语解释
同一比特宽度下浮点数可分配不同的 exponent/mantissa 位数（如 4-bit 可配 E0M3、E1M2、E2M1、E3M0，S1ExMy 记法：1 符号位 + x 指数位 + y 尾数位），指数位多则动态范围大、分辨率低，尾数位多则分辨率高、范围窄——构成"范围-分辨率"（range-resolution）权衡。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：浮点数由 sign/exponent/mantissa 三段组成；固定总位宽（如 4-bit）下三段位宽互相挤占；指数位决定可表示的最大/最小指数（范围），尾数位决定相邻可表示值的间距（分辨率）；因此 E0M3（全尾数）只适合值域窄但需高精度的数据，E3M0（全指数）适合动态范围大但精度要求低的数据，E1M2/E2M1 是中间权衡（Fig.3 显示四种 4-bit 配置的可表示值分布明显不同）；没有单一配置对所有数据分布最优——这是 MXFFP 的核心动机。论文实测：oracle（逐 block 选最优配置）下 E1Mx 与 E2Mx 两类配置覆盖 97.2% 的最低 MSE 选择，E0/E3 等极端配置仅 2.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在量化 pipeline 中，位配置直接决定 block 内每个元素的量化误差。以 MXFFP4 的相对指数比较为例（E1M2 vs E2M1，表 I）：元素相对指数 E_i^r=E_i−E_b^MAX（≤0）为 0 时，E1M2 用 2 位尾数（2^1×1.xx₂）分辨率高于 E2M1（2^1×1.x₂）；E_i^r∈{−2,−3} 时 E2M1 凭借额外指数位（2^0×1.x₂）比 E1M2（2^1×0.01₂）表示得更细；E_i^r=−1 或极小值两者等价。选择规则（Algorithm 1）：count_E1=|{E_i^r=0}|（偏好 E1M2）、count_E2=|{E_i^r∈{−2,−3}}|（偏好 E2M1），若 count_E1²>count_E2 选 E1M2（E_element^MAX=1）否则选 E2M1（E_element^MAX=2），共享指数 E_b^shared=E_b^MAX−E_element^MAX。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上每种配置是数据通路中的一段位映射（exponent 位域长度、bias、mantissa 位域长度不同）；MXFFP 用每 block 1-bit 配置字段在 E1Mx/E2Mx 两配置间选择（8 个配置位合成 1B configuration set 保持字节寻址）；Tensor Core 用 bit mapper（多路选择器阵列）按配置位把操作数重排进统一算术核（FP4 E2M1 核加宽为内部 E2M2）。若需更多配置可换 preset 对（E0/E2、E2/E3）或加宽选择器。使用场景：低位宽（4/6/8-bit）post-training quantization 中按数据分布动态选择位分配，提升表示精度。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
