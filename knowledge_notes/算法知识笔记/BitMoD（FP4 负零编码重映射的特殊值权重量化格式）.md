## BitMoD（FP4 负零编码重映射的特殊值权重量化格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BitMoD 是 P3-LLM 采用的 4-bit 权重量化格式（论文引用自文献 [6]），核心思想：FP4（4-bit 浮点）的基本量化值集合为 {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}，其中 ±0（尤其负零）的编码是冗余的——同一数值 0 对应多个编码。BitMoD 把这个冗余的负零编码重映射为 4 个预定义的特殊值 {±5, ±8}，并按权重分组搜索每组最优的一个特殊值来替换负零，从而比非对称整数量化（INT4）更小地降低量化误差，硬件开销极低（仅需解码器将特殊值映射回）。相比 MANT 的自适应数值类型（需把乘法分解成两个高精度部分和，增加面积/能耗）与 Ecco 的 k-means codebook + Huffman 编码（需在线解压回 FP16），BitMoD 的 6-bit 定点表示（4-bit 值 + 组级缩放相关位）能被低精度乘法器直接消费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BitMoD 权重离线量化与在线解码：
```
# 离线（每 group 128 个权重）：
Q_base = {-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6,SPECIAL}
for v in {±5, ±8}:          # 候选特殊值
    Q = Q_base with 负零 -> v
    err(v) = Σ |w_i - nearest(Q, w_i)|²
v* = argmin err(v)           # 选最优特殊值
Wq[i] = index of nearest(Q(v*), w_i)   # 4-bit 索引 + 组级 scale + v* 元数据
# 在线 MAC（PIM PCU，见硬件架构条目）：
product = w_mantissa(6-bit 定点，含 v* 解码) * x_mantissa(6-bit)
product <<= x_exponent(4-bit)          # 指数移位
acc += product                          # 32-bit 定点累加
# 线性层：GEMM 完成后统一乘组级 dequant scale（fusion 后置）
```
效果：消融实验（Table VI）显示 INT4 权重量化使 Llama-2-7B Wikitext-2 PPL 增 0.13，换用 BitMoD 后降至 0.01 增量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧在 AWQ 流程之上实现（开源仓库 https://github.com/yc2367/P3-LLM 的 `wq_dtype=bitmod`，group size 128）；硬件侧 PCU 为权重与 KV-cache 共享同一 6-bit 定点乘法器——由于权重（BitMoD）与 KV-cache（INT4-Asym）映射到 MAC 硬件的同一操作数位置，PE 内需要一个小解码器同时支持两种格式（BitMoD 特殊值 6-bit、INT4-Asym 5-bit 含 zero-point）。与仅权重量化（weight-only）不同，BitMoD 在 P3-LLM 中是 W4A8KV4P8 的一部分：权重 4-bit 压缩内存、激活保持 8-bit 降低精度损失，从而在低精度 PIM 上同时获得带宽与计算收益。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
