## Dual Redundancy in Visual Autoregressive Models

术语是什么？
Dual Redundancy（双重冗余）是 VAR-Turbo 论文的核心理论框架，从信息论角度刻画 VAR 图像生成模型的两类冗余：(1) Image Redundancy（图像空间冗余）—— 图像 visual token 的空间相关性远高于语言 token，相邻像素/token 高度相关，熵值低，使得无需 draft model 即可在同一轮中预测和选择多个 token 并行解码；(2) Model Redundancy（模型计算冗余）—— attention 机制作为低通滤波器，随层数加深使 attention map 趋于相似、token 重要性分化，导致深层的大部分 token 可安全跳过完整计算。Dual Redundancy 为 PD（利用 Image Redundancy）和 TA+DB（利用 Model Redundancy）提供了理论依据。

从算法pipeline角度拆解术语：
Dual Redundancy 驱动的算法设计：
1. Image Redundancy → Draft-Free Parallel Decoding (PD)：
   - 图像中相邻 visual token 高度空间相关→模型对相邻位置的预测置信度高
   - 可直接按置信度选 TopK token 并行解码（无需 draft model）
   - 每轮可解码 8-64 token（vs speculative decoding 的 2-3 token）
2. Model Redundancy → Token Aggregation (TA) + Dynamic Bypass (DB)：
   - 浅层 attention 局部 token 相似但全局 pattern 不同 → TA：Small Attention 压缩局部 + Big Attention 保留全局
   - 深层 attention 全盘趋于相似、token 重要性分化 → DB：仅 TopK 重要 token 完整计算，其余 bypass

术语一般如何实现？如何使用？
Image Redundancy 通过 entropy/redundancy 量化分析验证（对比语言 token 和图像 token 的条件熵分布）。Model Redundancy 通过逐层 attention map cosine similarity 和低通滤波频率响应分析验证。PD、TA、DB 三项算法各自对应其中一类冗余，三者叠加实现跨迭代和迭代内的联合加速。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

