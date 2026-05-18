## Learning Region / Inert Region

术语是什么？
Learning Region 和 Inert Region 是 VAR-Turbo 论文对 VAR Transformer 层按其 attention 行为特征所做的两类分区。论文将 attention 解释为低通滤波（low-pass filter）：随着层数堆叠，attention map 逐渐趋于相似（softmax 后的分布平滑化），高频信息被削弱。浅层 attention 仍有较强的学习能力（不同 token/head 的 attention pattern 差异化明显），称为 Learning Region；深层 attention map 高度相似、token 重要性分化、层间信息增量低（趋于"惯性化"），称为 Inert Region。此分区决定 TA 应用于 Learning Region、DB 应用于 Inert Region。

从算法pipeline角度拆解术语：
分区判断依据（基于 attention entropy / similarity 分析）：
- Learning Region（如 0-15 层）：inter-layer attention similarity 低，attention 仍在主动学习 token 间关系。适用 TA：local window 内高度相关但全局需要区分，通过 Small+Big Attention 压缩局部相似性。
- Inert Region（如 16+ 层）：inter-layer attention similarity 高（cosine similarity 接近 1），深层 attention map 趋于均匀和相似。适用 DB：仅重要 token 进入完整计算，其余 bypass。
分区是经验性的，VAR-Turbo 通过实验确定分界层。

术语一般如何实现？如何使用？
分区通过测量各层 attention map 间的 cosine similarity 确定，实际部署时使用固定分界层（如第 15 层为界）。Learning Region 使用 TA（Small+Big hierarchical attention），Inert Region 使用 DB（importance-based token bypass）。此分区方法可推广到其他 deep Transformer 模型以识别不同层的计算冗余特征。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

