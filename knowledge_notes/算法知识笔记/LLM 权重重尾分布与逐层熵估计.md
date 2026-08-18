## LLM 权重重尾分布与逐层熵估计

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 权重呈重尾（heavy-tailed）统计分布：大部分值集中在 0 附近（尖峰）、存在长尾离群值（outlier），且分布跨层显著变化。量化文献以 kurtosis、α-stable 分布刻画（QuaRot/Q-Palette 等用旋转把分布"Gaussianize"、KurTail 用 kurtosis 度量尾部、SmoothQuant 平滑激活离群）。对无损压缩的含义：重尾 → 符号频率高度偏斜 → 经验熵远低于名义位宽 → 熵编码可获得大压缩率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文的逐层熵估计即重尾分布的信息论量化（见 Shannon 熵条目伪代码）：逐 tensor 建直方图（而非全模型单一码本），因为分布跨层差异大。实测结果（1.5B–405B 模型一致）：bf16 熵 10–12 bits、int8 4–5 bits、int4 0.6–1.0 bits；低比特格式符号表小且分布尖锐 → ANS 几乎达熵界；群量化（sq8/awq4）的 per-group scale 元数据引入 1.1–1.3× 额外冗余。层共享 codebook 的策略正是"分布重尾但跨层相似"与"逐层有差异"之间的折中（聚合统计覆盖 + 元数据摊销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
统计实现 trivial（histogram）；工程意义在于指导码本粒度（per-layer/per-tensor codebook）与量化设计：AWQ 保护 salient 通道（0.1–1% 权重）、SmoothQuant 平滑离群、旋转类方法 Gaussianize 都是为了驯服重尾；无损压缩则直接利用偏斜分布做熵编码，与量化正交叠加。注意：经验熵是估计值，符号表大、样本少时偏高；论文对 bf16 的 0.1–0.2 bits 偏差即有限精度表的来源。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
