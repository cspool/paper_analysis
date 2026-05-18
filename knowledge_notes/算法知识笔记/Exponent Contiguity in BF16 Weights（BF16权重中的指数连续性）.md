## Exponent Contiguity in BF16 Weights（BF16权重中的指数连续性）

术语是什么？通过联网搜索让回答具体和精准。
Exponent contiguity是指LLM的BF16权重中，最频繁出现的top-K个exponent值在数值上形成连续序列的现象。ZipServ论文通过分析3875个weight matrices（覆盖Gemma-3、Mistral、Qwen2.5、LLaMA3.1四个LLM家族）发现：99.6%的矩阵中top-7 exponent构成连续序列（e*, e*+1, ..., e*+6），平均覆盖97.1%的权重。论文在Appendix A中给出数学证明：假设权重服从零均值正态分布N(0,σ²)，exponent概率函数P(X=x)=erf(2^(x+1)/(σ√2))−erf(2^x/(σ√2))是单峰的(unimodal)，因此top-K集合必然是连续的。这个性质是TCA-TBE设计的基础——它使编码仅需记录一个base_exp而非7个独立exponent值，将解码简化为base_exp + codeword的算术操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在TCA-TBE中，exponent contiguity被利用的方式：
1. Profiling阶段：扫描weight matrix exponent直方图 → 排序找top-7频率的exponent值
2. Contiguity检查：验证top-7是否连续 → 对于99.6%矩阵为连续，此时选择contiguous window覆盖，设base_exp = min(top_exponents) - 1
3. 编码：codeword 001–111直接映射到base_exp+1 ~ base_exp+7
4. 解码：exponent = base_exp + codeword，一条IADD指令完成
若无contiguity性质，则需7-entry lookup table（额外的shared memory访问和地址计算），contiguity将lookup简化为算术，消除shared memory往返。对比：Huffman/ANS等方法使用通用frequency-sorted codebook，无contiguity利用，每个symbol解码需查表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该性质在LLM中广泛存在（源自权重的高斯分布假设），可用于任何基于exponent压缩的方案。实现时：
1. 对目标模型做exponent profiling（轻量，仅需一次前向扫描）
2. 验证contiguity（top-7是否连续）
3. 若不连续（<0.4%矩阵），可fallback到通用lookup table编码
该性质的数学基础（单峰性+高斯权重）使得它在不同LLM family间可迁移，无需per-model调优。

涉及论文标题：
- ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

