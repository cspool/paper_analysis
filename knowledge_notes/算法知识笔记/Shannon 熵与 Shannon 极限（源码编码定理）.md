## Shannon 熵与 Shannon 极限（源码编码定理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shannon 熵 H(X) = −Σᵢ pᵢ·log₂pᵢ 是离散随机变量 X 的平均信息量（自信息 −log₂pᵢ 按概率加权）。Shannon 源码编码定理（无噪编码定理，1948）确立无损压缩的比特数下界：对任意前缀码平均码长 L ≥ H(X)，且存在码使 L ≤ H(X)+1；对 n 个 i.i.d. 符号，n 足够大时可压到 n·H(X) 比特而几乎必然无损，少于 n·H(X) 则几乎必然有损——H(X) 即"无损编码的 Shannon 极限"，与具体编码算法、数值格式、硬件布局无关。符号码等长只在 p=2⁻ᵏ 形概率下才可能（Huffman 离熵界的差距即源于此）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM 权重压缩中的用法（论文附 B）——把每个权重张量视为离散信源、逐层估计熵：
```
for l in layers:
    symbols = weight[l] as discrete codes        # 按数值格式取符号
    p_i = histogram(symbols) / |W^(l)|           # 经验分布
    H^(l) = -sum_i p_i * log2(p_i)               # 逐层熵（bits/weight）
H_model = sum_l H^(l) * |W^(l)| / sum_l |W^(l)|  # 按参数量加权平均
```
Annotations：|W^(l)| 为层 l 参数量；H_model 即任意无损编码下的 bits/weight 下界。论文实测：bf16 名义 16 bits 但熵仅 10–12 bits（冗余 4–5 bits/weight，约 1.5× 空间）；int8 熵 4–5 bits；int4 熵仅 0.6–1.0 bits（熵比 6–10×）；sq8/awq4 仍有 1.1–1.3× 冗余——由此证明无损压缩理论上最多 10× 空间。论文同时以 H 为基准对照 ANS 码率：实测与熵界差 0.01–0.05 bits/weight。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：直方图统计 + 熵求和（numpy/GPU 均可）；关键工程点是逐 tensor（而非全模型）建直方图——LLM 权重分布跨层差异大，层共享 codebook 是统计覆盖与元数据开销的折中。注意经验熵受样本量与符号表大小影响：bf16 的 2¹⁶ 符号表使有限精度 ANS 表（b=12）产生约 0.1–0.2 bits 定标偏差。使用：评估量化/压缩方法剩余冗余、为无损编码选型提供下界（本论文据此选 ANS）；可作为"无损 vs 有损"正交性论证的信息论依据。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices


（补充：Shannon 熵在 RNG/TRNG 评估中的用法——μRNG 论文把 Shannon 熵与 min-entropy 作为熵估计弱点测试）在随机数发生器评估语境下，Shannon 熵度量"无先验知识的盲猜攻击者"猜中下一输出的平均不确定性（H = −Σ p(xi)·log₂p(xi)，对所有 m 个样本求和）；而 min-entropy = −log₂(max_i p(xi)) 度量"拥有 RNG 历史先验知识的最强攻击者"猜中最可能输出的最坏情况不确定性。对输出 n-bit 完美均匀随机数发生器，两者都达理论最大值 n。µRNG 用二者量化 RNG 输出的"非均匀性"（TRNG 语境即"操作噪声的下降"）：在环境 corner（温度/电压）变化时观察熵的下降以暴露熵源退化。SRAM 熵源实测：名义条件 4KB 上电态每 bit 熵 0.149；-68°C 数据保持使熵崩至 0.004；+85°C 快速爬坡熵升至 0.108（但被布局偏置的 Moran's I 条带化掩盖）。RO 熵源实测（图 6）：8-bit 块熵随采样时间增大后饱和（jitter 随振荡波形多次穿越累积），低温+高电压熵最高，高温+低电压熵最低。
