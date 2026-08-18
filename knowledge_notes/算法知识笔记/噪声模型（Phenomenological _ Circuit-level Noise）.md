## 噪声模型（Phenomenological / Circuit-level Noise）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QEC 模拟的三级噪声抽象（越来越真实）：(1) code-capacity——只假设数据 qubit 以概率 p 出错；(2) phenomenological——数据 qubit 错误 + syndrome 测量结果按概率翻错，但不模拟具体测量电路（本论文主分析用此模型，与 AFS、Predecoder 一致）；(3) circuit-level——症候提取电路每一处（门、idle、初始化、测量）都可能出错。web：现象学模型 MWPM 阈值 ~2%（新研究最高 ~6%），circuit-level 阈值 ~0.5–0.7%（新研究到 ~1.4%），因为电路级错误位置多出 ~5×；p=10^-2 高于 circuit-level 阈值，不是可工作点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
噪声模型决定 syndrome 的时空统计结构，进而决定压缩参数与策略（本论文）：
```
phenomenological:  数据错误 p + 测量翻转 p
  -> 非零 syndrome 模式干净：X/Z 对、Y cross、测量错误对（相邻轮）
circuit-level:     5p 测量噪声+2p reset / 2p 测量+1p reset（两配置）
  -> 中途出错产生虚假 opcode 0 -> 策略调整：
     "仅当所有相邻 syndrome 不活跃时才记录 opcode 0 预测"
  -> 空间+时间聚类仍去 34% index，加 RGE 达 2.1–3.1×（vs AFS）
```
VI-A 其余场景：非 IID qubit（Willow 检测概率分布）、错误率漂移 10×、multi-bit burst errors（16 ancilla 区域、约 3 亿轮一次）、leakage（经 LRC 后类 circuit-level 特征）——论文逐一论证 IcePack 在每种噪声特征下无损或可忽略损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Stim 的噪声参数化：after_clifford_depolarization（现象学数据错误）、before_measure_flip_probability / after_reset_flip_probability（测量/复位噪声，组合即 circuit-level）。使用时注意 p 上界：现象学可到 10^-2，circuit-level 需排除 10^-2（超阈值）。噪声模型选择影响压缩评估的结论：IcePack 在 circuit-level 下相对 AFS 仍有 1.9–3.1×，证明压缩方法对噪声模型鲁棒。

补充（Coset Ensemble Decoder 论文）：该文在两种模型下同时评估解码器精度以证明通用性——circuit-level depolarizing（p=0.002 固定，d∈{3..19}）与 biased/unbiased phenomenological（repetition code，d∈{5,7}、p∈[0.04,0.08]，bias η=0.5/1/10，X-biased 下 vanilla UF 落后 MWPM 6.2× LER）。biased 噪声改变 syndrome 各向异性，是该文验证"陪集集成解码填补 UF-MWPM 差距"（~94%）的载体。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
