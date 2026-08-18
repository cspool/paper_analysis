## Weight Nulling 与 VAPI（基于权重值分布的容错 ECC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
两条利用神经网络权重分布特性的轻量容错 baseline 路线（RangeGuard 论文 §IV-B 对比对象）：(1) Weight Nulling（Qin et al. arXiv:1709.06173）——每个权重借用最低有效位（LSB）放 1 bit parity，读到 parity 不匹配时把损坏权重直接置零而非纠错；缺点：parity 只检奇数位错、对大幅值权重置零会丢失重要信息。(2) VAPI（Value-Aware Parity Insertion，Lee & Yang DATE 2022）——针对 8-bit 量化 CNN 权重：观察多数权重靠近零，采用 sign-magnitude 表示使高序位（如 b6、b5）很少使用，把这些"不重要位"覆写为 parity；用 DEC(64,50) 码每 64-bit 权重块纠 2 个位错、无需重训练；缺点：只针对 8-bit 量化权重的特定值分布，只保护存储的权重。共同点：仍在保护原始数据比特，只是按值分布"挑选值得保护的位"，因此对 FP 类 DNN 的 exponent 位故障（位数不固定、位置随值分布变化）覆盖不足。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VAPI 执行流程（8-bit 权重）：① 离线统计权重分布，确定哪些高序位几乎恒为零 → 标记为 parity 位；② 存储时把 LSB/低价值位替换为 DEC(64,50) 编码生成的 parity 覆盖 64-bit 块；③ 读时解码，纠 ≤2 bit 错、否则 DUE；④ 纠错后恢复原值。Weight Nulling 执行流程：① 每权重存 1 bit LSB parity（奇偶）；② 读时校验，奇数个 bit 错 → 检错 → 该权重置 0（不纠值）；③ 偶数个 bit 错漏检（SDC）。RangeGuard 的对比结果（Table III）：两种方案对 32E（32-bit 簇错）几乎无纠正能力（Weight Nulling 32E SDC≈75%、VAPI 32E DUE≈99.994%），而 RangeGuard 4b DSC 对单/双故障场景 CE/BE 覆盖显著更强——证明"按位挑选保护"不如"按范围语义保护"。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：都是内存写入/读取路径上的轻量编解码逻辑（parity 位与权重位复用的 bit 重排 + 线性码编码/解码），无需重训练、无需模型架构改动，适合量化模型权重存储保护。使用场景与限制：VAPI 只适用于值分布稳定的 8-bit 量化权重（高序位确实冗余），Weight Nulling 适用于奇偶错误为主的场景但无纠错能力；两者都难以覆盖 HBM 的 SWL/SWD 类 16–32-bit 簇错与 FP 模型的 exponent 灾难性错误——这是 RangeGuard 论文将其作为"bit-centric"代表进行对比的原因。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
