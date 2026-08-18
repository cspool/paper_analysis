## 错误症候与症候测量轮（Syndrome & Measurement Round）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Syndrome 是稳定子测量的输出模式，标记错误发生的位置但不直接给出错误类型；数据错误只在其发生轮产生非零 syndrome（后续轮保持一致），测量错误则在相邻两轮同位置产生成对的瞬态非零 syndrome——这是解码图上"横向边 = 数据错误、纵向边 = 测量错误"的来源（本论文 Fig. 3）。测量轮（measurement round）指完成一次全体稳定子测量的周期，超导系统典型间隔 ~1 μs，是 QEC 系统时序的基本单位：串行化上传 syndrome 必须在一个测量轮内完成，且从测量到解码到控制的反应时间不能超过 ~10 μs。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文的压缩 pipeline 以 syndrome 位流为输入，以"每轮 index 流"为输出，三类 syndrome 来源对应三类压缩规则：
```
输入: 每轮 syndrome 位图 B_t（ancilla 行主序索引 0..N-1）
输出: 每轮 (index, opcode) 流
# 数据错误: X/Z -> 水平/垂直对（2 个非零），Y -> cross（4 个非零）
#          -> 空间聚类：1 个 index + opcode∈{1,2,3}
# 测量错误: 孤立非零 syndrome，连续两轮同位置复现
#          -> 时间聚类：预测 + 命中丢弃/失败补发
# 错误链:   链两端孤立非零 syndrome -> 时间聚类误预测时补发 index
```
测量错误远比错误链常见（这是时间聚类预测成立的统计前提）；每轮 syndrome 位流中零占绝大多数（p=10^-3 时非零占比 <0.1%），是全零块过滤（PPU）与稀疏编码的直接依据。时序约束：1 μs 测量轮内串行化上限 → IcePack 目标 500 ns（压缩后 300× 数据量减少才能满足）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上每个 ancilla 一轮一 bit（1 次测量 = 1 syndrome bit）；本论文假设数字读出（如 Josephson photomultiplier）下每轮每 ancilla 恰好 1 bit。使用时以轮为粒度流水：round t 的 syndrome 驱动压缩硬件，同时 TCU 用上一轮（round t−1）的预测流做预测对比并生成 round t+1 的预测（存于 PTL 环形延迟线，延迟 = 一个测量轮时长）。300 K 端解码器按轮序重建完整 syndrome 历史。

补充（Coset Ensemble Decoder 论文）：该文把 d 个 syndrome 测量轮 XOR 相邻轮输出形成 detector（detector events 为解码图顶点），并强调实时约束：超导平台上解码器需在 <1 μs（一次 syndrome 提取轮时长）内完成一个 d 轮任务；其系统指标把"解码延迟 R 折算为提取轮数 R=L/l"来量化反馈解码场景的保真度损耗。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
