## 时间聚类压缩（Temporal Syndrome Clustering / Prediction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对空间聚类后的孤立 syndrome（opcode 0）做时间维压缩：由于测量错误远多于错误链，opcode 0 在下一轮同位置大概率复现（测量错误成对出现），因此对每个孤立 syndrome 预测"下一轮同位置也是 0"并只发 1 个 index；若预测失败（下一轮该位置无 syndrome），补发一个 (index, opcode=0) 条目，300 K 端"未收到 index 即默认该位置有 syndrome"的约定使该补发条目表示"此处无 syndrome"——无损。与层次化解码器（Clique/Predecoder）"等下一轮才能处理测量错误"不同，本方法不增加任何测量轮延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TCU 真值表（本论文 Fig. 11c），P_in = 上一轮同位置预测位：
if P_in == 0:
    if V_in == 1 and OP_in == 0:      # 本轮出现孤立 syndrome
        P_o = 1;  V_o = 1; OP_o = 0   # 发 index 并预测下一轮复现
elif P_in == 1:                       # 上轮预测这里会复现
    if V_in == 1 and OP_in == 0:      # 预测命中（测量错误对）
        V_o = 0                       # 丢弃 index，不发
    else:                             # 预测失败（错误链等）
        V_o = 1; OP_o = 0             # 补发 index，表示"此处无 syndrome"
# 多 syndrome 簇 OP_in ∈ [1,3] 不参与预测（防丢数据）
```
预测流按 index 对齐、延迟一个测量轮循环存储。效果：index 减少从纯空间聚类的 32–35% 提升到 41–55%（p=0.01%–0.1% 时 1.6× 提升），p=1% 时仅 1.3×（错误链多、误预测多）。circuit-level 噪声下调整策略：仅当所有相邻 syndrome 均不活跃时才记录 opcode 0 预测，避免误预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现（TCU）：SCU 输出流与预测流 P_in（来自 PTL 环形延迟结构，延迟 = 1 测量轮）按位对比，更新 (V_o, OP_o) 并生成 P_o；运行 index 由共享计数器采样，有效条目（index, opcode）写入队列。300 K 解压端用升序 FIFO 存预测，算术比较校验。适用前提：测量错误发生率高于错误链，该条件在可行工作区间内始终成立（本论文 VI-A3 论证）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
