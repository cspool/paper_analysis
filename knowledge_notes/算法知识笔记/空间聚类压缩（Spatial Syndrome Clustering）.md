## 空间聚类压缩（Spatial Syndrome Clustering）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把单个数据错误引起的多 syndrome 激活模式（spatial cluster）编码为"1 个索引 + 2-bit opcode"：水平对（X/Z 错误，opcode=1）、垂直对（X/Z 错误，opcode=2）、cross（Y 错误，opcode=3），孤立 syndrome 记 opcode=0。本质是复用层次化解码器（Clique、Predecoder）的局部模式规则，但只用于压缩而非解码决策——因此局部视野不会带来精度损失（解压在 300 K 无损完成）。优先级 cross > vertical > horizontal（按 index 减少率 75% > 50% = 50%，本论文 Table I）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SCU（硬件滑窗实现）对行主序位流逐位扫描：
for i in stream:                      # i 为当前 ancilla 索引
    w = 5-ancilla 滑窗(i 及其右/下/对角邻域)  # 2D 邻域映射到时间偏移
    if w 匹配 cross:      emit(i, OP=3); 清除 4 个匹配位
    elif w 匹配 vertical: emit(i, OP=2); 清除 2 个匹配位
    elif w 匹配 horizontal: emit(i, OP=1);清除 2 个匹配位
    else:                 emit(i, OP=0)   # 孤立：留给时间聚类
```
格点二维邻域映射为时间偏移：右侧 ancilla = 下一拍，下方 ancilla = 2d−1 拍之后（d 为码距）——这是 SCU 用移位寄存器/PTL 行缓冲做滑窗的数学基础。效果：仅数据错误时 index 减少 57–61%；加入测量错误后降到 32–35%（测量错误不形成单轮内多 syndrome 簇）。与 AFS 对比：AFS 不处理非零 syndrome，index reduction = 0.0。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
本论文以 PU 的 SCU 单元硬件实现：5-ancilla 搜索窗（行缓冲 + 固定偏移抽头）+ 组合逻辑真值表（Fig. 11b）输出 (opcode, valid)；命中后清除行缓冲中匹配位。边界处可能出现跨边界的假阳性（false positive），在 300 K 解压时无损反转，不需压缩端处理、也不增加编码位数。软件复现（artifact）中由 icepack.py 的空间聚类函数实现同一规则。扩展：换码型只需替换 SCU 匹配的局部模式集合（如 color code）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
