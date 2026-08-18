## 冲突无关多 bank 哈希存储（Multi-Bank Hashed Memory / Conflict-Free Hashing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
针对聚类增长阶段高并发局部访存的定制存储：把 3D 格点 (x,y,z) 的顶点数据用线性同余哈希 b_v(x,y,z)=(αx+βy+γz) mod M（α=1、β=3、γ=5、M=22，22 个 bank）散列到多 bank；系数选取保证中心顶点与其 6 个轴对齐邻居、以及入射边的并发访问 pairwise 落在不同 bank——"冲突无关"由构造保证（conflict-free by construction），而非运行时检测/仲裁。需求来源：聚类引擎每周期需同时访问 (i) 中心顶点与轴对齐邻居（7 顶点邻域）与 (ii) 全部入射边。d≤15 单周期访问；更长码距有稍高周期开销（脚注）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
输入: 每周期 VID 坐标 (x,y,z)
bank = (1·x + 3·y + 5·z) mod 22          # 哈希 -> bank
bank 内地址 = 同 bank 三元组中该点的秩:
a_v(x,y,z) = Σ_{0≤i,j<L, 0≤k<R} [[(i+3j+5k) mod 22 == b_v]] · [[(i,j,k) ≺_lex (x,y,z)]]
# 同 bank 顶点按 (i,j,k) 字典序密排(ix 最外, 后 j, k)；边按"正向端点"坐标标识
# 7-顶点邻域 + 入射边并发读 -> 22 路无冲突并行读出 -> S4 判定
```
设计原则：让并发访问集合在 bank 维度正交（中心与邻居的哈希值 pairwise distinct），从源头消除增长阶段的 bank 冲突 stall（baseline 两大 stall 源之一）。消融：d=11、p=0.0015 时 Multi-bank Hashing 单独贡献 2.30× 加速（三项优化中最大），与 Hierarchical ID Mapping（1.03×）、Graph Compression（1.18×）协同合计 3.24×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：22 bank 的 BRAM 阵列（本文共 252 BRAM tiles 含此结构）+ 硬连线模运算（乘系数后 mod 22 可用查表/加法网络）；bank 内密排地址 = 同哈希秩计数（可预计算表驱动）。类似技术：多 bank/异或哈希是 cache/内存冲突规避的常见手段，本处针对"3D 格点 + 7 邻域 + 入射边"这一特定并发模式定制系数（通用构造：保证系数在邻域位移向量上产生的差非零 mod M）。使用场景：任何把规则格点邻域访问映射到片上多 bank 的加速器（图处理、格点 QCD、图像处理）。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
