## Union-Find 解码器（UF Decoder）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Union-Find 解码器（Delfosse & Nickerson, Quantum 5:595, 2021）是 MWPM 的近似替代，几乎线性时间：① 聚类/增长阶段——每个非平凡 syndrome 初始化一个簇，奇数奇偶的簇半径增长并与相邻簇融合，直到所有簇均为偶数奇偶（并查集 disjoint-set 数据结构 + 加权合并 + 路径压缩，O(N·α(N))，α 为 Ackermann 反函数；Phys. Rev. Research 6:013154 (2024) 进一步证明规模上可线性）；② peeling 阶段——簇内生成生成树，从叶到根逐层剥除（吸收 syndrome），O(N)。精度低于 MWPM，但并行度高、易硬件化（顶点/簇可映射到分布式 PE）。本文将其定位为"低延迟但精度 suboptimal"的 baseline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
UF(G, s):
  for v in V: if s[v]==-1: 新建簇 C(v)      # 非平凡 syndrome
  while 存在奇数奇偶簇:
      增长所有奇数簇半径; 相邻簇相遇则合并   # union-find
  for 每个簇: T=生成树(cluster);  peeling(T)  # 叶到根剥除
```
本论文用法：① 陪集集成解码的 Phase I Clustering 是 UF-equivalent 聚类（把 stabilizer 群划分为局部子空间）；② 精度 baseline 用自研 UF 软件实现（避免边界条件处理差异混淆），硬件 baseline 为 Helios（QCE 2023，d=17 889k LUT @75 MHz，per-vertex PE 空间并行、sublinear 延迟但有 per-iteration 地板）与 QUEKUF（TRETS 2025，d=8 309k LUT / 548 BRAM @238 MHz）。对比结果：UF 类 LER 落后 MWPM 2.7–5.7×（biased X 噪声下 6.2×），本文方法填补该差距（repetition code 上距 MWPM 仅 1.0–1.4×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：Qsurface（Python，Delfosse 算法实现）、AFS（Das et al., HPCA 2022：UF + 3 级流水硬件 + syndrome 压缩，平均 42 ns）。硬件：Liyanage et al. TQE 2024（FPGA 分布式 UF）、QUEKUF、Helios。使用场景：需要亚微秒实时解码的超导平台；精度敏感场景可叠加本文的陪集集成/投票增强。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
