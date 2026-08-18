## 层级（树形）一致性协议（Hierarchical / Tree-Based Coherence）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
层级一致性协议把机器组织成多层 cache/目录树：每层 cache 项内嵌目录，记录下一层哪些缓存持有该行；行随访问迁移到相应子树，本地性好的读共享不需上升到树根。Web 证据补充其公认缺陷：多级目录引入 indirection 延迟（HPCA Token Coherence 论文，https://scholar.archive.org/work/j6x6rhlvmfcu5i6cjxhbky5pny）、中间层 cache 兼做目录与叶缓存导致状态爆炸与协议竞态（ISCA 层级私有/共享分类论文，https://ieeexplore.ieee.org/document/7056032）、需形式化验证防竞态（J. Supercomputing 2013, https://dl.acm.org/doi/10.1007/s11227-012-0865-8）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文的论证（以 Hier2/Hier4/Hier16 为对照）：(1) 需大总缓存（层级越多、越靠根越大，论文给 Hier4/Hier2 加到 8/12MB per core 才可比）；(2) 写共享行必须上行到覆盖所有共享者的层 L，再向下发无效化、向上收 ACK、回响应，每层都查/改目录，多级遍历延迟长；(3) 消息路径与主流扁平协议不同——扁平协议消息直接"簇→home"、home 串行化；层级协议沿树串行化于各子树根，硬件非模块化（扩容要新增不同类别的目录/cache 层硬件，而非同构加簇）。实验结果：Hier2/Hier4/Hier16 平均 1.17/1.20/1.18×，Dorado 1.36×；跨簇读写共享多的 SocNet 中 Hier4 的 L1-miss load 平均 72 cycles、Hier16 76 cycles（根节点争用），Dorado 57 cycles；MapRed（簇内只读为主）两者接近（53 vs 49 cycles）。RC 一致性下 Dorado 对 Hier4 优势从 1.13× 降到 1.11×（写重叠后 Hier4 受益更多）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SCI 链表式、DASH 式多级目录、H-tree 簇协议（Kapoor 等，Murϕ 验证）；未成为主流商用形态——商用多核走扁平目录 + 簇（Intel CHA、AMD EPYC）。使用要点：适合"读写都在簇内/子树内"的负载；跨簇写共享负载是层级协议最坏情形；与 TLH 类方案（单目录类型、单协议、扁平消息流）的选择取决于硬件模块化与工作负载共享模式。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
