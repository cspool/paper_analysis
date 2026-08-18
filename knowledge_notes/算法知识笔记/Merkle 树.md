## Merkle 树

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Merkle 树是哈希树：叶为数据块的哈希，内部节点为子节点哈希之哈希，根承诺整棵树。FRI 型 PCS（Plonky2）用 Merkle 树承诺多项式求值表，打开某点需给出叶子到根的兄弟路径哈希。Plonky2 中 Merkle 树是最大 kernel（占 prover 时间 68.84%），算术强度高（191 modmul/元素）但树结构导致并行度在近根处下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 构建/打开例子（N 个叶子，哈希 h）：
```
build: for level in 1..log2(N): 每对兄弟 h(左||右) → 父节点（并行）
open: 对叶子 i，输出路径 (i, 兄弟哈希序列) → verifier 逐层重算到根
```
- Annotations：GenZA 把树式 workload 分成子树 fit 片上，每层节点并行处理；打开路径的哈希是串行链（沿路径逐层），用向量 PE 并行处理多条路径。树的每层并行度按 2 递减，近根层是资源利用率瓶颈——GenZA 的混合时空映射（见硬件架构库）正好缓解此类低并行度场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Plonky2 用 Poseidon 哈希建树；硬件按"分片子树+层内并行"映射（GenZA Section VI-E）。使用：FRI 承诺与打开、区块链轻客户端验证等；内存/带宽友好型（逐层流式），对哈希吞吐要求高。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
