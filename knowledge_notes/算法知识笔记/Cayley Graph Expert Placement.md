## Cayley Graph Expert Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cayley Graph Expert Placement 是 FineMoE 用于构造 symmetric expert placement 的图论方法。Cayley graph 从群 A 及其生成集 S 构造：群中每个元素为 vertex，对每个 a∈A 和 s∈S，存在一条从 a 到 a·s 的边。FineEP 将 GPU 映射为 vertices，expert（edge 连接其 EDP group 中的 GPU）映射为 edges。利用 Cayley graph 的对称性保证 edges 在 vertices 间均匀分布，使得 graph 的 max induced subgraph density（Equation 3 中的最大化项）最小化。这从理论上保证：在未知 load 分布时，任何 GPU 子集的 expert 集中度都有上界。

从算法pipeline角度拆解术语：
构造示例（p GPU 数 exponent, q expert/GPU exponent）：
- 8 GPU, 8 experts (p=3, q=1): group (Z_8, +), generators {1,-1} → cycle graph。每个 vertex 度=2。
- 16 GPU, 32 experts (p=4, q=2): group (Z_4×Z_4, +), generators {(0,1),(0,-1),(1,0),(-1,0)} → 4×4 toroidal grid。每个 vertex 度=4。
- 8 GPU, 16 experts (p=3, q=2): group (Z_2×Z_4, +), generators {(0,1),(0,-1),(1,1),(1,-1)} → isomorphic to K_{4,4}。满足性质: ∀i, 所有 i-vertex induced subgraph 的最大 edge count 最小。
- 8 GPU, 32 experts (p=3, q=3): 先构造 complete graph（28 edges），剩余 4 edges 补为额外配对。

术语一般如何实现？如何使用？
- 论文假设 GPU 数和 expert/GPU 数为 2 的幂（实际常见于数据中心配置）。
- Cayley graph placement 生成在 Placement Manager 中完成，训练开始时 broadcast 到所有 GPU。
- 需要满足同步一致性约束：同一 expert 的所有 replica 必须具有相同 local expert index（Appendix B.3）。
- 仅用于 d=2 场景（hypergraph 退化为常规 graph）；d>2 尚未系统研究。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
