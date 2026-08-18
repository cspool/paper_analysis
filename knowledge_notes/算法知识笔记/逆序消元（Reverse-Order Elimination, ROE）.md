## 逆序消元（Reverse-Order Elimination, ROE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROE 是 peeling（剥除）的流水友好变体：传统 UF/MWPM 后处理需要反复全局寻找叶节点并重算度数；ROE 则利用建森林阶段的遍历结果——BFS 建森林时已按根→叶访问一遍，记录发现序 σ 后，逆序弹出顶点：对奇偶 p[x]=1 的顶点收集边 (x, parent[x]) 并翻转 x 与其父的奇偶。单趟、线性时间，免去第二次叶子发现遍历与度数重算，直接把解码延迟砍掉一趟全图扫描。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Require: parent[]; 发现序 σ; 奇偶 s            # Algorithm 3
E_i ← ∅; p ← s
for t = |σ| down to 1:
    x ← σ_t; r ← parent[x]
    if r ≠ NIL and p[x] == 1:
        E_i ← E_i ∪ {(x, r)}
        p[x] ← p[x] ⊕ 1; p[r] ← p[r] ⊕ 1   # 奇偶吸收到父
L_i = DECODELOGICAL(E_i)
```
关键观察：Algorithm 2（PriorityForests）在建森林时已经完成根→叶遍历，ROE 复用该顺序逆序剥除，等价于传统生成树 peeling 但省一趟。在陪集集成解码中，K 个候选各自执行一次 ROE，故其单趟性与无全局探测特性直接放大为 K 倍收益；硬件上 EFE 实例内的遍历状态不可时分复用（会覆盖在途邻接数据），故复制 K 份并行执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现随开源仓库发布（Python）；硬件在每个 EFE 实例中以状态机执行（邻接表构建与聚类重叠进行）。使用方式：任何"先建森林/树、后剥除"的解码流程都可替换为 ROE；前提是建树阶段能顺带记录发现序（BFS/DFS 均满足）。本论文配套的 lossless graph compression 进一步缩小 σ 与 parent 的规模。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
