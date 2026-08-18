## TDS（Topology Dependency Subgraph，拓扑依赖子图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TDS 是 TAGT（ISCA 2026）为图 Transformer 提出的拓扑感知稀疏子图：把全局 O(N²) 注意力近似为在 TDS 上的注意力，将每个目标顶点 attend 的边数从 O(N) 降到平均 O(m·log_m N)（m=2 时为 O(log N)），总边数 O(m·N·log_m N)=O(N log N)。TDS 由三类边构成：(1) original edges——保留输入图原始局部邻域；(2) fusion edges——自底向上的分层聚合边：沿原生 1D 输入顺序每次递归合并 m 个内存连续顶点为 fusion 顶点（约 log_m N 层直到单根），fusion 顶点持有全部子顶点的聚合特征，作为"高阶代理顶点"保留远程/全局上下文；(3) association edges——指向目标顶点的递归边：每层从目标顶点左右各取 m 个关联顶点（起始下标 p_{l+1}=parent(p_l+m)，若 p_l+m-1 为奇数则再纳入下一顶点且 p_{l+1}=parent(p_l+m+1) 保证集合互斥），提供多粒度远程上下文。TDS 总顶点数约 N + N/(m-1)（m≥2），构造保证任意两个原始顶点通过 fusion+association 边最多 2-hop 可达，因此每个目标顶点的 1-hop 注意力邻域同时含局部邻居、多粒度上下文与全局根——一次稀疏注意力即等效多跳 message passing 的全局效果。
- 论文理论分析（式2-4）：设 A 与 Â 为全/稀疏注意力矩阵，‖Δh_i‖₂ ≤ L‖V‖₂·Σ_{j∉T_i(m)} α_ij + ε_fus(m)，其中 ε_fus(m) 为融合粗粒度误差（随 m 单调不减）；在注意力量重尾衰减 α_{i,(k)} ≤ c·k^{-β}（β>1）假设下尾质量 ≤ O((m·log_m N)^{1-β})。m 控制保真度-效率权衡：m=N 时退化为精确 O(N²) 全局注意力，m=2 时准确率最优且复杂度 O(N log N)。
- 与既有稀疏 GT 方法的区别（论文 Related Work）：AnchorGT/ANS-GT 等用采样/锚点启发式近似长程依赖，拓扑敏感、需按数据集重调；TDS 是确定性、无参数、硬件友好的稀疏化——通过拓扑感知合并而非概率选择保留全局上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TDS 构造与注意力的算法pipeline（m=2）：
  ```
  # 输入：顶点特征 x[0..N-1]（1D 内存顺序），合并基数 m，原始边集 E_orig
  # 阶段1：自底向上分层聚合——生成 fusion 顶点特征
  cur = list(range(N))
  while len(cur) > 1:
      nxt = []
      for i in range(0, len(cur), m):
          fus = 新建 fusion 顶点
          feature(fus) = aggregate(feature(c) for c in cur[i:i+m])   # 如 mean/sum
          for c in cur[i:i+m]: 添加 fusion 边 c -> fus               # 底向上有向边
          nxt.append(fus)
      cur = nxt                                    # 约 log_m N 层到根
  # 阶段2：目标顶点 association 边（左右各取 m 个，递归到上层）
  for 目标顶点 v_k (1D 下标 k):
      p = k + 1                                    # 右侧起始（左侧用 k-1 递减对称）
      for l in range(log_m N):
          在层 l 取下标 p..p+m-1 的 m 个顶点，添加 association 边 v_k -> 它们
          if (p + m - 1) 为奇数:  p = parent(p + m + 1)              # 集合互斥
          else:                   p = parent(p + m)
  # 阶段3：目标顶点注意力（式1）——只在其 TDS 1-hop 邻域上做
  H^v = concat({h_u^l | u ∈ N_TDS(v)})              # K = O(m·log_m N) 个顶点
  h̄_v^{l+1} = softmax(h_v^l·W_Q·(H^v·W_K)^T / √d_K) · (H^v·W_V)
  h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}            # FFN + 残差
  ```
- 张量计算示例（N=4, m=2）：基层 v_0..v_3 → fusion u_0=agg(v_0,v_1)、u_1=agg(v_2,v_3) → 根 w_0=agg(u_0,u_1)。目标 v_1 的 1-hop 邻域 = 局部邻居 v_0（original）+ 祖先 u_0、w_0（fusion）+ 远程 u_1（association，覆盖全图上下文），仅对 ~4 个顶点算注意力分数而非全图 4 个顶点（更一般地 O(log N) vs O(N)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件侧：TAGT-S（修改 DGL v2.4.0 的软件实现，跑 A100）验证算法收益——比 TorchGT 快 1.8×–2.5×（TorchGT 依赖 Hamiltonian path、现实图不满足时回退 O(N²)），带宽利用 >60%，但软件 runtime overhead 占 69.8%–86.1% 执行时间，需硬件消除（TAGT 加速器，见 硬件架构 层 TDL/TCU 条目）。
- 硬件侧：TAGT 用 TDS-CSR Table 存 TDS 稀疏图结构，TDL/Topology Data Loader 取数、TCU（FUU+MOU）实时构造 TDS（去重共享 fusion 祖先），FAU 在 TDS 邻域上做流式注意力、SCU 做块级异步 softmax。
- 实验效果：准确率相对 DGL-CPU 全注意力参考下降 <1pp（GT 0.11–0.91pp、Graphormer 0.03–0.55pp、UGformer 0.22–0.84pp、EGformer 0.08–0.88pp），且高于 TorchGT；m=2 为准确率最优；对顶点排序鲁棒（random→METIS 排序准确率 65.08%–65.48% 稳定，local-only 断全局边则掉到 56.12%）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
