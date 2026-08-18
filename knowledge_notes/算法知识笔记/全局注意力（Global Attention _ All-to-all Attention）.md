## 全局注意力（Global Attention / All-to-all Attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 全局注意力指图 Transformer 采用的全对（all-to-all）注意力范式：每个顶点与图中所有其他顶点（含自环）计算注意力分数，邻居集 N(v) = 全图顶点。相比局部 message-passing，它能捕获长程依赖，但产生 O(N²) 计算与 O(N²) 中间注意力矩阵。论文量化（Fig.2/3）：attention+FFN 占 91.61% 执行时间；off-chip 访问占 60.5% 执行时间、SM 利用率 <25%；TorchGT 上 60.3% 取回数据不必要、cache line 利用率仅 18.27%；N=256K 时中间矩阵无法片上缓存、强制 off-chip spilling。大特征矩阵 O(N·D_feat + Nd) 与中间数据双重挤压内存带宽。
- 现有缓解的局限：FlashAttention 类 IO-aware 注意力 kernel 通过 tiling+online softmax 降低稠密注意力的显存流量，但**不消除 O(N²) 顶点对交互次数**；且 GT 的图结构编码使注意力矩阵不规则（非文本 token 的规则模式），块矩阵优化难以直接套用。因此全局注意力仍是 GT 扩展性的根本瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 全对注意力 pipeline（式1，全连接图 G 上）：对每个目标顶点 v，H^v = concat({h_u | u ∈ 全图})（N 个顶点），计算 h_v W_Q · (H^v W_K)^T 得 1×N 分数向量，softmax 后加权 H^v W_V。总体即 N×N 的 QK^T + softmax + PV。复杂度和中间数据均为 O(N²)。
- 对比示例（N=16K 序列）：DGL-CPU 保留 O(N²) 全对注意力作精度参考（Table VI 准确率 65.98%/98.02% 等）；TorchGT 的 dual-interleaved 注意力在缺少 Hamiltonian path 时回退 O(N²)；TAGT/TDS 把每个顶点 attend 的顶点数从 N=16K 降到 O(m·log_m N)=O(log 16K)≈28（m=2），精度下降 <1pp。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：DGL/TorchGT 直接实现；FlashAttention 类 kernel 用 tiling+online softmax 减 IO 但不减交互数。论文通过把全对注意力替换为 TDS 稀疏注意力（original/fusion/association 三类边）消除大部分不必要的注意力计算与 off-chip 移动，同时以"任意两点 ≤2-hop 可达"的拓扑保证保留全局建模能力（多粒度上下文+根顶点直接进入 1-hop 邻域）。
- 硬件：TAGT 的 FAU 把注意力分数流式送 SCU 做块级异步 softmax，全程不物化 N×N 注意力矩阵；GNN 加速器（FlowGNN/MEGA/BingoGCN）因专为稀疏 message-passing 设计，被改造执行 O(N²) 全对注意力时性能远逊于 TAGT（TAGT 平均快 8.2×/6.9×/4.7×）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
