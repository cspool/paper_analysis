## Feature Computation（FC，特征计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Feature Computation（FC）是 point-based 点云网络（PCN）Building Block 中的第二步：对 DS 形成的 point subsets（输入特征图）执行 MLP 卷积层与 Pooling 层，把每个 subset 的 K 个点的特征聚合到中心点。FC 与传统 DNN 的卷积/全连接层功能相同，可直接由商用 DLA（NPU）或脉动阵列加速。例如 PointNet++ 中一个 subset 的特征维度在 MLP 中从 (32,6) 变成 (32,128)，最后 max pooling 把 32 点聚合为中心点的 128 维特征。论文指出：当 DS 被专用单元加速后，FC 成为当前 PCN 加速器的主要延迟来源（占比可超 85%），而 FC 中 MLP 占 FC 计算量的 98%+——因此削减 MLP 输入量是加速关键。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中 FC 的 pipeline 伪代码（K=32，MLP 6→128）：
```
# FC pipeline（Building Block 第二步）
Input: 输入特征图 Fmap[c] = (32, d_in)
for c in C:
    # 每个 subset 的 32 个点依次过共享权重 MLP
    H[c] = MLP(Fmap[c])          # (32,6) -> (32,128)，同一权重作用于每点
    Out[c] = MaxPool(H[c])       # 沿 32 点维池化 -> (1,128)，聚合到中心点
# 冗余：相邻 subset 共享的重叠点重复进入 MLP（如 subset A 与 G 共享的 D,E,F）
#   -> 重复的 MLP 计算量 = 重叠点数/32 * 总计算量（论文测量可达 ~90%）
```
  - L-PCN 的优化：Islandization Unit 阻止重叠点重复进入 FCU 的 MLP，只对非重叠点计算，重叠点直接复用 Hub Cache 中的 MLP 结果（经结果增量补偿），从而在算法层面把 MLP 输入量本质减少。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU 上用 batch GEMM 把 K 点 × MLP 权重批量矩阵乘，再 max-pool；加速器上用脉动阵列（如 L-PCN 的 16×16 systolic array FCU）或商用 NPU 执行。L-PCN 的 FCU 由现有 AI 加速器 + Dataflow Controller 组成，Dataflow Controller 区分两条数据流：Hub point subset 全量计算并缓存结果，non-Hub point subset 只把非重叠点送入 MLP、重叠点从 Hub Cache 取缓存结果直接进 Pooling。论文未明确说明 FCU 的软件框架；参考实现为 PointNet++ 官方 PyTorch 代码的 shared MLP 层。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
