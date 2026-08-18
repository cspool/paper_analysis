## Result Delta Compensation（结果增量补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Result Delta Compensation 是 L-PCN 使"跨 point subset 复用缓存 MLP 结果"成为可能的关键算法技术。现代 PCN（如 PointNet++）在 MLP 前把非中心点的 XYZ 坐标按中心点归一化（减去中心点坐标），因此两个不同中心点子集的共享点（如 subset A 与 G 共享的 D,E,F）在各自子集中的输入不同，缓存的 MLP 结果不能直接复用。L-PCN 利用 MLP 的线性部分：w·(P−P_G) = w·(P−P_A) + w·Δ(A−G)，即用缓存的 w·(P−P_A) 加上由中心点差 Δ(A−G) 计算的增量 w·Δ(A−G) 补偿出实际需要的值。由于 MLP 含非线性激活，MLP(A−B) ≈ MLP(A)−MLP(B) 仅近似成立（Mesorasi 可因此损失至多 0.9% 精度）；L-PCN 只对重叠点做补偿（选择性近似），对非重叠点保持精确计算，因此精度损失更小；当激活只在 Building Block 末尾应用时（DGCNN(c)、PointVector-L），CONV(A−B)=CONV(A)−CONV(B) 严格成立，可完全补偿、零精度损失。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中的补偿公式与流程（Figure 8 示例，N=3 重叠点）：
```
# 处理 Point-subset G 时，复用 subset A 缓存的 N 个重叠点结果
for d in overlapping(D,E,F):
    cached = HubCache[d]                      # w·(P_d - P_A) 已缓存
    w_delta = MLPlinear(w, Delta(A-G))        # 增量 w·Δ(A-G)（送入 FCU 计算）
    result[d] = cached + w_delta              # Eq.1 补偿后的实际复用值
# Eq.1:  w·(P-P_G) = w·((P-P_A) + Δ(A-G)) = w·(P-P_A) + w·Δ(A-G)
# 补偿带来一次性额外计算开销 -> feature computation 节省略低于访存节省
```
  - 与 Mesorasi 的 Delayed-Aggregation 对比：Mesorasi 对全部 MLP 结果做近似（MLP(A−B)≈MLP(A)−MLP(B)），L-PCN 只对重叠点近似——非重叠点（多为点云边界点，max pooling 常保留其高值）精确计算，故精度更优、跨域鲁棒性更好。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：L-PCN 中把 Δ(A−G) 输入 Feature Computation Unit 计算 w·Δ(A−G)，与 Hub Cache 取出的缓存结果相加完成补偿；这是 Islandization Unit 内 Hub Cache 读路径上的附加加法器逻辑。论文未提供补偿单元的独立 RTL；GDPCA 的 Geometry-aware Differential Update 与 Mesorasi 的 Delayed-Aggregation 是相关先例（GDPCA 用 Bit-Pragmatic 加速器 PRA 利用低位宽差分输入）。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
