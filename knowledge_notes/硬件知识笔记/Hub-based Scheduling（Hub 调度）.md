## Hub-based Scheduling（Hub 调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hub-based Scheduling 是 L-PCN 在 Island 内部调度 intra-island 计算的方法，类比 Cache-guided scheduling（Mukkara 等 AGP'17/MICRO'18），通过动态缓存、更新、复用重叠点的运行时结果，使数据流在时间上呈现高局部性。流程：对每个 Island，先完整计算 Hub point subset（Island 中心、与其他 subset 重叠最多）并把 32 个结果缓存进 Hub Cache；其余 non-Hub subset 沿 Island List 自上而下（岛内由内向外）处理——对每个 subset 先用 Octree 搜索检出与 Hub Octree 的重叠点，重叠点复用缓存结果，只有非重叠点取内存并进 MLP，其新结果用 Tree-updating Method 更新进 Hub Octree 与 Hub Cache。相比 Mesorasi 的"precomputation-and-refetch"方案（需大缓冲存全部预计算数据、refetch 延迟无法隐藏、受限于内存带宽），Hub-based Scheduling 是运行时复用机制，可隐藏访存延迟、在 off-chip 设置下仍保持加速。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中 Hub-based Scheduling 的调度流程（Figure 11/14，K=32）：
```
for island in Islands:
    # 1) Hub point subset（Island List 首行）全量计算并缓存
    compute_all(HubSubset);  HubCache[0..31] = MLP_results(HubSubset)
    # 2) 其余 non-Hub subsets 沿 Island List 自上而下
    for subset in island (non-Hub):
        (K, idx) = OverlapDetect(subset, HubOctree)   # 双 OSE 并行搜索
        out[overlap]   = DeltaCompensate(HubCache[idx])   # 复用缓存+补偿
        out[non_overlap] = MLP(FetchFeatures(non_overlap)) # 仅非重叠点计算
        Pool(out);  HubCache[new] = out[non_overlap];  TreeUpdate(HubOctree, new)
    # 3) 下个 Island 时替换 Hub Cache（no-replacement within island）
```
  - 效果：岛内每 non-Hub subset 仅 (32−K) 个点进 MLP；论文测量相邻 subset 重叠可达 ~90%，故 MLP 输入量本质减少（feature computation 减 45.4%–80.6%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Overlap Detection Module（两个 OSE + Hub-Octree Buffer）+ Hub Cache 硬件；FCU 侧 Dataflow Controller 按 Case1（Hub subset）/Case2（non-Hub subset）两条数据流调度 16×16 脉动阵列。论文未提供开源实现。一般参考：Cache-guided scheduling（Mukkara 等）、I-GCN 的 island 内调度。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
