## Farthest Point Sampling（FPS，最远点采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Farthest Point Sampling（FPS）是点云下采样的经典算法：迭代地从未选点中选出"到已选集最远距离最大"的点加入选集，直到选够目标数量。相比随机采样，FPS 保证采样点在空间上均匀覆盖整个点云（避免只采到局部区域）。PointNet++ 的 Sampling Layer 用 FPS 从 N 个点选 N' 个中心点（如 1024→512）。FPS 也是 L-PCN 论文指出的一个 PCN 特有挑战的来源：由于 FPS 倾向选相距最远的点作为中心点，标准 PCN 流程中相邻迭代处理的 point subsets 往往空间相距很远（破坏空间相邻执行顺序），这使"相邻子集重叠"在原始执行顺序下不可见，需要 Octree-based Islandization 重排处理粒度才能利用空间局部性。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - FPS 伪代码（PointNet++ Sampling Layer）：
```
def FPS(X, M):                      # X: N 点云, M: 目标中心点数
    C = [random_point(X)]           # 随机起点
    dist = ||p - C[0]||_2 for p in X
    while len(C) < M:
        c = argmax_p(dist[p])       # 到已选集最远距离最大的点
        C.append(c)
        dist[p] = min(dist[p], ||p - c||_2)   # 更新最近距离
    return C                        # M 个均匀覆盖的中心点
```
  - L-PCN 中 FPS 由 DSU 的 Sampling Module 执行，输出 Sampled Point Cloud（中心点集）；后续 Octree-based Islandization 基于这些中心点选 Hub points 并做邻接聚类。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：PyTorch 的 farthest_point_sample 算子（GPU 并行版），复杂度 O(M·N)；加速器用硬件采样模块。L-PCN 论文未给出 FPS 的自定义硬件实现细节（沿用现有 PCN 采样方法），强调 FPS 与岛化/调度互补。参考开源：PointNet++ 官方仓库 https://github.com/charlesq34/pointnet2。
  - **NS-FPS 补充（ISCA'26，硬件-软件协同设计）**——FPS 是内存受限而非计算受限：RTX 3090 上 120k 点 25% 采样需 >900ms、95% 时间耗在内存事务，profile 显示 600M 次请求、164.65GB L1 + 74.81GB L2 请求，缓存吞吐 95.62% vs SM 指令吞吐 27.17%；FPS 在 16k 输入的点云网络中占 30–70% 总运行时间。维护 N 长距离缓存 T（Eq.2 递推 t_p^m=min(t_p^{m-1},d(p,s_m))）把每轮复杂度降到 O(N)，但仍有 O(MN) 全量扫描。NS-FPS 把 FPS 重述为邻居搜索：用 Voronoi 图证明距离更新只发生在以最新采样点 s_k 为球心、半径 d_k=min_{s_i∈S_{k-1}}||s_k−s_i||² 的球内（该球严格包含真实 Voronoi cell），每轮更新点数从 O(N) 降到 O(N/k)，总复杂度 O(N log N)，且采样结果与传统 FPS 完全一致（lossless）。CPU 版在 16k–120k 点相对 vanilla FPS 加速 100.1×/130.3×/106.2×/191.7×，相对 QuickFPS-CPU 在 64k/120k 上 1.22×/1.80×；ASIC 版相对 GPU 加速 17.2×–81.6×、内存访问降 1700×。
  - NS-FPS 算法 pipeline 伪代码（Algorithm 1）：初始化：Morton 码桶排序建索引、T←∞、随机选 s_0；每轮 k：(a) d_k=T[s_{k−1}]；(b) 枚举与球 B(s_{k−1},d_k) 相交的 Morton cube，对 cube 内点 p 更新 T[p]=min(T[p],||p−s_{k−1}||²)；(c) 对受更新块刷新 16:1 层次 max 缓存；(d) s_k=argmax_p T[p]，加入采样集。搜索半径自适应收缩是剪枝关键：120k 帧前 100 轮半径覆盖很多 cube、占 27.3% 总迭代时间，随后半径快速降到 1m 以下。
  - NS-FPS CPU 实现开源：https://github.com/satreeby/ns-fps/（C++17 + pybind11，`pip install -e .`，`yf.fps(points, n_samples=..., range=SpaceRange)` 返回采样索引；README 声称 CPU 上较 naive FPS 最高 191×、较 QuickFPS-CPU 1.72×、较 naive GPU FPS 4.2×）。GPU 侧 baseline 为 OpenPCDet 的 CUDA FPS 实现，profiling 用 NVIDIA Nsight Systems/Compute。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
