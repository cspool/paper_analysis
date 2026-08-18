## Don't Surrender to Low QPS/$: Fast and Cost-Efficient ANNS with TridentANN

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是面向 SSD 上十亿级向量 ANNS 的混合索引（hybrid noise-clusters index），三步构建（Algorithm 1-3）：(1) 对采样数据分层 KMeans 生成初始质心，每个向量分配给最近质心作 member、top-2~m 近质心作 candidate（解决边界向量漏检，避免 SPANN 式 RNG checking）；(2) 用 KMeans 递归拆分超过 list size r 的超大簇，子簇直接继承父簇 candidate（减少重复加载）；(3) 滤除 member 数低于阈值 n 的小簇（其向量归为 noise），用 candidate 补齐簇列表并重校准质心。最终索引 = 内存 HNSW 质心图（约 15M-60M 质心）+ 内存 SPTAG-BKT noise 索引（约 100M 向量）+ SSD 上的簇列表。实验比较 DiskANN、SPANN、PipeANN、FusionANNS（REF 原文 + REP 乐观复现），指标为 QPS、latency、Recall@10、P99.9、QPS/$、QPS/watt；8 SSD 配置下 1.8-3.4× QPS、延迟降为 21-70%。
- 硬件平台是什么，配置是什么。
  AMD EPYC 7453 28 核 CPU；8×32 GiB DDR4（实际内存占用 <32 GiB）；8× 1-TiB Samsung 980 Pro PCIe-Gen4 NVMe SSD；2× NVIDIA A2000-6GB GPU。Ubuntu 22.04 + Linux 6.8 + CUDA 12.1 + GCC 11.4。10B 规模建索引另用 256 GB DRAM + NVIDIA A6000-48GB。
- 模型是什么。数据集和bench分别是什么。
  无神经网络模型（纯向量检索系统）。数据集：SIFT1B（128-dim uint8 图像描述子）、SPACEV1B（100-dim int8 文档向量）、GloVe（1.2M 向量、100-dim）、NYTimes（0.3M、256-dim，后两者研究数据分布偏斜）；距离度量 L2。Benchmark：BigANN（Recall@10=90% 为合格线），另测 recall 90-98%、top-1 到 top-1000。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文正文未给出 TRIDENTANN 自身代码的开源链接（联网检索仅在作者主页看到 Code 入口，具体仓库地址未能确认）；实现基于 Faiss 1.12（hierarchical KMeans）。算法 pipeline 伪代码：
  ```
  # Algorithm 1: 建簇（member/candidate）
  C_init, I = H-K(sampling(X), |sampling(X)|/r)   # 分层 KMeans 生成初始质心
  for x in X:
      [c1..cm] = KNN(x, C_init, m)                # m=3 (SIFT1B)
      I[c1] += x                                  # member
      E[c2..cm] += x                              # candidate（边界冗余）
  # Algorithm 2: 平衡簇
  while C_init != ∅:
      若 |I[c]| > r: 拆为 ceil(|I[c]|/r) 个子簇(KMeans)，子簇继承 E[c]
  # Algorithm 3: 分离 noise 并成列表
  for c: 若 |I[c]| > n 且候选足够: 从 E[c] 选离质心最近向量补齐列表至 r，重校准质心
  N = X - 所有簇列表并集（noise 向量）
  建内存 HNSW(质心 C) + SPTAG-BKT(N)，簇列表 R 落 SSD
  ```
  查询：CPU 查质心 HNSW 定位最近簇 → 簇列表经 GPU-SSD P2P 直通加载 → GPU cuBLAS 算距离 → CPU partial_sort 排序并合并 noise 结果（详见 kernel调度条目）。建索引：CPU-only 版 2-2.5 天、GPU 加速 KMeans 版 20-22 小时（对比 SPANN 4-5 天、DiskANN/PipeANN 2.5-3 天）；10B 规模 1 周内。
