## Bitonic Sort（双调排序 / 双调排序网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bitonic sort 是一种并行排序算法/排序网络（Batcher 1968；优化见 Ionescu & Schauser, IPPS 1997）：把输入序列视为双调序列（bitonic sequence，先单调增后单调减），通过反复做 compare-exchange 归并为有序序列，总复杂度 O(N log²N)。它的核心优势是无数据依赖的比较结构固定（比较器网络），非常适合硬件实现为固定拓扑的 sorting network（比较器级联，流水/并行执行），因此被 3DGS 硬件加速器（GSCore[25] 及其层级排序）用作 tile 内 Gaussian 按深度排序的排序引擎；32 并行 bitonic 网络是硬件消融中的 baseline 排序配置。本论文用其作为"排序硬件"的代表来论证排序的痛点：硬件面积随输入并行度 k 按 k·log²k 增长、复杂度 O(N log²N) 与光栅化 O(N) 异构导致 pipeline 失衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
对 8 个深度值 d0..d7 的双调归并（bitonic merge，升序）：
```
# 输入为双调序列（先增后减），目标升序；距离从 N/2 递减到 1
for (k = N/2; k >= 1; k /= 2):          # N=8 → k=4,2,1
    for (j = k; j >= 1; j /= 2):
        for i in [0, N):
            l = i XOR j                  # 比较对索引（XOR 位翻转）
            if (l > i):
                # 按方向决定升降：取 (k&i)==0 时升序、否则降序
                if ((i & k) == 0 and d[i] > d[l]): swap(d[i], d[l])
                if ((i & k) != 0 and d[i] < d[l]): swap(d[i], d[l])
```
每轮 j 下所有比较对可并行执行（网络的一级）；硬件实现为 k·log²k 个比较器节点，k=32 并行时固定 32 输入并行度——tile 内 Gaussian 数（80~10000+）与固定并行度失配：小负载利用不足、大负载需多轮（层级排序）或成为瓶颈。本论文硬件消融：BS 变体用 32 并行 bitonic 网络[20]（按 GSCore 风格）+ 层级排序；与 MLP-OIT（替代排序）对比：21.1~32.4× 加速，且消除 pipeline 失衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：比较器（compare-exchange）单元级联成网络，支持完全流水/并行；GPU 实现：CUB/thrust 等库有 device bitonic sort kernel（warp 内共享内存 + 全局归并）。3DGS 场景（本论文 II-B2 章）：每 tile 对相交 Gaussian 按深度升序排序后再 α-blending；tile 负载 80~10000+ 波动两个数量级（MipNeRF-360 profiling），固定并行度模块要么小负载闲置、要么大负载成为光栅化瓶颈（Fig.4 调度图）——这是本论文用 MLP-OIT 完全替换排序的动机。

DMSU/Bitonic-16 补充视角（ParetoES，ISCA'26）：Bitonic 网络在检索加速器中被用作核内局部排序器——ParetoES 用 32 个核内 Bitonic-16 替换单体 Bitonic-512 全局排序器（Distributed Micro-Sorting Unit，DMSU），比较器从 11,520 降到 2,560、流水 45 级降到确定性 10 级，32 排序器完全并行、每 ACPE 各持一个；每核做两阶段：质心分数筛选（簇探测选 Top-nprobe）与簇内 Top-16 排序（索引与分数按 score 联合排序），32 核 Top-16 在 host 聚合为 Top-512 候选超集。32×Top-16 分解的精度代价：K≤200 时 Recall 恒 100%，仅 K>200 尾部略有偏差（Table III）。对比：3DGS 场景用 32 并行 bitonic 排 tile 内 Gaussian（负载 80~10000+ 波动导致失衡），ParetoES 用 16 输入微排序 + 分解规避单体网络资源爆炸（N=512 需 11,520 比较器/45 级，耗尽 FPGA）。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
