## 稀疏矩阵乘法（Sparse Matrix Multiplication，SpGEMM / SpMSpM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
稀疏矩阵乘法 C=A×B 是只对两个稀疏矩阵的非零元素做乘累加的线性代数运算（SpGEMM 泛指任意稀疏×稀疏，SpMSpM 特指 sparse-matrix × sparse-matrix，SpMM 为 sparse × dense）。它是图分析（三角计数、PageRank、图神经网络）、科学计算（有限元、电路仿真）、稀疏 CNN 与剪枝 LLM 推理的核心 kernel。相比稠密 GEMM，稀疏乘法的难点是：(1) 数据访问不规则——非零位置由矩阵结构决定，无法按稠密网格预测；(2) 数据复用低、计算强度低（M/K/N 循环中大量迭代无有效乘法）；(3) 负载不均衡——不同行/列的非零数差异大；(4) 输出也不确定——C 的稀疏结构在计算前未知，需要合并（merge）各中间部分和。Web sources（Gamma ASPLOS'21、Flexagon、SpecBoost）确认内积/外积/Gustavson 三类数据流以不同方式权衡这些难点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
算法级 SpGEMM 的执行骨架（外积视角，HiT 采用）：
```
输入: A (MxK, 稀疏), B (KxN, 稀疏), 输出 C (MxN, 稀疏)
# 外积：对每个 k，A 的第 k 列 x B 的第 k 行 生成一个 rank-1 片
C = {}
for k in nonzeros(K):
    for (m, a) in A.col[k]:        # A 列 k 的每个非零 (行 m, 值 a)
        for (n, b) in B.row[k]:    # B 行 k 的每个非零 (列 n, 值 b)
            C[m][n] += a * b       # 累加进输出位置 (m,n) —— 跨多个 k 的合并
# 输出合并：同一 (m,n) 被多个 k 贡献，需按 (m,n) 聚合
```
关键计算特性：每个 (m,n) 输出需要"列-行索引匹配"（A 非零的列索引 == B 非零的行索引，即 k 相同）才产生有效乘法；交叠率（有效匹配/总比较数）在高度稀疏时极低（HiT 实测 HS×HS geomean 仅 0.12%），决定 MAC 利用率。HiT 的算法变体：按列索引再行索引连续排列的 COO-like 片上格式使非零按外积顺序流式读取；HS×HS 输出同样稀疏，用压缩格式累积 psum（见 DMAccum）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用分三层：(1) 软件——SuiteSparse 等库提供 SpGEMM，支持 CSR/CSC/COO 格式与并行调度；HiT 用 27 个 SuiteSparse 真实矩阵（p2p-Gnutella24 密度 9.3e-5、cage12 1.2e-4、poisson3Da 1.9e-3、opt1 8.1e-3 等覆盖 1e-5~1e-2 密度段）做 HS workload，HS×HS 计算 M×M^T；(2) 硬件——专用稀疏加速器（HiT、Trapezoid、SpArch、OuterSPACE、Sigma、Flexagon、Spada）用专用数据流/相交单元/psum 合并网络加速，避免对零做无效 MAC；(3) 数据生成——MS/D 段用剪枝 DNN（ResNet50/VGG16 40% 密度非结构化稀疏 + im2col 转矩阵乘、Llama2-7B 幅值剪枝 0.2/0.4/0.6）构造。HiT 的意义：首次用统一架构把 HS/MS/D 三段都跑到高吞吐（全谱 performance/area 比 Trapezoid 高 1.93×）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication

SegFold 补充视角（ISCA'26，动态数据流下的 SpGEMM）：SegFold 针对双端稀疏（dual-side sparsity，A 与 B 都稀疏）的 SpGEMM，核心是把 SpGEMM 数据流从"静态循环序"扩展为"细粒度动态"：
- 静态数据流的公共缺陷：inner product 只复用 C（行-列点积交点数随非零位置变化）、outer product 复用 A/B 但每次迭代生成整个 T_{M,N,k} 部分和矩阵（输出约简距离最远达 M×N）、Gustavson 牺牲 B 复用且中间输出行大小不定；没有任何单一静态调度能同时最大化 A/B/C 三操作数复用，且静态循环对非均匀非零分布产生负载/计算失衡。
- SegFold 的动态扩展：SELECTA 利用 K 维约简的结合律，在 active window（默认 32 个 k）内逐周期贪心重排 (m,k) 顺序（优先共享 k 以复用 B 行、避免同 m 冲突）；SEGMENTBC 在虚拟坐标空间 V 中即时定位/创建 C 部分和，让 C 元素在 PE 间动态迁移以平衡约简负载——同时拿到 element-wise A 复用、row-wise B 复用与 tensor-wise C 复用。
- 实验意义：15 个 SuiteSparse 矩阵上 geomean 1.95× over Spada（runtime-adaptive baseline）、5.3× over 最佳 Flexagon 静态配置，证明"动态"是数据流设计空间中静态调度无法覆盖的维度。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
