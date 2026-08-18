## IDD-Scan（Intra-Segment Dependency Decoupled Scan，段内依赖解耦扫描）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ENEC 提出的、专为 Ascend NPU 设计的并行前缀和（scan）实现。动机：AscendC 要求操作数 32 字节对齐，half 类型（2 字节）每行 M=16 个元素恰好 32 字节，架构禁止对同一 32 字节硬件段内相邻元素直接做 SIMD 运算——即 row[i] += row[i-1] 这类"段内依赖"被硬件锁定、无法直接计算；同时 Ascend 无 CUDA 式轻量线程同步（每 AI core 是单一重线程）。IDD-Scan 用"转置 + 列方向扫描 + 层级行扫描"把禁止的段内依赖改写成硬件友好的跨行向量加法，把计算受限问题转化为一系列向量化操作（多阶段算法）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
目标：对 N×M 局部张量（M=16）做全前缀和。两阶段（以 8×16 half 张量为例，Figure 8）：
```
# Stage 1: 行内前缀和 via 转置（把行内依赖变行间）
T = transpose(M_local)              # 8×16 → 16×8：原每行元素散到 16 个新行
R_T = column_prefix_sum(T)          # 对 16 行的每列做 log2(M)=4 步向量加法
R = transpose(R_T)                  # 16×8 → 8×16：R 每行含正确的行内局部前缀和
# Stage 2: 行间传播与最终修正
C = R                               # 保存局部结果副本
for k in 1..log2(N)=3:              # 层级行扫描
    C[i] += C[i - 2^k]              # 行间元素级加法
offset = exclusive_scan(C[:, last]) # 末列含包含式行偏移 → 去尾补 0 得排他偏移
result = R + broadcast(offset)      # 偏移矩阵广播加到 R
```
Annotations：转置把"段内（同行 32B）相邻依赖"变成"跨段列方向依赖"，列方向扫描每步只做整列向量加法（合法）；行间传播用 log2(N) 步，无需原子/同步；最终排他偏移广播回加。代价是多几次转置/内存搬运，但把不可行的操作变成全向量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 的向量加/转置（half 张量）在 AIV 上执行，是 ENEC 解压 kernel 的核心原语——把 bit mask 转 0/1 整数后求前缀和，得到逆 gather 的偏移。效果：V3 相比 V2（朴素 scan）解压吞吐提升近 100%（IDD-Scan 直接贡献）。通用性：任何 Ascend 上需要 scan/前缀和的 kernel（如 softmax、GEMV 累加、压缩位重排）都可复用；GPU 移植版直接改用 CUB 库并行前缀和（ENEC-GPU-V1，吞吐 419.2 GB/s），说明该算法是 Ascend 对齐约束下的专用替代品。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
