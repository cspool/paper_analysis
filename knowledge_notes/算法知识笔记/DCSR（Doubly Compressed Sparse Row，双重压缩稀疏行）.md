## DCSR（Doubly Compressed Sparse Row，双重压缩稀疏行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCSR（Doubly Compressed Sparse Row）是稀疏矩阵的一种两级压缩存储格式，在标准 CSR（Compressed Sparse Row）之上再压缩一层"空行"：CSR 用 row_ptr[ ] 记录每行的起始位置、col_idx[ ] 记录非零列号、val[ ] 记录非零值，但空行仍占用 row_ptr 中的一个槽位；DCSR 增加一个行索引表，只保留非空行，从而跳过整行全零的存储。对高度稀疏（大量整行无非零）的矩阵，DCSR 把"定位下一非空行"从遍历 O(M) 个槽位降到 O(1)（直接查表），显著减少元数据开销与无效访存。SegFold 论文用它作为矩阵 B 的片上存储格式：B 按行粒度处理（row-wise），DCSR 的第二级压缩在调度时以 O(1) 跳过 active window 中无数值交集的空 B 行——这对高度稀疏矩阵（window 内许多 k 行不贡献任何 A-B 交集）至关重要，否则这些空行会被显式枚举浪费周期。论文还在此基础上为每个 active 行增加一个额外 start pointer，跟踪该行尚未消费的非零元素（支持 partial B 行的交错处理）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DCSR 的构造与 SpGEMM 中 B 行读取流程：
```
# CSR 表示（M 行，nnz 非零）
row_ptr[0..M]   # row_ptr[i] 是第 i 行首个非零在 col_idx/val 中的下标
col_idx[0..nnz)
val[0..nnz)
# DCSR：去掉空行，加一行索引
nonempty_row[i]      # 第 i 个非空行的原行号（升序）
row_ptr_dcsr[0..R+1] # 只对 R 个非空行建指针
# 读取第 k 行非零（SegFold 中 B 以行粒度处理）：
r = 查找 nonempty_row 中 <= k 的位置      # O(log R)，或哈希/指针 O(1)
for j in row_ptr_dcsr[r] .. row_ptr_dcsr[r+1]:
    process B_val[j]                       # 该 B 行的第 col_idx[j] 列非零
# SegFold 的 partial-row 扩展：每个 active 行额外存 start_ptr[r]，
# 记录该行已被消费到第几个非零，支持跨周期交错处理多个 B 行
```
空行占比越高，DCSR 相对 CSR 的节省越大：跳过空行查找 O(1)，避免对 window 内大量无交集 k 的显式枚举。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：DCSR 常见于稀疏线性代数库与稀疏加速器（GPU SpGEMM 等用于跳过空行降低调度开销）；SegFold 中 B 以 DCSR + per-active-row start pointer 存储于片上，A 则因 SELECTA 需要按列扫描而采用 column-major（列主序）格式，两者都只存非零元素，内存控制器含 coalescing unit 合并细粒度请求后再发往 cache/DRAM。FuseFlow（稀疏深度学习编译框架）也采用 DCSR/COO 等格式表达稀疏张量，说明该格式在编译框架与加速器两条路径上都是处理"空行密集型稀疏"的标准手段。证据说明：论文未明确说明 DCSR 的具体位宽/实现细节，以上为基于论文描述（DCSR [1] 引用）与通用知识的推断。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
