## 稀疏张量收缩（Sparse Tensor Contraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
张量收缩是矩阵乘法到多维张量的推广：对两个张量共有的模式求和，产生降阶输出张量。形式化（TensorPrism 式 1）：$C_{f_1,f_2}=\sum_c A_{\{f_1\},\{c\}}B_{\{c\},\{f_2\}}$，其中 $\{f_1\}$、$\{f_2\}$ 为不参与收缩的自由模式（free modes），$\{c\}$ 为收缩模式（contraction modes）；沿每个收缩模式的 fiber（单模式向量切片）长度必须匹配；输入含 m、n 个模式时输出含 $m+n-2|\{c\}|$ 个模式。稀疏版指参与张量含大量零元素、只存/算非零（NNZ），稀疏度跨模式变化且存在跨模式稀疏依赖，是高阶稀疏计算的核心原语。例：3D 张量收缩 $C_{i,j,l}=\sum_k A_{i,j,k}B_{k,l}$；单收缩模式+单自由模式退化为 SpMM $C_{M,L}=A_{M,K}B_{K,L}$。应用：LLM 多头注意力 4D 张量（batch/head/seq/channel）、3D 卷积（多收缩模式 f=2）、科学计算、推荐系统用户-物品-上下文交互、量子态模拟（20-400 阶）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
收缩模式决定稠密输入 B 的取行位置（B[K,:]），自由模式决定输出 C 的位置（C[I,J,:]）——这一"模式角色划分"是数据流设计（push/pull）的依据。TensorPrism 执行伪代码：
```
for each partition P_i (CoGTP 划分, N=PE 数):
    for each clique (I,J,K) in P_i:      # clique=非零元素
        # contraction 顶点 K PUSH 稠密行: 标量-向量乘+向量累加
        partial_C[I,J,:] += B[K,:] * A[I,J,K]
    # 自由顶点 PULL 累加输出行 C[I,J,:]
```
评估覆盖 k∈{64,128}（特征长度）与 f∈{1,2}（收缩阶数）。三类执行路线：mode unfolding（矩阵化）、einsum lowering、张量原生循环变换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：COO/CSR 稀疏格式 + einsum 记号描述收缩 + 展开成 SpMM 或张量原生循环。TensorPrism 的做法：NNZ 坐标→共现图（顶点=索引、边权=共现次数）→ CoGTP 按式 6 划分（决定各模式 tiling 因子）→ 图数据流 push/pull 执行。应用场景：FROSTT 8 数据集（Uber/Nips/Nell-1/Nell-2/Flickr/LBNL-Networks/Chicago-Crime/Amazon-Reviews，3-5 阶、密度 1e-14~1e-2、NNZ 百万到十亿级）+ LLaMA 注意力张量（稀疏化 1%/10%/20%）。性能：相对 SPADE/HotTiles/GSpTC/TCP/HyperSB 几何平均 2.22×/2.40×/1.71×/1.76×/1.49×。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
