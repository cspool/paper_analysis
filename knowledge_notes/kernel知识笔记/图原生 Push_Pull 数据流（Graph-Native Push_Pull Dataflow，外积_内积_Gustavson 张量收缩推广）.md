## 图原生 Push/Pull 数据流（Graph-Native Push/Pull Dataflow，外积/内积/Gustavson 张量收缩推广）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
数据流决定"谁驻留、谁流动、谁广播"——即循环嵌套中稀疏/稠密操作数的数据移动策略。三类经典 SpMM 数据流：Pull/Inner-product（输出驻留点积，输出复用高但稠密输入重复取）；Push/Outer-product（稠密行 B[k,:] 广播与相关非零外积累加 $Partial\_C_{M,L}=\sum_k A_{:,k}B_{k,:}$，输入复用高但部分和散布需大量缓冲/同步）；Gustavson/row-wise（每次取一条稀疏行并流式取对应稠密行累加，平衡输入输出局部性）。TensorPrism 的图原生数据流（§V）不是固定单一模式，而是按共现图遍历顺序动态切换：收缩模式顶点 PUSH 稠密行到目标顶点集（等效外积）、自由模式顶点 PULL 从源顶点拉特征累加（等效内积/row-wise），且一个收缩顶点向多个不同模式的自由顶点集广播实现 inter-mode（跨模式）复用——超出 2D 空间的表达能力。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TensorPrism PE 内收缩引擎执行一个分区（图遍历序）
for clique (I,J,K) in partition P_i:    # 非零元素
    # 收缩顶点 K PUSH: feed unit 广播稀疏输入 A[I,J,K] 给 8 个 MAC
    # 寄存器堆供稠密行 B[K,:] (32 FP32), 多累加器存不同部分和
    partial_C[I,J,:] += B[K,:] * A[I,J,K]   # 标量-向量乘+向量累加
# 列向遍历完成后转 PULL: 自由顶点从剩余源顶点拉特征累加输出行
```
例（Fig.7）：contraction 顶点 K0 向目标集 {I2,J0}/{I0,J0} PUSH（B[K0,:] 与稀疏切片 A[J0,:,K0] 逐非零乘、部分积存 C[J0,:,:]）；K2 向 J1/J2（不同模式）目标集广播实现 inter-mode 复用。硬件支撑：feed unit 8 路广播+连续周期重发（空间×时间复用，最高 128× 复用/取数）；PUSH 写不同输出地址→无写冲突免同步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PE 微架构（fetch unit 分片缓存稠密行、ring 跨单元转发；feed unit 广播；寄存器堆单端口 SRAM+cache；多累加器；commit unit+MAG 映射输出地址）+ CoG Scheduler 按式 6 划分后按分区分派。相比 baseline：inner-product 牺牲输入局部性、outer-product 需昂贵部分和同步（GSpTC 用 outer 在 chcr 上归约竞争占 73% 执行时间）、Gustavson 平衡但限 2D；TCP 编译期固定数据流+电路交换网络无法适配不规则（power-of-2 padding 浪费 2.89× 带宽）。图数据流动态切换使吞吐量平均 2.07×/1.71×/1.55×（vs GSpTC/TCP/SPADE&HotTiles），nel1 上 2KB feature 时 2.95×。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
