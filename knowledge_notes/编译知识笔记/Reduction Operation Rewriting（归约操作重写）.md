## Reduction Operation Rewriting（归约操作重写）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CODO 细粒度违例消除的第一个子方法，解决 FIFO 通道两侧访问计数不一致。多数计数 mismatch 来自归约操作（全连接、最大池化、归一化等）：归约引入的循环维度不直接对应数组索引，导致归约迭代期间产生冗余 FIFO 访问，producer 写数与 consumer 读数不等，最终 FIFO 死锁（Fig. 5：max pooling 写 out 与初始化读 out 计数不一致）。
- 变换核心：分析 producer/consumer 访问同一数组的循环结构 → 按数组被访问的循环层级与包围循环迭代计数乘积算总写数/读数 → 检测到 mismatch 后，把对应 FIFO 数组索引的循环维度分类为 index dimension、其余为 reduction dimension → 把 reduction dimension 移到最内层 → 把对输出数组的写移出归约区 → 引入临时数组聚合中间结果。这样 producer 每次归约只向 FIFO 写一次最终结果、计数与 consumer 一致，且中间结果 just-in-time 计算/传输（提早 FIFO 写，避免 StreamHLS 式延迟写）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 伪代码（max pooling 例）：
```
# 变换前（违例）：producer 每归约迭代写一次 out，consumer 只读一次
for i, j:                      # index dims
    for k in window:           # reduction dim
        out[i,j] = max(out[i,j], in[i+di,j+dj])   # 写 count = 窗口元素数
# 变换后：
for i, j:
    tmp = -inf
    for k in window:           # reduction 移到最内层
        tmp = max(tmp, in[i+di,j+dj])
    out[i,j] = tmp             # 写移到归约区外，写 count = 1
```
- 在 CODO 流程中的位置：细粒度违例消除 pass 的一部分，与 permutation map 互补（计数 vs 顺序）；该 pass 同时"提早 FIFO 写"，体现 co-optimization 原则——正确性变换兼顾通信效率，并为后续 reuse buffer/调度 pass 提供指导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 MLIR affine dialect 上的模块化 pass：识别归约区域（基于累加/求最大等运算模式）→ 循环重组 → 临时数组插入，对用户透明、codo-opt 自动执行。效果例：Padding→Conv2D 尾迭代死锁（Padding 写完最后数据后 Conv2D 仍等待）被消除，全模型可端到端 FIFO 流式执行。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
