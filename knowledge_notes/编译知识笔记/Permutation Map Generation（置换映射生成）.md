## Permutation Map Generation（置换映射生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CODO 细粒度违例消除的第二个子方法，解决 FIFO 通道两侧数据访问顺序不一致（如 Padding 按 (3,34,34) 循环序写 FIFO、Conv2D 按 (34,34,3) 读）。思路：以计算最重的 bottleneck 循环（如卷积、attention 的 Q*K，按各嵌套循环 trip count 与计算强度选定）为 reference loop，分析其输入/输出数组的访问顺序，再调整与之相连的 producer/consumer（target loop）的访问模式使其一致。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 四步流程（Fig. 6）：Step1 对 reference 与 target loop 分别建立"连接数组维度→循环深度"映射（如 reference loop 中 out 的维度 {n,co,h,w} 对应深度 {0,3,1,2}）；Step2 对 reference loop 做 tiling size=1 的循环切分（把 h、w 各拆两层），使两边循环深度对齐；Step3 构造深度-深度映射（如 2→1 表示 target loop 深度 2 的循环应换到深度 1）；Step4 按映射对 target loop 做循环置换。
- 置换后 producer 与 consumer 的 FIFO 写读顺序逐元素一致、FIFO 流式执行合法；对比 StreamHLS 靠插控制逻辑推迟写（近 8/9 迭代后才写）与 StreamTensor 的类型系统（不匹配即回退 ping-pong），CODO 用循环变换从根上统一顺序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 MLIR 上基于 affine 分析的 pass：数组索引仿射函数 → 维度-深度映射表 → tiling/permutation 变换。DNN 层间普遍存在的布局变换（NCHW↔NHWC 类）在 CODO 中退化为循环置换、不产生额外数据搬移。与 reduction rewriting 共同构成 Systematic Read-Write Coordination，把层间通信升级为 FIFO（Table VIII 100% FIFO 场景）。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
