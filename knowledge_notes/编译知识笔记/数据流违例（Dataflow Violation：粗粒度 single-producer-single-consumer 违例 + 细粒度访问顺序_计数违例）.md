## 数据流违例（Dataflow Violation：粗粒度 single-producer-single-consumer 违例 + 细粒度访问顺序/计数违例）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据流违例是使 HLS dataflow 优化无法正确/高效生效的代码模式，CODO 系统分为两级：
  (1) 粗粒度违例：违反商用 HLS 的 single-producer-single-consumer 约束——同一 buffer 被多节点读写，分三类访问模式：single-producer-multi-consumer（残差/旁路结构，如 ResNet-18、GPT-2 的 bypass）、multi-producer-single-consumer（初始化+padding 配对）、multi-producer-multi-consumer。Vitis HLS 检测到即跳过该段 dataflow 优化（区域退化为串行执行）。
  (2) 细粒度违例：FIFO 通道要求 producer 与 consumer 的写/读访问顺序与访问计数严格一致；不一致（如 Padding 按 (3,34,34) 循环序写、Conv2D 按 (34,34,3) 读；或 Padding 写完最后数据后 Conv2D 仍在等待→FIFO 死锁）导致数据丢失、FIFO 上溢/下溢甚至死锁，且综合阶段不报错。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CODO 在 MLIR 上分两个 pass 消除：粗粒度 pattern-aware 变换（Algorithm 1：遍历 buffer → 收集访问节点 → analyze_access_pattern 分类 → apply_transformation）按三类模式分别处理——single-producer-multi-consumer 插入复制中间节点（Node1' 读 a 写复制缓冲 b/b'）；multi-producer-single-consumer 用 node fusion 融合共享外层迭代域且无 loop-carried 依赖的写循环（内层结构差异插控制逻辑、中间结果暂存并最终并入最后一次写）；multi-producer-multi-consumer 复制 buffer 使每个 buffer 单写单读。
- 细粒度用 Systematic Read-Write Coordination 两个子方法：reduction rewriting（统一访问计数）与 permutation map generation（统一访问顺序），见各自条目。
- 检测方式：在 IR 层面静态检测（比对 producer/consumer 的循环结构与数组索引），而非靠 HLS cosim；细粒度计数检测 = 找目标数组被访问的循环层级、以包围循环迭代计数之积计算写数/读数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 效果：违例消除后全图可连续 dataflow 且层间可安全用 FIFO（Table VIII：Gesummv/Residual Block/MobileNet/ResNet-18 100% FIFO，Multi-Head Attention 84%、GPT-2 89%）。对照 prior work：ScaleHLS 只部分消多消费者违例、POM 靠用户手动管理依赖、HIDA 只消粗粒度、Allo 不处理违例（上板死锁）、StreamHLS 单任务部分消解（FIFO 写推迟到近 8/9 迭代后）、StreamTensor 类型不匹配即回退 ping-pong。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
