## Scale-Free Intermediate Representation for Distributed Computation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Scale-Free IR 是 Diffuse 系统提出的分布式计算中间表示，其核心特性是 IR 的大小和在其上执行的分析的复杂度**与目标机器的处理器数量无关**（scale-free）。IR 包含两部分：(1) **数据模型**：以 store（分布式数组，由唯一 ID 和矩形 domain 标识）和 partition（从 processor domain 到 sub-store 的映射）表示分布式数据。partition 通过结构化种类（None 表示 replication，Tiling(shape, offset, proj) 表示 n 维仿射 tiling）隐式表达映射关系，无需显式存储每个 sub-store 的边界。(2) **计算模型**：以 index task（IndexTask(d, A)，在 domain d 上的一组 parallel point task）表示分布式计算，每个 point task 操作于特定 processor 对应的 sub-store。区别于 Legion 等低层 runtime 的 scale-aware 表示（显式存储每个 partition piece 的边界），Diffuse IR 是 scale-free 的——task 数和 partition 数决定了 IR 大小，而非 processor 数。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Scale-Free IR 在 Diffuse 的编译框架中的关键作用：

```
高层 Python 程序 (cuPyNumeric / Legate Sparse)
    │
    ▼
库操作分解为 index task stream → 每个 task 标注 store/partition/privilege
    │  例: ADD([(center, P_center, R), (north, P_north, R), (t1, P_t1, W)])
    │  P_center = Tiling(shape=(8,8), offset=(1,1), proj=id)
    │  P_north = Tiling(shape=(8,8), offset=(0,1), proj=id)
    │  → partition 通过公式隐式定义: sub-store-bounds(Tiling(shape,offset,proj), p) = [proj(p)*shape, proj(p+1)*shape) + offset
    ▼
Diffuse IR (scale-free)
    │  IR 大小 ∝ task 数 × store 数 × partition 数
    │  IR 大小 与 GPU 数量无关! (1 GPU vs 128 GPUs → 相同 IR)
    ▼
Fusion Constraints 数据流分析 (Section 4)
    │  Partition equality check: O(1) 时间 (比较 Tiling 的 shape/offset/proj)
    │  而非 O(N²) 检查所有 sub-store pair 的 intersection
    ▼
Fused Task + MLIR Kernel Generation (Section 6)
    ▼
Legion Runtime → GPU Execution
```

Scale-free 的关键设计：partition 的结构化语法分组。同一 syntactic kind 的两个 partition 可通过常量时间比较检测 equality/aliasing。这使得 fusion constraint 检查（true-dependence/anti-dependence）可以在不显式计算 sub-store pair intersection 的情况下完成——后者在 Legion 等低层系统中随 processor 数二次增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) Store: 唯一 ID + 矩形 domain (tuple of non-negative integers)；SubStore(S, P, p) 表示 store S 通过 partition P 在点 p 的 sub-store。(2) Partition: None (全复制) 和 Tiling(shape, offset, proj) 是最基本的两种，实际实现支持更多种类。每种 partition 有共同的 syntactic structure，允许同种类间的 O(1) equality check。(3) IndexTask(d, A): d 是 launch domain，A 是 (S, P, pr) 列表，pr ∈ {R, W, Rd, RW}。(4) 语义通过翻译到 Legion runtime 定义：store → Legion region，partition → Legion partition，index task → Legion index launch。Scale-free 性源于 partition 的隐式映射公式，而非显式存储每个 piece。论文未明确开源 Diffuse 独立仓库（ASPLOS '25），但其底层组件（Legion, MLIR, cuPyNumeric）均为开源。

涉及论文标题：
- Composing Distributed Computations Through Task and Kernel Fusion
