## Bounding Volume Hierarchy（BVH，包围体层次结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BVH 是光线追踪的标准加速结构：把场景图元递归组织进包围盒（AABB）层次树，遍历时通过 ray-box 求交逐层剔除不相交子树，把求交测试次数从 O(N) 降到 O(log N)。Vulkan 采用两级结构（GauTracer Fig. 2(b)）：每场景一个顶层 TLAS（内部节点是层次 box，叶节点引用 object instance 及其变换矩阵），每个唯一对象几何一个底层 BLAS（叶节点为三角形网格或 procedural 原语）。procedural 叶是非内置类型，由通用 shader core 处理求交与响应。GauTracer 用 BVH 组织 Gaussian 原语：每个 Gaussian 一对一映射为一个 procedural 叶节点（避免 [27] 的 icosahedron 网格代理每高斯展开 20 三角形，BVH 体积减 26×、构建快 1.5×），AABB 由高斯空间缩放参数保守定义覆盖体积跨度；BVH 用 Intel Embree 以分支因子 6 构建。
- 从硬件架构角度拆解术语，给出运转流程具体例子：RTA 遍历 BVH 的流程 = ray 进入根节点 → node decode 判类型 → 内部节点做 ray-box 测试（RBIU）→ 命中则子节点入遍历栈 → 叶节点按类型分派（triangle→RTIU，Gaussian→TRAN+RGIU）→ 记录最近命中。GauTracer 采用 treelet 化 BVH：全局 BVH 划分成紧凑子树（treelet），每 treelet 256 节点（64B/节点 = 16KB）连续存放、适配 L1 cache 与 ping-pong 预取；射线先在当前 treelet 内处理完再进下一个（counter table 调度），遍历栈分两级（treelet stack + node stack）存每射线寄存器，溢出按 short-stack 策略落 local memory。far-node pruning 进一步用"当前最大命中距离"作为几何屏障，裁剪 ray-box 命中距离超阈值的节点，减少冗余访问（每射线访问节点削减 1.2~1.9×）。
- 术语一般如何实现？如何使用？：软件/驱动构建常用 Intel Embree（https://github.com/RenderKit/embree，GauTracer 用其建 BVH）、OptiX 的 accel 构建；硬件上 RTA 执行遍历（NVIDIA RT Core、Vulkan-Sim 的 RTA 模型）。GauTracer 修改 Vulkan-Sim 的 BVH 节点打包/解码（Gaussian 叶 64B 与三角形叶一致，含 12 元素逆变换矩阵 T' + opacity + GID + 2B descriptor，SH 颜色等外观参数单独存纹理经 GID 索引）。

TTP 补充视角（ISCA'26，BVH 遍历趋势与 DFS/BFS）：BVH 遍历可用 DFS（LIFO 栈，栈存节点地址而非数据）或 BFS（FIFO 队列）。Embree 构建 6-ary 树（每叶 1 primitive）。DFS closest-hit 遍历（Algorithm 1）：根命中则压栈→循环弹栈→读节点→内部节点对 6 个子 AABB 做 ray-box 测试、命中的压栈→叶子做 ray-triangle 测试更新最近命中距离。关键观测：BVH 遍历呈"下-上"交替趋势（连续 pop=向上/同层遍历），长 pop streak 占 RT read miss 的大头，而栈中已存这些节点地址→TTP 的预取机会。DFS vs BFS 对比（表 I）：平均每射线访问节点 DFS 49.0 vs BFS 70.0（+42.9%），最大 487.8 vs 636.9（+30.6%），chsnt 差最大（59.0 vs 137.4，+132.9%）——BFS 不可跳过更远节点；但 BFS 可预测（FIFO 队头即下一节点），加 TTP 后 BFS 反超 DFS。
涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing

RoboCortex 视角（ISCA'26，物理局部性在 BVH 上的泛化）：BVH 是空间数据结构之一（包围体层次），RoboCortex 的物理局部性方法（缓存相邻查询的共享搜索路径）不限于点云 CPU 场景——论文在 Vulkan-Sim（GPU ray tracing 模拟器）上对 BVH 遍历应用同一思想：与 Treelet 式优化（[8][9]，树簇预取优化 BVH 遍历局部性）相比，RoboCortex 挖掘的是"连续搜索操作之间"的跨查询局部性而非单查询内节点局部性，因此与 treelet 正交；实验结果在已充分优化的 Treelet 方案之上再获 4%-34% 额外性能提升（不同物体）。泛化前提是稳定的包含关系：目标叶节点必须包含在中间节点内（点作 key、点作 value、最小距离为准则），因此可推广到包围盒碰撞检测（ray tracing、obstacle avoidance）；对"线作 key、点线距离为准则"的 NNS 需进一步特化 belief space。
