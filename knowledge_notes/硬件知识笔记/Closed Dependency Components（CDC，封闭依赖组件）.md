## Closed Dependency Components（CDC，封闭依赖组件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CDC 是 MLX 对结构化算子数据流图 G=(V,E) 的形式化分解单元：CDC 是子图 C⊆V，对入边封闭（∀v∈C：v 的前置依赖⊆C∪In(C)，In(C) 为 C 的外部输入）——即 C 构成自包含的局部更新区域、有界局部性。与任意 tiling 不同，CDC 由算子的封闭依赖模式定义而非启发式分块；每个 CDC 有固定输入/输出接口（交换值只由模板参数如蝴蝶宽度或 MM/CONV 块形状决定，不随整体问题规模增长）。结构化算子含大量接口相同的重复 CDC 实例，故 MLX 可跨实例复用同一 tagged-block 模板。
- 两个关键性质：(1) 前向-only 分层——算子可表达为 CDC 层 {C_0,...,C_K}，每条边在层内或到下一层（ℓ(v)=ℓ(u) 或 ℓ(u)+1），层内 CDC 并行、层间严格前向，形成无长程/循环依赖的流水；(2) 编码分层路由——每 CDC 带轻量层索引 ℓ，直接选择下一级路由类，端点 PE 由 CDC-to-PE 静态放置决定（传输属少量仿射偏移，可参数化为 (Δx,Δy)），实现"有限路由类"。原则：任何可表达为 CDC 层 {C_0..C_K}（严格前向依赖、闭工作集 S_0⊆...⊆S_K）的结构化算子都可 MLX 执行；层间边形成确定性前向流水、消除全局调度，并支持空间折叠（多个逻辑 CDC 层覆盖到固定网格、逻辑深度与物理阵列大小解耦）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CDC 在 MLX 中的运转流程（BSMM 例子）：n=8 向量蝴蝶分解 → 每 B=2^k 个元素构成一个 closed set（n/B 个不相交闭集），块内所有蝴蝶交互不越界 → 块大小 L（FFT chunk）或 B（BSMM 块）增长时默认索引/布局把层间交换变成半阵列长 stride 洗牌，破坏空间局部性 → MLX 利用蝴蝶依赖图的代数可划分性重排 FFT/BSMM 使其严格尊重闭集 → 长蝴蝶流水分解为可复用数据流 stage；stage 间 I/O shuffle 用 scratchpad 访问原语重索引（Fig.11(c)：shuffled stage-2 值 B_8'(0) 逻辑上继承 stage-1 的 B_8(2)，但被重映射到与 B_8(0) 相同的空间足迹与执行模板）→ 长程蝴蝶依赖转成重复的紧凑本地数据流 + 有界次数的 stage 间交换。SWA 例子（Fig.12）：windowed score accumulation（QK^T，FMA）→ row-wise max（FMAX）→ exponentiation + normalization stats（FEXP + sum/broadcast）→ weighted accumulation（SV，FDIV/FMA）——各阶段为相邻依赖链的 CDC 层，折叠到同一 2D 阵列，不同层消费不同 FU 原语（FMA/FMAX/FEXP）故 tagged-block 执行可利用异构性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：编译器/运行时把算子数据流图静态划分为 CDC 层（Sec. V 映射：BSMM 三层嵌套循环——内层 i2 全展开于网格、中层 i1 PE 内执行、外层 i0 由序列器迭代）；每 CDC 映射为 loop-driven tagged block，按 tag 依赖触发（C_i ↦ loop(k) tagged_kernel_i(k)），PE 跨 CDC 实例重放短 tagged block 摊薄译码与调度开销；折叠不需要所有逻辑层同时活跃，小 in-flight 窗口即可维持 FU 利用率并约束片上缓冲。轻量辅助运行时按需调度预定义 CDC 序列处理粗粒度不规则（如 bucketed MoE），极不规则模式靠 credit-based 流控保正确（极端失衡产生气泡、利用率下降）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
