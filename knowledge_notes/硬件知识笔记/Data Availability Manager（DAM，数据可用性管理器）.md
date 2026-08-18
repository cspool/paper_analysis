## Data Availability Manager（DAM，数据可用性管理器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAM 是 MoE-Hub 放在消费者 GPU hub 中的硬件模块，用硬件管理的数据就绪信号替代软件轮询（busy-waiting），实现事件驱动的线程块（TB）调度。动机（Insight-2 消费者侧）：MoE 细粒度 token 到达使专家 kernel 需要持续轮询 semaphore 检查数据可用性——大量 warp 只为查信号而活跃，占用内存带宽与计算周期，通信粒度越小 busy-waiting 越严重（论文 Fig.6 显示轮询占消费者执行可观比例）。DAM 把"数据何时就绪"的追踪交给硬件。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DAM 结构：核心是 CAM 结构的 **Dependency Table**，映射内存地址范围 → 依赖它的 TB 组集合；该表由编译器对消费者 kernel 做静态 tiling 分析生成（TB 计算某输出 tile 依赖输入激活矩阵对应切片；依赖关系不随动态 token 顺序变化），每个模型（由 HiddenSize、FFNHiddenSize 定义）与目标硬件生成一次并按 auto-tuned tiling 方案缓存复用。每个唯一 TB 组配一个 **TB Status Counter**。运转流程：普通远程 st. 或 st.rowsp 的写应答返回 hub → DAM 在 Dependency Table 做范围查找 → 每个匹配条目对应 TB Status Counter 自增 → 计数器达阈值（该 TB 组所需全部数据已到本地内存）→ DAM 向 GPU 线程块调度器发 Ready 信号 → 这些 TB 立即可被派发到空闲 SM。此外 **Global Counter** 追踪消费者 kernel 期望的写应答总数（如 HiddenSize×SequenceLength×k）做运行时适应：达目标即发 AllReady 信号，识别并回收那些因编译器保守分配而生、计数器仍为 0、永远不会被调度的 TB，避免占用 SM 资源（MoE 专家动态 token 数下防资源浪费/过度订阅）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 hub 内 CAM（Dependency Table）+ 计数器（TB Status Counter/Global Counter）+ 触发 TB dispatcher 的逻辑；编译器负责静态 tiling 依赖分析生成表。对软件透明：消费者 kernel 无需任何轮询代码。消融数据（MH-DEP，routing→GEMM1 窗口）：硬件信号机制平均加速 1.14×，收益在小序列长度最大（软件原子轮询控制开销占比高、暴露通信时间在关键窗口），随序列变长 GEMM 摊薄控制开销后收益趋缓但仍为正。

涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
