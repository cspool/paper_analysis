## 执行阶段（Program Phase）与数据中心工作负载阶段表征

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Program phase（程序执行阶段）指程序运行期间计算行为相对稳定的时间区间，期间对硬件资源的占用模式（IPC、cache 行为、访存/网络强度）保持一致，随后切换到特征显著不同的另一阶段。PhaseWeave 将其定义并应用于数据中心负载：毫秒/亚毫秒尺度的 phase 交替由三类来源驱动——(1) 微服务架构中不同角色的服务（Nginx 网络密集、Memcached/MySQL 内存密集、HHVM 计算密集）；(2) 连接微服务的 datacenter tax 操作（加密/序列化/压缩，各自 compute- 或 memory-bound）；(3) 单服务内部多阶段（AdSim 的 GEMM→DeepCopy→PtrChase）。传统研究（SimPoint、Dhodapkar-Smith 的指令工作集、Isci-Martonosi 的事件计数签名）针对粗粒度、秒级 phase；PhaseWeave 针对毫秒/亚毫秒、输入相关的细粒度 phase。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
phase 概念是 PhaseWeave 硬件-软件协同的支点：硬件以 100µs epoch 为粒度采样 IPC、cache/TLB/branch MPKI、分类系统调用频率等 15 个特征，把每个 epoch 分类为 compute-/memory-/network-/low-power 四类 phase，据此把线程迁移到最匹配的 chiplet。例：Mediawiki 的一次请求经历 Nginx 网络 phase（高频网络系统调用、低 IPC）→ Protobuf 反序列化 phase（混合）→ HHVM 计算 phase（高 IPC、高分支 MPKI）→ Memcached 查表 phase（高 L1 MPKI 指针追逐）；硬件预测器在每个 100µs epoch 末判断下一 phase，调度器据此把该线程迁到 near-network/compute/fast-memory chiplet。phase 时长（数十-数百 µs）直接约束了检测/迁移开销预算：软件 RF 推理 50-250µs 已与 phase 等长，故必须硬件化（<100 cycles）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线用 DCPerf 的 WDLBench 微基准在每类 chiplet 上做灵敏度扫描（哪个硬件配置对该 phase 最优就标为该 phase 的最优 chiplet），为每个训练样本打标签；在线以硬件计数器+系统调用分布为特征做分类。使用：phase 概念可作为数据中心服务器设计的通用分析单位——同类工作可复用于"同构 vs 异构"的量化比较（论文用 IPC 时序图、top-down 微架构分析 [31][94]、资源灵敏度扫描——频率/内存带宽/L2 容量/代际——来表征 phase 的资源敏感性）；也可用于 SimPoint 式采样或 DVFS 等下游优化。注意 phase 分类不能用语义标签推断（相同操作因输入不同行为差异大），必须动态自适应。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
