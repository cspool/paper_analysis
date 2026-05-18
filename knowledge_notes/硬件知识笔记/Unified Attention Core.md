## Unified Attention Core

术语是什么？
Unified Attention Core 是 VAR-Turbo Accelerator 中的可重构注意力计算核心，通过 PE Cluster/Node/Cell 三级层次结构和 Snooper+Fat Tree 分发网络，在同一 PE array 上支撑两种不同的 attention dataflow：BA（Big Attention）模式使用 Row dataflow（按 attention head 分配 PE Cluster），SA（Small Attention）模式使用 OP dataflow（output-stationary dataflow）。通过动态切换 dataflow 避免为两类 attention 分别放置独立 core 导致的硬件利用率不足。

从硬件架构角度拆解术语：
Unified Attention Core 的硬件层次和 dataflow 切换流程：
1. PE Cell：最小计算单元，执行 MAC 操作
2. PE Node：多个 PE Cell 组成，执行向量级操作
3. PE Cluster：多个 PE Node 组成，对应一个 attention head
4. Snooper（配置单元）：根据当前 layer region 发送配置信号，设定每个 PE Cell 应接收的 packet ID
5. Fat Tree（分发网络）：根据 Snooper 的配置将数据包路由到对应 PE Cell lanes
   - BA 模式（Big Attention/Row dataflow）：PE Cluster 按 attention head 分配，执行标准 Row-stationary attention dataflow
   - SA 模式（Small Attention/OP dataflow）：PE Cell/Node 切换为 output-stationary dataflow，执行 local window 内 token 的压缩 attention
6. Shared FP Accumulator：Row+OP MAC 通过 Divide-and-Conquer 和 Fluid Zone Detection 技术共享 FP 累加器，降低 FP 累加的功耗和面积

术语一般如何实现？如何使用？
Unified Attention Core 的 RTL 作为 VAR-Turbo accelerator 的一部分实现，通过 Synopsys 工具链综合（TSMC 28nm）。Row dataflow 适合大矩阵的全局 attention（高吞吐），OP dataflow 适合小窗口的局部 attention（低延迟）。Snooper 的配置是运行时动态的——每层推理前根据层索引选择 dataflow 模式，通过配置 packet ID 路由实现单 cycle 内切换。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

