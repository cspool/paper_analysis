## 环形拓扑（Ring Topology，chiplet 互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 环形拓扑是 chiplet/节点按首尾相接的环组织互连的结构：每个节点只与相邻两个节点直连（前驱+后继），数据沿环单向/双向逐跳传递，首尾闭合形成环路。相比全互连（all-to-all）/mesh，环形拓扑的链路数量最少（每节点 2 条 D2D 链路）、布线/面积开销最低，但最坏跳数与链路负载随节点数线性增长（跨环远端传输需多跳）。在 CASCADE 中，12 个 HMUX Chiplet（HC）按 4×3 网格布局、经 UCIe D2D 链路组成环形拓扑（HC_{C-1} 回传数据给 HC_0），用于支撑跨 HMUX 流水线：中间 RLWE 密文（ACC）沿环在相邻 HC 间传递。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CASCADE 的环运转流程（HMUX 链，n 可远大于 C=12）：HC 在每个时间槽同时"把输出传给下游 + 从上游接收密文"（全 HC 并行）→ ACC 沿环逐跳流动，HMUX_i 在某个 HC 完成后经 1 跳 D2D 到下一 HC 执行 HMUX_{i+1} → 当 n>C 时，RLWE 在环上循环多圈直到 n 次迭代完成 → 最后 HC0 的 VPU 做 key-switching。BSK-stationary 数据流保证 ICT 只发生在物理相邻 chiplet 间（单跳、无跨环拥塞）。环的芯片设计权衡：D2D 时延（16 GT/s/1024 Gbps 下仍高于 intra-die 访问）> HMUX 计算时间时 HC 欠利用 → 用 Interleaved-Fusion 把连续 HMUX 融合在本地（组内不跨环）、仅组间经环传输，把 D2D 通信频率降下来；环上 ACC 的 D2D 传输经输入/输出 double buffer（128 KB 各）与计算重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每 HC 集成 D2D PHY（UCIe Advanced，16 GT/s、64-bit → 1024 Gbps）+ 输入/输出 double buffer；环的物理层是 2.5D 无源硅中介层上的短距 D2D 链路。使用：环形拓扑适合"数据依赖链式流动 + 需要任意长链（n 不受 C 限制）"的流水线场景（CASCADE 的 HMUX 链即流水线阶段）；与 mesh（Task/tile 并行、任意路由）相比，环把通信局部化到相邻节点、硬件开销最低，但要求调度把通信量集中到相邻节点——这正是 OIFS 的 Interleaved-Fusion 映射（f(t,c) 二维时空矩阵，组内本地、组间沿环交错）与 BSK-stationary 数据流的设计目标。可扩展性：CASCADE-x（不同 HC 数）在 DeepCNN-100 上端到端时延随芯粒数增加持续下降，得益于环上通信不拥塞（ICT 只发生在相邻 chiplet 间）。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
