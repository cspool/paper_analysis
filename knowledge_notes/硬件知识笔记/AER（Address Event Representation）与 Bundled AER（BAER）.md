## AER（Address Event Representation）与 Bundled AER（BAER）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AER（地址事件表示）是 Carver Mead 团队（Caltech）开创的神经形态系统通信协议：把神经元发放事件（spike）编码为"只含发放神经元身份/地址"的多 bit 包，在片上/片间以包交换转发，从而以极少布线承载大量稀疏事件。TrueNorth 的 spike 包含 9-bit delta-x、9-bit delta-y、4-bit 投递时间、8-bit 目标 axon 索引、2 debug bits，经异步包交换 mesh NoC 路由；SpiNNaker 用固定宽度包（8-bit 头 + 32-bit 内容 + 可选 32-bit payload）。
- 问题（ELSA 动机）：SNN 硬件以 AER 逐 spike 传包（如 32-bit），头部含空间位置与时间步信息；SNN 激活稀疏（ViT 上 >80% 稀疏）但随时间步重复传输，TrueNorth 相对 QANN 可产生 8× 流量（Fig.6）。ELSA 提出 Bundled AER（BAER）：把同一膜行（row）的 spike 聚合进单个 flit，头部只写一次，摊薄 per-spike 包头开销。例：传统 AER 传 17 个 spike 需 17×25-bit=425 bits，BAER 压到 256-bit。
- BAER flit 字段（ELSA Fig.12b）：6-bit 路由目的（hop 计数 m,n）+ 2-bit type（flit 在 spine/token 内位置：begin/body/end）+ 12-bit spine/token ID + 12-bit spike 位置 + 1-bit 极性（sign）+ 15-bit 校验（ECC）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 ELSA 路由器中的运转流程（Fig.11 五条数据通路）：
```
本地通路（①②）：PE 产出的 spike → Local Input Reducer 收集 →
   Flit Generator 攒够同行 spike 打包成 BAER flit（不足则零填充）
   → 注入 NoC（Routing Engine 按多路径路由概率选输出端口）
远端通路（③④⑤）：收到 flit → Arbiter 检查 hop 计数 (m,n) 是否归零 →
   Flit Decoder 解包还原 spike → 入 FIFO Queue（核间 pipeline register）
   → Output Scheduler 按 spine/token 依赖顺序调度进 PE 计算
```
- 例：层 2 产出的第 1 个 spine 的 17 个 spike（同一膜行）被打包成 1 个 256-bit BAER flit，经 2D-mesh 多路径转发到层 3 所在核，解码后立即触发层 3 该 spine 的计算——BAER 的行捆绑同时为 mini-batch Gustavson 提供了行对齐 mini-batch（见 kernel 层术语）。
- Annotations：本地/远端通路分离（①⑤）避免五通路争用；type 字段标记 flit 在 spine/token 内的位置使解码端知道何时 spine 完整、可触发下一层；hop 计数使路由器免查路由表即可判到达。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ELSA 路由器内的 Flit Generator/Decoder、FIFO、Arbiter、Routing Engine；多路径路由的传输概率由遗传算法离线优化（见编译框架层术语）。效果：Tab.VIII 中 ELSA 相对 TrueNorth 平均降 20.5% NoC 流量、24.3% NoC 能量（ResNet18/34/50/ViT-S 同 6×6 mesh 公平对比）；Fig.25 显示 flit 尺寸存在最优区间（48-bit 太小把 spine 拆碎、256-bit 太大负载利用率低），ELSA 用 256-bit；Fig.22-C 消融显示 BAER 主要贡献在降低通信（PE 计算仍是主导，故延迟增益有限但能耗显著）。Web 证据：AER 家族还包括光学的 O-BAR（Princeton）等变体；TrueNorth 的异步 AER 对外暴露为全双工 packet-bundled 接口，可接硅视网膜/硅耳蜗。

WaferBRAIN 补充视角（ISCA'26，NAHP 的 AER 压缩）：WaferBRAIN 把 AER 称为 "Addressing Event Representation"（Fig.2），BPU 的 axon-in 模块"ingests AER spikes from local and global sources"——spike 以 AER 包形式进入处理节点。NAHP 是对 AER 包格式的进一步压缩与路由/寻址解耦：本地广播包只带 LNid=<LNode,Neuron>（27bit，比 neuron-centric 的 FNid 37bit 少 10bit）、全局单播包带 <POD,Wafer,Node>+GAid（33bit，比 axon-centric 的 NodeID&FAid 46bit 少 4bit）；路由只用包头物理 ID（广播路由器只看 LNode、单播路由器只看目标坐标），GAid 仅接收端用于索引突触——"路由与寻址解耦"是 AER 支持广播/定向投递混合的基础。与 ELSA BAER 的"同膜行 spike 聚合摊薄包头"正交互补：NAHP 从包宽（ID 压缩）与包数（边界触发减少单播）两个维度降 AER 流量。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
