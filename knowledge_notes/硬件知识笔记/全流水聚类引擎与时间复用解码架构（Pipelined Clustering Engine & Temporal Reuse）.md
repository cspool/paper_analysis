## 全流水聚类引擎与时间复用解码架构（Pipelined Clustering Engine & Temporal Reuse）

术语解释
本论文 FPGA 解码器的组织方式：以时间复用替代空间展开——单条 7 级深度流水聚类引擎流式处理 + K=24 个复制 EFE 实例并行 + Voting 模块。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QEC 解码硬件的两种组织范式的对立：spatial（空间）架构把顶点/簇映射到分布式 PE 阵列（Helios per-vertex PE、QUEKUF），延迟随 d 亚线性但资源随码距快速增长且存在 per-iteration 协调地板；本论文采用 temporal reuse（时间复用）：一条 deeply-pipelined clustering engine 每周期流式处理一个 VID（7 级 S1–S7），K 个 Ensemble Forest Exploration 实例复制并行（遍历态不可时分复用——会覆盖在途邻接数据），Voting 聚合逻辑结果。资源集中在单条深度流水而非 d² 规模的 PE 阵列，配合 forwarding bypass 与冲突无关存储把流水打满。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
每周期 1 个 VID 流过 7 级流水:
S1–S3: 出队 VID; 并发取 RID 与边权 -> 解析 CID
S4:    grow/merge 判定  —— bypass 网络把判定回喂 S1–S3 解 RAW
S5–S7: priority FIFO 管理 active CID; 更新边界顶点状态
聚类完成后: K=24 EFE 并行(优先级森林+ROE) -> Voting 聚合
```
效果数据：baseline 两段式（聚类流水 + 后聚类遍历）stall 占 48–58% 总延迟；加优化后降至 1–7%，整体 2.2–3.6×（d=9 为 3.0–3.4×、d=11 为 3.2–3.6×）；d=9 p95 2.12→0.65 μs、p99 3.09→0.90 μs。与 Helios 对比：d=3 时本文快 3–5×（Helios 有 per-iteration 地板），d=7–9 收敛，更大 d 时 Helios 亚线性更优——本文定位"资源高效的 latency/area 点"（d=15 时较 Helios 少约 6× LUT、3× FF）。残余 1–7% stall 来自聚类尾部 parity-update RAW hazard（新顶点 merge 判定需等前一步奇偶提交），论文论证 speculative parity 评估/乱序发射的控制与带宽开销不值，保留顺序流水。资源随 d 缓增：仅格点坐标寻址组件（多 bank 顶点/边缓冲、每 EFE 邻接存储）按 O(2^{⌈log2 d⌉}) 增长（2 的幂地址量化），其余保持 d=15 尺寸。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog HDL，Xilinx Virtex UltraScale+ VU19P，Vivado 2024.2 综合，163 MHz，108k LUT / 43k FF / 252 BRAM（d=15）；单 RTL 覆盖全部码距（同一设计跑 d=3..15+）。先以 Python cycle-accurate simulator 镜像微架构数据流验证（报 LER 与 cycle 计数、可逐项开关优化），再交叉验证 RTL。K=24 分支不占关键路径，Vivado 功耗报告：每 EFE 分支 ~50 mW，合计 ~1.2 W 动态功耗，增 K 仅线性加分支项。使用场景：需单 RTL 可扩展、低 LUT 的实时 QEC 解码部署；对照 Helios/QUEKUF 的 spatial 组织选型。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
