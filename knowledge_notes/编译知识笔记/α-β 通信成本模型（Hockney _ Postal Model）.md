## α-β 通信成本模型（Hockney / Postal Model）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
α-β 模型（Hockney 1994，也称 postal model）是并行/分布式计算中最基础的通信成本抽象：发送大小为 s 字节的消息耗时 T(s) = α + β·s，其中 α 是固定启动/延迟开销（协议握手、启动传输，单位秒），β 是单位字节传输时间（β = 1/bandwidth，单位秒/字节）。PipeComm 用该模型作为成本分析的底层：每条链路以其带宽推导单 chunk 传输延迟 w（w = chunk_size / BW），MILP 深度约束（Eq.4）与 II 容量约束（Eq.5）都基于 w；性能分析章节（Section V-B2）用 αβ 模型推导最优 chunk 数与通信总成本下界。vault 中已有条目：NetMoE 用 αβ 建模三类通道（intra-device/intra-node NVLink/inter-node IB）取瓶颈 max；ASTRA-sim analytical backend 亦用线性成本模型（启动延迟+传输时间）评估通信。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
PipeComm 用 αβ 模型做流水线成本分析与 chunk 数优化：
```
数据 D 在一节点、切 C 个 chunk、II 启动间隔、R 个 root、S 步调度:
  每步成本 = α + D/(R·C)·β          # 单 chunk 传输: 启动延迟 + chunk/带宽
  总步数   = S + II·(C−1)            # 流水线: 前 S 步 + 每新 chunk 仅 II 步
  Cost     = (S + II·(C−1))·(α + D·β/(R·C))
  # 展开 = (S−II)α + D·II·β/R + (II·C·α + D·(S−II)·β/(R·C))
最优 chunk 数: 对 C 求导 → C* = sqrt( D·(S−II)·β / (α·R·II) )
  例: α=200ns, 1/β=50GB/s, D=16MB, R=3, S=10, II=2 → C*≈46
成本下界: Cost ≥ (S−II)α + D·II·β/R + 2·sqrt( D·(S−II)·II·α·β/R )
```
这使 PipeComm 能解析地确定最优分区（消除 Themis/TACOS 依赖经验 tiling 的不确定性），并用于判断流水线化在小消息下的收益边界。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：αβ 参数通过 ping-pong 测试（两节点间发不同大小消息线性拟合）或硬件规格获得（PipeComm 仿真配置直接给 α=200ns、1/β=50GB/s 等；异构配置给每维不同 α/β，如 0.2µs/50GB/s vs 0.05µs/200GB/s）；PipeComm 把它编码进 MILP（w=chunk/β）与调度（RT 按 w 占用相位）并用于 chunk 优化。使用场景：一切通信成本估计与算法选择（NCCL 的 eager/rendezvous 协议阈值、chunk 数选择、是否流水线化的判断）。局限：αβ 假设理想通信（无路由冲突/拥塞），实际性能偏离（NCCL 多通道并行、协议切换、ring/chunked 影响）；更精确模型有 LogP/LogGP/LogGPS。

涉及论文标题：
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
