## On-Chip Ring Network for NMP (近存计算片上环形网络)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
On-Chip Ring Network 是 Stratum NMP logic die 上连接 16 个 Processing Units (PUs) 的 bidirectional ring-based interconnect 网络，用于支持 MoE 推理中的 collective 通信模式（all-gather, reduce-scatter, scalar exchange），避免 centralized global buffer 和 crossbar 带来的 scalability 和 physical design 复杂度。每个 ring link 提供 128 GB/s 带宽，aggregated ring bandwidth = 2.048 TB/s。Ring router 内含 local switch（数据路由）+ aggregator（in-situ data reduction：incoming data streams 可直接在 router 内累加，无需经过 shared memory）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Ring network 在 NMP 执行中的三种关键通信模式：
1. **All-Gather**（Expert Processing）：xPU 将 input token matrix 分片发送到各 DRAM channel → sub-ring all-gather 复制完整矩阵到所有 PUs → 每个 PU 获得完整 input 以便 tensor-parallel expert 计算。
2. **Reduce-Scatter**（Expert Processing）：GeMM3 后各 PU 产生 partial output → ring-based reduce-scatter 沿 ring 传递并累加 partial sums → 各 PU 获得最终 output 的一个分片 → 可与下一 expert 的 GeMM1 并行（overlap optimization）。
3. **Scalar Exchange**（Attention Processing）：Softmax 需要 global max 和 global sum → 每 PU 独立计算 local max/sum → ring 上仅传输标量值交换（极小带宽需求） → 各 PU 独立完成最终 Softmax 归一化。

Ring topology 相比 crossbar 的优势：(1) 物理设计简单——每个 PU 仅连接 2 个 neighbor，routing 逻辑轻量；(2) scalability——增加 PU 数量不增加 per-PU 互联复杂度；(3) reduced wiring area。缺点：all-gather 需要 log 步，latency 高于 crossbar，但 Stratum 通过通信-计算 overlap 隐藏了这部分延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Ring network 在 Stratum 中通过 per-PU ring router 实现：(1) 每个 router 有 2 个 bidirectional ports（prev 和 next PU）；(2) aggregator 在数据流经时执行 element-wise add（用于 reduce-scatter 的 in-flight reduction，无需先存后算）；(3) 仅 NMP mode 激活（regular memory mode 下 inactive 以省电）。Stratum 的 ring network 设计借鉴了 systolic array 的 data stationary 思想和 Google TPU 的 ring-based reduce 设计。实现复杂度：ring router area 占 logic die 的较小比例（未单独 breakdown，但参考类似设计，ring interconnect 通常 <5% die area）。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
