## PSum Router 与 Ring Network（psum 路由与环网互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSum Router 与 Ring Network 是 HiT 在 HSparse 模式下把部分和（psum）从"产生它的 Compute Row"路由到"应存储它的 Compute Row"进行归约的硬件互连。核心设计动机：外积数据流中 psum 需要在存储目标处合并，而乘法与累积被解耦（乘法可在任意 Row 进行、psum 可被路由到指定 Row），从而避免朴素外积的"psums 必须在同一 PE 内累积"限制。Ring network 是沿 y 轴连接 Cluster 内相邻 Compute Group 的轻量双向点对点环（每 Cluster 4 条 ring），每链接每周期在宽并行总线上传一个 psum+列索引对向量（带行元数据），总线宽度匹配累加器并行度。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程（图 8 示例，1 个 A 非零 + 4 个 B 非零流入 Compute Group）：PIDU 匹配分发 → 乘法器产出 psum → PSum Router 检查结果行索引：若该行的目标 C 行属于本 Compute Row 则送本地 DMAccum，否则经 ring 的 downlink/uplink 路由到目标 Row（示例中 psum 属 Compute Row 2 的 C 行，经 downlink 送出）。每个 Router 配 4 个 ring buffer 防拥塞：up/down 各 6、multiplier 来 4、DMAccum 转发 6；buffer 满则源 Row stall。buffer 大小是性能/成本权衡——加倍结构只降 10% 延迟却翻倍面积功耗。MSparse 下 ring network 与 PSum Router 被绕过（psum 立即在同一 Row 内累积），Router 时钟门控；D×D 下改走 systolic 垂直连接。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：点对点双向链接阵列（ring topology）而非 crossbar/多级网络——HiT 论证 ring 足够支撑 HS 模式（psums 经多条独立 ring 并行流动），避免了 Trapezoid 32×32 crossbar 的功耗与面积；缓冲用多级 FIFO（4 个 ring buffer）吸收突发。使用：仅 HSparse 模式激活（HS×HS/HS×MS/HS×D），因为只有 HS 输出稀疏、psums 路由量可控；MS 交叠率高、psums 量大，走 ring 会数据移动过大，故 MSparse 改为"同一 Row 内立即累积 + B 广播"。评估：Router/Buffer 在 HS 模式下几乎每周期活跃（图 18a），是支撑 HiT HS 高吞吐的关键组件。论文未开源。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
