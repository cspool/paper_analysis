## Parameter Server（PS，参数服务器）与 MiSDP（Model-in-Server DP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PS 架构 = 集中式键值服务器存模型参数：worker 算梯度后 push 给 PS，PS 聚合并更新参数，worker 再 pull 最新参数；源自 Li et al. OSDI'14（Web 证据：CMU 参数服务器工作）。MiSDP = PS 用于数据并行的变体：每 worker 每层 1 push（梯度）+ 2 pull（前/反向各一次参数）。
- 流量对比：MSDP 每层 2×AG+1×RS 每方向 3(N-1)S/N；MiSDP 每方向 2S 收 + S 发，流量最多减 1/3。
- 三个痛点：1) 需大量 CPU 机器做 PS——16 worker 线速聚合需 29 台非 colocated PS（或 13 台 colocated+非 colocated 混合），因每台服务器内存子系统压力大；2) 传统 MiSDP 把整模型梯度 push 后再 pull 整模型参数，GPU 需装整个模型，训不了 100B 模型；3) GPU 直连 NIC（IBGDA/GPUDirect Async）路径仍有算通干扰。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运行流程（MiSDP 一层反向）：worker GPU 算 partial gradient → push 到 PS → PS 累加 N 路梯度 → Adam 更新 → 前向/反向时 worker pull 参数。请求路径经过网络往返，PS 是汇聚点（incast）与算力热点。
- DisDP 的重构：SmartSwitch 在网内把 N 路部分梯度聚合为 1 份 → PS 只收 1 份聚合梯度；参数反向由 SmartSwitch 广播（one-to-many）。PS 侧 step-centric 流水把 SSD 读、CPU Adam、网卡收发深度流水，单台 PS（双 Xeon Gold 5320 + 12 SSD）即可线速服务任意数量 worker；配合 many-to-one 心跳聚合可靠协议，PS 确认包数与 worker 数无关。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：参数按 key 哈希环分片、BSP/SSP/异步一致性、梯度压缩与显著梯度过滤（Li et al.）。使用场景：稀疏模型（推荐系统）仍广泛用 PS；LLM 稠密训练以 MSDP 为主，DisDP 证明「单 PS + 网内聚合」可把 PS 架构带回 LLM 训练（流量减 1/3 + 单机 PS）。信息缺口：论文未说明 PS 侧参数分片（跨多 PS 扩展仅靠多机架 SmartSwitch 层级）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
