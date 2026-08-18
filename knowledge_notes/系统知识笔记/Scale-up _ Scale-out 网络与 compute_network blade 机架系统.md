## Scale-up / Scale-out 网络与 compute/network blade 机架系统

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MTIA 300 训练系统（ISCA'26）的两级网络与刀片式机架组织：(1) **Scale-up 网络**——每加速器 800 GB/s（可选至 1000 GB/s），16 节点域（一 rack），用于域内高带宽通信（对比 H100 8 加速器、450 GB/s）；(2) **Scale-out 网络**——200 GB/s，第一级 4096 节点域（可选第二级交换机扩到 16K+ 节点），采用 disaggregated scheduled fabric（packet spray 避免 hot link + 保序投递 + fabric 级端到端 credit 可靠性）。机架：MTIA training chassis 含 16 个 compute blade 槽 + 6 个 network blade 槽（cable backplane，compute blade 垂直放置缩小背板）；**compute blade** = 单 CPU（512 GB RAM）+ 单 MTIA 300（1:1 映射，避免 PCIe 争用、容纳不可预测 CPU 需求）；**network blade** 分两种——scale-up blade（低延迟低功耗 ASIC，多 rack 组合成大域）与 scale-out blade；每 compute blade 200 GB/s I/O，均液冷。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
一次 40 卡 DLRM 训练迭代的网络流程：40 个 MTIA 300 分布在多个 rack（每 rack 16 节点 scale-up 域）→ 域内 AllReduce（1.6 GB）走 scale-up 800 GB/s（16 节点内高带宽）+ 跨域通信走 scale-out 200 GB/s（AllToAllv 1 KB-1 GB 可变消息）→ 12 个内置 RDMA NIC 按需切分给 scale-up/scale-out 域（网络 bytes-to-FLOPS >5× H100）→ 数据经 RoCE 网络 chiplet（112G SerDes）进出。整体通信性能超 H100 3.9×（16 节点域 + 2.2× scale-up 带宽是 16+ 卡大消息占优的主因）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：scale-up blade 用 ASIC（低延迟低功耗）、scale-out blade 支持 packet spray 调度 fabric；两种 blade 均可按 scale-up/scale-out 设置配置数量；liquid cooling。使用场景：DLRM 训练（16 节点域规模匹配、扁平网络最高 1.2 TB/s）+ LLM 推理 8 卡配置（TP/EP）。对比 GPU：H100 scale-up 8 节点 450 GB/s、scale-out 50 GB/s。信息缺口：论文未给出 fabric switch ASIC 的规格与 packet spray 的具体算法。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
