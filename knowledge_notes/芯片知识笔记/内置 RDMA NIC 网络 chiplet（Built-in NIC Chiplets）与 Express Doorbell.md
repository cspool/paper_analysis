## 内置 RDMA NIC 网络 chiplet（Built-in NIC Chiplets）与 Express Doorbell

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
内置 RDMA NIC 网络 chiplet 是 MTIA 300（ISCA'26，Meta 首款训练芯片）把 RDMA 网络接口控制器（NIC，基于第三方商用 RDMA IP 定制）直接集成进芯片封装的技术：2 个网络 chiplet（25.6×9.3 mm）各含 6 个定制 800 Gbps（100 GB/s）RoCE RDMA NIC IP，共 12 NIC、1.2 TB/s I/O，经 die-to-die 接口 + 112G SerDes 与 compute chiplet 直连。其目的是消除传统"加速器经 PCIe 外挂 NIC"数据通路的开销（PCIe 协议开销 + 主机 CPU 参与 work submission/CQ 处理——1.2 TB/s IO 若由 host 管理会耗尽 host 核），且 NIC 可灵活用于 scale-up（800 GB/s、16 节点域）或 scale-out（200 GB/s、4096 节点）网络。网络 chiplet 的 NIC 做了四项定制：(1) **Express Doorbells**——用 work request（WR）本身作为 doorbell 写，避免额外的 HBM ring-buffer 读（每事务省约 800 ns），每 IP 支持 24576 个 outstanding WR / 1024 QP；(2) **移除 QP caching**——QP cache 占芯片面积大，去掉后每 NIC 限制 1100 active QP（远高于实际几百 rank 需求）；(3) **简化包处理流水线**——去掉 virtual switching、TC offload（如 cls_flower）等不需要的特性；(4) **AXI steering tag**——自定义 steering tag 使不同流量类型走 compute chiplet 的独立 cache partition。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MTIA 300 芯片数据路径中：collective 数据（如 AllReduce ring 的梯度分片）由 Message Engine（ME）的 NIC interface 收 work request（WR）进单 FIFO，分发给 12 个 NIC 的 express doorbell（WR 即 doorbell 写、无需读 HBM ring buffer），NIC 经 112G SerDes + die-to-die 接口把数据发出/接收。HCCL 库用 RDMA verbs（ibv_create_qp/ibv_modify_qp）创建 QP 并映射到 express doorbell，因无硬件 QP caching，12 NIC 提供 13056 个 QP 可切分/共享于 scale-up/scale-out 域。chiplet 级意义：网络功能从"板卡外设"变为"芯片内 IP 块"，使单颗 MTIA 300 包内同时具备 216 GB HBM3E、72 PE、16 ME 与 1.2 TB/s 网络 I/O，是"训练芯片内嵌网络"的首例（对比 TPU 稀疏核的非 RDMA torus 专用网络）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于商用 RDMA NIC IP（论文未给厂商），定制四项（express doorbell/去 QP cache/简化流水线/steering tag），经 die-to-die（112G SerDes）与 compute chiplet 互联；软件侧 HCCL 以 RDMA verbs 管理控制路径（QP 创建/状态迁移/异步事件），以 MTIA streaming interface 提交数据路径。使用场景：DLRM 训练（AllReduce/AllToAllv/AllGather 频繁、消息 1 KB-1 GB，通信性能超 H100 3.9×）与 LLM 推理（DeepSeek-R1 8 卡 TP/EP 分片）。局限/信息缺口：论文未说明 NIC IP 供应商、流片面积/功耗分解、以及 12 NIC 与 scale-up/scale-out 网络 blade 的具体映射细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
