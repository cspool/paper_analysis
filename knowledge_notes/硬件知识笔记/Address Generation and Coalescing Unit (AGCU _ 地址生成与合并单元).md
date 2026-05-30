## Address Generation and Coalescing Unit (AGCU / 地址生成与合并单元)

术语是什么？
Address Generation and Coalescing Unit (AGCU) 是 SN40L RDU tile 的可重构数据流桥接单元，连接片上 RDN 和片外 HBM/DDR/host memory/remote RDU。Tile 侧：AGCU 像普通 dataflow core 一样暴露 RDN vector/scalar/control 端口。TLN (Top Level Network) 侧：AGCU 生成读写请求、合并响应、提供地址转换层。AGCU 内含 scalar 地址生成流水线（类似 PMU 但无 SRAM）和 counters。关键功能：(1) Peer-to-Peer (P2P) 协议 — 支持跨 socket 的 RDU tile 之间直接流式传输数据（不经过 DDR/HBM）；(2) Kernel Launch Orchestration — 实现硬件级的 kernel 调度（Program Load → Argument Load → Kernel Execute），支持硬件 orchestrated 执行静态 kernel schedule。

从硬件架构角度拆解：
AGCU 在跨 socket 多 RDU 系统中的 role：
```
RDU Socket 0: Tile → AGCU → TLN → P2P interface → P2P link → RDU Socket 1: P2P interface → TLN → AGCU → Tile
```
P2P 通信使 collective operations（如 AllReduce）可融合进 dataflow kernel，类似片上 PCU-PMU 的操作融合，避免中间结果回 HBM。这使跨 socket 的 data/tensor/pipeline parallelism 与 socket 内部映射问题在编译器高层统一处理。

术语一般如何实现？如何使用？
AGCU 由编译器配置：为每个 off-chip tensor 分配 read/write stream；编程 AGCU 地址生成器；配置 P2P 路由（multi-RDU 场景）；选择 kernel launch 模式（SO/HO）。用户通过 PyTorch 模型代码 + 模型并行配置（如 TP=8）间接使用 AGCU 功能。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
