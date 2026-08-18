## MHD / SHD（多/单头内存设备）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SHD（Single-Headed Device）只有一个上行接口接一台主机。MHD（Multi-Headed Device）在单个物理设备内集成多个逻辑 head，每个 head 有独立 CXL.io/CXL.mem 栈与独立地址空间，可各自连接不同主机——免除交换机即可多主机并行接入与容量池化。注意 MHD 的"池化"是容量聚合而非数据共享。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MHD 的硬件局限（本论文）：(1) head 间无缓存一致性，跨 head 访问需 NUMA 事务；共享需软件显式 cache flush 或非标准 tag/back-invalidation 机制（额外 tag 跟踪开销）；(2) 总带宽被活跃 head 均分，每主机带宽随 head 数下降（PCIe bifurcation）；(3) 端口数受 die 面积限制约 4。评估数据：4N1M_private（4 节点共享 512GB MHD）4 节点相对单节点 -35% 吞吐；YCSB-A 下 MHD 写流量集中于主节点 head、只读节点链路利用率 <6%；对比 4N4S_SWopt（PBR 交换机）达 >95% 带宽利用率与 4× 吞吐。NUMA 内 4 socket 各接一个 head（4N1M_private）可降 28% 延迟，但访问他人 head 仍需 inter-socket 通信。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
内存扩展器形态：每 head 一个控制器接口，用于免交换机多主机扩容（单节点内多 socket NUMA、多租户隔离）。规范示例配置最多 4 head。长期定位为过渡方案：交换机 PBR 池化在一致性、带宽与扩展性上全面替代之（本论文 64 节点近线性扩展 vs MHD 约 4 头上限）。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
