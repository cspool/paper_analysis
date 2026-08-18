## VCS（Virtual CXL Switch）与 Dynamic Port binding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VCS 把一个物理 CXL 交换机划分为隔离的虚拟根域：单根 VCS 一个上游端口（USP）接主机、多个下游端口（DSP）接设备；多根 VCS 多个 USP 各属不同主机共存于同一交换机，各虚拟根保持独立地址空间。带宽与地址空间分配由硬件 Dynamic Port（DP）绑定机制管理：把虚拟端口映射到物理下行端口，使主机在硬件层分配或共享 MHD 资源。本论文扩展支持 MHD/多逻辑设备：交换机把 MHD 每个 head 当作独立逻辑端点，跨主机共享时保持排序与一致性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
多根 VCS 数据路径：主机 A/B 各经自己 USP 进入交换机 → 地址空间隔离在各虚拟根域 → 共享内存池经 DP 绑定映射到物理 DSP（硬件分配/共享 MHD head）→ CXL.cache 一致性跨根域维护（CacheID/BI-ID 与 SPID/DPID 映射）→ 各虚拟根延迟一致。DP 绑定实现硬件级可组合性（composability）：无需软件中介即可并发访问共享内存池，隔离与确定性延迟跨域保持。系统侧利用：修改版 PostgreSQL 17 借助硬件一致性实现跨节点并发写与跨节点缓存复用，替代单主节点串行写。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VCS 是 CXL 交换机的标准化虚拟化机制，由 FM 配置域划分；DP 绑定以硬件表实现（论文未给出表结构细节）。使用场景：多租户隔离、多主机内存池化/共享、MHD 接入的排序与一致性维护。与 PCIe SR-IOV 思路类似但作用于交换/内存域，且本论文实现为无固件参与的纯硬件路径。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
