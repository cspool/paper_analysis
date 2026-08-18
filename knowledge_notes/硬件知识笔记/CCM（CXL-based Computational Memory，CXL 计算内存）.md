## CCM（CXL-based Computational Memory，CXL 计算内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CCM 是在 CXL Type 3 内存扩展设备之上叠加计算资源（近内存处理 PNM 单元）形成的新兴设备：既可作为内存被主机 load/store 访问（CXL.mem），又可作为设备执行卸载的计算任务。计算能力弱（低频处理单元、无大缓存），但设备本地内存带宽/容量高，因此主要用途是 PNM——把应用中的内存密集操作（KNN 向量距离、图遍历、OLAP 过滤、LLM attention、DLRM 嵌入查找/SLS 等）部分卸载到设备本地执行，避免数据经 CXL 链路搬回主机。CCM 具有"二象性"（duality）：设备中心视角（device-centric，当作加速器走 CXL.io 邮箱）与内存中心视角（memory-centric，当作内存走 CXL.mem），两种视角衍生出不同的卸载机制与性能权衡。行业原型：SK hynix 定制 add-in card（Xilinx Versal VP1502 FPGA + CXL 内存控制器 + PFL 硬核 + Cortex-A72），另有 UPMEM/CXL 类研究原型。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AXLE 论文的 CCM 架构（基于 M²NDP）：主机对远端特定地址发 CXL.mem store → CCL 内存控制器上的自定义 packet filter 区分"普通访存"与"kernel 启动"（区分地址范围/命令编码）→ CCM 调度器把任务划分为固定大小输入向量分派给各处理单元的 µthreads（AXLE 配置 2GHz、16 处理单元、每单元 16 µthreads、DDR5-4800 16 通道）→ 结果写入设备内存 → AXLE 的 DMA executor 经 CXL.io 反流给主机。AXLE 论文指出固定卸载划分不一定最优：数据搬运量与主机处理量随负载变化，瓶颈可能在主机（KNN 低维多行，主机占 64.67% 时间）或数据搬运（PageRank 数据搬运占 47.77%），这是设计 CCM 卸载机制必须考虑的架构约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现路线：ASIC/FPGA 把 PNM 引擎做进 CXL 控制器（PFL 硬核 IP 或通用核），设备固件处理卸载请求与结果监控；主机侧内核驱动管理 DMA 区域（预 pin + scatter-gather 描述符影子化）。使用方式：在应用内部分卸载内存密集操作（Table I 场景），把数据搬运量从 {#edge×#vertex} 降到 {#vertex} 量级；局限是跨迭代依赖与同步结果加载导致主机/CCM 交替空闲（AXLE 的动机）。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
