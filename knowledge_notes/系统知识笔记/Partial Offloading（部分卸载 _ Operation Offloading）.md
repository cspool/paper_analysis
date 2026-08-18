## Partial Offloading（部分卸载 / Operation Offloading）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Partial Offloading 指不把整个应用搬到 CCM/加速器，而只把应用中的内存密集操作部分卸载到近内存设备执行：主机执行其余计算，主机与 CCM 通过 CXL 协议交换数据与命令。典型卸载目标（AXLE Table I）：OLAP/OLTP 的过滤（SELECT 内谓词）、图分析的边遍历+顶点更新、KNN/ANN 的向量距离计算、LLM 推理的 attention 块、DLRM 的 embedding 表查找+SLS。收益来源：结果只需搬 {#vertex}/{#rows} 量级数据而非 {#edge×#vertex} 量级原始数据。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PageRank 迭代的部分卸载例子（AXLE）：第 t 轮——主机把邻接表与当前顶点值留在设备内存 → 主机经卸载协议启动 CCM kernel（边遍历+顶点值更新）→ CCM 在本地内存上执行 → 更新后的顶点值经卸载协议回传 → 主机据其结果计算新 frontier/PageRank 值、准备下一轮。关键观察（Observation #2）：固定卸载划分不保证端到端最优——同一操作随输入规模与粒度变化，瓶颈会转移到主机处理（KNN 低维多行，主机时间占 64.67%）或数据搬运（PageRank 搬运占 47.77%），因此卸载机制（how）与卸载内容（which）同等重要。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：应用内插入卸载调用点（kernel 描述符 + 输入/输出缓冲区），由 CCM 设备驱动与固件完成远端启动；卸载机制有 RP（CXL.io 邮箱）、BS（M²NDP CXL.mem store）、AXLE 异步背流三种。使用方式：图分析/向量检索/OLAP/LLM 推理/DLRM 等内存密集负载加速；配套主机侧并行任务调度器（[19][14][18][17][33]）时需接口隔离（AXLE 的 OoO 流式为此设计）。局限：跨迭代数据依赖使主机与 CCM 交替空闲，需流水重叠机制弥补。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
