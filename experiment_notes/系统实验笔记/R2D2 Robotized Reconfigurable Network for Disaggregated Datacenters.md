## R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters

- 属于Serving调度的实现是什么？实验比较什么？
  - R2D2 software runtime 的联合任务分配与网络重构调度（近似匹配本层：非请求级 serving 框架，而是 datacenter VM/任务分配调度）。两层设计：系统控制器（datacenter orchestration 层，全局资源管理）+ 机器人控制器（嵌入式，低层运动规划/闭环控制/故障处理）。核心是 Joint Allocation and Reconfiguration 算法（Alg.1）：两阶段分层（先选 datacenter row，再选 row 内 compute-memory 节点）；优先复用已建立链路避免 reconfiguration、fitness 函数考虑资源匹配与链路利用率、必要时允许 reconfiguration 并按实时空闲机器人列表并行分发（异步不阻塞，避免系统控制器串行化瓶颈）；重配成本计入分配决策以主动工程化流量稀疏/稳定。
  - 实验比较：与 best-fit baseline（云常用策略，同跑 R2D2 硬件）——联合算法平均与 p99 allocation latency 低 10-20×（best-fit 机器人无关、触发过量重配级联延迟）；与 fat-tree/OCS 相比 allocation latency 高 41-51%(avg)/27-30%(p99)（512 节点）、5-39%/21-29%（2048 节点 + spine），但仅增加 VM 总运行时间的 0.49%；利用率达 99% CPU / 69% memory；机器人并行度 37-45% 重配重叠 2+ 机器人（2/4 robot 配置）。

- 硬件平台是什么，配置是什么。
  - 512/2048 节点、400 GbE；Broadcom P1400GD 400Gb NIC（$2198）；R2D2 fabric 为 2-robot 或 4-robot 配置（400G 拆 2×200G / 4×100G breakout transceivers），每 R2D2 unit 576/2304 端口、120W。分配延迟用自定义 discrete-event simulator 在 R2D2 runtime 上测量。VM trace：Protean [32] 生产集群 trace（2064 机器、48 核/384GB 每机，每次请求含 VM CPU/内存/存储需求与时长，排除存储与应用网络需求，含 1000s 请求/秒 burst）。

- 开源Serving框架是什么。修改了什么。
  - 论文未修改开源 serving 框架（非 LLM serving 论文）：R2D2 runtime 是自研两层调度系统，通过标准 API（兼容 Azure Protean rules [32]）暴露给上层 orchestrator（hypervisor/scheduler），上层无需修改编排逻辑即可请求资源分配与网络配置。最接近"多请求调度"的层面是 VM/任务分配调度，而非请求级 serving——近似匹配本层次。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：R2D2 runtime 未提供公开仓库（联网搜索无法确认）；输入 trace 公开：Protean VM allocation traces [32]、Gao et al. [25]/Shoal [62] 内存流量 trace。
  - 使用例子（一次 VM 分配）：hypervisor 通过标准 API 提交任务（compute 需求 C + memory 需求 M）→ 系统控制器执行 Alg.1：①FEASIBLEROWS(C,M) 按 best-fit fitness 选最高分行；②先试 row 内已连接（无需重配）的 compute 节点（FEASIBLECNODESNORECONF 排序、COMMIT 校验资源并绑定，成功即返回）；③无则枚举可行 compute-memory 对，对每对查 AVAILABLEROBOTS 实时空闲列表，向机器人控制器分发"断开端口 A、连接端口 B"命令（多机器人可并行），RECONFIGURENETWORKBYROBOT 成功后 COMMIT 并返回节点分配；④失败标记机器人 down、换下一候选；全部失败则 QUEUE FOR RETRY。
  - 输入到硬件执行全过程：输入=VM 分配请求流（Protean trace，含 1000s 请求/秒 burst）→ 系统控制器联合调度（异步分发、不阻塞后续分配）→ 机器人控制器（嵌入式板）把高层命令翻译成 stepper 轨迹/G-code，闭环控制（编码器位置、插入力反馈）执行物理插拔 → 光纤 latch 后链路变被动直连，数据面由 NIC 直通跑 400 GbE（无逐包交换）→ 完成通知回系统控制器，hypervisor 调度 VM 运行。作用：联合优化任务放置与物理拓扑，促进流量稀疏/稳定、最小化机器人移动，使分配延迟增量仅占 VM 运行时间 0.49%；故障时（机器人 down、重配失败）快速 failover（备选机器人/备选放置），已分配应用性能不受影响。
