## R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters

- 属于硬件架构的实现是什么？实验比较什么？
  - R2D2 硬件架构：switch-less、由商品 gantry 机器人（改装 3D 打印机）动态建立 compute/memory 节点间直接单跳光纤连接的 reconfigurable network fabric。单个 R2D2 unit = 封装机柜（encapsulation housing，防尘防振）+ 前后 48×96 密集光纤 patch panel（96 光纤接口/4.5U 机架单元，前后共 192 接口/U）+ cable retensioner（恒力弹簧卷收 slack 光纤，防缠绕/防碰撞）+ reconfiguring robot（gripper + gantry 导轨 + 线性驱动，2D 平面运动）。机器人从 retensioner 取光纤、对准 patch panel 目标 receptor、插入完成 latch；latch 后链路为全被动光通路，机器人可离开继续其他任务。
  - 单元角色：MoR（Middle-of-Row）unit 居中服务整行服务器（最大化行内连通，任意 compute 可连任意行内 memory）；Spine unit 用 MoR 剩余端口互连跨行（datacenter 级连通与负载均衡）。单 unit 高 radix 4608 端口（前后面板 48×48×2），可服务 27 个高密度机架/4536 服务器；2 层 MoR-Spine 可扩展至千万级服务器。多机器人可算法级并行重构（软件调度）。
  - 实验比较：与 SOTA fat-tree（NDP，[34]）和 OCS（3D MEMS，Google Jupiter [55] 部署技术；POLATIS 384 端口数据线性放大到 576/2304 端口，最优情形 baseline）比较硬件成本、功耗、Flow Completion Time（FCT）、吞吐；512/2048 节点（另含 8192 节点 3-tier、100G 512 节点）。结果：成本降最多 38%、能耗降最多 84%（vs fat-tree）；vs OCS 成本降 24-35%、功率降 1-6%；FCT 平均改善最多 43.3%、吞吐最多 70.2%（vs fat-tree），与 OCS 延迟/吞吐相同；应用性能相当或略优（+0.59% 平均）。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - ns-3 packet-level 网络模拟器（https://www.nsnam.org/）：模拟 400 GbE 网络带宽与延迟（500 ns per-hop 延迟 [71]），输出 FCT 与吞吐。
  - 自定义 discrete-event simulator（论文自研，无公开代码）：执行 R2D2 runtime 与联合分配-重构调度算法，测 allocation latency（含 4%/天机械故障注入）。
  - ASTRA-sim（https://github.com/astra-sim/astra-sim）：ML workload（LLaMA 4 Behemoth [2] 配置）应用级评估。
  - 物理原型：商品 3D 打印机 Creality Ender 3 V3 KE（$279）改装为 R2D2 原型；rack-scale 参数取自 Elegoo OrangeStorm Giga（0.8×0.8×1.0m 运动范围，$2499）。
  - 参数化 cost/power 模型：用原型成本分析与功率测量校准，放大到全规模部署（Table III：576 端口 unit $5.4K/118W、2304 端口 $13K、4608 端口 $23K）。

- 模拟器模拟什么的性能，修改了什么。
  - ns-3：模拟 R2D2/fat-tree/OCS 数据面，per-hop 500ns，输出平均/p99 FCT 与平均/p01 吞吐；论文未说明对 ns-3 的具体修改（采用 packet-level 模型）。
  - 自定义 DES：模拟系统控制器+机器人控制器异步执行联合调度（Alg.1），输出平均/p99 allocation latency；模拟 best-fit baseline（10-20× 更高延迟）、机器人并行度（37-45% 重配重叠 2+ 机器人）、4%/天故障率下恢复（延迟 +2.7%）。
  - 原型：替换 filament feeder/extruder 为自定义 3D 打印 gripper（卡 LC 光纤连接器、自对准+闩锁传感、无 servo 用 3D 打印材料弹性替代）、build plate 换成光学 patch panel 支架、重刷固件支持 G-code 直控 XYZ stepper。实测：500 次随机位置重配 100% 连接成功率、最坏 0.02mm 定位误差（与 spec 一致）、端到端重配延迟（与 §III-A 估算匹配，含运动/对准/插拔/线缆管理 <15s）、空闲 <10W/峰值 30W 功耗（AC 功率计测量）。
  - 修改点总结：硬件层（gripper 自对准、集成闩锁传感器、可选 mini servo 释放）、固件层（G-code 直控）、软件层（Alg.1 联合分配+重构、异步分发、fault handling）。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源情况：论文未提供 R2D2 代码/仓库链接，联网搜索未能确认公开实现（ISCA 2026，UPenn ESE/CIS）。输入数据公开：Gao et al. [25]（OSDI'16）disaggregated memory traffic traces（bdb/GraphLab/Memcached/Terasort Hadoop/Terasort Spark，按 Shoal [62] 方法缩放）、Protean VM allocation traces（[32]，2064 机器）、ASTRA-sim 开源、ns-3 开源、3D 打印机与 FS.com 组件为商品件。
  - 使用例子（复现 ns-3 网络微基准）：①配置 512/2048 节点拓扑（R2D2 直连 / NDP fat-tree / OCS）；②注入 Gao et al. 内存流量 trace（每条目=资源访问流：请求时间+流大小，去存储流量）；③ns-3 packet-level 模拟，400 GbE，per-hop 500ns；④输出 FCT（平均/p99）与吞吐（平均/p01）。
  - 模拟器输入到性能输出全过程：输入=拓扑+流量 trace+链路参数（400 GbE、每跳 500ns）→ ns-3 逐包模拟转发/排队/拥塞（fat-tree 多跳多流竞争、incast 碰撞；R2D2 单跳直连无中间转发、无共享带宽瓶颈）→ 输出流完成时间与吞吐。自定义 DES：输入=VM 分配请求流（Protean trace，含 1000s 请求/秒 burst）+ R2D2 fabric 状态（已建链路、空闲机器人列表）→ 模拟 Alg.1（选 row→选 no-reconfig 计算节点→必要时按可用机器人列表并行 reconfiguration、异步不阻塞）→ 输出 allocation latency（avg/p99）与机器人并行度。作用：量化 R2D2 在成本/功耗/延迟/吞吐/分配延迟上的优势，并评估机械故障影响（4%/天故障率下分配延迟仅 +2.7%、不影响已分配应用性能）。
