## 光路交换机（Optical Circuit Switch，OCS）

术语解释
在光域直接完成端口间物理连接的交换机，无需光电转换与逐包转发；R2D2 的 OCS baseline 为 3D MEMS OCS（论文引用 POLATIS 384 端口数据放大到 576/2304 端口，即 Google Jupiter Evolving 部署的光学电路交换技术）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- OCS 通过微镜阵列（MEMS）或压电执行器（Polatis DirectLight）把输入光纤的光束导向目标输出光纤，建立端到端光路；切换为电路级（ms 级），建立后数据透明直通、无逐包处理。Web 证据：POLATIS Series 7000 384×384 端口全光矩阵交换机，4RU、ms 级路径重配、SDN 使能（OpenFlow/NETCONF/RESTCONF），Google 为其 OCS 架构核心供应商（2025 年约 4600 台订单用于 TPU 集群）。
- 论文观点：OCS（RotorNet、ProjecToR、Firefly、Jupiter Evolving 等）仍静态供给 all-to-all、基于离线需求分析做固定 schedule 拓扑变化，且 3D MEMS 用非闩锁微镜、插损更高、需主动稳定；商用 OCS radix 有限（≤384 端口）需线性放大成本/功耗建模（最优情形）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 R2D2 评估中，OCS fabric 与 R2D2 fabric 同构（NIC 相同、OCS 替换 R2D2 单元）：512 节点用 2-OCS/4-OCS 576 端口配置（~$300K/台、~200W/台），2048 节点用 2304 端口（~$1.2M/台、~800W/台）。一次远端内存访问：compute NIC → OCS 光路 → memory NIC，单跳光路直通。
- 对比 R2D2：OCS 切换 ms 级（快于机器人秒级），但需持续供电维持光路（非闩锁）、插损更高、成本远高（576 端口 OCS ~$300K vs R2D2 unit $5.4K）；两者延迟/吞吐评估结果相同（都免逐包转发），但 R2D2 成本（512 节点 -24.6%~-34.5%）与故障隔离（控制面与数据面物理解耦）更优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MEMS 微镜阵列（Huber+Suhner POLATIS 系列、Calient）或压电 DirectLight；控制面 SDN 控制器下发交叉连接，数据面光路直通。使用：Google Jupiter 用 OCS 做集群内动态互连（TPU 训练流量），替代 spine 层降低功耗；论文将其作为"电路交换 baseline"与 R2D2 比较成本/功耗/性能。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
