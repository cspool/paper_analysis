## R2D2 单元（R2D2 Unit：patch panel / cable retensioner / gripper / MoR-Spine）

术语解释
R2D2 硬件架构的基本构建块：一个独立数据中心机架单元，由封装机柜、前后 48×96 光纤 patch panel、cable retensioner 和 gantry 重构机器人组成；以 MoR（Middle-of-Row）或 Spine 角色组网。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 四大组件：(1) Encapsulation housing——标准机架机箱，密封连接器防气流污染、衰减服务器振动；(2) 前后 patch panel——48×96 密集受体端口矩形阵列，外部端口接服务器 NIC、内部端口供机器人访问，布局与机器人运动平面协同设计（数据中心现有光纤面板可 96 连接器/4.5U，前后共 192/U）；(3) Cable retensioner——恒力弹簧卷收机构，收纳 slack 光纤、重构时完全收回断开电缆防缠绕；(4) Reconfiguring robot——增强版商品 gantry（gripper+gantry 导轨+线性驱动），在 2D 平面内取线、对准、插入完成建链。
- 高 radix：前后面板 48×48×2=4608 端口，42U 机架内可服务 27 个高密度机架（4 服务器/U、168 服务器/机架）=4536 服务器。
- 角色：MoR 单元居中于服务器行、提供行内任意 compute-memory 直连（集中式优于分布式 ToR）；Spine 单元互连 MoR、跨行提供 datacenter 级连通与负载均衡。两角色硬件相同。2 层 MoR-Spine 可扩展到千万级服务器。
- 成本/功耗（Table III）：576 端口 $5.4K/118W、2304 端口 $13K、4608 端口 $23K；空闲 <10W、重构峰值 ~120W。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一次跨行内存访问：compute 节点 NIC → 所在行 MoR 单元前端受体 → MoR 内部被动光纤 → MoR 后端口 → Spine 单元 → 目标行 MoR → 目标 memory 节点。全部为机器人预先建立的单跳（MoR-Spine-MoR 两段）被动直连，无逐包交换、无共享瓶颈。
- 机架空间核算：512 compute+512 memory pod 需 512×4=2048 条 compute-memory 链路=4096 连接点，/192 端口每 U×4.5cm≈96cm≈21.3U（占 48U 机架一半，留 2U 给电源/控制器/机器人 home dock）；1024+1024 pod 约 42.6U。因为端口与机架空间随 fanout 线性增长，单个低成本商品 3D 打印机即可覆盖整个 fabric。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：全商品件组装——42U 机架（$307）、96 光纤 patch panel（$109）、3m 光纤跳线（$6.5）、Elegoo OrangeStorm Giga 3D 打印机（$2499，含 X/Y/Z 步进电机 36W×3、控制器板 5W、电源 5W）、gripper（~$0，3D 打印）。多机器人单元天然算法级并行重构（软件异步分发）。
- 使用：MoR 部署在每行中央最小化线缆，Spine 用 MoR 剩余端口做跨行；故障时（机器人 down/控制器失联）由系统控制器排除该单元、备选机器人接管，被动闩锁链路不受影响。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
