## 龙门机器人（Gantry Robot / Cartesian Gantry，商品 3D 打印机）

术语解释
Cartesian 坐标型机器人，由正交导轨约束末端执行器沿三个相互垂直轴平移；R2D2 将其从 3D 打印/半导体制造场景"挪用"为数据中心网络物理重构执行器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 组成：末端执行器 End-effector（gripper/喷嘴/探针）+ 安装在移动托架上的正交导轨（Orthogonal guide rails）+ 线性驱动模块（Linear drive modules，步进电机/丝杠）——对应 3D 打印机的打印头、X/Y/Z 轴框架、电机驱动。
- 关键特性：正交结构在运动学上解耦各轴，天然支持笛卡尔运动、无需逆运动学，误差限制在单轴内不跨关节累积 → 低成本下实现亚 10µm 可重复定位精度（比同量程关节臂便宜 1-2 个数量级）。Web 证据：高精密龙门（直驱直线电机+气浮轴承+光栅编码器）可达 ±1µm 甚至亚微米重复度（Dover SAX ±1µm、Intellidrives ±5µm、气浮 PBA ±0.2-0.5µm），而 3D 打印机级别（Kyrus、Macron MCS-R6Y）重复度 0.1mm。
- 论文实测的商品 3D 打印机参数（Table II）：Creality Ender-3 V3 KE $279/350W/0.6m/s/0.1mm、Prusa MK4S $729/120W、Elegoo OrangeStorm Giga $2499/1524W/0.8×0.8×1.0m 构建体积。R2D2 用 rack-scale 的 OrangeStorm Giga（0.8×0.8×1.0m 包络覆盖 48.3cm 机架面）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- R2D2 unit 内的重构流程：系统控制器下达"断开端口 A、连接端口 B"→ 机器人控制器把命令翻译为 stepper 轨迹（G-code）→ gantry 将 gripper 移动到 retensioner 取 LC 光纤 → 2D 平面内运动到 patch panel 目标受体坐标 → 插入并 latch（自对准 gripper 容忍错位、传感器验证闩锁）→ 链路建立后 gripper 与链路解耦，机器人去执行下一任务。运动范围：512-compute/512-memory pod 对角最坏移动 111cm，300mm/s 下约 3.7s，含对准/插拔 <15s 完成一次重构。
- 与 OCS 对比：OCS 用微镜/压电执行器 ms 级切换但非闩锁（需持续供电稳定、更高插损）；R2D2 机器人秒级切换但链路闩锁后全被动零功耗。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现路径：直接改装商品 3D 打印机——原型用 Creality Ender-3 V3 KE，移除 filament feeder/extruder 换成 3D 打印 gripper（卡标准 LC 光连接器、自对准+闩锁传感，可用材料弹性替代 servo）、build plate 换成光学 patch panel 支架、重刷固件支持 G-code 直控 XYZ 步进电机；rack-scale 用 Elegoo OrangeStorm Giga。原型 500 次随机位置重配 100% 成功率、最坏 0.02mm 定位误差、空闲 <10W/峰值 30W。
- 用途：在成本敏感场景（3D 打印、PCB 组装、半导体制造）做高精度重复定位；R2D2 将其扩展为数据中心按需物理建链执行器。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
