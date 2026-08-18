## Motion-to-Sound Latency（运动到声音延迟，T_m-s）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Motion-to-sound latency 是 VR 空间声化的端到端关键延迟：从用户头部运动发生到耳机输出反映该运动的声音之间的时间差（类比图形学的 motion-to-photon latency）。人类感知研究表明该延迟须低于 50-60ms 才能保持沉浸感（[13]、[108]）。ECHO 论文把 SS 流水线建模为传感（T_S：相机 S^M 与 IMU S^I 捕获+传输）→ 位姿估计（T_P）→ 声传播+BRIR 生成（T_R1）→ 可听化（T_R2）→ 音频输出（T_O），加上传感器捕获间隔 T_IN，得到 T_m-s = T_IN + T_S + T_P + T_R1 + T_R2 + T_O。各阶段可流水化提升稳态吞吐，但流水不缩短端到端关键路径。
- 从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - ECHO 用 T_m-s 作为系统级优化目标与评估指标。建模细节：T_IN 固定 5ms（Hybrid 模式 100Hz 更新率、10ms 周期内采样延迟 [0,10]ms 的期望）；T_S^MI 按 AEA 文档设相机/IMU 传感延迟 + MIPI CSI 接口的传感器到 SoC 传输延迟模型；T_O=1ms 音频输出 + 5ms 播放缓冲（音频渲染与播放异步，已渲染块最多等一个缓冲周期）；T_P^MI、T_R1、T_R2 按实际执行位置测量求和。ECHO 加速器 Hybrid 模式在 MI（单目+IMU SLAM）与 IO（IMU-RNN）间交错：T_m-s = max(T_m-s^MI, T_m-s^IO)，因 T_S^MI+T_P^MI 远大于 IO 模式，整体由 MI 主导。系统级调度例子（Fig.11a）：相机捕获→CPU 建金字塔切 cell→加速器 ORB 检测→CPU 点过滤→加速器投影/Jacobian→CPU Gauss-Newton→CPU 声学注视聚类→GPU 渲染双耳音频→DAC 输出，九阶段流水协同，最终平均 39.2ms（<50ms）、256 源 <60ms。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：作为延迟分解度量，把端到端延迟拆成组件级可测/可建模部分再求和（ECHO 的做法）；或直接端到端实测（用户研究 + 硬件测量）。用于评估系统优化效果：ECHO 在九种配置（Table VII：Jet ORB+Full CPU/GPU、Jet {OKVIS,HybVIO,VINS,ORB}+Foveated GPU、Jet ECHO+Foveated GPU、ECHO+Full GPU、ECHO）上对比，分离算法与硬件贡献——声学注视 1.29×（vs Full GPU）、ECHO 算法 1.28×（vs ORB-SLAM3）、加速器 1.41×（vs Jet ECHO），合计相对 Jet ORB+Full GPU 在 256 源时 AEA 2.91×/TUM VI 2.79×。使用注意：该指标同时含算法、系统（异步缓冲）与硬件（卸载）因素，优化需跨层协同（ECHO 的算法-硬件 co-design 正是为此）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality
