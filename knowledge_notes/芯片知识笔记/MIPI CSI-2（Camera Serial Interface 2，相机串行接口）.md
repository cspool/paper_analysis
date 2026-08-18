## MIPI CSI-2（Camera Serial Interface 2，相机串行接口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MIPI CSI-2 是 MIPI 联盟定义的相机传感器到应用处理器（SoC/NPU）的串行接口标准：物理层为 D-PHY 或 C-PHY（差分管脚/三相编码），含单向高速数据通道（差分 lane，每 lane 百 Mbps–数 Gbps）与双向低速控制通道（CCI/I2C）；以包（packet）为单位传输图像/事件数据。DESSCam 用它把传感器 AER 包传给 host NPU：假设有效带宽 2.5 Gbps（与 BlissCam 对比对齐）、能量成本约 100 pJ/byte（ISSCC 2022 的 AR 图像传感器分析值）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
DESSCam 输出链路：输出 FIFO -> MIPI CSI-2 PHY -> host NPU
MIPI 功耗 = 平均数据率(激活 patch 率 x AER 包大小) x 100 pJ/byte
MIPI 延迟 = 数据量(AER 包大小 x 每帧激活 patch 数) / 2.5 Gbps
```
CSI-2 是芯片到芯片的物理层：事件驱动传感器下数据率随眼动速度动态变化（13.56–5,347.56 Hz 等效帧率 × 每帧 12 patch），接口功耗与延迟因此自适应；ESS/PAC 把每帧输出压缩到 12 个 patch（约 70 B/包），使 2.5 Gbps 带宽下接口延迟进入微秒级、无需高带宽物理层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：传感器侧 D-PHY TX + 控制器、SoC 侧 RX + CSI 主机控制器（商用 CIS/DVS 标配，DAVIS346 等经 USB 转换暴露）；版本演进 CSI-2 v2–v4 提升每 lane 速率（v4 达 6 Gbps/lane 以上）。使用方式：移动/AR 相机的标准接口；系统设计要点：带宽选择与输出数据量匹配（稀疏事件数据可用低带宽 lane 数）、能量成本按 ~100 pJ/byte 量级估算（DESSCam 沿用该假设评估系统功耗）。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
