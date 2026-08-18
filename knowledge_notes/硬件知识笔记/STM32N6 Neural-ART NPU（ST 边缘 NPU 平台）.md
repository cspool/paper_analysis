## STM32N6 Neural-ART NPU（ST 边缘 NPU 平台）

术语解释
STM32N6 是意法半导体 16 nm 工艺的边缘 MCU：Arm Cortex-M55 内核 + Neural-ART NPU（约 600 GOPS、3 TOPS/W），面向边缘 AI 推理。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Neural-ART 是 ST 自研 NPU：原生支持 INT8 卷积类算子（CNN 为主），片内 4.2 MB RAM；执行 MobileNet v2 基准约 6 mJ/inference。对 transformer 类模型支持受限（ST 社区报告 ViT/自注意力模型转换失败或部分层不支持），因此 transformer 块的 LayerNorm/Softmax/多头注意力等非线性算子需回落 Cortex-M55 或改造模型。DESSCam 用它作为 off-sensor gaze 估计处理器：Robust ViT 计算量集中在早期卷积层，恰好匹配 NPU 的 CNN 优势。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
DESSCam 部署数据流：
传感器 AER 包 --MIPI CSI-2--> STM32N6
  -> 卷积/线性层（INT8，LSQ 量化）: Neural-ART NPU 执行
  -> LayerNorm/Softmax 等非线性: Cortex-M55 执行
  -> 输出 gaze 坐标
```
性能估算：NPU 功耗 = MAC 数/秒 ÷ 能效（按 STM32N6x7 官方 MobileNet v2 基线外推）；NPU 延迟随激活 patch 数（token 数）变化，12 patch 触发一次推理。ESSCam 因 ESS/PAC 把 patch 数压到 12，NPU 负载极低；论文指出未来可把专用 NPU 直接集成到 3D 堆叠传感器底层（near-sensor 计算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工具链：STM32Cube.AI / ST Edge AI Core 完成模型转换（Keras/TFLite/ONNX → C 代码）与算子级 NPU/CPU 分派（见编译框架层 STM32Cube.AI 条目）；使用注意：只支持 int8 量化格式时需先做 QAT/PTQ，transformer 模型需确认算子覆盖（社区建议 --use-onnx-simplifier、去除不支持的 Constant/LayerNormalization/Pow 等算子）。适用场景：电池供电的轻量边缘推理（AR 眼镜、穿戴设备）。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
