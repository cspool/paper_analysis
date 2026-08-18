## STM32Cube.AI / ST Edge AI Core（边缘模型部署工具链）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STM32Cube.AI（现整合为 STM32Cube AI Studio / ST Edge AI Core 工具链）是意法半导体的模型转换与部署工具：输入预训练模型（Keras/TFLite/ONNX，支持 float32 与 int8 QDQ 量化格式，以及 QKeras/Larq 低位宽网络），输出针对 STM32 MCU 优化的 C 代码与工程；对带 Neural-ART NPU 的目标（STM32N6）做算子级映射——支持的算子下沉 NPU、不支持算子回落 Cortex-M55 CPU，自动生成内存布局与运行时调度。DESSCam 用它部署 Robust ViT：LSQ INT8 量化 → ONNX 导出 → 工具链切分（卷积/线性层在 Neural-ART NPU、LayerNorm/Softmax 等非线性在 Cortex-M55）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
训练模型（LSQ int8）-> ONNX（QDQ）-> [解析/图优化：算子融合、常量折叠、内存布局]
-> 算子级分派：NPU 支持集(CNN/GEMM 类) vs CPU(非线性/不支持算子)
-> 代码生成：NPU 微码序列 + C 库调用 + 内存分配
-> STM32CubeMX 工程集成 -> 烧录运行
```
DESSCam 的用法要点：Neural-ART NPU 无 transformer 原生加速，故依赖"计算量集中在早期卷积层"的模型设计——卷积生成稀疏 token 使 transformer 块计算量极小，整体仍高效（按官方 MobileNet v2 基线 6 mJ/inference 估算功耗/延迟）。工具链使用示例（ST 社区）：stedgeai analyze/generate --target stm32n6 --st-neural-art，配 --use-onnx-simplifier 处理图规约问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用流程：模型训练/量化（TFLite converter 或 ONNX Runtime int8 量化）→ 工具链 analyze（报告每算子 NPU/CPU 分配与内存）→ generate（生成 C 工程）→ 集成到 STM32CubeMX 应用。已知限制（社区报告）：ViT/自注意力模型转换常失败（Constant/LayerNormalization/Pow/ReduceMean 等算子不支持或需要清理），1×1 Conv/GEMM 存在不落入 NPU 的情况——需确认算子覆盖或在 CPU 上执行；适用于电池供电边缘设备（穿戴/AR 眼镜/传感器节点）的轻量模型部署。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
