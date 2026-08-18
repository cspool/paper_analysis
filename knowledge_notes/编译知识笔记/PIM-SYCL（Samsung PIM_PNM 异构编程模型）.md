## PIM-SYCL（Samsung PIM/PNM 异构编程模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM-SYCL 是 Samsung 面向其 PIM/PNM 硬件（HBM-PIM、LPDDR5X-PIM 等）的异构编程模型：基于 Khronos SYCL（单源 C++ 异构编程标准，host/device 代码同源），为 PIM 设备提供类 CUDA 的 kernel 编写与调度抽象，把 PIM 算力作为可编程设备暴露给应用，同时管理 PIM 内存空间、数据搬运与设备执行。区别于传统 SYCL 把 kernel 卸载到 GPU/NPU，PIM-SYCL 把计算卸载到内存内/近存单元（如 bank 级 MAC、PNM 控制器），使 GEMM/GEMV 等 kernel 在数据所在的内存侧执行。论文 [29]（Samsung Hot Chips'23，PIM/PNM for transformer-based AI）描述其 PIM/PNM 集群上的能量效率编程。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26）采用与 CUDA 和 PIM-SYCL 类似的异构编程模型，其编译-执行流程即 PIM-SYCL 式模型的实例：
```
# 1) 主机侧（host）高层 API 暴露关键 RAG 算子与系统配置：
#    GEMV、GeLU、设备初始化、并行策略选择（tensor/pipeline/hybrid）
# 2) 编译：高层 API → 低层 PIM 指令（设备控制器可解码的指令流）
#    PIM 计算命令：PIM_MAC（乘累加）、PIM_CMP（比较）、
#                   PIM_EW_MULT（逐元素乘）、PIM_EW_ADD（逐元素加）——用于组合非线性函数
#    PIM 数据移动命令：PIM_ACT（全 bank 行激活）、PIM_WR_PB（写 PU buffer）、
#                      PIM_RD_PB（读 PU buffer）
#    标准 load/store 经 CXL.mem 接口直接访问 PIM 内存（文档 KV 写入/更新）
# 3) 运行时：指令派发到设备控制器 → 控制器解码并协调设备内组件 →
#    需 PU 计算时广播 PIM 指令到相关 channel 与 PU
```
编译框架的职责：把高层算子（GEMV/GeLU）与并行策略（DAC/CEC 映射）编译为上述指令流，主机负责全局控制与任务编排；设备控制器承担"指令解码→组件协调→PU 广播"的运行时职责。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Samsung PIM-SYCL 开源/发布面向其 PIM/PNM 硬件；通用做法是扩展 SYCL device selector 与 backend 支持 PIM 设备，kernel 内用 SYCL buffer/accessor 或 USM 管理 PIM 内存。使用方式：应用只需用高层 API 表达算子与配置，编译/运行时把计算映射到内存侧单元——对 PIM 加速的 LLM/RAG 推理，开发者无需手写 bank 级指令；MERIDIAN 的 PIM 命令集（PIM_MAC/PIM_CMP/PIM_EW_MULT/PIM_EW_ADD/PIM_ACT/PIM_WR_PB/PIM_RD_PB）即此类编程模型在 LPDDR5X-PIM 上的落地，支撑文档 KV 的直接放置与更新（head-sharded 位置，无需系统级重排）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
