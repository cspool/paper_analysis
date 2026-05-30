## CANN (Compute Architecture for Neural Networks)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CANN 是华为开发的异构计算架构，类似 NVIDIA CUDA 的角色，为 Ascend NPU 提供从底层驱动到上层框架的完整软件栈。包含：算子库（提供优化的 AI 算子实现）、图编译器（将框架图 IR 编译为 Ascend 可执行指令）、驱动（管理 NPU 硬件资源）和 toolkit（开发工具和调试工具）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 PanGu-Σ 训练中，CANN 的软件栈层次：
1. **MindSpore 框架**（上层）：定义模型计算图、并行策略
2. **CANN GE engine**（中层）：图编译和优化，将 MindSpore IR 转换为 Ascend 可执行的计算图
3. **CANN Runtime + Driver**（底层）：管理 NPU 内存、调度 AI Core/Vector Core 执行、管理 HCCS/HCCL 通信
4. **Ascend 910A 硬件**：32 AI Core 执行计算

LocMoE 使用的 CANN 版本为 5.1.RC2.1（toolkit 1.84, driver 23.0.rc2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CANN 通过统一的 AscendCL API 向上提供编程接口，支持 C/C++ 和 Python。提供 ATC（Ascend Tensor Compiler）进行离线模型转换和优化，以及 ACL（Ascend Computing Language）进行在线算子开发和调优。CANN 同时支持 MindSpore、PyTorch（通过 torch_npu 适配插件）和 TensorFlow 等框架。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
