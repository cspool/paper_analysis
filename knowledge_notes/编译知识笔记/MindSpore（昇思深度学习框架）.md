## MindSpore（昇思深度学习框架）

术语是什么？
MindSpore 是华为开源的深度学习框架，原生支持静态计算图（static graph）模式，通过编译时图优化提升训练和推理效率。与 PyTorch（默认动态图）不同，MindSpore 的静态图要求在编译前固定 tensor shape 和 operator graph，编译后直接执行——这带来更高的计算效率，但也限制了运行时的动态调整能力。EfficientMoE 论文基于 MindSpore 2.0 + Mindformers 1.0 实现了所有算法优化。

从编译框架角度拆解术语：
MindSpore 的训练执行流程：
1. **图构建**：用户定义模型（Python API），MindSpore 将 Python 计算描述转换为中间表示（IR）。
2. **图优化**：编译器对 IR 进行算子融合、内存优化、并行策略切分（DP/MP/EP）。
3. **静态编译**：优化后的 IR 编译为可执行图——所有 tensor shape、expert capacity 等参数在此阶段固定。
4. **执行**：编译后的图在 Ascend accelerator 上直接执行，无前端语言（Python）中间介入，效率高。

EfficientMoE 在 MindSpore 上的关键修改：(1) 在静态图编译前插入 Load Prediction Model，周期性评估 expert 负载；(2) 在 cycle 边界修改 expert placement（replica 调度到目标 accelerator）；(3) 编译前为每个 expert 注入差异化 capacity（C_j^i）替代统一的固定 capacity。这些修改保持在静态图框架内——不依赖运行时 shape 变化，而是通过周期性的编译前重配置实现动态性。

术语一般如何实现？如何使用？
MindSpore 分为 MindSpore Lite（端侧推理）、MindSpore 训练框架、MindFormers（大模型套件）三层。代码开源在 Gitee (https://gitee.com/mindspore)。静态图模式通过 `@mindspore.jit` 装饰器或 `model.train()` 中的 `mode=ms.GRAPH_MODE` 启用。与 PyTorch 生态的主要区别：原生支持 Ascend 硬件、静态图优先、自动并行切分。

涉及论文标题：
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
