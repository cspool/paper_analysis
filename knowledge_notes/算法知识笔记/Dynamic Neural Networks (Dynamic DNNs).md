## Dynamic Neural Networks (Dynamic DNNs)

术语是什么？
Dynamic Neural Networks（动态神经网络）是一类在执行路径或架构配置上依赖于运行时输入的深度神经网络。与静态 DNN（所有输入沿同一计算图执行）不同，动态 DNN 根据不同输入选择不同的执行路径、激活不同的子网络或调整网络宽度/深度。典型例子包括：(1) InstaNAS——根据输入图像由 controller 网络动态选择最优的子架构路径（如跳过某些卷积层、使用不同 kernel 大小）；(2) Dynamic Routing Networks——根据输入特征自适应选择从不同路径的语义分割专家网络；(3) Conditional Convolution——根据输入动态计算卷积权重（Mixture of Experts 风格，多个 expert 的权重由 gating 网络按输入加权）。动态 DNN 的设计目标是减少 FLOPs 和推理延迟（特别适合边缘设备），但其 input-dependent 的计算图给 GPU 并发调度带来挑战。

从算法pipeline角度拆解术语：
InstaNAS（实例感知神经架构搜索）动态推理的伪代码：
```
Algorithm: InstaNAS Dynamic Inference
Input: image x, supernet with N possible paths
Output: class prediction

// Phase 1: Controller selects architecture based on input
architecture_config = ControllerNet(x)  
// config: {layer1_skip: True, layer3_filters: 64, layer5_path: "B", ...}

// Phase 2: Execute selected sub-graph
for each layer in supernet:
    if architecture_config[layer].skip:
        continue  // 跳过该层
    
    if architecture_config[layer].is_dynamic:
        // 动态选择路径 (如不同kernel size)
        path = architecture_config[layer].path
        x = execute_path[layer][path](x)
    else:
        x = execute_static[layer](x)

return classifier(x)
```

在 GPU 上执行时，每个 controller 选择的子图对应不同的 kernel 序列（不同的 kernel 类型、不同的 kernel 大小），且每个 input image 产生不同的序列。这导致：(1) 无法提前构建全局 kernel DAG；(2) 大量小 kernel（Conv 2D 被分成多个小的 tile-based kernel，每层可能有多个 kernel）；(3) GPU occupancy 低（InstaNAS-A 在 RTX 3060 上仅 39%）。

术语一般如何实现？如何使用？
动态 DNN 的实现框架：PyTorch（通过 `if/else` 控制流和动态图机制天然支持）、TensorFlow（需要 `tf.cond` 等动态控制算子）。常见动态机制：(1) early exit（提前退出，如 BranchyNet，当中间层置信度足够高时提前输出预测）；(2) layer skipping（如 SkipNet，用轻量 gate 决定跳过哪些层）；(3) adaptive width/depth（如动态选择 channel 数或 block 数）。ACS 论文评估了 InstaNAS（CIFAR10, InstaNAS-A 架构）、Dynamic Routing（Cityscapes, Dynamic-A 16-layer）、Conditional Convolution（4 experts, EfficientNet-B4 backbone, ImageNet）三类动态 DNN。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
