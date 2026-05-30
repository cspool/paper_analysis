## Graph Optimization Pipeline for MoE Inference (MoE推理图优化流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Graph Optimization Pipeline 是 MoESys 将训练完成的 MoE 模型部署为线上推理服务所经历的 6 步编译优化流程。训练阶段使用动态图（eager execution，调试灵活），推理阶段转换为静态图（graph mode，性能稳定），并通过一系列图级别的 pass 优化模型结构和执行效率。

从编译框架角度拆解术语：
MoESys Graph Optimization Pipeline 的 6 步流程：
```
# Step 1: Graph Fusion
原始计算图 + 分布式策略(op placement)
→ 合并冗余参数(如 expert parallelism 下的重复 op)
→ 消除不必要的 H2D/D2H 节点

# Step 2: Distillation & Compression
Teacher MoE (E_t experts) → Knowledge Distillation
→ Student MoE (E_s experts, E_s < E_t)
→ 减少总参数量和推理计算量

# Step 3: Graph Conversion
Dynamic Graph → paddle.jit.to_static()
→ Static Graph (PaddlePaddle ProgramDesc IR)
→ 消除 Python 解释器开销，启用 ahead-of-time 优化

# Step 4: Graph Segmentation
Static Graph + 硬件拓扑(可用 GPU 数, 显存)
→ 手动或自动选择分布式策略
→ 切分为多个 distributed sub-graphs
→ 在切分处插入必要的通信 op (AlltoAll/AllReduce)

# Step 5: IR Pass Optimization
Distributed Sub-graphs → Apply optimization passes:
  - Kernel Fusion (如 Fused Multi-Head Attention)
  - Memory Planning (复用中间 tensor 的显存)
  - Layout Optimization (NCHW→NHWC 等)
→ 优化后的 IR

# Step 6: Deployment
Optimized IR → 生成各 GPU 的执行计划
→ 加载到推理服务器
→ 提供线上推理服务
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现基于 PaddlePaddle 的 `paddle.jit.to_static` 动态图转静态图机制（类似 PyTorch 的 `torch.jit.trace`/`torch.compile`）。
- 图融合阶段消除了分布式策略产生的冗余通信和参数复制——例如 data parallelism 下各 GPU 的参数 slice 在融合后被标记为同一参数的 shard，而非独立变量。
- Kernel fusion（如 Fused MHA）将多个细粒度 CUDA kernel（QKV projection + attention score + softmax + weighted sum）合并为一个 kernel，减少 kernel launch overhead 和 global memory round-trips。
- 图分割的自动化程度论文未详细说明——标注为"手动或自动"。
- 该 pipeline 与 TensorRT、ONNX Runtime 等通用推理优化器的编译流程有相似之处，但专门针对 MoE 模型的层间独立性和 expert parallelism 做了优化。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
