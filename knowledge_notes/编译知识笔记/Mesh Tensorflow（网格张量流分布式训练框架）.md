## Mesh Tensorflow（网格张量流分布式训练框架）

术语是什么？
Mesh Tensorflow 是 Google 开发的分布式深度学习框架（Shazeer et al., 2018），专为超级计算机规模的 TPU 集群设计。核心抽象是"mesh"——一个逻辑多维处理器网格，将张量维度（batch、model hidden size）拆分到网格的不同轴上，以统一抽象表达 data parallelism、model parallelism 以及 MoE 的 expert parallelism。ST-MoE 所有实验基于 Mesh Tensorflow 在 Google TPU 集群进行。开源：https://github.com/tensorflow/mesh。

从编译框架角度拆解术语：
Mesh Tensorflow 编译流程：
```
# 1. Define logical mesh
mesh = tf.mesh.Mesh(layout=['data', 'model'], devices=tpu_cores)

# 2. Split tensors across mesh dimensions
x = mesh.split(x, 'batch', mesh_dim='data')
W = mesh.split(W, 'hidden', mesh_dim='model')

# 3. MoE routing via einsum (automatically compiles to all2all)
dispatch = einsum('bt,be->bte', one_hot_tokens, token_tensors)
expert_out = experts(dispatch)
combined = einsum('bte,be->bt', expert_out, gates)

# 4. Compilation chain:
# Mesh Tensorflow Python API → XLA HLO graph → TPU executable
# Communication primitives (allreduce, all2all) auto-injected
```

对于 expert 数少于 data parallel rows 的情况，ST-MoE 引入 3D mesh（outer_batch × inner_batch × model）支持 ≤ 1 expert per core 以确保高 compute-to-memory ratio。

术语一般如何实现？如何使用？
Mesh Tensorflow 提供 `moe.py` 参考实现，包含 capacity factor 控制、einsum dispatch/combine、load balancing loss。训练配置：mixed precision（权重 float32, matmul bfloat16, allreduce float32）、Adafactor optimizer。ST-MoE 通过 performance benchmarking 选择最优 mesh layout 和分片策略。

涉及论文标题：
- ST-MoE Designing Stable and Transferable Sparse Expert Models
- Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（Mesh TensorFlow 是 Switch Transformer 所有实验的基础框架，在其上实现了 Switch routing、load balancing loss、capacity factor 控制和 expert parallelism）
