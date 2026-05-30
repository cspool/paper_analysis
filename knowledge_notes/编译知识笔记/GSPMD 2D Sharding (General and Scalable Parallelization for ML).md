## GSPMD 2D Sharding (General and Scalable Parallelization for ML)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GSPMD 是 Google 提出的基于 XLA 编译器的自动并行化系统（Xu et al. 2021, arXiv:2105.04663），通过用户标注少量张量（通常 <10 个）的 sharding hint，自动推演整个计算图的张量划分策略，并生成跨设备的 SPMD 程序。2D sharding 指沿 device mesh 的两个维度同时划分张量，实现在单模型层内同时复用多种并行策略。

核心 API：`mesh_split(tensor, device_mesh, dims_mapping)`，其中 `dims_mapping[i]` 指定 tensor 的 dim i 沿 device mesh 的哪个维度划分（-1 表示复制）。支持的并行模式包括 Data Parallelism、In-layer Model Parallelism、Spatial Partitioning、Weight-update Sharding、Pipeline Parallelism。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 GLaM 中，GSPMD 2D sharding 的编译流程：

```
# Step 1: 用户标注 key tensors 的 sharding hint
# Expert 权重 W[E=64, M=8192, H=32768] → dims_mapping=[0, -1, 1]
#   dim E 沿 mesh dim_0, dim H 沿 mesh dim_1, dim M 不划分
# 输入激活 x[B, S=1024, M=8192] → dims_mapping=[0, -1, 1]
#   dim B 沿 mesh dim_0, dim M 沿 mesh dim_1, dim S 不划分

# Step 2: GSPMD Compiler Pass — 自动传播 sharding 属性
# 优先级驱动迭代传播（elementwise 优先）→ 收敛

# Step 3: SPMD 代码生成
# XLA → 各 device 独立程序 + 必要 collective comm

# Step 4: GLaM 特有优化 — while_loop 包装
# 同 index expert 跨层放同 device → identical computation graph
# → XLA while_loop 减少编译时间
```

GLaM 中 GSPMD 效果：1,024 TPU-v4 芯片上实现 50-62% 计算利用率，支持 1.2T 参数训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

基于 XLA 编译器实现，支持 TensorFlow、JAX、PyTorch。Sharding propagation 使用优先级队列算法（elementwise 优先、MatMul 次之）。Pipeline parallelism 通过 wrapper library 映射为 stage dimension 的 tensor sharding。约束：生成 SPMD 程序而非 MPMD。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
