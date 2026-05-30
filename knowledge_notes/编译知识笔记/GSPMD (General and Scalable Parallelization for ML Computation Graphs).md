## GSPMD (General and Scalable Parallelization for ML Computation Graphs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GSPMD (Xu et al., 2021) 是 Google 提出的基于编译器的自动并行化系统，将 ML 计算图自动分布到多设备（TPU/GPU）。核心思想是分离模型编程与并行化：用户编写单设备程序，仅对少量关键 tensor（通常 <1% 的图中 tensor）添加 sharding annotation，GSPMD 编译器自动推导全图的分区方案并插入必要的 collective communication（AllReduce/AllGather/ReduceScatter/All-to-All/CollectivePermute），生成等价但已并行化的 SPMD 程序。支持的并行范式：Data Parallelism（batch 维度切分）、In-layer Model Parallelism（权重切分）、Spatial Partitioning（图像维度切分）、Weight-update/Optimizer-state Sharding、Pipeline Parallelism（经由 wrapper library 归约为 tensor sharding）及嵌套混合模式。基于 XLA 编译器实现，支持 TensorFlow/JAX/PyTorch，硬件覆盖 CPU/GPU/TPU。在 2048 TPUv3 上达到 50-62% compute utilization，已用于 LaMDA (137B)、GShard-M4 (577B)、BigSSL 等 production 模型。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
GSPMD 的编译流程：
```
1. 用户标注 (User Annotation):
   仅对少数关键 tensor 添加 sharding 标记（通常 <10 个 tensors）:
   mesh_split(tensor, device_mesh, dims_mapping)
   # device_mesh: 逻辑多维设备网格 (如 [8, 64] 表示 512 devices)
   # dims_mapping: 每个 tensor dim 映射到 device mesh 的哪个维度

2. Sharding Completion Pass (编译器自动推导):
   Input: partial sharding annotations
   Process: propagate constraints through all operators
     - 前向传播: 根据 operator 语义推导 output sharding
     - 反向传播: 根据 output sharding 约束推导 input sharding
     - 当冲突时: 插入 Reshard (collective communication)
   Output: complete per-tensor sharding + communication schedule

3. SPMD Program Generation:
   - 根据 complete sharding 转化每个 operator 为设备本地计算
   - 插入 collective communication (AllReduce/AllGather/ReduceScatter/All-to-All)
   - 生成单程序 (同一 program 在所有 device 上执行，但处理不同数据分片)

4. XLA Backend:
   - 标准 XLA 优化 (fusion, layout, scheduling)
   - 生成设备代码 (TPU/GPU/CPU)
```

GShard/Expert Choice/LaMDA 等 Google 大模型均基于 GSPMD 实现分布式训练。以 Expert Choice MoE 为例：用户标注 expert 参数沿 expert 维度 partition（`mesh_split(expert_weight, mesh, [0, -1])`），GSPMD 自动插入 token dispatch 的 All-to-All collective 和 expert 梯度同步的 AllReduce。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源实现：GSPMD 已作为 XLA 编译器的一部分开源（`tensorflow/compiler/xla`）。JAX 通过 `pjit`/`xmap` API 直接使用 GSPMD。
- 使用方式：在 JAX 中使用 `shard_map` 或 `pjit`；在 TensorFlow 中使用 `dtensor` API。典型用法（JAX）：
  ```python
  import jax
  from jax.sharding import PartitionSpec as P, Mesh
  devices = jax.devices()  # 512 TPU chips
  mesh = Mesh(jax.devices().reshape(8, 64), ('data', 'expert'))
  # 标量 weight 沿 expert 维度 sharding
  sharding = jax.sharding.NamedSharding(mesh, P('expert', None))
  ```
- GSPMD 自动推导能力使得用户仅需标注 ~10 个 tensor（<1% 图中 tensor），编译时间通常 <1 min，远快于手动并行化。
- Expert Choice (Zhou et al. 2022) 使用 GSPMD 的 2D sharding 充分利用 TPU V4 的 2D torus 拓扑，在 512 TPU V4 上训练 8B/64E 模型。

涉及论文标题：
- Mixture-of-Experts with Expert Choice Routing
