## Logical Sharding（strided GEMM 逻辑分片，零开销 TP/DP 切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Logical Sharding 是 RESONATOR 实现"零开销、Just-in-Time 并行切换"的机制，解决 canonical TP 的根本局限：canonical TP 把 transformer 层权重物理切分到多 GPU、每 GPU 只存自己的分片，改 TP 度需在网络里重分布权重数据（成本高、不可能逐 batch 适应）。logical sharding 利用 cuBLAS/CUTLASS 等现代 GPU 计算库的 strided memory access 能力——GEMM 可通过 leading dimension（ld）参数在非连续内存切片上计算：启动时把完整未分片的 encoder 模型预载到每张 GPU，运行时只改 kernel launch 参数（ld），把计算逻辑约束到想要的 1/k* 分片上，把"数据搬移问题"变成"元数据更新"。DP 执行时 kernel 作用在完整本地 tensor；TP=k* 时每 worker 用 strided GEMM 只算自己的 1/k* 逻辑分片。代价：encoder 权重全量复制到每 GPU（ViT-675M 1.3GB、MoonViT 0.8GB，A100 80GB 上 HBM 开销 1.6%/1.0%），且 strided 布局的 GEMM 内存合并/缓存局部性略差。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
logical sharding 的 GEMM 执行（Figure 16 微基准，Qwen2-VL-7B 三类 encoder 线性层）：
```
# canonical TP（materialized）：权重物理连续分片 W_shard [h, 4h/k]
Y = GEMM(X, W_shard)                       # 每 GPU 算自己的列分片，再 all-reduce
# logical sharding：完整权重 W_full [h, 4h] 预载每 GPU，只改 ld
Y = GEMM_strided(X, W_full, ld=4h, n_cols=4h/k, col_offset=worker*4h/k)
# DP 执行：kernel 用完整本地 tensor
Y = GEMM(X, W_full)
```
Annotations：ld（leading dimension）=行间内存步长；strided GEMM 通过指定 ld 与起始偏移让每 worker 只计算连续权重中的 1/k 列；因逻辑分片仍含宽矩阵 tile（分片列宽 896–7168 元素，远大于 64 元素 L2 cache line），kernel 在 tile 内保持规整向量化访存，多出的 stride 只影响行间地址计算而非内层循环——微基准显示 strided vs contiguous 中位差仅 0.7%、91% 配置 <2%（延迟与 MFU 归一化到 A100 FP16 峰值 312 TFLOPS）；少量低序列长配置有残余差距（固定 launch/寻址开销未被摊薄）。Performance Atlas 直接 profile strided 路径，布局代价已计入调度延迟估计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 cuBLAS/CUTLASS 的 strided GEMM API（ld/batch stride 参数），权重全量预载 + 控制面改 launch 参数。使用：Inter-GPU Parallelism Engine 每 batch 跑 PRISM 选好 DP/TP 计划后，统一运行时按计划给各 worker 下 ld/offset 参数即可，不触碰数据面（无 reshuffle/reload）——这正是"动态 per-batch 并行"可行的使能器。论文 §IV-H 用三类 encoder GEMM（QKV projection、FFN up、FFN down）× L_seq∈{1k,4k,8k,16k} × TP∈{1,2,4} 验证布局代价可忽略。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
