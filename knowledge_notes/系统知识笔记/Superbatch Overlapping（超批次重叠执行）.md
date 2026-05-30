## Superbatch Overlapping（超批次重叠执行）

术语是什么？
Superbatch Overlapping 是 ScaleMoE 论文提出的 CPU-GPU 并行优化技术。Dynamic Expert Clustering 和 Topology-aware Expert Remapping 需要 CPU 执行时间（聚类计算、遗传算法搜索、expert 数据搬运），若与 GPU 训练 iteration 串行执行会显著增加总训练时间。Superbatch 将完整 epoch 划分为更小的 superbatch（每个 superbatch = N 个 training iterations），每个 superbatch 内的 GPU iterations 使用相同的 cluster-to-GPU 映射。关键优化：superbatch n 的 clustering + remapping（CPU）与 superbatch n-1 的 GPU iterations 并行执行——即 CPU 为下一 superbatch 准备 expert 布局的同时，GPU 继续执行当前 superbatch 的训练。

从系统架构角度拆解术语：
Superbatch Overlapping 的时间线：
```
无 Overlap（串行）:
|--Clustering&Remap--|------Superbatch Iterations------|--Clustering&Remap--|...
     7.79s CPU              N × 6.68s GPU                  7.79s CPU
总时间增加: 12.48%

有 Overlap（并行）:
Superbatch 0: |--Clustering-->|---Superbatch-0 Iterations---|
Superbatch 1:                 |--Clustering-->|---Superbatch-1 Iterations---|
Superbatch 2:                                  |--Clustering-->|---...
                     CPU 与 GPU 完全重叠，clustering overhead → 0.001%
```
论文 sensitivity analysis 表明 superbatch=100 在 load-imbalanced 和 balanced 场景下均达到最优：clustering overhead 可忽略且 iteration time 不受影响。更小的 superbatch 导致频繁 clustering（效率下降），更大的 superbatch 导致单次 clustering 时间过长（无法完全 overlap）。

术语一般如何实现？如何使用？
在 PyTorch 训练循环中实现：每个 superbatch 开始时，CPU 线程异步启动 clustering + remapping；GPU 继续执行当前 superbatch 的 iterations；下一个 superbatch 开始时切换到新的 expert 布局。论文在 ScaleMoE 中实现，所有 clustering 相关 overhead 从 568.91ms/iteration（8.51%）降至 16.27ms/iteration（0.26%）。

涉及论文标题：
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
