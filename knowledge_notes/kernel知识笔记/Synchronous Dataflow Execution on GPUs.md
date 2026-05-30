## Synchronous Dataflow Execution on GPUs

术语是什么？
Synchronous Dataflow Execution 是一种 GPU 执行模型，将 DL 计算图中的不同算子映射到不同 CTA，通过片上队列（on-chip queue）传递 tile 级中间数据，使多个算子在不同 SM 上 **空间并发**（spatial concurrency）执行，而非 BSP 的串行执行或 Vertical Fusion 的时间复用（temporal multiplexing）。在 Kitsune 的形式定义中，属于 synchronous dataflow（SDF）：当数据到达输入队列时，CTA 开始执行；完成计算后将结果写入生产者队列以触发后继 CTA。子图的首节点从主存读取激活，末节点将结果写回主存。

从kernel调度角度拆解术语：
Kitsune 的 synchronous dataflow 执行通过两个原语实现：(1) 软件 ring buffer queue（L2-resident, atomics-based）实现 inter-CTA 通信；(2) modified grid scheduler（双 arbiter：SIMT + Tensor）实现异构 CTA 在 SM 上 colocate。

以 MLP spatial pipeline（Linear → ReLU → Linear）的 dataflow 执行为例：

```
// Host: 配置 spatial pipeline
cudaPipeline pipeline;
pipeline.addKernel(kernel_Linear1, CTA_count=64, type=TENSOR);
pipeline.addKernel(kernel_ReLU,   CTA_count=44, type=SIMT);
pipeline.addKernel(kernel_Linear2, CTA_count=44, type=TENSOR);
pipeline.addQueue(queue0, producer=Linear1, consumer=ReLU);
pipeline.addQueue(queue1, producer=ReLU, consumer=Linear2);
pipeline.launch();

// GPU 端执行:
// SM_0: Linear1_CTA_0 (TensorCore: 执行GEMM tiles)
//       ReLU_CTA_0   (SIMT Core:  执行elementwise) ← 双arbiter确保co-location
// SM_1: Linear1_CTA_1 + ReLU_CTA_1
// ...
// 不同stage CTA并发: dataflow触发执行
//   Linear1写tile→queue0→ReLU消费, ReLU写result→queue1→Linear2消费
//   全程无global barrier, 无DRAM round-trip
```

与 BSP 的关键区别：(a) No global barrier between operators；(b) intermediate data 通过 L2-resident queue 传递（非 DRAM round-trip）；(c) 多类型 CTAs 并发执行，充分利用 TensorCore 和 SIMT Core。

术语一般如何实现？如何使用？
Kitsune 通过 PyTorch Dynamo compiler backend 自动将 DL 图 lowering 到 dataflow pipeline：(a) Subgraph Selection → 模式匹配识别 sf-node；(b) Pipeline Design → 插入 queue 节点；(c) Load Balance → ILP 求解 CTA 分配；(d) Code Generation → CUDA kernel 改写为读写 queue。cudaPipeline API 指定 kernel type（SIMT/TENSOR），modified grid scheduler 用双 arbiter 配对不同类型 CTA 到同一 SM。Queue 性能：54 queues × 2 CTAs = 108 CTAs 对应 A100 108 SMs，aggregate bandwidth 2 TB/s（37 GB/s/queue @ 128-256KB payload），同步 overhead <63% @ ≥64KB payload。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
