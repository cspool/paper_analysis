## NULL Stream (Default Stream)

术语是什么？
NULL Stream（也称为 Default Stream、Stream 0）是 CUDA 中隐式的默认流。在传统行为（legacy default stream）中，NULL stream 具有特殊的同步语义：当在 NULL stream 中执行操作时，先等待所有 blocking stream 完成，操作排队后，所有 blocking stream 再等待 NULL stream 完成。这导致 NULL stream 成为一个隐式全局同步屏障，可能严重破坏并发性能。

从kernel调度角度拆解术语：
论文通过实验发现了 NULL stream 在 TX2 GPU scheduler 中的具体调度规则（N1-N2，扩展自 Rule G2）：
- Rule N1: NULL stream queue 头部的 kernel Kk 入队 EE queue 的条件是——对于每个其他 stream queue，该 queue 为空或该 queue 头部的 kernel 在 Kk 之后 launch。
- Rule N2: 非 NULL stream queue 头部的 kernel Kk 入队 EE queue 的条件是——NULL stream queue 为空或 NULL stream queue 头部的 kernel 在 Kk 之后 launch。

具体例子（论文 Fig. 5, Table 2）：
```
假设三个 stream: S1(normal), S2(normal), NULL stream
Launch order: K1(S1, t=0) → K2(NULL, t=0.2) → K3(S2, t=0.2) → K4(S2, t=0.4) → K5(NULL, t=0.6) → K6(S3, t=0.8)

Rule N1 效果：
  K2(NULL stream head) 入 EE queue 条件: S1的head(K1)在K2之前launch → K2必须等K1完成后才能入EE queue
  → K2被K1阻塞，直到K1执行完。在此期间：
  Rule N2 效果：K3 虽在 S2 头部，但 NULL stream queue 非空且 head(K2)在K3之前launch → K3不能入EE queue
  → K3和K4都被K2阻塞（即使GPU有空闲资源）

结论：NULL stream 造成严重的不必要阻塞和GPU容量浪费。K6本可与K3/K4并发执行，但因NULL stream的同步语义而被全部串行化。
```

术语一般如何实现？如何使用？
NULL stream 是 CUDA 向后兼容的设计。在现代 CUDA 开发中，最佳实践是：(1) 使用显式 stream 替代 NULL stream；(2) 编译时使用 --default-stream per-thread 标志使每个 CPU 线程有独立的默认流；(3) 使用 cudaStreamNonBlocking 标志创建不与 NULL stream 同步的 stream。在实时系统中，NULL stream 的隐式同步行为应完全避免——论文明确指出"usage of the NULL stream is problematic if real-time predictability and efficient platform utilization are desired"。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
