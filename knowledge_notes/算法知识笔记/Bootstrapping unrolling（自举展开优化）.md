## Bootstrapping unrolling（自举展开优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bootstrapping unrolling 是 TFHE 自举的算法级优化（思想源自 Bourse 等 CRYPTO'18、MATCHA 的 bootstrapping-key unrolling 实践）：自举主循环的 n 次迭代本身串行，把 r 次相邻迭代合并为一次展开迭代，每轮同时处理 r 个密钥位/旋转量，迭代深度 n→n/r、FFT/IFFT 次数 n→n/r（计算量下降），代价是需要 (2^r−1) 大小的展开阵列缓存中间结果、且 r 倍并发密钥访问（内存带宽压力上升）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 原循环每迭代一次 CMux/external product；展开后：
```
for j in 0..(n/r)-1:
    # 一次迭代同时应用 r 个密钥位，需 (2^r - 1) 展开阵列
    ACC = UnrolledCMux([BK_{j*r}, ..., BK_{j*r+r-1}],
                       [a_{j*r}, ..., a_{j*r+r-1}], ACC)
```
- Annotations：r 增大 → FFT 次数下降（计算减）但每轮并发密钥访问上升（带宽增）。本论文实测最优 r 与带宽耦合：Strix 场景（300 GB/s）最优 r=2，MATCHA 场景（640 GB/s）最优 r=3，r 过大带宽压力导致性能回退——最优展开因子无法跨硬件场景手工迁移，必须自动搜索（AutoFHE DSE 的核心论据）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件：bootstrapping 模板内置 (2^r−1) 展开阵列；AutoFHE 把 r 作为 DSE 设计变量自动选择（禁用 unrolling 性能降 39.5%）。手工实践：MATCHA 手工选定 r=3；MOSFHET 的 blind rotation unfolding 属同类思想（多值自举，2× 加速）。带宽有限场景（如 300 GB/s）不宜盲目增大 r。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
