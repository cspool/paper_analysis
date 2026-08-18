## Adam 优化器（AdamW、fp32 主副本与 CPU Adam 卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Adam = 一阶自适应矩估计优化器：维护梯度一阶矩 m 与二阶矩 v 的指数滑动平均（β1≈0.9、β2≈0.999），偏差校正后按 m̂/(√v̂+ε) 方向更新权重。混合精度训练（bf16/fp16）中：前/反向用低精度权重与梯度，Adam 状态与主权重副本保持 fp32——每参数共 2B 梯度 + 12B 状态（m 4B + v 4B + 主权重 4B）+ 2B 参数副本（Web 证据：HF model memory anatomy）。
- CPU Adam（ZeRO-Offload 风格）：SIMD + 循环展开 + 多线程，每参数 17 次浮点运算；DisDP 以此估算 PS 算力需求（100Gbps 聚合梯度需 99 GFLOPS）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 每参数更新（x 为权重、g 为梯度、lr 学习率）：
```
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
m_hat = m / (1-β1^t);  v_hat = v / (1-β2^t)
x = x - lr * m_hat / (sqrt(v_hat) + ε)
```
- DisDP 中 Adam 全部在 PS 的 CPU 上执行（worker GPU 不碰优化器），是 step-centric 流水中的第 3 步（读 2B 梯度 + 12B 状态、写 12B 更新状态 + 2B 参数副本）；收敛与 GPU 侧 Adam 一致（OPT-66B rm-static 微调 loss 曲线与 ZeRO-Infinity 重合）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch FusedAdam、DeepSpeed CPUAdam/ZenFlowCPUAdam；fp32 状态放 CPU/SSD（ZeRO-Offload/Infinity）或单台 PS（DisDP）。使用：bf16 训练需考虑动态范围（论文用 bf16 + 激活检查点）；卸载场景需与参数服务流水重叠以隐藏优化器时延。信息缺口：论文未给出 β1/β2/ε 具体值。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
