## Expert Loading Optimization

术语解释
Expert加载优化是减少expert offloading场景中从CPU内存/SSD加载expert到GPU的延迟的技术，因为expert loading通常占总推理时间的80%以上。

术语是什么？
加载优化策略：
- **低精度加载**（EdgeMoE、HOBBIT）：用低精度expert替代高精度expert来减少加载时间。低精度版本的字节数更少，PCIe传输时间更短
- **自适应精度选择**（HOBBIT）：动态决定加载精度——基于gating输出的重要性分数，低于阈值加载低精度版本，高于阈值加载高精度版本
- **重要性跳过**（AdapMoE）：使用Fisher信息矩阵计算每个expert的重要性，完全跳过不重要的expert（不激活也不加载）
- **CPU辅助计算**（Fiddler、HOBBIT）：将激活值拷贝到CPU后在CPU上执行expert计算，结果拷回GPU。相比通过PCIe加载expert权重到GPU再计算，拷贝激活值的开销更小
- **CPU-GPU-I/O流水线**（MoE-Lightning）：同时利用CPU、GPU和I/O资源的三级流水线

从系统架构角度拆解术语。
以Fiddler的CPU辅助计算为例：
```
# 传统方式（加载expert到GPU）
latency_traditional = PCIe_read(expert_weights) + GPU_compute(weights, x)
# expert weights 很大 → PCIe latency高

# Fiddler方式（激活值到CPU）
latency_fiddler = PCIe_write(activations_to_CPU) + CPU_compute(weights_in_CPU, x) + PCIe_read(results_to_GPU)
# 激活值 << expert权重 → 总延迟更低
```
结论：expert在GPU miss时，使用CPU计算比加载expert到GPU更快，前提是CPU计算资源充足。

术语一般如何实现？如何使用？
- 混合精度expert存储（FP16高精度 + INT4低精度两份副本）
- 异步加载（CUDA streams + CPU线程）
- 权衡：精度损失 vs 加载延迟减少
- 适用平台：有充足CPU资源的场景（PC、服务器），不适合共享内存设备（Jetson Orin）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
