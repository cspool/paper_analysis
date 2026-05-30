## Roofline Model for Quantization Decision（量化决策中的 Roofline 模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Model（Williams et al. 2009）是一个可视化性能模型，用于描述计算平台的理论性能上限。模型将操作的算术强度（Arithmetic Intensity, AI = FLOPs / Bytes accessed）作为 x 轴，可达到的吞吐量（FLOPs/s）作为 y 轴。模型包含两个"天花板"：峰值计算吞吐量（水平线，由硬件算力决定）和峰值内存带宽（斜线，斜率 = 带宽）。当 AI 低于"ridge point"时，操作为 memory-bound；高于 ridge point 则为 compute-bound。

在 MxMoE 中，Roofline 用于判断不同 MoE block 内 GEMM 的计算特性。对于 GEMM shape [m, n, k]，当 n, k >> m 时，AI ≈ m（token 数），意味着低激活频率 expert 的 GEMM 为 memory-bound，高激活频率 expert 为 compute-bound。MxMoE 的硬件感知分配据此决定：memory-bound GEMM 用 W4A16（减少 memory traffic），compute-bound GEMM 用 W8A8 或 W4A4（利用 Tensor Core 高吞吐）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MxMoE 中 Roofline 指导量化方案选择：

```
给定 RTX 4090：
  Peak FP16 compute: 82.6 TFLOPS
  Peak memory bandwidth: 1008 GB/s
  Ridge point: 82.6e12 / 1008e9 ≈ 82 FLOPs/Byte

对于 MoE block 中 expert e 的 linear block j：
  GEMM: X_e [m_e × k] × W_{e,j} [k × n]
  m_e = tokens assigned to expert e
  AI ≈ m_e (当 n,k >> m_e)

  如果 m_e < 83:
    → memory-bound → 用 W4A16 (减少 4x weight memory traffic)
  如果 m_e ≥ 83:
    → compute-bound → 用 W8A8 或 W4A4 (利用低精度算术)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MxMoE 将 Roofline 分析嵌入 ILP 的目标函数 T 中：tile 执行时间 c_{i,j,k,t} 通过 ahead-of-time profiling 获取，隐式捕获了 memory-bound 和 compute-bound 的性能差异。不需要显式计算每 expert 的 AI 值后再决策——profiling 数据自动反映了 GEMM shape 和精度对应的性能，ILP 在全局优化中选择最优组合。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
