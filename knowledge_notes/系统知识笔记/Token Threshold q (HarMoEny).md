## Token Threshold q (HarMoEny)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Threshold q 是 HarMoEny token rebalancing 算法中的唯一超参数（Algorithm 2, Line 11）。当某 expert e_max 的待转移 token 数 t_move < q 时，rebalancing 停止——因为转移过少 token 无法使 expert computation time 超过 expert loading time（从 system memory 到 GPU memory），prefetching 无法被计算掩盖。

从系统架构角度拆解术语：

q 的数学推导（Section 4.4, Appendix B）：要求 expert computation time > expert loading time

$$\frac{|O|}{\phi} > \frac{|E|}{\beta}$$

对于 2-layer MLP expert（m×p → p×m），简化后得：

$$q > \frac{\phi \cdot d_{type}}{2\beta}$$

- φ: GPU FLOPS, d_type: 精度字节数, β: PCIe bandwidth
- **示例** (V100 FP16): q > 14×10¹² × 2 / (2 × 32×10⁹) ≈ 438 tokens
- **关键性质**: q 仅依赖硬件规格（φ, β, d_type），与 workload 的 dynamic properties 无关；HarMoEny 实验表明对 q 不敏感，下界提供可靠近似

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 HarMoEny 初始化时根据硬件规格计算一次（static），无需 per-model 或 per-workload 调参。系统设计者根据实际 GPU FLOPS 和 PCIe 带宽调整即可。

涉及论文标题：
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
