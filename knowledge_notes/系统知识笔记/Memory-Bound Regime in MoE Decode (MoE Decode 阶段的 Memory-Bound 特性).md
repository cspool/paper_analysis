## Memory-Bound Regime in MoE Decode (MoE Decode 阶段的 Memory-Bound 特性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 MoE 模型的 decode 阶段，由于每 token 仅激活 k/N 比例的 expert（如 Qwen3 的 k=8, N=128，稀疏因子 16×），平均每 expert 负载（cnt^i）随 batch size B 仅以 k/N 速率增长。根据 Roofline 模型，decode 的算术强度（Arithmetic Intensity）约为 100-200 FLOPs/Byte，而稀疏因子 N/k 导致进入 compute-bound 所需 batch size 约为 N/k × ridge_point ≈ 1600（以 Qwen3 为例）。因此在中等到较大的 batch size（如 B=16、32、64）下，即使各 token 激活了 k 个 expert，MoE 层仍处于 memory-bound 状态——延迟由从 HBM 加载 expert 权重到 SRAM 的时间（b·T 项）主导，而非计算时间（a·Bk 项）。其中 T 为 batch 内唯一激活 expert 数，期望值 T = N(1-(1-k/N)^B)，B 增大时 T 快速逼近 N。

从系统架构角度拆解术语：
OEA 论文的延迟模型刻画了这一现象：
```
总延迟 = Σ_i f(cnt^i) = b · T + a · B · k
其中:
  f(0) = 0, f(n) = a·n + b (n > 0)
  cnt^i = routed到expert E_i 的 token 数
  T = |{i : cnt^i > 0}|  (唯一激活 expert 数)
  b = 从HBM加载一个expert权重的固定延迟
  a = 处理一个token的计算延迟
```
当 expert 处于 memory-bound 时（cnt^i 很小，b >> a·cnt^i），b 主导延迟。因此 T 越大延迟越高。实际在 H100 上验证：MoE 层平均延迟与 T 呈严格线性关系（R² > 0.99）。

术语一般如何实现？如何使用？
- 判断方法：profile decode 阶段的 DRAM read bytes 和 FLOPs，计算算术强度，与 GPU ridge point 比较。NVIDIA Nsight Compute 可直接提供 Roofline 分析。
- 优化方向：在 memory-bound 下应减少 T（减少加载的 expert 数），而非减少 Bk（总计算量）。这就是 OEA 的核心动机——通过 batch-aware routing 减少 T。
- 与 dense 模型的区别：dense FFN 权重固定（T=1 恒成立），B 增大不改变 memory fetch 次数。MoE 的 T 随 B 快速增大，在 B=16 时 T≈48（Qwen3），是 B=1 时 T=8 的 6 倍。

涉及论文标题：
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
