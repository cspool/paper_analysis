## Roofline Model and Ridge Point（Roofline模型与Ridge Point / RP_acc）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Model 是 Williams 等人 (CACM 2009) 提出的性能分析模型，将计算任务的性能上界表示为峰值计算吞吐和峰值内存带宽的函数。在 Roofline 图中，横轴为 ArI (Op/B)，纵轴为可达到的吞吐 (FLOPS)。Ridge Point (RP_acc) 是 memory-bound 和 compute-bound 之间的转折点，定义为 $\text{RP}_{\text{acc}} = \frac{\text{Peak Throughput (FLOPS)}}{\text{Peak Memory Bandwidth (Bytes/s)}}$。对于 B200 GPU：RP = 2250 TFLOPS / 8 TB/s = 281.25 Op/B。ArI < RP 的任务 memory-bound（实际吞吐 = ArI × BW），ArI > RP 的任务 compute-bound（实际吞吐 = Peak Throughput）。论文发现现代加速器的 RP 在一个狭窄范围内（138～320 Op/B），且 B200 相比 V100 的 RP 仅增加 2×（虽然 FLOPS 增加了 18×）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文使用 Roofline 分析 LLM 推理的典型案例（Figure 3, H100 GPU, L=4096）：

| 层 | ArI (Op/B) | vs RP=206 | Bound |
|----|-----------|-----------|-------|
| GPT-3 Core-Attention (MHA) | ~1 | 远低于 RP | Memory-bound |
| Llama4 Core-Attention (GQA, deg_grp=5) | ~5 | 远低于 RP | Memory-bound |
| DeepSeek-R1 Core-Attention (MLA+reordering) | ~200 | 接近 RP | Balanced |
| FC layers (large B) | >200 | 高于 RP | Compute-bound |
| FC layers (small B, GEMV) | ~100 | 低于 RP | Memory-bound |

Roofline 图直观展示了 MLA + reordering 将 Core-Attention 从最左侧（memory-bound 极端）移动到 RP 附近的关键效果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Roofline 分析在论文中用于：(1) 论证 attention-specialized PIM 不再必要（MLA 已使 attention ArI 匹配 GPU RP）；(2) 推导 MoE 的 $B_{\text{MoE}} = RP_{\text{acc}} \cdot n_e/n_k$，一旦选定加速器和模型就固定；(3) 分析低精度（FP8）权重下 RP 加倍但 $B_{\text{RP}}$ 不变（因 memory access 也减半），而 $B_{\text{cap}}$ 增加。常用工具：NVIDIA Nsight Compute 可直接测量 kernel 的 FLOPs 和 memory traffic 计算实际 ArI。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
