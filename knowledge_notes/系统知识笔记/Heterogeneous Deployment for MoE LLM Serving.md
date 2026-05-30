## Heterogeneous Deployment for MoE LLM Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Heterogeneous Deployment for MoE LLM Serving（MoE 大模型推理的异构部署）是一种利用不同类型 GPU 分别部署 MoE 模型的 attention 模块和 expert/FFN 模块以最大化性价比的部署策略。核心原理：attention 是 memory-intensive（需频繁访问 KV cache）且需要大 GPU 内存（存储 KV cache），应使用 per-cost 内存带宽和容量更高的 GPU（如 NVIDIA H20：51.9 GB/$, 2214.1 GB/s/$）；expert/FFN 是 compute-intensive（batch GEMM），应使用 per-cost 计算能力更高的 GPU（如 NVIDIA L40S：335.2 TFLOPS/$）。这一策略是 MegaScale-Infer 的 Disaggregated Expert Parallelism 的自然延伸——因为 attention 和 expert 已物理分离，可独立选择硬件。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MegaScale-Infer 中的异构部署硬件选择和性能：
| GPU | Price (norm) | Memory (GB) | BW (GB/s) | Compute (TFLOPS) | Per-$ GB | Per-$ GB/s | Per-$ TFLOPS |
|-----|-------------|-------------|-----------|-------------------|----------|------------|--------------|
| H20 | 1.85 | 96 | 4096 | 148 | 51.9 | 2214.1 | 80.0 |
| L40S | 1.08 | 48 | 864 | 362 | 44.4 | 800.0 | 335.2 |
| A800 | 2.26 | 80 | 2039 | 312 | 35.4 | 902.2 | 138.1 |
| H800 | 5.28 | 80 | 3430.4 | 989 | 15.2 | 649.7 | 187.3 |
| L20 | 1.00 | 48 | 864 | 119.5 | 48 | 864 | 119.5 |

部署策略：attention node → H20（最高 per-$ memory bandwidth）；expert node → L40S（最高 per-$ compute）。H20 节点 900 GB/s NVLink + 4×400 Gbps NICs；L40S 节点 PCIe intra-node + 2×400 Gbps NICs。

性能结果（Figure 9）：
- Decoding throughput per unit cost：最高 3.24× vs vLLM on H20，1.86× vs TensorRT-LLM on H20
- End-to-end throughput per unit cost（含 prefill）：最高 1.66× vs baselines
- Throughput per unit power：1.80× (decoding), 1.72× (end-to-end) vs baselines（H20 500W vs L40S 350W）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Deployment plan search 自动枚举 GPU 类型组合（attention GPU type × expert GPU type）与 parallelism configs，选最优 throughput per unit cost。
- 关键实现挑战：(a) 异构 GPU 间的通信性能不对称（L40S 仅 2 NICs vs H20 4 NICs），需在 M2N 通信中做好拥塞控制和流量调度；(b) attention 和 expert 的计算时间需在异构硬件下重新平衡（T_a ≈ T_e 约束）；(c) 异构部署的 GPU 内存容量不对称影响 batch size 上限（attention KV cache → H20 96GB 优势）。
- 生产部署：MegaScale-Infer 已在 ByteDance ~10,000 GPU 集群上异构部署，降低 serving cost 1.5–2.0×。
- 扩展：异构不仅限于 attention vs expert——也可在 expert node 内部使用不同 GPU（如热门 expert 放在高带宽 GPU，冷门 expert 放在低成本 GPU），论文未实现但提及此为 future work。

涉及论文标题：
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---
