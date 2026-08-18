## Cached Expert Ratio（CER，专家缓存率 / 显存约束控制指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cached Expert Ratio（CER）是 STEP 定义的控制变量：CER = GPU 可用专家槽位数 / 模型专家总数，量化"GPU 显存能缓存多大比例的专家"，用于模拟不同内存约束环境（论文取 25%/50%/75%）。它是 expert offloading 系统的核心内存预算度量：CER 越低，显存中驻留的专家越少、CPU→GPU 的 PCIe 取数越频繁，推理延迟越高。STEP 用它做敏感性分析——在不同 CER 下比较预取/剪枝策略的有效性（Fig.10/11/18/19），并观察到低 CER（25%）下"削减冗余计算 + 高命中率预取"的收益最显著（此时 offloading 主导延迟、专家复用率低）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# CER 控制的内存预算 → 缓存替换策略
CER = available_expert_slots / total_experts        # 如 0.25/0.50/0.75
# 每个 GPU expert 槽位被以下对象占用（按优先级）：
#   1) 常驻：共享专家 + 当选的临时共享专家（整窗口常驻，STEP 固定预算下替换低使用率专家）
#   2) 热专家：窗口投票中高票者
#   3) 其他：按淘汰策略（STEP 用"当选替换低使用率"而非 LRU）
# 未驻留专家：命中→直接计算；未命中→PCIe 加载（与计算重叠）或 CPU 计算
```
Annotations：available_expert_slots=GPU 显存按专家大小折算的槽位总数，total_experts=模型全部专家数；CER 越低，缓存替换越频繁、预取命中率与推理性能对策略越敏感。相关但不同：SMoE 用 GPU cache ratio（GPU 命中率）作结果指标，CER 是 STEP 的输入控制变量；STEP 同时报告 Prefetch Hit Rate（预取命中率）作为策略质量指标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：评估框架按 CER 设置 GPU 上专家缓存容量（如把可驻留专家数设为 total×CER），逐层按容量执行缓存/替换/预取。使用场景：①内存约束敏感性分析（STEP Fig.10/11：不同 CER 下 TTFT/TPOT 对比七个 baseline）；②batch 与硬件敏感性分析（Fig.18/19：batch 1–8、V100/A100/H20 下不同 CER 的速度对比）；③指导部署——给定实际显存，按 CER 预估可缓存专家比例与预期加速。STEP 结果：各 CER 下 prefill/decode 均全面领先，低 CER（25%）优势最大（TTFT 3.12× vs llama.cpp），高 CER 下因专家复用率高、与 DAOP/APTMoE 差距缩小但仍最优。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
