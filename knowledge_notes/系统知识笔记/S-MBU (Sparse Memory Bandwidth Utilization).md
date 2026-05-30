## S-MBU (Sparse Memory Bandwidth Utilization)

术语是什么？
S-MBU（稀疏内存带宽利用率）是MoE-CAP提出的专用于稀疏MoE系统的内存带宽利用率度量指标。传统MBU = B_achieved / B_peak = ((S_model + S_KV) / TPOT) / B_peak，其中S_model使用全部模型参数量，在MoE场景下会严重高估实际内存访问量。S-MBU将S_model替换为S_activated = n_layer × S_attn + Σ_{l=1}^{n_layer} Σ_{i=1}^{n_expert} 𝟙[l,i] × S_expert，其中𝟙[l,i]是布尔变量，表示第l层第i个expert是否被当前batch的token激活。S-MBU仅统计实际激活expert的参数内存访问，精度与profiler实测值误差<1%（Mixtral-8x7B验证）。dense模型（n_expert=1, ∀i 𝟙[l,i]=1）自动兼容为S-MBU=MBU。动态batching场景下，S-MBU = Σ_forward (S_activated + S_KV) / Σ_forward Latency，逐次forward累加实际激活参数。

从系统架构角度拆解术语：
S-MBU在MoE系统评测中作为性能维度的核心指标，直接影响硬件选型决策。流程：(1) 在SGLang/HuggingFace Transformers的每个MoE layer路由器后植入probe，记录forward pass中的𝟙[l,i]值；(2) 从模型配置获取S_attn（attention层参数量）和S_expert（单个expert参数量）；(3) 根据当前batch的token路由结果计算S_activated；(4) 结合实测TPOT（Time-Per-Output-Token）或Latency计算B_achieved；(5) S-MBU = B_achieved / B_peak。通过S-MBU可计算Practical Bandwidth = Theoretical Bandwidth / S-MBU，用于确定满足给定延迟SLO所需的最低硬件带宽。例如DeepSeek-R1在batch size=1时S-MBU揭示实际带宽需求仅1040 GB/s，远低于full activation的18901 GB/s，证明RTX 4090等消费级GPU配合offloading即可部署。

术语一般如何实现？
MoE-CAP在SGLang和HuggingFace Transformers中实现轻量级expert activation profiler：在每个MoE layer的router forward后插入probe tensor操作记录top-k选择结果，兼容CUDA graph编译以确保overhead < 2.7%。Profiler输出activation sheet持久化存储，后续评测复用。多节点场景验证：2节点×8 H20 GPU, 400 GB/s InfiniBand, 运行DeepSeek-R1在LongBench上，计算S-MBU与torch.profiler实测值对比，delta < 1%。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
