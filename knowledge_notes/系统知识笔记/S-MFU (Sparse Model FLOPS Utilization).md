## S-MFU (Sparse Model FLOPS Utilization)

术语是什么？
S-MFU（稀疏模型FLOPS利用率）是MoE-CAP提出的专用于稀疏MoE系统的计算利用率度量指标。传统MFU = (T_token × F_token) / F_peak，其中F_token假设所有参数参与计算。S-MFU修正为S-MFU = (T_token × S-F_token) / F_peak，其中S-F_token = F_attn + 2N_router + 2k_expert × N_expert。F_attn为attention模块所需FLOPs，N_router为router参数量，k_expert为实际激活expert数（含shared experts），N_expert为单个expert参数量。k_expert从模型配置直接获取无需运行时追踪。由于每个矩阵乘法的FLOPs固定且确定，S-MFU精度极高：与profiler实测值误差<0.05%（Mixtral-8x7B和DeepSeek-V2-Lite验证，batch size 1-32）。

从系统架构角度拆解术语：
S-MFU衡量MoE系统在给定硬件上实际的计算效率，用于识别compute-bound vs memory-bound瓶颈。流程：(1) 从模型config获取n_layer, n_expert, k_expert, shared_expert数, attention配置；(2) 根据模型结构解析F_attn（attention FLOPs = 4 × d_model² + 2 × seq_len × d_model，简化估算）、N_router（router参数量）、N_expert（单个FFN expert的参数量）；(3) 计算S-F_token = F_attn + 2 × N_router + 2 × k_expert × N_expert；(4) S-MFU = (实测T_token × S-F_token) / F_peak。通过S-MFU可计算Practical OPS = Theoretical OPS / S-MFU，用于确定满足吞吐要求的硬件算力。S-MFU揭示MoE模型的实际算力需求远低于dense假设：batch size=1时Mixtral-8x7B仅需0.06-0.08% MFU（A100）。

术语一般如何实现？
S-MFU为纯分析性指标，基于模型配置和token吞吐计算，无需profiler介入。MoE-CAP在自动化流水线中集成S-MFU计算模块：从HuggingFace model config自动解析F_attn/N_router/N_expert参数，结合benchmark实测T_token计算。S-MFU随batch size增大而上升（更多expert被激活），但不线性增长（token间共享expert激活）。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
