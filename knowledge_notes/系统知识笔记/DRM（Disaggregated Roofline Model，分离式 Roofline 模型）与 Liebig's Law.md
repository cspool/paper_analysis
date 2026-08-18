## DRM（Disaggregated Roofline Model，分离式 Roofline 模型）与 Liebig's Law

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DRM 是 CHIME 提出的第一个面向 AFD 系统的通用性能分析模型，统一刻画 GPU 与加速器两条 "batch size—token 吞吐" 曲线及其交互：Line-FC（GPU 执行 FC 的 token 吞吐，随 batch 增长至 B_target 后饱和在 T_max_gpu）、Line-ATTN（加速器执行 attention 的吞吐，与带宽正相关、与 batch 基本无关，因 attention 算术强度恒定且低）、Line-Cap（加速器容量允许的最大 batch B_hp/B_dp/B_cpu）。三条推论：I 系统吞吐 = 两设备吞吐较低者；II attention 成为瓶颈时吞吐受加速器带宽限制（T_bw_cpu 类）；III FC 成为瓶颈时吞吐受 KV cache 容量限制（T_cap_hp 类）。由此归纳 Liebig's Law（借用农学"最小养分律"）：AFD 系统吞吐由加速器带宽/容量中较弱的一维决定，只扩较强一维收益递减甚至为零。DRM 还能刻画通信开销（PCIe 3.0 vs 4.0 使 ATTN 线下移）与动态负载（更长上下文使容量线左移、ATTN 线下移而 Line-FC 几乎不变）。DRM 是设计期分析模型，不预测单个请求的精确延迟。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
案例（Fig.2-c/Fig.3，8×A100 DGX-A100、L_t=2048、GPT-175B、AttAcc 模拟）：CPU 加速器的弱点 = 主机内存带宽（吞吐被 T_bw_cpu 钳制，容量再大也没用）；HBM-PIM 的弱点 = 容量（部署模型后仅 ~310GB 存 KV cache，batch 被 B_hp 卡死，GPU 利用率上不去，吞吐被 T_cap_hp 钳制，260.8TB/s 带宽大量浪费）；DIMM-PIM（2TB + 13.0TB/s）两维均衡 → 吞吐最高。部署配置会平移瓶颈：换更先进 GPU 上移 Line-FC 后，HBM-PIM 相对 DIMM-PIM 的差距收窄甚至反超（32× DIMM-PIM-L 配 A100 胜过 AttAcc、配 B200 则不如）——DRM 因此可用于加速器选型与容量/带宽配置决策（LoL-PIM/CXL-PNM/HPU 的配置可用 DRM 指导）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：曲线可理论推导也可 profiling 实测（token 吞吐 = FLOP/s ÷ FLOP/token，FLOP/s 由传统 roofline 按 batch 相关的算术强度给出）；对部署配置（GPU 型号/数量、上下文长度、模型尺寸/架构）做参数扫描得轮廓图。使用方式：设计期判断"瓶颈在带宽还是容量"，指导同时扩容量与带宽（CHIME 实测只扩容量 8× 得 2.28×、只扩带宽 8× 得 1.01×、两者同扩 8× 得 8.23×）；也可扩展分析 MoE（FC 层行为类似稠密模型）。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
