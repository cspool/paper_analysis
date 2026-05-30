## Expert Activation Profiler

术语是什么？
Expert Activation Profiler是MoE-CAP实现的轻量级运行时profiling组件，用于追踪MoE推理过程中每层每个expert的激活状态。核心机制：在SGLang和HuggingFace Transformers的每个MoE layer的路由器（router/gate network）输出后植入probe tensor操作，记录当前forward pass中哪些expert被激活（布尔变量𝟙[l,i]），以及每个token被路由到哪些expert。Profiler兼容CUDA graph编译以最小化性能干扰——benchmark实测最大overhead仅2.7%（TTFT +8ms, TPOT +4ms）。Profiling数据持久化为activation sheet，后续评测可直接复用避免重复运行。

从系统架构角度拆解术语：
Expert Activation Profiler在CAP评测流水线中的角色：(1) 部署：在目标serving框架（SGLang/HuggingFace Transformers）的每个MoE block中，router.forward()之后插入hook记录top-k expert索引；(2) 数据采集：每次forward pass记录{batch_size, layer_idx, activated_experts[], token_to_expert_mapping[]}，prefill和decode阶段均采集；(3) 统计收敛：模型在代表性数据上运行至activation分布稳定；(4) 持久化：activation sheet存储每层每个batch size下的expert激活分布，支持后续S-MBU/S-MFU计算无需重跑模型；(5) 动态batching适配：逐次forward累加S_activated和KV cache size，按Σ_forward计算S-MBU。

术语一般如何实现？
使用轻量级tensor操作实现，兼容CUDA graph（不破坏graph capture）。支持vLLM和SGLang等主流框架。MoE-CAP开源实现：https://github.com/Auto-CAP/MoE-CAP。统计指标（S-MBU/S-MFU标准差）在各项实验中始终很低，不影响核心分析结论。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
