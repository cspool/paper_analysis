## MoE (Mixture-of-Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种神经网络架构，将传统 dense 模型中的前馈网络（FFN）替换为多个并行的"专家"子网络和一个可学习的门控机制（gate）。对于每个输入 token x，gate 计算一个分数向量 g(x) ∈ R^{|E|}，表示该 token 与每个专家 E_i 的亲和度，然后选择 top-k 个专家处理该 token，各专家输出加权求和得到最终结果：

$$MoE(x) = \sum_{i \in \tau} g(x)_i \cdot E_i(x)$$

其中 gate 通常是一个线性变换后接 softmax：g(x) = softmax(W_g · x)。专家网络 E_i 通常实现为标准 FFN。MoE 的核心优势是 sparse activation——每个 token 只激活 k 个专家（k << |E| 总数），因此在增大模型总参数量（capacity）的同时，计算量仅随 k 线性增长而非随专家总数增长。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FOLDMOE 论文中，GPT-MoE 模型每隔一个 Transformer block 将 FFN 替换为 MoE 层（使用 top-1 GShard gate）。一个 Transformer-MoE block 的算法 pipeline：

```
# 输入: sequence X = [x_1, x_2, ..., x_n]
# 阶段1: Attention
for t in 1..n:
    q_t, k_t, v_t = W_q(x_t), W_k(x_t), W_v(x_t)
    z_t = softmax(q_t @ K_{1:t-1}^T / sqrt(d_k)) @ V_{1:t-1}

# 阶段2: MoE with top-1 gating
for t in 1..n:
    g(z_t) = softmax(W_g @ z_t)         # gate scores
    expert_idx = argmax(g(z_t))          # top-1 routing
    y_t = E_{expert_idx}(z_t) * g(z_t)[expert_idx]
```

在分布式训练中，专家分布在多张 GPU 上（Expert Parallelism），gate 计算后需要 all-to-all dispatch 将 token 发送到对应专家所在 GPU，计算完成后 all-to-all combine 收集结果回原 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE 的实现通常基于现有训练框架：
- **Megatron-LM**: 提供 Megatron-MoE 实现，支持 EP + TP + DP 混合并行
- **Tutel**: 提供自适应 MoE，支持 token-level overlapping
- **DeepSpeed-MoE**: 微软的 MoE 实现，集成在 DeepSpeed 中
- **FairScale / PyTorch FSDP**: 通过专家并行方式支持 MoE

FOLDMOE 基于 Megatron-LM 框架，将每层 MoE 隔一层插入 Transformer block（alternating dense-MoE pattern），使用 top-1 GShard gate 和 capacity factor=1.0。

FSMoE 通过模块化设计将 MoE 层分解为 6 个子模块（Gate、Order、I-Order、Dispatch、Combine、Expert），预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice）。这种非侵入式模块化使得新增路由函数或通信算法只需继承对应基类，无需修改调度器代码。FSMoE 的在线 profiler 对各子模块执行时间建模，为调度器提供性能数据。

Fair-MoE 提出 FO-MoE（Fairness-Oriented MoE）变体，在医疗 VLM 中将 embedding-based MoE 和 feature-based MoE 两级结构集成到 CLIP encoder 中，通过 expert capacity 过滤偏置 patch embedding 以提取公平特征。

FasterMoE（PPoPP'22）进一步从分布式训练角度分析了 MoE 的动态特性：(1) 训练数据的偏斜分布导致 expert 热度高度不均衡且随 iteration 动态变化（图 4 可视化），热门 expert 可接收 3.2× 平均的 tokens；(2) MoE 允许在不增加计算量的前提下增大模型参数量——weight matrices 沿特定维度切分，每部分仍产生同尺寸输出，但 GeMM 计算量保持较小；(3) 在 transformer 中，MoE 层通常替换 MLP 层中的密集 FC 层，gate 是一个小型 FC 层（计算 fit score 并取 top-k）。

Fiddler 从资源受限环境推理角度利用了 MoE 的两个关键特性：(1) MoE 的参数量-计算量不对称性——模型总参数可极大（>90GB for Mixtral-8x7B FP16），但每 token 仅激活 top-2 expert（~12.5% 参数），使得 CPU 在 small-batch 场景下的低计算能力仍可承接部分 expert 计算；(2) Expert 的独立可分离性——每个 expert 的权重和计算是完全独立的，可以被独立分配到不同设备（GPU 或 CPU）执行，无需跨 expert 通信。这是 Fiddler 的 per-expert CPU/GPU 动态调度策略成立的前提——若 expert 之间共享权重或有数据依赖，则无法独立决策每个 expert 的执行后端。

Flex-MoE 将 SMoE 应用于**多模态 missing modality** 场景：将 Transformer 的 FFN 替换为 SMoE layer，expert 索引按 modality combination 分配（如 "IGCB"=0, "IGC"=1, ..., "B"=14），剩余 index 作为 buffer expert。核心创新是两阶段训练：(1) Generalization——全模态样本通过 G-Router 训练通用 expert 知识；(2) Specialization——S-Router 通过 cross-entropy loss 将 top-1 gate 绑定到目标 modality combination expert。batch 内 samples 按可用模态数降序排列（课程学习），encoder 仅用对应 modality 的 observed 样本训练，缺失嵌入从 Missing Modality Bank 查找。

GLaM (Google, 2022) 进一步展示了 MoE 在 decoder-only LLM 上的大规模实践。GLaM (64B/64E) 拥有 1.2T 总参数，64 个 expert，每 token 通过 top-2 softmax gating 仅激活 2 个 expert（96.6B 活跃参数，占总参数 8%）。GLaM 的架构特征：(1) 每隔一层 Transformer FFN 替换为 MoE 层（alternating pattern），MoE 层 expert 的 hidden dim H=32768；(2) top-2 gating：gating_logits = softmax(x @ W_gate)，选 top-2 expert 并归一化 gate 值，加权组合输出；(3) GShard auxiliary load balancing loss 系数 0.01；(4) GSPMD 2D sharding 将 expert 权重 [E, M, H] 沿 E 维和 H 维划分到 TPU-v4 集群的 2D device mesh 上。GLaM 证明了 sparse MoE 在 few-shot in-context learning 任务上超越同等计算量的 dense 模型：at similar FLOPs/token, MoE (64B/64E) 在 29 个 NLP benchmark 上平均 zero/one/few-shot 性能均高于 GPT-3 (175B dense)，而推理 FLOPs/Token 仅为 GPT-3 的 51.4%，训练能耗仅为 GPT-3 的 1/3。

Hecate 进一步揭示了 MoE 训练中的 **expert load 动态性**：gate 的频繁演化导致 expert load 快速波动和不平衡（图 3 可视化，不同 expert 的 token 比例在 iteration 间显著变化），导致 EP 的严重 straggler 效应（最坏情况下性能下降 5.18×）。为解决此问题，Hecate 提出 FSSDP 范式：将 MoE layer 的 parameters 和 optimizer states 完全分片到所有 device，每次 iteration 用 SparseAllGather 从 shards 零构建临时 expert placement，用 SparseReduceScatter 同步 gradients 回 source device，消除 traditional expert rearrangement 的 memory/timeliness 两难困境。

LExI (Chitty-Venkata et al., 2025) 进一步从 layer sensitivity 角度丰富了 MoE 的理解：(1) 不同层的 expert 冗余程度差异显著——某些层对减少 active expert 数量高度不敏感（低 Frobenius 范数扰动），而其他层则高度敏感；(2) 传统的固定 top-k 假设所有层需要相同数量的 active expert 是不合理的，layer-adaptive top-k 可以更高效地分配计算资源；(3) MoE 推理中 load imbalance 不仅来自 token-to-expert 路由的不均匀，也来自固定 top-k 导致的跨层计算冗余——即使每层 expert 负载均衡，统一 top-k 仍可能在低敏感层浪费计算；(4) Expert 冗余可以通过仅使用模型权重的 data-free sensitivity profiling 量化，无需 calibration 数据集。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Layerwise Recurrent Router for Mixture-of-Experts
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

Hunyuan-Large 是目前最大的开源 Transformer-based MoE 模型（389B 总参数, 52B 激活参数, 256K 上下文），采用 64 layers、1 shared expert + 16 specialized experts（top-1 激活）、GQA (8 KV groups) + CLA (每 2 layers 共享 KV)、SwiGLU 激活、RoPE、128K tokenizer。预训练 7T tokens（含 1.5T 合成数据），后训练 SFT + DPO。证明了 MoE 在大规模开源模型上的有效性——52B 激活参数在 MMLU (88.4)、MATH (69.8)、HumanEval (71.4) 等 benchmark 上超越 LLama3.1-405B。

Joint MoE Scaling Laws (Ludziejewski et al., 2025) 进一步从 Scaling Laws 角度研究了 MoE 的 compute/memory efficiency。使用 Switch MoE (top-1 routing) 训练 280+ 模型（最高 5B total params, E∈{1,2,4,8,16,32}），推导 joint scaling law L(N_act, D, Ê) = aÊ^δ·N_act^(α+γ·ln(Ê)) + bÊ^ω·D^(β+ζ·ln(Ê)) + c，证明 MoE 在 memory-constrained 场景下可超越 dense 模型，打破"MoE memory-inefficient"的传统认知。

LYNX (Gupta et al., 2025) 从 MoE 推理角度揭示了 batch 级别 expert 选择的关键特性：(1) Training 时的 load-balancing loss 导致 inference 时 batch 级 expert activation 存在系统性冗余——虽然 aggregate 分布均匀（变异性 ~1.2%），batch 级分布显著偏斜（变异性 ~15-20%）；(2) Decode 阶段的 arithmetic intensity = B×k/N，在 moderate batch size 下 MoE decode 是 memory-bandwidth-bound——42% decode latency 花在 HBM 加载 expert 权重上；(3) Top-1 expert 主导输出质量，lower-ranked experts 高度冗余，这种 Expert Rank Hierarchy 跨 tasks 一致；(4) Prefill 和 Decode 对 expert fidelity 的敏感度存在根本性不对称——prefill 需严格 fidelity（建立 context），decode 因 attention/residual/context 补偿而高度容错。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Layerwise Recurrent Router for Mixture-of-Experts
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
