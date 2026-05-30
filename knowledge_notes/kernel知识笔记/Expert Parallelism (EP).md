## Expert Parallelism (EP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Parallelism (EP) 是 MoE 模型分布式训练/推理的核心并行策略。在 EP 中，不同 expert 的权重被分配到不同 GPU 上（每个 GPU 持有部分 expert），token 通过 All-to-All 通信被路由到对应 expert 所在的 GPU 进行计算，结果再通过 All-to-All 返回原 GPU。EP 解决了 MoE 模型参数量巨大（如 256 experts × per-expert FFN）无法放入单 GPU 的问题，但也引入显著的 All-to-All 通信开销。

EP 在训练和推理中的关键差异：(1) 训练时通信常跨节点（多机），All-to-All 是主要瓶颈；(2) 推理时通信通常在节点内（单机多卡），NVLink 带宽充足，All-to-All 不再是首要瓶颈，反而是共享参数的每卡全量复制造成的内存浪费成为主要问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

经典 EP 推理流程（IFMoE 论文所述 Baseline）：

```
# EP=4, 每 GPU 持有不同 expert subset
# 每 GPU 复制完整 Attention + Norm + Shared Expert 参数

# Dispatch phase:
each GPU:
    router_outputs = Router(hidden)        # [local_tokens, num_experts]
    topk_indices = TopK(router_outputs, k=6)
    # All-to-All: send tokens to expert GPUs
    tokens_by_expert = AllToAll_Scatter(hidden, topk_indices)

# Compute phase:
each GPU:
    for expert_j in local_experts:
        expert_out[j] = ExpertFFN(tokens_by_expert[j])  # GroupedGEMM

# Combine phase:
each GPU:
    output = AllToAll_Gather(expert_out)   # return to origin GPU
    output += SharedExpert(hidden)         # dense
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EP 的通信模型：每个 token 传输 d_model 维数据两次（dispatch + combine），总通信量 = 2 × total_tokens × d_model × sizeof(dtype)。典型实现中 EP 与 DP（数据并行）正交组合——EP group 内切分专家，DP group 内复制模型。

IFMoE 对 EP 推理的改进：用 EP+TP hybrid（共享参数用 TP 切分，expert 参数用 EP），通信从 All-to-All 改为 double All-Gather（因节点内通信带宽充足，All-Gather 通信量与 All-to-All 相当但内存效率更高）。

**Lancet 对 EP 的通信开销量化**（Lancet, MLSys 2024）：

Lancet 在 GPT-2 MoE 训练的实验中量化了 EP 的 all-to-all 瓶颈：(1) all-to-all 通信占总训练时间最高 40%；(2) all-to-all 执行时间可达 expert 计算时间的 3.36x。因此传统仅重叠 all-to-all+expert 的方案（Tutel, FasterMoE）只能隐藏 expert 计算，all-to-all 仍主导 critical path。Lancet 通过全图重叠解决此问题——前向 pipelining non-MoE 计算（self-attention, FFN）与 all-to-all 重叠，反向调度 weight gradient computation (dW) 与 all-to-all 重叠。在 A100/V100 集群上减少 non-overlapped communication 最多 77% vs Tutel，端到端加速 1.3x。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Llama 3 Meets MoE: Efficient Upcycling

**LLEP 对 EP 的核心洞察与扩展**：

LLEP 指出标准 EP 的设计假设（每 GPU 负载始终近似均衡）在实践中不成立——训练良好的 MoE 模型会表现出持续的不均衡路由（专家专业化），且这种不均衡在 domain-specific post-training 或推理中是正确的/可取的。LLEP 从系统层面缓解不均衡，而不修改模型行为：
- **Least-Loaded Assignment (LLA)**: 在 dispatch 前，通过贪心算法将超载 GPU 的多余 token + expert 权重溢出到欠载 GPU。使用容量因子 α 硬限制每 GPU token 数，最小 GEMM token 数 m 约束避免低效微小 chunk 传输。
- **自适应 λ**: 当 max(l)/mean(l) < λ（如 λ=1.3）时回退标准 EP，避免不必要的 LLA 开销。
- **Backward 支持**: foreign expert 梯度通过 P2P 传回原生 GPU 累加，支持训练。
- **性能**: MoE 层 up to 6.1× speedup, 5× memory 节省 (H200, gpt-oss-120b)；端到端 gpt-oss-120b 1.88× speedup。
- **代码**: github.com/SalesforceAIResearch/LeastLoadedEP

**Lazarus 对 EP 的扩展分析**：

Lazarus 指出传统 EP 的两个关键缺陷：(1) Expert load 不均衡——gate network 动态路由导致某些 expert 收到远多于其他 expert 的 token（up to 87% tokens routed to 2 experts），等分 expert 到 GPU 导致 GPU 间计算不均衡；(2) 无弹性——EP 要求 GPU 数为 EP size 的整数倍，故障后可能有多余 GPU 空闲。Lazarus 通过 adaptive expert replica allocation（为 popular experts 分配更多 replicas 和 GPUs）+ flexible token dispatcher（CUDA kernel 处理非对称 placement 下的 token dispatch）+ flexible all-to-all（无 padding）来解决这些缺陷，允许任意 GPU 数下完全利用所有资源。

**LatentMoE 对 EP 的分析扩展**：

LatentMoE 从 hardware-software co-design 角度量化分析 EP 在不同 deployment regime 下的瓶颈（Section 2.1-2.2, GB200 NVL72）：

Memory BW Regime（低延迟，latency-critical）：
- Per-GPU memory traffic per MoE layer: M_exp = d·m + t_exp·(d+m) per expert
- 需要 t_exp ≥ 1418 (for Qwen3-235B, d=4096, m=1536) 才进入 compute-bound
- 典型 latency-critical serving 中 t_exp ~ 数百 → firmly memory BW bound

Communication Regime（高吞吐，throughput-oriented）：
- Communication cost per GPU: M_comm = 2.5·(N/EP)·t_exp·d (FP4+BF16 mixed precision)
- Communication-to-compute ratio ≈ 9:1 for GB200 + Qwen3-235B
- All-to-All 是主要瓶颈，占总执行时间的 ~90%

LatentMoE 的解决方案：通过在 latent space ℓ 中进行 EP 的 All-to-All 通信，per-token message size 从 d 降至 ℓ = d/α，同时通过增加 K'=αK 保持总通信量不变（ℓ-MoE_acc）或降低 α 倍（ℓ-MoE_eff）。Expert 权重加载的 memory BW 从 d·m 降至 ℓ·m（降低 α×）。

**Llama 3 Meets MoE 对 EP 的实践分析**：

论文在 128-512 H100 GPU 上使用 EP=8 训练 Llama 3-E8T2，总结了 EP 的关键调优实践：
- EP 通信是每层的 All-to-All token dispatch + combine，将其保持在 NVLink 域内（单节点 8 GPU）可最小化延迟
- MoE 层 EP 性能优于 TP（expert 独立计算，EP 仅需 token dispatch），TP 更适合 Attention 层
- AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
- 通过 MoE Parallel Folding 实现 Attention (TP1CP2) 和 MoE (EP8) 的异构并行映射，将 TP/CP group 折叠到 EP group 的 NVLink 域内
- 总 5-D Hybrid Parallelism: TP=1, EP=8, CP=2, PP=4, VPP=8, DP with ZeRO-1

**LSH-MoE 对 EP 通信瓶颈的分析与优化**：

LSH-MoE 在 V100 (100Gbps) 和 A100 (200Gbps) 集群上对 EP 的 all-to-all 通信瓶颈进行了详细 profiling：GPT-MoE (15B) 的 all-to-all 占训练总时间约 30%，RoBERTa-MoE 约 40%，Swin-MoE-L 约 70%，平均约 45%。LSH-MoE 的 scalability analysis 表明，该比例在更大模型和更多 GPU 下保持恒定——$\frac{T_{all\_to\_all}}{T_{compute}} = \frac{\text{FLOPs}}{6B_{inter}} \times \frac{k}{1+2k} \times \frac{w-1}{wh}$，其中 h 增长缓慢而 l 和 expert 数量增长较快。LSH-MoE 通过 LSH 聚类压缩 all-to-all 通信数据量（仅传 centroids 而非全部 tokens），实现 1.28×-2.2× 端到端加速，同时保持模型精度。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Llama 3 Meets MoE: Efficient Upcycling
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
