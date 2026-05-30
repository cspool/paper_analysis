## Shared Expert (Residual MoE)

术语解释
Shared Expert 是 MoE 中预留一部分对所有 tokens 始终激活的固定专家，与动态路由专家配合。DeepSpeed-MoE (2022) 的 Residual-MoE 首次提出，DeepSeekMoE (2024) 推广为多共享专家的 fine-grained 设计。

术语是什么？
核心动机是解决"知识冗余"：多个 routed expert 可能重复学习通用知识。Shared Expert 捕获共性知识，让 routed experts 专注于细粒度专业化。

DeepSpeed-MoE: 1 fixed + 1 routed = top-1 通信开销获得 top-2 精度。
DeepSeekMoE: 多个 shared experts（如 2 shared + 64 routed），shared:routed ≈ 1:3。

从算法pipeline角度拆解术语。
```
def moe_with_shared_expert(x, shared_experts, routed_experts, router, K):
    y_shared = sum(SE_i(x) for SE_i in shared_experts)  # 固定激活
    logits = router(x)                                    # [batch, N_routed]
    topk_vals, topk_idx = TopK(softmax(logits), K)       # 稀疏激活
    y_routed = sum(topk_vals[e] * routed_experts[e](x) for e in topk_idx)
    return y_shared + y_routed
```
使用模型：DeepSpeed-MoE, DeepSeekMoE/V2/V3, OpenMoE, Qwen1.5-MoE, MoCLE, ARIA。

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 每 MoE 层有 2 shared experts (always active) + 64 routed experts (Top-6 per token)；shared experts 捕获通用跨模态知识，routed experts 发展出 modality-specific specialization）

术语一般如何实现？如何使用？
- 减少专家间知识冗余，提升参数效率
- 减少 All-to-All 通信量（shared expert 固定本地），有利于通信-计算 overlap
- 典型配置：总参数量的 10-20% 分配给 shared experts

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework（DeepSeek-MoE-16B 使用 2 shared + 64 routed 残差 MoE 架构；发现 shared experts 相比 routed experts 更不可压缩——pruning 不包含 shared experts 提升 Wanda 平均精度 +3.6%、SparseGPT +1.5%，因为 shared experts 对所有 token 激活，承载更关键和更通用的知识，对压缩更敏感）
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism（FinDEP 在 DEP 架构下对 Shared Expert 的调度视角：Shared Expert 在 DEP 中置于 Attention Group(AG)因需对所有 token 计算。关键发现：Shared Expert 与 A2E 通信无数据依赖——A2E 仅需 attention 输出即可发送，无需等 Shared Expert 完成。FinDEP 支持两种调度策略：AASS(All Attention then All Shared)使 A2E 最早启动，ASAS(Alternating Attention-Shared)使 AG GPU 利用率最高；通过 Algorithm 1 自适应选择最优顺序，解决原 PP-Pipe 将 Shared Expert 与 Attention 串行导致 GPU 空闲的问题。）
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs（Ling 采用单个 Shared Expert 配合 Fine-Grained Routed Experts，公式为 o_t' = o_t + E_share(h_t)，其中 o_t 为 routed experts 的加权输出。Shared Expert 无需路由，所有 token 均通过其计算，提供通用语言能力，使得 routed experts 可专注于专业化。Paper 指出仅靠 fine-grained experts 不足以同时发展通用和专用能力，Shared Expert 是必要补充。）
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（2 shared experts + 160 routed experts per MoE layer, shared experts 捕获通用知识，routed experts 通过 fine-grained segmentation 实现专业化）

---
