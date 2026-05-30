## Prefill-Decode Asymmetry in MoE Expert Sensitivity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Prefill-Decode Asymmetry 是 LYNX 发现并利用的 MoE 推理阶段非对称属性：prefill 和 decode 阶段对 expert selection fidelity 的敏感度存在根本性差异。在 prefill 阶段，expert reassignment 会显著降低模型性能（特别是在 code generation 和 complex reasoning 任务上）；在 decode 阶段，相同的 expert modification 仅产生 minimal accuracy impact。这种不对称性跨 task types（code, math, reasoning）一致成立，暗示它是 auto-regressive inference 的根本属性——prefill 建立 context 指导所有后续计算，而 decode 受益于 attention、residual connections 和累积 context 的补偿机制。

LYNX 利用此不对称性设计 Phase-Aware Optimizer：仅在 memory-bound decode iteration 中启用 expert remapping，prefill 和其他 compute-bound 阶段直接绕过。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Phase-aware expert reduction policy（基于 Prefill-Decode Asymmetry）

def moe_forward_with_lynx(batch, phase, memory_bound):
    if phase == "prefill":
        # Prefill: 严格保留所有 router 选择的 expert
        # 原因：prefill 建立 full context, expert fidelity critical
        return standard_moe_forward(batch)
    
    elif phase == "decode" and memory_bound:
        # Decode: 可以安全地 remap low-confidence experts
        # 原因：attention/residual/accumulated context 补偿 suboptimal selection
        return lynx_expert_remapping(batch)
    
    else:
        # Compute-bound decode (rare): skip LYNX overhead
        return standard_moe_forward(batch)

# Arithmetic intensity 差异（§2.1）:
#   Prefill: AI high (many tokens) → compute-bound → remapping 无益
#   Decode: AI = B × k / N → memory-bandwidth-bound → remapping 直接减少 HBM 流量

# LYNX Figure 4 实验验证:
#   Prefill expert reassignment → HumanEval accuracy 显著下降
#   Decode expert reassignment → accuracy minimal impact
#   跨 GSM8K 和 HumanEval 一致性 → 结构属性而非 task artifact
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 的 Phase-Aware Optimizer 集成在 vLLM batch scheduler 中，支持三种常见 serving policy：(1) Co-located prefill/decode：识别 pure-decode batches 为 memory-bound；(2) Disaggregated serving：直接标记 decode 实例为 memory-bound；(3) Chunked prefill：标记仅含 decode tokens 的 batch 为 memory-bound。含 prefill chunks 的混合 batch 被认为是 edge case，留给 future work。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
