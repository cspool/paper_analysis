## Self-Draft Speculative Decoding with KV-cache Revision

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IFMoE 提出的 Self-Draft Speculative Decoding 是一种专门针对 fine-grained MoE 的投机解码变体。核心洞察：fine-grained MoE 模型在激活更少 expert 时（如 top-2 vs top-6）仍能保持较好的输出质量，因此无需额外 draft model——MoE 模型自身在"少 expert 模式"下就能作为 draft model。

与标准 SD 的关键区别：(1) 不用额外的 draft model，而是同一模型的不同配置（Dk=2 vs Ek=6）；(2) 接受所有 draft token（不做 rejection sampling），信任 fine-grained MoE 在小 expert 数下的输出质量；(3) KV-cache revision——每 α 步后用全量 experts 重算 KV-cache，因为 draft 阶段 attention 看到的 KV 是基于 2 个 expert 的 residual stream 产生的，与全量 6 experts 的 KV 存在偏差，revision 补偿这个偏差以保证后续 decode 的 attention 质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

IFMoE Self-Draft 算法（来自论文 Algorithm 1）：

```
Input: α=10 (draft steps), encode_topk Ek=6, decode_topk Dk=2, MoE model M
Initialize: terminate = False, buffer = []

while not terminate:
    # Draft phase: α steps with fewer experts
    for step in 1..α:
        token = M.decode(topk=Dk)   # 仅激活 top-2 experts
        buffer.append(token)
    
    # KV-cache Revision: recompute KV with full experts
    M.encode(buffer, topk=Ek)       # 全量 top-6 experts 重算
    
    terminate = detect_terminate()
    buffer = []
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

超参选择：α=10（每 10 步 draft 后 revision），Ek=6（全量 expert），Dk=2（draft expert）。IFMoE 在 Qwen2-57B-A14B 和 Deepseek-Lite-Chat 上验证，下游性能（XSum, GSM8K, TruthfulQA, IFEval）与 full model 接近（如 Qwen2 GSM8K: 75.4→71.1），推理延迟和吞吐均提升 >30%。

论文列为 Future Work：(1) 在高要求任务（如代码生成）中引入 logits-based rejection 和 rollback 机制；(2) 动态选择 expert 数——探索何时可减少 expert 数、何时需全量 expert。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
