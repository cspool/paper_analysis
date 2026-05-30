## Fine-grained MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fine-grained MoE（细粒度专家混合）是一种 MoE 架构变体，相比传统 MoE（如 GShard、Mixtral）使用更多但更小的 expert。传统 MoE 通常每层配备 8-16 个 expert（每个 expert FFN 隐藏维度与 dense 模型相同或接近），而 fine-grained MoE 将 expert 数量增加到 64-256 个，每个 expert 的参数量相应缩小。典型代表为 DeepSeek-V2/V3 系列（每层 1 shared expert + 256 routed experts，top-8 激活）、Qwen2-57B-A14B（64 experts，top-6 激活）、Deepseek-Lite（64 experts）。

设计动机：(1) 更多 expert 意味着更细粒度的知识分工，每个 expert 可专精于更窄的知识领域，提升专家专业化程度；(2) 更小的 expert 使单次前向计算量降低（虽然激活更多 expert 以保持总参数量），训练成本降低；(3) 路由灵活性更高——64 选 6 的组合数（C(64,6) ≈ 7.5×10^7）远超 8 选 2（C(8,2)=28）。

推理挑战：(1) 激活 expert 数增多（如 top-6 vs top-2），GroupedGEMM 的 expert 数量增加导致 memory-bound 延迟上升；(2) 共享参数（Attention、Norm、Shared Expert）在传统 EP 下每 GPU 复制，expert 数量增多使共享参数内存占比相对降低但绝对值仍然可观；(3) 负载均衡在训练阶段通常良好（expert 小而多使 token 分布更均匀），但推理时 batch size 增大导致几乎所有 expert 被激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fine-grained MoE 的解码 pipeline（以 IFMoE 论文中 Deepseek-Lite-Chat 为例）：

```
# Fine-grained MoE (64 experts/layer, top-6 routing, 1 shared expert)
For each layer:
    hidden = Norm(input)
    
    # 1. Attention (dense)
    attn_out = Attention(hidden)
    
    # 2. Router
    gate_logits = Router(hidden)           # [B, 64]
    gate_probs = Softmax(gate_logits)      # [B, 64]
    topk_weights, topk_indices = TopK(gate_probs, k=6)  # 选 6 个 experts
    
    # 3. Routed Experts (sparse via GroupedGEMM)
    routed_out = GroupedGEMM(hidden, topk_weights, topk_indices, expert_weights)
    
    # 4. Shared Expert (dense, all tokens)
    shared_out = SharedExpertMLP(hidden)
    
    output = attn_out + routed_out + shared_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fine-grained MoE 的典型实现：DeepSeek-V2 使用 256 routed experts（top-8）+ 1 shared expert，每 expert FFN 维度约为 dense 模型的 1/16-1/32。训练使用 DeepSpeed-MoE 或 Megatron-LM 的 EP 并行。推理方面，IFMoE 提出用 EP+TP hybrid parallelism 减少共享参数内存浪费，并用 self-draft speculative decoding（减少激活 expert 数从 6→2）降低 GroupedGEMM 延迟。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- Deepseekmoe: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
