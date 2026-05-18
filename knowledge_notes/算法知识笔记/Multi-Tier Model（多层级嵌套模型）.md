## Multi-Tier Model（多层级嵌套模型）

术语是什么？通过联网搜索让回答具体和精准。
Multi-Tier Model（多层级嵌套模型）是MFS论文提出的一种模型结构设计，将LLM model family中不同规模的模型（如7B/13B/70B）折叠为单一checkpoint中的多个执行层级（tier）。每个tier在transformer的不同layer深度有独立的输出头（lm_head），低tier对应小模型（浅层，低延迟），高tier对应大模型（深层，高质量）。所有tier共享低层Transformer参数，不同tier之间通过Knowledge Precipitation保证各自的语言建模能力。该结构是MFS实现高效model family serving的基础——使得不同"模型大小"的请求可以共享参数、KV cache和batch执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-Tier Model在推理时的执行流（以Llama2-13B转3-tier模型：tier-1=18层/~3B等效, tier-2=32层/~10B等效, tier-3=40层/13B为例）：

```
def multi_tier_inference(x, requested_tier):
    hidden = x
    # Tiers 1-3 all share these layers
    for layer in range(0, 18):  # Tier-1 prefix
        hidden = transformer_block[layer](hidden)
    kv_cache.store(layers_0_17, hidden)  # Mark as shareable
    
    if requested_tier == 1:
        return tier1_lm_head(hidden)  # Early exit: ~3B quality
    
    # Tiers 2-3 share layers 18-31
    for layer in range(18, 32):  # Tier-2 extension
        hidden = transformer_block[layer](hidden)
    kv_cache.store(layers_18_31, hidden)
    
    if requested_tier == 2:
        return tier2_lm_head(hidden)  # ~10B quality
    
    # Tier-3 only: layers 32-39
    for layer in range(32, 40):
        hidden = transformer_block[layer](hidden)
    return tier3_lm_head(hidden)  # Full 13B quality
```

关键设计约束：
1. **Layer-only split**：只按layer切分tier，不切head维度。因为attention计算需要所有head一致性→切head会破坏各tier KV cache的维度一致性→无法跨tier共享KV cache。
2. **连续性**：tier边界是连续的layer范围，不是删除中间层（deep pruning）。层连续性保证增量计算正确（tier切换时第N+1层的输入hidden state = 第N层的输出）。
3. **Latency-aligned边界**：tier边界不按参数量线性切分（如一半参数量取一半层），而是测量最大模型前若干层在目标硬件的实际推理latency，使各tier的端到端latency对齐对应规模独立模型。例如Llama2-13B前24层latency对齐Llama2-7B，取24层而非20层（按参数量）。
4. **Nested hierarchy**：tier-1 ⊂ tier-2 ⊂ tier-3，即低tier层是高tier层的严格前缀。这确保了group batching和KV cache sharing的正确性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MFS的Multi-Tier Model通过Knowledge Precipitation（step-by-step fine-tuning）从最大模型的checkpoint构造。论文在Llama2-7B-chat → 2-tier、Llama2-13B-chat → 2-tier和3-tier（13B/7B/3B、13B/10B/7B组合）上验证，并在Qwen-14B → Qwen-7B上做泛化验证。质量评估显示各tier在MMLU/PIQA/OpenBookQA/HellaSWAG/BoolQ/ARC/ANLI等10个benchmark上的表现接近或优于对应规模的独立模型。该设计目前仅见于MFS系统，未见其他serving系统使用类似结构。该方法的局限：(1) 仅适用于同一model family（共用相同架构、tokenizer和vocabulary的模型）；(2) tier数量受层数和layer粒度限制；(3) fine-tuning成本随tier数和最大模型规模增长。

涉及论文标题：
- MFS: An Efficient Model Family Serving System for LLMs

