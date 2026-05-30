## Hybrid Model Architecture (混合模型架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Model Architecture 是将异构架构层混合在同一模型中的设计范式。在 LLM 中特指将标准 Transformer attention 层与内存高效替代层（Mamba/SSM、sliding/streaming attention、lightning attention）混合使用，平衡表达力与内存效率。已知实例：Jamba（Transformer + Mamba）、Gemma 2（sliding window + full attention 交替）、Minimax-01（lightning attention + full attention）、LightTransfer（streaming attention + full attention）。

从算法pipeline角度拆解术语。

**LightTransfer 的 Hybrid 转换（prefilling 阶段）**：
```
标准 Transformer → Hybrid Model:
  Layer_0: FullAttn  → 识别为 lazy → StreamingAttn (KV cache 缩减)
  Layer_1: FullAttn  → 非 lazy     → FullAttn (保留完整 KV cache)
  ...
  Layer_L: FullAttn  → 识别为 lazy → StreamingAttn (KV cache 缩减)
```

关键设计决策：Layer-wise（非 head-wise）hybrid。在 TP 下，head-wise hybrid 导致不同 GPU 的 KV cache 大小不一致产生同步瓶颈；layer-wise 保持同层内所有 head 一致，与 vLLM/SGLang 的 KV cache 粒度兼容。

术语一般如何实现？如何使用？

两种路径：(1) 从头训练（Jamba, Gemma 2——需大规模预训练）；(2) 从预训练 Transformer 转换（LightTransfer——~5K 训练样本或 zero-shot）。LightTransfer 仅修改 attention mask pattern（不改变权重），在 lazy 层丢弃 {sink + recent} 之外的 KV cache。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
