## Fine-grained SSM State Checkpointing

术语是什么？
Fine-grained SSM State Checkpointing是将纯Attention prefix caching扩展至Hybrid LLMs的naive baseline方案。由于SSM state的in-place递推特性（无法回滚），该方法每隔固定k个token保存一次SSM layer的完整recurrent state作为checkpoint。当后续请求匹配到某个checkpoint对应的前缀时，从该checkpoint恢复SSM state并继续forward计算。

该方案存在两个致命缺陷（Marconi论文核心motivation）：(1) Cache entries are sparsely-hit——大部分checkpoint位于无人复用的token位置（如对话中间而非开始或结尾）；(2) Cache entries are huge——单checkpoint覆盖所有SSM layers，NVIDIA Mamba2-Hybrid-7B约48MB/checkpoint（24 SSM layers × 2MB/layer），大量低命中率checkpoint迅速填满缓存。

从算法pipeline角度拆解术语：
```
Naive Algorithm:
  Prompt: "NYC is a busy city" (5 tokens), k=2
  Checkpoints:
    h_2 (position 2: "NYC is")      → 缓存
    h_4 (position 4: "NYC is a busy") → 缓存
    h_5 (position 5: full sequence)  → 缓存 (最后一个)

  新请求: "NYC is" + "new query" (shared prefix: "NYC is" = position 2)
    → 命中h_2 checkpoint → 恢复 → 从position 2继续prefill

  Problems:
  - h_4被缓存但几乎不会被复用（"NYC is a busy"不是自然的对话开始点）
  - 每k token一个checkpoint → k越小checkpoint越多 → 缓存越满
  - 长对话: 1000 token sequence, k=5 → 200 checkpoints × 48MB = 9.6GB仅一个序列
```

术语一般如何实现？如何使用？
vLLM+/SGLang+ baseline采用此方案（扩展原始框架的prefix caching以支持SSM states）。使用标准LRU eviction管理所有检查点。Marconi替代方案：每序列至多2个checkpoint（purely-input分支点 × 1 + leaf末尾 × 1），通过radix tree的复用模式识别精准定位值得缓存的token位置。vs fine-grained checkpointing token hit rate提升4.5×–34.4×。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---
