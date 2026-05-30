## Radix Tree for Prefix Caching

术语是什么？
Radix Tree在prefix caching中作为请求历史bookkeeping数据结构。节点=tokens序列前缀，边=单个token。Marconi利用其两个性质：(1) Intermediate nodes（非叶）→ purely-input前缀（多请求共享）→ 高复用；(2) Leaf nodes → input+output前缀（对话末尾）→ 仅末尾复用。通过speculative insertion（admission前tentatively插入请求）检测branching point判断复用模式。

从系统架构角度拆解术语：
```
Radix Tree: root → node_0 → node_01 → {leaf_A, leaf_B, leaf_C}
  node_01有3 descendant → purely-input → admit
  leaf_A/B/C → input+output → 仅缓存末尾SSM state
节点存储：token序列、KV cache、SSM states（仅admission节点）、复用标记
```

术语一般如何实现？如何使用？
Marconi在radix_cache_hybrid.py中实现统一radix tree（管理KV+SSM states，因所有layer states须代表同一prefix）。radix_cache_vllm.py提供vLLM适配。支持纯Transformer、纯SSM和Hybrid LLMs三种架构。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs
