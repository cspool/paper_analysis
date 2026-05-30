## Logical Fusion of Attention Masks（and_mask / or_mask）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Logical fusion 是 FlexAttention 提供的 attention mask 组合机制，通过 `and_mask` 和 `or_mask` 两个组合算子，允许用户将两个独立的 mask_mod 函数按逻辑 AND / OR 组合成一个新的 mask_mod 函数。这解决了 attention 变体的"组合爆炸"问题：当研究者想组合多种 attention 修改（如 sliding window + document mask + causal mask）时，无需为每种组合手写新 kernel。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 FlexAttention 中：
1. `and_masks(mask1, mask2)` 返回一个新函数 `lambda b, h, q, kv: mask1(b, h, q, kv) and mask2(b, h, q, kv)`
2. `or_masks(mask1, mask2)` 返回一个新函数 `lambda b, h, q, kv: mask1(b, h, q, kv) or mask2(b, h, q, kv)`
3. 组合后的 mask 可以继续与其他 mask 组合（链式组合）
4. TorchDynamo 捕获组合后的完整函数图，TorchInductor 降低为 fused Triton 代码（两个 mask 的条件判断被融合到一个位置检查中）
5. `create_block_mask` 对组合 mask 生成 BlockMask，自动处理各种 mask 叠加后的 block sparsity

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
```python
from torch.nn.attention.flex_attention import and_masks, or_masks

# PrefixLM = causal OR prefix
lm_mask = or_masks(causal_mask, prefix_mask)

# Document causal = causal AND document
doc_causal = and_masks(causal_mask, document_mask)

# 三层组合
complex_mask = and_masks(or_masks(causal_mask, prefix_mask), sliding_window_mask)
```

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
