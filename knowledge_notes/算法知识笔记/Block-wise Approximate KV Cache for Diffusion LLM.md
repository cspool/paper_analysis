## Block-wise Approximate KV Cache for Diffusion LLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Approximate KV Cache是Fast-dLLM提出的针对双向注意力扩散LLM的KV缓存机制。自回归模型使用causal attention mask，已生成的token不会受未来token影响，因此KV Cache可以精确复用。但扩散LLM使用full bidirectional attention——任意token的计算依赖所有其他token，每步生成后所有token的注意力分布都可能改变，导致标准KV Cache不可用。Fast-dLLM观察到相邻推理步之间的KV激活余弦相似度接近1（Figure 3, red boxed region），意味着在块内（block）解码的多个步中，前缀token的Key和Value几乎不变，可以安全近似复用缓存。基于此，Fast-dLLM采用分块生成策略：将输出序列分为K个块（每块B个token，默认B=32），块内多步解码复用cache的prefix K/V，块完成后全序列forward更新cache再进入下一块。

块大小B是关键的精度-速度trade-off：B太小→频繁cache更新增加开销；B太大→缓存失配精度下降（Figure 4，B=32最佳）。

两种变体：(1) **PrefixCache**：仅缓存prefix（prompt+已生成块）的K/V；(2) **DualCache**：额外缓存suffix（全[MASK]末尾块）的K/V，进一步减少attention计算量。DualCache在长序列上加速更强（8-shot gen_len=1024: DualCache 27.6× vs PrefixCache 18.6×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PrefixCache单块解码流程（DualCache在step 2额外缓存suffix K/V）：

```
Input: x (全序列: prompt + [MASK]×L), block k, block_size B, steps T
   s = |prompt| + (k-1)*B         # 当前块起始索引
   e = |prompt| + k*B              # 当前块结束索引
   
   # Step 1: Cache初始化（仅首次调用）
   K_prefix, V_prefix = compute_KV(x[0:|prompt|])    # 缓存prompt的K/V
   
   # Step 2: 块内迭代解码（复用cache）
   for t = 1 to T:
       Q = x[s:e] * W_Q                             # 仅当前块作为query
       K_rest = x[s:] * W_K                          # 剩余部分（当前块+suffix）的K
       V_rest = x[s:] * W_V                          
       
       S_prefix = Q * K_prefix^T                     # (B, |p|)
       S_rest   = Q * K_rest^T                       # (B, L_rest)
       S = concat([S_prefix, S_rest])
       P = softmax(S)
       
       O = P_prefix * V_prefix + P_rest * V_rest     # attention输出
       # ... 后续FFN层 ...
       
       # confidence计算 + 解码 ...
       if all_unmasked(x[s:e]): break
   
   # Step 3: Cache更新（块完成后，与forward融合无额外开销）
   K_full = compute_KV(x[:])                          # 全序列KV重算
   K_prefix = K_full[:e]                              # 扩展prefix cache到当前块结束
   V_prefix = V_full[:e]
```

计算量对比（单attention step）：
- 无Cache: QK^T 需要 (B, |p|+L) × (|p|+L, d)^T → O(B·(|p|+L)·d)
- PrefixCache: 仅计算Q×K_rest^T，O(B·L_rest·d)，省去prefix部分（大比例）
- DualCache: 仅计算Q×K_block^T，O(B²·d)，省去prefix+suffix

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fast-dLLM在PyTorch eager模式实现：forward pass中，使用kvcache标志位控制是否跳过prefix/suffix的attention计算。cache存储在Python dictionary中（key: layer_id, value: (K_tensor, V_tensor)）。block size通过--block_size参数控制（默认32），使用lm-eval框架进行评估。DualCache需额外存储suffix位置的K/V，实现上将后缀token的position标记并在attention中对后缀部分使用null op。开源代码：https://github.com/NVlabs/Fast-dLLM（v1目录）。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
