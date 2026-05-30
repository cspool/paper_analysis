## Streaming KV Cache (Dual-Level)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Streaming KV Cache（流式键值缓存）是 LiveStar 中用于加速流式视频推理的双级缓存架构。标准 KV Cache 将 Transformer decoder 每层已计算的 Key/Value 投影存储在 GPU 显存中，避免新 token 生成时重算历史的 K/V。LiveStar 在此基础上扩展为双级结构：(1) **Intra-dialogue KV Cache**（对话内缓存）—— 在一个对话轮次内（即一个语义片段内），缓存逐帧处理的帧级 K/V，使得每帧的 vision encoding 结果只需计算一次，后续 verification 复用；(2) **Inter-dialogue Streaming Cache**（对话间流式缓存）—— 跨多个对话轮次（多个语义片段），缓存历史帧的 K/V 表示，使得新的 incoming frame 只需与新帧交互而不重算整个视频历史。双级 cache 的关键挑战是：(a) SVeD swap 操作（沉默时交换 Ctx 最后两个元素）后的 cache 序列完整性维护；(b) Peak-End Memory Compression 剪枝操作后的动态 cache 长度适配。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 LiveStar 流式推理系统架构中的运转流程：

```
Streaming SVeD Inference Loop:
  for each incoming frame Frm^{t_j}:
      1. Vision Encoding (InternViT) → frame tokens [16, D]
      
      2. Cache Lookup (Inter-dialogue):
         # 从 inter-dialogue cache 加载历史帧的 K/V
         # 无需重算之前所有帧的 K/V
         hist_K, hist_V = inter_dialogue_cache.load(up_to=t_j-1)
      
      3. Forward Pass (Verification):
         # 仅计算新帧的 K/V，concat 到历史 cache
         new_K, new_V = compute_KV(Frm^{t_j})
         K = concat(hist_K, new_K); V = concat(hist_V, new_V)
         PPL = attention_and_ppl(Q, K, V, Dec_tokens)
      
      4. Gate Decision:
         if PPL > alpha * PPL_ref:
             # 解码: 需要完整generate
             new_tokens = generate(K, V)
             Dec = tokenizer.decode(new_tokens)
             # 更新 intra-dialogue cache（当前clip的K/V）
             intra_cache.update(Dec_tokens)
             inter_cache.append(Dec_tokens)
         else:
             # 沉默: swap操作
             # Cache中Dec从响应位置移到末尾
             swap_last_two_in_cache(inter_cache)
      
      5. Memory Management (every W frames):
         if cache_size > threshold:
             # Peak-End 剪枝: 移除低重要性旧帧的K/V
             pruned_indices = peak_end_select(cache, PPL_scores)
             inter_cache = inter_cache[pruned_indices]
             intra_cache = intra_cache[pruned_indices]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于 PyTorch 的 KV Cache 接口扩展：(1) 在 forward 时传入 `past_key_values`，模型返回 `present_key_values`；(2) 双级缓存通过两个独立的 `past_key_values` 列表管理（intra 存当前 clip 帧级 cache，inter 存跨 clip 历史）；(3) SVeD swap 操作后需同步更新 cache 中的位置索引，确保 positional encoding 对应正确；(4) Peak-End 剪枝时需对每层的 K/V 做索引切片 (`K_pruned = K[:, valid_indices]`)。性能：在 5 分钟视频上，双级 cache (Both) 实现 FPS 3.82，vs 仅 intra-dialogue (w/o Inter-Dialog) FPS 2.92，vs 无 cache (Neither) FPS 2.50，加速 1.53×。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
