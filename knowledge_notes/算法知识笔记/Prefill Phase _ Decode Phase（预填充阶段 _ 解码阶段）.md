## Prefill Phase / Decode Phase（预填充阶段 / 解码阶段）

术语是什么？

LLM推理的两个核心阶段：
- **Prefill Phase（预填充阶段）**：处理输入prompt的所有token，一次性计算所有位置的Key和Value并存入KV Cache，同时生成第一个输出token。计算特征为**compute-bound（计算密集）**，涉及大规模矩阵乘法（prompt_length × hidden_dim）、并行注意力计算和FFN。
- **Decode Phase（解码阶段）**：每步生成一个新token，计算该token的Query并与KV Cache中的所有历史Key/Value进行注意力计算。计算特征为**memory-bound（内存密集）**，每步计算量极小（约2×参数量FLOPs），HBM访问是主要瓶颈。

从算法pipeline角度拆解术语：

完整LLM推理pipeline伪代码：

```
// ===== Prefill Phase =====
function prefill(input_tokens[0..L-1], model_weights, kv_cache):
    // Step 1: 逐层处理
    for layer in 0..num_layers-1:
        // Step 1a: QKV投影（矩阵乘法，compute-bound）
        Q[layer] = input_tokens[layer] @ W_Q[layer]    // [L, d] @ [d, d]
        K[layer] = input_tokens[layer] @ W_K[layer]
        V[layer] = input_tokens[layer] @ W_V[layer]
        
        // Step 1b: 将KV存入Cache
        kv_cache.store(layer, K[layer], V[layer])       // HBM写入
        
        // Step 1c: 自注意力（FlashAttention kernel）
        attn_out[layer] = flash_attention(Q[layer], K[layer], V[layer])
        // 计算复杂度: O(L^2 * d)  — L为prompt长度
        
        // Step 1d: FFN + 残差
        input_tokens[layer+1] = ffn(attn_out[layer]) + input_tokens[layer]
    
    // Step 2: 从最后一层采样第一个token
    first_token = sample(input_tokens[num_layers])
    return first_token
// Prefill阶段总计算量: O(L * d^2 + L^2 * d)
// GPU利用率特征: 高SM占用率(>80%)，高计算/访存比

// ===== Decode Phase =====
function decode(last_token, model_weights, kv_cache):
    // Step 1: 逐层处理
    for layer in 0..num_layers-1:
        // Step 1a: 仅计算新token的Query
        q[layer] = last_token[layer] @ W_Q[layer]       // [1, d] @ [d, d]
        
        // Step 1b: 从KV Cache加载所有历史K, V（HBM读取瓶颈）
        K_all = kv_cache.load(layer)                     // HBM读取 [L, d]
        V_all = kv_cache.load(layer)                     // HBM读取 [L, d]
        
        // Step 1c: 单token注意力
        attn_out[layer] = paged_attention(q[layer], K_all, V_all)
        // 计算复杂度: O(L * d) — 远小于Prefill
        
        // Step 1d: FFN + 残差
        last_token[layer+1] = ffn(attn_out[layer]) + last_token[layer]
    
    // Step 2: 采样下一个token
    next_token = sample(last_token[num_layers])
    return next_token
// Decode阶段每token计算量: O(d^2 + L * d) — L为累积序列长度
// GPU利用率特征: 低SM占用率(<10-30%)，低计算/访存比，HBM带宽瓶颈

// ===== LLM推理主循环 =====
kv_cache = init_kv_cache()
next_token = prefill(prompt_tokens, weights, kv_cache)
output_tokens = [next_token]

while next_token != EOS and len(output_tokens) < max_len:
    next_token = decode(next_token, weights, kv_cache)
    output_tokens.append(next_token)
```

两阶段的GPU利用率特征差异是PD Multiplexing的理论基础：Prefill高计算低访存，Decode低计算高访存，两者在同一GPU上并行可以互补。

术语一般如何实现？如何使用？

实际推理框架中：
- **Prefill**：通常使用FlashAttention/FlashInfer等高效kernel，将prompt tokens一次性经所有Transformer层处理。
- **Decode**：使用PagedAttention管理KV Cache分页，减少HBM碎片；每token逐层处理，利用KV Cache避免重复计算。
- **Chunked Prefill**：将长Prefill按token切成多块交替执行，平衡TTFT和ITL。
- **PD Multiplexing**：利用两阶段不同的GPU利用率特征，在同一GPU上SM空间分区并行执行。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

