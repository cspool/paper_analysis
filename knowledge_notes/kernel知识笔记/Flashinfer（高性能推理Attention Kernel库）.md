## Flashinfer（高性能推理Attention Kernel库）

术语是什么？

Flashinfer是一个针对LLM推理优化的高性能attention kernel库，由华盛顿大学和CMU开发。它提供模块化的GPU kernel实现，涵盖prefill attention、decode attention、paged attention等多种场景。Flashinfer的核心优势是**将（原本分开的）prefill attention和decode attention融合为单一kernel**（即POD-attention的前身），使chunked-prefill场景下prefill chunk的attention和decode batch的attention在同一kernel内完成，消除两次kernel launch和KV cache重复读取的开销。该特性被SGLang默认集成。

从kernel调度角度拆解术语：

Flashinfer的融合attention kernel执行流程（伪代码）：

```
// SGLang + Flashinfer 融合 attention kernel
// 输入: prefill_chunk_tokens (Q_prefill, K_prefill, V_prefill)
//       decode_batch_tokens (Q_decode)
//       kv_cache (所有历史K, V)
function fused_prefill_decode_attention(
    Q_prefill, K_prefill, V_prefill,  // prefill chunk的数据
    Q_decode,                          // decode batch的query
    kv_cache                           // 历史KV Cache
):
    // Step 1: Prefill self-attention
    // Q_prefill @ K_prefill^T (当前chunk内的attention)
    attn_prefill = flash_attention_block(
        Q_prefill, K_prefill, V_prefill
    )
    
    // Step 2: Prefill cross-attention with KV cache
    // Q_prefill @ KV_cache^T (当前chunk与历史context的attention)
    K_cached = kv_cache.load_all_layers()
    V_cached = kv_cache.load_all_layers()
    attn_prefill_cross = paged_attention_block(
        Q_prefill, K_cached, V_cached
    )
    
    // Step 3: Decode attention
    // Q_decode @ [KV_cache + new_KV]^T (decode token与所有历史的attention)
    K_all = concat(K_cached, K_prefill)
    V_all = concat(V_cached, V_prefill)
    attn_decode = paged_attention_block(
        Q_decode, K_all, V_all
    )
    
    // Key insight: 三步在单一kernel内完成
    // - 避免prefill和decode的两次独立kernel launch
    // - KV cache在shared memory / register中复用
    // - 实现了POD-attention的等效性能
    return attn_prefill + attn_prefill_cross, attn_decode
```

相比非融合方案（SARATHI-Serve原始实现的serial execution），Flashinfer融合kernel通过shared memory复用KV cache，消除了一次HBM往返。

术语一般如何实现？如何使用？

Flashinfer在SGLang中通过`from flashinfer import ...`直接调用。其kernel实现以CUDA/CUTLASS模板编写，针对不同head_dim（64/128/256）、不同dtype（fp16/bf16）和不同GPU架构（SM 8.0/9.0）编译多组特化版本。在MuxWise中，Flashinfer被用于decode iteration的attention kernel（通过CUDA Graph launch）。开源地址：https://github.com/flashinfer-ai/flashinfer。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

