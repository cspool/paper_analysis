## GOLD (GPTAlpha Over Linear transformer Decoder)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GOLD是GoldFinch后1/3层的Transformer变体，基于GPTAlpha但移除per-layer W^K和W^V权重，改为从TokenCat解压的proto-keys和原始embedding生成k/v。GPTAlpha是独立改进版Transformer：将Llama SwiGLU FFN替换为Finch Channel Mixer (RWKV FFN)，attention层添加ddlerp token shift和额外LayerNorm。GOLD = GPTAlpha - W^K - W^V + TokenCat输入 + DDLoRAdapt。由于无per-layer K/V权重，所有GOLD层共享同一压缩key cache，无需per-layer K cache和value cache。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// GOLD Attention (per layer, per head):
q_t = LayerNorm(ddlerp_q(x_t, x_{t-1}) @ W^Q)           // query: per-layer W^Q
a_t = lerp(x_t^0, x_{t-1}^0, μ_a)                        // embedding token-shift data
k_t = LayerNorm(loradapt_k(lerp(k_t^D, k_{t-1}^D, lora_k(a_t))))  // key from proto-keys
v_t = LayerNorm(loradapt_v(lerp(x_t^0, x_{t-1}^0, lora_v(a_t))))  // value from embeddings
o_t = LayerNorm(concat(attention(q_t, K_{1:t}, V_{1:t}))) @ W^O

// vs GPTAlpha standalone (has per-layer W^K, W^V):
k_t = LayerNorm(ddlerp_k(x_t, x_{t-1}) @ W^K)
v_t = LayerNorm(ddlerp_v(x_t, x_{t-1}) @ W^V)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTAlpha可独立使用（ablation中GPTAlpha+RoPE loss=2.6684，vs Llama 2.7125，L12 D768）。GOLD专为GoldFinch hybrid设计。K/V重建利用token shift的隐式位置信息（Finch-C2 RNN自动编码位置），训练context内无需显式位置编码；需extrapolation时可选RoPE。开源于https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
