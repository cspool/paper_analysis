## DDLoRAdapt

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DDLoRAdapt (Data-Dependent LoRA Adaptation) 是GoldFinch在GOLD Attention中使用的参数高效token shift增强。定义为loradapt_□(x)=x+tanh(xC_□)D_□，C_□∈R^{H×r}、D_□∈R^{r×H}为低秩矩阵。与标准ddlerp（乘性插值：a+(b-a)⊙lora(...)）不同，DDLoRAdapt是加性的：在输入上叠加低秩tanh偏移。用于GOLD层从共享proto-keys和embedding生成层特异的k/v，使所有GOLD层共享压缩cache时仍学习不同attention模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
loradapt_□(x) = x + tanh(x @ C_□) @ D_□      // rank r, additive LoRA offset

// GOLD key with DDLoRAdapt:
a_t = lerp(x_t^0, x_{t-1}^0, μ_a)                     // embedding shift data
k_t_raw = lerp(k_t^D, k_{t-1}^D, lora_k(a_t))          // token-shifted proto-keys
k_t = LayerNorm(loradapt_k(k_t_raw))                   // DDLoRAdapt: layer-specific adaptation

// vs standard ddlerp (multiplicative):
ddlerp(a,b) = a + (b-a) ⊙ lora(a+(b-a)⊙μ_x)           // elementwise product with ratio
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
每GOLD层分别应用loradapt_k和loradapt_v。C/D rank推测≤32，初始化≈0使初始行为接近无adaptation。代码：https://github.com/recursal/GoldFinch-paper。

涉及论文标题：
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
