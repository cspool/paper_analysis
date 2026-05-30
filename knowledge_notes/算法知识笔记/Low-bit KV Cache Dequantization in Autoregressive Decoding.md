## Low-bit KV Cache Dequantization in Autoregressive Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Low-bit KV Cache Dequantization 是在自回归解码过程中将低比特（INT4/INT2）量化存储的 Key-Value cache 在 attention 计算前恢复为 FP16 精度以参与混合精度矩阵乘法的过程。与 low-bit weight 的 dequantization 不同，weight 是静态的、可离线预处理的——Marlin/Ladder 等 kernel 可在模型加载时完成所有 layout transformation。而 KV cache 是动态生成的：每个 decode step 都产生新的 K/V 并需要在线 quantize→pack→store，下一次 step 又需要 load→dequant→compute。这种"在线量化+在线解量化"的循环特性使 CUDA kernel 设计极具挑战——dequantization 成为 attention 计算的 critical path 而非一次性开销。

从算法pipeline角度拆解术语。

BitDecoding 中低比特 KV cache 的自回归解码全流程：

```
// Prefill 后
KV_cache_fp16 = [FP16 K/V of all prompt tokens]    // shape: L×d
Partition: X_pack = quantize_and_pack(KV[:L-N_r])   // → low-bit packed
           X_res  = KV[L-N_r:]                      // → FP16 residual

// Decode step t (autoregressive loop)
while not EOS:
    // Step 1: 新 token embedding
    x_t = embedding(token_t)

    // Step 2: QKV Projection (FP16 GEMM on TC)
    Q_t, K_t, V_t = x_t @ W_Q, x_t @ W_K, x_t @ W_V

    // Step 3: Dequant + Attention on packed KV cache
    for each tile in X_pack:
        K_fp16_tile = dequant(load_packed_K(tile), load_K_params(tile))
        V_fp16_tile = dequant(load_packed_V(tile), load_V_params(tile))
        S += Q_t @ K_fp16_tile^T         // mixed-precision GEMM
    S = softmax(S / sqrt(d))
    O = S @ V_fp16  (over same tiles)

    // Step 4: Residual KV attention (标准 FP16)
    O += FlashAttention(Q_t, X_res_K, X_res_V)

    // Step 5: Append new K/V to residual
    X_res_K.append(K_t)
    X_res_V.append(V_t)
    if len(X_res_K) == N_r:
        quantize_and_pack(X_res_K, X_res_V) → append to X_pack
        X_res_K, X_res_V = [], []

    // Step 6: FFN + LM head → next token
    token_{t+1} = argmax(lm_head(FFN(O)))
```

术语一般如何实现？如何使用？

BitDecoding 通过 Residual Kernel（在线量化+pack）和 Packing Kernel（在线 dequant+compute）实现。典型 4-bit 配置下 dequant 开销 <15% kernel time（vs QServe ~50%）。2-bit 下 <35%（dequant 更昂贵但 memory savings 更大）。开源：https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
