## Hybrid LLMs (Attention-SSM混合架构)

术语是什么？
Hybrid LLMs是将Attention层和State Space Model (SSM)层混合组成的语言模型架构。典型设计：少量Attention层（如4层）捕捉token间显式交互和检索能力，大量SSM层（如24层）通过recurrent state高效处理长序列。这种架构平衡了Attention的语言建模能力强（O(L²)计算复杂度、O(L) KV cache内存）和SSM的推理效率高（O(L)计算复杂度、O(1) fixed-size state内存）之间的tradeoff。

代表模型：NVIDIA Mamba2-Hybrid系列（4 Attention + 24 SSM + 28 MLP layers, 7B参数）、Jamba系列（1 Attention per 6-10 SSM layers, 最高398B参数）、Phi-4等。Marconi论文评估的默认模型为Mamba2-Hybrid-7B。

从算法pipeline角度拆解术语：
```
Hybrid LLM Forward Pass (per layer):

Attention layer (位置: 特定层索引):
  Input: X ∈ R^{L×d}
  1. Q, K, V = W_Q·X, W_K·X, W_V·X    // Linear projections
  2. A = softmax(Q·K^T/√d_k)·V         // Multi-head attention, O(L²·d)
  3. O = W_O·A                          // Output projection
  4. Store KV cache: (K, V) per token   // O(L·d) 内存, 可切片复用

SSM layer (位置: 大多数层):
  Input: X ∈ R^{L×d}
  1. B_t, C_t, Δ_t = Project(X_t)      // 输入依赖的选择性参数 (Mamba2)
  2. h_t = Ā_t·h_{t-1} + B̄_t·X_t        // Recurrent state update (in-place!)
  3. Y_t = C_t·h_t                      // Output
  4. Store SSM state: h ∈ R^{d_state×d} // O(1) 固定内存, 无法回滚!

MLP layer (每层后):
  gate = SiLU(W_g·X); up = W_u·X; down = W_d·(gate ⊙ up)
```

关键差异: Attention的KV cache是per-token的（可任意切片→prefix caching直接），SSM的recurrent state是per-sequence的（in-place更新→无法回滚→prefix caching需额外checkpoint机制）。
```

术语一般如何实现？如何使用？
实现：基于Mamba2 selective SSM kernel（CUDA官方实现），与FlashAttention交替排列构建Hybrid architecture。Marconi通过radix_cache_hybrid.py统一管理两者的prefix caching。评估workloads：LMSys/ShareGPT (conversational)和SWEBench (agentic)。趋势：Hybrid LLMs中SSM比例不断增加（更高效率），Marconi在higher SSM ratio下性能增益更大。

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---
