## HashEncode for LLM Attention (面向LLM注意力的哈希编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HashEncode是HATA中定义的哈希编码函数，将d维query或key向量编码为rbit-bit的二进制hash code，并以packed integer格式存储。其计算过程为：V_H = BitPack(Sign(V @ W_H))，其中W_H∈R^{d×rbit}是经过训练得到的hash权重矩阵。

HashEncode的三个步骤：(1) MatMul：V @ W_H将输入向量从d维投影到rbit维；(2) Sign：对投影结果逐元素取符号函数sign(x)∈{-1,1}，得到rbit个二进制位；(3) BitPack：将rbit个二进制位打包为rbit/32个INT32整数（如128-bit → 4 INT32），便于在GPU上高效存储和计算。

HashEncode的复杂度为O(s×d×rbit)，其中s是序列长度。由于rbit=128远小于s（典型值s≥32K），额外prefill overhead<1%。

从算法pipeline角度拆解术语：

```
# HashEncode (Algorithm 2 in HATA)
Input:  V ∈ R^{s×d}         # s tokens, d-dim each
Param:  W_H ∈ R^{d×rbit}    # trained hash weights, rbit=128

Step 1: V_H = V @ W_H       # [s, d] × [d, 128] → [s, 128], float
Step 2: V_H = Sign(V_H)     # [s, 128], values ∈ {-1, 1}
Step 3: V_H = BitPack(V_H)  # [s, 4], 128 bits → 4 INT32
Output: V_H ∈ N^{s×4}       # compact hash codes

# BitPack detail (128-bit → 4 INT32):
# binary_code = [b0, b1, ..., b127] where b_i = 1 if V_H[i] > 0 else 0
# packed[0] = Σ_{i=0}^{31} b_i * 2^i     (INT32, bits 0-31)
# packed[1] = Σ_{i=32}^{63} b_i * 2^{i-32} (INT32, bits 32-63)
# packed[2] = Σ_{i=64}^{95} b_i * 2^{i-64} (INT32, bits 64-95)
# packed[3] = Σ_{i=96}^{127} b_i * 2^{i-96} (INT32, bits 96-127)
```

HashEncode在prefill阶段编码所有keys并缓存K_H_cache；在decode阶段每步编码新query和新key，仅需O(d×rbit)计算。

术语一般如何实现？如何使用？

在HATA实现中（https://github.com/gpzlx1/HATA），HashEncode的MatMul在GPU Tensor Cores上执行（cuBLAS或custom CUDA kernel），Sign和BitPack通过fused CUDA kernel完成（kernelfusion for hash encoding）。推理时W_H作为固定权重加载。与FlashAttention-2和FlashInfer框架兼容。适用场景：任何需要快速query-key相似度比较的长上下文LLM推理任务。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference
