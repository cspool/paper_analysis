## GLU / GeGLU (Gated Linear Unit / GELU-gated Linear Unit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gated Linear Unit (GLU) 是一种门控前馈网络结构，由 Dauphin et al. (2017) 在语言建模中首次提出。GLU 计算两个线性投影的 component-wise 乘积，其中一个经过激活函数作为门控：

$$GLU(x) = (xW_1 + b_1) \otimes \sigma(xW_2 + b_2)$$

其中 $\otimes$ 表示逐元素乘积，$\sigma$ 为 sigmoid 激活函数。

Shazeer (2020) "GLU Variants Improve Transformer" 系统研究了用不同激活函数替代 sigmoid 的变体，发现 GELU-gated (GeGLU) 和 Swish-gated (SwiGLU) 表现最佳：

$$GeGLU(x) = (xW_g + b_g) \otimes GELU(xW_v + b_v)$$

其中 GELU (Hendrycks & Gimpel, 2016) 为 $GELU(x) = x \cdot \Phi(x)$，$\Phi$ 是标准正态分布的 CDF。

在 GLaM 中，非 MoE 层使用 GLU + GeGLU 激活替代标准 ReLU+Linear：先计算 gate = GeGLU(x·W_g) 和 value = x·W_v，逐元素乘积 gate * value，最后通过 W_o 映射回模型维度。MoE expert FFN 内部也使用 GeGLU 激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GLaM 非 MoE FFN 层（使用 GLU + GeGLU）前向传播
# 输入: x [B, S, M=8192]
# 权重: W_g [M, H=32768], W_v [M, H=32768], W_o [H, M]

gate_logits = x @ W_g    # [B, S, H]
value = x @ W_v          # [B, S, H]

# GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
gate = GELU(gate_logits)  # GeGLU 激活

gated_output = gate * value  # 逐元素门控
output = gated_output @ W_o  # 输出投影

# 注：为保持参数量对等，H 约为标准 FFN hidden dim 的 2/3
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GLU 变体已成为现代 LLM 的标准组件：LLaMA 系列使用 SwiGLU（SiLU/Swish gate），GLaM 使用 GeGLU。实现时 hidden dim 缩减为 2/3 以补偿第三个 weight matrix 的参数开销。GELU 通过近似公式高效计算：`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`。在 HuggingFace 中 LLaMA 的 SwiGLU 实现为 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

**SwiGLU 变体**：SwiGLU 使用 SiLU (Swish) 作为门控函数替代 GELU：`SwiGLU(x) = (xW_g + b_g) ⊗ SiLU(xW_v + b_v)`，其中 `SiLU(x) = x · σ(x)`。Hunyuan-Large 在所有 FFN（包括 shared/specialized expert FFN）中使用 SwiGLU 作为激活函数。SwiGLU 已成为现代 LLM 最常用的激活函数，LLaMA、Qwen、DeepSeek 系列均采用。
