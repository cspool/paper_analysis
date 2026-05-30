## RoPE (Rotary Position Embedding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Rotary Position Embedding (RoPE) 是 Su et al. (2024, Neurocomputing) 提出的位置编码方法，通过旋转矩阵将位置信息编码到 attention 的 query 和 key 向量中。核心思想：对 Q 和 K 的每对维度施加基于绝对位置 m, n 的旋转，使 attention score Q_m^T K_n 仅依赖于相对位置 (m-n)：

$$f_Q(x_m, m) = R_{\Theta,m} W_Q x_m$$
$$f_K(x_n, n) = R_{\Theta,n} W_K x_n$$
$$f_Q(x_m, m)^T f_K(x_n, n) = x_m^T W_Q^T R_{\Theta,m-n} W_K x_n$$

旋转矩阵 R_{\Theta,m} 是分块对角矩阵，每块施加二维旋转：`[cos(mθ_i), -sin(mθ_i); sin(mθ_i), cos(mθ_i)]`，其中 θ_i = base^{-2i/d}, i=0..d/2-1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# RoPE: 对 Q 和 K 施加位置依赖旋转
# 输入: q, k [B, L, H, d], 位置索引 position_ids [L]

def rope_forward(q, k, position_ids, base=10000):
    d = q.shape[-1]
    # 计算频率: θ_i = base^{-2i/d}
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    
    # 计算 cos, sin 表
    pos = position_ids.unsqueeze(-1)  # [L, 1]
    freqs = pos * theta               # [L, d/2]
    cos = freqs.cos().repeat(1, 2)    # [L, d]
    sin = freqs.sin().repeat(1, 2)
    
    # 旋转: 每对维度 (2i, 2i+1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def rotate_half(x):
    # x = [..., x0, x1, x2, x3, ...]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    # x_rot = [..., -x1, x0, -x3, x2, ...]
    return x_rot
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RoPE 已成为现代 LLM 的标准位置编码：LLaMA、Qwen、DeepSeek、Hunyuan-Large 等均使用。Hunyuan-Large 在长上下文预训练阶段（256K）将 RoPE base frequency 从标准 10000 扩展到 1 billion (10^9)（参考 Xiong et al., 2023），以支持更长的上下文长度。实现上，HuggingFace Transformers 的 `LlamaRotaryEmbedding` 类可作为参考。RoPE 的效率优化包括：(1) 使用预计算的 cos/sin 表，(2) 与 FlashAttention 结合时在 kernel 内部完成旋转。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
