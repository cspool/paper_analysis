## SUPRA (Scalable UPtraining for Recurrent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SUPRA 是 Mercat et al. (2024, COLM) 提出的将预训练 softmax Transformer 大规模转换为线性 RNN 的方法。核心操作：(1) 用可学习 MLP kernel φ(x)=ReLU(Wx+b)（Q/K 共享 W）替换 softmax；(2) 用 GroupNorm 替换传统线性注意力的分母除法归一化，解决大规模数值不稳定性；(3) 引入 RoPE 相对位置编码；(4) 使用固定衰减向量 γ∈(0,1)^h。最终注意力形式：v'_i=GroupNorm(Σ_{j=1}^{i} γ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)。训练仅需约 5% 预训练 tokens（20B-100B），基于 OpenLM + Lightning Attention 2 Triton kernel，PyTorch FSDP + H100 集群。支持 Llama2 和 Mistral 作为 base model。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SUPRA uptraining 流程:
model = load_pretrained("Mistral-7B")
# 添加 MLP kernel: W_phi, b_phi (Q/K 共享)
for layer in model.layers:
    layer.W_phi = Linear(D, D, bias=True)

# 替换 attention + uptraining 5% tokens
def supura_attention(q, k, v):
    phi_q = RoPE(ReLU(q @ W_phi + b_phi))
    phi_k = RoPE(ReLU(k @ W_phi + b_phi))
    out = lightning_attn_ops(phi_q, phi_k, v, gamma)  # Triton kernel
    return GroupNorm(num_heads)(out)

# 推理切换为循环模式: s_i = diag(γ)·s_{i-1} + φ(k_i)·v_i^T, O(1) per token
```

关键洞察：不同于 T2R 的"近似 softmax"策略，SUPRA 直接替换 attention 机制，通过 uptraining 让模型学习新的计算范式。Appendix A 热力图证实 SUPRA 和 softmax 的 attention 矩阵差异很大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/TRI-ML/linear_open_lm (MIT)，模型：Mistral-SUPRA (https://huggingface.co/TRI-ML/mistral-supra)。7B 模型在 128 H100 GPU 上 uptraining 约 1.5 天。Mistral-SUPRA +100B tokens avg 64.0（vs Mamba-7B 1.2T tokens avg 64.7），仅 5% 训练成本。局限性：MMLU 大幅退化（34.2 vs Mistral 62.4），ICL 能力丧失（线性模型的已知瓶颈）。

涉及论文标题：
- Linearizing_Large_Language_Models

---
