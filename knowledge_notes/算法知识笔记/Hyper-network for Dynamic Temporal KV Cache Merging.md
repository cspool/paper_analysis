## Hyper-network for Dynamic Temporal KV Cache Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hyper-network for Dynamic Temporal KV Cache Merging 是 MTLA 中用于动态生成 temporal merge weight 的轻量网络。背景：不同输入序列长度和内容各异，fixed-weight merging（如 static averaging）无法自适应。Hyper-network 以 latent vector C 为条件，通过 Sigmoid gate 输出 per-position merge weight w_i，使合并策略数据驱动。

结构（training batch）：W = Sigmoid(Linear(PE) × Linear(C)) ∈ R^{T×T}，chunk mask 后乘 C 得 Ĉ'。Inference single-token：w_i = Sigmoid(Linear(c_i) · Linear(pe_j))。两个 Linear 层各将 256-dim 映射到 64-dim。Sigmoid（非 Softmax）：不同 position 间无需归一化竞争——每个 w_i 仅控制对应 c_i 对 merged vector 的贡献比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Training batch 伪代码（s=2, r=256）**：

```
# 输入: C ∈ R^{T×256}, pe ∈ R^{t×64}
# 参数: W_pe: 256×64, W_c: 256×64

pe_proj = pe.repeat_interleave(s, dim=0) @ W_pe    # R^{T×64}
c_proj = C @ W_c                                    # R^{T×64}
W_raw = pe_proj @ c_proj.T                          # R^{T×T}
W = Sigmoid(W_raw)
W = chunk_mask(W)  # 仅保留 chunk 内连接
Ĉ' = W @ C         # R^{T×256}
```

**Inference single-token 伪代码**：
```
c_i ∈ R^{1×256}
j = ceil(i/s)
w_i = Sigmoid(Linear_pe(pe_j) @ Linear_c(c_i).T)   # scalar weight
```

术语一般如何实现？如何使用？

超参数：两个 Linear 层 256×64 = 32768 params（<0.04% 模型总量）。训练时与主模型共同训练，无额外损失。推理每次新增 token 仅两次 Linear + Sigmoid。Chunk mask 维持 streaming 属性，避免全局依赖导致训练不稳定。Hyper-network 输出 w_i 决定 c_i 对当前 slot merged vector ĉ_j 的贡献；当 i%s==0 时 ĉ_j 固化。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---
