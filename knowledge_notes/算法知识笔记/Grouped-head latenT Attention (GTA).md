## Grouped-head latenT Attention (GTA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-head latenT Attention (GTA) 是一种结合了 **shared attention map（共享注意力矩阵）** 和 **nonlinear value decoder（非线性值解码器）** 的高效注意力机制，由 Sun 等人于 2025 年提出。其核心思想是利用注意力计算中的冗余性——不同 head 的 attention map 高度相似，且 KV cache 可显著压缩——来同时减少预填充 FLOPs 和解码阶段的 KV cache 大小。

GTA 包含两个关键组件：

**(1) Shared Attention Map（共享注意力矩阵）**：将 query heads 分为 n_q 组、key heads 分为 n_k 组（n_q, n_k << n_h）。每个 head i 通过映射函数 q(i) 和 k(i) 分别分配到 Q group 和 K group，同一 Q group 内的 heads 共享同一套 QK^T 注意力计算。这从 MHA 的每 head 独立计算 n_h 次降至 n_q 次（n_q 为 Q group 数，<< n_h）。Key cache 仅需存储 n_k × d_h 维而不是 n_h × d_h 维。

**(2) Nonlinear Value Decoder（非线性值解码器）**：将 value cache 压缩为 latent space：引入 C = XW_C ∈ R^{N × n_c × d_l}（共享 latent value），每个 head 的 value V_i 由 V_i = (C_{c(i)}W_{P,i}) ⊙ Sigmoid(x_tW_{G,i}) 动态生成，而不是存储独立的 V_i。Latent dimension d_l ≥ d_h 以保证全秩投影不损失信息。Sigmoid gate 提供 context-adaptive 的非线性调制，增广 value 表示的有效秩。

这两个机制结合的效果是：KV cache 从 MHA 的 2n_h d_h N 降至 (n_k d_h + n_c d_l)N（1B 模型下仅 30% of GQA）；attention FLOPs 从 2n_h d_h N^2 降至 n_q(d_h + d_l)N^2（1B 模型下仅 37.5% of GQA）。

GTA 的预填充计算复杂度为 O(2NH^2 + (n_q d_h + n_k d_h + n_c d_h + d_l)NH + n_q(d_h + d_l)N^2)。解码时每步生成复杂度为 O(2H^2 + (n_q d_h + n_k d_h + n_c d_h + d_l)H + 2n_h d_h N)。关键效率优势来自 Eq 8 的 reformulation：将 attention 计算放在 latent space 上执行，decode 时无需从 latent vector 为每个 token 重新解压完整 value。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GTA 张量计算流程（以 500M GTA4 配置为例：n_h=20, n_q=10, n_k=1, n_c=2, d_h=64, d_l=256, H=1280）：**

```
# === Prefill Phase (N tokens) ===

# 1. 输入投影 (Eq 5)
Q = X @ W_Q            # W_Q ∈ R^{1280×640},  Q ∈ R^{N×640}  (10 Q groups × 64)
K = X @ W_K            # W_K ∈ R^{1280×64},   K ∈ R^{N×64}   (1 K group × 64)
C = X @ W_C            # W_C ∈ R^{1280×512},  C ∈ R^{N×512}  (2 C groups × 256)

# 2. Head-to-Group 映射
# q(i): head 0→Q0, head 1→Q0, ..., head 9→Q9, head 10→Q0, ...
# k(i): all heads → K0  (n_k=1)
# c(i): heads 0-9 → C0, heads 10-19 → C1  (n_c=2)

# 3. 分组计算 attention (仅 n_q=10 次，非 20 次)
for g in 0..9:
    Q_g = Q[:, g*64:(g+1)*64]       # (N, 64)
    S_g = Q_g @ K^T / sqrt(64)       # (N, N)  attention scores
    A_g = softmax(S_g)               # (N, N)  attention weights
    
    for each head i where q(i) == g:
        c_idx = c(i)                 # 0 或 1
        C_ci = C[:, c_idx*256:(c_idx+1)*256]  # (N, 256)
        
        # 4. 非线性 Value Decoder (Eq 6 → Eq 8 reformulation)
        # Latent-space attention: 直接对 latent C 做加权
        O_i_latent = A_g @ C_ci      # (N, 256)
        
        # Head-specific 投影 + context-adaptive gate
        O_i = (O_i_latent @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
        # W_{P,i} ∈ R^{256×64}, W_{G,i} ∈ R^{1280×64}
        
        # 5. 输出投影
        O_i = O_i @ W_{O,i}          # (N, 1280)

# 6. 合并所有 heads
O = sum(O_i for all heads i)        # (N, 1280)

# === KV Cache 写入 ===
# 仅存储 K (64 dims/token) 和 C (512 dims/token)
# 共计 576 dims/token/layer
# vs MHA: 2560 dims/token/layer = 22.5%
# vs GQA: 512 dims/token/layer (8×64) = 112.5% 反而更大
# 但实际 GTA4 的 n_k=1 让 K=64 极小，整体仍远小于 MHA

# === Decode Phase (1 new token) ===
# 追加 K_new (1,64) 和 C_new (1,512) 到 cache
# 对每组重新计算 score: S_g_new = Q_g_new @ K_all^T / 8
# A_g_new = softmax(S_g_new)  # 仅对 1 行 query 做
# O_i_latent_new = A_g_new @ C_all  # latent-space attention
# O_i_new = (O_i_latent_new @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
```

**GTA 配置变体（论文 Table 5）：**

| 配置 | n_q | n_k | n_c | d_l | KV cache dims | vs MHA |
|------|-----|-----|-----|-----|---------------|--------|
| GTA1 (160M) | 3 | 1 | 1 | 128 | 192 (64+128) | 12.5% |
| GTA2 (160M) | 6 | 1 | 1 | 128 | 192 (64+128) | 12.5% |
| GTA3 (500M) | 5 | 1 | 1 | 128 | 192 (64+128) | 7.5% |
| GTA4 (500M) | 10 | 1 | 1 | 256 | 320 (64+256) | 12.5% |
| GTA-1B | 5 | 1 | 1 | 128 | 192 (64+128) | 7.5% |

术语一般如何实现？如何使用？

实现方式（基于论文和 GitHub repo https://github.com/plm-team/GTA）：

1. **训练实现**：在 PyTorch 中替换标准 MultiHeadAttention 模块。关键实现点：(a) W_Q/W_K/W_C 投影——W_Q 输出 n_q×d_h 维、W_K 输出 n_k×d_h 维、W_C 输出 n_c×d_l 维；(b) Head-to-group 映射表维护；(c) Eq 8 reformulation 实现——在 latent space 计算 attention，避免 decode 时重复解压；(d) Gate 生成 ——对每个 head 维护 W_{G,i}，gate 仅在当前 token x_t 上计算（与序列长度无关）；(e) RoPE 应用于 Q 和 K 的投影后添加。

2. **训练配置**：AdamW optimizer，cosine LR scheduler，global batch size 800-2048，训练于 4 节点 32×A800 GPU。160M 和 500M 模型使用 C4 数据集（1 epoch），1B 模型使用 smollm-corpus（220B tokens）。

3. **SFT 微调**：使用 LlamaFactory [39] 框架和 tulu3-sft-mixture 数据集。

4. **推理部署**：使用 HuggingFace Transformers v4.36.0。支持 DynamicCache（标准）和 OffloadedStaticCache（缓存卸载）。FP16/BF16 和 FP32 均支持。

5. **评价**：使用 lm-evaluation-harness [25] 在 PIQA、HellaSwag、ARC、Winogrande、BoolQ、MathQA、TruthfulQA 等 benchmark 上评估。使用 LLM-Viewer [38] 做 roofline 模拟评估预填充/解码时延。

适用场景：资源受限设备上的 LLM 部署（NVIDIA H100/A800/RTX 3060/Apple M2/BCM2712），尤其是长上下文生成（论文测试至 4096 tokens）和需要同时优化预填充+解码延迟的场景。论文坦承缺乏工程级 kernel 优化（"The limitation stems from our lack of engineering-focused optimization efforts"），理论效率增益的上限尚未达到，未来结合自定义 GPU kernel（如 FlashAttention 风格融合 kernel）可进一步提升。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention

---
