## LoRA (Low-Rank Adaptation / 低秩适应)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LoRA 是一种参数高效微调（PEFT）方法，由 Hu et al.（2021）提出。核心思想：冻结预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，注入可训练的低秩分解矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$（$r \ll \min(d, k)$），将权重更新约束为低秩形式：

$$h = W_0 x + \frac{\alpha}{r} BA x$$

其中 $\alpha$ 为缩放超参数，$r$ 为秩（通常 2-64）。仅 $A$ 和 $B$ 可训练，参数量从 $d \times k$ 降至 $r(d+k)$，在 $r=8, d=k=4096$ 时可减少 >99% 可训练参数。推理时 $BA$ 可融合回原权重：$W = W_0 + \frac{\alpha}{r} BA$，无额外推理延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# LoRA Forward Pass (per linear layer)
def lora_forward(x, W_0, A, B, alpha, r):
    # W_0 frozen, A and B trainable
    # A init: Kaiming uniform, B init: zeros
    h_base = W_0 @ x              # frozen pretrained pathway
    h_lora = (alpha / r) * (B @ A) @ x  # low-rank adaptation pathway
    return h_base + h_lora

# Training: only A and B receive gradients
# Inference: fuse B@A into W_0, then just W @ x

# QLoRA variant: add 4-bit quantization
def qlora_forward(x, W_0_quantized, A_16bit, B_16bit, alpha, r):
    W_0_dequant = dequant(W_0_quantized)  # NF4 → BF16
    h_base = W_0_dequant @ x
    h_lora = (alpha / r) * (B_16bit @ A_16bit) @ x
    return h_base + h_lora
    # Key: double quantization of quantization constants for further memory saving
```

关键变体：QLoRA（将 W_0 量化为 4-bit NF4，再在 BF16 下训练 LoRA 参数，使 65B 模型可在单 48GB GPU 上微调）、DoRA（将预训练权重分解为 magnitude 和 direction 分量分别微调）、PiSSA（使用 SVD 初始化 A/B，将大奇异值分配给可训练矩阵加速收敛）、LoRA+（为 B 和 A 设置不同学习率）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HuggingFace PEFT 库（https://github.com/huggingface/peft）提供了 LoRA 标准实现，支持对 Transformer 的 Q/K/V/O 投影和 FFN 层注入 adapter。典型用法：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])`。秩选择经验：r=8 适用于大多数任务，r=64 接近 full fine-tuning 性能。QLoRA 由 bitsandbytes 库提供 4-bit 量化后端。Punica 和 S-LoRA 系统支持多租户 LoRA serving（多个 adapter 共享同一 base model）。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
