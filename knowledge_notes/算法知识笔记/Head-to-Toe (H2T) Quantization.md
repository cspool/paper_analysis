## Head-to-Toe (H2T) Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Head-to-toe (H2T) quantization 指从 embedding 层到 SSM blocks 到 lm_head（输出层）的全模型量化——不留任何 FP16 层。之前的 SSM 量化方法（MambaQuant, Quamba）仅量化 SSM blocks，embedding 和 lm_head 保持 FP16（见表 2）。Quamba2 在 W4A8 设置下，embedding 层使用**per-token quantization**（每 token 独立 scale），lm_head 权重使用**per-group quantization**（分组量化），实现了 4× 全模型显存减小（如 Mamba2-2.7B W4A8：从 FP16 5.2GB 降至 1.4GB）。H2T 的关键价值在于**边缘设备部署**：仅当 embedding 和 lm_head 也被量化，8B 模型才能在 Orin Nano 8G 上运行（FP16 OOM, W8A8 也 OOM, 仅 W4A8/W4A16 可行），实现 13 tokens/s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 各层量化方式
# Embedding: per-token 量化（输入 token embedding）
e = embedding[token_ids]                              # FP16 lookup
s_e = max(|e|) / 127.0                                # per-token scale
e_quant = clamp(round(e / s_e), -127, 127)            # INT8

# SSM blocks: W4A8/W4A16（sort-and-cluster + per-state-group）
# ... (标准 SSM block 量化流程)

# lm_head: 4-bit per-group weight + FP16 activation
W_lm_head_4bit, s_lm = quantize_per_group(W_lm_head)  # per-group 4-bit
logits = (W_lm_head_4bit * s_lm) @ x_final             # GEMM with dequant
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
小型模型（130M, 370M）的 embedding 量化精度损失较大（约 3-7% LAMBADA），大型模型（2.7B, 8B）几乎无损失（约 0-0.5%），说明大模型对 embedding 量化更鲁棒。论文实现了 4-bit 和 8-bit 的 CUDA embedding/lm_head kernel。H2T 是实现 "部署到边缘设备" 的必要条件——否则 FP16 embedding/lm_head 成为显存瓶颈。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
