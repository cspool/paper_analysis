## Transfusion (Multimodal Training Framework)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transfusion（Zhou et al., 2024）是一种统一的多模态生成模型训练框架，通过单个Transformer模型同时学习文本自回归生成和图像扩散去噪。其核心思想是：将文本token和图像latent patch交替排列为一个长序列，使用标准Transformer（与Llama等主流LLM架构一致）通过end-to-end训练同时优化language modeling loss（cross-entropy on discrete tokens）和DDPM loss（MSE on predicted noise）。与分别训练语言模型和扩散模型再拼接的多阶段方法不同，Transfusion在同一个Transformer backbone内联合训练两种模态，使文本和图像在attention层中进行双向信息交互（文本可条件化图像生成，图像可辅助文本理解）。Transfusion引入U-Net downsampler/upsampler在Transformer前后压缩/还原图像latent的维度，降低attention计算的开销。该框架是从头训练的——不使用预训练LLM的权重，需要同时使用大量language-only data和image-caption data训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Transfusion的训练pipeline（以caption→image数据为例）：
```
# 输入: text tokens x_txt (discrete), image x_img (raw pixels)
# Step 1: Image encoding
z_img = VAE_encoder(x_img)              # [H, W, 3] → [h, w, c] continuous latent

# Step 2: Image diffusion forward process
t ~ Uniform(1, T)                        # 随机采样扩散时间步
noise ~ N(0, I)                          # 高斯噪声
z_t = sqrt(ᾱ_t) * z_img + sqrt(1-ᾱ_t) * noise  # 加噪latent

# Step 3: U-Net downsampling
h_img = UNet_Down(z_t, t)               # [h*w, c] → [N_patches, d]

# Step 4: Input embedding
h_txt = Embedding(x_txt)                 # [M, d] text token embeddings

# Step 5: Unified Transformer (single shared QKV/FFN/O)
h_all = concat(h_txt, h_img)            # [M+N_patches, d]
for layer in 1..L:
    Q, K, V = QKV_proj(h_all)           # 共享QKV，同时处理text和image
    # 混合attention mask: text用因果mask（autoregressive），image用双向mask（diffusion）
    A = softmax(Q@K^T/sqrt(d) + M)      # M为混合mask
    h_all = FFN(A @ V)                  # 共享FFN

# Step 6: Separate output heads
h_txt_out, h_img_out = split(h_all)
p_logits = LM_Head(h_txt_out)            # [M, vocab_size] 文本logits
ε_pred = UNet_Up(h_img_out, t, z_t)     # [N_patches, c] 预测噪声

# Step 7: Combined loss
L = CrossEntropy(p_logits, x_txt_labels) + λ * MSE(ε_pred, noise)
```
关键特性：所有Transformer参数（QKV, O, FFN）跨模态共享；混合attention mask使文本token只能attend之前的文本（因果），图像patch可双向attend；不使用预训练LLM权重，全部从头训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Transfusion由Meta FAIR提出（arXiv 2408.11039, 2024年8月）。实现基于PyTorch，使用标准Transformer + U-Net模块。训练数据含language-only text data（0.25T tokens）和image-caption pairs（0.25T image tokens）。Transfusion 7B与Llama-3 8B有相同Transformer尺寸（差异仅来自vocabulary大小影响embedding层）。LMFusion将其作为核心baseline对比：Transfusion从头训练虽架构统一，但(1) 需大量language-only data维持语言能力，(2) 语言benchmarks仍低于专用text-only LLM，(3) 总FLOPs是LMFusion（冻结文本模块）的2倍。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
