## DDPM in Multimodal LLMs（多模态LLM中的扩散图像生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Denoising Diffusion Probabilistic Models (DDPM, Ho et al., 2020) 是一种通过迭代去噪生成图像的生成模型框架。在LMFusion/Transfusion等多模态LLM中，DDPM被集成到Transformer backbone内部——不同于传统扩散模型使用独立U-Net backbone（如Stable Diffusion），多模态LLM在Transformer hidden state空间中运作。流程：(1) VAE encoder将图像压缩到连续latent space，(2) 前向扩散逐步加噪（x_0→x_T），(3) 反向去噪时基于文本条件预测每步噪声，(4) VAE decoder解码去噪latent为像素图像。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练阶段
z_0 = VAE_encoder(x_img)                   # 编码到latent
t ~ Uniform(1, T); ε ~ N(0, I)             # 时间步+噪声
z_t = sqrt(ᾱ_t) * z_0 + sqrt(1-ᾱ_t) * ε    # 前向加噪
h_txt = text_forward(x_txt)                 # 文本表征
ε_pred = image_forward(z_t, t, h_txt)      # 文本条件化噪声预测
L_DDPM = MSE(ε_pred, ε)

# 推理阶段（T步去噪）
z_T ~ N(0, I)
for t = T...1:
    ε_pred = image_forward(z_t, t, text_context)
    z_{t-1} = denoise_step(z_t, ε_pred, t)  # DDPM/DDIM sampler
generated_image = VAE_decoder(z_0)
```
关键设计：噪声预测以文本为条件——text tokens通过cross-modal attention注入去噪过程；图像token使用双向attention mask（去噪需全局上下文），文本token使用因果mask；cosine noise schedule（Nichol and Dhariwal, 2021）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion使用256×256图像压缩为32×32×8 latent（VAE f=8下采样），经U-Net downsampler降至256 patches。Loss权重λ平衡LM loss和DDPM loss。此设计使单一模型同时具备文本生成和图像生成能力。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
