## VAE Tokenizer for Multimodal Models（多模态模型中的VAE图像分词器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VAE (Variational Autoencoder) Tokenizer是扩散模型和多模态LLM中将图像编码为连续latent representation的模块。LMFusion/Transfusion使用预训练VAE encoder（stabilityai/sd-vae-ft-mse）将256×256 RGB图像压缩为32×32×8连续latent tensor（f=8下采样，8通道），在latent space中进行扩散去噪。与离散tokenizer（VQ-VAE, VQGAN）不同，VAE tokenizer产生连续latent配合DDPM做连续空间扩散。VAE encoder冻结（不参与训练），latent进一步经可训练U-Net downsampler压缩为256 patches送入Transformer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 编码: x_img [3,256,256] → z [8,32,32] → h_img [256,d]
z = VAE_encoder(x_img)
h_img = UNet_Down(reshape(z, [1024,8]), t)

# 解码: z_0 [8,32,32] → generated_image [3,256,256]
generated_image = VAE_decoder(z_0)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion使用sd-vae-ft-mse (https://huggingface.co/stabilityai/sd-vae-ft-mse)，在OpenImages上fine-tuned，MSE loss优化重建质量。选择VAE over VQ-VAE的优势：连续latent更适合diffusion连续加噪/去噪；无codebook collapse问题；与主流扩散模型兼容。代价是latent无离散token语义，不能用自回归next-token方式解码，必须用diffusion框架。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
