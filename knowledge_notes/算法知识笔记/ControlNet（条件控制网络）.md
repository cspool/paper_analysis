## ControlNet（条件控制网络）

术语是什么？通过联网搜索让回答具体和精准。

ControlNet是由Lvmin Zhang等人在ICCV 2023提出的扩散模型条件控制扩展架构。它通过复制预训练扩散模型（如Stable Diffusion）的encoder层并注入额外条件输入（如Canny edge maps、pose skeletons、depth maps、segmentation masks等），实现对生成图像内容的精确空间控制，而无需从头训练或微调原始模型。ControlNet的核心设计是"zero convolution"（零初始化卷积层），在训练初期输出为零以保持原始模型行为不变，逐步学习条件控制信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ControlNet在扩散pipeline中的运转流程：

```
// ControlNet block（添加到U-Net的每个encoder layer）
1: x ← UNet_encoder_block(latent, timestep_embedding)    // 原始U-Net encoder输出
2: c ← ControlNet_encoder_block(latent, timestep_embedding, control_input)
   // ControlNet复制原始encoder结构+额外control input通道
3: c_transformed ← zero_convolution(c)                     // 1×1 zero-initialized conv
4: output ← x + c_transformed                              // 残差加法注入控制信号
```

Diffusion pipeline中的ControlNet使用：
```
5. 输入：prompt ("a dog") + control_image (Canny edge map)
6. CLIP text encoder → text_embedding
7. ControlNet encoder → extract control features from control_image at multiple resolutions
8. VAE encoder → latent (if image-to-image)
9. for t in denoising_steps:
10.    U-Net(latent_t, t, text_embedding) + ControlNet(control_features) → predicted_noise
11.    latent_{t-1} = denoise_step(latent_t, predicted_noise, t)
12. VAE decoder → output image
```

ControlNet inputs属于Difflow论文分析的扩散pipeline多种输入之一。Difflow评估的edit应用同时使用ControlNet（Canny edge spatial control）+ LoRA（style adaptation）。Symbolic property analysis发现ControlNet inputs共享相同的优化条件（同时冗余或同时非冗余），因此可将多个ControlNet inputs在属性条件枚举时合并为一个，指数级减少dEngine数量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ControlNet开源实现：https://github.com/lllyasviel/ControlNet（基于Stable Diffusion 1.5）和Diffusers库中的ControlNetModel类。使用时加载预训练的Stable Diffusion backbone + 对应control type的ControlNet权重（如canny、depth、pose、scribble等）。支持multi-ControlNet（同时使用多个control信号）。在Difflow中，启用ControlNet会使U-Net输入从4个激增至14个，通过symbolic analysis将ControlNet inputs视为共享优化条件的一个整体避免了2^14=16384 engines的组合爆炸。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---

