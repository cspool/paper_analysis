## Masked Conditional Video Diffusion (掩码条件视频扩散)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Masked Conditional Video Diffusion 是 SlowFast-VGen 中慢学习（slow learning）阶段使用的条件视频生成方法，基于 Voleti et al. (MCVD, 2022) 的框架。核心思想：给定 fp 帧过去帧和 fg 帧待生成帧，过去帧的 latent 保持干净（不加噪）作为条件，仅对待生成的 fg 帧添加高斯噪声；将干净条件帧和加噪生成帧拼接后送入 UNet 去噪；在计算 loss 时 mask 掉条件帧部分，仅在生成帧上计算 MSE loss。这使得模型学会基于前序视频和语言 action 生成后续视频 chunk。该方法支持任意长度 ≤ context window（32 帧）的 fp 和 fg 组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Slow Learning: Masked Conditional Video Diffusion
# fp: 过去帧数, fg: 待生成帧数

# 前向扩散
z_{0,:fp} = z_{0,:fp}                                              # 条件帧 latent (clean)
z_{t,fp:(fp+fg)} = sqrt(ᾱ_t)·z_{0,fp:(fp+fg)} + sqrt(1-ᾱ_t)·ε    # 生成帧 latent (加噪)
z_t = concat(z_{t,:fp}, z_{t,fp:(fp+fg)})                         # 拼接送入 UNet

# UNet 去噪
ε_pred = ε_Φ(z_t, t, c)   # c = CLIP 编码的语言 action text

# Masked Loss（仅在后 fg 帧计算）
loss = ||ε - ε_pred[fp:(fp+fg)]||²
```

Annotations:
- z_{0,:fp}: 前 fp 帧的 VAE 编码 latent（clean，作为条件）
- z_{0,fp:(fp+fg)}: 后 fg 帧的 ground-truth latent
- ε: 采样的标准高斯噪声
- ᾱ_t: 扩散累积噪声系数
- c: CLIP text encoder 编码的 action 文本条件
- UNet 输出 ε_pred 包含所有 fp+fg 帧的噪声预测，但仅取 [fp:(fp+fg)] 范围计算 loss
- 条件帧的 latent 在去噪推理时保持已知（不参与 DDIM 采样）
- fp=1 时退化为单帧条件生成（如 robot 数据）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SlowFast-VGen 基于预训练 ModelScopeT2V（latent video diffusion model）实现 masked conditional video diffusion。ModelScopeT2V 使用 3D UNet 架构（spatial convolutions + temporal convolutions + attention blocks），VAE 编码视频到 latent，CLIP ViT-H/14 编码文本。慢学习在约 64 张 V100 GPU 上训练，batch size=128，slow learning rate=5e-6，冻结 VAE 和 CLIP Encoder，仅训练 UNet。训练视频长度 ≤ 32 帧（context window）。与标准 video diffusion 的区别：标准方法对全部帧加噪+去噪；MCVD 的 masking 机制使条件帧保持干净信号，引导生成帧与条件帧一致。MCVD 原论文（Voleti et al., 2022, arXiv:2205.09853）支持 prediction、generation 和 interpolation 三种任务。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation
