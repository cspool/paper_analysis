## LMFusion: Adapting Pretrained Language Models for Multimodal Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LMFusion框架，核心创新为：(1) 模态特异性Transformer模块——将预训练text-only LLM (Llama-3 8B) 的权重复用于文本处理，同时引入并行的图像专用Transformer模块用于图像扩散（diffusion）处理；(2) 模态特异性设计——FFN层、QKV投影、LayerNorm均为模态独立，各模态数据路由到各自的专用模块处理；(3) 共享self-attention层——跨模态交互通过共享的自注意力实现，text和image的Q/K/V在attention层内concat后进行统一注意力计算；(4) 文本模块冻结、仅训练图像模块——通过设text学习率η_text=0保持Llama-3的语言能力；(5) 学习率解耦——text和image参数组使用独立学习率。

  实验比较：(1) **主实验** vs Transfusion 7B（从头训练的多模态生成模型）——0.5× FLOPs配置（仅用image data，匹配Transfusion的图像数据量）和1× FLOPs配置（匹配总FLOPs）；(2) **Ablation实验**——No separation（dense Llama-3直接finetune）、Shallow separation（仅FFN模态特异性）、Deep separation（FFN+Attention模态特异性，即LMFusion）三种架构对比，以及不同学习率比η_text/η_image ∈ {0, 0.1, 1}的影响；(3) **LLaVAFusion扩展**——从LLaVA-NeXT 8B出发延续LMFusion范式，与EMU-3、Show-O、Janus、Chameleon、MetaMorph、Transfusion对比image understanding（MMMU/ChartQA/RealWorldQA/MME-P）和generation（FID）。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练和评估的具体GPU型号及数量。论文提及使用的是Llama-3 8B模型初始化和Transfusion训练recipe，推测类似Meta FAIR的基础设施（通常为NVIDIA H100集群）。训练配置：最大上下文长度4096 tokens，batch size 250K tokens/image tokens。论文未说明推理延迟或GPU利用率等硬件性能指标。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3 8B（text backbone），额外引入U-Net downsampler/upsampler（0.27B参数，从头训练）。VAE encoder使用stabilityai/sd-vae-ft-mse（将256×256图像压缩为32×32×8 tensor，经2-block U-Net downsampler后得256 patches）。LLaVAFusion扩展基于LLaVA-NeXT 8B。
  
  数据集：训练数据——380M Shutterstock image-caption pairs（与Transfusion一致），80% caption-before-image顺序，20% image-before-caption。图像编辑finetuning——8K MagicBrush image editing examples。
  
  Benchmarks：
  - 语言能力：HellaSwag、PIQA、SIQA、WinoGrande（accuracy）
  - 图像理解：MS-COCO Captioning test split（CIDEr scores）
  - 图像生成：MS-COCO 30K validation prompts（FID、CLIP scores），含无classifier-free guidance (CFG=1.0) 和CFG=1.55两版本
  - LLaVAFusion额外benchmark：MMMU、ChartQA、RealWorldQA、MME-Perception、FID

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文发表于NeurIPS 2025 (https://papers.nips.cc/paper_files/paper/2025/hash/0d33b1148c6ee6bb3ea9f056ae3301e6-Abstract-Conference.html)，arXiv: 2412.15188。截至搜索时，论文未明确提供官方代码仓库链接，但有HuggingFace论文页面 (https://huggingface.co/papers/2412.15188)。基础模型Llama-3 8B为Meta开源权重，VAE encoder (sd-vae-ft-mse) 为Stability AI开源。

  LMFusion算法pipeline核心流程（以1个Transformer layer，text token数为M，image patch数为N为例）：

  **初始化阶段：**
  ```
  # 文本模块初始化（从Llama-3加载，冻结）
  Proj_text = Llama3.embed_tokens          # 线性embedding
  QKV_text = Llama3.self_attn.qkv_proj     # QKV投影 [d, 3d]
  O_text = Llama3.self_attn.o_proj         # O投影 [d, d]
  FFN_text = Llama3.mlp                    # FFN (gate+up+down)
  LM_Head_text = Llama3.lm_head            # 输出投影 head

  # 图像模块初始化（从Llama-3加载，训练）
  QKV_img = copy(Llama3.self_attn.qkv_proj)  # 并行QKV投影
  O_img = copy(Llama3.self_attn.o_proj)      # 并行O投影
  FFN_img = copy(Llama3.mlp)                # 并行FFN
  UNet_Down_img = random_init()             # 从头训练 (0.27B)
  UNet_Up_img = random_init()
  ```

  **前向传播（per layer）：**
  ```
  # Step 1: Input Projection
  h_txt = Proj_text(x_txt)                 # [M, d]  文本embedding
  h_img = UNet_Down_img(x_img_t, t)        # [N, d]  图像下采样，t为扩散时间步

  # Step 2: Modality-specific QKV projection
  Q_txt, K_txt, V_txt = QKV_text(h_txt)   # 各 [M, d]
  Q_img, K_img, V_img = QKV_img(h_img)    # 各 [N, d]

  # Step 3: Cross-modal self-attention
  # 文本token的attention（Eq.9）:
  K_all = concat(K_img, K_txt)            # [M+N, d]
  V_all = concat(V_img, V_txt)            # [M+N, d]
  A_txt = softmax(Q_txt @ K_all^T / sqrt(d) + M)  # [M, M+N]，M为混合mask
  h_O_txt = O_text(A_txt @ V_all)        # [M, d]

  # 图像token的attention（Eq.10）:
  K_all' = concat(K_txt, K_img)           # [M+N, d]
  V_all' = concat(V_txt, V_img)
  A_img = softmax(Q_img @ K_all'^T / sqrt(d) + M)  # [N, M+N]
  h_O_img = O_img(A_img @ V_all')        # [N, d]
  # M: causal mask for text tokens (i<=j), bidirectional for image tokens

  # Step 4: Modality-specific FFN
  h_FFN_txt = FFN_text(h_O_txt)           # [M, d]  冻结参数，无梯度
  h_FFN_img = FFN_img(h_O_img)            # [N, d]  可训练参数

  # Step 5: Output projection
  p_logits = LM_Head_text(h_FFN_txt)      # [M, vocab_size]  文本logits
  ε_pred = UNet_Up_img(h_FFN_img, t, h_img)  # [N, 32*32*8]  预测噪声
  ```

  **训练目标（Eq.4）：**
  ```
  # LM loss on text tokens
  L_LM = CrossEntropy(p_logits, x_txt_labels)

  # DDPM loss on image tokens  
  L_DDPM = MSE(ε_pred, ε)   # ε ~ N(0,I) 为真实噪声

  # Total loss
  L = L_LM + λ * L_DDPM

  # 参数更新（仅图像模块）：
  θ_img = {UNet_Down_img, QKV_img, O_img, FFN_img, UNet_Up_img}
  θ_img = θ_img - η_img * ∇L(θ_img)
  # θ_text = {Proj_text, QKV_text, O_text, FFN_text, LM_Head_text} 冻结
  ```

  **推理阶段图像生成：**
  ```
  # 文本条件编码（单次前向）
  h_txt_all = text_forward(prompt)        # 文本token通过冻结的文本模块

  # 扩散去噪循环（T步）
  x_T ~ N(0, I)  # 初始纯噪声
  for t = T, T-1, ..., 1:
      ε_pred_t = image_forward(x_t, t, h_txt_all)  # 图像模块 + cross-attn to text
      x_{t-1} = denoise_step(x_t, ε_pred_t, t)     # DDPM/DDIM sampler
  generated_image = VAE_decoder(x_0)
  ```

  关键设计要点：
  - 文本和图像模块均从Llama-3初始化，使图像模块获得文本预训练的knowledge transfer
  - 虽然参数量是Transfusion的2倍，但每个token仅激活对应模态的模块（一半参数），FLOPs与Transfusion相同
  - Cross-modal attention是双向的——文本可attend到图像、图像可attend到文本，实现模态间信息融合
  - 混合attention mask：文本使用因果mask（autoregressive），图像使用双向mask（diffusion的去噪特性）
  - 训练时80%数据为caption→image顺序（训练图像生成）使模型学习文本条件下的图像生成
