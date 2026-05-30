## LMFusion: Adapting Pretrained Language Models for Multimodal Generation

- baseline方法是什么？
  Baseline是Transfusion（Zhou et al., 2024），一种从头训练的unified multimodal model。Transfusion使用标准Transformer架构（与主流LLM如Llama相同），通过end-to-end训练同时学习语言建模（next-token prediction with cross-entropy loss）和图像扩散（DDPM loss on continuous image latents）。架构特点：所有参数跨模态共享——同一套QKV、O、FFN同时处理文本token和图像patch。训练数据包含language-only text data（0.25T tokens）和image-caption pairs（0.25T image tokens）。虽然架构统一，但存在两个核心缺陷：
  1. **计算资源浪费**：训练从头开始需要大量language-only data维持语言能力，即使已有强大的预训练text-only LLM（如Llama-3 8B已训练15T+ tokens），仍需重新学习语言知识。Transfusion 7B在language-only benchmarks上比Llama-3 8B低11.6%（HellaSwag 51.0 vs 60.0），说明从头训练的多模态模型语言能力不如专用text-only LLM。
  2. **naive finetuning导致灾难性遗忘**：直接在Llama-3上继续用Transfusion recipe训练（dense finetuning），会导致语言能力显著退化——HellaSwag下降15%（初始阶段），即使后续有所恢复，仍存在~7%的永久性差距。

  Baseline全栈执行例子（以Transfusion 7B从头训练后生成"a cat with secrets to keep"对应图像为例）：
  - 算法层：Transfusion统一Transformer + U-Net下采样/上采样。文本token和图像patch交替排列为一个长序列，共享QKV/FFN/O参数处理。文本使用因果mask（autoregressive），图像使用双向mask（diffusion）。训练目标L = L_LM + λ·L_DDPM。从头训练需要language-only data + image-caption data混合。
  - 系统框架层：论文未明确说明训练框架。基于标准PyTorch + FSDP/DeepSpeed分布式训练，使用标准Transformer训练pipeline。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。标准Transformer attention kernel + U-Net convolution kernel，无特殊kernel优化。
  - 硬件架构层：论文未明确说明硬件平台。训练7B+规模多模态模型通常需要数百张H100/H800，FLOPs规模极大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LMFusion，核心设计：(1) 模态特异性模块分离（modality-specific QKV/FFN/LayerNorm）将文本和图像处理路径解耦；(2) 共享自注意力层保持跨模态信息融合；(3) 文本模块冻结（η_text=0）仅训练图像模块。

  **解决"计算资源浪费"缺陷**：通过冻结Llama-3文本模块并排除训练数据中的language-only data，LMFusion用0.5× total FLOPs即达到甚至超越Transfusion的性能。具体而言，Transfusion需要0.25T text tokens + 0.25T image tokens训练，而LMFusion仅需0.25T image tokens（文本模块已预训练）。在匹配0.25T image data的情况下，LMFusion比Transfusion在image understanding上高20%（CIDEr 38.3 vs 32.0），image generation FID好3.6%（13.9 vs 14.4），language benchmarks保持Llama-3原水平。这证明**预训练LLM的语言知识可以通过冻结+模态分离无损复用于多模态任务**，无需重新学习。

  **解决"naive finetuning灾难性遗忘"缺陷**：通过deep modality separation（QKV+FFN都分离）+ 文本模块冻结（lr_ratio=0），LMFusion完全避免了语言能力的退化。Ablation实验证实：(a) 无分离（dense model）finetune时即使降低文本lr（lr_ratio=0.1），语言能力仍有2%退化且image性能受影响；(b) 浅层分离（仅FFN分离）能减轻但不足以消除退化；(c) 深层分离（QKV+FFN都分离）+ 冻结文本模块，在保持语言能力的同时image understanding/generation性能even超越全参数调优的dense模型（Figure 5 vs Figure 4）。深层原因：当文本和图像共享QKV参数时，image diffusion的梯度会通过attention层反向传播到文本参数的优化空间，干扰文本表征——文本token的Q/K/V被"拉向"适应图像噪声预测的方向，破坏了其在语言任务上的有效表征。

  **模态特异性QKV的深层作用**：看似是"增加参数"的朴素设计，实际解决了多模态训练中的**梯度冲突（gradient conflict）**问题。在dense模型中，一个token的QKV同时被LM loss和DDPM loss更新——这两个loss的梯度方向可能矛盾：LM loss要求text token的attention pattern保持语言上的coherence（更关注语义相关的token），而DDPM loss要求image patch的attention关注视觉上相关的区域。模态特异性QKV使这两种优化目标在独立的参数空间中完成，消除了梯度冲突。

  LMFusion方法全栈执行例子（以相同"a cat with secrets to keep"→图像生成为例）：
  - 算法层：Llama-3 8B text modules（冻结）+ 并行image modules（从Llama-3初始化，可训练）+ U-Net down/up（0.27B，从头训练）。文本token通过Proj_text → QKV_text → 在共享attention中text Q attend到[K_img, K_txt] → O_text → FFN_text → LM_Head_text。图像patch通过UNet_Down_img → QKV_img → 在共享attention中image Q attend到[K_txt, K_img] → O_img → FFN_img → UNet_Up_img。仅image路径参数有梯度更新。关键：文本和图像在attention层有双向cross-modal交互，但由于QKV分离，两者的attention计算是独立的——text的attention不改变image的QKV参数，反之亦然。这比完全独立的两个模型（no cross-modal at all）更优越，因为共享attention允许text context condition image diffusion（文本条件在去噪每一步都参与）。
  - 系统框架层：论文未明确说明。基于标准PyTorch训练，使用AdamW optimizer (β1=0.9, β2=0.95)，cosine decay LR schedule with 4000-step warmup，η_img=1e-4→1.5e-5。Image data 380M Shutterstock captions，80% caption→image顺序。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。与Transfusion相同——标准Transformer attention + U-Net卷积分支，无特殊kernel优化。注意：虽然LMFusion参数量是Transfusion的2倍（两套QKV/FFN），但每个token仅激活对应模态的模块（一半参数），因此每次前向的FLOPs与Transfusion相同。
  - 硬件架构层：论文未明确说明硬件平台。

  设计思路核心：
  LMFusion的本质洞察是**模态特异性参数化不是"增加参数量"的成本支出，而是"消除梯度冲突"的架构投资**。传统观点认为MoE或模态分离是computation-vs-capacity的trade-off（更多参数=更多计算），但LMFusion证明了在multimodal generation场景下，模态分离反而比dense model在**同等可训练参数量和同等FLOPs**下表现更好。深层原因：文本和图像的优化动态（optimization dynamics）在本质上是不同的——文本是离散token的自回归生成（需要精确的next-token probability），图像是连续latent的扩散去噪（需要smooth的噪声预测）。当两者共享参数时，这两种动态相互干扰（gradient interference）；分离参数使每种模态在其自身的优化空间中自由演化。LMFusion的扩展（LLaVAFusion）进一步验证了这一原则的普适性——同样的冻结+分离范式可以直接应用于已有多模态理解能力的VLM（LLaVA-NeXT），在不损失其已有能力的前提下赋予图像生成能力。
