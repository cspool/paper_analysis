## SigLIP Vision Encoder（SigLIP 视觉编码器）

术语解释
SigLIP (Sigmoid Loss for Language-Image Pre-training) 是 Google DeepMind 提出的视觉-语言预训练方法，使用 sigmoid loss 替换 CLIP 的 contrastive softmax loss，在 batch 规模上更鲁棒且对 batch size 不敏感。DeepSeek-VL2 使用 SigLIP-SO400M-384 作为 vision encoder。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP 的核心创新：将 CLIP 的多分类 softmax loss（需在 batch 内对所有 negative pairs 做归一化）替换为独立的二分类 sigmoid loss —— 每对 (image, text) 独立判断是否匹配，其他对作为独立的 negative。SigLIP Loss = -(1/(|B|)) Σ [log σ(z_ii · t_exp) + Σ_j≠i log(1-σ(z_ij · t_exp))]，其中 z_ij = f_img(I_i)·f_txt(T_j)/τ。优势：(1) 不依赖 large batch size（CLIP 需 32k+ batch，SigLIP 可小 batch 训练）；(2) 每个 negative pair 独立处理，对于 noisy image-text pairs 更鲁棒；(3) 对 batch 内负样本分布不敏感。SigLIP 训练出的 vision encoder 在 VLM 任务（尤其是 OCR/文档理解）上表现优异。DeepSeek-VL2 使用的 "SO400M" 变体是 SigLIP 中最大的公开模型之一（~400M params）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SigLIP 在 VLM 中的应用：作为 frozen 或 finetuned vision encoder，将图像编码为 visual tokens 供给 LLM。DeepSeek-VL2 在所有训练阶段（Alignment/Pretraining/SFT）均对 SigLIP encoder 进行 finetuning（vision encoder LR multiplier = 0.1× LLM LR），而非保持 frozen。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：Google 官方 tf/keras 仓库 + HuggingFace models（google/siglip-so400m-patch14-384）。常见替代方案包括：OpenAI CLIP-ViT, EVA-CLIP, DFN-CLIP, InternViT（InternVL 自研）。VLM 选择 vision encoder 的考量：(1) 分辨率支持（SigLIP-384 固定 384×384，需配合 dynamic tiling 实现高分辨率）；(2) 输出 token 数（384/14=27×27=729 tokens）；(3) OCR 能力（SigLIP 在 OCR 任务上表现较好）。SigLIP 的一个变体 SigLIP2 进一步改进了多分辨率支持和训练效率。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
