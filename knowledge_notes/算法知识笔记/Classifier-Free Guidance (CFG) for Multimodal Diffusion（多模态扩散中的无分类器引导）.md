## Classifier-Free Guidance (CFG) for Multimodal Diffusion（多模态扩散中的无分类器引导）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Classifier-Free Guidance（CFG, Ho and Salimans, 2021）是提升扩散模型图像生成质量的技术。核心思想：推理时结合条件预测和无条件预测引导生成方向，ε̂ = ε_uncond + w*(ε_cond - ε_uncond)，w为guidance scale。w>1时模型放大条件信息影响，使生成图像更贴合文本（更高CLIP score）；w=1等价标准条件生成。需每扩散步两次前向pass（conditional + unconditional），推理延迟翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for t = T...1:
    ε_cond = model(z_t, t, text_prompt)     # 条件前向
    ε_uncond = model(z_t, t, null_text)     # 无条件前向
    ε_guided = ε_uncond + w*(ε_cond - ε_uncond)  # CFG组合, w=1.55
    z_{t-1} = denoise_step(z_t, ε_guided, t)
```
LMFusion在MS-COCO上评估：无CFG (w=1.0)的FID和带CFG (w=1.55)的结果，通常w在1.5-3.0间选择以平衡质量和多样性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMFusion采用w=1.55。CFG代价是每步两次前向，但显著提升文本-图像对齐度（CLIP score）。实际使用中训练时可随机drop文本条件（如10%概率）使模型学习无条件生成能力，推理时即可使用CFG。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
