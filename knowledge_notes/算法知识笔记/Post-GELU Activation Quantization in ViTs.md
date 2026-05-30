## Post-GELU Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-GELU Activation Quantization 指 ViT MLP 中经过 GELU(x) = x·Φ(x) 激活后的激活值量化挑战。GELU 导致：(1) 分布极不平衡——负值集中在窄区间 [-0.17, 0]，正值稀疏且范围可达 40；(2) 层间激活范围差异巨大。均匀量化器低效——密集负值的细微变化被粗糙量化，稀疏正值的宽范围浪费格点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GELU: 密集负值 + 稀疏宽范围正值 → 量化低效
GELU(-3)≈ -0.004, GELU(-1)≈ -0.159, GELU(0)=0
密集区间 [-0.17, 0], 正值可达 40

# ReLU: 精确零 + 线性正值 → 量化高效
ReLU(x<0) = 0 (精确零), ReLU(x>0) = x (可 clamp)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
已有方法：PTQ4ViT 的 Twin-Uniform Quantizer、AdaLog 的对数量化器、DopQ-ViT 的 Tangent 量化器——均需专用硬件。APHQ-ViT 的 MR 方法通过 GELU→ReLU 替换从根本上消除该问题，使用标准均匀量化器达到优于专用量化器精度。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
