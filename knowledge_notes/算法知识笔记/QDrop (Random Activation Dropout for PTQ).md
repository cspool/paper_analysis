## QDrop (Random Activation Dropout for PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QDrop (ICLR 2022) 在 block-wise 重建中随机丢弃部分激活量化器以提升泛化性。每次前向以概率 p 随机将某些层激活"掉回"为 FP，迫使模型学习在混合精度条件下重建。随机性打破量化误差的确定性模式，防止过拟合特定量化配置，原理类似 Dropout。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for iter in range(max_iter):
    for layer in B.layers:
        w_hat = quant_dequant(layer.weight)     # 始终量化权重
        x_q = quant(layer.input) if rand() > p else layer.input
        output = w_hat @ x_q
    O_hat = B.forward_quantized(X)
    L = APH_loss(O_hat, O_fp)
    L.backward(); update(AdaRound_weights)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QDrop 在 BRECQ 基础上添加随机激活 dropout（p=0.5），不改变优化目标。APHQ-ViT 直接使用 QDrop 作为量化重建框架，将 BRECQ Hessian loss 替换为 APH loss。额外开销极小（仅前向随机判断），显著提升 W3/A3 精度。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
