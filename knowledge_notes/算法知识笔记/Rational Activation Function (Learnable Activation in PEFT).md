## Rational Activation Function (Learnable Activation in PEFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rational Activation Function 由 Molina et al. (2019) 提出，是有理函数形式的可学习激活函数：Ra(x) = Σ_{j=0}^{m} a_j·x^j / (1 + ||Σ_{i=1}^{n} b_i·x^i||)，其中 a_j, b_i 为可学习参数，m, n 为阶数。通过调整参数可逼近 ReLU/GeLU/Swish 等，也可学习全新形态。MiLoRA 将其用于每层 LoRA router 的激活函数（m=6, n=5，初始化为 GeLU 逼近），使不同深度的 Transformer 层学习最适合路由的激活函数。通过 bi-level optimization（DARTS 风格）训练：inner level 优化 LoRA+router 参数 Ω（lr=1e-4），outer level 优化 activation 参数 Θ（lr=1e-6，仅 ~12 scalars/layer）。Ablation 表明 learnable activation 在 BoolQ/PIQA/MMLU 上优于固定 GeLU 或 ReLU+GeLU 混合方案。

从算法pipeline角度拆解术语：
```
# 逐元素计算（per element x in pooled hidden state h^l）
num = a_0 + a_1·x + a_2·x^2 + a_3·x^3 + a_4·x^4 + a_5·x^5 + a_6·x^6
den = 1 + |b_1·x + b_2·x^2 + b_3·x^3 + b_4·x^4 + b_5·x^5|
Ra(x) = num / den

# 每层参数: a_j (7个), b_i (5个) = 12 scalars/layer
# 32层总计: 384 scalars, overhead trivial
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 实现: 自定义 nn.Module，forward 中预计算 x 各次幂避免重复。参数初始化为 GeLU Padé 逼近系数（保证训练初期稳定）。
- 训练: bi-level optimization，alternating 更新（每 step 更新 Ω，每若干 step 更新 Θ）。
- 适用: 任何需要为不同层学习不同激活函数的场景。论文验证在 3 个 benchmark 上一致优于固定激活函数。

涉及论文标题：
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

---
