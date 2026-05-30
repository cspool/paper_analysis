## PredKL and AttnKL Distillation Losses（预测级和注意力级KL蒸馏损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PredKL 和 AttnKL 是 TwigVLM++ 在第一训练阶段（蒸馏学习）中引入的两个辅助损失函数，用于提升 multi-head twig 的训练质量。

- **PredKL (Prediction-level KL Divergence Loss)**：L_PredKL = KL(p_Mb || p_Ms)，其中 p_Ms 和 p_Mb 分别是浅层 draft model Ms 和深层 target model Mb 的 next-token 预测概率分布。这是一种"强到弱"的蒸馏（strong-to-weak distillation）：用更强的 target model (Mb) 的预测分布作为 soft target，蒸馏到较弱的 draft model (Ms)。这为 twig block 提供了更丰富的监督信号，增强了其对视觉 token 的理解能力，从而间接提升 P-Head 的剪枝质量。

- **AttnKL (Attention-level KL Divergence Loss)**：L_AttnKL = KL(a_b || s)，其中 a_b ∈ R^M 是 target model Mb 某指定层的文本到视觉 token 的 attention 分布（各头平均），s ∈ R^M 是 P-Head 输出的 token 重要性分数（Eq.7）。该 loss 直接监督 P-Head，使其重要性分数与深层模型的 attention pattern 一致——而深层 attention 已被证明能提供更精准的 token 选择信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Stage-1 Distillation Training (TwigVLM++)
# α=0.1, γ=1.0

for batch in dataloader:
    # forward shallow model (draft)
    logits_s, p_head_scores = Ms.forward(X)  # p_head_scores = s (Eq.7)
    # forward deep model (target, frozen)
    logits_b, attn_b = Mb.forward(X)          # attn_b = a_b
    
    # Loss 1: Standard AR next-token prediction
    L_NTP = CrossEntropy(logits_s, y_true)
    
    # Loss 2: PredKL - 预测分布对齐
    p_s = softmax(logits_s)
    p_b = softmax(logits_b)
    L_PredKL = KL(p_b || p_s)  # target分布为指导
    
    # Loss 3: AttnKL - attention/重要性对齐
    L_AttnKL = KL(a_b || s)    # 深层attention为指导
    
    # 总损失
    L = L_NTP + α * L_PredKL + γ * L_AttnKL
    # α=0.1, γ=1.0
    
    # 仅更新 twig block (包括 P-Head)
    L.backward()
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PredKL 和 AttnKL 仅在 TwigVLM++ 的 Stage-1 训练中使用。消融实验表明：(a) 仅用 D-Head + L_NTP = 96.0% RelAcc；(b) D-Head + P-Head + L_NTP + L_AttnKL = 95.0%（多 head 分散训练能力导致退步）；(c) D-Head + P-Head + L_NTP + L_AttnKL + L_PredKL = 96.4%（PredKL 补偿多 head 的训练不足）。经过 Stage-2 RL 优化后，配置 (c) 达到最佳 97.7% RelAcc。AttnKL 的 teacher attention 取自与 twig 最后一层深度相同的 base VLM 层（即第 K+T 层），确保深度一致。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
