## Long-Tailed Distribution in Vision MoE Token Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Long-Tailed Distribution in Vision MoE Token Routing 是 LTDR 论文揭示的 LVLM MoE 中 vision token 路由分布的核心特性。Vision tokens（CLIP 编码器输出，~576/image）天然包含大量低信息背景 patches（head, ~87%）和少量高信息前景 patches（tail, ~13%），导致 vision TER 的 routing 呈现 long-tailed 分布。具体表现：大部分 vision tokens 的 RPV 较低（router 不确定分配，token 含信息量低 → 类似 long-tail 分类问题中的 head classes），少数 vision tokens 的 RPV 较高（router 明确偏好某些 expert，token 含信息量高 → 类似 tail classes）。这与 language tokens 的近似 uniform TER 分布形成对比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Vision Token Long-Tailed 特性分析流程**：
```
# Step 1: 计算 vision token RPV distribution
for each image in batch:
    V = CLIP_encoder(image)  # [576, D] vision tokens
    probs = Softmax(V @ W_g)  # [576, K] routing probabilities
    rpv = Variance(probs, dim=1)  # [576] per-token RPV

    # Step 2: 绘制 RPV 直方图 (Fig. 1(b) in paper)
    # x-axis: RPV range (0.00-0.01, 0.01-0.02, ...)
    # y-axis: token count in each bin
    # Result: long-tailed shape
    #   - 442 tokens in 0.00-0.01 bin (head)
    #   - rapidly decreasing counts in higher RPV bins (tail)

# Step 3: Load balancing impact analysis
# with L_balancing:  高 RPV token 数量减少 → expert specialization 受阻
# without L_balancing: 保留 long-tailed shape → tail tokens 获得专业 expert

# Step 4: Modality comparison
# Language RPV distribution: near-uniform (各 RPV 区间 count 接近)
# Vision RPV distribution: long-tailed (集中低 RPV, 长尾高 RPV)
```

**关键发现**：
- Vision 的 long-tailed 特性来自于视觉内容的固有结构——大多数图像区域是背景/纹理（低信息量），少数是目标/文本/关键细节（高信息量）
- Load balancing 的正则化效果类似 undersampling tail tokens → 使 critical foreground tokens 被分散到各 expert，降低 expert 对视觉关键信息的特化能力
- GMoE 实验：移除 load balancing 后性能提升，验证了 load balancing 对 vision TER 的负面影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **识别方法**：在 MoE 训练的每个 batch 中计算 per-token RPV，绘制分布直方图确认 long-tailed 特性
- **应对策略**：
  1. 移除 vision TER load balancing（MsDaR）→ 保持 long-tailed 天然分布
  2. 增强 tail token expert activation（VsDEA）→ 补偿 tail token 的稀少性
- **与其他 long-tailed 研究的关系**：传统 long-tailed classification（RIDE, BBN, LDAM）主要处理 sample-level class imbalance，LTDR 是首次在 token-level TER distribution 层面处理 long-tailed 问题

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
