## Mutual Information Gap for Expert Routing in MoE (MI Gap / 专家路由互信息间隙)

术语解释
MI Gap (Mutual Information Gap) 是 CoMoE 提出的量化 MoE 中 expert 专业化和冗余程度的信息论度量：定义为输入 token x 与 top-k routing 下激活专家 M⁺ 之间的互信息，减去 x 与非激活专家 M⁻ 之间的互信息。

术语是什么？
给定输入 token x 和专家集合 M，在 top-k routing 下定义：

$$\Delta I = I_{\text{top-}k}(x, M^{+}) - I_{\neg \text{top-}k}(x, M^{-})$$

其中 $I(x; M) = \mathbb{E}_{x, \mathcal{M} \sim \mathcal{D}}[\log \frac{p(M|x)}{p(M)}]$ 为标准互信息。

MI Gap 的直观含义：
- **最大化 I(x; M⁺)**：促进激活 expert 对高度匹配的输入做出响应，鼓励专业化；同时作为信息瓶颈过滤无关噪声
- **最小化 I(x; M⁻)**：抑制非激活 expert 对无关输入的响应，防止多个 expert 学习相似表示

通过 Jensen 不等式，可得到 I(x; M) 的下界：
$$I(x; M) \ge \mathbb{E}_{x,e,\mathcal{M}}\left[\log \frac{p(e|x)}{p(e)}\right]$$

其中 e = E(x) 为 expert 的输出表示。

从算法pipeline角度拆解术语：
MI Gap 不直接计算，而是通过对比学习进行估计。CoMoE 的理论核心是 **InfoNCE 定理**：

**Theorem (InfoNCE)**：MI Gap ΔI = I_top-k(x, e⁺) - I_¬top-k(x, e⁻) 可通过对比目标下界估计：

$$\Delta I \ge \log(N) - \mathcal{L}_{\text{NCE}}$$

```
# MI Gap 估计流程
# 1. 收集样本
(x, e⁺) ~ D_top-k           # 激活 expert 的表示（正样本分布）
(x, e⁻) ~ D_¬top-k          # 非激活 expert 的表示（负样本分布）

# 2. 计算得分函数（信息密度比估计）
h₁(x,e⁺) ∝ p(e⁺|x)/p(e⁺)   # 激活 expert 的信息密度比
h₂(x,e⁻) ∝ p(e⁻|x)/p(e⁻)   # 非激活 expert 的信息密度比

# 3. InfoNCE 对比损失
L_NCE = -E[log( h₁(x,e⁺) / (h₁(x,e⁺) + Σ h₂(x,e⁻)) )]

# 4. MI Gap 下界
ΔI ≥ log(N) - L_NCE        # N 为负样本数，随 N 增大越紧
```

当 expert 专业化程度高时：每个 expert 仅对特定 token 子集产生高互信息（I_top-k 大），同时不同 expert 之间知识冗余最小化（I_¬top-k 趋近于 0），MI Gap 达到最大值。

术语一般如何实现？如何使用？
- **实践中的近似**：不分别估计 I_top-k 和 I_¬top-k，而是统一对比目标直接估计 ΔI。通过将 h₁ 和 h₂ 合并为单一得分函数 h(x,e) = exp(E⁺(x)·e)/τ，使正负样本形成双向样本对
- **评分函数选择**：指数余弦相似度是最常用的选择，温度 τ 控制对比分布的锐度
- **理论保证**：InfoNCE 提供的是 MI Gap 的紧下界（tight lower bound），随负样本数 N 增加而收紧
- **与 Load Balance 的区别**：Load balance loss 强制 expert 使用频率均匀（量的平衡），MI Gap 强制 expert 功能差异化（质的专业化），两者互补但 CoMoE 实验表明仅 MI Gap 即可自然产生负载均衡

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---
