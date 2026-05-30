## Contrastive Representation for MoE (CoMoE) / InfoNCE Contrastive Loss in MoE

术语解释
Contrastive Representation for MoE (CoMoE) 是在 MoE-based PEFT 训练中引入基于 InfoNCE 的对比学习辅助损失，利用 top-k routing 将专家分为正负样本以促进专家专业化和模块化的方法。其核心创新在于将非激活专家（routing 中未被选中的 expert）从"浪费的计算资源"重新定义为"对比学习的负样本"。

术语是什么？
CoMoE 在标准 supervised fine-tuning 损失 L_CE 基础上添加对比辅助损失 L_con：

$$L_{\text{total}} = L_{CE} + \lambda \cdot L_{\text{con}}$$

其中对比损失基于 InfoNCE（Oord et al., 2018）：

$$L_{\text{con}} = \sum_{i=1}^{k} -\log\left(\frac{\exp(q_i \cdot k_i^+/\tau)}{\exp(q_i \cdot k_i^+/\tau) + \sum_{k_i^-} \exp(q_i \cdot k_i^-/\tau)}\right)$$

具体构造：
- **Query (anchor)**：从 k 个激活 expert 中随机选一个的输出表示 E_a(x) 作为 query q
- **Positive keys**：其余 k-1 个激活 expert 的输出表示（同属激活集的正样本）
- **Negative keys**：n-k 个非激活 expert 的输出表示（路由未选中的负样本）
- **Score function**：指数余弦相似度 h(x,e) = exp(q·e)/τ，τ 为温度超参数

CoMoE 可无缝集成到任何 top-k routing 的 MoE 架构中（LoRA-MoE、DoRA-MoE 等），不需要预训练，仅作为辅助目标添加。推理时无需对比损失——仅标准 top-k routing 前向。

从算法pipeline角度拆解术语：
CoMoE 的完整训练 pipeline（基于论文 Algorithm 1）：

```
# CoMoE Training Forward Pass (single token x)
输入: x, experts {E_j}_{j=1}^n, router g, top-k=2, τ, λ

# Step 1: Standard MoE forward
g(x) = softmax(W_g · x)                    # router logits [n]
T = topk(g(x), k)                           # activated expert indices
ŷ_i = g_i / Σ_{j∈T} g_j  if i∈T else 0     # renormalized weights
y' = W_0·x + Σ_{i∈T} ŷ_i · E_i(x)          # residual output

# Step 2: Expert representations (所有 expert 的中间输出)
e_j = E_j(x)  for j = 1..n                # 每个 expert 的输出表示 [D]

# Step 3: Contrastive loss construction
r = randint(1, k)                          # 随机 anchor 位置
a = T[r]                                   # anchor expert index
q = Normalize(e_a)                         # query [D]
P = {Normalize(e_{T[j]}) | j ≠ r}          # positive set, size k-1
N = {Normalize(e_j) | j ∉ T}              # negative set, size n-k

# Step 4: Similarity scores
s_pos = (q · P^T) / τ                      # [k-1]
s_neg = (q · N^T) / τ                      # [n-k]

# Step 5: InfoNCE loss
logits = [s_pos, s_neg]
L_con = -log( Σexp(s_pos) / (Σexp(s_pos) + Σexp(s_neg) + ε) )

# Step 6: Total loss
L_total = L_CE(y', y_true) + λ · L_con
L_total.backward()
```

CoMoE 还可以应用固定大小负采样策略：从 n-k 个非激活 expert 中随机采样固定数量作为负样本，将复杂度从 O(n) 降至 O(1)，训练时间与 expert 数量解耦（论文验证无性能损失）。

术语一般如何实现？如何使用？
- **实现框架**：基于 HuggingFace PEFT + transformers，在 MixLoRA / OMoE 等现有 MoE-LoRA 实现之上添加对比损失模块
- **关键超参数**：λ = 0.01（对比损失权重，通过消融确定），τ = 论文未明确说明（典型值 0.07-0.1），n = 4 experts，k = 2 (top-2 routing)
- **适用范围**：任何 top-k routing 的 MoE 架构（LoRA-MoE, DoRA-MoE 等），LLaMA-2 7B / Gemma 2B 已验证
- **训练成本**：n=4 时 3.5h on A6000（multi-task），固定采样策略下 O(1) 复杂度
- **推理效率**：对比损失仅用于训练，推理时无额外开销。CoMoE 推理延迟 3,789ms（vs MixLoRA 4,217ms，降低 10%）
- **论文未开源独立代码仓库**，但方法简单可基于标准 MoE-LoRA 实现复现
- **性能**：multi-task avg +1.3 accuracy (LLaMA-2 7B)，1.45% 可训练参数（vs MixLoRA 2.9%）

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---
