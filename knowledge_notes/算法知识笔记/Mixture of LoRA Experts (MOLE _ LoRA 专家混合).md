## Mixture of LoRA Experts (MOLE / LoRA 专家混合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture of LoRA Experts (MOLE) 由 Wu et al. (Microsoft Research Asia, 2024) 提出，是一种用于组合多个预训练 LoRA adapter 的方法。核心设计：(1) 将每个已训练 LoRA 的每一层视为一个独立 expert（而非整个 LoRA 为一个 expert），即 N 个 LoRA × M 个 transformer block 产生 N×M 个 expert；(2) 在每个 transformer block 层级嵌入一个可学习的 gating function $\mathcal{G}(\cdot)$，接收该层所有 LoRA 的输出 $\{E_{\Delta\theta_i}(x)\}_{i=0}^{N-1}$，输出 N 维 softmax 分布作为组合权重；(3) 训练时仅优化 gating 参数（e 向量和 τ 温度标量），冻结所有 LoRA 和预训练模型权重，极低训练开销；(4) 推理时支持双模式：全专家模式（所有 LoRA 参与，gating 自动分配权重）和 mask 模式（手动排除不需要的 LoRA，gating 按比例重新分配剩余权重，无需重训练）。

从算法pipeline角度拆解术语：
MOLE 单 transformer block 的前向计算（§3.2 Eq.5-13）：
```
输入: x ∈ R^{L×d}, 预训练 block θ, N 个 LoRA {Δθ_i}

# Step 1: 预训练 block 前向
F_θ(x) = x + f_Attn(LN(x)|θ) + f_FFN(LN(x + f_Attn(LN(x)|θ))|θ)

# Step 2: 每个 LoRA expert 独立前向（可并行）
for i in range(N):
    E_Δθi(x) = x + f_Attn(LN(x)|Δθ_i) + f_FFN(LN(...)|Δθ_i)

# Step 3: Gating 计算组合权重
E_Ω(x) = Normalize(concat([E_Δθ0(x), ..., E_Δθ{N-1}(x)]))  # [N·L·d]
ε = flatten(E_Ω(x))^T @ e                                    # e ∈ R^{N·L·d × N}
G_i = exp(ε_i / τ) / Σ_j exp(ε_j / τ)                        # τ learnable

# Step 4: 加权组合 + 残差融合
O(x) = F_θ(x) + Σ_i G_i · E_Δθi(x)
```
与 NLA 的关键区别：MOLE 在"block 输出空间"（而非"权重空间"）组合，每个 LoRA 需独立计算完整 block 输出。

MOLE 与 LoRA-based MoE（如 MixLoRA、MOELoRA）的本质区别：
| 维度 | LoRA-based MoE | MOLE |
|------|---------------|------|
| Expert 定义 | LoRA adapter 作为 expert | 整个 LoRA block 输出作为 expert |
| Router 位置 | 每个 token 路由到 expert | 每层 gating 加权组合所有 expert |
| 训练方式 | 联合训练 router + LoRA | 仅训练 gating（LoRA 预训练并冻结） |
| LoRA 来源 | 同一训练任务中学习 | 独立预训练的多个 LoRA |
| 应用场景 | 多任务微调 | 多 LoRA 组合/融合 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文声明开源在 github.com/yushuiwx/MoLE.git（2026年已 404）。
- V&L 域训练：基于 DreamBooth + Stable Diffusion V2.1，400 iterations，lr=1e-5，batch=2，α=0.5（L = L_CLIP + α·L_balance）。CLIP 提供 local+global guidance 作为无监督训练目标。
- NLP 域训练：基于 FLAN-T5，800 iterations，lr=1e-5，batch=12，α=0.5（L = L_task + α·L_balance）。
- 适用：需要将多个独立获取的 LoRA adapter（如社区发布的角色/风格 LoRA）组合使用的场景。推理灵活性（mask 模式）使其适合交互式 LoRA 组合（用户手动选择保留哪些 LoRA 特征）。

涉及论文标题：
- Mixture of LoRA Experts
