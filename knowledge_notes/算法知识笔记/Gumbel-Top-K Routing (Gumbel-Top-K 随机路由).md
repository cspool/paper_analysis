## Gumbel-Top-K Routing (Gumbel-Top-K 随机路由)

术语解释
一种基于 Gumbel-Max Trick 的无放回随机采样方法，在 MoE 推理时通过向 router logits 添加 Gumbel 噪声来实现受控的 expert 选择随机化，等价于从 router 定义的 categorical 分布中无放回采样 k 个 expert。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gumbel-Top-K Routing 是将 Gumbel-Max Trick 的 TopK 扩展版本应用于 MoE expert routing 的机制。其数学基础为：

1. **Gumbel-Max Trick**（Gumbel, 1954）：给定 logits $\phi_1, ..., \phi_E$，若 $G_i \sim \text{Gumbel}(0,1)$ 独立同分布，则 $\arg\max_i(\phi_i + G_i)$ 等价于从 $\text{Categorical}(\text{softmax}(\phi))$ 中采样一次。
2. **Gumbel-Top-K 扩展**（Kool et al., ICML 2019）：$\text{TopK}(\phi + G, k)$ 等价于从该 categorical 分布中**无放回**顺序采样 k 个元素。
3. **RoE 中的应用**：在 MoE router logits $\mathbf{R} \in \mathbb{R}^E$ 上添加缩放 Gumbel 噪声后做 TopK 选择：$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$，其中 $\tau \geq 0$ 为温度参数控制随机性程度。

温度参数 $\tau$ 的作用：
- $\tau = 0$：退化为标准确定性 TopK routing
- $\tau$ 中等：高 logit expert 仍更可能被选中（Gumbel-Max 性质保证），但低 logit expert 也有机会被激活
- $\tau \to \infty$：退化为纯均匀随机选择，预测质量下降

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RoE 单 token 生成中，Gumbel-Top-K Routing 的 pipeline 位置如下：

```
# 单 MoE 层的 Gumbel-Top-K Routing Forward
输入: hidden_state h ∈ R^d, router weight W_r ∈ R^{E×d}, 温度 τ, top-k 数 k
输出: expert 输出 y ∈ R^d

# Step 1: Router logits
R = W_r @ h          # (E,)  router logits per expert

# Step 2: Add scaled Gumbel noise
U = rand(E)          # Uniform(0,1) i.i.d.
G = -log(-log(U))    # Gumbel(0,1) via inverse CDF: F^{-1}(u) = -log(-log(u))
noisy_R = R + τ * G  # (E,)  perturbed logits

# Step 3: Top-K expert selection (无放回采样)
topk_values, topk_indices = topk(softmax(noisy_R), k)

# Step 4: Weighted expert aggregation (standard MoE)
y = Σ_i topk_values[i] * Expert_FFN_i(h)

return y
```

在整个 RoE pipeline 中，上述过程在每层 MoE 对 batch 中的每个 sample 独立执行一次，产生 n 条不同的内部计算路径。batch 内第一个 sample（index 0）在启用 Clean Cache 时使用 τ=0 确定性路由作为"clean path"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
- **Gumbel 噪声生成**：标准实现为 $G = -\log(-\log(U))$，其中 $U \sim \text{Uniform}(0, 1)$。需加 $\epsilon$（如 1e-20）防止 $\log(0)$。
- **PyTorch 等效**：`torch.distributions.Gumbel(0, 1).sample()` 或手动 `-torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)`。
- **温度调优**：$\tau$ 为逐层超参数，通过 Optuna TPE 在验证集上搜索最佳值。搜索空间约束为 $[0, 0.5]$（论文经验观察：$\tau > 0.5$ 引入过多噪声导致性能下降）。
- **首尾层保护**：前 k 层和最后 k 层固定 $\tau = 0$（确定性路由），仅中间层参与 Gumbel-Top-K 随机化。论文实验表明初始层处理 raw embedding、最终层整合输出信息，对路由扰动更敏感。
- **跨任务差异**：不同任务（数学/常识/代码）的最优 $\tau$ 分布差异显著，需分别调优。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE
