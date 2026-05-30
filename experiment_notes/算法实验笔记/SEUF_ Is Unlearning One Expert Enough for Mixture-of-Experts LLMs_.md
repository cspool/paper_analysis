## SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **SEUF (Selected Experts Unlearning Framework)**——一种针对 MoE LLM 的参数高效 unlearning 框架。核心包含三步：(1) **Expert Attribution**：对 forget set 中每个 token，记录 Router 输出的 gating score g_{i,t}，按式 s_i = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t} 计算每个 expert 的 affinity score，跨所有 layer 排序选出 top-M affinity 最高的 expert 作为"target expert"（默认 M=1）；(2) **Router Anchor Loss**：式 L_anchor = ||g - a||_2^2，其中 a_i = 1 当第 i 个 expert 为 target expert 否则 a_i = 0，强制 router 在 unlearning 过程中持续激活该 target expert，防止"expert selection shift"（router 切换激活非目标 expert 来作弊降低 forget loss）；(3) **Focused Unlearning**：仅对选中的 target expert 及其对应 router 进行梯度更新（冻结其他所有参数），更新参数仅约 0.06%。总损失为 min_θ l_f(θ; D_f) + λ l_r(θ; D_r) + α L_anchor。SEUF 为即插即用框架，可集成 GA、GDIFF、NPO、RMU 等任意现有 unlearning 方法。
  - 实验比较：(a) 四种 unlearning 方法（GA、GDIFF、NPO、RMU）在 w/ vs w/o SEUF 下的 FE（Forget Efficacy，越低越好）和 UT（Model Utility on MMLU，越高越好）对比（Table 3），涵盖 Qwen1.5-MoE-A2.7B-Chat 和 DeepSeek-V2-Lite 两个模型，WMDP 和 RWKU 两个 benchmark；(b) SEUF vs PEFT baseline（LoRA、ESFT）的参数效率和 unlearning 效果对比（Table 3, Table 4）；(c) Top-1 expert selection（affinity score-based）vs Random selection 的消融实验（Table 3 最后一行）；(d) 不同 affinity score 排名的 expert 对 utility 的影响（Table 5，排名 #1 → UT 0.5485，排名 #26 → UT 0.2355）；(e) Top-M 选择消融（M=1/3/6，same layer vs different layers, Table 2+Table 7）；(f) 对抗攻击鲁棒性：GCG jailbreak 攻击下 FE 保持 0.01 不变（Sec. 5），证明 unlearned knowledge 不可恢复；(g) Mixtral-8x7B 大模型扩展实验（Table 9）；(h) hyperparameter α 敏感性分析（Table 6，α=1 最优）。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU（论文在 Sec. 5 中明确提到 "∼ 1 GPU hour on an A100 per soft prompt" 用于 GCG 攻击实验）。其他训练/推理所用 GPU 具体型号和数量论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) **Qwen1.5-MoE-A2.7B-Chat**（Qwen）：14.3B 总参数，2.7B 激活参数，upcycle-from-dense 训练方案；(2) **DeepSeek-V2-Lite**（DeepSeek）：16B 总参数，2.4B 激活参数，train-from-scratch 训练方案，包含 shared experts；(3) **mistralai/Mixtral-8x7B-Instruct-v0.1**（Mixtral）：45B 总参数，12.9B 激活参数，仅用于 SEUF vs PEFT 比较。
  - 数据集：(1) **WMDP** benchmark（Li et al., 2024）：评估移除 biosecurity、cybersecurity、chemical security 领域的 hazardous knowledge，使用 WMDP-Cyber 子集作为 forget set，MMLU 作为 utility evaluation；(2) **RWKU** benchmark（Jin et al., 2024）：评估消除 200 个真实世界名人信息，选取 100 人作为 unlearning target，使用 train_original_passage 中 Wikipedia 描述作为 forget set。
  - Benchmarks/metrics：**FE (Forget Efficacy)**——WMDP 上为 forget set 的四选一多选题准确率（理想值 0.25 即随机猜测），RWKU 上为 fill-in-the-blank 和 QA 任务的 Rouge-L recall（理想值 0.0）；**UT (Model Utility)**——MMLU zero-shot 准确率（越高越好）。使用 LM Evaluation Harness 进行评测。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供代码仓库链接。评估使用开源 LM Evaluation Harness（Gao et al., 2024）。
  - 算法 Pipeline 核心伪代码：

```
# === SEUF Unlearning (Algorithm 1) ===
# Input: pretrained_model θ_o, forget_set D_f, retain_set D_r
# Output: unlearned_model θ_u

# Step 1-3: Expert Attribution
D_s = sample_subset(D_f, n_tokens=100000)     # ~100K tokens for robust attribution
affinity_scores = {}                            # dict: expert_id → score
for each sample x_j in D_s:
    for each layer l in model:
        for each token t at position:
            g_{i,t}^{(l)} = Softmax(Router(u_t^{(l)}))[i]  # router gating score
            # accumulate: s_i = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t}^{(l)}
            affinity_scores[(l,i)] += g_{i,t}^{(l)} / len(x_j)

# Rank experts globally by affinity score, select top-M (M=1)
all_experts = sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)
e_M = all_experts[:M]                           # selected target expert(s)

# Step 4: Activate only target expert & its router
for param_name, param in model.named_parameters():
    param.requires_grad = False                  # freeze all
for expert in e_M:
    expert.expert_weights.requires_grad = True   # unfreeze target expert
    expert.router.requires_grad = True           # unfreeze corresponding router

# Step 5: Unlearn with anchor loss
for batch in D_f ∪ D_r:
    # Forward pass
    for layer l:
        g^{(l)} = Router(u_t^{(l)})              # router output probabilities

        # Anchor loss: force router to keep target expert active
        a^{(l)} = [1 if i in e_M else 0 for i in range(num_experts)]
        L_anchor += ||g^{(l)} - a^{(l)}||_2^2

        # Expert computation (standard MoE)
        s^{(l)} = Softmax(g^{(l)})
        topk_indices = TopK(s^{(l)}, K)
        h' = u + Σ_{i∈topk} s_i * FFN_i(u)

    # Compute unlearning loss
    L = L_forget(D_f) + λ * L_retain(D_r) + α * L_anchor

    # Gradient update (only target expert + router params updated)
    θ ← θ - η * ∇_θ L
```

  - 核心张量计算——Router Anchor Loss：
    - Router 对第 l 层输出 gating score: g^{(l)} ∈ R^{E} （E 个 expert 的概率分布，经由 Softmax）
    - Anchor target: a^{(l)} ∈ {0,1}^{E}，target expert 对应位置为 1，其余为 0
    - Anchor Loss: L_anchor^{(l)} = Σ_i (g_i^{(l)} - a_i^{(l)})^2，对所有 MoE layer 求和
    - 效果：MSE loss 强制 router 在 unlearning 时保持 target expert 的 gating score 接近 1，防止 router 将 token 路由到非目标 expert 来"作弊"降低 forget loss
  - Expert Attribution 核心张量计算：
    - 对 calibration set Z 个样本，第 j 个样本序列长度 L_j
    - 第 l 层第 i 个 expert 的 affinity: s_i^{(l)} = (1/Z) Σ_{j=1}^{Z} (1/L_j) Σ_{t=1}^{L_j} g_{i,t}^{(l)}
    - g_{i,t}^{(l)} 是 Router Softmax 后第 i 维的标量值
    - 跨所有 layer 全局排序，选 top-1 expert（M=1 时性能最优，Insight 4）
