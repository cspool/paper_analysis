## Mixture of LoRA Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **Mixture of LoRA Experts (MOLE)**，将每个已训练的 LoRA 的每一层视为一个独立 expert，在每层引入可学习的 gating function，通过 hierarchical weight control 对多个 LoRA 的层输出加权组合，实现灵活、动态、低开销的多 LoRA 组合。训练时仅优化 gating function 参数（冻结所有 LoRA 和预训练模型参数），推理时支持两种模式：(1) 使用全部已训练 LoRA 并自动分配权重；(2) 手动 mask 不需要的 LoRA 后按比例重新分配权重。
  - V&L 域实验比较 MOLE vs (a) Normalized Linear Arithmetic composition (NLA, Eq.2) 和 (b) SVDiff（均为 LoRA composition baseline），以及 full-parameter training baseline (Custom, Textual Inversion)。评估指标为 CLIP feature space 下的 Text-alignment 和 Image-alignment。
  - NLP 域实验比较 MOLE vs (a) LoRAHub 和 (b) PEMs。评估任务包括 Translation（WMT14/16, BLEU）、Struct to Text（CommonGen/DART/E2ENLG/WebNLG, Rouge-1/2/L）、Closed-Book QA（ARC-c/ARC-e/NQ/TQA, EM）、BBH（7 subtasks, EM）、NLI（ANLI-R1/R2/R3/QNLI, EM）。
  - 消融实验：w/ vs w/o gating balancing loss L_balance；仅调大温度 τ 替代 L_balance 的多组对比；coarse-to-fine gating granularity（matrix-wise m-MoLE / layer-wise l-MoLE / block-wise b-MoLE / network-wise n-MoLE）；LoRA 数量扩展（8/24/48/128 在 NLP，3/4/5/6 在 V&L）；跨任务泛化（NLI 任务训练 → BBH 评估）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的具体 GPU 硬件配置。
  - V&L 域：DreamBooth 基于 Stable Diffusion V2.1，图像分辨率 512×512，DDPM sampler 50 steps，scale=7.5。
  - NLP 域：基于 FLAN-T5（Chung et al., 2022），具体参数量论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - V&L 域模型：DreamBooth (Ruiz et al., 2023) 基于 Stable Diffusion V2.1，以 Stable Diffusion V2.1 为 base generator。
  - NLP 域模型：FLAN-T5（Chung et al., 2022）。
  - V&L 域数据集：15 组不同三概念组合（如"Fancy boot + Monster + Clock"等，见 Table 1），每组 200 张生成图像 × 5 个 text prompt 评估。训练数据未明确说明（使用 CLIP 的 local + global guidance 做无监督训练优化 MoLE）。
  - NLP 域数据集/benchmark：Translation（WMT'14 En↔Fr, WMT'16 En↔De/En↔Ro）、Struct to Text（CommonGen, DART, E2ENLG, WebNLG）、Closed-Book QA（ARC-c, ARC-e, NQ, TQA）、Big-Bench Hard（Boolean Expressions, Causal Judgement, Date Understanding, Disambiguation, Penguins in a Table, Reasoning about Colored Objects, Ruin Names）、NLI（ANLI-R1, ANLI-R2, ANLI-R3, QNLI, WNLI）。
  - NLP 域 LoRA 训练数据：各 LoRA 从 FLAN 数据集的不同子集训练获得。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明代码开源在 https://github.com/yushuiwx/MoLE.git，但该链接返回 404（2026-05-27 确认，可能已迁移或删除）。
  - 算法pipeline 核心计算流程（单 transformer block，基于论文 §3.2 Eq.5-13）：

  ```
  # 输入: x ∈ R^{L×d}, 预训练 block θ, 已训练 LoRA 集合 Ω={Δθ_i}_{i=0}^{N-1}
  
  # 1. 预训练 block 前向
  x_θ'    = x + f_Attn(LN(x) | θ)                         # Eq.5
  F_θ(x)  = x_θ' + f_FFN(LN(x_θ') | θ)                    # Eq.6
  
  # 2. 每个 LoRA expert i 的前向
  x_Δθi'       = x + f_Attn(LN(x) | Δθ_i)                  # Eq.7
  E_Δθi(x)     = x_Δθi' + f_FFN(LN(x_Δθi') | Δθ_i)        # Eq.8
  
  # 3. Gating 函数计算组合权重
  E_Ω(x) = Normalization(E_Δθ0(x) ⊕ ... ⊕ E_Δθ{N-1}(x))   # Eq.9, concat: R^{N·L·d}
  ε      = Flatten(E_Ω(x))^T · e                           # Eq.10, e ∈ R^{N·L·d × N}
  G_i    = exp(ε_i/τ) / Σ_j exp(ε_j/τ)                     # Eq.11, τ learnable
  
  # 4. 加权组合
  Ẽ_Ω(x) = Σ_i G_i · E_Δθi(x)                              # Eq.12

  # 5. 最终输出
  O(x) = F_θ(x) + Ẽ_Ω(x)                                   # Eq.13
  ```

  - 训练时仅优化 gating function 参数 e 和 τ（冻结 θ 和所有 Δθ_i），总可训练参数量为 O(N·L·d·N) + 1（仅 Eq.10 的 e 和 Eq.11 的 τ）。
  - V&L 域训练目标：L = L_CLIP（local + global guidance） + α · L_balance（α=0.5），400 iterations，lr=1e-5，batch size=2。
  - NLP 域训练目标：L = L_FLAN-T5（cross-entropy） + α · L_balance（α=0.5），800 iterations，lr=1e-5，batch size=12。
  - Gating balancing loss：L_balance = -log(Π_i q^(i))，其中 q^(i) = (1/M)·Σ_k exp(ε_i^k/τ) / Σ_j exp(ε_j^k/τ)，M 为嵌入 gating 的 block 数。该 loss 在 gating 均匀分布时最小化，防止 gating 坍塌到少数 LoRA。

- 模型是什么。数据集和bench分别是什么。
  - 模型：M3oE，包含 Domain Representation Extraction Layer（含 domain-specific/shared 权重矩阵元素乘 + domain-agnostic mapping）、Multi-View Expert Learning Layer（共享专家 N 个、域专家 D 个、任务专家 T 个，均为单层 MLP + LayerNorm + ReLU）、MDMT Objective Prediction Layer（D×T 个两层 MLP prediction tower，Sigmoid 输出）。
  - 数据集：(1) MovieLens-1M（~100万评分，~3900电影，用 "age" 特征切分为 3 个域，"click"/"like" 2 个任务）；(2) KuaiRand-Pure（快手短视频平台数据，用 "tab" 特征切分为 3 个域，"click"/"long-view" 2 个任务）。训练/验证/测试分割比例为 8:1:1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Applied-Machine-Learning-Lab/M3oE
  - 算法pipeline 伪代码：

```
输入: 样本 x_d (来自域 d), 域数 D, 任务数 T
输出: 所有域-任务对的预测 y_hat_{d,t}

// 1. Domain Representation Extraction
for each domain d:
    W_hat_d = W_d ⊙ W_sh                          // element-wise product
    h_d = W_c * (W_hat_d * x_d + b_d + b_sh) + b_c + f_DA(x_d)

// 2. Shared Expert Module (N experts)
for each shared expert e in {1..N}:
    f_E^e(h_d) = ReLU(LayerNorm(W_e * h_d + b_e))
// Shared fusion with D×T gates
for each (d,t):
    S_{d,t}(h_d) = softmax(f_gate_{d,t}(h_d)) · [f_E^1(h_d), ..., f_E^N(h_d)]

// 3. Domain Expert Module (D experts)
for each domain expert k in {1..D}:
    f_E^k(h_d) = ReLU(LayerNorm(W_k * h_d + b_k))
// Domain fusion (biased)
D(h_d) = β_d * f_E^d(h_d) + (1-β_d)/(D-1) * Σ_{k≠d} f_E^k(h_d)

// 4. Task Expert Module (T experts)
for each task expert k in {1..T}:
    f_E^k(h_d) = ReLU(LayerNorm(W_k * h_d + b_k))
// Task fusion (biased)
T(h_d) = β_t * f_E^t(h_d) + (1-β_t)/(T-1) * Σ_{k≠t} f_E^k(h_d)

// 5. Multi-View Representation Balancing (Two-Level Fusion)
h̄_d = S_{d,t}(h_d) + α_d * T(h_d) + α_t * D(h_d)

// 6. Prediction
for each (d,t):
    y_hat_{d,t} = Sigmoid(W2_{d,t} * ReLU(W1_{d,t} * h̄_d + b1_{d,t}) + b2_{d,t})

// 7. AutoML - Bi-Level Optimization
for epoch in 1..E:
    更新模型参数 W = argmin_W L(W, α, β)         // 外层
    更新融合权重 α, β = argmin_{α,β} L(W*, α, β)  // 内层（基于一个 mini-batch）
    // α_d, α_t, β_d, β_t 由可训练标量经 Sigmoid 生成: w = Sigmoid(e_w)
```

- 关键参数配置：
  - embedding size = 16
  - MovieLens: N=1 shared experts, lr=1e-2
  - KuaiRand-Pure: N=4 shared experts, lr=3e-3
  - D=3 domain experts, T=2 task experts, D×T=6 prediction towers
  - Loss: Binary Cross Entropy，所有域和任务加和
