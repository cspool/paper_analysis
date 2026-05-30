## MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MC（Mixture Compressor），一种 training-free 的 MoE-LLM 混合压缩策略，包含两个阶段：
    - **PMQ（Pre-Loading Mixed-Precision Quantization）**：基于 expert 重要性（访问频率 ϕ × 激活权重 w × 量化重构误差 ϵ）构建 Integer Programming 模型，为每个 expert 分配 1/2/3-bit 的最优位宽，使用 GPTQ 执行量化。其余 attention/gating 模块统一 4-bit。
    - **ODP（Online Dynamic Pruning）**：基于 routing weight ratio w₁/w₀ 动态剪枝低置信度 expert，同时引入 token importance Iⱼ = ‖tⱼ‖₁ · (Σ Aⱼ,ᵢ)/(L-j) 保护关键 token（仅需保护 2%），防止 attention decay。
  - 实验比较：
    - PMQ vs Uni（GPTQ uniform quantization）、BSP（block score predictor, Li et al. 2024）、Hessian-based（HAWQ V2, Dong et al. 2020），在 1.57~2.54-bit 范围。
    - PMQ+ODP vs PMQ-only vs Uni，在不同 bit-width 下。
    - 压缩后 MoE vs 同规模 FP16 dense LLM（LLaMA2-7b/13b）。
    - 消融：bit-width 分配指标（random/routing weight/activation frequency/Hessian/F-norm/PMQ）、token protection ratio、pruning threshold μ、专家显著性权重 α/β/γ。
    - 不同量化技术兼容性：GPTQ vs Omniquant。
    - 挑战性 benchmark：GSM8K, HumanEval, Needle-in-a-haystack。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB（Mixtral 8×7b 用 2 卡，Mixtral 8×22b 用 4 卡用于 FP16 baseline；量化后模型在单张 A100-80GB 上测试），也测试了 RTX 3090。
  - CPU/内存：论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral 8×7b：总参数 49B（~96.8 GB FP16），32 decoder blocks，hidden_dim=4096，8 experts/layer，每 token 激活 top-2，激活参数 13B（~26.3 GB）。
    - Mixtral 8×22b：总参数 141B（~281.2 GB FP16），56 decoder blocks，hidden_dim=6144，8 experts/layer，每 token 激活 top-2，激活参数 39B（~76.5 GB）。
    - 对比 dense model：LLaMA2-7b, LLaMA2-13b（16-bit）。
  - 数据集/benchmark：
    - 校准数据：C4（128 组随机序列，每组 2048 tokens），用于计算 expert 显著性指标和 bit-width 配置。
    - 评估数据：
      - Perplexity（PPL↓）：WikiText2
      - 8 个 zero-shot benchmark（EleutherAI LM Harness, ↑）：PIQA, ARC-easy, ARC-challenge, BoolQ, HellaSwag, Winogrande, MathQA, MMLU
      - Few-shot：MMLU（5-shot）
      - 挑战性 benchmark：GSM8K（推理↑）, HumanEval（pass@10↑）, Needle-in-a-haystack（长上下文检索↑, NIAH）
    - 额外分析：MATH 数据集（用于观察 expert 激活分布差异）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Aaronhuang-778/MC-MoE
  - 量化工具链：GPTQ（Frantar et al. 2022）执行 expert 量化，HQQ（Badri & Shaji 2024）用于保存量化权重和反量化，Omniquant（Shao et al. 2023）作为替代量化方法验证兼容性。
  - 算法pipeline 伪代码：

```
输入: MoE-LLM 模型 M（含 L 层 MoE block，每层 N 个 expert），校准数据集 C4，目标平均位宽 k
输出: 压缩后模型 M_compressed

// ===== 阶段0: Expert 显著性分析（在原始 16-bit 模型上）=====
for each MoE block l in {1..L}:
    for each expert e_i in {1..N}:
        // 计算访问频率
        ϕ_i = n_i / N_calib                    // n_i: expert i 被激活的总次数
        // 计算激活权重和
        w_i = Σ_{j=1}^{N_calib} σ_i^j / N_calib  // σ_i^j: 第 j 次推理中 expert i 的 routing weight
        // 计算量化重构误差（F-norm）
        for each bit j in {1, 2, 3}:
            ε_{i,j} = ||F(θ) - F(θ[e_i → Q(e_i, j)])||_F

// ===== 阶段1: PMQ — Integer Programming 求解最优位宽分配 =====
for each MoE block l in {1..L}:
    // 定义 binary 决策变量 x_{i,j} ∈ {0,1}: expert i 分配 j-bit
    // 求解 Integer Programming:
    MINIMIZE  Σ_i Σ_j ϕ_i^α · w_i^β · (ε_{i,j} · x_{i,j})^γ
    Subject to:
        Σ_i Σ_j j · x_{i,j} = N · k           // 平均位宽约束
        Σ_j x_{i,j} = 1, ∀i                    // 每个 expert 只分配一个位宽
        Σ_i x_{i,3} ≥ 1, Σ_i x_{i,2} ≥ 1      // 至少一个 3-bit 和 2-bit expert
        x_{i,j} ∈ {0,1}
    // 得到位宽配置 B_i ∈ {1,2,3} for each expert i

// ===== 阶段1b: 应用 GPTQ 量化 =====
for each MoE block l:
    for each expert e_i:
        位宽 b = B_i
        if b == 1:
            // 二值化（见附录 A.2）
            B̃ = (sign(W) + 1) / 2             // 映射到 {0,1}
            s = ||W||_ℓ1 / (d × m)             // scaling factor
            存储: B̃ (bool) + s (float)
        else:  // b ∈ {2,3}
            使用 GPTQ 量化: W_q = GPTQ(W, X, b)
            // GPTQ: Hessian H=2XX^T + 逐列量化 + 误差补偿
    // Attention/gating 模块统一 4-bit GPTQ

// ===== 阶段2: ODP — Online Dynamic Pruning =====
// 在推理时对每个 token t 动态执行：

for each MoE block l in {1..L}:
    // 2a. 计算 token importance（基于上一层 attention map）
    for each token j:
        I_j = ||t_j||_1 · (Σ_{i≥j} A_{j,i}) / (L - j)
    // 保护 top-2% 重要 token：这些 token 的所有 top-k expert 都保留

    // 2b. 对非保护 token，基于 routing weight 剪枝
    {w_0, w_1} = Top-2{G(t)}                  // routing scores
    if token 未被保护 AND w_1/w_0 < μ:        // μ 取 calibration 数据的中位数
        剪枝 w_1 对应的 expert，仅用 w_0 对应的 expert 计算
        y = w_0 · E_0(t)                       // 从 top-2 降为 top-1
    else:
        y = w_0 · E_0(t) + w_1 · E_1(t)       // 保留 top-2

// ===== 一比特权重反量化（推理时）=====
// 对 b=1 的 expert，反量化为:
// s · xB = s(Σ_{j: B̃_{ij}=1} x_j - Σ_{j: B̃_{ij}=0} x_j)
// MACs: 仅 m 次乘法（vs FP16 的 d×m 次），复杂度 O(m) vs O(m²)
```

  - 关键超参数配置：
    - α=1, β=1, γ=2（expert 显著性权重因子，消融实验验证稳定）
    - token 保护比例：2%（ODP 阶段）
    - pruning threshold μ：取 calibration 数据上 w₁/w₀ 的中位数
    - 校准数据：C4，128 序列 × 2048 tokens
    - 量化时间：Mixtral 8×7b 约 90 分钟（GPTQ）
  - 1-bit 权重存储格式：通过 B̃ = (sign(W)+1)/2 将 ±1 映射到 {0,1}，真正用 1-bit 内存存储每个元素。反量化仅需 m 次乘法（vs FP16 的 dm 次）。
