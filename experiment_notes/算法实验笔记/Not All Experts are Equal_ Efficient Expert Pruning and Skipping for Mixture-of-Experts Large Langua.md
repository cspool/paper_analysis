## Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出两种后训练 expert-level 稀疏化方法：(1) **Expert Pruning**：逐层枚举 expert 组合，以最小化 token 重建损失（Frobenius norm）选择保留 r 个 expert，永久丢弃 n−r 个不重要 expert；(2) **Dynamic Expert Skipping**：推理时根据 routing weight 比值 w_{e1}/w_{e0} < β 动态跳过次要 expert，β 从校准集每层中位数确定。实验比较：(a) 与 Wanda 2:4 结构化剪枝的性能/内存/速度对比；(b) 与 Random Pruning、Frequency-based Pruning 等 expert 剪枝 baseline 的 zero-shot 精度对比；(c) task-agnostic (C4 校准) vs task-specific (MATH 校准) 的 domain 效果对比；(d) expert pruning + dynamic skipping 组合的 LM-eval 精度与 token 生成速度 trade-off。

- **硬件平台是什么，配置是什么。**
  NVIDIA A100-80G GPU。原始 Mixtral 8x7B (bf16) 需 2 块 A100-80G 加载；prune 2 个 expert（r=6）后仅需 1 块 80G GPU；prune 4 个 expert（r=4）内存降至 46,879 MB。fine-tuning 实验使用 16 块 A100-80G GPU。推理速度测试基于 AutoGPTQ speed benchmark 脚本修改。

- **模型是什么。数据集和bench分别是什么。**
  模型：Mixtral 8x7B 和 Mixtral 8x7B Instruct。校准集：task-agnostic 用 C4（128 序列×2048 tokens），task-specific 用 MATH training set。Benchmarks：(a) EleutherAI LM Harness 8 项 zero-shot（ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande）；(b) GSM8K 5-shot；(c) MATH zero-shot；(d) fine-tuning 用 MetaMathQA（训练 900 steps, lr=2e-5, cosine scheduler）。

- **开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。**
  代码开源：https://github.com/Lucky-Lance/Expert_Sparsity。使用 HuggingFace Transformers，prune 后仅需修改 model config 即可加载。

  **Expert Pruning 伪代码（逐层）：**
  ```
  # 第1步: 用校准集对原始模型做推理，缓存每层 MoE 输入输出
  for each sample in calibration_set:
      for each MoE_layer l:
          cache: X_l (input tokens), Y_l = F_l(X_l) (original output)

  # 第2步: 逐层枚举 expert 组合
  for each layer l:
      best_loss = inf
      for each subset C of {expert_0,...,expert_{n-1}} with |C| = r:
          # 构建 prune 后 MoE 层 F'_l(·, C)，仅保留 C 中 expert 及对应 routing weight
          Y'_l = F'_l(X_l, C)
          loss = ||Y'_l - Y_l||_F   # Frobenius norm 重建损失
          if loss < best_loss:
              best_loss = loss
              best_C = C
      保留 best_C，丢弃其余 n−r 个 expert

  # 逐层拼接得到 r-expert MoE 模型
  ```

  **Dynamic Skipping 伪代码（推理时逐 token, top-2 场景）：**
  ```
  for each token x in sequence:
      计算 routing weights w = Softmax(l)
      选 top-2 expert: e0 (w_{e0} 最大), e1 (w_{e1} 次大)
      if w_{e1} < β * w_{e0}:    # β per-layer 超参，取校准集中位数
          仅使用 expert e0：z = E_{e0}(x)
      else:
          使用两个 expert：z = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)
  ```

  张量计算流程：input token x ∈ R^{d} → Router 计算 logits l ∈ R^n → Softmax → w ∈ R^n → top-k 选择 → 对选中 expert e_j 计算 SwiGLU FFN：x → W_gate·x ⊙ SiLU(W_up·x) → W_down·(result) → output = Σ w̃_{e_j}·E_{e_j}(x)。Prune 后仅保留 r 个 expert，移除其他 expert 的权重矩阵及 routing weight。Dynamic skipping 在不修改模型参数的前提下运行时决定调用 1 或 2 个 expert。
