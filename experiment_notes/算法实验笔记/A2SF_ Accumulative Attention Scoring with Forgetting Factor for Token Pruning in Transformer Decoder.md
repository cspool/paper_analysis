## A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 A2SF（Accumulative Attention Score with Forgetting Factor）算法，在 Attention Score 累积过程中引入 Forgetting Factor α（0 < α < 1），对过去的 Attention Score 施加指数衰减惩罚，解决 Transformer Decoder 中因 Causal Mask 导致早期 token 累积次数过多而产生的不公平比较问题。实验比较 A2SF 与 Full cache（无剪枝上限）、Local Attention（仅保留最近 token）和 H2O（基于 A2S 的 token 剪枝）在不同 cache ratio [0.1, 0.8] 下的准确率和与 Ideal Mask 的 cosine similarity。

- 硬件平台是什么，配置是什么。
  RTX 3090 GPU，FP16 精度。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B、LLaMA-7B、OPT-6.7B、OPT-2.7B。
  数据集/Benchmark：OpenbookQA、Winogrande、PiQA、COPA、MathQA、ARC-easy、ARC-challenge，使用 lm-eval-harness (v0.4.0) 在 0-shot 和 1-shot 设置下评估 Commonsense-reasoning 性能。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Dirac-Notation/A2SF

  **算法核心（伪代码）**：
  输入：每层每头的 Attention Score 矩阵 S ∈ R^{N×N}，Forgetting Factor α
  输出：每 token 的重要性分数 A_k，用于决定保留/剪枝

  ```
  // H2O baseline: 直接累积 Attention Score（考虑 causal mask）
  for n in 1..N (generation step):
      for k in 1..n (key token, k <= n due to causal mask):
          A_k += S[n][k]  // 早期 token (k 小) 被累加更多次 → 不公平

  // A2SF: 引入 Forgetting Factor 的累积
  for n in 1..N (generation step):
      for k in 1..n:
          A_k += α^(n - generation_step_of_score) × S[n][k]
  ```

  **张量计算形式**：
  公式 (5-6)：
  $$A_{n,k}^h = \sum_{q=1}^n \alpha^{n-q} \times S_{q,k}^h$$
  $$A_{n,k}^{h} = S_{n,k}^{h} + \alpha \cdot S_{n-1,k}^{h} + \alpha^{2} \cdot S_{n-2,k}^{h} + \dots + \alpha^{N-k} \cdot S_{k,k}^{h}$$

  其中 α ∈ (0, 1)。每次生成新 token 时，所有历史 Attention Score 乘以 α 后再加入新的 Score。越早的 Score 经历越多次 α 乘法，趋近于 0。这使得近期 Attention Score 权重更大，消除 token 生成顺序造成的累积次数不平衡。

  **执行流程**：
  1. 每层每头计算 Attention Score 矩阵（带 Causal Mask）
  2. 按 A2SF 公式累积带遗忘因子的重要性分数
  3. 在下一 Generation Step 前，按 A_k 排序，保留前 K 个 token（K = cache_ratio × N），剪枝其余 token 的 KV Cache
  4. A2SF 不分配 local cache，全部 cache budget 用于 selective cache（与 H2O 各半分配不同）

  **关键超参数**：
  - Forgetting Factor α：实验表明最优范围为 [0.1, 0.3]
  - Cache Ratio：在 [0.1, 0.8] 范围内评估
  - α = 0.0 等价于完全不用历史，仅用当前步 Attention Score
  - α = 1.0 等价于 H2O 的原始 A2S（无衰减）
