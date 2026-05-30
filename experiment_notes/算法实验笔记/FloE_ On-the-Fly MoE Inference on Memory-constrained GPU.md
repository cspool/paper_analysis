## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  - FloE 提出 **Hybrid Compression（混合压缩）** 方案，对 MoE expert 内部的三组投影矩阵采用不同压缩策略：
    1. **Contextual Sparsification（上下文化稀疏化，Section 3.2.1）**：对 gate projection W_gate 和 down projection W_down 执行基于激活幅值的剪枝。具体为对 up projection 的输出激活 a_up = x·W_up 按阈值 t 做稀疏化 S_t(a_up)，保留 |a_up|≥t 的通道，将对应 gate projection 列和 down projection 行一同剪枝。阈值 t 由目标稀疏率 k 从样本数据集的激活绝对值经验CDF反向确定（t = min{t': F(t') ≥ k}）。
    2. **Ultra-Low-Bit Quantization（超低位量化，Section 3.2.2）**：仅对 up projection W_up 施加 INT2 HQQ 量化，因为 up projection 对量化最不敏感（在 INT2 下 perplexity 仅为 gate 量化的 46% 和 down 量化的 27%）。
    3. **Dual Sparsity Predictors（双稀疏预测器，Section 3.3）**：
       - Inter-expert 学习型预测器：利用当前层 hidden state 和历史 expert 选择轨迹，通过单层/双层 MLP（32K~2M 参数）预测下一层激活 expert，平均 precision 0.88。
       - Intra-expert 复用型预测器：用当前层 hidden state 与下一层 W_up（复用）直接做矩阵乘法，近似估计 up projection 输出激活，预计算上下文稀疏分布，平均 recall 0.95，零额外内存开销。
  - 实验比较：在 7 个下游任务（ARC-Easy/Challenge, BoolQ, SciQ, OpenBookQA, Winogrande, MMLU@5）上对比 CATS（激活稀疏化）、CHESS（通道级阈值稀疏化）、HQQ 量化；在 WikiText-2 上评估 perplexity；在 ShareGPT 上评估端到端生成速度。

- 硬件平台是什么，配置是什么。
  - 端到端延迟测试：GeForce RTX 3090（24GB VRAM），64核 CPU @2.3GHz，256GB DRAM，PCIe 4.0。
  - 单 expert 延迟测试：H100, A100, A6000, GeForce RTX 3090。

- 模型是什么。数据集和bench分别是什么。
  - 主要模型：Mixtral-8×7B（FP16，每层 8 expert，每 token 激活 2 expert）。
  - 验证模型：Phi-3.5-MoE-instruct, DeepSeek-MoE-16B-Base, Qwen1.5-MoE-A2.7B, DeepSeek-V2。
  - 数据集：C4（稀疏性分析和单 expert 延迟），WikiText-2（perplexity），ShareGPT（端到端生成效率），EleutherAI LM Harness（下游任务：ARC-Easy/Challenge, BoolQ, SciQ, OpenBookQA, Winogrande, MMLU@5）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文发表于 ICML 2025。论文未明确给出代码开源链接。可参考的相关开源项目：HQQ（https://github.com/mobiusml/hqq）、CATS（https://openreview.net/forum?id=v3w2a7EInO）。
  - 算法 pipeline 伪代码（基于 Algorithm 1）：
    ```
    Input: hidden state x, sparse threshold t_ij, expert weights {W_gate, W_down^T, W_up}
    1. v = x @ W_up                          # 全精度 up projection
    2. mask = (|v| > t_ij)                   # 按阈值生成稀疏掩码
    3. x' = SiLU(x @ W_gate[mask]) ⊙ v[mask] # 仅加载被掩码选中的 gate 列
    4. y = (W_down^T[mask] @ x')^T           # 仅加载被掩码选中的 down 列（列主序转置存储）
    Return: y
    ```
    核心张量操作链：hidden state x (1×4096) → up projection 全精度计算 → 幅值阈值化 → 掩码选择 gate 列和 down 行 → 稀疏 gate GEMV + SiLU → Hadamard 积 → 稀疏 down GEMV → 输出。结合混合压缩，实际传输量从 ~300MB/expert（FP16）降至约 ~32MB/expert（9.3×压缩）。
