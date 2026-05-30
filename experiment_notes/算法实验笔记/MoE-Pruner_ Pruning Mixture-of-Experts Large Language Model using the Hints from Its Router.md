## MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Pruner 是一种针对 MoE LLM 的一站式（one-shot）后训练剪枝方法，核心创新在于将 MoE router 的 gating 权重引入剪枝重要性度量：S = |W_ij| * ||X_j * Gate_j||，即对每个输出神经元，剪掉 weight magnitude × 输入激活 norm × router 权重最小的权值。相比 Wanda（S = |W_ij| * ||X_j||），MoE-Pruner 多乘了一个 router 权重项，利用 MoE routing 信息识别 expert 层中不重要的权值。
  - 实验比较：（1）One-shot 剪枝：MoE-Pruner vs SparseGPT vs Wanda，在 Mixtral-8x7B (base/instruct) 和 Mixtral-8x22B (base/instruct) 上以 50% 非结构化稀疏度和 2:4 半结构化稀疏度进行对比，指标为 WikiText perplexity 和 9 个 zero-shot 任务准确率（ARC-c, ARC-e, Boolq, HellaSwag, MMLU, OBQA, PIQA, RTE, WinoGrande）；（2）Expert-wise Knowledge Distillation 恢复：以未剪枝 pretrained model 为 teacher，对剪枝后 student 做逐 expert 的 MSE 蒸馏，评测 zero-shot 准确率恢复；（3）消融：校准样本数量（2-256）和剪枝率（10%-70%）对 perplexity 的影响。

- 硬件平台是什么，配置是什么。
  - 剪枝实验：单张 NVIDIA H100-80GB GPU。
  - 微调/蒸馏实验：2 台服务器，每台 8×NVIDIA H100-80GB GPU（共 16 卡）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B、Mixtral-8x7B-Instruct、Mixtral-8x22B、Mixtral-8x22B-Instruct（Jiang et al., 2024）。
  - 校准数据：C4（Raffel et al., 2020），固定 128 条序列用于所有 one-shot 剪枝实验。
  - 评估数据集：WikiText 验证集（perplexity）。
  - Benchmarks：EleutherAI LM Harness（Gao et al., 2023）上的 9 个 zero-shot 任务 — ARC-easy、ARC-challenge、Boolq、HellaSwag、MMLU、OpenBookQA、PIQA、RTE、WinoGrande。
  - 蒸馏训练集：C4 子集，仅需 1000 条训练样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方代码仓库链接，arXiv 页面和 HuggingFace papers 页面均未找到 GitHub URL。
  - 算法 Pipeline（基于论文 Algorithm 1）：
    1. **初始化**：给定 MoE 模型 M（l 个 MoE layer，每层 n 个 expert），校准数据 X ∈ R^{b×d_col}，目标稀疏度 p%。
    2. **逐层处理**：对每一层 t=1,...,l：
       a. Forward 前一层：X', G ← forward(layer_t, X)，得到当前层的输入激活 X' 和 router 权重 Gate ∈ R^{b×n}。
       b. 对每个 expert e=1,...,n：
          - 初始化 binary pruning mask M ← 1_{d_row × d_col}
          - 计算重要性分数：S ← |W_ij| * ||X_j * Gate_j||（对每个输出神经元 j，Gate_j 是 router 分配给该 expert 的归一化权重广播到所有输入维度，X_j 是输入激活的第 j 列，逐元素乘法后取 L2 norm）
          - 沿 dim=1 对 S 排序，取最不重要的 d_col*p% 个位置
          - M 中对应位置置 0，W ← M ⊙ W（剪枝后的权重为零）
       c. X ← X' 传递给下一层。
    3. **返回**：剪枝后的模型 M'。
