## MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoDE（Mixture of Dyadic Experts），一种新的多任务 PEFT 算法。核心创新：(1) 所有 expert 共享同一个 down-projection 矩阵 A（基于 PCA 分析发现 down-projection 向量跨任务聚类，即 task-agnostic），消除 LoRA-MoE 中的参数冗余；(2) 将 LoRA 更新分解为 rank-one dyadic product 之和 $\Delta\mathbf{W} = \sum_{j=1}^r (\mathbf{a}_j \otimes \mathbf{b}_j)$，每个 rank 维度独立路由（fine-grained routing），允许 $m^r$ 种专家组合（传统 LoRA-MoE 仅 m 种）；(3) 广义 MoDE 支持 rank-p adapter，router 选择 p 列为一组。
  - 实验比较：(1) Multi-task 全量评估（756 tasks, SNI）：LoRA 64 vs MoLORA 16×4 vs MoLORA-SD 16×4 vs MoDE 16×4/8×4/6×4/4×4/4×6/4×8/4×16；(2) 广义 MoDE ablation：固定 m/r 变化 expert rank p (1→16)；(3) Iso-parametric 配置：固定总参数量变化 LoRA rank r、expert rank p；(4) Case study：15 类任务、固定参数预算约 6M，比较 LoRA 15×4（baseline，每任务独立 LoRA）vs LoRA 1×60 vs MoLORA vs MoLORA-SD vs MoDE。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台。作者来自 Google DeepMind，推测使用 Google Cloud TPU 或 GPU。Gemma 2B 模型规模较小，可在单卡运行。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Gemma 2B（Google 开源的 2B 参数 decoder-only LLM）。
  - 数据集：Supernatural Instructions (SNI)，含 1,616 个指令遵循任务。实验使用 756 个英文任务的训练集，每任务 90/10 切分。Case study 选取 15 类任务（QuestionAnswering, WrongCandidateGeneration, QuestionGeneration, GrammarErrorDetection 等，每类 ≥5k 训练样本）。
  - 评估指标：ROUGE-L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：**论文未开源代码**（Papers with Code 显示 "No code implementations yet"）。GitHub 上存在同名的 MoDE 项目（CLIP Data Experts 和 Diffusion Policy），与本论文无关。
  - 算法 Pipeline 伪代码：

```
# MoDE 前向传播 (per transformer layer)
# 输入: x ∈ R^{1×P}, 冻结权重 W0 ∈ R^{P×Q}
# 可训练参数: A ∈ R^{P×r} (共享), B_j^i ∈ R^{Q×1} (per rank per expert),
#              W_R;j ∈ R^{P×m} (per rank router)

def mode_forward(x):
    # 1. 冻结层输出
    y = x @ W0  # R^{1×Q}

    # 2. 共享 down-projection
    h = x @ A  # R^{1×r}, 其中 A = [a_1, ..., a_r], a_j ∈ R^{P×1}

    # 3. 对每个 rank 维度独立路由
    dyadic_sum = 0
    for j in range(r):  # 遍历每个 rank
        # 路由权重: softmax per-rank
        R_j = softmax(x @ W_R_j)  # R^(1×m)

        for i in range(m):  # 遍历每个专家
            # B_j^i ∈ R^{Q×1}, h_j 为标量
            dyadic_sum += R_j[i] * (h[:, j] * B_j^i)  # R^{1×Q}

    return y + dyadic_sum
```

张量计算等效形式：

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^m \sum_{j=1}^r \mathcal{R}_j^i(\mathbf{x}) \cdot (\mathbf{x} (\mathbf{a}_j \otimes \mathbf{b}_j^{iT}))$$

其中 $\mathcal{R}_j^i(\mathbf{x}) = \text{softmax}(\mathbf{x} \cdot \mathbf{W}_{\mathcal{R};j})_i$，$\mathbf{a}_j$ 是共享 down-projection 矩阵 A 的第 j 列，$\mathbf{b}_j^i$ 是第 i 个 expert 在第 j 个 rank 的 up-projection 向量。

广义 MoDE (rank-p adapter)：

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^{m} \sum_{k=1}^{r/p} \mathcal{R}_k^i(\mathbf{x}) \cdot \mathbf{x}\mathbf{A}_k \mathbf{B}_k^{iT}$$

其中 $\mathbf{A}_k \mathbf{B}_k^{iT} = \sum_{j=1}^p (\mathbf{a}_{j+p(k-1)} \otimes \mathbf{b}_{j+p(k-1)}^i)$。

- 训练配置：Adafactor 优化器，lr=1e-3，total sequence length=1024，batch size=128，训练 20,000 steps。
- MoDE $1 \times r \times r$ 等价于标准 LoRA rank r；MoDE $m \times r \times r$ 等价于 LoRA-MoE-SD。
