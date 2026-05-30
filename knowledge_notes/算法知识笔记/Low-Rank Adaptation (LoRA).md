## Low-Rank Adaptation (LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Low-Rank Adaptation (LoRA) 由 Hu et al. (2022, ICLR) 提出，是当前最主流的 Parameter-Efficient Fine-Tuning (PEFT) 方法之一。核心原理基于大语言模型的参数更新具有内在低秩性（intrinsic low-dimensionality）：全参数微调中权重更新矩阵 ΔW ∈ R^{m×n} 的实际有效自由度远低于 mn，可由两个低秩矩阵 B ∈ R^{m×r} 和 A ∈ R^{r×n} 的乘积近似，其中 r ≪ min(m, n)。正向传播：W'·x = W₀·x + (α/r)·B·(A·x)，其中 W₀ 冻结，A 使用 Kaiming 均匀初始化，B 零初始化。推理时可将 ΔW 合并到 W₀ 中无额外延迟。

FlyLoRA 论文基于 LoRA 的标准公式 (Eq. 1-2) 构建改进：标准 LoRA 中不同 rank（即 A 的不同行和 B 的不同列）之间存在参数耦合——梯度在所有 rank 间密集计算，导致 rank 间梯度协方差非零 (intra-task interference)；多任务 LoRA 合并时可训练的 A 之间无正交性保证 (inter-task interference)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 标准 LoRA 训练流程:
Forward:
  h_proj = A @ x               // [r×n] @ [n] → [r]
  delta = (α/r) * B @ h_proj  // B 的 r 个列加权组合
  output = W₀ @ x + delta     // W₀ 冻结 (无梯度)

Backward:
  grad_B = (α/r) * grad_output @ h_proj^T   // 所有 r 列有梯度
  grad_A = (α/r) * B^T @ grad_output @ x^T  // 所有 r 行有梯度

// LoRA rank-wise 展开 (Eq. 6, FlyLoRA):
// f_LoRA(x) = W₀·x + (α/r)·Σ_{i=1}^r b_i·(a_i·x)
// 每对 (a_i, b_i) 是一个 rank-1 组件, 类比一个 expert
// 标准 LoRA 中所有 r 个 rank-1 组件始终全激活 (k=r)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 标准实现：HuggingFace PEFT (`peft.LoraConfig`)，指定 target_modules（通常为 q/k/v/o/gate/up/down_proj）、rank r (8/16/32)、alpha scaling。保存为 adapter_model.bin。
- LoRA 变体：LoRA-FA (冻结 A 节省激活内存)、AsymmetryLoRA (固定 A 为随机投影)、AdaLoRA (自适应 rank 分配)、DoRA (幅值-方向解耦)、MoE-based LoRA (多 expert + router)。
- FlyLoRA 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
