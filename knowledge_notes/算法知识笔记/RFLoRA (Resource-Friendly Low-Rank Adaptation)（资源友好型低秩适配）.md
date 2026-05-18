## RFLoRA (Resource-Friendly Low-Rank Adaptation)（资源友好型低秩适配）

术语是什么？通过联网搜索让回答具体和精准。

RFLoRA是TailorLLM提出的LoRA变体，针对端云协同LLM推理中的adapter传输和存储开销优化。核心创新：(1) 参数解耦：观察到跨任务微调时LoRA的A矩阵趋于收敛（capture domain-invariant encoder features），B矩阵呈现任务特异性（adapt to domain-specific transformations），因此冻结共享A矩阵（所有任务共用一份）、仅训练和传输B矩阵；(2) 方向-幅度分解：将预训练权重W解耦为magnitude（列范数m = ||W||_c ∈ R^d）和direction（列归一化矩阵W/||W||_c），仅对direction分量应用LoRA低秩分解，幅度分量作为可训练标量独立优化。此设计使端侧仅需预存一份共享A矩阵，传输量从标准LoRA的22MB降至~11.56MB（约50% reduction），同等存储空间可容纳的adapter数量翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 预训练权重 W₀ ∈ R^{d×k}, rank r, 下游任务数据
输出: 压缩的task-specific adapter (B ∈ R^{d×r}, m ∈ R^d)

// Step 1: 权重分解
1: m = ||W₀||_c ∈ R^d           // column-wise L2 norm (magnitude)
2: V = W₀ / ||W₀||_c             // column-wise normalized (direction)
3: // W₀ = diag(m) · V

// Step 2: 初始化低秩矩阵（仅一次，跨任务共享A）
4: A ∈ R^{r×k} ~ Kaiming(seed=固定)  // 全局冻结，所有任务共享
5: B ∈ R^{d×r} = zeros(d, r)         // zero初始化，任务特定可训练

// Step 3: 训练（仅更新B和m）
6: for each training step:
7:     ΔV = B @ A                    // [d×r] @ [r×k] → [d×k]
8:     V' = V + ΔV                   // direction update
9:     W' = diag(m) · V' / ||V'||_c  // 重新组合并归一化direction
10:     // 等价于: W' = m · (W₀ + BA) / ||W₀ + BA||_c
11:     loss = task_loss(model(x|W'), y)
12:     m.grad, B.grad ← backward(loss)  // A无梯度
13:     m, B ← optimizer.step()

// Step 4: 传输（仅B + m，约标准LoRA的50%）
14: 云端→端侧: 传输B (d×r FP16) + m (d FP16)
// Step 5: 端侧推理
15: W' = m · (W₀ + BA) / ||W₀ + BA||_c  // 合并后推理，无额外延迟
```

RFLoRA的关键设计决策：(1) A用Kaiming初始化并冻结（与LoRA的随机初始化+可训练不同），因为A主要作为encoder投影输入到子空间；(2) B用zero初始化确保训练起始W'=W₀；(3) 方向归一化||W₀+BA||_c使magnitude和direction分量解耦，magnitude分量不受frozen A约束而独立优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RFLoRA在TailorLLM中通过PyTorch实现，在Llama3-1B的所有linear层上应用（r=16）。训练在云端RTX 3090上进行，与标准LoRA共享相同的训练schedule和超参数。每任务仅需存储和传输B矩阵(约11.56MB)和magnitude参数m(约几KB)，端侧预存一份共享A矩阵(~10.5MB)。在8个NLP benchmark上，RFLoRA以3.4M trainable params（0.273% of full model）达到81.6% avg accuracy，超越标准LoRA（81.2%, 0.454% params）和AdaLoRA（81.0%, 0.680% params），与DoRA（82.1%, 0.484% params）精度接近但参数量仅为其56%。论文未明确说明代码开源。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

