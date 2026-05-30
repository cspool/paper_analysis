## MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

- baseline方法是什么？
  Baseline 为 Full-Parameter MoE Training，即传统的完整参数 MoE 训练方式。以 Qwen1.5-MoE-2.7B（E=4 experts）在 C4 数据集上训练为例说明全栈执行路径：
  - **算法层（MoE Training）**：完整的 MoE 模型（含共享 backbone + E 个 expert + gating network）在整个训练过程中全部加载到 GPU 内存中。每个 forward pass 经过 embedding → attention → LayerNorm → gating 选择 Top-K experts → 对应 expert MLP 计算 → 输出。Backpropagation 必须遍历所有 expert 路径（即使 inference 时只激活 Top-K），所有 expert 参数同时更新。**缺陷**：(1) 内存和计算开销随 expert 数量线性增长，导致无法在低内存 GPU（如 RTX 4090 ≤ 24GB）上训练；(2) 整个训练过程必须在昂贵的高带宽 GPU（A100, $2.28/GPU·h）上完成，训练成本极高——Qwen on C4 需 $22.50，Llama on OpenWebText 需 $32.13；(3) 大规模多 GPU 训练时，梯度同步和 activation 传输等通信开销导致 per-GPU MFU 随 GPU 数量增加而下降。
  - **系统框架层**：标准 PyTorch 分布式训练，使用 AdamW optimizer (LR=3e-4)、Cosine LR scheduler、warmup_ratio=0.03、weight_decay=0.01、batch_size=16、bf16、seq_len=1024。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用标准 PyTorch kernel，无自定义 kernel 优化。
  - **硬件架构层**：NVIDIA A100 80GB × 1 进行全参数训练，GPU 租赁 $2.28/GPU·h。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-DisCo——基于 BCD + SimulParallel SGD 的分阶段 MoE 训练框架，将 MoE 训练分解为"低成本并行子模型训练 + 短时高成本全局微调"。以 Qwen1.5-MoE-2.7B（E=4）在 C4 上的执行为例说明全栈执行路径：
  - **算法层（分阶段 MoE Training）**：
    1. **Model Decoupling**：完整 MoE 参数 Θ = (θ_shared, θ_1, ..., θ_E) 分解为 E 个子模型 Θ_k = (θ_shared^(k), θ_k)。每个子模型为 dense 模型，参数量大幅减小（仅 1/E 的 expert 参数），可直接放入 RTX 4090 的 24GB 显存。**解决 baseline 缺陷(1)**：将 O(E) 的内存需求降为 O(1)。
    2. **Data Decoupling**：用预训练 embedding 提取句子向量 → K-Means 聚类 → E 个语义区分的子数据集，分配给不同 expert。最大化子数据集间的分布差异以促进 expert 专业化。**解决 baseline 缺陷(1)延伸**：通过数据-模型联合解耦确保每个 expert 学到互补表征而非冗余知识。
    3. **Independent Parallel Training (S-phase)**：E=4 个子模型在 4 块 RTX 4090 上完全并行训练（最慢子模型 4200 steps/2.09h），零通信开销。S-phase 成本仅 $2.93。**解决 baseline 缺陷(3)**：完全消除跨设备通信开销，MFU 不随训练规模下降。**解决 baseline 缺陷(2)**：将大部分训练从 A100（$2.28/h）移至 RTX 4090（$0.35/h），成本降低约 6.5 倍。
    4. **Reintegration + Global Fine-Tune (F-phase)**：Expert 参数直接拼接，共享参数按 WP-SGD 加权平均。组装完整 MoE 后在单块 A100 上短时微调（1730 steps/0.76h），成本 $1.55。总成本 $6.87 vs baseline $22.50，节省 69.5%。
  - **系统框架层**：S-phase 使用标准 PyTorch + RTX 4090 × 4（完全独立，无分布式框架），F-phase 在单块 A100 上使用标准 PyTorch + AdamW。超参数：S-phase 用 AdamW (LR=1e-4, constant scheduler)，F-phase 用 AdamW (LR=3e-4, Cosine scheduler)。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。使用标准 PyTorch kernel。
  - **硬件架构层**：S-phase 使用 4 块 NVIDIA RTX 4090（$0.35/GPU·h），F-phase 使用 1 块 NVIDIA A100 80GB（$2.28/GPU·h）。训练时间从 baseline 9.87h 降至 3.82h。
