## MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-DisCo，一种基于 Block Coordinate Descent (BCD) 和 SimulParallel SGD 的 MoE 分阶段训练框架，由四个阶段组成：
    1. **Model Decoupling（模型解耦）**：将完整 MoE 模型（E 个 expert）分解为 E 个独立的 dense 子模型，每个子模型 = 完整共享 backbone（embedding, attention, LayerNorm 等）+ 单个 expert。MoE 层中移除 gating 机制，仅保留一个 expert，形成紧凑的 dense 子模型。
    2. **Data Decoupling（数据解耦）**：使用预训练 embedding 层对每个句子提取 token embedding 并做 mean pooling 得到句子向量 h_x，通过 K-Means（K=E）将训练数据聚类为 E 个语义区分的子集，每个子集分配给一个 expert 子模型。
    3. **Independent Parallel Training（独立并行训练）**：各子模型在其分配的数据子集上独立训练，无任何跨设备通信（无 gradient/parameter 交换），可在低成本 GPU（RTX 4090）上并行执行。
    4. **Model Reintegration & Fine-Tune（模型重组与微调）**：采用 "direct integration" 策略将各 expert 参数拼接为完整 expert 层；共享参数按 WP-SGD 加权平均融合；最后在完整数据集上进行短时间 global fine-tune（A100）恢复协调的 gating 行为。
  - 实验比较：(1) MoE-DisCo vs Full-Parameter MoE training，按 training loss、PPL、downstream tasks 和训练经济成本比较；(2) 消融：K-Means clustering vs random data assignment；(3) 消融：2 experts vs 4 experts 对收敛的影响。

- 硬件平台是什么，配置是什么。
  - S-phase（子模型训练）：NVIDIA RTX 4090 × 4（并行，无通信），价格 $0.35/GPU·hour
  - F-phase（fine-tune）：NVIDIA A100 80GB × 1，价格 $2.28/GPU·hour
  - Full-Parameter baseline：NVIDIA A100 80GB × 1
  - 计算精度：bfloat16
  - 序列长度：1024
  - Batch size：16

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Qwen1.5-MoE-2.7B：约 2.7B 激活参数，性能与 Mistral-7B 相当，实验中设 E=4 experts
    - LLaMA-MoE-3.5B：基于 LLaMA 架构的 MoE 设计，实验中设 E=4 experts
  - 数据集（预训练）：C4、WikiText-2、OpenWebText
  - Benchmark/评估指标：
    - 语言建模：Training Loss、Perplexity (PPL)
    - Downstream：ARC-e（5-shot）、MMLU（5-shot）、HellaSwag（0-shot）、PIQA（0-shot）
    - 经济成本：GPU 租用费用（$）、训练时长（hours）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://anonymous.4open.science/r/MoE-DisCo-4835/
  - 框架：PyTorch（论文未明确说明具体版本）
  - 算法 Pipeline 伪代码（对应 Algorithm 1）：

```
# ===== MoE-DisCo 算法流程 =====
# 输入: 原始数据集 D, MoE 共享参数 θ_shared, E 个 expert 参数 θ_1..θ_E
# 输出: 训练完成的全局 MoE 模型 M(Θ, D)

# --- Stage 1: Data Clustering ---
for x in D:
    # 对句子中所有 token 取 embedding 后 mean pooling
    h_x = MeanPool(Embed(x))    # h_x shape: [d_embed]

# K-Means 聚类，K = E
{D_1, ..., D_E} = KMeans({h_x}, K=E)

# --- Stage 2: Independent Submodel Training (S-phase, RTX 4090) ---
for k in 1..E:    # 完全并行，无跨设备通信
    θ_shared^(k) = θ_shared          # 复制共享参数
    Θ_k = (θ_shared^(k), θ_k)       # 子模型参数 = 共享 backbone + 第 k 个 expert
    Train M(Θ_k, D_k)               # 在数据子集 D_k 上训练子模型

# --- Stage 3: Reintegration ---
θ_exp* = Concat(θ_1, ..., θ_E)     # 拼接所有 expert 参数
θ_shared* = (1/E) * Σ_{k=1}^{E} θ_shared^(k)   # 共享参数加权平均

# --- Stage 4: Global Fine-Tune (F-phase, A100) ---
Θ = (θ_shared*, θ_exp*)             # 组装完整 MoE 参数
FineTune M(Θ, D)                    # 全数据集短时间微调
```

  - 超参数（S-phase）：Optimizer=AdamW, LR=1e-4, scheduler=constant, batch=16, bf16
  - 超参数（F-phase/Full-Param）：Optimizer=AdamW, LR=3e-4, weight_decay=0.01, warmup_ratio=0.03, scheduler=Cosine, batch=16, bf16
