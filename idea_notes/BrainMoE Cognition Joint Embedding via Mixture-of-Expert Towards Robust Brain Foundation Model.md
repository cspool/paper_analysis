## BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

- baseline方法是什么？
  **Baseline 为单一 brain foundation model 预训练于 resting-state fMRI**：现有 brain foundation models（BrainLM, BrainJEPA, BrainMass）将 fMRI 分析建模为自监督预训练+下游微调范式。预训练阶段仅使用 resting-state fMRI 数据（或最多加入一种 tasking state），通过 mask reconstruction（MAE 或 JEPA）学习 BOLD 或 FC 的 latent feature representation。下游微调使用 SVM 或简单 MLP 做分类/回归。

  **Baseline 全栈执行例子（以 BrainMass 处理 ABIDE Autism 分类为例）**：
  - **算法层**: fMRI raw BOLD → AAL atlas 分区 → FC 矩阵 X ∈ R^{116×116} → 随机 mask → Encoder(Bottleneck) → Z ∈ R^{2048} → Decoder → X̂ → L = ||X̂ - X||² → 预训练 29,951 resting-state scans → 下游：提取 Z → SVM/MLP → 2-class Autism 分类（F1=67.81% with MLP+68k）
  - **系统框架层**: PyTorch（推断）→ 单 GPU RTX 6000 Ada → 训练/推理在单卡完成 → 无分布式通信
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）
  - **Kernel/运行时调度层**: 标准 cuBLAS GEMM 执行 transformer encoder FFN 和 attention 计算
  - **硬件架构层**: 1× NVIDIA RTX 6000 Ada GPU → 所有模型参数常驻显存 → 推理 37.08ms/sample

  **Baseline 的核心缺陷**：
  1. **忽略认知状态异质性**：大规模 fMRI 数据集（UKB、HCP）包含多种认知状态（resting + 11 种 tasking），但现有模型仅使用 resting-state 数据（~30k scans），忽略了 >38k tasking fMRI。直接混合所有认知状态训练单一模型反而因不同认知状态间的异质性导致 suboptimal 特征表示（信息瓶颈理论）。
  2. **数据扩展边际收益递减**：从 30k→68k 预训练数据（加入 11 种 tasking states），BrainMass+MLP 在 ABIDE 上仅 +1.00 F1，在 SZ 上反而 -2.22 F1；BrainJEPA 在 HCPA 上 -6.42 F1。说明简单扩大数据规模不解决多认知状态异质性问题。
  3. **输入类型和预处理管线依赖**：BOLD 模型和 FC 模型对不同下游数据集的性能差异大，模型缺乏对输入类型和预处理管线的鲁棒性。不同预处理 pipeline（如 [33] vs 本文 pipeline）的数据分布差异进一步降低模型泛化。
  4. **Late Fusion MoE 无效**：直接使用 Late Fusion（各 expert 独立预测后加权融合）无法有效利用 expert 多样性——路由器总是倾向于选择数据量最大的单一 expert（如 Rest, n=29,971），无法学习专家间的协同。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **论文方法**: BrainMoE = 将 fMRI 数据按认知状态分层预训练多个 brain expert + Router 做 expert 选择 + Cognition Adapter（Transformer Decoder with cross-attention）混合 cognition embeddings。

  **Defect→Design 映射**:

  | Baseline 缺陷 | BrainMoE 设计选择 | 解决机制 |
  |---|---|---|
  | 忽略认知状态异质性 | 按 12 种认知状态分层预训练 12 个独立 expert | 每个 expert 专门学习一种认知状态下的 brain activity pattern，避免不同状态间相互干扰 |
  | 数据扩展边际收益递减 | Stratified pre-training + MoE fine-tuning | 不是简单混合所有数据训练一个模型，而是让每个 expert 成为特定认知状态的"专家"，再通过 adapter 联合利用。Taowu (n=40) 上 +18.28 F1 over 68k BrainMass |
  | 输入类型/预处理管线依赖 | Cognition Adapter 对 expert 架构和数据格式无要求 | Adapter 通过 cross-attention 将 FC 矩阵信息注入 task embeddings，不依赖 expert 的具体内部实现。支持 BOLD/FC 混合 expert |
  | Late Fusion MoE 无效 | Router + Cognition Adapter（Transformer Decoder）替代 weighted sum | Router 学习多样化的 dual expert 组合（而非单一 expert 主导），Adapter 通过 self-attention 混合 expert 嵌入产生新表示 |

  **BrainMoE 方法全栈执行例子（以 ABIDE Autism 分类为例）**：
  - **算法层**: fMRI → AAL116 FC 矩阵 X ∈ R^{116×116} → 12 个冻结 expert 分别产 Z_rest, Z_emotion, Z_gambling, ..., Z_language ∈ R^{2048} → Router: P = Softmax(W_r · [Z_1,...,Z_12]), Top-k → 选 Rest+Emotion expert → Cognition Adapter: Self-Attention(Z_topk + Q_task) → Cross-Attention(Q=FC, K=Z_bar, V=FC) → FFN → Linear → 2-class Autism 分类（F1=70.26% vs Baseline 67.81%）
  - **系统框架层**: PyTorch → 1× RTX 6000 Ada → 12 experts + adapter 常驻显存 → 推理 157.60ms/sample（vs baseline 37.08ms，4× 时间增加）
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）
  - **Kernel/运行时调度层**: 12 experts 前向并行 → Router Top-k → Adapter Self-Attention + Cross-Attention → 所有计算在单 GPU 上顺序执行
  - **硬件架构层**: 1× NVIDIA RTX 6000 Ada → 709M params 常驻显存 → 4× 推理时间 overhead（trade robustness for latency）

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 按认知状态分层预训练 12 experts | 解决简单混合多状态数据的 suboptimal 问题 | BrainMoE 在所有 7 个数据集上超越单个 expert（Fig 4），task-specific experts（Language→AD, WM→PD）超越 Rest expert |
  | Cognition Adapter (Transformer Decoder) | 替换 MLP adapter 的可扩展性不足 | 709M params, FC recon. expert 在 phenotypic 分类 4/7 数据集 rank 1st（Table 3） |
  | Cross-attention with FC matrix | 解决输入类型依赖，统一 BOLD/FC 多模态信息 | All-in-one BrainMoE (36 experts) 在 sex 分类 4/7 数据集 rank 1st（Table 4） |
  | Router Top-k with adaptive selection | 解决 Late Fusion 单一 expert 主导 | BrainMoE 学习 diverse dual expert 组合（Fig 5a vs 5b），expert 嵌入相关性 < 0.5（Fig 5c） |
  | 多 cognitive state 数据利用 | 解决 >38k tasking fMRI 被忽略的浪费 | 68,251 scans 全部利用，smallest dataset (Taowu n=40) 上 BrainMoE +43.76 F1 over BrainJEPA |
  | Age regression 泛化 | 验证方法在连续回归任务上的效果 | ABIDE (6-58 yrs): MSE 36.77→4.86（↓87%），HCPYA: 5.46→3.45 |
  | fMRI-EEG 多模态 | 验证跨模态鲁棒性 | CBraMod(EEG)+BrainMoE(fMRI) 在 8-task 分类 68.73% vs CBraMod only 67.66% |

  **创新总结**: BrainMoE 首次将 MoE 框架引入脑 fMRI 基础模型，核心洞察是：大规模脑影像数据中丰富的认知状态信息不应被忽视，也不应被简单混合——而应该通过"分而治之"（stratified pre-training）+ "智能混合"（cross-attention adapter）的方式利用。每个 brain expert 成为特定认知状态的"脑活动专家"，Cognition Adapter 通过 Transformer decoder 的自注意力和交叉注意力机制学习如何为每类下游任务组合这些专家知识。方法对 expert 架构和数据格式无要求，可适配任意 brain foundation model。
