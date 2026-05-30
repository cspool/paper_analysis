## Brain Foundation Model (脑基础模型)

术语解释
Brain Foundation Model 是将深度学习中"基础模型"（Foundation Model）范式迁移到脑神经影像领域的模型类别：在大规模 fMRI 数据上通过自监督预训练学习通用的脑活动特征表示，再微调到下游任务（疾病诊断、行为识别、年龄预测等）。

术语是什么？
类似于 LLM 在通用文本上预训练后适配多种 NLP 任务，Brain Foundation Model 在多个数据集的 resting-state 和 tasking-state fMRI 数据上预训练，学习 BOLD 信号或 FC 矩阵的通用表示。核心流程：
1. 预训练：在大规模 fMRI 数据（UKB ~50k scans, HCP ~15k scans, OpenNeuro）上，通过自监督目标（mask reconstruction, JEPA prediction, contrastive learning）学习脑区活动模式的 latent representation
2. 微调：在目标下游数据集（如 ABIDE Autism, ADNI Alzheimer's）上，用少量标注数据微调预训练 encoder + 分类头
3. 推理：输入新 subject 的 fMRI → 预训练 encoder 提取特征 → 分类器输出诊断/预测

代表性模型：BrainLM (2023, MAE on BOLD, 650M params)、BrainJEPA (2024, JEPA masking, 307M)、BrainMass (2024, FC reconstruction + pseudo-FC, 34M)、BrainMoE (2025, MoE with 12 experts, 709M)。

从算法pipeline角度拆解术语。
```
# 通用 Brain Foundation Model Pipeline
# Phase 1: Pre-training (self-supervised)
for each fMRI_scan in large_scale_dataset:  # 来自多个数据源
    # Step 1: 预处理
    T1w_MRI → FSL segmentation → atlas parcellation (AAL/Schaefer)
    BOLD_4D → regional_mean_timeseries → FC_matrix (Pearson corr)
    
    # Step 2: 自监督预训练 (以 BrainMass 为例)
    FC = compute_FC(BOLD)           # [M, M] functional connectivity
    FC_masked = random_mask(FC)     # mask 部分 brain region pairs
    Z = Encoder(FC_masked)          # bottleneck → latent representation
    FC_hat = Decoder(Z)
    L = ||FC_hat - FC||²            # reconstruction loss

# Phase 2: Fine-tuning (supervised)
for each subject in downstream_dataset:
    Z = frozen_encoder.extract(FC)  # 提取预训练特征
    y_pred = classifier(Z)          # 分类/回归头 (SVM, MLP, Transformer)
    L = CrossEntropy(y_pred, y_true)
```

术语一般如何实现？如何使用？
- 框架：基于 PyTorch，使用 FSL (FMRIB Software Library) 做 MRI 预处理
- 脑图谱：常用 AAL (116 ROIs)、Schaefer (400 ROIs)、C200 等，将全脑分为 ~100-400 个区域
- 输入模态：BOLD timeseries → FC 矩阵、或直接使用 BOLD latent features
- 预训练数据：UK Biobank (UKB)、Human Connectome Project (HCP)、OpenNeuro 等公开数据集
- 下游应用：疾病早期诊断（Alzheimer's, Parkinson's, Autism, Schizophrenia）、性别/年龄预测、行为识别、fMRI-EEG 多模态融合

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
