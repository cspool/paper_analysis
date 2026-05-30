## Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Flex-MoE 框架，核心是两个算法组件：(1) **Missing Modality Bank Completion**：对缺失模态基于已观测模态组合从 learnable embedding bank 中查找补充，避免 zero-padding/imputation 破坏编码器训练；(2) **Expert Generalization & Specialization**：先用全模态样本通过 G-Router 训练通用 expert（含 load/importance balancing loss），再用 S-Router 结合交叉熵损失将 top-1 gate 绑定到目标 modality combination expert，剩余 top-(k-1) expert 仍做 load/importance balancing。实验比较 Flex-MoE 与单模态 baseline（3D CNN、VGG、ResNet-18/34）、多模态 baseline（TF、MulT、MAG、LIMoE、ShaSpec、mmFormer、FuseMoE）在 11 种 missing modality combination 场景下的 ACC、Macro-F1、AUC，以及消融实验（去除 expert specialization/generalization、去除 embedding bank、改变排序策略）、敏感度分析（expert 数量、SMoE 层数、top-k）和计算复杂度对比（mean time/iteration、GFLOPs、参数量）。
- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU。
- 模型是什么。数据集和bench分别是什么。
  - 模型：modality-specific encoders（3D-CNN 处理 MRI 图像、ResNet-34 处理 Genetic SNP 数据、MLP encoder 处理 Clinical/Biospecimen tabular 数据），输出 concat 后经 Transformer（FFN 替换为 SMoE layer，ADNI 用 16 experts, top-4 gating；MIMIC-IV 用 32 experts, top-3 gating），最后 1-layer MLP prediction head 做 AD 三分类或 MIMIC 二分类。Hidden dim=128, attention heads=4, batch size=8, learning rate=1e-4, 50 epochs（含 5 warm-up epochs），load/importance balancing loss coefficient=0.01。
  - 数据集：ADNI（Image/Genetic/Clinical/Biospecimen 四个模态，AD 阶段三分类：Dementia/CN/MCI，70/15/15 train/val/test split，test 和 val 取全模态交集以保证公平），MIMIC-IV（Lab&Vital/Clinical Notes/ICD-9 Codes 三个模态，一年死亡率二分类，每患者取最后一次访问）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/UNITES-Lab/flex-moe
  - 算法流程伪代码：
    ```
    # 输入: samples 按可用模态数降序排列
    # M = {I, C, B, G} = 4个模态
    # Missing modality bank: B ∈ R^(2^|M|-1 × |M| × d), d=128
    # MC_index: modality combination 索引, 如 "IGCB"=0, "IGC"=1, ..., "B"=14

    for each sample i in batch:
        for each modality m in {I, C, B, G}:
            if m is observed in sample i:
                e_i^m = Encoder_m(sample_i)    # 仅用有模态的样本训练encoder
            else:
                mc_idx = MC_index(observed_combinations(i))
                e_i^m = B[mc_idx][m]           # 从bank按观测组合查找缺失embedding
        h_i = concat([e_i^I, e_i^C, e_i^B, e_i^G])  # h_i ∈ R^(4×128)

    # === Phase 1: Expert Generalization (warm-up epochs, 仅全模态样本) ===
    if sample_i has ALL modalities:
        gate_logits = g(h_i)                   # g is 1-2 layer MLP
        gate_vals = TopK(softmax(gate_logits), k)
        y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
        # G-Router uses standard load + importance balancing loss:
        #   L_balance = CV^2(importance) + CV^2(load)

    # === Phase 2: Expert Specialization (remaining epochs, all samples) ===
    gate_logits = g(h_i)
    top1_pred = argmax(gate_logits)
    target_expert = MC_index(observed_modalities_of(i))
    # Cross-entropy loss to bind top-1 gate to target expert:
    #   L_ce = - Σ_j one_hot(MC(x_j)) · log(softmax(gate_logits))
    # Load/importance balancing only on remaining top-(k-1) experts:
    #   L_balance = CV^2(Σ_i importance_{e≠etop1}) + CV^2(Σ_i load_{e≠etop1})
    gate_vals = TopK(softmax(gate_logits), k)
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)

    # Inference: 任意模态组合 → S-Router 激活对应expert + 其他top-(k-1)
    pred = MLP_head(y_i)
    ```
  - 张量计算要点：missing modality bank `B` 的索引由观测模态组合的位掩码确定，共 2^4-1=15 种非全组合。bank embedding 维度 d=128。Router 为 1-2 层 MLP，输出经 softmax 后 Top-K 选择（ADNI 用 k=4, MIMIC-IV 用 k=3）。GFLOPs 约 59.06-59.07（极低，因 SMoE 稀疏激活），参数量约 36.5M-36.9M（远低于 FuseMoE 的 264.7M-340.9M）。mean time/iteration 约 12.73-16.00s，优于 FuseMoE（18.68-20.71s）。expert 总数 16（ADNI）或 32（MIMIC-IV），expert indices 对应所有可能的 modality combinations 加 buffer experts。Encoder 训练只使用该 modality 被 observed 的样本，不做 zero-imputation。
